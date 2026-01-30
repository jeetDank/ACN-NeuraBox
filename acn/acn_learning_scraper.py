#!/usr/bin/env python3
"""
ACN Learning Center Scraper
Scrapes paginated course listings (Page 1 → N)
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
import logging

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from posthog import page

from config import RAGConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ACNLearningScraper:

    # BASE_URL = "https://learning.appliedclientnetwork.org/"
    # MAX_PAGES = 130   # safe upper bound


    def __init__(self, config: RAGConfig):
        self.config = config
        self.output_dir = Path(config.RAW_DIR) / "learning"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = config.LEARNING_BASE_URL
        self.max_pages = config.LEARNING_MAX_PAGES

    async def scrape(self):
        empty_page_retries = 0
        MAX_EMPTY_RETRIES = 2
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(self.base_url, wait_until="networkidle")

            for page_no in range(1, self.max_pages + 1):
                logger.info(f"Scraping Learning Center – Page {page_no}")

                #await page.wait_for_selector(".card, .course, article", timeout=10000)
                # await page.wait_for_selector('text=Recorded On:', timeout=30000)
                await page.wait_for_load_state("networkidle")
                await page.wait_for_timeout(2000)

                # html = await page.content()
                # items = self.extract_items(html)
                # course_blocks = await page.locator('text=Recorded On:').locator('xpath=ancestor::*[self::div or self::li][1]').all()
                # Select all visible catalog items (not anchored to Recorded On)
                # 1. Find all occurrences of "Component(s)"
                component_nodes = await page.locator(
                    'text=/Contains\\s+\\d+\\s+Component\\(s\\)/'
                ).all()

                # 2. Climb up to the item container for each
                item_blocks = []
                seen_texts = set()

                for node in component_nodes:
                    block = node.locator(
                        'xpath=ancestor::*[self::div or self::li][1]'
                    )
                    text = (await block.inner_text()).strip()

                    # De-duplicate blocks
                    if text and text not in seen_texts:
                        seen_texts.add(text)
                        item_blocks.append(block)

                # 3. Keep only expected number (5 except last page)
                item_blocks = item_blocks[:5]

                # items = await self.extract_items_from_blocks(course_blocks)
                items = await self.extract_items_from_blocks(item_blocks)

                if not items:
                        empty_page_retries += 1
                        logger.warning(
                            f"No items found on page {page_no} "
                            f"(retry {empty_page_retries}/{MAX_EMPTY_RETRIES})"
                        )

                        if empty_page_retries >= MAX_EMPTY_RETRIES:
                            logger.warning("Confirmed end of pagination. Stopping.")
                            break

                        # Retry: reload page and continue
                        await page.reload(wait_until="networkidle")
                        await page.wait_for_timeout(3000)
                        continue
                else:
                        empty_page_retries = 0


                self.save_page(page_no, items)

                # Try clicking next page
                next_clicked = await self.go_to_next_page(page)
                if not next_clicked:
                    logger.info("Next page not available, scraping complete.")
                    break

                await page.wait_for_timeout(1500)

            await browser.close()

    def parse_date(self, text: str):
        import re
        match = re.search(r"(\d{2}/\d{2}/\d{4})", text)
        if match:
            return datetime.strptime(match.group(1), "%m/%d/%Y").date().isoformat()
        return None


    def extract_items(self, html: str):
        soup = BeautifulSoup(html, "html.parser")

        # Get visible text only
        body = soup.find("body")
        if not body:
            return []

        text = body.get_text("\n", strip=True)

        blocks = text.split("Recorded On:")
        items = []

        for block in blocks[1:]:
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            if len(lines) < 3:
                continue

            recorded_on_raw = lines[0]
            title = lines[-1] if len(lines) > 3 else None

            items.append({
                "title": title,
                "components": self.extract_components(" ".join(lines)),
                "recorded_on": self.parse_date(recorded_on_raw),
                "description": " ".join(lines[1:4])
            })

        return items

    def extract_components(self, text: str):
        import re
        match = re.search(r"Contains\s+\d+\s+Component\(s\).*?(Credits)?", text)
        if match:
            return match.group(0).strip()
        return None

    def extract_date(self, text: str):
        import re
        match = re.search(r"Recorded On:\s*(\d{2}/\d{2}/\d{4})", text)
        if match:
            return datetime.strptime(match.group(1), "%m/%d/%Y").date().isoformat()
        return None

    async def go_to_next_page(self, page):
        try:
            next_btn = page.locator('a[rel="next"], a:has-text(">"), li.active + li a').first
            if await next_btn.count() == 0:
                return False

            await next_btn.click()
            return True

        except Exception:
            return False
    
    def extract_title(self, lines):
        for line in lines:
            if (
                len(line) > 10 and
                "Component(s)" not in line and
                "Recorded On" not in line and
                "Register" not in line and
                "Continue" not in line and
                "More Information" not in line
            ):
                return line
        return None
    
    def classify_item_type(self, text):
        if "Course:" in text or "Applied Systems Course" in text:
            return "course"
        if "Live Web Event" in text:
            return "webinar"
        return "learning_resource"

    def extract_description(self, lines, title):
        for line in lines:
            if line == title:
                continue

            if (
                len(line) > 40 and
                "Component(s)" not in line and
                "Recorded On" not in line and
                "Register" not in line and
                "Continue" not in line and
                "More Information" not in line
            ):
                return line
        return None

    async def extract_items_from_blocks(self, blocks):
        items = []

        for block in blocks:
            text = (await block.inner_text()).strip()

            lines = [l.strip() for l in text.split("\n") if l.strip()]
            joined = " ".join(lines)

            title = self.extract_title(lines)

            items.append({
                "type": self.classify_item_type(joined),
                "title": title,
                "components": self.extract_components(joined),
                "recorded_on": self.parse_date(joined),
                "description": self.extract_description(lines, title)
            })

        return items

    def save_page(self, page_no: int, items: list):
        data = {
            "page": page_no,
            "scraped_at": datetime.now().isoformat(),
            "source": self.base_url,
            "items": items
        }

        file_path = self.output_dir / f"page_{page_no:03d}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {file_path.name}")


async def main():
    config = RAGConfig()
    scraper = ACNLearningScraper(config)
    await scraper.scrape()


if __name__ == "__main__":
    asyncio.run(main())