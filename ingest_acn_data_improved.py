#!/usr/bin/env python3
"""
ACN Web Scraper + RAG Ingestion - IMPROVED & HARDENED VERSION 2025
Focus: Maximum prevention of word concatenation, spacing errors, date garbage
"""

import os
import re
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser

try:
    from dateutil.parser import parse as parse_date
except ImportError:
    parse_date = None

# LangChain
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings

print("=" * 80)
print("ACN Ingestion Pipeline - HARDENED VERSION (better spacing & cleaning)")
print("=" * 80)


# ==================== CONFIG ====================

class Config:
    START_URL       = "https://www.appliedclientnetwork.org/"
    ALLOWED_DOMAINS = ["appliedclientnetwork.org", "learning.appliedclientnetwork.org", "community.appliedclientnetwork.org"]
    MAX_DEPTH       = 5
    MAX_PAGES       = 350
    REQUEST_TIMEOUT = 12
    DELAY           = 0.7

    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    CHUNK_SIZE      = 850
    CHUNK_OVERLAP   = 120

    DATA_DIR    = "./acn_data"
    RAW_DIR     = f"{DATA_DIR}/raw"
    CHUNKS_DIR  = f"{DATA_DIR}/chunks"
    CHROMA_DIR  = f"{DATA_DIR}/chroma_db"

    def __init__(self):
        for d in [self.RAW_DIR, self.CHUNKS_DIR, self.CHROMA_DIR]:
            Path(d).mkdir(parents=True, exist_ok=True)


# ==================== TEXT CLEANING (the most important part) ====================

def clean_text(text: str, debug_url: str = "") -> str:
    """
    Aggressive cleaning to prevent:
    - Missing spaces (especially around acronyms / product names)
    - Concatenated words
    - Garbage numbers that look like dates
    """
    if not text:
        return ""

    original = text

    # 1. Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # 2. Fix missing spaces before/after uppercase sequences (ACN, EZLynx, etc)
    text = re.sub(r'([a-z0-9])([A-Z]{2,})(?=[A-Z\s]|$)', r'\1 \2', text)   # acnEZLynx → acn EZLynx
    text = re.sub(r'([A-Z]{2,})([a-z])', r'\1 \2', text)                   # EZLynx → EZ Lynx
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)                       # AppliedClient → Applied Client

    # 3. Known replacement dictionary (expand as you find more issues)
    replacements = {
        'AppliedClientNetwork': 'Applied Client Network',
        'Applied Client Network s': 'Applied Client Network',
        'ACNs': 'ACN',
        'As': 'ACN',                    # very common broken abbreviation
        'Ezylynx': 'EZLynx',
        'EZLynx': 'EZLynx',             # protect correct one
        'hanson': 'hands-on',
        'Summitt': 'Summit',
        'summsits': 'summits',
        'summite': 'summit',
        'peer -': 'peer-to-peer',
        'peer-education': 'peer education',
        'To': 'TX',                     # likely Dallas, To → Dallas, TX
        'A': 'AB',                      # Calgary, A → Calgary, AB ?
        '11016': '2026',                # very common parse garbage
        '2o26': '2026',
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)

    # 4. Year / number fixes
    text = re.sub(r'\b(1[89]\d{2}|20\d{2})\b', r'\1', text)  # protect real years
    text = re.sub(r'\b[12]0\d{3}\b', lambda m: '20' + m.group(0)[-2:], text)  # 20226 → 2026 etc

    # 5. Punctuation spacing
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'([.,;:!?])([^\s])', r'\1 \2', text)

    # 6. Multiple dashes / messy ranges
    text = re.sub(r'[-–—]{2,}', '–', text)  # unify dashes

    # Optional: log big changes (helps debugging)
    if len(text) < 0.6 * len(original) or 'As ' in text or 'hanson' in text:
        diff = f"Cleaned {len(original)} → {len(text)} chars | URL: {debug_url}"
        print(f"  [CLEAN] {diff[:120]}")

    return text.strip()


# ==================== SCRAPER ====================

class ACNWebScraper:
    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ACN-Ingestion/1.0'
        })
        self.visited = set()
        self.failed = []
        self.robots = RobotFileParser()
        self._init_robots()

    def _init_robots(self):
        try:
            self.robots.set_url(urljoin(self.config.START_URL, "/robots.txt"))
            self.robots.read()
        except:
            pass

    def is_allowed(self, url: str) -> bool:
        return self.robots.can_fetch("*", url)

    def is_event_page(self, url: str) -> bool:
        path = urlparse(url).path.lower()
        return any(x in path for x in ["/event", "/summit", "/conference", "/calendar"])

    def scrape(self, url: str) -> Optional[Dict]:
        if url in self.visited:
            return None
        if not self.is_allowed(url):
            return None

        try:
            print(f"  → {url}")
            r = self.session.get(url, timeout=self.config.REQUEST_TIMEOUT)
            r.raise_for_status()
            html = r.text

            soup = BeautifulSoup(html, "html.parser")

            # Remove junk
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                tag.decompose()

            title = (soup.title.string or "No Title").strip()

            # ── KEY: Use space separator EVERYWHERE ──
            raw_text = soup.get_text(separator=" ", strip=True)

            # Apply strong cleaning
            clean_content = clean_text(raw_text, debug_url=url)

            if len(clean_content) < 350:
                print(f"  SKIP (too short after clean): {url}")
                return None

            data = {
                "url": url,
                "title": title,
                "clean_title": clean_text(title),
                "content": clean_content,
                "crawled_at": datetime.now().isoformat(),
                "content_type": "event" if self.is_event_page(url) else "page",
            }

            # Try to extract event date only for event-like pages
            if data["content_type"] == "event":
                data["event_date"] = self.extract_event_date(clean_content)

            self.visited.add(url)
            time.sleep(self.config.DELAY)
            return data

        except Exception as e:
            print(f"  ✗ {url} → {type(e).__name__}")
            self.failed.append({"url": url, "error": str(e)})
            return None

    def extract_event_date(self, text: str) -> Optional[str]:
        patterns = [
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:[-–]\d{1,2})?(?:,\s*\d{4})?',
            r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
        ]
        for pat in patterns:
            m = re.search(pat, text, re.I)
            if m:
                try:
                    if parse_date:
                        dt = parse_date(m.group(), fuzzy=True)
                        return dt.strftime("%Y-%m-%d")
                    else:
                        return m.group()
                except:
                    pass
        return None

    def get_links(self, html: str, base: str) -> List[str]:
        soup = BeautifulSoup(html, "html.parser")
        links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"].split("#")[0].strip()
            if not href:
                continue
            abs_url = urljoin(base, href)
            parsed = urlparse(abs_url)
            if any(parsed.netloc.endswith(d) for d in self.config.ALLOWED_DOMAINS):
                links.add(abs_url)
        return list(links)

    def crawl(self) -> List[Dict]:
        from collections import deque
        queue = deque([(self.config.START_URL, 0)])
        pages = []

        while queue and len(pages) < self.config.MAX_PAGES:
            url, depth = queue.popleft()
            if depth > self.config.MAX_DEPTH:
                continue
            if url in self.visited:
                continue

            page = self.scrape(url)
            if page:
                pages.append(page)

                if "html" in page:  # only follow links from full HTML pages
                    try:
                        links = self.get_links(page.get("html", ""), url)
                        for link in links:
                            if link not in self.visited:
                                queue.append((link, depth + 1))
                    except:
                        pass

            print(f"  {len(pages)} pages | queue: {len(queue)}")

        return pages


# ==================== MAIN FLOW ====================

def main():
    config = Config()
    print("Starting improved ingestion...\n")

    # 1. Scrape
    scraper = ACNWebScraper(config)
    pages = scraper.crawl()

    # Save raw (cleaned) data
    for i, page in enumerate(pages):
        fname = f"{i:04d}_{urlparse(page['url']).path.replace('/', '_')[:80]}.json"
        path = Path(config.RAW_DIR) / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(page, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(pages)} cleaned pages → {config.RAW_DIR}")

    # 2. Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    documents = []
    for page in pages:
        if len(page["content"]) < 400:
            continue
        doc = Document(
            page_content=page["content"],
            metadata={
                "source": page["url"],
                "title": page.get("clean_title", page["title"]),
                "content_type": page.get("content_type", "page"),
                "event_date": page.get("event_date"),
            }
        )
        chunks = splitter.split_documents([doc])
        documents.extend(chunks)

    print(f"Created {len(documents)} chunks")

    # 3. Chroma
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=config.CHROMA_DIR
    )
    print(f"ChromaDB saved → {config.CHROMA_DIR}")
    print(f"Collection count: {db._collection.count()}")

    print("\nIngestion finished. You should now get much cleaner answers.")


if __name__ == "__main__":
    main()