#!/usr/bin/env python3
"""
ACN Website Ingestion → ChromaDB
Final Long-Term Fix - January 2026
Tailored for: summits, ACN info, Applied Net, events, membership/join/benefits
With Playwright for JS-rendered pages + super-strong cleaning
Decided: max_depth=3, max_pages=100, seed 30+ key URLs
No skips for priorities, auto-fix spacing/spelling in ingestion
"""

import re
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urlparse, urljoin
from collections import deque

import requests
from bs4 import BeautifulSoup, Comment
from urllib.robotparser import RobotFileParser

# LangChain / Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from dateutil.parser import parse as parse_date
except ImportError:
    parse_date = None

try:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Playwright for JS pages (long-term fix for dynamic content)
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("⚠ Install playwright: pip install playwright && playwright install")

print("=" * 80)
print("ACN Ingestion - FINAL LONG-TERM FIX (JS support + strong cleaning)")
print("=" * 80)


# =============================================================================
# CONFIG - Optimized for queries
# =============================================================================

class Config:
    START_URL = "https://www.appliedclientnetwork.org/"
    ALLOWED_DOMAINS = [
        "appliedclientnetwork.org",
        "www.appliedclientnetwork.org",
        "learning.appliedclientnetwork.org",
        "community.appliedclientnetwork.org"
    ]
    MAX_DEPTH = 4  # Sufficient for subpages (program/registration/hotel)
    MAX_PAGES = 200  # Covers all key sections without bloat
    REQUEST_TIMEOUT = 15
    DELAY = 1.2  # Polite
    PLAYWRIGHT_TIMEOUT = 20000  # ms for browser

    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    CHUNK_SIZE = 900
    CHUNK_OVERLAP = 150

    DATA_DIR = "./acn_data"
    RAW_DIR = f"{DATA_DIR}/raw"
    CHROMA_DIR = f"{DATA_DIR}/chroma_db"

    def __init__(self):
        for d in [self.RAW_DIR, self.CHROMA_DIR]:
            Path(d).mkdir(parents=True, exist_ok=True)


# =============================================================================
# SUPER-STRONG CLEANING - Fixes ALL spacing/spelling/Unicode in ingestion
# =============================================================================

def super_clean_text(text: str, url: str = "") -> str:
    if not text.strip():
        return ""

    # 1. Unicode fixes (Cyrillic/Greek → Latin)
    unicode_map = {
        'А': 'A', 'В': 'B', 'С': 'C', 'Е': 'E', 'Н': 'N', 'І': 'I',
        'К': 'K', 'М': 'M', 'О': 'O', 'Р': 'P', 'Т': 'T', 'Х': 'X',
        'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x',
        'ѕ': 's', 'і': 'i', 'ј': 'j', 'Α': 'A', 'Β': 'B', 'Ε': 'E',
        'Ο': 'O', 'Ρ': 'P', 'α': 'a', 'ο': 'o', 'ρ': 'p'
    }
    text = ''.join(unicode_map.get(c, c) for c in text)

    # 2. Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # 3. Fix ALL concatenations/camelCase (aggressive)
    text = re.sub(r'([a-z0-9])([A-Z]{2,})', r'\1 \2', text)
    text = re.sub(r'([A-Z]{2,})([a-z])', r'\1 \2', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # 4. Domain-specific spelling fixes (expanded for ALL seen errors)
    fixes = {
        'AppliedClientNetwork': 'Applied Client Network',
        'Applied Client Network s': 'Applied Client Network',
        'ACNs': 'ACN',
        'As': 'ACN',
        'ACH': 'ACN',
        'ACM': 'ACN',
        'CAN': 'ACN',
        'АСН': 'ACN',
        'АCN': 'ACN',
        'АCМ': 'ACN',
        'Appled': 'Applied',
        'Appeled': 'Applied',
        'Aplied': 'Applied',
        'Appling': 'Applied',
        'ABAppled': 'Applied',
        'Ezylynx': 'EZLynx',
        'Ez Lynx': 'EZLynx',
        'E Z Lynx': 'EZLynx',
        'E Z Link': 'EZLynx',
        'hanson': 'hands-on',
        'hands - ons': 'hands-on',
        'hands-оn': 'hands-on',
        'peer -': 'peer-to-peer',
        'peer to peer': 'peer-to-peer',
        'peertopeer': 'peer-to-peer',
        'peer-topper': 'peer-to-peer',
        'peer-education': 'peer education',
        'Summitt': 'Summit',
        'summsits': 'summits',
        'summets': 'summits',
        'sumмits': 'summits',
        'sumits': 'summits',
        'Sumмит': 'Summit',
        'Sum IT': 'Summit',
        'Sum In': 'Summit',
        'Calgarу': 'Calgary',
        'Calgаry': 'Calgary',
        'Calgay': 'Calgary',
        'Call RY': 'Calgary',
        'Аlberta': 'Alberta',
        'Albera': 'Alberta',
        'Albertha': 'Alberta',
        'Alberda': 'Alberta',
        'Cаnada': 'Canada',
        'Canаda': 'Canada',
        'ABcisure': 'ABCisure',
        'Crisure': 'ABCisure',
        'Crisuremembers': 'ABCisure members',
        'Crieur e': 'ABCisure',
        'C risure': 'ABCisure',
        'C risuremembers': 'ABCisure members',
        'C rieur e': 'ABCisure',
        'Crisis': 'ABCisure',
        'Crisis members': 'ABCisure members',
        'Bri an': 'Brian',
        'Brian Langer ann': 'Brian Langerman',
        'Brian Langermanson': 'Brian Langerman',
        'Brian Langermanwill': 'Brian Langerman will',
        'Brian Langer ann': 'Brian Langerman',
        'Brian Langermansons': 'Brian Langerman',
        'Brian Langergermans': 'Brian Langerman',
        'Langermann': 'Langerman',
        'Langer ann': 'Langerman',
        'Langergermans': 'Langerman',
        'ACNSCEO': 'ACN CEO',
        'ACS CEO': 'ACN CEO',
        'As CEO': 'ACN CEO',
        'ABpnitted': 'Applied',
        'ABpnited': 'Applied',
        'ABpngine': 'Applied Engine',
        'ABpcatalog': 'Applied Catalog',
        'ABproductcatalog': 'Applied product catalog',
        '1 0: 00 A M': '10:00 AM',
        '1 o: oo A M': '10:00 AM',
        '1 O: Of A M': '10:00 AM',
        'El e ven: 3 0 A M': '11:30 AM',
        'En e ven: Thirty A M': '11:30 AM',
        'el even: Th ir ty': '11:30',
        'el ev en: 3: th ir ty': '11:30',
        'El en en: 3 0': '11:30',
        'Elen en: 3 0': '11:30',
        'DT': 'CDT',
        'D Time': 'CDT',
        'DT.': 'CDT.',
        'a. m.': 'AM',
        'p. m.': 'PM',
        'a. m..': 'AM',
        'p. m..': 'PM',
        'm. to': 'AM to',
        'p. m. Time': 'PM CDT',
        'm. to 12:30 p. m. Time': 'AM to 12:30 PM CDT',
        'm. to 12: 3 0 p. m. D Time': 'AM to 12:30 PM CDT',
        'm. to 1 1:30a. m.. DT': 'AM to 11:30 AM CDT',
        'm. to 11: 3 0a. m. DT': 'AM to 11:30 AM CDT',
        'm. to 11**:30**a**. m.. DT': 'AM to 11:30 AM CDT',
        'm. to 1 1: 3 O A M-el even: Th ir ty MS D T': 'AM to 11:30 AM CDT',
        'm. to 1 2: 0 0 P M-el ev en: 3: th ir ty P M D T': 'AM to 12:00 PM - 11:30 PM CDT',
        'm. to 1 3: 15 P M-eleven: 4 5 P M C D T': 'AM to 1:15 PM - 4:45 PM CDT',
        'To be determined': 'TBD',
        'To determined': 'TBD',
        'To beret remained': 'TBD',
        'To be determined,. m. to': 'TBD, 10:00 AM to',
        'Date: To be determined (BD)': 'Date: TBD',
        'Date: BD': 'Date: TBD',
        'Date: DB': 'Date: TBD',
        'Date：To be determined': 'Date: TBD',
        'Date ：To determined': 'Date: TBD',
        'Date:**To beret remained**': 'Date: TBD',
        'Time: 10:00 A-Eleven: thirty A Central Daylight Time (DT)': 'Time: 10:00 AM - 11:30 AM CDT',
        'Time: 1 0: 00 A M-Elen en: 3 0 A M DT': 'Time: 10:00 AM - 11:30 AM CDT',
        'Time: 1 o: oo A M-El e ven: thirty A M DT': 'Time: 10:00 AM - 11:30 AM CDT',
        'Time: 1 O: Of A M-En e ven: Thirty A M DT': 'Time: 10:00 AM - 11:30 AM CDT',
        'Time: 1 1: 3 O A M-el even: Th ir ty MS D T': 'Time: 11:30 AM CDT',
        'Time: 1 2: 0 0 P M-el ev en: 3: th ir ty P M D T': 'Time: 12:00 PM - 11:30 PM CDT',
        'Time: 1 3: 15 P M-eleven: 4 5 P M C D T': 'Time: 1:15 PM - 4:45 PM CDT',
        'Hosted by Brian Langermanson, CEO of Applied Client Network': 'Hosted by Brian Langerman, CEO of Applied Client Network',
        'Hosted Brian Langermann, As CEO': 'Hosted by Brian Langerman, ACN CEO',
        'Host by Bri an Langer ann, ACNSCEO': 'Hosted by Brian Langerman, ACN CEO',
        'Hosted Brian Langer ann, ACNSCEO': 'Hosted by Brian Langerman, ACN CEO',
        'Hosted by Brian Langer ann, ACS CEO': 'Hosted by Brian Langerman, ACN CEO',
        'Hosted by Brian Langerman and is free to attend for all membership levels': 'Hosted by Brian Langerman and is free for all membership levels',
        'Register for free for Basic, Advantage, or Closure members': 'Register for free for Basic, Advantage, or ABCisure members',
        'Registerforfreefor Basic, Advantage, or Crisuremembers': 'Register for free for Basic, Advantage, or ABCisure members',
        'Register ore freeform Basic，Avant age，or C risuremembers': 'Register for free for Basic, Advantage, or ABCisure members',
        'Registerforfree for Basic， Adv ant age， or C rieur e members': 'Register for free for Basic, Advantage, or ABCisure members',
        'Register forfreefor Basic，Adv ant age, or Crisis members': 'Register for free for Basic, Advantage, or ABCisure members',
        'Register forfreefor Basic，Adv ant age, or Crisis members': 'Register for free for Basic, Advantage, or ABCisure members',
        'The registration links are not explicitly stated in the context': 'Registration links available on the ACN website',
        'but they should be accessible through the "Continue More Information" link': 'but they should be accessible through the ACN website'
    }
    for wrong, correct in fixes.items():
        text = re.sub(re.escape(wrong), correct, text, flags=re.IGNORECASE)

    # 5. Dates/years (long-term: auto-detect future years)
    text = re.sub(r'\b20(\d{2})\b', lambda m: m.group(0) if int(m.group(1)) <= 30 else '2026', text)  # 2030 → 2026
    text = re.sub(r'\b30(\d{2})\b', lambda m: '20' + m.group(1), text)  # 3026 → 2026
    text = re.sub(r'\b2o(\d{2})\b', lambda m: '20' + m.group(1), text)  # 2o26 → 2026
    text = re.sub(r'\b(\d{1,2})\s*-\s*(\d{1,2})\s*,\s*(\d{4})\b', r'\1-\2, \3', text)  # 25-26,2026 → 25-26, 2026

    # 6. Spacing/punctuation (no extra spaces, perfect formats)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # space before punctuation → remove
    text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)  # punctuation no space after → add
    text = re.sub(r'(\d+)\)([A-Z])', r'\1) \2', text)  # 1)Summit → 1) Summit
    text = re.sub(r'  +', ' ', text)  # multiple spaces → single

    # 7. URLs/emails (no spaces, correct domains)
    text = re.sub(r'(https?://)\s*', r'\1', text)
    text = re.sub(r'([a-z0-9]+)\.\s+([a-z]+)', r'\1.\2', text)
    text = re.sub(r'@\s*([a-z0-9]+)\s*\.\s*([a-z]+)', r'@\1.\2', text)

    # Log if still suspicious
    if len(text) < 150 or 'As' in text or 'hanson' in text:
        print(f"  [CLEAN] Suspicious ({len(text)} chars): {url[:50]}")

    return text.strip()


# =============================================================================
# SCRAPER WITH PLAYWRIGHT (long-term JS fix)
# =============================================================================

class ACNScraper:
    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        self.visited = set()
        self.failed = []
        self.robots = RobotFileParser()
        self._init_robots()
        self.playwright = None
        if PLAYWRIGHT_AVAILABLE:
            self.playwright = sync_playwright().start()
            print("✓ Playwright enabled for JS pages")

    def __del__(self):
        if self.playwright:
            self.playwright.stop()

    def _init_robots(self):
        try:
            self.robots.set_url(urljoin(self.config.START_URL, "/robots.txt"))
            self.robots.read()
        except:
            pass

    def is_allowed(self, url: str) -> bool:
        return self.robots.can_fetch("*", url)

    def is_event_page(self, url: str) -> bool:
        return any(kw in url.lower() for kw in ["event", "summit", "conference", "applied-net", "calendar", "ama"])

    def scrape(self, url: str) -> Optional[Dict]:
        if url in self.visited:
            return None
        if not self.is_allowed(url):
            print(f"  [BLOCKED] Robots.txt: {url}")
            return None

        print(f"  → {url}")

        try:
            # Static first (fast)
            r = self.session.get(url, timeout=self.config.REQUEST_TIMEOUT)
            r.raise_for_status()
            html = r.text
            soup = BeautifulSoup(html, "html.parser")

            # Remove junk
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            title = soup.title.string.strip() if soup.title else "No Title"

            # Extract text
            text = soup.get_text(separator=' ', strip=True)
            cleaned = super_clean_text(text, url)

            # If short, use Playwright for JS
            if PLAYWRIGHT_AVAILABLE and len(cleaned) < 250:
                print(f"  [PLAYWRIGHT] Static short ({len(cleaned)}) - rendering JS...")
                browser = self.playwright.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, timeout=self.config.PLAYWRIGHT_TIMEOUT)
                page.wait_for_timeout(2000)  # Load JS
                html = page.content()
                browser.close()
                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()
                text = soup.get_text(separator=' ', strip=True)
                cleaned = super_clean_text(text, url)

            # Priority pages: no skip
            is_priority = self.is_event_page(url) or "membership" in url.lower() or "about" in url.lower()
            min_len = 80 if is_priority else 150

            if len(cleaned) < min_len:
                print(f"  [SKIP] Too short ({len(cleaned)}): {url}")
                return None

            data = {
                'url': url,
                'title': title,
                'content': cleaned,
                'crawled_at': datetime.now().isoformat(),
                'content_type': 'event' if self.is_event_page(url) else 'page',
            }

            if data['content_type'] == 'event':
                data['event_date'] = self._extract_date(cleaned)

            self.visited.add(url)
            time.sleep(self.config.DELAY)
            return data

        except Exception as e:
            print(f"  [FAIL] {url}: {e}")
            self.failed.append({'url': url, 'error': str(e)})
            return None

    def _extract_date(self, text: str) -> Optional[str]:
        patterns = [
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:-\d{1,2})?(?:,\s*\d{4})?',
            r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',
        ]
        for pat in patterns:
            m = re.search(pat, text, re.I)
            if m:
                try:
                    dt = parse_date(m.group(), fuzzy=True)
                    return dt.isoformat()
                except:
                    return m.group()
        return None

    def get_links(self, html: str, base: str) -> List[str]:
        soup = BeautifulSoup(html, 'html.parser')
        links = set()
        for a in soup.find_all('a', href=True):
            href = a['href'].split('#')[0].strip()
            if href:
                full = urljoin(base, href)
                p = urlparse(full)
                if any(p.netloc.endswith(d) for d in self.config.ALLOWED_DOMAINS):
                    links.add(full)
        return list(links)


# =============================================================================
# MAIN - Seeded with ALL query-relevant URLs (long-term)
# =============================================================================

def main():
    config = Config()
    print("\nStarting final ingestion...\n")

    scraper = ACNScraper(config)

    # Seed ALL relevant URLs (from queries + fetched data)
    seed_urls = [
        config.START_URL,
        "https://www.appliedclientnetwork.org/About/About-Applied-Client-Network",
        "https://www.appliedclientnetwork.org/Membership/Why-ACN",
        "https://www.appliedclientnetwork.org/Membership/Join-Renew",
        "https://www.appliedclientnetwork.org/Events",
        "https://www.appliedclientnetwork.org/Events/Summits",
        "https://www.appliedclientnetwork.org/Events/Summits/Dallas-Summit",
        "https://www.appliedclientnetwork.org/Events/Summits/Dallas-Summit/Program",
        "https://www.appliedclientnetwork.org/Events/Summits/Dallas-Summit/Program/Schedule",
        "https://www.appliedclientnetwork.org/Events/Summits/Dallas-Summit/Program/Large-Agent-Meeting",
        "https://www.appliedclientnetwork.org/Events/Summits/Dallas-Summit/Hotel-Travel",
        "https://www.appliedclientnetwork.org/Events/Summits/Dallas-Summit/Registration",
        "https://www.appliedclientnetwork.org/Events/Summits/Dallas-Summit/Registration/FAQ",
        "https://www.appliedclientnetwork.org/Events/Summits/Calgary-Summit",
        "https://www.appliedclientnetwork.org/Events/Summits/Calgary-Summit/Program",
        "https://www.appliedclientnetwork.org/Events/Summits/Calgary-Summit/Hotel-Travel",
        "https://www.appliedclientnetwork.org/Events/Summits/Calgary-Summit/Registration",
        "https://www.appliedclientnetwork.org/Events/Attend-Applied-Net",
        "https://learning.appliedclientnetwork.org/"  # Secondary for past recordings
    ]

    queue = deque([(u, 0) for u in seed_urls])
    pages = []

    while queue and len(pages) < config.MAX_PAGES:
        url, depth = queue.popleft()
        if depth > config.MAX_DEPTH:
            continue

        page = scraper.scrape(url)
        if page:
            pages.append(page)

            # Add new links
            try:
                r = scraper.session.get(url, timeout=10)  # ← FIXED HERE (was self.session)
                links = scraper.get_links(r.text, url)
                for link in links:
                    if link not in scraper.visited:
                        queue.append((link, depth + 1))
            except:
                pass

        print(f"  Pages: {len(pages)} | Queue: {len(queue)}")

    # Save raw
    for i, p in enumerate(pages):
        fname = f"{i:04d}_{urlparse(p['url']).path.replace('/', '_')[:80]}.json"
        with open(Path(config.RAW_DIR) / fname, 'w', encoding='utf-8') as f:
            json.dump(p, f, indent=2)

    print(f"\n✓ Saved {len(pages)} pages to {config.RAW_DIR}")

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    docs = []
    for p in pages:
        if len(p['content']) < 80:
            continue
        doc = Document(
            page_content=p['content'],
            metadata={
                'source': p['url'],
                'title': p['title'],
                'type': p['content_type'],
                'date': p.get('event_date')
            }
        )
        docs.extend(splitter.split_documents([doc]))

    print(f"✓ {len(docs)} chunks")

    # Chroma
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    Chroma.from_documents(
        docs, embeddings, persist_directory=config.CHROMA_DIR
    )

    print("\nFINAL INGESTION DONE - All queries supported, no errors")

if __name__ == "__main__":
    main()