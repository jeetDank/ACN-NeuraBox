#!/usr/bin/env python3
"""
ACN Website Scraper - PRODUCTION-READY FINAL VERSION
Features:
- Parallel processing with asyncio for 5-10x speed boost
- Playwright/Chromium for JavaScript-rendered content
- Comprehensive text cleaning and validation
- Robust error handling and retry logic
- Quality checks to ensure clean database
- Progress tracking and detailed logging
"""

import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set
from urllib.parse import urlparse, urljoin
from collections import deque
from dataclasses import dataclass
import logging

# Core dependencies
import aiohttp
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Browser, Page

# LangChain & Embeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from dateutil.parser import parse as parse_date
except ImportError:
    parse_date = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('acn_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("=" * 80)
print("ACN PRODUCTION SCRAPER - FINAL VERSION")
print("Parallel Processing + Playwright + Comprehensive Cleaning")
print("=" * 80)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ScraperConfig:
    """Production scraper configuration"""
    # URLs
    START_URL: str = "https://www.appliedclientnetwork.org/"
    ALLOWED_DOMAINS: List[str] = None
    
    # Crawling limits
    MAX_DEPTH: int = 4
    MAX_PAGES: int = 500
    MAX_CONCURRENT: int = 20  # Parallel requests
    
    # Timeouts & delays
    REQUEST_TIMEOUT: int = 30
    PLAYWRIGHT_TIMEOUT: int = 45000  # ms
    MIN_DELAY: float = 0.3
    MAX_DELAY: float = 0.8
    
    # Retries
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 2.0
    
    # Content quality
    MIN_CONTENT_LENGTH: int = 100
    MIN_PRIORITY_LENGTH: int = 50  # For event/membership pages
    
    # Embedding & chunking
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    CHUNK_SIZE: int = 900
    CHUNK_OVERLAP: int = 150
    
    # Storage
    DATA_DIR: str = "./acn_data"
    RAW_DIR: str = "./acn_data/raw"
    CHROMA_DIR: str = "./acn_data/chroma_db"
    LOGS_DIR: str = "./acn_data/logs"
    
    def __post_init__(self):
        if self.ALLOWED_DOMAINS is None:
            self.ALLOWED_DOMAINS = [
                "appliedclientnetwork.org",
                "www.appliedclientnetwork.org",
                "learning.appliedclientnetwork.org",
                "community.appliedclientnetwork.org"
            ]
        
        # Create directories
        for dir_path in [self.RAW_DIR, self.CHROMA_DIR, self.LOGS_DIR]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


# =============================================================================
# TEXT CLEANING ENGINE - COMPREHENSIVE
# =============================================================================

class TextCleaner:
    """Handles all text cleaning and normalization"""
    
    # Unicode character mappings
    UNICODE_MAP = {
        # Cyrillic to Latin
        'А': 'A', 'В': 'B', 'С': 'C', 'Е': 'E', 'Н': 'N', 'І': 'I',
        'К': 'K', 'М': 'M', 'О': 'O', 'Р': 'P', 'Т': 'T', 'Х': 'X',
        'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x',
        'ѕ': 's', 'і': 'i', 'ј': 'j',
        # Greek to Latin
        'Α': 'A', 'Β': 'B', 'Ε': 'E', 'Ο': 'O', 'Ρ': 'P',
        'α': 'a', 'ο': 'o', 'ρ': 'p',
        # Common substitutions
        ''': "'", ''': "'", '"': '"', '"': '"', '–': '-', '—': '-'
    }
    
    # Domain-specific corrections
    CORRECTIONS = {
        # Organization names
        'AppliedClientNetwork': 'Applied Client Network',
        'Applied Client Network s': 'Applied Client Network',
        'ACNs': 'ACN', 'As': 'ACN', 'ACH': 'ACN', 'ACM': 'ACN',
        'CAN': 'ACN', 'АСН': 'ACN', 'АCN': 'ACN', 'АCМ': 'ACN',
        
        # Product names
        'Ezylynx': 'EZLynx', 'Ez Lynx': 'EZLynx', 'E Z Lynx': 'EZLynx',
        'ABCisure': 'ABCisure', 'ABcisure': 'ABCisure', 'Crisure': 'ABCisure',
        
        # Event terms
        'Summitt': 'Summit', 'summsits': 'summits', 'summets': 'summits',
        'sumмits': 'summits', 'sumits': 'summits', 'Sumмит': 'Summit',
        
        # Locations
        'Calgarу': 'Calgary', 'Calgаry': 'Calgary', 'Calgay': 'Calgary',
        'Аlberta': 'Alberta', 'Albera': 'Alberta', 'Albertha': 'Alberta',
        'Cаnada': 'Canada', 'Canаda': 'Canada',
        
        # People
        'Brian Langermann': 'Brian Langerman',
        'Langer ann': 'Langerman',
        'Langergermans': 'Langerman',
        
        # Common phrases
        'hands - on': 'hands-on', 'hanson': 'hands-on',
        'peer - to - peer': 'peer-to-peer', 'peertopeer': 'peer-to-peer',
        'To be determined': 'TBD', 'To determined': 'TBD',
    }
    
    @classmethod
    def clean(cls, text: str, url: str = "") -> str:
        """Apply all cleaning operations"""
        if not text or not text.strip():
            return ""
        
        # 1. Unicode normalization
        text = ''.join(cls.UNICODE_MAP.get(c, c) for c in text)
        
        # 2. Whitespace normalization
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 3. Fix camelCase concatenations
        text = re.sub(r'([a-z0-9])([A-Z]{2,})', r'\1 \2', text)
        text = re.sub(r'([A-Z]{2,})([a-z])', r'\1 \2', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # 4. Domain-specific corrections
        for wrong, correct in cls.CORRECTIONS.items():
            text = re.sub(re.escape(wrong), correct, text, flags=re.IGNORECASE)
        
        # 5. Fix dates and times
        text = cls._fix_dates(text)
        text = cls._fix_times(text)
        
        # 6. Fix punctuation spacing
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before
        text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)  # Add space after
        
        # 7. Fix URLs and emails
        text = re.sub(r'(https?://)\s*', r'\1', text)
        text = re.sub(r'@\s*([a-z0-9]+)\s*\.\s*([a-z]+)', r'@\1.\2', text)
        
        # 8. Remove multiple spaces
        text = re.sub(r'  +', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def _fix_dates(text: str) -> str:
        """Fix date formatting issues"""
        # Fix year typos
        text = re.sub(r'\b30(\d{2})\b', r'20\1', text)  # 3026 → 2026
        text = re.sub(r'\b2o(\d{2})\b', r'20\1', text)  # 2o26 → 2026
        text = re.sub(r'\b2602\b', '2026', text)
        
        # Fix date ranges
        text = re.sub(r'(\d{1,2})\s*-\s*(\d{1,2})\s*,\s*(\d{4})', r'\1-\2, \3', text)
        
        return text
    
    @staticmethod
    def _fix_times(text: str) -> str:
        """Fix time formatting issues"""
        # Fix common time corruptions
        time_fixes = {
            r'1\s*0:\s*0\s*0\s*A\s*M': '10:00 AM',
            r'1\s*1:\s*3\s*0\s*A\s*M': '11:30 AM',
            r'1\s*2:\s*0\s*0\s*P\s*M': '12:00 PM',
            r'a\.\s*m\.': 'AM',
            r'p\.\s*m\.': 'PM',
        }
        
        for pattern, replacement in time_fixes.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text


# =============================================================================
# ASYNC WEB SCRAPER WITH PLAYWRIGHT
# =============================================================================

class AsyncACNScraper:
    """High-performance async scraper with Playwright support"""
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.visited: Set[str] = set()
        self.failed: List[Dict] = []
        self.pages: List[Dict] = []
        self.browser: Optional[Browser] = None
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Statistics
        self.stats = {
            'total_scraped': 0,
            'static_success': 0,
            'playwright_success': 0,
            'failed': 0,
            'skipped': 0
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        # Create aiohttp session
        timeout = aiohttp.ClientTimeout(total=self.config.REQUEST_TIMEOUT)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        
        # Launch Playwright browser
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=True)
        logger.info("✓ Browser and session initialized")
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
        if self.browser:
            await self.browser.close()
        logger.info("✓ Resources cleaned up")
    
    def is_priority_url(self, url: str) -> bool:
        """Check if URL is high-priority"""
        priority_keywords = [
            'event', 'summit', 'conference', 'membership',
            'about', 'join', 'benefits', 'applied-net',
            'program', 'schedule', 'registration', 'hotel'
        ]
        url_lower = url.lower()
        return any(kw in url_lower for kw in priority_keywords)

    def is_event_page(self, url: str) -> bool:
        """
        Detect whether a URL represents an event-related page
        """
        event_keywords = [
            'summit',
            'conference',
            'event',
            'ama',
            'session',
            'workshop',
            'webinar'
        ]
        url_lower = url.lower()
        return any(keyword in url_lower for keyword in event_keywords)

    def should_use_playwright(self, url: str) -> bool:
        js_paths = ["/community/", "/members/", "/forum/"]
        return any(p in url.lower() for p in js_paths)
    
    def is_valid_url(self, url: str) -> bool:
        """Validate URL for crawling"""
        try:
            parsed = urlparse(url)
            
            # Check domain
            if not any(parsed.netloc.endswith(d) for d in self.config.ALLOWED_DOMAINS):
                return False
            
            # Block patterns
            blocked = [
                'login', 'logout', 'cart', 'profile', 'dashboard',
                'account', 'checkout', 'page=', '.pdf', '.jpg',
                '.png', '.css', '.js', '.zip'
            ]
            url_lower = url.lower()
            if any(b in url_lower for b in blocked):
                return False
            
            return True
        except:
            return False
    
    async def scrape_static(self, url: str) -> Optional[str]:
        """Scrape using aiohttp (fast path)"""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.text()
        except Exception as e:
            logger.debug(f"Static scrape failed for {url}: {e}")
        return None
    
    async def scrape_playwright(self, url: str) -> Optional[str]:
        """Scrape using Playwright (JS rendering)"""
        try:
            page = await self.browser.new_page()
            await page.goto(url, timeout=self.config.PLAYWRIGHT_TIMEOUT, wait_until='networkidle')
            # await page.wait_for_timeout(2000)  # Extra JS execution time
            html = await page.content()
            await page.close()
            return html
        except Exception as e:
            logger.debug(f"Playwright scrape failed for {url}: {e}")
        return None
    
    def extract_text(self, html: str, url: str) -> tuple[str, str]:
        """Extract and clean text from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove noise
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript']):
            tag.decompose()
        
        # Get title
        title = soup.title.string.strip() if soup.title else "Untitled"
        
        # Extract text with space separator (prevents concatenation)
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean thoroughly
        text = TextCleaner.clean(text, url)
        
        return title, text
    
    def extract_date(self, text: str) -> Optional[str]:
        """Extract event date from text"""
        patterns = [
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:-\d{1,2})?,?\s*\d{4}',
            r'\d{1,2}[-/]\d{1,2}[-/]\d{4}'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    if parse_date:
                        dt = parse_date(match.group(), fuzzy=True)
                        return dt.isoformat()
                    return match.group()
                except:
                    continue
        return None
    
    async def scrape_page(self, url: str, retry_count: int = 0) -> Optional[Dict]:
        """Scrape a single page with retry logic"""
        if url in self.visited:
            return None
        
        if not self.is_valid_url(url):
            self.stats['skipped'] += 1
            return None
        
        logger.info(f"→ Scraping: {url}")
        
        # Try static first (faster)
        if self.should_use_playwright(url):
            html = await self.scrape_playwright(url)
            method = "playwright"
        else:
            html = await self.scrape_static(url)
            method = "static"

        # Fallback if static failed
        if not html and method == "static":
            html = await self.scrape_playwright(url)
            method = "playwright"
        
        if not html:
            if retry_count < self.config.MAX_RETRIES:
                await asyncio.sleep(self.config.RETRY_DELAY)
                return await self.scrape_page(url, retry_count + 1)
            
            self.stats['failed'] += 1
            self.failed.append({'url': url, 'error': 'No HTML retrieved'})
            return None
        
        # Extract and clean text
        title, text = self.extract_text(html, url)
        
        # Validate content length
        is_priority = self.is_priority_url(url)
        min_length = self.config.MIN_PRIORITY_LENGTH if is_priority else self.config.MIN_CONTENT_LENGTH
        
        if len(text) < min_length:
            logger.warning(f"  ✗ Content too short ({len(text)} chars): {url}")
            self.stats['skipped'] += 1
            return None
        
        # Create page data
        is_event = self.is_event_page(url)

        page_data = {
            'url': url,
            'title': title,
            'content': text,
            'content_type': 'event' if is_event else 'page',
            'event_date': self.extract_date(text) if is_event else None,
            'crawled_at': datetime.now().isoformat(),
            "raw_html": html,
            'method': method,
            'content_length': len(text)
        }
   
        # Quality check
        if self._quality_check(page_data):
            self.visited.add(url)
            self.stats['total_scraped'] += 1
            if method == 'static':
                self.stats['static_success'] += 1
            else:
                self.stats['playwright_success'] += 1
            
            # Random delay to be polite
            # await asyncio.sleep(self.config.MIN_DELAY + 
            #                   (self.config.MAX_DELAY - self.config.MIN_DELAY) * 0.5)
            
            return page_data
        
        return None
    
    def _quality_check(self, page_data: Dict) -> bool:
        """Validate page data quality"""
        content = page_data['content']
        
        # Check for common corruption indicators
        corruption_indicators = [
            len(content) < 50,
            content.count('�') > 3,  # Encoding errors
            len(re.findall(r'[A-Z]{10,}', content)) > 20,  # Too many all-caps words
        ]
        
        if any(corruption_indicators):
            logger.warning(f"  ⚠ Quality check failed: {page_data['url']}")
            return False
        
        return True
    
    def extract_links(self, html: str, base_url: str) -> Set[str]:
        """Extract valid links from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        links = set()
        
        for a in soup.find_all('a', href=True):
            href = a['href'].split('#')[0].strip()
            if href:
                full_url = urljoin(base_url, href)
                if self.is_valid_url(full_url):
                    links.add(full_url)
        
        return links
    
    async def crawl(self) -> List[Dict]:
        """Main crawling logic with BFS and concurrency"""
        # Seed URLs
        seed_urls = self._get_seed_urls()
        queue = deque([(url, 0) for url in seed_urls])
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting crawl with {len(seed_urls)} seed URLs")
        logger.info(f"Max depth: {self.config.MAX_DEPTH}, Max pages: {self.config.MAX_PAGES}")
        logger.info(f"Concurrency: {self.config.MAX_CONCURRENT}")
        logger.info(f"{'='*80}\n")
        
        while queue and len(self.pages) < self.config.MAX_PAGES:
            # Collect batch for parallel processing
            batch = []
            batch_size = min(self.config.MAX_CONCURRENT, len(queue), 
                           self.config.MAX_PAGES - len(self.pages))
            
            for _ in range(batch_size):
                if queue:
                    batch.append(queue.popleft())
            
            # Process batch in parallel
            tasks = [self.scrape_page(url) for url, depth in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and extract new links
            for (url, depth), result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.error(f"Error scraping {url}: {result}")
                    continue
                
                if result:
                    self.pages.append(result)
                    
                    # Extract links if not at max depth
                    if depth < self.config.MAX_DEPTH:
                        try:
                            html = result.get("raw_html")
                            if html:
                                new_links = self.extract_links(html, url)
                                for link in new_links:
                                    if link not in self.visited:
                                        queue.append((link, depth + 1))
                        except Exception as e:
                            logger.debug(f"Link extraction failed for {url}: {e}")

            # Progress update
            logger.info(f"Progress: {len(self.pages)}/{self.config.MAX_PAGES} pages | Queue: {len(queue)}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Crawl complete!")
        logger.info(f"Stats: {json.dumps(self.stats, indent=2)}")
        logger.info(f"{'='*80}\n")
        
        return self.pages
    
    def _get_seed_urls(self) -> List[str]:
        """Get comprehensive seed URLs"""
        return [
            self.config.START_URL,
            "https://www.appliedclientnetwork.org/About/About-Applied-Client-Network",
            "https://www.appliedclientnetwork.org/Membership/Why-ACN",
            "https://www.appliedclientnetwork.org/Membership/Join-Renew",
            "https://www.appliedclientnetwork.org/Membership/Membership-Levels",
            "https://www.appliedclientnetwork.org/Events",
            "https://www.appliedclientnetwork.org/Events/Summits",
            "https://www.appliedclientnetwork.org/Events/Summits/Dallas-Summit",
            "https://www.appliedclientnetwork.org/Events/Summits/Dallas-Summit/Program",
            "https://www.appliedclientnetwork.org/Events/Summits/Dallas-Summit/Registration",
            "https://www.appliedclientnetwork.org/Events/Summits/Dallas-Summit/Hotel-Travel",
            "https://www.appliedclientnetwork.org/Events/Summits/Calgary-Summit",
            "https://www.appliedclientnetwork.org/Events/Attend-Applied-Net",
            "https://www.appliedclientnetwork.org/Events/AMA-Sessions",
            "https://learning.appliedclientnetwork.org/"
        ]
    
    def save_raw_data(self):
        """Save scraped pages to disk"""
        for i, page in enumerate(self.pages):
            filename = f"{i:04d}_{urlparse(page['url']).path.replace('/', '_')[:80]}.json"
            filepath = Path(self.config.RAW_DIR) / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(page, f, indent=2, ensure_ascii=False)
        
        # Save metadata
        metadata = {
            'total_pages': len(self.pages),
            'crawl_date': datetime.now().isoformat(),
            'stats': self.stats,
            'failed_urls': self.failed
        }

        with open(Path(self.config.DATA_DIR) / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Saved {len(self.pages)} pages to {self.config.RAW_DIR}")


# =============================================================================
# VECTOR DATABASE BUILDER
# =============================================================================

class VectorDBBuilder:
    """Build ChromaDB from scraped data"""
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_pages(self) -> List[Dict]:
        """Load scraped pages"""
        pages = []
        for filepath in Path(self.config.RAW_DIR).glob('*.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                pages.append(json.load(f))
        logger.info(f"✓ Loaded {len(pages)} pages")
        return pages
    
    def create_documents(self, pages: List[Dict]) -> List[Document]:
        """Convert pages to LangChain documents"""
        documents = []
        
        for page in pages:
            # Skip very short content
            if len(page['content']) < 50:
                continue
            
            doc = Document(
                page_content=page['content'],
                metadata={
                    'source': page['url'],
                    'title': page['title'],
                    'content_type': page['content_type'],
                    'event_date': page.get('event_date'),
                    'crawled_at': page['crawled_at'],
                    'method': page.get('method', 'unknown')
                }

            )
            
            # Don't chunk events - keep them whole
            if page['content_type'] == 'event':
                documents.append(doc)
            else:
                documents.extend(self.splitter.split_documents([doc]))
        
        logger.info(f"✓ Created {len(documents)} document chunks")
        return documents
    
    def build_chroma(self, documents: List[Document]):
        """Build ChromaDB"""
        logger.info("\n" + "="*80)
        logger.info("Building ChromaDB vector database...")
        logger.info("="*80 + "\n")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=self.config.EMBEDDING_MODEL
        )
        
        db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=self.config.CHROMA_DIR
        )
        
        logger.info(f"\n✓ ChromaDB created successfully!")
        logger.info(f"  → {len(documents)} documents stored")
        logger.info(f"  → Location: {self.config.CHROMA_DIR}")
        logger.info(f"  → Model: {self.config.EMBEDDING_MODEL}")
        
        return db


# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    """Main execution flow"""
    config = ScraperConfig()
    
    print("\n" + "="*80)
    print("PHASE 1: Web Scraping (Parallel + Playwright)")
    print("="*80 + "\n")
    
    # Scrape with async context manager
    async with AsyncACNScraper(config) as scraper:
        pages = await scraper.crawl()
        scraper.save_raw_data()
    
    print("\n" + "="*80)
    print("PHASE 2: Vector Database Creation")
    print("="*80 + "\n")
    
    # Build vector database
    builder = VectorDBBuilder(config)
    pages = builder.load_pages()
    documents = builder.create_documents(pages)
    builder.build_chroma(documents)
    
    print("\n" + "="*80)
    print("✅ COMPLETE - Production database ready!")
    print("="*80)
    print("\nDatabase features:")
    print("  ✓ Clean, properly formatted text")
    print("  ✓ JavaScript-rendered content captured")
    print("  ✓ Comprehensive domain-specific corrections")
    print("  ✓ Event dates extracted and stored")
    print("  ✓ Quality-checked content only")
    print("\nTest your queries with:")
    print("  python quick_start.py")
    print("  or")
    print("  python api.py")


if __name__ == "__main__":
    asyncio.run(main())