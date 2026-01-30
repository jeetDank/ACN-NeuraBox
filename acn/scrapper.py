#!/usr/bin/env python3
"""
ACN Ultra-Smart Playwright Scraper
Detects interactive elements by HTML structure, not keywords
More robust and comprehensive
"""

import asyncio
import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse
from datetime import datetime
from typing import Set, Dict, List, Optional, Tuple
import logging

from playwright.async_api import async_playwright, Page, Browser, Locator
from bs4 import BeautifulSoup

from config import RAGConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UltraSmartDetector:
    """
    Detects interactive elements by HTML structure and attributes,
    not just text keywords. Much more robust!
    """
    
    @staticmethod
    async def find_all_interactive_elements(page: Page, current_url: str) -> List[Tuple[Locator, str, str]]:
        """
        Find ALL interactive elements on page using multiple detection methods.
        Returns: List of (locator, description, element_type) tuples
        """
        interactive_elements = []
        current_domain = urlparse(current_url).netloc
        
        # ========== METHOD 1: Real Buttons ==========
        buttons = await page.locator('button').all()
        for btn in buttons:
            try:
                # Check if visible and enabled
                if not await btn.is_visible():
                    continue
                if await btn.is_disabled():
                    continue
                
                text = await btn.inner_text()
                text = text.strip()[:100]  # Limit length
                
                if text:
                    interactive_elements.append((btn, f"Button: {text}", "button"))
            except:
                continue
        
        # ========== METHOD 2: Links with Button Classes ==========
        # Common button class patterns
        button_class_patterns = [
            'btn', 'button', 'cta', 'action', 'link-button',
            'primary', 'secondary', 'submit', 'call-to-action'
        ]
        
        for pattern in button_class_patterns:
            links = await page.locator(f'a[class*="{pattern}"]').all()
            for link in links:
                try:
                    if not await link.is_visible():
                        continue
                    
                    href = await link.get_attribute('href')
                    
                    # Skip if external domain
                    if href and href.startswith('http'):
                        link_domain = urlparse(href).netloc
                        if link_domain and link_domain != current_domain:
                            continue
                    
                    text = await link.inner_text()
                    text = text.strip()[:100]
                    
                    if text and len(text) > 2:
                        interactive_elements.append((link, f"Link-Button: {text}", "button-link"))
                except:
                    continue
        
        # ========== METHOD 3: Elements with onclick ==========
        onclick_elements = await page.locator('[onclick]').all()
        for elem in onclick_elements:
            try:
                if not await elem.is_visible():
                    continue
                
                text = await elem.inner_text()
                text = text.strip()[:100]
                
                if text:
                    interactive_elements.append((elem, f"Clickable: {text}", "onclick"))
            except:
                continue
        
        # ========== METHOD 4: Tabs (role="tab") ==========
        tabs = await page.locator('[role="tab"]').all()
        for tab in tabs:
            try:
                if not await tab.is_visible():
                    continue
                
                text = await tab.inner_text()
                text = text.strip()[:100]
                
                if text:
                    interactive_elements.append((tab, f"Tab: {text}", "tab"))
            except:
                continue
        
        # ========== METHOD 5: Accordions (aria-expanded) ==========
        accordions = await page.locator('[aria-expanded]').all()
        for accordion in accordions:
            try:
                if not await accordion.is_visible():
                    continue
                
                is_expanded = await accordion.get_attribute('aria-expanded')
                if is_expanded == 'true':
                    continue  # Already expanded
                
                text = await accordion.inner_text()
                text = text.strip()[:100]
                
                if text:
                    interactive_elements.append((accordion, f"Accordion: {text}", "accordion"))
            except:
                continue
        
        # ========== METHOD 6: Links with data-toggle ==========
        toggle_links = await page.locator('a[data-toggle], [data-toggle]').all()
        for link in toggle_links:
            try:
                if not await link.is_visible():
                    continue
                
                text = await link.inner_text()
                text = text.strip()[:100]
                
                if text:
                    interactive_elements.append((link, f"Toggle: {text}", "toggle"))
            except:
                continue
        
        # ========== METHOD 7: Modal Triggers ==========
        modal_triggers = await page.locator('[data-target*="modal"], [data-bs-toggle="modal"]').all()
        for trigger in modal_triggers:
            try:
                if not await trigger.is_visible():
                    continue
                
                text = await trigger.inner_text()
                text = text.strip()[:100]
                
                if text:
                    interactive_elements.append((trigger, f"Modal: {text}", "modal"))
            except:
                continue
        
        # ========== METHOD 8: Any clickable element (cursor: pointer) ==========
        # This catches custom interactive elements
        try:
            clickable = await page.evaluate('''
                () => {
                    const elements = [];
                    const all = document.querySelectorAll('*');
                    
                    for (let elem of all) {
                        const style = window.getComputedStyle(elem);
                        if (style.cursor === 'pointer' && elem.offsetParent !== null) {
                            // Element is visible and has pointer cursor
                            const text = elem.innerText?.trim().substring(0, 100);
                            if (text && text.length > 2) {
                                elements.push(text);
                            }
                        }
                    }
                    
                    return elements.slice(0, 20);  // Limit to 20
                }
            ''')
            
            for text in clickable:
                try:
                    # Find element with this text
                    elem = page.locator(f'text="{text}"').first
                    if await elem.count() > 0 and await elem.is_visible():
                        interactive_elements.append((elem, f"Clickable: {text}", "pointer"))
                except:
                    continue
        except:
            pass
        
        # Remove duplicates (same text)
        seen_texts = set()
        unique_elements = []
        
        for elem, desc, elem_type in interactive_elements:
            if desc not in seen_texts:
                seen_texts.add(desc)
                unique_elements.append((elem, desc, elem_type))
        
        return unique_elements


class ACNUltraSmartScraper:
    """
    Ultra-smart scraper with robust element detection and error handling
    """
    
    NAVIGATION_STRUCTURE = {
        'About': {
            'base': '/About',
            'priority_subpages': [
                '/About/About-Applied-Client-Network',
                '/About/Leadership',
                '/About/Contact-Us',
                '/About/FAQs',
            ],
            'max_pages': 15,
            'max_depth': 3,
        },
        'Get-Involved': {
            'base': '/Get-Involved',
            'priority_subpages': [
                '/Get-Involved/Join-the-Conversation',
                '/Membership/Join-Renew',
                '/Get-Involved/Volunteer',
                '/Get-Involved/Get-Active-in-the-Industry',
                '/Get-Involved/Attend-an-Upcoming-Event',
                '/Get-Involved/Sponsorship',
            ],
            'max_pages': 20,
            'max_depth': 4,
        },
        'Membership': {
            'base': '/Membership',
            'priority_subpages': [
                '/Membership/Why-ACN',
                '/Membership/Join-Renew',
                '/Membership/Access-Benefits',
                '/Membership/EZLynx-Membership',
                '/Membership/Member-Alliance-Program',
                '/Membership/Member-Alliances',
                '/Membership/Member-Alliance-Leaders',
            ],
            'max_pages': 30,
            'max_depth': 5,
        },
        'Resources': {
            'base': '/Resources',
            'priority_subpages': [
                '/Resources/Member-Alliances',
                '/Resources/Job-Roles',
                '/Resources/Mobile-App',
                '/Connections-Publication',
                '/Get-Involved/Sponsorship/Sponsor-Directory',
                '/Resources/EZLynx-Member-Hub',
                '/Resources/Epic-Member-Hub',
                '/Resources/New-Hire-Onboarding',
                '/Resources/EZLynx-Blueprint',
                '/resources/office-managers',
                '/resources/customer-service-reps',
                '/resources/it-operations-managers',
                '/resources/accountants',
                '/resources/principals-owners',
                '/resources/employee-benefits',
                '/resources/commercial-lines',
                '/resources/personal-lines',
            ],
            'max_pages': 40,
            'max_depth': 4,
            'exclude_patterns': [
                r'/Connections-Publication/Article/',
            ],
        },
        'Community': {
            'base': '/Community',
            'priority_subpages': [
                '/Community/Product-Forums',
                '/Community/Member-Alliance-Communities',
                '/Community/Job-Board',
                '/Community/Member-Directory',
                '/Community/Mobile-App',
            ],
            'max_pages': 25,
            'max_depth': 4,
        },
        'Events': {
            'base': '/Events',
            'priority_subpages': [
                '/Events/Summits',
                '/Events/Attend-Applied-Net',
                '/Events/Event-Calendar',
                '/Events/Summits/Dallas-Summit',
                '/Events/Summits/Dallas-Summit/Program',
                '/Events/Summits/Dallas-Summit/Registration',
                '/Events/Summits/Dallas-Summit/Hotel-Travel',
                '/Events/Summits/Dallas-Summit/Exhibitors-Sponsors',
                '/Events/Summits/Dallas-Summit/Program/Speakers',
                '/Events/Summits/Dallas-Summit/Program/EZLynx-Program',
                '/Events/Summits/Calgary-Summit',
                '/Events/Summits/Calgary-Summit/Program',
                '/Events/Summits/Calgary-Summit/Registration',
                '/Events/Summits/Calgary-Summit/Hotel-Travel',
                '/Events/Summits/Calgary-Summit/Exhibitors-Sponsors',
            ],
            'max_pages': 50,
            'max_depth': 5,
        },
    }
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.base_url = config.BASE_URL
        self.base_domain = urlparse(self.base_url).netloc
        self.global_visited: Set[str] = set()
        self.browser: Optional[Browser] = None
        self.context = None
        self.total_pages_scraped = 0
        self.section_stats = {}
        self.detector = UltraSmartDetector()
        self.interactions_found = []
    
    def normalize_url(self, url: str) -> str:
        """Normalize URL"""
        url = url.split('#')[0].split('?')[0]
        url = url.rstrip('/')
        return url
    
    def is_same_domain(self, url: str) -> bool:
        """Check if URL is same domain as base"""
        try:
            domain = urlparse(url).netloc
            # Allow main domain and subdomains
            return domain == self.base_domain or domain.endswith(f'.{self.base_domain}')
        except:
            return False
    
    def should_scrape_url(self, url: str, section_name: str) -> bool:
        """Check if URL should be scraped"""
        if not url.startswith('http'):
            return False
        
        # Must be same domain or subdomain
        if not self.is_same_domain(url):
            return False
        
        exclude_global = [
            r'/cdn-cgi/',
            r'\.(pdf|jpg|jpeg|png|gif|css|js|ico|zip|xml)$',
            r'/feed/',
            r'/wp-',
            r'^https://learning\.appliedclientnetwork\.org',  # Separate platform
            r'^https://community\.appliedclientnetwork\.org',  # Forum (separate)
            r'^https://members\.appliedclientnetwork\.org',    # Member portal (login required)
        ]
        
        for pattern in exclude_global:
            if re.search(pattern, url, re.IGNORECASE):
                return False
        
        section_config = self.NAVIGATION_STRUCTURE.get(section_name, {})
        exclude_patterns = section_config.get('exclude_patterns', [])
        
        for pattern in exclude_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False
        
        return True
    
    def belongs_to_section(self, url: str, section_name: str) -> bool:
        """Check if URL belongs to section"""
        section_config = self.NAVIGATION_STRUCTURE[section_name]
        base_path = section_config['base'].lower()
        url_path = urlparse(url).path.lower()
        
        if url_path.startswith(base_path.lower()):
            return True
        
        for subpage in section_config['priority_subpages']:
            if url_path.startswith(subpage.lower()):
                return True
        
        return False
    
    async def click_and_extract(self, page: Page, element: Locator, description: str, elem_type: str) -> Optional[Dict]:
        """
        Try to click an element and extract any new content
        """
        try:
            current_url = page.url
            current_domain = urlparse(current_url).netloc
            
            logger.info(f"  → Trying to click: {description}")
            
            # Try clicking with timeout
            try:
                # Click and wait for possible navigation
                async with page.expect_navigation(wait_until='networkidle', timeout=3000):
                    await element.click(timeout=2000)
                
                # Navigation happened
                new_url = page.url
                new_domain = urlparse(new_url).netloc
                
                # Check if we left the main domain
                if new_domain != current_domain and not self.is_same_domain(new_url):
                    logger.info(f"    ✗ External domain: {new_domain} - skipping")
                    # Try to go back (with error handling)
                    try:
                        await page.go_back(wait_until='networkidle', timeout=5000)
                    except:
                        # If can't go back, navigate to original URL
                        await page.goto(current_url, wait_until='networkidle', timeout=10000)
                    return None
                
                if new_url != current_url:
                    logger.info(f"    ✓ New page: {new_url}")
                    self.interactions_found.append({
                        'type': 'navigation',
                        'description': description,
                        'from_url': current_url,
                        'to_url': new_url
                    })
                    
                    await page.wait_for_timeout(1000)
                    html = await page.content()
                    
                    # Go back safely
                    try:
                        await page.go_back(wait_until='networkidle', timeout=5000)
                        await page.wait_for_timeout(500)
                    except:
                        # If can't go back, navigate to original
                        await page.goto(current_url, wait_until='networkidle', timeout=10000)
                    
                    return {
                        'url': new_url,
                        'html': html,
                        'trigger': description
                    }
            
            except Exception:
                # No navigation - try clicking anyway for dynamic content
                try:
                    await element.click(timeout=2000)
                    await page.wait_for_timeout(1000)
                    
                    # Check if still on same page
                    if page.url == current_url:
                        new_html = await page.content()
                        logger.info(f"    ✓ Clicked (checking for changes)")
                        
                        # Don't try to detect content changes, just return the new state
                        return {
                            'url': current_url,
                            'html': new_html,
                            'trigger': description
                        }
                    else:
                        # URL changed unexpectedly, go back
                        try:
                            await page.go_back(wait_until='networkidle', timeout=5000)
                        except:
                            await page.goto(current_url, wait_until='networkidle', timeout=10000)
                        return None
                
                except Exception as e:
                    logger.debug(f"    ✗ Could not click: {e}")
                    return None
        
        except Exception as e:
            logger.debug(f"    ✗ Error: {e}")
            return None
    
    async def discover_and_interact(self, page: Page) -> List[Dict]:
        """
        Discover all interactive elements and try clicking them
        Limit to avoid spending too much time on one page
        """
        new_content = []
        
        current_url = page.url
        
        # Find all interactive elements
        interactive_elements = await self.detector.find_all_interactive_elements(page, current_url)
        
        if interactive_elements:
            logger.info(f"  Found {len(interactive_elements)} interactive elements")
        
        # Limit clicks per page to avoid infinite loops
        MAX_CLICKS_PER_PAGE = 10
        clicks_done = 0
        
        # Try clicking each element
        for element, description, elem_type in interactive_elements:
            if clicks_done >= MAX_CLICKS_PER_PAGE:
                logger.info(f"  Reached max clicks ({MAX_CLICKS_PER_PAGE}) for this page")
                break
            
            result = await self.click_and_extract(page, element, description, elem_type)
            
            if result:
                new_content.append(result)
                clicks_done += 1
                
                # Make sure we're back on the original page
                if page.url != current_url:
                    try:
                        await page.goto(current_url, wait_until='networkidle', timeout=10000)
                    except:
                        logger.warning(f"  Could not return to {current_url}")
                        break
        
        return new_content
    
    async def extract_links(self, page: Page, section_name: str) -> List[str]:
        """Extract all links from the page"""
        links = []
        link_elements = await page.locator('a[href]').all()
        
        for element in link_elements[:100]:  # Limit to first 100 links
            try:
                href = await element.get_attribute('href')
                if href:
                    absolute_url = urljoin(page.url, href)
                    absolute_url = self.normalize_url(absolute_url)
                    
                    if (self.should_scrape_url(absolute_url, section_name) and
                        self.belongs_to_section(absolute_url, section_name) and
                        absolute_url not in self.global_visited):
                        
                        links.append(absolute_url)
            except:
                continue
        
        return list(set(links))
    
    async def scrape_page(self, page: Page, url: str, section_name: str) -> Optional[Tuple[List[Dict], List[str]]]:
        """Scrape a single page with automatic interaction detection"""
        try:
            logger.info(f"[{section_name}] Scraping: {url}")
            
            # Navigate to page
            await page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Wait for main content
            try:
                await page.wait_for_selector('main, article, .content', timeout=5000)
            except:
                pass
            
            # Extract initial content
            initial_html = await page.content()
            initial_soup = BeautifulSoup(initial_html, 'html.parser')
            
            # AUTOMATICALLY find and click interactive elements (with limits)
            discovered_content = await self.discover_and_interact(page)
            
            # Combine all content
            all_page_data = []
            
            # 1. Add original page
            page_data = self.extract_page_data(initial_soup, url, await page.title(), section_name)
            if page_data:
                all_page_data.append(page_data)
            
            # 2. Add discovered content
            for content in discovered_content:
                soup = BeautifulSoup(content['html'], 'html.parser')
                discovered_data = self.extract_page_data(
                    soup, 
                    content['url'], 
                    await page.title(), 
                    section_name,
                    trigger=content.get('trigger')
                )
                if discovered_data:
                    all_page_data.append(discovered_data)
            
            # Extract new links
            new_links = await self.extract_links(page, section_name)
            
            return all_page_data, new_links
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return [], []  # Return empty lists instead of None
    
    def extract_page_data(self, soup: BeautifulSoup, url: str, title: str, section: str, trigger: str = None) -> Optional[Dict]:
        """Extract clean data from BeautifulSoup"""
        # Remove unwanted elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        
        # Extract text
        main_content = soup.find('main') or soup.find('article') or soup.body
        
        if not main_content:
            return None
        
        text = main_content.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text or len(text) < 50:
            return None
        
        page_data = {
            'url': url,
            'title': title,
            'section': section,
            'content': text,
            'scraped_at': datetime.now().isoformat(),
            'content_length': len(text),
        }
        
        if trigger:
            page_data['discovered_via'] = trigger
        
        return page_data
    
    def save_page_data(self, page_data: Dict, section_name: str, page_index: int):
        """Save page data to JSON"""
        url_path = urlparse(page_data['url']).path.strip('/').replace('/', '_')
        
        if not url_path:
            url_path = 'home'
        
        section_dir = Path(self.config.RAW_DIR) / section_name.lower()
        section_dir.mkdir(parents=True, exist_ok=True)
        
        suffix = ""
        if page_data.get('discovered_via'):
            suffix = "_discovered"
        
        filename = f"{page_index:04d}__{url_path}{suffix}.json"
        filepath = section_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(page_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[{section_name}] Saved: {filepath.name}")
    
    async def scrape_section(self, page: Page, section_name: str) -> Dict:
        """Scrape a complete section"""
        logger.info("=" * 80)
        logger.info(f"SCRAPING SECTION: {section_name}")
        logger.info("=" * 80)
        
        section_config = self.NAVIGATION_STRUCTURE[section_name]
        max_pages = section_config['max_pages']
        max_depth = section_config['max_depth']
        
        from collections import deque
        queue = deque()
        section_visited = set()
        
        base_url = urljoin(self.base_url, section_config['base'])
        base_url = self.normalize_url(base_url)
        queue.append((base_url, 0))
        section_visited.add(base_url)
        self.global_visited.add(base_url)
        
        for subpage in section_config['priority_subpages']:
            url = urljoin(self.base_url, subpage)
            url = self.normalize_url(url)
            if url not in section_visited:
                queue.append((url, 1))
                section_visited.add(url)
                self.global_visited.add(url)
        
        pages_scraped = 0
        
        while queue and pages_scraped < max_pages:
            url, depth = queue.popleft()
            
            if depth > max_depth:
                continue
            
            result = await self.scrape_page(page, url, section_name)
            
            if result:
                all_page_data, new_links = result
                
                # Save all discovered content
                for page_data in all_page_data:
                    if page_data:
                        self.save_page_data(page_data, section_name, pages_scraped)
                        pages_scraped += 1
                        self.total_pages_scraped += 1
                        
                        if pages_scraped >= max_pages:
                            break
                
                # Add new links
                for link in new_links:
                    if link not in section_visited and pages_scraped < max_pages:
                        queue.append((link, depth + 1))
                        section_visited.add(link)
                        self.global_visited.add(link)
                
                await asyncio.sleep(0.5)
            
            if pages_scraped % 5 == 0 and pages_scraped > 0:
                logger.info(f"[{section_name}] Progress: {pages_scraped}/{max_pages} pages, Queue: {len(queue)}")
        
        stats = {
            'section': section_name,
            'pages_scraped': pages_scraped,
            'pages_visited': len(section_visited),
            'max_pages': max_pages,
            'completion': (pages_scraped / max_pages) * 100 if max_pages > 0 else 0,
        }
        
        logger.info(f"[{section_name}] Complete: {pages_scraped}/{max_pages} pages ({stats['completion']:.1f}%)")
        
        return stats
    
    async def crawl_all_sections(self):
        """Main crawl function"""
        logger.info("=" * 80)
        logger.info("ULTRA-SMART PLAYWRIGHT SCRAPER - STRUCTURE-BASED DETECTION")
        logger.info("=" * 80)
        logger.info(f"Base URL: {self.base_url}")
        logger.info("Features: HTML structure analysis, robust error handling")
        
        async with async_playwright() as p:
            logger.info("Launching browser...")
            self.browser = await p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            
            self.context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            )
            
            page = await self.context.new_page()
            
            for section_name in self.NAVIGATION_STRUCTURE.keys():
                try:
                    stats = await self.scrape_section(page, section_name)
                    self.section_stats[section_name] = stats
                except Exception as e:
                    logger.error(f"Error in section {section_name}: {e}")
                    self.section_stats[section_name] = {
                        'section': section_name,
                        'pages_scraped': 0,
                        'error': str(e)
                    }
                
                logger.info(f"\nSection '{section_name}' complete\n")
            
            await self.browser.close()
        
        self.print_final_summary()
        self.save_crawl_summary()
    
    def print_final_summary(self):
        """Print final summary"""
        logger.info("=" * 80)
        logger.info("CRAWL COMPLETE - FINAL SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total pages scraped: {self.total_pages_scraped}")
        logger.info(f"Interactive elements found: {len(self.interactions_found)}")
        logger.info("")
        
        logger.info("Section Breakdown:")
        logger.info("-" * 80)
        
        for section_name, stats in self.section_stats.items():
            if 'error' in stats:
                logger.info(f"❌ {section_name:20s}: ERROR - {stats['error']}")
            else:
                completion = stats.get('completion', 0)
                status = "✓" if completion >= 100 else "⚠"
                pages = stats.get('pages_scraped', 0)
                max_pages = stats.get('max_pages', 0) or self.NAVIGATION_STRUCTURE[section_name]['max_pages']
                logger.info(
                    f"{status} {section_name:20s}: {pages:3d}/{max_pages:3d} "
                    f"({completion:5.1f}%)"
                )
        
        logger.info("=" * 80)
    
    def save_crawl_summary(self):
        """Save summary"""
        summary = {
            'total_pages': self.total_pages_scraped,
            'base_url': self.base_url,
            'scraped_at': datetime.now().isoformat(),
            'strategy': 'ultra_smart_structure_based',
            'interactions_found': self.interactions_found,
            'sections': self.section_stats,
        }
        
        summary_path = Path(self.config.RAW_DIR) / '_crawl_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to: {summary_path}")


async def main():
    """Run ultra-smart scraper"""
    config = RAGConfig()
    scraper = ACNUltraSmartScraper(config)
    await scraper.crawl_all_sections()


if __name__ == "__main__":
    asyncio.run(main())