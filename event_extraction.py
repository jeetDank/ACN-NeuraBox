#!/usr/bin/env python3
"""
ACN Event Extraction Pipeline
Extracts structured event data from ACN website pages.
Fixes text corruption and extracts complete event metadata.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

from event_models import (
    Event, EventType, EventFilter,
    EVENT_URL_PATTERNS, SUBPAGE_PATTERNS, KNOWN_CITIES
)


class ACNTextCleaner:
    """
    Comprehensive text cleaning with full Unicode support.
    Fixes Cyrillic character corruption and other text issues.
    """
    
    # Complete Unicode character mappings (Cyrillic + lookalikes)
    UNICODE_MAP = {
        # Cyrillic letters that look like Latin
        'А': 'A',  # U+0410 Cyrillic Capital Letter A
        'В': 'B',  # U+0412 Cyrillic Capital Letter Ve
        'С': 'C',  # U+0421 Cyrillic Capital Letter Es (MISSING IN ORIGINAL!)
        'Е': 'E',  # U+0415 Cyrillic Capital Letter Ie
        'Н': 'N',  # U+041D Cyrillic Capital Letter En
        'І': 'I',  # U+0406 Cyrillic Capital Letter Byelorussian-Ukrainian I
        'К': 'K',  # U+041A Cyrillic Capital Letter Ka
        'М': 'M',  # U+041C Cyrillic Capital Letter Em
        'О': 'O',  # U+041E Cyrillic Capital Letter O
        'Р': 'P',  # U+0420 Cyrillic Capital Letter Er
        'Т': 'T',  # U+0422 Cyrillic Capital Letter Te
        'Х': 'X',  # U+0425 Cyrillic Capital Letter Ha
        'У': 'Y',  # U+0423 Cyrillic Capital Letter U
        'а': 'a',  # U+0430 Cyrillic Small Letter a
        'е': 'e',  # U+0435 Cyrillic Small Letter ie
        'о': 'o',  # U+043E Cyrillic Small Letter o
        'р': 'p',  # U+0440 Cyrillic Small Letter er
        'с': 'c',  # U+0441 Cyrillic Small Letter es
        'у': 'y',  # U+0443 Cyrillic Small Letter u
        'х': 'x',  # U+0445 Cyrillic Small Letter ha
        'і': 'i',  # U+0456 Cyrillic Small Letter byelorussian-ukrainian i
        'ј': 'j',  # U+0458 Cyrillic Small Letter je
        'ѕ': 's',  # U+0455 Cyrillic Small Letter dze
        # Greek letters that look like Latin
        'Α': 'A', 'Β': 'B', 'Ε': 'E', 'Ο': 'O', 'Ρ': 'P',
        'α': 'a', 'ο': 'o', 'ρ': 'p',
        # Smart quotes and dashes
        ''': "'", ''': "'", '"': '"', '"': '"',
        '–': '-', '—': '-', '‑': '-',
        # Common substitutions
        '`': "'",
    }
    
    # Domain-specific corrections
    CORRECTIONS = {
        # Organization names
        'AppliedClientNetwork': 'Applied Client Network',
        'Applied Client Network s': 'Applied Client Network',
        'ACNs': 'ACN', 'As': 'ACN', 'ACH': 'ACN', 'ACM': 'ACN',
        'CAN': 'ACN',
        # Product names
        'Ezylynx': 'EZLynx', 'Ez Lynx': 'EZLynx', 'E Z Lynx': 'EZLynx',
        'ABCisure': 'ABCisure', 'ABcisure': 'ABCisure',
        # Event terms
        'Summitt': 'Summit', 'summsits': 'summits', 'summets': 'summits',
        'sumмits': 'summits', 'sumits': 'summits',
        # Locations
        'Calgarу': 'Calgary', 'Calgаry': 'Calgary', 'Calgay': 'Calgary',
        'Аlberta': 'Alberta', 'Albera': 'Alberta', 'Albertha': 'Alberta',
        'Cаnada': 'Canada', 'Canаda': 'Canada',
        # Common phrases
        'hands - on': 'hands-on', 'hanson': 'hands-on',
        'peer - to - peer': 'peer-to-peer', 'peertopeer': 'peer-to-peer',
        'To be determined': 'TBD',
        # Fix common OCR/encoding errors
        '1 0:': '10:',  # Time corruption
        '1 1:': '11:',
        '1 2:': '12:',
        'A M': 'AM', 'P M': 'PM',
        'Ja n': 'Jan', 'Fe b': 'Feb', 'Ma r': 'Mar',
        'Ap r': 'Apr', 'Ma y': 'May', 'Ju n': 'Jun',
        'Ju l': 'Jul', 'Au g': 'Aug', 'Se p': 'Sep',
        'Oc t': 'Oct', 'No v': 'Nov', 'De c': 'Dec',
    }
    
    @classmethod
    def clean(cls, text: str, url: str = "") -> str:
        """Apply all cleaning operations to text"""
        if not text or not text.strip():
            return ""
        
        # 1. Unicode normalization
        text = ''.join(cls.UNICODE_MAP.get(c, c) for c in text)
        
        # 2. Domain-specific corrections
        for wrong, correct in cls.CORRECTIONS.items():
            text = re.sub(re.escape(wrong), correct, text, flags=re.IGNORECASE)
        
        # 3. Fix camelCase concatenations
        text = re.sub(r'([a-z0-9])([A-Z]{2,})', r'\1 \2', text)
        text = re.sub(r'([A-Z]{2,})([a-z])', r'\1 \2', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # 4. Whitespace normalization
        text = re.sub(r'\s+', ' ', text)
        
        # 5. Fix punctuation spacing
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)
        
        # 6. Fix date/time patterns
        text = cls._fix_dates(text)
        text = cls._fix_times(text)
        
        # 7. Fix URLs
        text = re.sub(r'(https?://)\s*', r'\1', text)
        
        # 8. Final cleanup
        text = re.sub(r'  +', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def _fix_dates(text: str) -> str:
        """Fix common date formatting issues"""
        # Fix year typos (3026 → 2026, 2o26 → 2026)
        text = re.sub(r'\b30(\d{2})\b', r'20\1', text)
        text = re.sub(r'\b2o(\d{2})\b', r'20\1', text)
        text = re.sub(r'\b2602\b', '2026', text)
        
        # Fix date ranges (ensure proper spacing)
        text = re.sub(r'(\d{1,2})\s*-\s*(\d{1,2})\s*,\s*(\d{4})', r'\1-\2, \3', text)
        
        return text
    
    @staticmethod
    def _fix_times(text: str) -> str:
        """Fix common time formatting issues"""
        time_fixes = {
            r'1\s*0:\s*0\s*0\s*A\s*M': '10:00 AM',
            r'1\s*1:\s*3\s*0\s*A\s*M': '11:30 AM',
            r'1\s*2:\s*0\s*0\s*P\s*M': '12:00 PM',
            r'a\.\s*m\.': 'AM',
            r'p\.\s*m\.': 'PM',
            r'\s*pm\b': ' PM',
            r'\s*am\b': ' AM',
        }
        for pattern, replacement in time_fixes.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text


class EventPageDetector:
    """
    Detects ACN event and summit detail pages.
    Excludes generic calendars, listing pages, and subpages (FAQ, registration, agenda).
    
    IMPORTANT: Only MAIN event/summit DETAIL pages should create events.
    Subpages (/faq, /registration, /agenda, etc.) must be MERGED into parent event.
    """
    
    # Subpage patterns that should be MERGED into parent event, not create separate events
    SUBPAGE_PATTERNS = {
        "program": ["program", "agenda", "schedule"],
        "speakers": ["speakers"],
        "registration": ["registration", "register"],
        "faq": ["faq"],
        "hotel": ["hotel", "travel", "accommodations"],
        "exhibitors": ["exhibitors", "sponsors"],
    }
    
    # BLOCKED subpage patterns - these should NEVER create events
    # URLs containing these patterns should be merged into parent event
    SUBPAGE_BLOCK_PATTERNS = [
        '/faq',
        '/registration',
        '/agenda',
        '/program',
        '/speakers',
        '/hotel',
        '/travel',
        '/accommodations',
        '/register',
    ]
    
    # Exclude these patterns (generic pages, not event details)
    EXCLUDE_PATTERNS = [
        r'/event-calendar',
        r'/events\?',
        r'/events/category/',
        r'/webinars',
        r'/learning',
        r'/ama$',  # AMA landing page, not specific AMA session
        r'/community/event-calendar',
        r'/topics/',  # Forum topics
        r'/forum/',
        r'/community/',
    ]
    
    @classmethod
    def is_event_detail_page(cls, url: str, content: str = "", title: str = "") -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Detect if URL is an ACN event detail page.
        
        Returns:
            Tuple of (is_main_event, event_type, parent_event_id)
            
        IMPORTANT: Only MAIN event/summit DETAIL pages create events.
        Subpages (/faq, /registration, /agenda, etc.) return:
            - is_main_event = False (don't create new event)
            - parent_event_id = ID of parent event to merge into
        """
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        path = parsed.path.lower()

        # 🚫 HARD EXCLUDE: ACN Learning / product catalog pages
        if 'learning.appliedclientnetwork.org' in host:
            return False, None, None

        if '/products/' in path:
            return False, None, None

        url_lower = url.lower()
        path = urlparse(url).path.lower()
        
        # STEP 1: Check BLOCKED subpage patterns FIRST
        # These URLs should NEVER create events - they must be merged into parent
        for block_pattern in cls.SUBPAGE_BLOCK_PATTERNS:
            if block_pattern in url_lower:
                # Extract parent event ID for merging
                path_parts = [p for p in path.split('/') if p]
                parent_event_id = None
                
                # Build parent event ID from path
                # e.g., /events/summits/dallas-summit/faq → acn_events_summits_dallas_summit
                if len(path_parts) >= 3:
                    # Remove the subpage part (last part)
                    parent_path = '/' + '/'.join(path_parts[:3])
                    parent_event_id = Event.generate_event_id_from_path(parent_path)
                elif len(path_parts) == 2:
                    # e.g., /events/dallas-summit/faq
                    parent_path = '/' + '/'.join(path_parts[:2])
                    parent_event_id = Event.generate_event_id_from_path(parent_path)
                
                print(f"  ⚠ Blocked subpage: {url}")
                print(f"    → Will merge into parent event: {parent_event_id}")
                return False, None, parent_event_id
        
        # STEP 2: Check exclusion patterns
        for pattern in cls.EXCLUDE_PATTERNS:
            if re.search(pattern, url_lower):
                return False, None, None
        
        # STEP 3: Check URL patterns for event detail pages
        event_type = None
        parent_event_id = None
        
        # Summit detection
        if '/summits/' in url_lower:
            event_type = "Summit"
            
            # Check if this is a subpage (should NOT create new event)
            path_parts = [p for p in path.split('/') if p]
            is_subpage = len(path_parts) > 3  # More than /events/summits/dallas-summit
            
            if is_subpage:
                # Check for subpage type - these should NOT create separate events
                for subpage_type, patterns in cls.SUBPAGE_PATTERNS.items():
                    for pattern in patterns:
                        if pattern in path_parts:
                            # This is a subpage - return parent event ID for merging
                            # Build parent URL from path parts
                            if len(path_parts) >= 3:
                                # Parent path: /events/summits/dallas-summit
                                parent_path = '/' + '/'.join(path_parts[:3])
                                parent_event_id = Event.generate_event_id_from_path(parent_path)
                            # Mark as NOT a main event (don't create new event)
                            return False, None, parent_event_id
            
            # This is the main summit page - create event
            return True, "Summit", None
        
        # Conference detection
        elif '/conference/' in url_lower or '/applied-net' in url_lower:
            event_type = "Conference"
            
            # Check for subpages
            path_parts = [p for p in path.split('/') if p]
            is_subpage = len(path_parts) > 3
            
            if is_subpage:
                for subpage_type, patterns in cls.SUBPAGE_PATTERNS.items():
                    for pattern in patterns:
                        if pattern in path_parts:
                            if len(path_parts) >= 3:
                                parent_path = '/' + '/'.join(path_parts[:3])
                                parent_event_id = Event.generate_event_id_from_path(parent_path)
                            return False, None, parent_event_id
        
        # General event detection
        elif any(re.search(p, url_lower) for p in EVENT_URL_PATTERNS):
            event_type = "Event"
            
            # Check for subpages
            path_parts = [p for p in path.split('/') if p]
            is_subpage = len(path_parts) > 3
            
            if is_subpage:
                for subpage_type, patterns in cls.SUBPAGE_PATTERNS.items():
                    for pattern in patterns:
                        if pattern in path_parts:
                            if len(path_parts) >= 3:
                                parent_path = '/' + '/'.join(path_parts[:3])
                                parent_event_id = Event.generate_event_id_from_path(parent_path)
                            return False, None, parent_event_id
        
        # Check content signals if provided (only for main event pages)
        if event_type and content:
            # Verify page actually has event content
            event_signals = [
                r'(?:register|registration)',
                r'(?:date|time).*?:?\s*\w+\s+\d{1,2}',
                r'(?:join|attend).*(?:event|summit|conference)',
                r'agenda|schedule|speakers?',
                r'(?:hotel|travel|venue|location)',
            ]
            has_event_content = any(re.search(p, content, re.IGNORECASE) for p in event_signals)
            if not has_event_content:
                # Still mark as event but content might be minimal
                pass
        
        return event_type is not None, event_type, parent_event_id
    
    @classmethod
    def is_exclude_page(cls, url: str) -> bool:
        """Check if page should be excluded from event extraction"""
        url_lower = url.lower()
        
        for pattern in cls.EXCLUDE_PATTERNS:
            if re.search(pattern, url_lower):
                return True
        
        # Also exclude non-event URLs
        if not any(p in url_lower for p in EVENT_URL_PATTERNS):
            return True
        
        return False


class EventDataExtractor:
    """
    Extracts structured event data from ACN page content.
    """
    
    # City patterns (city name followed optionally by state)
    CITY_PATTERNS = [
        r'((?:Dallas|Chicago|Orlando|Calgary|Toronto|Salt Lake City|Las Vegas|Seattle|Boston|Atlanta|Philadelphia|Phoenix|Minneapolis|San Antonio|San Diego|Denver|Austin)[,\s]*(?:Texas|TX|Alberta|AB|Ontario|ON|Florida|FL|Utah|UT|Nevada|NV|Washington|WA|Massachusetts|MA|Georgia|GA|Pennsylvania|PA|Arizona|AZ|Colorado|CO|Minnesota|MN|California|CA)?)',
    ]
    
    # Date patterns (multiple formats)
    DATE_PATTERNS = [
        # "February 25-26, 2026" or "February 25, 2026"
        r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:[-–]\d{1,2})?,?\s*\d{4})',
        # "Feb 25-26, 2026"
        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:[-–]\d{1,2})?,?\s*\d{4})',
        # "2026-02-25"
        r'(\d{4}[-]\d{2}[-]\d{2})',
    ]
    
    # Venue patterns
    # VENUE_PATTERNS = [
    #     r'(?:at|venue|location[:\s]+)([^.\n]{10,80})',
    #     r'(?:held at|hosted at|taking place at)([^.\n]{10,80})',
    # ]
    VENUE_PATTERNS = [
    r'(?:Venue|Location)\s*:\s*([A-Za-z0-9 ,&\-]{5,60})',
    r'(?:at|held at|hosted at)\s+([A-Z][A-Za-z0-9 ,&\-]{5,60})',
    ]

    # Description extraction patterns
    DESCRIPTION_PATTERNS = [
        r'<h2[^>]*>([^<]+)</h2>.*?<p>([^<]+)</p>',
        r'class="[^"]*Head[^"]*">([^<]+)</',
        r'<h[1-3][^>]*>([^<]+)</h[1-3]>.*?<p>([^<]+)</p>',
    ]
    
    @classmethod
    def extract_event_data(
        cls,
        url: str,
        title: str,
        content: str,
        html_content: str = "",
        event_type: str = "Event",
        parent_event_id: Optional[str] = None
    ) -> Event:
        """
        Extract structured event data from page.
        
        All fields are extracted from the actual page content.
        DO NOT infer or hallucinate missing values.
        """
        # Clean the text
        cleaned_title = ACNTextCleaner.clean(title, url)
        cleaned_content = ACNTextCleaner.clean(content, url)
        
        # Generate event ID
        # event_id = Event.generate_event_id(url)
        # Canonical event ID
        if parent_event_id:
            event_id = parent_event_id
        else:
            parsed = urlparse(url)
            event_id = Event.generate_event_id_from_path(parsed.path)

        
        # Extract location
        # city, state = cls._extract_location(cleaned_content)
        
        # # Extract dates
        # start_date, end_date = cls._extract_dates(cleaned_content)

        # Extract location and dates ONLY from MAIN event page
        if not parent_event_id:
            city, state = cls._extract_location(cleaned_content)
            start_date, end_date = cls._extract_dates(cleaned_content)
        else:
            # Subpages must NEVER override core event facts
            city = None
            state = None
            start_date = None
            end_date = None
  
        # Extract venue
        venue = cls._extract_venue(cleaned_content, html_content)
        
        # Extract description
        description = cls._extract_description(cleaned_content, html_content)
        
        # Extract agenda (if present)
        agenda = cls._extract_agenda(cleaned_content, html_content)
        
        # Extract speakers (if present)
        speakers = cls._extract_speakers(cleaned_content, html_content)
        
        # Extract registration URL
        registration_url = cls._extract_registration_url(url, html_content)
        
        # Create Event object
        event = Event(
            event_id=event_id,
            title=cleaned_title or url.split('/')[-1],
            event_type=EventType(event_type),
            city=city,
            state=state,
            venue=venue,
            start_date=start_date,
            end_date=end_date,
            description=description,
            agenda=agenda,
            speakers=speakers,
            registration_url=registration_url,
            source_url=url,
            parent_event_id=parent_event_id,
        )
        
        return event
    
    @classmethod
    def _extract_location(cls, content: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract city and state from content"""
        city = None
        state = None
        
        for pattern in cls.CITY_PATTERNS:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                location_str = match.group(1).strip()
                # Parse city and state
                parts = re.split(r'[,\s]+', location_str)
                city = parts[0]
                # Check if last part is a state
                if len(parts) > 1:
                    last_part = parts[-1].upper()
                    state_map = {
                        'TX': 'TX', 'TEXAS': 'TX',
                        'AB': 'AB', 'ALBERTA': 'AB',
                        'ON': 'ON', 'ONTARIO': 'ON',
                        'FL': 'FL', 'FLORIDA': 'FL',
                        'UT': 'UT', 'UTAH': 'UT',
                        'NV': 'NV', 'NEVADA': 'NV',
                    }
                    if last_part in state_map:
                        state = state_map[last_part]
                    else:
                        # Check if last part matches known city
                        if parts[-1] in KNOWN_CITIES:
                            city = parts[-1]
                            if len(parts) > 2:
                                state = KNOWN_CITIES.get(parts[-1])
                break
        
        return city, state
    
    @classmethod
    def _extract_dates(cls, content: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract start and end dates from content"""
        start_date = None
        end_date = None
        
        # Try to find date patterns
        for pattern in cls.DATE_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                # Use first date as start, last as end if range
                date_str = matches[0]
                # Parse to ISO format
                parsed = cls._parse_date_to_iso(date_str)
                if parsed:
                    start_date = parsed
                    if len(matches) > 1:
                        end_parsed = cls._parse_date_to_iso(matches[-1])
                        if end_parsed:
                            end_date = end_parsed
                break
        
        # Validate date range: ensure end_date >= start_date
        if start_date and end_date:
            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                
                if end_dt < start_dt:
                    # Swap dates if end is before start
                    # This handles cases where the format was misinterpreted
                    start_date, end_date = end_date, start_date
                    print(f"  ⚠ Swapped invalid date range: {start_date} to {end_date}")
            except ValueError:
                # If parsing fails, keep original values
                pass

        if start_date and end_date:
            if end_date < start_date:
                print(f"⚠ Invalid date range fixed: {start_date} → {end_date}")
                start_date, end_date = min(start_date, end_date), max(start_date, end_date)

        return start_date, end_date
    
    @staticmethod
    def _parse_date_to_iso(date_str: str) -> Optional[str]:
        """Parse various date formats to YYYY-MM-DD"""
        from dateutil.parser import parse as parse_date
        
        try:
            # Clean the date string
            date_str = date_str.strip()
            # Handle dash-separated dates
            if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                return date_str
            
            # Try dateutil parser
            dt = parse_date(date_str, fuzzy=True)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            # Return original if parsing fails
            return date_str if re.match(r'\d{4}-\d{2}-\d{2}', date_str) else None
    
    @classmethod
    def _extract_venue(cls, content: str, html_content: str = "") -> Optional[str]:
        """Extract venue name from content"""
        
        # 1️⃣ Try from clean plain text (preferred)
        for pattern in cls.VENUE_PATTERNS:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                venue = match.group(1).strip()

                # 🚨 SANITY CHECK — kill navbar / junk text
                if len(venue.split()) <= 1:
                    return None

                return venue

        # 2️⃣ Fallback: try from HTML (very conservative)
        if html_content:
            soup = BeautifulSoup(html_content, 'html.parser')

            venue_element = soup.find(string=re.compile(r'\b(venue|location|hotel)\b', re.I))
            if venue_element:
                parent = venue_element.find_parent()
                if parent:
                    venue = parent.get_text(strip=True)[:60]

                    # Same sanity check here
                    if len(venue.split()) <= 1:
                        return None

                    return venue

        return None


    @classmethod
    def _extract_description(cls, content: str, html_content: str = "") -> Optional[str]:
        """Extract event description from content"""
        # Get first substantial paragraph(s)
        if html_content:
            soup = BeautifulSoup(html_content, 'html.parser')
            # Remove navigation, scripts, etc.
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                tag.decompose()
            
            # Find main content area
            main = soup.find('main') or soup.find(id='body_content')
            if main:
                # Get first meaningful paragraph
                paragraphs = main.find_all('p')
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if len(text) > 50:  # Substantial paragraph
                        return text[:500]
        
        # Fallback: extract first long paragraph from plain text
        paragraphs = content.split('\n\n')
        for p in paragraphs:
            if len(p.strip()) > 50:
                return p.strip()[:500]
        
        return None
    
    @classmethod
    def _extract_agenda(cls, content: str, html_content: str = "") -> Optional[str]:
        """Extract agenda/schedule from content"""
        # Look for agenda-related content
        agenda_keywords = ['agenda', 'schedule', 'program', 'sessions', 'timeline']
        content_lower = content.lower()
        
        for keyword in agenda_keywords:
            if keyword in content_lower:
                # Extract section around keyword
                idx = content_lower.find(keyword)
                start = max(0, idx - 50)
                end = min(len(content), idx + 500)
                section = content[start:end]
                
                # If substantial, return it
                if len(section) > 100:
                    return section.strip()
        
        return None
    
    @classmethod
    def _extract_speakers(cls, content: str, html_content: str = "") -> Optional[List[str]]:
        """Extract speaker names from content"""
        speakers = []
        
        # Look for speaker patterns
        speaker_patterns = [
            r'Speakers?[:\s]+([^.\n]+)',
            r'Featuring[:\s]+([^.\n]+)',
            r'Presenter[s]?[:\s]+([^.\n]+)',
        ]
        
        for pattern in speaker_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                for match in matches:
                    # Split by common delimiters
                    names = re.split(r'[,;]|\band\b', match)
                    for name in names:
                        name = name.strip()
                        if len(name) > 2 and len(name) < 50:
                            speakers.append(name)
                break
        
        # Remove duplicates
        speakers = list(dict.fromkeys(speakers))
        
        return speakers if speakers else None
    
    @classmethod
    def _extract_registration_url(cls, url: str, html_content: str) -> Optional[str]:
        """Extract registration URL from page"""
        if not html_content:
            return None
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Look for registration links
        for a in soup.find_all('a', href=True):
            text = a.get_text(strip=True).lower()
            href = a['href']
            
            if 'register' in text or 'registration' in text:
                # Make relative URLs absolute
                if href.startswith('/'):
                    parsed = urlparse(url)
                    href = f"{parsed.scheme}://{parsed.netloc}{href}"
                elif not href.startswith('http'):
                    href = urljoin(url, href)
                return href
            
            # Check href directly
            if 'registration' in href.lower() or 'register' in href.lower():
                if href.startswith('/'):
                    parsed = urlparse(url)
                    href = f"{parsed.scheme}://{parsed.netloc}{href}"
                elif not href.startswith('http'):
                    href = urljoin(url, href)
                return href
        
        return None


class EventStore:
    """
    Stores and queries extracted ACN events.
    Uses JSON files for persistence.
    """
    
    def __init__(self, data_dir: str = "./acn_data"):
        self.data_dir = Path(data_dir)
        self.events_file = self.data_dir / "events.json"
        self.events_dir = self.data_dir / "events"
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        
        self._events: Dict[str, Event] = {}
        self._load_events()
    
    def _load_events(self):
        """Load events from JSON file"""
        if self.events_file.exists():
            try:
                with open(self.events_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for event_data in data:
                        event = Event.from_dict(event_data)
                        self._events[event.event_id] = event
                print(f"✓ Loaded {len(self._events)} events from {self.events_file}")
            except Exception as e:
                print(f"⚠ Failed to load events: {e}")
    
    def save_events(self):
        """Save events to JSON file"""
        events_data = [event.to_dict() for event in self._events.values()]
        with open(self.events_file, 'w', encoding='utf-8') as f:
            json.dump(events_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved {len(self._events)} events to {self.events_file}")
    
    def add_event(self, event: Event):
        """Add or update an event"""
        self._events[event.event_id] = event
    
    def get_event(self, event_id: str) -> Optional[Event]:
        """Get event by ID"""
        return self._events.get(event_id)
    
    def get_all_events(self) -> List[Event]:
        """Get all events"""
        return list(self._events.values())
    
    def query_events(self, filter_criteria: EventFilter) -> List[Event]:
        """
        Query events by filter criteria.
        
        Returns events sorted by start_date ascending.
        """
        results = []
        for event in self._events.values():
            if filter_criteria.matches(event):
                results.append(event)
        
        # Sort by start date
        results.sort(key=lambda e: e.start_date or "9999-12-31")
        
        return results
    
    def get_upcoming_events(self, event_type: Optional[EventType] = None) -> List[Event]:
        """Get all upcoming events"""
        filter_criteria = EventFilter(
            upcoming_only=True,
            event_type=event_type,
        )
        return self.query_events(filter_criteria)
    
    def get_events_by_city(self, city: str, event_type: Optional[EventType] = None) -> List[Event]:
        """Get events in a specific city"""
        filter_criteria = EventFilter(
            city=city,
            event_type=event_type,
        )
        return self.query_events(filter_criteria)
    
    def get_events_in_date_range(
        self,
        start_date: str,
        end_date: str,
        event_type: Optional[EventType] = None
    ) -> List[Event]:
        """Get events within a date range"""
        filter_criteria = EventFilter(
            start_date_from=start_date,
            start_date_to=end_date,
            event_type=event_type,
        )
        return self.query_events(filter_criteria)
    
    def get_event_by_title(self, title_query: str) -> Optional[Event]:
        """Get event by partial title match (case-insensitive)"""
        title_query_lower = title_query.lower().strip()
        
        for event in self._events.values():
            event_title_lower = event.title.lower()
            
            # Direct substring match
            if title_query_lower in event_title_lower:
                return event
            
            # Check for city + summit pattern variations
            # "Dallas Summit" should match "Dall ACN Summit" (with corruption)
            if 'summit' in title_query_lower:
                # Extract city from query
                cities = ['dallas', 'calgary', 'toronto', 'orlando', 'salt lake', 'chicago', 'las vegas']
                for city in cities:
                    if city in title_query_lower:
                        # Check if event title contains corruption of that city
                        city_corruptions = {
                            'dallas': ['dall acn', 'dall', 'dall a', 'dall '],
                            'calgary': ['calg', 'cal ', 'calgary'.replace('a', '')],
                            'toronto': ['tor', 'tor '],
                            'orlando': ['orl', 'orl '],
                        }
                        if city in city_corruptions:
                            for corruption in city_corruptions[city]:
                                if corruption in event_title_lower:
                                    return event
                        # Also check if city name is in event title
                        if city in event_title_lower:
                            return event
        
        return None
    
    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        """Get all events of a specific type"""
        return [e for e in self._events.values() if e.event_type == event_type]
    
    def print_summary(self):
        """Print summary of stored events"""
        print(f"\n{'='*60}")
        print(f"EVENT STORE SUMMARY")
        print(f"{'='*60}")
        print(f"Total events: {len(self._events)}")
        
        # Count by type
        type_counts = {}
        for event in self._events.values():
            type_counts[event.event_type] = type_counts.get(event.event_type, 0) + 1
        
        print("\nBy type:")
        for event_type, count in type_counts.items():
            print(f"  {event_type.value}: {count}")
        
        # Count by city
        city_counts = {}
        for event in self._events.values():
            if event.city:
                city_counts[event.city] = city_counts.get(event.city, 0) + 1
        
        print("\nBy city:")
        for city, count in sorted(city_counts.items()):
            print(f"  {city}: {count}")
        
        # Upcoming events
        upcoming = self.get_upcoming_events()
        print(f"\nUpcoming events: {len(upcoming)}")
        
        print(f"{'='*60}\n")


def process_scraped_pages(
    raw_dir: str = "./acn_data/raw",
    output_dir: str = "./acn_data"
) -> EventStore:
    """
    Process all scraped pages and extract events.
    
    Args:
        raw_dir: Directory containing scraped page JSON files
        output_dir: Directory for output data
    
    Returns:
        EventStore with all extracted events
    """
    print(f"\n{'='*60}")
    print("PROCESSING SCRAPED PAGES FOR EVENTS")
    print(f"{'='*60}")
    
    # Initialize store
    store = EventStore(output_dir)
    
    # Get all scraped pages
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        print(f"⚠ Raw data directory not found: {raw_dir}")
        return store
    
    json_files = list(raw_path.glob("*.json"))
    print(f"Found {len(json_files)} scraped pages\n")
    
    # Process each page
    events_found = 0
    pages_checked = 0
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                page_data = json.load(f)
            
            pages_checked += 1
            
            url = page_data.get('url', '')
            title = page_data.get('title', '')
            content = page_data.get('content', '')
            html_content = page_data.get('raw_html', '')
            
            # Check if this is an event page
            is_event, event_type, parent_id = EventPageDetector.is_event_detail_page(
                url, content, title
            )
            
            # if not is_event:
            #     continue
            # MAIN EVENT PAGE
            if is_event and not parent_id:
                event = EventDataExtractor.extract_event_data(
                    url=url,
                    title=title,
                    content=content,
                    html_content=html_content,
                    event_type=event_type or "Event",
                    parent_event_id=None,
                )
                store.add_event(event)
                events_found += 1
                continue

            # SUBPAGE → MERGE INTO PARENT
            if parent_id:
                parent_event = store.get_event(parent_id)
                if not parent_event:
                    continue  # parent not processed yet

                merged = EventDataExtractor.extract_event_data(
                    url=url,
                    title=title,
                    content=content,
                    html_content=html_content,
                    event_type=parent_event.event_type.value,
                    parent_event_id=parent_id,
                )

                # MERGE RULES (CRITICAL)
                if merged.registration_url and not parent_event.registration_url:
                    parent_event.registration_url = merged.registration_url

                if merged.agenda and not parent_event.agenda:
                    parent_event.agenda = merged.agenda

                if merged.speakers:
                    parent_event.speakers = list(
                        dict.fromkeys((parent_event.speakers or []) + merged.speakers)
                    )

                # NEVER overwrite location or dates from subpages
                store.add_event(parent_event)
 
            # # Extract event data
            # event = EventDataExtractor.extract_event_data(
            #     url=url,
            #     title=title,
            #     content=content,
            #     html_content=html_content,
            #     event_type=event_type or "Event",
            #     parent_event_id=parent_id,
            # )
            
            # # Add to store
            # store.add_event(event)
            # events_found += 1
            
           # print(f"  ✓ {event.title}")
            print(f"    Type: {event.event_type.value}")
            print(f"    Location: {event.location_str}")
            print(f"    Date: {event.date_range_str}")
            print()
            
        except Exception as e:
            print(f"  ⚠ Error processing {json_file.name}: {e}")
            continue
    
    # Save events
    store.save_events()
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Pages checked: {pages_checked}")
    print(f"Events extracted: {events_found}")
    print(f"Events stored: {len(store.get_all_events())}")
    
    # Print summary
    store.print_summary()
    
    return store


if __name__ == "__main__":
    # Process all scraped pages
    store = process_scraped_pages()
    
    # Test queries
    print("\n" + "="*60)
    print("TEST QUERIES")
    print("="*60)
    
    # Show upcoming events
    upcoming = store.get_upcoming_events()
    print(f"\nUpcoming events ({len(upcoming)}):")
    for event in upcoming[:5]:
        print(f"  - {event.title}: {event.date_range_str} ({event.location_str})")
    
    # Show Dallas Summit
    dallas = store.get_events_by_city("Dallas")
    print(f"\nDallas events ({len(dallas)}):")
    for event in dallas:
        print(f"  - {event.title}: {event.date_range_str}")
        print(f"    Registration: {event.registration_url}")

