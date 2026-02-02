# response_adapter.py
import re
from typing import Dict, List

# -----------------------------
# Helpers
# -----------------------------

DATE_REGEX = r"(\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b)"
TIME_REGEX = r"(\b\d{1,2}:\d{2}\s?(?:AM|PM|am|pm)?\b)"
LOCATION_KEYWORDS = [
    "online", "virtual", "las vegas", "dallas", "chicago",
    "new york", "calgary", "washington", "texas", "canada", "usa"
]


def looks_like_event_block(text: str) -> bool:
    """Check if text contains event indicators"""
    has_date = bool(re.search(DATE_REGEX, text))
    has_location = any(loc in text.lower() for loc in LOCATION_KEYWORDS)
    has_event_keywords = any(keyword in text.lower() for keyword in ['event', 'summit', 'conference', 'webinar', 'ama', 'session', 'registration'])
    
    # Match if has date/location OR event keywords with date/location
    return (has_event_keywords and (has_date or has_location)) or (has_date and has_location)


# -----------------------------
# Event Extraction
# -----------------------------

def extract_events_from_text(text: str) -> List[Dict]:
    """
    Extract multiple events from free-form LLM text.
    Handles events marked as "1. Title... 2. Title... 3. Title..."
    Key insight: Event numbers (1, 2, 3) are single digits followed by ". " and CAPITAL letter.
    Date patterns like "2024." have 4 digits, so we can distinguish.
    """
    events = []
    
    # Pattern: single digit (1-9), period, space, THEN capital letter
    # This distinguishes "1. Event Name" from "2024. Location"
    # Split pattern: Find positions where "N. " occurs before a capital letter
    split_pattern = r'(?<![0-9])\b([1-9])\.\s+(?=[A-Z])'
    
    # Split by this pattern (keeping the separators)
    parts = re.split(split_pattern, text)
    
    # Process: ['', '1', 'content', '2', 'content', '3', 'content', '']
    # pairs of (number, content)
    i = 1
    while i < len(parts):
        if i + 1 < len(parts):
            event_num = parts[i]  # "1", "2", "3"
            event_text = parts[i + 1]  # The content after "N. "
            
            event_text = event_text.strip()
            if not event_text or len(event_text) < 5:
                i += 2
                continue
            
            # Check if this looks like an event
            if not looks_like_event_block(event_text):
                i += 2
                continue
            
            # Extract title: text up to first date or location keyword
            title_end = len(event_text)
            
            # Look for "Date:" or "Location:" or " - " to find title boundary
            date_pos = event_text.find('Date:')
            location_pos = event_text.find('Location:')
            dash_pos = event_text.find(' - ')
            
            # Use the first boundary found
            boundaries = [p for p in [date_pos, location_pos, dash_pos] if p != -1]
            if boundaries:
                title_end = min(boundaries)
            
            # Also limit by period or sentence end
            period_pos = event_text.find('. ')
            if period_pos != -1 and period_pos < title_end:
                title_end = period_pos + 1
            
            title = event_text[:title_end].strip()
            # Remove trailing punctuation
            title = re.sub(r'[\.\-\s]+$', '', title)
            title = (title[:150] + "...") if len(title) > 150 else title
            
            # Extract date using regex
            date_match = re.search(DATE_REGEX, event_text)
            date_str = None
            
            # Also try to match patterns like "Date: May 15-17, 2024"
            date_pattern_match = re.search(r'Date:\s*([^.]+?)(?:\.|Location:|$)', event_text, re.IGNORECASE)
            if date_pattern_match:
                date_str = date_pattern_match.group(1).strip()
            elif date_match:
                date_str = date_match.group(0)
            
            # Extract time
            time_match = re.search(TIME_REGEX, event_text)
            time_str = time_match.group(0) if time_match else None
            
            # Extract location
            location_str = None
            location_match = re.search(r'(?:Location|Venue):\s*([^.]+?)(?:\.|$)', event_text, re.IGNORECASE)
            if location_match:
                location_str = location_match.group(1).strip()
                location_str = (location_str[:100] + "...") if len(location_str) > 100 else location_str
            
            # If no explicit location found, check for location keywords in remaining text
            if not location_str:
                remaining = event_text[event_text.find('Location:') + 9:] if 'Location:' in event_text else event_text
                for line in remaining.split('.'):
                    for loc_keyword in LOCATION_KEYWORDS:
                        if loc_keyword in line.lower():
                            location_str = line.strip()
                            location_str = (location_str[:100] + "...") if len(location_str) > 100 else location_str
                            break
                    if location_str:
                        break
            
            # Only add if we have a meaningful title
            if title and len(title) > 3:
                events.append({
                    "title": title,
                    "date": date_str,
                    "time": time_str,
                    "location": location_str,
                    "raw_text": event_text
                })
            
            i += 2
        else:
            i += 1
    
    return events


# -----------------------------
# Rich Message Blocks
# -----------------------------

def text_to_blocks(text: str) -> List[Dict]:
    """
    Convert plain / markdown-ish text into UI blocks
    Handles:
    - Bullet lists (-) 
    - Numbered lists (1., 2., 3., etc) on separate lines
    - Numbered patterns inline (1. Text 2. Text 3. Text)
    - Regular paragraphs
    """
    blocks: List[Dict] = []
    
    # First, try to detect inline numbered patterns like "1. Text 2. Text 3. Text"
    # This regex finds patterns like "1. Something", "2. Something", etc
    inline_numbered_pattern = r"\d+\.\s+[^0-9]*?(?=\d+\.|$)"
    inline_matches = re.findall(inline_numbered_pattern, text)
    
    if inline_matches and len(inline_matches) >= 2:
        # This looks like an inline numbered list
        bullet_items = []
        for match in inline_matches:
            # Remove the number prefix
            item_text = re.sub(r"^\d+\.\s+", "", match.strip()).strip()
            if item_text:
                bullet_items.append(item_text)
        
        if bullet_items:
            blocks.append({
                "type": "list",
                "items": bullet_items
            })
            return blocks  # Done - return just the list
    
    # Fallback: Process line by line for proper line-broken numbered lists
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    bullet_items = []

    for line in lines:
        # Check if line is a bullet point
        is_bullet = line.startswith("-") or line.startswith("•")
        
        # Check if line is a numbered point (1., 2., etc)
        is_numbered = bool(re.match(r"^\d+\.\s+", line))
        
        if is_bullet or is_numbered:
            # Extract the text without the bullet/number prefix
            if is_bullet:
                item_text = line.lstrip("-• ").strip()
            else:  # numbered
                item_text = re.sub(r"^\d+\.\s+", "", line).strip()
            
            bullet_items.append(item_text)
        else:
            # This is a paragraph line
            if bullet_items:
                # Save the accumulated list before starting a paragraph
                blocks.append({
                    "type": "list",
                    "items": bullet_items
                })
                bullet_items = []
            
            # Only add non-empty paragraphs
            if line:
                blocks.append({
                    "type": "paragraph",
                    "text": line
                })

    # Don't forget remaining bullet items
    if bullet_items:
        blocks.append({
            "type": "list",
            "items": bullet_items
        })

    return blocks


# -----------------------------
# Event Attribute Extraction
# -----------------------------

def extract_event_from_attributes(text: str) -> Dict:
    """
    Extract a single event from attribute-format answer.
    Format: "1. Event Name: X 2. Date & Time: Y 3. Location: Z 4. Description: ..."
    """
    event = {
        "title": None,
        "date": None,
        "time": None,
        "location": None,
        "raw_text": text
    }
    
    # Extract Event Name
    name_match = re.search(r'Event Name:\s*([^\n.]+?)(?:\d+\.|$)', text)
    if name_match:
        event["title"] = name_match.group(1).strip()[:150]
    
    # Extract Date & Time (may be combined)
    date_time_match = re.search(r'Date[s]?[^:]*?:\s*([^\n.]+?)(?:\d+\.|$)', text, re.IGNORECASE)
    if date_time_match:
        date_time_text = date_time_match.group(1).strip()
        
        # Try to extract just the date part
        date_match = re.search(DATE_REGEX, date_time_text)
        if date_match:
            event["date"] = date_match.group(0)
        
        # Try to extract time
        time_match = re.search(TIME_REGEX, date_time_text)
        if time_match:
            event["time"] = time_match.group(0)
    
    # Extract Location
    location_match = re.search(r'Location:\s*([^\n.]+?)(?:\d+\.|$)', text, re.IGNORECASE)
    if location_match:
        location_text = location_match.group(1).strip()
        event["location"] = (location_text[:100] + "...") if len(location_text) > 100 else location_text
    
    # Only return if we have a title
    if event["title"]:
        return event
    
    return None


# ---------------------
# Main Adapter
# ---------------------

def adapt_llm_response(answer: str, intent: str) -> Dict:
    """
    Single entry point for ALL LLM output.
    Decides UI type safely.
    
    Logic:
    1. If intent == "events", extract events from answer
    2. FALLBACK: If answer contains event-like content, try to extract events anyway
    3. FALLBACK 2: If answer is event attributes (Event Name, Date, Location), convert to event card
    4. Otherwise: Return formatted text blocks (point-wise list)
    """

    # 1️⃣ Primary: Check explicit intent
    if intent == "events":
        events = extract_events_from_text(answer)
        if events:
            return {
                "response_type": "event_cards",
                "data": {
                    "events": events
                }
            }
    
    # 2️⃣ FALLBACK: Try to detect events from content even if intent isn't "events"
    # This handles cases where RAG classifier says "general" but answer has event cards
    event_keywords = ['event', 'summit', 'conference', 'webinar', 'ama', 'session', 'registration', 'networking']
    has_event_keywords = any(keyword in answer.lower() for keyword in event_keywords)
    has_date = bool(re.search(DATE_REGEX, answer))
    
    # If answer looks like events, try to extract them
    if has_event_keywords and has_date:
        events = extract_events_from_text(answer)
        if events and len(events) >= 1:  # Require at least 1 valid event
            return {
                "response_type": "event_cards",
                "data": {
                    "events": events
                }
            }
    
    # 3️⃣ FALLBACK 2: Check if answer is in event attributes format
    # "1. Event Name: ... 2. Date & Time: ... 3. Location: ..."
    if "Event Name:" in answer or ("event" in answer.lower() and "date" in answer.lower() and "location" in answer.lower()):
        # Try to extract single event from attribute format
        event = extract_event_from_attributes(answer)
        if event:
            return {
                "response_type": "event_cards",
                "data": {
                    "events": [event]
                }
            }

    # 4️⃣ Default → rich message (point-wise formatted)
    return {
        "response_type": "rich_message",
        "data": {
            "blocks": text_to_blocks(answer)
        }
    }
