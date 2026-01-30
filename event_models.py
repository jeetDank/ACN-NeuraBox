#!/usr/bin/env python3
"""
ACN Event Data Models
Structured event representation for ACN summits, conferences, and events.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from enum import Enum
import json
import hashlib
import re


class EventType(Enum):
    SUMMIT = "Summit"
    CONFERENCE = "Conference"
    EVENT = "Event"
    WEBINAR = "Webinar"
    TRAINING = "Training"
    AMA = "AMA"
    WORKSHOP = "Workshop"
    NETWORKING = "Networking"


@dataclass
class Event:
    """
    Structured ACN event representation.
    
    All fields extracted directly from ACN website pages.
    DO NOT infer or hallucinate missing values.
    """
    # Core identity
    event_id: str                      # Stable ID derived from URL
    title: str                         # Event title from page
    event_type: EventType              # Summit, Conference, etc.
    
    # Location
    city: Optional[str] = None         # e.g., "Dallas"
    state: Optional[str] = None        # e.g., "TX"
    venue: Optional[str] = None        # e.g., "Hyatt Regency Dallas"
    
    # Dates (ISO format YYYY-MM-DD)
    start_date: Optional[str] = None   # e.g., "2026-02-25"
    end_date: Optional[str] = None     # e.g., "2026-02-26"
    
    # Content (verbatim from page)
    description: Optional[str] = None  # Event description
    agenda: Optional[str] = None       # Agenda/schedule
    speakers: Optional[List[str]] = None  # Speaker names
    
    # URLs
    registration_url: Optional[str] = None  # Registration link
    source_url: str = ""             # Original ACN URL
    
    # Metadata
    scraped_at: str = field(default_factory=lambda: datetime.now().isoformat())
    parent_event_id: Optional[str] = None  # For subpages (program, speakers)
    
    @classmethod
    def generate_event_id(cls, url: str) -> str:
        """Generate stable event ID from URL"""
        # Extract the event path from URL
        # e.g., https://www.appliedclientnetwork.org/Events/Summits/Dallas-Summit
        # → events_summits_dallas_summit
        parsed = url.rstrip('/').split('/')[-1]
        # Clean and normalize
        event_id = re.sub(r'[^a-zA-Z0-9]', '_', parsed.lower())
        # Remove duplicates and trailing underscores
        event_id = re.sub(r'_+', '_', event_id).strip('_')
        return f"acn_{event_id}"
    
    @classmethod
    def generate_event_id_from_path(cls, path: str) -> str:
        """
        Generate stable event ID from a path (for subpages).
        Returns the event_id of the parent event.
        
        Args:
            path: URL path like "/events/summits/dallas-summit"
            
        Returns:
            Event ID like "acn_events_summits_dallas_summit"
        """
        # Clean and normalize the path
        path = path.strip('/')
        if not path:
            return "acn_unknown"
        
        # Split path and extract meaningful parts
        parts = [p for p in path.split('/') if p and p.lower() not in ['events', 'event', 'summits', 'summit']]
        
        if not parts:
            # If only generic parts, use the last meaningful segment
            parts = [p for p in path.split('/') if p]
        
        # Build event_id from path parts
        event_id_parts = path.replace('/', '_').lower()
        # Clean: remove non-alphanumeric except underscore
        event_id = re.sub(r'[^a-z0-9_]', '_', event_id_parts)
        # Remove duplicates and trailing underscores
        event_id = re.sub(r'_+', '_', event_id).strip('_')
        
        return f"acn_{event_id}"
    
    @property
    def is_upcoming(self) -> bool:
        """Check if event is upcoming (start_date >= today)"""
        if not self.start_date:
            return False
        try:
            start = datetime.strptime(self.start_date, "%Y-%m-%d").date()
            return start >= date.today()
        except ValueError:
            return False
    
    @property
    def location_str(self) -> str:
        """Human-readable location string"""
        parts = []
        if self.city:
            parts.append(self.city)
        if self.state:
            parts.append(self.state)
        return ", ".join(parts) if parts else "Location TBD"
    
    @property
    def date_range_str(self) -> str:
        """Human-readable date range"""
        if not self.start_date:
            return "Date TBD"
        if self.end_date and self.end_date != self.start_date:
            return f"{self.start_date} to {self.end_date}"
        return self.start_date
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage"""
        data = {
            "event_id": self.event_id,
            "title": self.title,
            "event_type": self.event_type.value,
            "city": self.city,
            "state": self.state,
            "venue": self.venue,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "description": self.description,
            "agenda": self.agenda,
            "speakers": self.speakers,
            "registration_url": self.registration_url,
            "source_url": self.source_url,
            "scraped_at": self.scraped_at,
            "parent_event_id": self.parent_event_id,
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create Event from dictionary"""
        # Handle event_type enum
        event_type_str = data.get("event_type", "Event")
        try:
            event_type = EventType(event_type_str)
        except ValueError:
            event_type = EventType.EVENT
        
        return cls(
            event_id=data["event_id"],
            title=data["title"],
            event_type=event_type,
            city=data.get("city"),
            state=data.get("state"),
            venue=data.get("venue"),
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
            description=data.get("description"),
            agenda=data.get("agenda"),
            speakers=data.get("speakers"),
            registration_url=data.get("registration_url"),
            source_url=data["source_url"],
            scraped_at=data.get("scraped_at", datetime.now().isoformat()),
            parent_event_id=data.get("parent_event_id"),
        )
    
    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "Event":
        """Deserialize from JSON string"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class EventFilter:
    """Filter criteria for event queries"""
    event_type: Optional[EventType] = None
    city: Optional[str] = None
    state: Optional[str] = None
    start_date_from: Optional[str] = None  # YYYY-MM-DD
    start_date_to: Optional[str] = None    # YYYY-MM-DD
    upcoming_only: bool = False
    title_contains: Optional[str] = None
    
    def matches(self, event: Event) -> bool:
        """Check if event matches filter criteria"""
        # Event type filter
        if self.event_type and event.event_type != self.event_type:
            return False
        
        # City filter (case-insensitive partial match)
        if self.city and self.city.lower() not in (event.city or "").lower():
            return False
        
        # State filter
        if self.state and self.state.lower() != (event.state or "").lower():
            return False
        
        # Date range filter
        if self.start_date_from and event.start_date:
            if event.start_date < self.start_date_from:
                return False
        if self.start_date_to and event.start_date:
            if event.start_date > self.start_date_to:
                return False
        
        # Upcoming only filter
        # IMPORTANT: Include events without dates (we don't know if they're past)
        # Only exclude events that definitely have a past date
        if self.upcoming_only:
            if event.start_date:
                try:
                    from datetime import date
                    event_date = date.fromisoformat(event.start_date)
                    if event_date < date.today():
                        return False
                except ValueError:
                    pass  # Invalid date format, include the event
        
        # Title contains filter
        if self.title_contains and self.title_contains.lower() not in event.title.lower():
            return False
        
        return True


# Predefined ACN event locations for validation
KNOWN_CITIES = {
    "Dallas": "TX",
    "Calgary": "AB",
    "Toronto": "ON",
    "Orlando": "FL",
    "Salt Lake City": "UT",
    "Chicago": "IL",
    "Las Vegas": "NV",
}

# URL patterns for event page detection
EVENT_URL_PATTERNS = [
    r'/events/summits/',
    r'/events/conference/',
    r'/event/',
]

# Subpage patterns (inherit parent event)
SUBPAGE_PATTERNS = {
    "program": ["program", "agenda", "schedule"],
    "speakers": ["speakers"],
    "registration": ["registration", "register"],
    "hotel": ["hotel", "travel", "accommodations"],
    "exhibitors": ["exhibitors", "sponsors"],
}

