#!/usr/bin/env python3
"""
ACN Event Query Handler
Handles event-related queries with structured data lookup first.
Implements deterministic date filtering for "upcoming" queries.
"""

import re
from datetime import datetime, date, timedelta
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from event_models import Event, EventType, EventFilter


class EventQueryIntent(Enum):
    """Event query intent types"""
    EVENT_DETAIL = "event_detail"      # "Tell me about Dallas Summit"
    EVENT_LIST = "event_list"          # "Show me upcoming events"
    EVENT_COUNT = "event_count"        # "How many events are upcoming"
    UNKNOWN = "unknown"


@dataclass
class EventQueryResult:
    """Result of event query processing"""
    intent: EventQueryIntent
    events: List[Event]
    formatted_answer: str
    source_info: str
    confidence: float
    processing_time: float


class EventQueryClassifier:
    """
    Classifies event-related queries into intent types.
    """
    
    # Patterns for EVENT_DETAIL queries
    # IMPORTANT: These must be checked AFTER LIST_PATTERNS to avoid false positives
    DETAIL_PATTERNS = [
        r"tell me about\s+(?:the\s+)?(.+)",
        r"what is\s+(?:the\s+)?(.+)",
        r"describe\s+(?:the\s+)?(.+)",
        r"info(?:rmation)?\s+about\s+(?:the\s+)?(.+)",
        r"when is\s+(?:the\s+)?(.+)",
        r"where is\s+(?:the\s+)?(.+)",
        r"details?\s+about\s+(?:the\s+)?(.+)",
    ]
    
    # Patterns for EVENT_LIST queries
    LIST_PATTERNS = [
        r"(?:show|list|display|find|get)\s+(?:me\s+)?(?:the\s+)?(?:upcoming\s+)?(?:acn\s+)?(?:.+?)?\s*(?:events?|summits?|conferences?)",
        r"what\s+(?:events?|summits?|conferences?)\s+(?:are\s+)?(?:there\s+)?(?:coming|upcoming|scheduled|planned)",
        r"(?:upcoming|future|next|this\s+year|next\s+month)\s+(?:events?|summits?|conferences?)",
        r"events?\s+(?:next|in|during)\s+(?:this\s+)?(?:few?\s+)?month",
        r"acn\s+(?:events?|summits?|conferences?)",
    ]
    
    # Patterns for count queries
    COUNT_PATTERNS = [
        r"how many\s+(?:events?|summits?|conferences?)",
        r"count\s+(?:of\s+)?(?:events?|summits?|conferences?)",
        r"number\s+of\s+(?:events?|summits?|conferences?)",
    ]
    
    @classmethod
    def classify(cls, query: str) -> Tuple[EventQueryIntent, Optional[str], Dict[str, Any]]:
        """
        Classify query and extract relevant entities.
        
        Returns:
            Tuple of (intent, target_entity, extracted_params)
        """
        query_lower = query.lower().strip()
        target_entity = None
        params = {}
        
        # Check for EVENT_LIST patterns FIRST (before DETAIL)
        # This is important to correctly classify queries like "Show me upcoming events"
        for pattern in cls.LIST_PATTERNS:
            if re.search(pattern, query_lower):
                # Extract temporal modifiers
                params = cls._extract_temporal_params(query_lower)
                return EventQueryIntent.EVENT_LIST, None, params
        
        # Check for EVENT_DETAIL patterns
        for pattern in cls.DETAIL_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                target = match.group(1).strip()
                # Clean up the target
                target = re.sub(r'^(the|a|an)\s+', '', target)
                if target and len(target) > 2:
                    target_entity = target
                    return EventQueryIntent.EVENT_DETAIL, target_entity, params
        
        # Check for COUNT patterns
        for pattern in cls.COUNT_PATTERNS:
            if re.search(pattern, query_lower):
                params = cls._extract_temporal_params(query_lower)
                return EventQueryIntent.EVENT_COUNT, None, params
        
        return EventQueryIntent.UNKNOWN, None, {}
    
    @staticmethod
    def _extract_temporal_params(query: str) -> Dict[str, Any]:
        """Extract temporal parameters from query"""
        params = {}
        today = date.today()
        
        # Check for specific time periods
        # Check "next month" FIRST as it's more specific than just "month"
        if re.search(r'next\s+(?:few?\s+)?month', query):
            # Next month
            first_of_next_month = today.replace(day=1)
            if first_of_next_month.month == 12:
                last_of_next_month = first_of_next_month.replace(
                    year=first_of_next_month.year + 1, month=1, day=1
                ) - timedelta(days=1)
            else:
                last_of_next_month = first_of_next_month.replace(
                    month=first_of_next_month.month + 1, day=1
                ) - timedelta(days=1)
            params['start_date_from'] = first_of_next_month.isoformat()
            params['start_date_to'] = last_of_next_month.isoformat()
            params['temporal_mode'] = 'next_month'
        
        elif re.search(r'this\s+(?:few?\s+)?month', query):
            # This month
            params['start_date_from'] = today.replace(day=1).isoformat()
            first_of_next_month = today.replace(day=1)
            if first_of_next_month.month == 12:
                last_of_this_month = first_of_next_month.replace(
                    year=first_of_next_month.year + 1, month=1, day=1
                ) - timedelta(days=1)
            else:
                last_of_this_month = first_of_next_month.replace(
                    month=first_of_next_month.month + 1, day=1
                ) - timedelta(days=1)
            params['start_date_to'] = last_of_this_month.isoformat()
            params['temporal_mode'] = 'this_month'
        
        elif re.search(r'(?:events?|summits?|conferences?)\s+(?:in|during|next)\s+(?:the\s+)?(?:few?\s+)?month', query):
            # Handle "events next month", "events in next month", "events during next month"
            first_of_next_month = today.replace(day=1)
            if first_of_next_month.month == 12:
                last_of_next_month = first_of_next_month.replace(
                    year=first_of_next_month.year + 1, month=1, day=1
                ) - timedelta(days=1)
            else:
                last_of_next_month = first_of_next_month.replace(
                    month=first_of_next_month.month + 1, day=1
                ) - timedelta(days=1)
            params['start_date_from'] = first_of_next_month.isoformat()
            params['start_date_to'] = last_of_next_month.isoformat()
            params['temporal_mode'] = 'next_month'
        
        elif re.search(r'this\s+year|this\s+year\'?s?', query):
            # This year
            params['start_date_from'] = today.isoformat()
            params['start_date_to'] = today.replace(month=12, day=31).isoformat()
            params['temporal_mode'] = 'this_year'
        
        elif re.search(r'next\s+year|next\s+year\'?s?', query):
            # Next year
            next_year = today.year + 1
            params['start_date_from'] = f"{next_year}-01-01"
            params['start_date_to'] = f"{next_year}-12-31"
            params['temporal_mode'] = 'next_year'
        
        elif re.search(r'(?:in\s+)?(\d{4})\b', query):
            # Specific year (e.g., "in 2026", "events 2026")
            year_match = re.search(r'(?:in\s+)?(\d{4})\b', query)
            year = int(year_match.group(1))
            params['start_date_from'] = f"{year}-01-01"
            params['start_date_to'] = f"{year}-12-31"
            params['temporal_mode'] = 'specific_year'
            params['year'] = year
        
        elif re.search(r'upcoming|future|next|coming', query):
            # Default upcoming (today and beyond)
            params['start_date_from'] = today.isoformat()
            params['upcoming_only'] = True
            params['temporal_mode'] = 'upcoming'
        
        # Check for event type filter
        # IMPORTANT: Only set event_type if a SPECIFIC type is mentioned
        # "Show me upcoming events" should return ALL event types, not just type="Event"
        if re.search(r'\bsummit[s]?\b', query, re.IGNORECASE):
            params['event_type'] = 'Summit'
        elif re.search(r'\bconference[s]?\b', query, re.IGNORECASE):
            params['event_type'] = 'Conference'
        elif re.search(r'\bwebinar[s]?\b', query, re.IGNORECASE):
            params['event_type'] = 'Webinar'
        # Don't set event_type for generic "event" or "events" - return all types
        
        # Check for city filter
        cities = ['Dallas', 'Calgary', 'Toronto', 'Orlando', 'Salt Lake City', 'Chicago', 'Las Vegas']
        for city in cities:
            if re.search(rf'\b{city}\b', query, re.IGNORECASE):
                params['city'] = city
                break
        
        return params


class DateFilterEngine:
    """
    Deterministic date filtering for event queries.
    Implements exact date range logic without LLM interpretation.
    """
    
    @staticmethod
    def get_upcoming_date_range() -> Tuple[str, str]:
        """
        Get date range for "upcoming" events.
        
        Returns:
            Tuple of (start_date, end_date) in ISO format
            start_date = today
            end_date = end of next year
        """
        today = date.today()
        end_date = today.replace(year=today.year + 1, month=12, day=31)
        return today.isoformat(), end_date.isoformat()
    
    @staticmethod
    def get_month_date_range(year: int, month: int) -> Tuple[str, str]:
        """
        Get date range for a specific month.
        
        Args:
            year: 4-digit year
            month: 1-12 month number
        
        Returns:
            Tuple of (first_day, last_day) in ISO format
        """
        first_day = date(year, month, 1)
        
        if month == 12:
            last_day = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = date(year, month + 1, 1) - timedelta(days=1)
        
        return first_day.isoformat(), last_day.isoformat()
    
    @staticmethod
    def get_year_date_range(year: int) -> Tuple[str, str]:
        """
        Get date range for a specific year.
        
        Args:
            year: 4-digit year
        
        Returns:
            Tuple of (jan_1, dec_31) in ISO format
        """
        return f"{year}-01-01", f"{year}-12-31"
    
    @staticmethod
    def get_next_month_range() -> Tuple[str, str]:
        """Get date range for next month"""
        today = date.today()
        first_of_next_month = today.replace(day=1)
        
        if first_of_next_month.month == 12:
            next_month = 1
            next_year = first_of_next_month.year + 1
        else:
            next_month = first_of_next_month.month + 1
            next_year = first_of_next_month.year
        
        return DateFilterEngine.get_month_date_range(next_year, next_month)
    
    @staticmethod
    def get_this_month_range() -> Tuple[str, str]:
        """Get date range for current month"""
        today = date.today()
        return DateFilterEngine.get_month_date_range(today.year, today.month)
    
    @staticmethod
    def filter_events_by_date(
        events: List[Event],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        upcoming_only: bool = False
    ) -> List[Event]:
        """
        Filter events by date criteria.
        
        Args:
            events: List of events to filter
            start_date: Filter for events on or after this date (YYYY-MM-DD)
            end_date: Filter for events on or before this date (YYYY-MM-DD)
            upcoming_only: If True, filter for events with start_date >= today
        
        Returns:
            Filtered list of events sorted by start_date
        """
        today = date.today()
        filtered = []
        
        for event in events:
            # Skip events without dates for date-based queries
            if not event.start_date:
                if not upcoming_only and not start_date and not end_date:
                    filtered.append(event)
                continue
            
            try:
                event_date = datetime.strptime(event.start_date, "%Y-%m-%d").date()
            except ValueError:
                # Invalid date format, skip
                continue
            
            # Apply filters
            if upcoming_only and event_date < today:
                continue
            
            if start_date and event_date < date.fromisoformat(start_date):
                continue
            
            if end_date and event_date > date.fromisoformat(end_date):
                continue
            
            filtered.append(event)
        
        # Sort by start date
        filtered.sort(key=lambda e: e.start_date or "9999-12-31")
        
        return filtered


class EventQueryHandler:
    """
    Main handler for ACN event queries.
    Routes queries to appropriate handlers based on intent.
    """
    
    def __init__(self, event_store: 'EventStore'):
        """
        Initialize with an EventStore instance.
        
        Args:
            event_store: EventStore instance for querying events
        """
        self.store = event_store
        self.classifier = EventQueryClassifier()
        self.date_filter = DateFilterEngine()
    
    def query(self, query: str) -> EventQueryResult:
        """
        Process an event-related query.
        
        Args:
            query: User query string
        
        Returns:
            EventQueryResult with formatted answer
        """
        import time
        start_time = time.time()
        
        # Classify the query
        intent, target_entity, params = self.classifier.classify(query)
        
        # Handle based on intent
        if intent == EventQueryIntent.EVENT_DETAIL:
            result = self._handle_detail_query(target_entity, params)
        elif intent == EventQueryIntent.EVENT_LIST:
            result = self._handle_list_query(params)
        elif intent == EventQueryIntent.EVENT_COUNT:
            result = self._handle_count_query(params)
        else:
            # Not an event query - return empty result
            result = EventQueryResult(
                intent=EventQueryIntent.UNKNOWN,
                events=[],
                formatted_answer="This doesn't appear to be an event-related query.",
                source_info="",
                confidence=0.0,
                processing_time=time.time() - start_time,
            )
        
        result.processing_time = time.time() - start_time
        return result
    
    def _handle_detail_query(
        self,
        target: str,
        params: Dict[str, Any]
    ) -> EventQueryResult:
        """Handle event detail query like 'Tell me about Dallas Summit'"""
        
        # Clean up target (remove event type words)
        target = re.sub(r'\s*(summit|event|conference)\s*', '', target, flags=re.IGNORECASE)
        target = target.strip()
        
        # Try to find event by title (case-insensitive partial match)
        event = self.store.get_event_by_title(target)
        
        # If not found, try city match (look for events in that city)
        if not event:
            # Check if target is a city name - look for upcoming events in that city
            cities = ['Dallas', 'Calgary', 'Toronto', 'Orlando', 'Salt Lake City', 'Chicago', 'Las Vegas']
            for city in cities:
                if city.lower() in target.lower():
                    events = self.store.get_events_by_city(city)
                    if events:
                        # Get the next upcoming event in that city
                        upcoming = [e for e in events if e.is_upcoming]
                        event = upcoming[0] if upcoming else events[0]
                        break
        
        # If still not found, try fuzzy matching on title
        if not event:
            all_events = self.store.get_all_events()
            target_lower = target.lower()
            
            # Try to match with common variations/corruptions
            for ev in all_events:
                title_lower = ev.title.lower()
                # Check for partial matches
                if (target_lower in title_lower or 
                    title_lower.replace('acn', '').strip() == target_lower.replace('acn', '').strip() or
                    title_lower.replace('dall', 'dallas') == target_lower or
                    'dall' in title_lower and 'dallas' in target_lower):
                    event = ev
                    break
        
        if not event:
            return EventQueryResult(
                intent=EventQueryIntent.EVENT_DETAIL,
                events=[],
                formatted_answer=f"I couldn't find an event matching '{target}'. Please check the event name and try again.",
                source_info="",
                confidence=0.0,
                processing_time=0.0,
            )
        
        # Build formatted answer
        answer = self._format_event_detail(event)
        
        confidence = 1.0
        if not event.start_date or not event.city:
            confidence = 0.7
        if event.parent_event_id:
            confidence = min(confidence, 0.85)

        return EventQueryResult(
            intent=EventQueryIntent.EVENT_DETAIL,
            events=[event],
            formatted_answer=answer,
            source_info=f"Source: {event.source_url}",
            confidence=confidence,
            processing_time=0.0,
        )
    
    def _handle_list_query(
        self,
        params: Dict[str, Any]
    ) -> EventQueryResult:
        """Handle event list query like 'Show me upcoming events'"""
        
        # Build filter from params
        filter_criteria = EventFilter(
            event_type=EventType(params.get('event_type')) if params.get('event_type') else None,
            city=params.get('city'),
            start_date_from=params.get('start_date_from'),
            start_date_to=params.get('start_date_to'),
            upcoming_only=params.get('upcoming_only', False),
        )
        
        # Query events
        events = self.store.query_events(filter_criteria)
        # 🚫 Never return subpages (program, registration, exhibitors, etc.)
        events = [e for e in events if e.parent_event_id is None]

        
        # If no temporal params, default to upcoming
        # if not params.get('start_date_from') and not params.get('start_date_to') and not params.get('upcoming_only'):
        #     # Default to upcoming
        #     events = self.store.get_upcoming_events(
        #         event_type=EventType(params.get('event_type')) if params.get('event_type') else None
        #     )
        events = self.store.query_events(filter_criteria)
        # 🚫 Never return subpages (program, registration, exhibitors, etc.)
        events = [e for e in events if e.parent_event_id is None]

        
        # Format answer
        answer = self._format_event_list(events, params)
        
        source_info = f"Found {len(events)} events"
        
        if not events:
            confidence = 0.4
        else:
            confidence = min(1.0, 0.6 + 0.1 * len(events))

        return EventQueryResult(
            intent=EventQueryIntent.EVENT_LIST,
            events=events,
            formatted_answer=answer,
            source_info=source_info,
            confidence=confidence,
            processing_time=0.0,
        )

    
    def _handle_count_query(
        self,
        params: Dict[str, Any]
    ) -> EventQueryResult:
        """Handle event count query like 'How many upcoming events'"""
        
        filter_criteria = EventFilter(
            event_type=EventType(params.get('event_type')) if params.get('event_type') else None,
            start_date_from=params.get('start_date_from'),
            start_date_to=params.get('start_date_to'),
            upcoming_only=params.get('upcoming_only', False),
        )
        
        events = self.store.query_events(filter_criteria)
        # 🚫 Never return subpages (program, registration, exhibitors, etc.)
        events = [e for e in events if e.parent_event_id is None]

        
        answer = f"There {'is' if len(events) == 1 else 'are'} {len(events)} upcoming ACN event{'s' if len(events) != 1 else ''}."
        
        return EventQueryResult(
            intent=EventQueryIntent.EVENT_COUNT,
            events=events,
            formatted_answer=answer,
            source_info=f"Counted {len(events)} events",
            confidence=1.0,
            processing_time=0.0,
        )
    
    def _format_event_detail(self, event: Event) -> str:
        """Format single event as readable answer"""
        lines = []
        
        lines.append(f"## {event.title}")
        lines.append("")
        
        if event.description:
            lines.append(f"**Description:** {event.description}")
            lines.append("")
        
        # Date and time
        if event.start_date:
            lines.append(f"**Date:** {event.date_range_str}")
            lines.append("")
        
        # Location
        if event.city or event.state or event.venue:
            location_parts = []
            if event.venue:
                location_parts.append(event.venue)
            if event.city:
                location_parts.append(event.city)
            if event.state:
                location_parts.append(event.state)
            lines.append(f"**Location:** {', '.join(location_parts)}")
            lines.append("")
        
        # Registration
        if event.registration_url:
            lines.append(f"**Registration:** {event.registration_url}")
            lines.append("")
        
        # Source
        lines.append(f"_Source: {event.source_url}_")
        
        return '\n'.join(lines)
    
    def _format_event_list(
        self,
        events: List[Event],
        params: Dict[str, Any]
    ) -> str:
        """Format list of events as readable answer"""
        if not events:
            # Determine the context for the "no results" message
            temporal = params.get('temporal_mode', 'upcoming')
            if temporal == 'upcoming':
                return "No upcoming ACN events found."
            elif temporal == 'next_month':
                return "No ACN events found for next month."
            elif temporal == 'specific_year':
                year = params.get('year', '')
                return f"No ACN events found for {year}."
            else:
                return "No ACN events found for the specified criteria."
        
        # Build the answer
        lines = []
        
        # Add header
        temporal_mode = params.get('temporal_mode', 'upcoming')
        if temporal_mode == 'upcoming':
            lines.append("## Upcoming ACN Events")
        elif temporal_mode == 'next_month':
            lines.append("## ACN Events Next Month")
        elif temporal_mode == 'specific_year':
            year = params.get('year', '')
            lines.append(f"## ACN Events in {year}")
        elif temporal_mode == 'this_year':
            lines.append("## ACN Events This Year")
        else:
            lines.append("## ACN Events")
        
        lines.append("")
        
        # Add event count
        lines.append(f"_Found {len(events)} event{'s' if len(events) != 1 else ''}_\n")
        
        # List each event
        for i, event in enumerate(events, 1):
            lines.append(f"### {i}. {event.title}")
            lines.append("")
            
            if event.description:
                # Truncate long descriptions
                desc = event.description[:200] + "..." if len(event.description) > 200 else event.description
                lines.append(f"_{desc}_")
                lines.append("")
            
            details = []
            if event.start_date:
                details.append(f"📅 {event.date_range_str}")
            if event.city:
                details.append(f"📍 {event.location_str}")
            if event.registration_url:
                details.append(f"🔗 [Register]({event.registration_url})")
            
            if details:
                lines.append(" | ".join(details))
                lines.append("")
            
            lines.append(f"_Source: {event.source_url}_")
            lines.append("")
        
        return '\n'.join(lines)


# Convenience function for quick integration
def handle_event_query(query: str, event_store: 'EventStore') -> EventQueryResult:
    """
    Quick function to handle an event query.
    
    Args:
        query: User query string
        event_store: EventStore instance
    
    Returns:
        EventQueryResult
    """
    handler = EventQueryHandler(event_store)
    return handler.query(query)


if __name__ == "__main__":
    # Test the query handler
    from event_extraction import EventStore
    
    # Load events
    store = EventStore()
    
    # Create handler
    handler = EventQueryHandler(store)
    
    # Test queries
    test_queries = [
        "Tell me about Dallas Summit",
        "When is the Dallas Summit?",
        "Show me upcoming events",
        "Show me upcoming summits",
        "What ACN events are happening next month?",
        "Upcoming ACN events in 2026",
        "How many upcoming events are there?",
    ]
    
    print("\n" + "="*70)
    print("EVENT QUERY HANDLER TESTS")
    print("="*70 + "\n")
    
    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 50)
        
        result = handler.query(query)
        
        print(f"Intent: {result.intent.value}")
        print(f"Events found: {len(result.events)}")
        print(f"\nAnswer:\n{result.formatted_answer}")
        print("\n" + "="*70 + "\n")

