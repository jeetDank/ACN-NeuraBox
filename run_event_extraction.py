#!/usr/bin/env python3
"""
ACN Event Extraction Runner
Processes scraped pages and extracts structured event data.
Run this after running the web scraper.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

from event_models import Event, EventType
from event_extraction import (
    ACNTextCleaner,
    EventPageDetector,
    EventDataExtractor,
    EventStore,
    process_scraped_pages,
)


def main():
    """Main entry point for event extraction."""
    
    print("\n" + "="*70)
    print("ACN EVENT EXTRACTION PIPELINE")
    print("="*70)
    print(f"Started at: {datetime.now().isoformat()}")
    print()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        # Specific page file provided
        page_file = Path(sys.argv[1])
        if page_file.exists():
            process_single_page(page_file)
        else:
            print(f"File not found: {page_file}")
        return
    
    # Process all scraped pages
    store = process_scraped_pages(
        raw_dir="./acn_data/raw",
        output_dir="./acn_data"
    )
    
    # Print final summary
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    print(f"\nTo use extracted events:")
    print("  from event_query_handler import handle_event_query")
    print("  from event_extraction import EventStore")
    print()
    print("  store = EventStore()")
    print("  result = handle_event_query('Show me upcoming events', store)")
    print()
    
    # Verify the extracted data
    verify_extracted_events(store)


def process_single_page(page_file: Path):
    """Process a single scraped page file."""
    print(f"Processing single page: {page_file}")
    
    with open(page_file, 'r', encoding='utf-8') as f:
        page_data = json.load(f)
    
    url = page_data.get('url', '')
    title = page_data.get('title', '')
    content = page_data.get('content', '')
    html_content = page_data.get('raw_html', '')
    
    # Check if this is an event page
    is_event, event_type, parent_id = EventPageDetector.is_event_detail_page(
        url, content, title
    )
    
    print(f"\nURL: {url}")
    print(f"Title: {title}")
    print(f"Is Event Page: {is_event}")
    print(f"Event Type: {event_type}")
    
    if is_event:
        # Extract event data
        event = EventDataExtractor.extract_event_data(
            url=url,
            title=title,
            content=content,
            html_content=html_content,
            event_type=event_type or "Event",
            parent_event_id=parent_id,
        )
        
        print(f"\nExtracted Event:")
        print(f"  ID: {event.event_id}")
        print(f"  Title: {event.title}")
        print(f"  Type: {event.event_type.value}")
        print(f"  City: {event.city}")
        print(f"  State: {event.state}")
        print(f"  Start Date: {event.start_date}")
        print(f"  End Date: {event.end_date}")
        print(f"  Venue: {event.venue}")
        print(f"  Registration: {event.registration_url}")
        
        # Save to store
        store = EventStore()
        store.add_event(event)
        store.save_events()
        
        print(f"\n✓ Event saved to store")
    else:
        print("\nThis page is not an event detail page.")


def verify_extracted_events(store: EventStore):
    """Verify and display extracted events."""
    events = store.get_all_events()
    
    if not events:
        print("\n⚠ No events extracted. This may indicate:")
        print("  - Scraped data not yet generated (run scraper first)")
        print("  - Event detection logic needs adjustment")
        print("  - Pages are not being processed correctly")
        return
    
    print(f"\n✓ Verified {len(events)} events in store")
    
    # Show events by type
    by_type = {}
    for event in events:
        key = event.event_type.value
        by_type[key] = by_type.get(key, 0) + 1
    
    print("\nEvents by type:")
    for event_type, count in sorted(by_type.items()):
        print(f"  {event_type}: {count}")
    
    # Show upcoming events
    upcoming = store.get_upcoming_events()
    print(f"\nUpcoming events: {len(upcoming)}")
    
    for event in upcoming[:5]:
        print(f"  - {event.title}: {event.date_range_str} ({event.location_str})")
    
    # Test a query
    print("\n" + "-"*50)
    print("Testing event query:")
    
    from event_query_handler import EventQueryHandler
    
    handler = EventQueryHandler(store)
    
    # Test queries
    test_queries = [
        "Show me upcoming events",
        "Tell me about Dallas Summit",
        "What summits are coming up?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        result = handler.query(query)
        print(f"Intent: {result.intent.value}")
        print(f"Events: {len(result.events)}")


if __name__ == "__main__":
    main()

