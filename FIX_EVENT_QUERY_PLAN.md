# Event Query Fix Plan

## Issues Identified

1. **EventFilter.matches() bug**: When `upcoming_only=True` and event has no `start_date`, it's excluded
2. **Subpage merging logic issue**: Subpages creating separate events instead of merging
3. **Corrupted event data**: Dates are wrong (e.g., "2004-06-03" instead of "2026-06-03")
4. **Query classification edge cases**: Some query patterns may not be detected properly

## Fixes to Implement

### Fix 1: EventFilter.matches() - Handle events with missing dates
- **File**: `event_models.py`
- **Issue**: When `upcoming_only=True`, events without dates are excluded
- **Fix**: For `upcoming_only=True`, include events without dates (we don't know if they're past)

### Fix 2: EventPageDetector - Improve subpage detection
- **File**: `event_extraction.py`
- **Issue**: Subpage patterns may not properly identify parent events
- **Fix**: Ensure proper parent_event_id is returned for subpages

### Fix 3: process_scraped_pages - Fix subpage merging
- **File**: `event_extraction.py`
- **Issue**: Subpages creating separate events instead of merging
- **Fix**: Correct flow: if `parent_id` is returned, merge into parent

### Fix 4: Date parsing - Fix corrupted dates
- **File**: `event_extraction.py`
- **Issue**: Years like "2004" should be "2026"
- **Fix**: Add logic to correct obviously wrong years

### Fix 5: EventQueryClassifier - Improve query detection
- **File**: `event_query_handler.py`
- **Issue**: Some query patterns like "list all events" may not match
- **Fix**: Add patterns for "list all events", "show ACN events", etc.

## Status

- [ ] Fix 1: EventFilter.matches() - Handle events with missing dates
- [ ] Fix 2: EventPageDetector - Improve subpage detection
- [ ] Fix 3: process_scraped_pages - Fix subpage merging
- [ ] Fix 4: Date parsing - Fix corrupted dates
- [ ] Fix 5: EventQueryClassifier - Improve query detection
- [ ] Test all fixes

