# ACN Event Ingestion Fixes - TODO

## Phase 1: Text Cleaning Improvements
- [ ] 1.1 Add missing Cyrillic 'с' → 'c' mapping
- [ ] 1.2 Add corrections for "Dall ACN" → "Dallas", "Tex ACN" → "Texas"
- [ ] 1.3 Fix "EZL ynx" → "EZLynx" (space corruption)
- [ ] 1.4 Fix patterns like "ACN an ACN" → "an ACN"
- [ ] 1.5 Fix "Calgary" not leaking into Dallas events
- [ ] 1.6 Add better camelCase handling for "day-To-Day" → "day-to-Day"

## Phase 2: Event Page Detection Fixes
- [ ] 2.1 Block ALL subpage patterns from creating events
- [ ] 2.2 Only main event pages (depth 3) create events
- [ ] 2.3 Extract correct parent_event_id for subpages
- [ ] 2.4 Add better URL pattern matching

## Phase 3: Date Extraction & Validation Fixes
- [ ] 3.1 Fix date parsing to handle range formats correctly
- [ ] 3.2 Validate end_date >= start_date
- [ ] 3.3 Filter out impossible years (before 2023)
- [ ] 3.4 Swap dates if validation fails

## Phase 4: Location Extraction Fixes
- [ ] 4.1 Only extract location from main event pages
- [ ] 4.2 Inherit location from parent for subpages
- [ ] 4.3 Fix city extraction patterns

## Phase 5: Event Merging Logic
- [ ] 5.1 Subpage content merged into parent event
- [ ] 5.2 Use parent_event_id field correctly
- [ ] 5.3 Aggregate agenda from Program subpages
- [ ] 5.4 Aggregate speakers from Speakers subpages

## Phase 6: Testing & Validation
- [ ] 6.1 Re-run event extraction on raw data
- [ ] 6.2 Validate output events.json
- [ ] 6.3 Check for duplicate events
- [ ] 6.4 Verify date ranges are valid
- [ ] 6.5 Check text quality

## Files to Modify
- `event_extraction.py` - Main fixes
- `event_models.py` - Event validation
- `ingest_acn_data.py` - If needed

