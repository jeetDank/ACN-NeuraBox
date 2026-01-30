## Running the Complete Event Workflow

**Step-by-step:**

1. **Scrape data** (if needed)
   ```bash
   python ingest_acn_data.py
   ```

2. **Extract events** from scraped pages
   ```bash
   python run_event_extraction.py
   ```
   → Creates `acn_data/events.json`

3. **Run event query tests**
   ```bash
   python verify_event_workflow.py
   ```

4. **Start API server** (full RAG + Event handling)
   ```bash
   python api.py
   ```
   → API at `http://localhost:8000`

5. **Query via API**
   ```bash
   curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"question": "Show me upcoming events"}'
   ```

**Quick test:**
```bash
python event_query_handler.py
```

