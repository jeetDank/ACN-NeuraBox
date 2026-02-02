"""
FastAPI Server for ACN RAG System
Production-ready API with health checks, metrics, and proper error handling
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, List
import time
from datetime import datetime
import logging
import json
import re

# FIXED IMPORT - Use NEW engine
from query import ACNRAGEngine
from config import RAGConfig
from response_adapter import adapt_llm_response

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== FastAPI App Setup ====================

app = FastAPI(
    title="ACN RAG API",
    description="GPU-Optimized Retrieval-Augmented Generation for Applied Client Network",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Global State ====================

# Load engine ONCE at startup
rag_config = RAGConfig()
rag_engine = None
startup_time = None
query_count = 0
error_count = 0

# Load events data
events_data = []
def load_events():
    """Load events from events.json file"""
    global events_data
    try:
        import os
        events_file = os.path.join(os.path.dirname(__file__), '..', 'acn_data', 'events.json')
        if os.path.exists(events_file):
            with open(events_file, 'r') as f:
                events_data = json.load(f)
            logger.info(f"Loaded {len(events_data)} events from events.json")
        else:
            logger.warning(f"Events file not found at {events_file}")
            events_data = []
    except Exception as e:
        logger.error(f"Failed to load events: {e}")
        events_data = []

# ==================== Pydantic Models ====================

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="User question")
    k: Optional[int] = Field(5, ge=1, le=20, description="Number of documents to retrieve")
    
    class Config:
        schema_extra = {
            "example": {
                "question": "What are the membership benefits of ACN?",
                "k": 5
            }
        }


class QueryResponse(BaseModel):
    ui: Dict
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources: List[str]
    intent: str
    num_docs: int
    processing_time: float
    
    class Config:
        schema_extra = {
            "example": {
                "ui": {
                    "response_type": "rich_message",
                    "data": {"blocks": []}
                },
                "answer": "ACN membership benefits include...",
                "confidence": 0.85,
                "sources": ["https://..."],
                "intent": "membership",
                "num_docs": 5,
                "processing_time": 1.5
            }
        }


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    queries_processed: int
    error_count: int
    model_loaded: bool
    gpu_available: bool


class ModelInfoResponse(BaseModel):
    llm_model: str
    embedding_model: str
    reranker_enabled: bool
    use_4bit: bool
    device: str


class SuggestionRequest(BaseModel):
    user_query: str = Field(..., min_length=1, max_length=500)
    answer: str = Field(..., min_length=1, max_length=2000)
    intent: Optional[str] = Field("general", max_length=100)
    
    class Config:
        schema_extra = {
            "example": {
                "user_query": "What are upcoming events?",
                "answer": "Here are the upcoming ACN events...",
                "intent": "events"
            }
        }


class Suggestion(BaseModel):
    id: str
    text: str
    icon: str


class SuggestionResponse(BaseModel):
    suggestions: List[Suggestion]


class Event(BaseModel):
    event_id: str
    title: str
    event_type: str
    city: Optional[str] = None
    state: Optional[str] = None
    venue: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None
    registration_url: Optional[str] = None
    source_url: Optional[str] = None


class EventsResponse(BaseModel):
    events: List[Event]
    total_count: int
    filter_type: str


# ==================== Startup & Shutdown ====================

@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine on startup"""
    global rag_engine, startup_time
    
    logger.info("Starting ACN RAG API...")
    startup_time = time.time()
    
    # Load events data
    load_events()
    
    try:
        logger.info("Loading RAG engine (this may take 10-20 minutes on first run)...")
        rag_engine = ACNRAGEngine(rag_config)
        logger.info("✓ RAG engine loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load RAG engine: {e}")
        logger.error("API will return errors for all queries")
        rag_engine = None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info(f"Shutting down. Processed {query_count} queries with {error_count} errors")


# ==================== API Endpoints ====================

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API info"""
    return {
        "name": "ACN RAG API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "query": "/query (POST)",
            "health": "/health (GET)",
            "model_info": "/model-info (GET)",
            "docs": "/docs (GET)"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Health check endpoint for monitoring"""
    import torch
    
    uptime = time.time() - startup_time if startup_time else 0
    
    return {
        "status": "healthy" if rag_engine is not None else "unhealthy",
        "uptime_seconds": uptime,
        "queries_processed": query_count,
        "error_count": error_count,
        "model_loaded": (
        rag_engine is not None and
        hasattr(rag_engine, "llm") and
        rag_engine.llm.available
        ),
        "gpu_available": torch.cuda.is_available()
    }


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Info"])
async def model_info():
    """Get information about loaded models"""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return {
        "llm_model": rag_config.LLM_MODEL,
        "embedding_model": rag_config.EMBEDDING_MODEL,
        "reranker_enabled": rag_config.USE_RERANKER,
        "use_4bit": rag_config.USE_4BIT,
        "device": device
    }


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_acn(request: QueryRequest):
    """
    Process a query through the RAG system
    
    **Parameters:**
    - question: The question to ask (required)
    - k: Number of documents to retrieve (default: 5, max: 20)
    
    **Returns:**
    - answer: Generated answer
    - confidence: Confidence score (0-1)
    - sources: List of source URLs
    - intent: Detected query category (membership/events/howto/general)
    - num_docs: Number of documents used
    - processing_time: Time taken in seconds
    """
    global query_count, error_count
    
    # Check if engine is loaded
    if rag_engine is None:
        error_count += 1
        logger.error("Query attempted but engine not initialized")
        raise HTTPException(
            status_code=503,
            detail="RAG engine not initialized. Please wait or check logs."
        )
    
    # Validate input
    question = request.question.strip()
    if not question:
        error_count += 1
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Log query
    query_count += 1
    logger.info(f"Query #{query_count}: {question[:100]}")
    
    try:
        # Standard RAG query for all questions
        start_time = time.time()
        result = rag_engine.query(question, k=request.k)
        processing_time = time.time() - start_time

        # Extract intent
        intent_category = (
            result["intent"].category 
            if hasattr(result.get("intent"), "category") 
            else "general"
        )

        # 🔥 Single adapter entry
        # Check if result already has UI data (from structured query handlers like event queries)
        if isinstance(result.get("ui"), dict) and result["ui"]:
            ui_payload = result["ui"]
        else:
            ui_payload = adapt_llm_response(
                answer=result["answer"],
                intent=intent_category
            )
        
        # Build response
        response = {
            "ui": ui_payload,
            "answer": result["answer"],
            "confidence": result.get("confidence", 0.0),
            "sources": result.get("sources", []),
            "intent": intent_category,
            "num_docs": result.get("num_docs", 0),
            "processing_time": processing_time
        }
        
        logger.info(f"Query #{query_count} completed in {processing_time:.2f}s (intent: {intent_category})")
        
        return response
        
    except Exception as e:
        error_count += 1
        logger.error(f"Query #{query_count} failed: {e}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)[:200]}"
        )


@app.get("/metrics", tags=["Info"])
async def metrics():
    """
    Get API metrics for monitoring
    """
    uptime = time.time() - startup_time if startup_time else 0
    
    return {
        "uptime_seconds": uptime,
        "total_queries": query_count,
        "total_errors": error_count,
        "error_rate": error_count / max(query_count, 1),
        "average_time_estimate": "1.5-2.5s",  # Can track this more precisely
        "model_loaded": rag_engine is not None and rag_engine.llm.available
    }


@app.post("/suggest", response_model=SuggestionResponse, tags=["Suggestions"])
async def get_suggestions(request: SuggestionRequest):
    """
    Generate follow-up suggestion prompts based on user query and AI answer
    Uses the LLM to generate contextually relevant follow-up questions
    
    **Parameters:**
    - user_query: The original user question
    - answer: The AI-generated answer
    - intent: The detected query category (optional)
    
    **Returns:**
    - suggestions: List of contextually relevant follow-up questions with icons
    """
    try:
        if rag_engine is None:
            logger.warning("RAG engine not available for suggestions, returning empty")
            return SuggestionResponse(suggestions=[])
        
        logger.info(f"Generating dynamic suggestions for query: {request.user_query[:50]}...")
        # Check if answer contains event information
        is_event_response = any(keyword in request.answer.lower() for keyword in ['event', 'summit', 'webinar', 'conference', 'upcoming', 'registration', 'date:'])
        
        if is_event_response:
            # Extract event names from answer for event-specific suggestions
            event_pattern = r'([A-Z][A-Za-z\s&]+(?:\d{4})?)'
            event_matches = re.findall(event_pattern, request.answer)
            
            suggestions = []
            if event_matches:
                # Create suggestions based on first 1-2 events mentioned
                for idx, event_name in enumerate(event_matches[:2]):
                    event_name = event_name.strip()
                    if len(event_name) > 3 and event_name != request.user_query:
                        suggestions.append(Suggestion(
                            id=f"sugg_{idx}",
                            text=f"Tell me more about {event_name}",
                            icon="Calendar"
                        ))
                
                # Add a general follow-up suggestion
                if len(suggestions) < 2:
                    suggestions.append(Suggestion(
                        id=f"sugg_{len(suggestions)}",
                        text="Show me the upcoming events",
                        icon="Calendar"
                    ))
            
            if suggestions:
                logger.info(f"Returning {len(suggestions)} event suggestions")
                return SuggestionResponse(suggestions=suggestions[:2])  # Max 2 for events
        
        # Standard prompt for non-event responses
        prompt = f"""Based on this conversation:

User Question: {request.user_query}

AI Answer: {request.answer}

Generate exactly 2 contextually relevant follow-up questions that a user might naturally ask next. These should be:
- Natural and specific to the content discussed
- NOT repetitions of the original question
- Exploring different angles or deeper aspects
- Short and concise (under 12 words each)

Format ONLY as a numbered list:
1. First follow-up question
2. Second follow-up question"""

        # Generate suggestions using LLM
        try:
            # Call the LLM's generate method
            generated_text = rag_engine.llm.generate(prompt)
            logger.info(f"Generated suggestions: {generated_text[:150]}")
            
            # Parse the generated suggestions
            suggestions = []
            icon_list = ["BookOpen", "HelpCircle", "MessageSquare", "FileText", "Users"]
            icon_index = 0
            
            lines = generated_text.strip().split('\n')
            for line in lines:
                # Parse numbered list format
                line = line.strip()
                if not line:
                    continue
                
                # Extract question text (remove "1." or "1)" prefix)
                match = re.split(r'^\d+\.\s+', line)
                if len(match) > 1:
                    question_text = match[1].strip()
                else:
                    question_text = line
                
                # Only add non-empty questions
                if question_text and len(question_text) > 5 and not any(skip in question_text.lower() for skip in ['reject', 'unable', 'cannot']):
                    suggestions.append(Suggestion(
                        id=f"sugg_{len(suggestions)}",
                        text=question_text,
                        icon=icon_list[icon_index % len(icon_list)]
                    ))
                    icon_index += 1
                    
                    if len(suggestions) >= 2:
                        break
            
            if suggestions:
                logger.info(f"Returning {len(suggestions)} suggestions")
                return SuggestionResponse(suggestions=suggestions)
            else:
                logger.warning("No suggestions parsed from LLM output")
                return SuggestionResponse(suggestions=[])
        
        except Exception as llm_error:
            logger.warning(f"LLM generation failed: {llm_error}")
            # Return empty on LLM errors
            return SuggestionResponse(suggestions=[])
        
    except Exception as e:
        logger.error(f"Failed to generate suggestions: {e}", exc_info=True)
        return SuggestionResponse(suggestions=[])


@app.get("/events", response_model=EventsResponse, tags=["Events"])
async def get_events(filter_type: str = "upcoming", limit: int = 10):
    """
    Get events from the ACN knowledge base
    
    **Parameters:**
    - filter_type: Type of events to return - "upcoming", "past", or "all" (default: "upcoming")
    - limit: Maximum number of events to return (default: 10, max: 50)
    
    **Returns:**
    - events: List of Event objects
    - total_count: Total number of events matching filter
    - filter_type: The filter type that was applied
    """
    try:
        if not events_data:
            logger.warning("No events data available")
            return EventsResponse(events=[], total_count=0, filter_type=filter_type)
        
        # Limit max results
        limit = min(limit, 50)
        
        # Filter events based on filter_type
        filtered_events = []
        current_date = datetime.now().date()
        
        for event in events_data:
            try:
                # Try to parse start_date if available
                event_date = None
                if event.get("start_date"):
                    try:
                        event_date = datetime.strptime(event["start_date"], "%Y-%m-%d").date()
                    except:
                        pass
                
                if filter_type == "upcoming":
                    # Only include events with valid future dates
                    if event_date and event_date >= current_date:
                        filtered_events.append(event)
                elif filter_type == "past":
                    # Only include events with valid past dates
                    if event_date and event_date < current_date:
                        filtered_events.append(event)
                else:  # "all"
                    filtered_events.append(event)
            except Exception as e:
                logger.warning(f"Error filtering event {event.get('event_id')}: {e}")
                continue
        
        # Sort by start_date if available (most recent first for past, earliest first for upcoming)
        try:
            if filter_type == "upcoming":
                filtered_events.sort(
                    key=lambda e: datetime.strptime(e.get("start_date", "2099-12-31"), "%Y-%m-%d"),
                    reverse=False
                )
            else:
                filtered_events.sort(
                    key=lambda e: datetime.strptime(e.get("start_date", "2000-01-01"), "%Y-%m-%d"),
                    reverse=True
                )
        except Exception as e:
            logger.warning(f"Error sorting events: {e}")
        
        # Convert to Event models and limit
        event_models = [Event(**event) for event in filtered_events[:limit]]
        
        logger.info(f"Returned {len(event_models)} {filter_type} events")
        
        return EventsResponse(
            events=event_models,
            total_count=len(filtered_events),
            filter_type=filter_type
        )
        
    except Exception as e:
        logger.error(f"Failed to get events: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving events: {str(e)[:200]}"
        )


# ==================== Error Handlers ====================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler"""
    return {
        "error": "Endpoint not found",
        "available_endpoints": ["/", "/health", "/model-info", "/query", "/suggest", "/events", "/metrics", "/docs"]
    }


# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set True for development
        log_level="info"
    )