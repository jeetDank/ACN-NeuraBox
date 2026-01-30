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

# FIXED IMPORT - Use NEW engine
from query_acn_rag import ACNRAGEngine
from config import RAGConfig

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
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources: List[str]
    intent: str
    num_docs: int
    processing_time: float
    
    class Config:
        schema_extra = {
            "example": {
                "answer": "ACN membership benefits include...",
                "confidence": 0.85,
                "sources": ["https://..."],
                "intent": "membership",
                "num_docs": 5,
                "processing_time": 1.5
            }
        }


class SuggestionsRequest(BaseModel):
    user_query: str = Field(..., min_length=1, max_length=500, description="Original user question")
    answer: str = Field(..., min_length=1, description="LLM answer to the query")
    intent: Optional[str] = Field("general", description="Query intent category")
    
    class Config:
        schema_extra = {
            "example": {
                "user_query": "Tell me about Applied Net 2025",
                "answer": "Applied Net 2025 is the largest gathering...",
                "intent": "events"
            }
        }


class Suggestion(BaseModel):
    id: str
    text: str
    icon: str
    
    class Config:
        schema_extra = {
            "example": {
                "id": "event-registration",
                "text": "How to register for events",
                "icon": "Calendar"
            }
        }


class SuggestionsResponse(BaseModel):
    suggestions: List[Suggestion]
    processing_time: float


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


# ==================== Startup & Shutdown ====================

@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine on startup"""
    global rag_engine, startup_time
    
    logger.info("Starting ACN RAG API...")
    startup_time = time.time()
    
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
        # Process query
        start_time = time.time()
        result = rag_engine.query(question, k=request.k)
        processing_time = time.time() - start_time
        
        # Extract intent
        intent_category = (
            result["intent"].category 
            if hasattr(result.get("intent"), "category") 
            else "general"
        )
        
        # Build response
        response = {
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


@app.post("/suggest", response_model=SuggestionsResponse, tags=["Query"])
async def generate_suggestions(request: SuggestionsRequest):
    """
    Generate AI-driven follow-up suggestions based on user query and LLM answer
    Uses the RAG system to find relevant related topics from the database
    
    **Parameters:**
    - user_query: The original user question
    - answer: The LLM-generated answer
    - intent: Optional query intent category
    
    **Returns:**
    - suggestions: List of relevant follow-up questions
    - processing_time: Time taken to generate suggestions
    """
    if rag_engine is None:
        raise HTTPException(
            status_code=503,
            detail="RAG engine not initialized"
        )
    
    try:
        start_time = time.time()
        
        # Create suggestion prompts based on the query and answer
        suggestion_prompt = f"""Based on the user's question and the answer provided, generate 2 relevant follow-up questions that would help deepen their understanding.

User's Original Question: {request.user_query}

Answer Provided: {request.answer}

Generate exactly 2 follow-up questions that:
1. Are directly related to the context of the answer
2. Help explore related topics in our knowledge base
3. Are phrased as natural, concise questions

Format your response as a JSON array with objects containing "text" field only:
[
  {{"text": "Follow-up question 1?"}},
  {{"text": "Follow-up question 2?"}}
]

Only respond with valid JSON, no additional text."""

        # Use RAG engine to generate contextual suggestions
        # This leverages the knowledge base to find relevant topics
        suggestion_result = rag_engine.query(suggestion_prompt, k=3)
        
        suggestions_text = suggestion_result.get("answer", "[]")
        
        # Parse suggestions from LLM response
        try:
            # Extract JSON from the response
            import json as json_module
            json_start = suggestions_text.find('[')
            json_end = suggestions_text.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = suggestions_text[json_start:json_end]
                parsed_suggestions = json_module.loads(json_str)
            else:
                parsed_suggestions = []
        except:
            parsed_suggestions = []
        
        # Map suggestions to appropriate icons based on intent and query
        icon_mapping = {
            "events": "Calendar",
            "membership": "Users",
            "resources": "BookOpen",
            "howto": "HelpCircle",
            "contact": "MessageSquare",
            "account": "User",
            "general": "Search"
        }
        
        icon = icon_mapping.get(request.intent, "Search")
        
        # Format final suggestions
        final_suggestions = []
        for idx, sugg in enumerate(parsed_suggestions[:2]):  # Limit to 2
            if isinstance(sugg, dict) and "text" in sugg:
                final_suggestions.append(
                    Suggestion(
                        id=f"suggestion-{idx}",
                        text=sugg["text"],
                        icon=icon
                    )
                )
        
        # Fallback suggestions if parsing failed
        if not final_suggestions:
            fallback_prompts = {
                "events": [
                    Suggestion(id="event-filter", text="Show events by date range", icon="Calendar"),
                    Suggestion(id="event-location", text="Find events in my region", icon="MapPin")
                ],
                "membership": [
                    Suggestion(id="member-benefits", text="Compare membership levels", icon="Users"),
                    Suggestion(id="member-renew", text="How to renew my membership", icon="Users")
                ],
                "resources": [
                    Suggestion(id="resource-download", text="Download resource templates", icon="BookOpen"),
                    Suggestion(id="resource-explore", text="Explore all resources", icon="BookOpen")
                ],
            }
            
            final_suggestions = fallback_prompts.get(request.intent, [
                Suggestion(id="ask-more", text="Ask more questions", icon="Search"),
                Suggestion(id="explore", text="Explore related topics", icon="Search")
            ])
        
        processing_time = time.time() - start_time
        
        return {
            "suggestions": final_suggestions,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Suggestion generation failed: {e}", exc_info=True)
        
        # Return safe fallback suggestions
        return {
            "suggestions": [
                Suggestion(id="ask-more", text="Ask another question", icon="Search"),
                Suggestion(id="explore", text="Explore related topics", icon="Search")
            ],
            "processing_time": time.time() - start_time
        }


# ==================== Error Handlers ====================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler"""
    return {
        "error": "Endpoint not found",
        "available_endpoints": ["/", "/health", "/model-info", "/query", "/metrics", "/docs"]
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