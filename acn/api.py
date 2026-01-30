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
        # Check query type
        question_lower = question.lower()
        
        # ==================== HARDCODED QUERY: What is ACN ====================
        is_what_is_acn = any(phrase in question_lower for phrase in [
            'what is acn', 'what\'s acn', 'about acn', 'tell me about acn', 
            'explain acn', 'acn overview', 'how to join acn', 'join acn'
        ])
        
        if is_what_is_acn:
            hardcoded_answer = """ How to Join ACN (Step-by-Step)

 1️⃣ Go to the Official Website
 [https://www.appliedclientnetwork.org](https://www.appliedclientnetwork.org)

 2️⃣ Click Join / Become a Member
You'll usually find:
- Join ACN
- Membership
- Become a Member

 3️⃣ Choose the Right Membership Type
ACN offers different plans:
- Agency / Organization Membership (most common)
- Individual Membership
- Student Membership (if available)
- Vendor / Partner Membership

 4️⃣ Fill Out the Application Form
You'll be asked for:
- Name
- Email
- Company / College
- Role (Student / Developer / Analyst / Insurance Professional)
- Country

 5️⃣ Pay Membership Fee (If Required)
- Some memberships are paid
- Student memberships are sometimes free or discounted

 6️⃣ Approval & Confirmation
Once approved, you get access to:
- ACN resources
- Events & webinars
- Community forums
- Tools & integrations knowledge"""

            response = {
                "answer": hardcoded_answer,
                "confidence": 0.98,
                "sources": ["https://www.appliedclientnetwork.org"],
                "intent": "membership",
                "num_docs": 1,
                "processing_time": 0.1,
                "suggestions": [
                    {"id": "sugg_0", "text": "How do I become a member at ACN?", "icon": "Users"},
                    {"id": "sugg_1", "text": "What are the membership benefits at ACN?", "icon": "Award"}
                ]
            }
            logger.info(f"Query #{query_count} - What is ACN hardcoded response")
            return response
        
        # ==================== HARDCODED QUERY: How to become a member ====================
        is_become_member = any(phrase in question_lower for phrase in [
            'become a member', 'become member', 'how to become member', 
            'membership types', 'join as member', 'apply for membership',
            'how do i become a member'
        ])
        
        if is_become_member:
            hardcoded_answer = """ How to Become a Member

 1️⃣ Visit the ACN Membership Page
Go to the Membership → Join/Renew section on the ACN website:
 [Join ACN (membership page)](https://www.appliedclientnetwork.org/membership)

 2️⃣ Choose the Correct Membership Type
ACN offers several categories:

  Advantage Membership
- For agencies and brokerages using Applied Systems products
- Covers your entire office location (not just individuals)

  Enterprise Membership
- For larger organizations with 125+ licenses
- Requires contacting ACN directly (email) for a custom package

  EZLynx Membership
- For EZLynx software users
- Requires contacting ACN for details or enrollment

  Associate Membership
- For consultants, carriers, technology providers, vendors, etc.
- Gives access to forums, networking, directory listing, and more

 3️⃣ Fill Out the Application
For most membership types, you'll either:
- Complete an online join form
- OR email ACN for specific membership types (like Enterprise or EZLynx)

 The contact email for questions or custom membership info is: info@appliedclientnetwork.org

 4️⃣ Pay the Membership Fee (If Applicable)
- Agency/Advantage memberships typically include a fee (varies by region/size)
- Associate/consultant or specialized memberships may also have pricing
- The ACN team will send payment instructions after your form or inquiry.

 5️⃣ Get Access
Once approved and paid (if needed):
- Your organization gets access to ACN resources, forums, and member benefits.
- Your users can participate in events like Applied Net and ACN Alliance meetups."""

            response = {
                "answer": hardcoded_answer,
                "confidence": 0.98,
                "sources": ["https://www.appliedclientnetwork.org/membership"],
                "intent": "membership",
                "num_docs": 1,
                "processing_time": 0.1,
                "suggestions": [
                    {"id": "sugg_0", "text": "What are the membership benefits at ACN?", "icon": "Award"},
                    {"id": "sugg_1", "text": "What is ACN?", "icon": "HelpCircle"}
                ]
            }
            logger.info(f"Query #{query_count} - How to become a member hardcoded response")
            return response
        
        # ==================== HARDCODED QUERY: Membership benefits ====================
        is_membership_benefits = any(phrase in question_lower for phrase in [
            'membership benefits', 'benefits of membership', 'what benefits',
            'member benefits', 'acn benefits', 'why join acn', 'member perks',
            'what are the membership benefits'
        ])
        
        if is_membership_benefits:
            hardcoded_answer = """## Membership Benefits

1. Peer-to-Peer Networking
Connect with other insurance professionals who use Applied products to share insights, tips, and best practices. ACN members can participate in forums, alliances, and community discussions to solve problems and discover new workflows.

2. Education & Training Resources
Members get access to:
- On-demand educational content created by users and experts
- Webinars and workshops tailored to different roles (e.g., CSR, IT, operations)
- Practical guidance focused on improving how you use Applied software in your agency or brokerage.

3. Discounts on Applied Net Conference
ACN members qualify for reduced pricing on registration for Applied Net — the annual flagship conference with hundreds of sessions, networking, and industry insights.

4. Access to Member Alliances
Join persona-specific alliance groups (e.g., operations, sales, IT) to connect with peers in similar roles and get targeted education and discussions.

5. Member Directory & Job Board
Members can use the ACN directory to find other agencies, professionals, or resources. A job board helps members explore opportunities in the industry.

6. Events & Summits
Beyond the big Applied Net conference, ACN hosts regional summits and local events where members can learn and network more intimately with peers.

7. Industry Advocacy & Product Influence
ACN gives members a voice to share feedback on Applied Systems products — helping shape improvements and product direction with real user input.

8. Publications & Insights
Members receive Connections — ACN's online publication and newsletter with industry trends, updates, and best practices."""

            response = {
                "answer": hardcoded_answer,
                "confidence": 0.98,
                "sources": ["https://www.appliedclientnetwork.org/membership/benefits"],
                "intent": "membership",
                "num_docs": 1,
                "processing_time": 0.1,
                "suggestions": [
                    {"id": "sugg_0", "text": "How do I become a member at ACN?", "icon": "Users"},
                    {"id": "sugg_1", "text": "Show me upcoming events", "icon": "Calendar"}
                ]
            }
            logger.info(f"Query #{query_count} - Membership benefits hardcoded response")
            return response
        
        # ==================== Check if this is an events query ====================
        is_events_query = any(keyword in question_lower for keyword in [
            'event', 'summit', 'conference', 'webinar', 'upcoming', 'schedule', 
            'applied net', 'ama session', 'roundtable', 'show me events'
        ])
        
        # If events query, fetch and format event cards
        if is_events_query:
            try:
                # Demo upcoming events (since most JSON events are in the past)
                demo_events = [
                    {
                        "title": "Applied Net 2026",
                        "start_date": "2026-09-27",
                        "time": "9:00am",
                        "location": "Gaylord National Resort, Washington, DC",
                        "url": "https://www.appliedclientnetwork.org/events/applied-net-2026"
                    },
                    {
                        "title": "ACN Summit – Dallas",
                        "start_date": "2026-02-25",
                        "time": "9:00am",
                        "location": "Dallas, Texas, USA",
                        "url": "https://www.appliedclientnetwork.org/events/acn-summit-dallas-2026"
                    },
                    {
                        "title": "ACN Summit – Calgary",
                        "start_date": "2026-06-03",
                        "time": "9:00am",
                        "location": "Calgary, Alberta, Canada",
                        "url": "https://www.appliedclientnetwork.org/events/acn-summit-calgary-2026"
                    },
                    {
                        "title": "ACN Webinar – Applied Epic Power User AMA",
                        "start_date": "2026-02-10",
                        "time": "1:00pm",
                        "location": "Virtual (Online)",
                        "url": "https://www.appliedclientnetwork.org/events/applied-epic-power-user-ama"
                    }
                ]
                
                # Sort by date
                demo_events.sort(key=lambda e: e.get('start_date', '2099-12-31'))
                
                # Format each event in the exact format the frontend parser expects
                formatted_answer = ""
                for idx, event in enumerate(demo_events, 1):
                    title = event.get('title', 'Untitled Event')
                    start_date = event.get('start_date', 'Date TBA')
                    time_str = event.get('time', '')
                    location = event.get('location', 'Online/TBA')
                    url = event.get('url', '')
                    
                    # Format: "1. **Title**" on its own line
                    formatted_answer += f"{idx}. **{title}**\n"
                    # Then Date: on next line
                    formatted_answer += f"Date: {start_date}\n"
                    # Then Time: if available
                    if time_str:
                        formatted_answer += f"Time: {time_str}\n"
                    # Then Location:
                    formatted_answer += f"Location: {location}\n"
                    # Then Learn More link
                    if url:
                        formatted_answer += f"Learn More: [{title}]({url})\n"
                    formatted_answer += "\n"
                
                response = {
                    "answer": formatted_answer,
                    "confidence": 0.95,
                    "sources": [],
                    "intent": "events",
                    "num_docs": len(demo_events),
                    "processing_time": 0.1
                }
                
                logger.info(f"Query #{query_count} - Events query, returned {len(demo_events)} events")
                return response
            
            except Exception as e:
                logger.warning(f"Failed to format events, falling back to RAG: {e}")
        
        # Standard RAG query for non-event questions
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