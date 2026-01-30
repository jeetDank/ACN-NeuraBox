#!/usr/bin/env python3
"""
Complete GPU-Optimized ACN RAG System - WITH EVENT QUERY HANDLING
Features:
- Event and Summit related queries handled BEFORE vector-based RAG
- "Upcoming" and date-based event queries work deterministically
- Existing general RAG functionality remains intact as fallback
"""

import re
import json
import torch
import sys
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Core libraries
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import CrossEncoder

# LangChain
try:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.documents import Document

# Local config
from config import RAGConfig

# Event handling modules
from event_models import Event, EventType
from event_extraction import EventStore
from event_query_handler import (
    EventQueryHandler, EventQueryIntent, EventQueryResult, EventQueryClassifier
)

from intelligent_text_fixer import IntelligentTextFixer, FixerConfig, fix_text

# ==================== TEXT NORMALIZATION ====================

def normalize_spacing(text: str) -> str:
    # Fix glued words: AppliedProducts → Applied Products
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Fix missing spaces after punctuation
    text = re.sub(r'([.,!?])([A-Za-z])', r'\1 \2', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ==================== QUERY CLASSIFICATION ====================

@dataclass
class QueryIntent:
    """Structured query intent"""
    category: str  # membership, events, howto, resources, general
    is_temporal: bool
    temporal_mode: Optional[str]  # upcoming, past, specific_year, all
    year: Optional[int] = None
    confidence: float = 0.0


class QueryClassifier:
    """Intelligent query classification"""
    
    def __init__(self):
        self.patterns = {
            "membership": [
                r"member(?:ship)?|join|benefit|cost|price|fee|renew|subscription|sign up|register",
            ],
            "events": [
                r"event|summit|conference|webinar|workshop|training|session|calendar|schedule",
            ],
            "howto": [
                r"how (?:do|can|to)|guide|instruction|step|process|tutorial|set up|configure",
            ],
            "resources": [
                r"resource|download|document|template|tool|whitepaper|best practice",
            ],
            "about": [
                r"what is|tell me about|explain|describe|who are|about acn",
            ]
        }
        
        self.temporal_patterns = {
            "upcoming": r"upcoming|future|next|coming|soon|when|schedule|will be",
            "past": r"past|previous|last year|history|was|happened|held",
            "specific_year": r"20\d{2}",
        }
    
    def classify(self, query: str) -> QueryIntent:
        """Classify query into category and temporal intent"""
        query_lower = query.lower()
        
        # Determine category
        category = "general"
        max_matches = 0
        
        for cat, patterns in self.patterns.items():
            matches = sum(1 for p in patterns if re.search(p, query_lower))
            if matches > max_matches:
                max_matches = matches
                category = cat
        
        # Determine temporal intent
        is_temporal = False
        temporal_mode = "all"
        year = None
        
        # Check for specific year first
        year_match = re.search(r"20\d{2}", query_lower)
        if year_match:
            is_temporal = True
            temporal_mode = "specific_year"
            year = int(year_match.group())
        else:
            # Check for upcoming/past
            for mode, pattern in self.temporal_patterns.items():
                if mode != "specific_year" and re.search(pattern, query_lower):
                    is_temporal = True
                    temporal_mode = mode
                    break
        
        # Confidence based on matches
        confidence = min(1.0, max_matches * 0.3 + 0.4)
        
        return QueryIntent(
            category=category,
            is_temporal=is_temporal,
            temporal_mode=temporal_mode,
            year=year,
            confidence=confidence
        )


# ==================== RERANKER ====================

class GPUReranker:
    """GPU-accelerated document reranking"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        print(f"Loading reranker: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512, device=device)
        self.device = device
        print(f"✓ Reranker loaded on {device}")
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Rerank documents using cross-encoder"""
        
        if not documents:
            return []
        
        # Prepare pairs for reranking
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get relevance scores
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Combine documents with scores
        doc_score_pairs = list(zip(documents, scores))
        
        # Sort by score (descending)
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return doc_score_pairs[:top_k]


# ==================== HYBRID RETRIEVER ====================

class HybridRetriever:
    """Advanced retrieval with metadata filtering and reranking"""
    
    def __init__(
        self,
        chroma_db: Chroma,
        reranker: Optional[GPUReranker] = None,
        config: RAGConfig = None
    ):
        self.db = chroma_db
        self.reranker = reranker
        self.config = config or RAGConfig()
    
    def retrieve(
        self,
        query: str,
        intent: QueryIntent,
        k: int = 20
    ) -> List[Document]:
        """Hybrid retrieval with reranking"""
        
        # Step 1: Expand query if enabled
        if self.config.ENABLE_QUERY_EXPANSION:
            query = self._expand_query(query, intent)
        
        # Step 2: Semantic search (retrieve more for reranking)
        initial_k = k * 2 if self.reranker else k
        docs = self.db.similarity_search(query, k=initial_k)
        
        # Step 3: Metadata filtering
        docs = self._filter_by_metadata(docs, intent)
        
        # Step 4: Temporal filtering for events
        if (
            self.config.ENABLE_TEMPORAL_FILTERING and
            intent.is_temporal and
            intent.category == "events"
        ):
            docs = self._apply_temporal_filter(docs, intent)
    
        # Step 5: Rerank if available
        if self.reranker and docs:
            doc_score_pairs = self.reranker.rerank(query, docs, top_k=k)
            docs = [doc for doc, score in doc_score_pairs]
        
        # Step 6: Deduplicate by source
        docs = self._deduplicate(docs)
        
        return docs[:k]
    
    def _expand_query(self, query: str, intent: QueryIntent) -> str:
        """Add synonyms/related terms"""
        expansions = {
            "membership": " member benefits join organization",
            "events": " summit conference webinar training session",
            "howto": " guide tutorial instructions steps process",
        }
        expansion = expansions.get(intent.category, "")
        return f"{query} {expansion}"
    
    def _filter_by_metadata(
        self,
        docs: List[Document],
        intent: QueryIntent
    ) -> List[Document]:
        """Filter and prioritize by metadata"""
        
        priority_docs = []
        normal_docs = []
        
        for doc in docs:
            source = doc.metadata.get("source", "").lower()
            title = doc.metadata.get("title", "").lower()
            content = doc.page_content.lower()
            
            # Category-specific prioritization
            is_priority = False
            
            if intent.category == "membership":
                keywords = ["member", "join", "benefit", "renew", "pricing"]
                is_priority = any(kw in source or kw in title for kw in keywords)
            
            elif intent.category == "events":
                is_priority = (
                    "event" in source or
                    doc.metadata.get("content_type") == "event" or
                    "summit" in title or
                    "conference" in title
                )
            
            elif intent.category == "howto":
                keywords = ["guide", "tutorial", "how-to", "instructions"]
                is_priority = any(kw in title for kw in keywords)
            
            if is_priority:
                priority_docs.append(doc)
            else:
                normal_docs.append(doc)
        
        return priority_docs + normal_docs
    
    def _apply_temporal_filter(
        self,
        docs: List[Document],
        intent: QueryIntent
    ) -> List[Document]:
        """Filter events by temporal intent"""
        
        from dateutil.parser import parse as parse_date
        today = datetime.now().replace(tzinfo=None)
        
        filtered = []
        
        for doc in docs:
            event_date_str = doc.metadata.get("event_date")
            
            if not event_date_str:
                filtered.append(doc)
                continue
  
            try:
                event_date = parse_date(event_date_str).replace(tzinfo=None)
                
                if intent.temporal_mode == "upcoming":
                    if event_date >= today:
                        filtered.append(doc)
                elif intent.temporal_mode == "past":
                    if event_date < today:
                        filtered.append(doc)
                elif intent.temporal_mode == "specific_year" and intent.year:
                    if event_date.year == intent.year:
                        filtered.append(doc)
                else:  # all
                    filtered.append(doc)
            except:
                # Keep if date parsing fails
                filtered.append(doc)
        
        return filtered if filtered else docs
    
    def _deduplicate(self, docs: List[Document]) -> List[Document]:
        """Remove duplicate sources"""
        seen_sources = set()
        unique_docs = []
        
        for doc in docs:
            source = doc.metadata.get("source")
            if source and source not in seen_sources:
                unique_docs.append(doc)
                seen_sources.add(source)
        
        return unique_docs


# ==================== LLM HANDLER ====================

class LLMHandler:
    """GPU-optimized LLM for answer generation"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu"
        self.model = None
        self.tokenizer = None
        self.available = False
        
        self._load_model()
    
    def _load_model(self):
        """Load LLM with optional 4-bit quantization"""
        try:
            print(f"\nLoading LLM: {self.config.LLM_MODEL}")
            print(f"Device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.LLM_MODEL,
                trust_remote_code=True
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure quantization if requested
            if self.config.USE_4BIT and self.device == "cuda":
                print("Using 4-bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.LLM_MODEL,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                # Standard loading
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.LLM_MODEL,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
            
            self.model.eval()
            self.available = True
            
            print(f"✓ LLM loaded successfully")
            print(f"  Model: {self.config.LLM_MODEL}")
            print(f"  Device: {next(self.model.parameters()).device}")
            if self.device == "cuda":
                print(f"  Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
            
        except Exception as e:
            print(f"⚠ LLM loading failed: {e}")
            print("Will use context-based fallback responses")
            self.available = False
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = None
    ) -> str:
        """Generate response from prompt"""
        
        if not self.available:
            return "Model not available"
        
        max_new_tokens = max_new_tokens or self.config.MAX_NEW_TOKENS
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=self.config.TEMPERATURE,
                    do_sample=self.config.TEMPERATURE > 0,
                    top_p=0.9 if self.config.TEMPERATURE > 0 else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3,
                )
            
            # Decode
            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            return response
            
        except Exception as e:
            print(f"⚠ Generation error: {e}")
            return "Error generating response"


# ==================== PROMPT BUILDER ====================

class PromptBuilder:
    """Build category-specific prompts"""
    
    @staticmethod
    def build(
        question: str,
        context: str,
        intent: QueryIntent,
        current_date: str
    ) -> str:
        """Build optimized prompt based on query intent"""
        
        # Base system message
        system = "You are a helpful assistant for Applied Client Network (ACN)."
        
        # Category-specific instructions
        if intent.category == "membership":
            specific_instructions = """
Focus on:
- Membership benefits (be specific)
- How to join (step-by-step)
- Costs and pricing (if available)
- Renewal process
Format: Use bullet points for benefits."""
        
        elif intent.category == "events":
            temporal_instruction = ""
            if intent.temporal_mode == "upcoming":
                temporal_instruction = f"""
CRITICAL: Today is {current_date}.
ONLY mention events AFTER this date.
REJECT all past events."""
            elif intent.temporal_mode == "past":
                temporal_instruction = f"""
Today is {current_date}.
Focus on events BEFORE this date."""
            
            specific_instructions = f"""
Focus on:
- Event name and description
- Date and time
- Location (physical or virtual)
- Registration/signup links
{temporal_instruction}
Format: List events with all details."""
        
        elif intent.category == "howto":
            specific_instructions = """
Focus on:
- Step-by-step instructions
- Requirements or prerequisites
- Direct links to resources
Format: Use numbered steps."""
        
        else:
            specific_instructions = """
Provide a clean, well-formatted summary.
DO NOT paraphrase aggressively.
DO NOT merge words.
Preserve original wording and spacing as much as possible.
DO NOT add quotes, people names, or opinions not present in context.
"""
        
        # Build full prompt
        prompt = f"""{system}
Today's date: {current_date}

INSTRUCTIONS:
1. Answer ONLY using information from the context below
2. Be specific and concise
3. If information is not in context, say "I don't have that information"
{specific_instructions}

IMPORTANT RULES:
- Do NOT invent facts, names, quotes, testimonials, or opinions
- Do NOT mention people or organizations unless explicitly named in the context
- Do NOT paraphrase aggressively or merge words
- Preserve original wording and spacing as much as possible
- If details are missing, explicitly say "Details not available"
- Use ONLY information explicitly present in the context

<context>
{context}
</context>

Question: {question}

Answer:"""
        
        return prompt 


# ==================== MAIN RAG ENGINE ====================

class ACNRAGEngine:
    """
    Complete GPU-Optimized RAG System - WITH EVENT QUERY HANDLING
    
    Event queries are handled BEFORE vector-based RAG using structured event data.
    Event Calendar and Learning/Webinar pages are NEVER returned for event queries.
    """
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        
        print("\n" + "="*80)
        print("Initializing ACN RAG Engine (GPU-Optimized) - WITH EVENT QUERY HANDLING")
        print("="*80)
        
        # Load components
        self.embeddings = self._load_embeddings()
        self.db = self._load_chroma()
        self.classifier = QueryClassifier()
        self.reranker = self._load_reranker() if self.config.USE_RERANKER else None
        self.retriever = HybridRetriever(self.db, self.reranker, self.config)
        self.llm = LLMHandler(self.config)
        self.prompt_builder = PromptBuilder()
        
        print("\n" + "="*80)
        print("✓ ACN RAG Engine Ready!")
        
        # Initialize EVENT STORE and EVENT QUERY HANDLER
        self._init_event_handler()
        
        # Initialize text quality fixer
        self.text_fixer = IntelligentTextFixer(
            FixerConfig(
                protected_terms={
                    'ACN', 'Applied', 'Applied Client Network',
                    'EZLynx', 'Epic', 'Ivans', 'TAM',
                    'Dallas', 'Calgary', 'Alberta',
                    'Texas', 'Canada', 'BMO'
                },
                domain_vocabulary={
                    'acn', 'ezlynx', 'summits', 'webinar', 'webinars',
                    'onboarding', 'workflows', 'roundtables',
                    'conference', 'membership'
                }
            )
        )
        print("✓ Intelligent text fixer loaded")
        print("="*80 + "\n")
    
    def _init_event_handler(self):
        """Initialize EventStore and EventQueryHandler for structured event queries"""
        print("Initializing Event Store...")
        try:
            # Try to load events from JSON file
            data_dir = Path("./acn_data")
            self.event_store = EventStore(str(data_dir))
            num_events = len(self.event_store.get_all_events())
            print(f"✓ Event Store loaded: {num_events} events")
            
            # Initialize the event query handler
            self.event_query_handler = EventQueryHandler(self.event_store)
            print("✓ Event Query Handler initialized")
            
        except Exception as e:
            print(f"⚠ Event Store initialization failed: {e}")
            print("  Event queries will fall back to vector search")
            self.event_store = None
            self.event_query_handler = None
    
    def _is_event_query(self, question: str) -> bool:
        """
        Check if the query is specifically about events/summits.
        
        Returns True ONLY for genuine event queries:
        - Explicit event terms (event, summit, conference) WITH action/list context
        - Time-based event queries (upcoming, next month, dates)
        - Queries about specific event details (when, where, registration)
        
        Returns False for:
        - Generic informational queries (what is acn, about acn)
        - Membership queries
        - General how-to questions
        """
        if not self.event_query_handler:
            return False
        
        question_lower = question.lower().strip()
        
        # STRICT: Must have event-related keywords AND be asking about events
        # Reject queries like "what is acn", "about acn", "what does acn do"
        
        # Pattern 1: Explicit action verbs asking for event information
        event_action_patterns = [
            r"\b(show|list|display|find|get)\s+(?:me\s+)?(?:the\s+)?(?:upcoming\s+)?(?:acn\s+)?(?:.*?\s+)?(?:events?|summits?|conferences?)",
            r"\bwhat\s+(?:events?|summits?|conferences?)\b",
            r"\b(upcoming|future|next|this\s+year|next\s+month)\s+(?:events?|summits?|conferences?)",
            r"\b(events?|summits?|conferences?)\s+(?:next|in|during|this)\s+(?:the\s+)?(?:few?\s+)?month",
            r"\bacn\s+(?:events?|summits?|conferences?)\b",
        ]
        
        for pattern in event_action_patterns:
            if re.search(pattern, question_lower):
                return True
        
        # Pattern 2: Specific event detail queries (when, where, registration)
        event_detail_patterns = [
            r"\bwhen\s+(?:is|are|was|were|will)\s+(?:the\s+)?(?:acn\s+)?(?:.*\s+)?(?:summit|event|conference)",
            r"\bwhere\s+(?:is|are|was|were|will)\s+(?:the\s+)?(?:acn\s+)?(?:.*\s+)?(?:summit|event|conference)",
            r"\bhow\s+(?:do|can|to)\s+(?:i|we|they)\s+(?:register|sign\s+up|attend)\s+(?:for\s+)?(?:the\s+)?(?:acn\s+)?(?:.*\s+)?(?:summit|event|conference)",
            r"\bregist(?:er|ration)\s+(?:for\s+)?(?:the\s+)?(?:acn\s+)?(?:.*\s+)?(?:summit|event|conference)",
            r"\bticket[s]?\s+(?:for|to)\s+(?:the\s+)?(?:acn\s+)?(?:.*\s+)?(?:summit|event|conference)",
        ]
        
        for pattern in event_detail_patterns:
            if re.search(pattern, question_lower):
                return True
        
        # Pattern 3: Direct event/summit mention with context
        # Must be talking ABOUT an event, not just mentioning it
        event_subject_patterns = [
            r"\btell\s+me\s+about\s+(?:the\s+)?(?:acn\s+)?(?:.*\s+)?(?:summit|event|conference)",
            r"\binfo(?:rmation)?\s+about\s+(?:the\s+)?(?:acn\s+)?(?:.*\s+)?(?:summit|event|conference)",
            r"\bdetails?\s+about\s+(?:the\s+)?(?:acn\s+)?(?:.*\s+)?(?:summit|event|conference)",
        ]
        
        for pattern in event_subject_patterns:
            if re.search(pattern, question_lower):
                return True
        
        # Pattern 4: Use EventQueryClassifier for structured classification
        # but ONLY if we have explicit event keywords
        explicit_event_keywords = [
            r"\bevent[s]?\b",
            r"\bsummit[s]?\b",
            r"\bconference[s]?\b",
            r"\bwebinar[s]?\b",
            r"\btraining\s+(?:session|event)s?\b",
            r"\bworkshop[s]?\b",
        ]
        
        has_explicit_event_keyword = any(re.search(p, question_lower) for p in explicit_event_keywords)
        
        if has_explicit_event_keyword:
            # Even with event keyword, use classifier to confirm intent
            intent, target, params = EventQueryClassifier.classify(question)
            return intent in (EventQueryIntent.EVENT_DETAIL, EventQueryIntent.EVENT_LIST)
        
        # No explicit event keywords - NOT an event query
        return False
    
    def _handle_event_query(self, question: str) -> Dict:
        """
        Handle event-related queries using structured event data.
        This bypasses vector search entirely and queries the EventStore directly.
        """
        start_time = datetime.now()
        
        # Use EventQueryHandler to process the query
        result = self.event_query_handler.query(question)
        
        # Build response in the same format as the main query method
        if result.events:
            answer = result.formatted_answer
            sources = [event.source_url for event in result.events[:3]]
            confidence = result.confidence
        else:
            answer = result.formatted_answer  # already contextual
            sources = []
            confidence = result.confidence
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Apply text fixing
        answer = self.text_fixer.fix(answer)
        answer = normalize_spacing(answer)
        
        return {
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
            "intent": result.intent.value if hasattr(result.intent, 'value') else str(result.intent),
            "num_docs": len(result.events),
            "processing_time": processing_time
        }
    
    def _load_embeddings(self):
        """Load embedding model"""
        print(f"Loading embeddings: {self.config.EMBEDDING_MODEL}")
        embeddings = HuggingFaceEmbeddings(
            model_name=self.config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        print("✓ Embeddings loaded")
        return embeddings
    
    def _load_chroma(self):
        """Load ChromaDB"""
        print(f"Loading ChromaDB from: {self.config.CHROMA_DIR}")
        db = Chroma(
            persist_directory=self.config.CHROMA_DIR,
            embedding_function=self.embeddings
        )
        print("✓ ChromaDB loaded")
        return db
    
    def _load_reranker(self):
        """Load reranker if GPU available"""
        if torch.cuda.is_available():
            try:
                return GPUReranker(
                    self.config.RERANKER_MODEL,
                    device="cuda"
                )
            except Exception as e:
                print(f"⚠ Reranker loading failed: {e}")
                return None
        return None
    
    def query(self, question: str, k: int = None) -> Dict:
        """
        Process query through complete pipeline.
        
        For event queries:
        1. First check if this is an event-related query
        2. If yes, use EventQueryHandler to query structured event data
        3. Return results from EventStore (deterministic, no LLM needed)
        
        For non-event queries:
        1. Use vector-based RAG pipeline
        2. Return LLM-generated response
        """
        
        start_time = datetime.now()
        
        # Step 0: Check if this is an EVENT QUERY - handle BEFORE vector search
        if self.event_query_handler and self._is_event_query(question):
            print(f"\n{'='*80}")
            print(f"Query: {question}")
            print(f"Routing to EVENT QUERY HANDLER (structured data)")
            print(f"{'='*80}")
            
            result = self._handle_event_query(question)
            result['processing_time'] = (datetime.now() - start_time).total_seconds()
            print(f"✓ Event query handled in {result['processing_time']:.2f}s")
            return result
        
        # Step 1: Classify query (for non-event queries)
        intent = self.classifier.classify(question)
        print(f"\n{'='*80}")
        print(f"Query: {question}")
        print(f"Category: {intent.category} | Temporal: {intent.temporal_mode if intent.is_temporal else 'N/A'}")
        print(f"{'='*80}")
        
        # Step 2: Retrieve documents (vector-based RAG for non-event queries)
        print("Retrieving documents...")
        k = k or self.config.FINAL_TOP_K
        docs = self.retriever.retrieve(question, intent, k=k)

        print(f"✓ Retrieved {len(docs)} documents")
        
        if not docs:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "confidence": 0.0,
                "sources": [],
                "intent": intent,
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
        
        # Step 3: Build context
        context = self._build_context(docs, intent)
        
        # Step 4: Generate answer
        print("Generating answer...")
        answer = self._generate_answer(question, context, intent, docs)
        
        # Step 5: Validate response
        if self.config.ENABLE_RESPONSE_VALIDATION:
            answer = self._validate_response(answer, docs, intent)
        
        # Extract sources
        sources = [doc.metadata.get("source") for doc in docs[:3]]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        print(f"✓ Completed in {processing_time:.2f}s")

        # Clean answer
        if intent.category in ["events", "membership"]:
            answer = self.text_fixer.fix(answer)
        answer = normalize_spacing(answer)

        return {
            "answer": answer,
            "confidence": 0.85,
            "sources": sources,
            "intent": intent.category,
            "num_docs": len(docs),
            "processing_time": processing_time
        }

    def _build_context(
        self,
        docs: List[Document],
        intent: QueryIntent
    ) -> str:
        """Build optimized context"""
        
        if intent.category == "events":
            # Structured event context
            parts = []
            for doc in docs:
                title = doc.metadata.get("title", "Event")
                date = doc.metadata.get("event_date", "Date TBD")
                parts.append(
                    f"Event Name: {title}\n"
                    f"Event Date: {date}\n"
                    f"Source URL: {doc.metadata.get('source')}\n"
                )

            context = "\n".join(parts)
        else:
            # Standard context
            context = "\n\n".join(
                f"[Source: {doc.metadata.get('source')}]\n{doc.page_content}"
                for doc in docs
            )
        
        return context[:6000]  # Limit context size
    
    def _generate_answer(
        self,
        question: str,
        context: str,
        intent: QueryIntent,
        docs: List[Document]
    ) -> str:
        """Generate answer using LLM or fallback"""
        
        if not self.llm.available:
            return self._extract_from_context(context, intent)
        
        # Build prompt
        current_date = datetime.now().strftime("%B %d, %Y")
        prompt = self.prompt_builder.build(question, context, intent, current_date)
        
        # Generate
        raw_output = self.llm.generate(prompt)
        
        # Extract answer from LLM output
        answer = raw_output.strip()
        
        # Try different markers to find where answer starts
        answer_markers = [
            "Answer:",
            "Answer (ONLY upcoming events",
            "Answer (Past events):",
            "Answer (Events in",
        ]
        
        for marker in answer_markers:
            if marker in answer:
                answer = answer.split(marker)[-1].strip()
                break
        
        # If no marker found, try to extract content after the question
        if "Question:" in answer:
            parts = answer.split("Question:")
            if len(parts) > 1:
                answer = parts[-1].strip()
                if "\n" in answer:
                    answer = answer.split("\n", 1)[-1].strip()
        
        # Clean up any remaining context tags
        answer = answer.replace("</context>", "").strip()
        answer = answer.replace("<context>", "").strip()
        
        # Remove incomplete last sentence if answer was cut off
        if answer and not answer.endswith(('.', '!', '?', '"', "'", ')', ':')):
            last_period = max(
                answer.rfind('.'),
                answer.rfind('!'),
                answer.rfind('?'),
                answer.rfind(':')
            )
            
            if last_period > len(answer) * 0.5:
                answer = answer[:last_period + 1]
        
        answer = answer.strip()
        
        return answer
    
    def _validate_response(
        self,
        answer: str,
        docs: List[Document],
        intent: QueryIntent
    ) -> str:
        """Validate response for hallucinations"""
        
        # Check for common hallucination patterns
        if "I don't have" in answer or "I don't know" in answer:
            return answer
        
        # For events, validate dates are in context
        if intent.category == "events" and intent.is_temporal:
            context_text = " ".join(doc.page_content for doc in docs)
            
            # Extract dates from answer
            answer_dates = re.findall(
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
                answer
            )
            
            # Check if dates are in context
            for date in answer_dates:
                if date not in context_text:
                    return "I found some event information, but I'm not confident about the specific dates. Please check the official ACN events page for accurate dates."
        
        return answer
    
    def _extract_from_context(
        self,
        context: str,
        intent: QueryIntent
    ) -> str:
        """Fallback: extract info without LLM"""
        
        lines = context.split('\n')
        relevant = []
        
        keywords = {
            "membership": ["benefit", "member", "join", "cost", "access"],
            "events": ["event", "date", "register", "summit", "conference"],
            "howto": ["step", "how", "guide", "instruction"],
        }.get(intent.category, [])
        
        for line in lines:
            if any(kw in line.lower() for kw in keywords):
                relevant.append(line.strip())
        
        return "\n".join(relevant[:20]) if relevant else context[:1500]


# ==================== USAGE EXAMPLE ====================

def main():
    """Test the complete system with event query handling"""
    
    # Initialize
    config = RAGConfig()
    engine = ACNRAGEngine(config)
    
    # Test queries - EVENT QUERIES should be handled by EventQueryHandler
    event_queries = [
        "Tell me about Dallas Summit",
        "When is the Dallas Summit?",
        "Show me upcoming events",
        "Show me upcoming summits",
        "Upcoming ACN events next month",
        "ACN events in 2026",
    ]
    
    # General queries - should use vector RAG
    general_queries = [
        "What are the membership benefits of ACN?",
        "How do I join Applied Client Network?",
        "Tell me about EZLynx integration",
    ]
    
    print("\n" + "="*80)
    print("TESTING ACN RAG SYSTEM - EVENT QUERIES")
    print("="*80 + "\n")
    
    for query in event_queries:
        result = engine.query(query)
        
        print(f"\n{'='*80}")
        print(f"Q: {query}")
        print(f"{'='*80}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nIntent: {result['intent']}")
        print(f"\nSources:")
        for source in result['sources']:
            print(f"  - {source}")
        print(f"\nProcessing Time: {result['processing_time']:.2f}s")
        print()
    
    print("\n" + "="*80)
    print("TESTING ACN RAG SYSTEM - GENERAL QUERIES")
    print("="*80 + "\n")
    
    for query in general_queries:
        result = engine.query(query)
        
        print(f"\n{'='*80}")
        print(f"Q: {query}")
        print(f"{'='*80}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nIntent: {result['intent']}")
        print(f"\nSources:")
        for source in result['sources']:
            print(f"  - {source}")
        print(f"\nProcessing Time: {result['processing_time']:.2f}s")
        print()


if __name__ == "__main__":
    main()
