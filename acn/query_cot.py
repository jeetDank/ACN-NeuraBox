#!/usr/bin/env python3
"""
Complete GPU-Optimized ACN RAG System - WITH EVENT QUERY HANDLING AND CoT
Features:
- Event and Summit related queries handled BEFORE vector-based RAG
- "Upcoming" and date-based event queries work deterministically
- Chain-of-Thought (CoT) reasoning for better answer quality
- ReAct framework for fact-grounding
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
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    text = re.sub(r'([.!?;,])([A-Z])', r'\1 \2', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ==================== QUERY INTENT ====================

@dataclass
class QueryIntent:
    """Structured query intent"""
    category: str  # membership, events, howto, resources, general
    is_temporal: bool
    temporal_mode: Optional[str]  # upcoming, past, specific_year, all
    year: Optional[int] = None
    confidence: float = 0.0


class IntentClassifier:
    """Classify query intent"""
    
    PATTERNS = {
        "membership": [
            r'\bmembership\b', r'\bjoin\b', r'\bmember\b',
            r'\benroll\b', r'\bsign\s*up\b', r'\bregister\b',
            r'\bbenefits?\b', r'\bprice\b', r'\bcost\b'
        ],
        "events": [
            r'\bevent\b', r'\bsummit\b', r'\bconference\b',
            r'\bwebinar\b', r'\bmeeting\b', r'\bgathering\b',
            r'\bworkshop\b', r'\bseminar\b'
        ],
        "howto": [
            r'\bhow\s+to\b', r'\bhow\s+do\s+i\b', r'\bhow\s+can\s+i\b',
            r'\bsteps?\b', r'\bprocess\b', r'\bprocedure\b',
            r'\bguide\b', r'\btutorial\b', r'\binstructions?\b'
        ],
        "resources": [
            r'\bresources?\b', r'\bdocument\b', r'\bfile\b',
            r'\blink\b', r'\bdownload\b', r'\baccess\b'
        ]
    }
    
    TEMPORAL_PATTERNS = {
        "upcoming": r'\bupcoming\b|\bnext\b|\bfuture\b|\bscheduled\b|\bcoming\s+up\b',
        "past": r'\bpast\b|\bprevious\b|\blast\b|\bearlier\b|\bhappened\b',
        "specific_year": r'\b(20\d{2})\b'
    }
    
    @classmethod
    def classify(cls, question: str) -> QueryIntent:
        """Classify query intent"""
        
        question_lower = question.lower()
        
        # Category
        category_scores = {}
        for cat, patterns in cls.PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, question_lower, re.I))
            if score > 0:
                category_scores[cat] = score
        
        category = max(category_scores, key=category_scores.get) if category_scores else "general"
        confidence = category_scores.get(category, 0) / 10.0
        
        # Temporal
        is_temporal = False
        temporal_mode = None
        year = None
        
        for mode, pattern in cls.TEMPORAL_PATTERNS.items():
            match = re.search(pattern, question_lower, re.I)
            if match:
                is_temporal = True
                if mode == "specific_year":
                    temporal_mode = "specific_year"
                    year = int(match.group(1))
                else:
                    temporal_mode = mode
                break
        
        return QueryIntent(
            category=category,
            is_temporal=is_temporal,
            temporal_mode=temporal_mode,
            year=year,
            confidence=confidence
        )


# ==================== LLM WRAPPER ====================

class LLMWrapper:
    """Wrapper for local LLM with GPU optimization"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.available = False
        self._load_model()
    
    def _load_model(self):
        """Load model with optimizations"""
        
        try:
            print("âš™ Loading LLM with GPU optimizations...")
            
            # Quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.llm_model,
                trust_remote_code=True
            )
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.llm_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            self.model.eval()
            self.available = True
            print(f"✓ LLM loaded: {self.config.llm_model}")
            
        except Exception as e:
            print(f"⚠ LLM not available: {e}")
            self.available = False
    
    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate response"""
        
        if not self.available:
            return "LLM not available"
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
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


# ==================== IMPROVED PROMPT BUILDER WITH CoT ====================

class ImprovedPromptBuilder:
    """Build intelligent prompts with Chain-of-Thought and ReAct reasoning"""
    
    @staticmethod
    def build(
        question: str,
        context: str,
        intent: QueryIntent,
        current_date: str,
        strategy: str = "standard"
    ) -> str:
        """Build optimized prompt with chain-of-thought and ReAct"""
        
        if strategy == "strict":
            return ImprovedPromptBuilder._build_strict_prompt(question, context, intent, current_date)
        else:
            return ImprovedPromptBuilder._build_standard_prompt(question, context, intent, current_date)
    
    @staticmethod
    def _build_standard_prompt(
        question: str,
        context: str,
        intent: QueryIntent,
        current_date: str
    ) -> str:
        """Standard prompt with CoT and ReAct"""
        
        # Build thinking instructions
        thinking = ImprovedPromptBuilder._build_thinking_instructions(intent)
        
        # Build category-specific instructions
        specific = ImprovedPromptBuilder._build_specific_instructions(intent, current_date)
        
        # Build validation rules
        rules = ImprovedPromptBuilder._build_validation_rules(intent, current_date)
        
        prompt = f"""You are a helpful assistant for Applied Client Network (ACN).
Today's date: {current_date}

{thinking}

REACT FRAMEWORK - Reason and Act:

REASON:
- What does the question require?
- What context is available?
- Are there temporal constraints (upcoming/past)?
- What facts are EXPLICITLY stated vs. inferred?

ACT:
- Extract only facts explicitly stated in context
- Do NOT invent details
- For temporal queries: Compare dates to TODAY ({current_date})
- Validate each claim against the source material

{specific}

{rules}

<context>
{context}
</context>

Question: {question}

Let's think through this step by step:

1. ANALYZE: What does the question ask for?
2. SEARCH: What relevant information exists in the context?
3. VALIDATE: Is each claim supported by the context?
4. FORMULATE: Provide the answer based only on validated facts

Answer:"""
        
        return prompt
    
    @staticmethod
    def _build_strict_prompt(
        question: str,
        context: str,
        intent: QueryIntent,
        current_date: str
    ) -> str:
        """Ultra-strict fact-grounded prompt"""
        
        specific = ImprovedPromptBuilder._build_specific_instructions(intent, current_date)
        
        prompt = f"""You are a fact-grounded assistant for Applied Client Network (ACN).
Today's date: {current_date}

STRICT GROUNDING MODE:

Your job: Extract and relay information from context ONLY.
You MUST NOT generate, invent, or infer anything beyond what's written.

For EVERY fact you mention, indicate its source.
Format: "Claim (Evidence: exact text from context)"

CRITICAL RULES FOR ALL QUERIES:
✓ MUST cite context for every claim
✓ MUST use exact dates/names/details from context
✓ MUST state "Information not available" for gaps
✓ MUST NOT invent dates, times, or details
✓ MUST NOT merge or paraphrase aggressively
✓ MUST NOT assume or infer beyond what's written
✓ MUST NOT mention names/organizations not in context

{specific}

<context>
{context}
</context>

Question: {question}

STEP-BY-STEP:
1. What facts are needed to answer?
2. Which of these facts appear explicitly in the context?
3. Which facts are missing?
4. Provide answer with evidence references only

Answer:"""
        
        return prompt
    
    @staticmethod
    def _build_thinking_instructions(intent: QueryIntent) -> str:
        """Build chain-of-thought thinking instructions"""
        
        thinking = """CHAIN-OF-THOUGHT THINKING:
Step 1: What is the user specifically asking?
Step 2: What category of information do they need?
Step 3: What key facts must be extracted?
Step 4: Are there dates/times that matter?
Step 5: Is any important information missing?"""
        
        if intent.category == "events":
            thinking += """
Step 6: What are the event dates? BEFORE or AFTER today?
Step 7: Do events match temporal requirement (upcoming/past/specific year)?"""
        
        return thinking
    
    @staticmethod
    def _build_specific_instructions(intent: QueryIntent, current_date: str) -> str:
        """Build category-specific instructions"""
        
        if intent.category == "membership":
            return """
MEMBERSHIP QUERY REQUIREMENTS:
- List membership benefits clearly and specifically
- Include pricing/cost information if available
- Explain how to join step-by-step
- Mention renewal process if relevant
- Use bullet points for clarity
- DO NOT add generic benefits not explicitly stated
- DO NOT assume membership tiers not in context"""
        
        elif intent.category == "events":
            return f"""
EVENT QUERY REQUIREMENTS:
- CRITICAL: Today is {current_date}
- Extract: Name | Date | Time | Location | Details
- UPCOMING queries: ONLY include events with dates >= {current_date}
- PAST queries: ONLY include events with dates < {current_date}
- SPECIFIC YEAR queries: ONLY include events from that year
- Include registration/links if available
- If dates missing: state "Date not specified"
- DO NOT assume future events if unclear
- DO NOT include events outside requested timeframe
- List each event separately with all details"""
        
        elif intent.category == "howto":
            return """
HOW-TO QUERY REQUIREMENTS:
- Provide numbered, step-by-step instructions
- Include prerequisites/requirements
- Link to relevant resources from context
- Explain technical terms
- DO NOT add steps not in context
- Use clear sequential numbering"""
        
        else:
            return """
GENERAL QUERY REQUIREMENTS:
- Provide accurate, factual information
- Preserve original wording and structure
- Be specific and detailed
- DO NOT paraphrase aggressively
- DO NOT merge or alter technical terms
- DO NOT add opinions or editorializing"""
    
    @staticmethod
    def _build_validation_rules(intent: QueryIntent, current_date: str) -> str:
        """Build fact validation rules"""
        
        rules = """
VALIDATION RULES:
✓ Answer ONLY from provided context
✓ Include specific dates/times for events
✓ State "I don't have that information" for gaps
✓ Ground every claim with context reference
✓ Preserve exact names, dates, terminology

✗ Do NOT invent facts or details
✗ Do NOT merge or alter words
✗ Do NOT add opinions or editorializing
✗ Do NOT mention people/orgs not named
✗ Do NOT assume missing information
✗ Do NOT hallucinate links or resources"""
        
        if intent.category == "events":
            rules += f"""

TEMPORAL VALIDATION:
- Compare event dates to TODAY ({current_date})
- If "upcoming" requested but event is past: EXCLUDE
- If "past" requested but event is future: EXCLUDE
- If date unclear: state "Date information not available"
- List dates in format: "Month DD, YYYY" or "Month DD-DD, YYYY\""""
        
        return rules


# Keep backward compatibility
class PromptBuilder:
    """Backward compatible wrapper"""
    
    @staticmethod
    def build(
        question: str,
        context: str,
        intent: QueryIntent,
        current_date: str
    ) -> str:
        """Build prompt - redirects to ImprovedPromptBuilder"""
        return ImprovedPromptBuilder.build(
            question=question,
            context=context,
            intent=intent,
            current_date=current_date,
            strategy="standard"
        )


# ==================== MAIN RAG ENGINE ====================

class ACNRAGEngine:
    """
    Complete GPU-Optimized RAG System - WITH EVENT QUERY HANDLING AND CoT
    
    Event queries are handled BEFORE vector-based RAG using structured event data.
    Event Calendar and Learning/Webinar pages are NEVER returned for event queries.
    """
    
    def __init__(self, config: RAGConfig = None):
        """Initialize RAG Engine"""
        
        self.config = config or RAGConfig()
        
        # Initialize components
        print("🚀 Initializing ACN RAG Engine with CoT...")
        
        # Event handling
        print("📅 Loading event store...")
        self.event_store = EventStore()
        self.event_store.load_events()
        
        self.event_classifier = EventQueryClassifier()
        self.event_handler = EventQueryHandler(self.event_store)
        
        # Vector store
        print("📚 Loading vector store...")
        self._init_vector_store()
        
        # LLM
        print("🤖 Loading LLM...")
        self.llm = LLMWrapper(self.config)
        
        # Reranker
        if self.config.use_reranker:
            print("🎯 Loading reranker...")
            self.reranker = CrossEncoder(self.config.reranker_model)
        else:
            self.reranker = None
        
        # Components
        self.intent_classifier = IntentClassifier()
        self.prompt_builder = PromptBuilder()
        
        print("✅ RAG Engine initialized with CoT support!")
    
    def _init_vector_store(self):
        """Initialize vector store"""
        
        embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        
        self.vector_store = Chroma(
            persist_directory=str(self.config.chroma_db_path),
            embedding_function=embeddings
        )
    
    def query(self, question: str, k: int = 5) -> str:
        """
        Main query method with event handling
        
        Flow:
        1. Check if this is an event query
        2. If yes, use EventQueryHandler (structured data)
        3. If no, use standard RAG pipeline
        """
        
        print(f"\n{'='*60}")
        print(f"QUERY: {question}")
        print(f"{'='*60}")
        
        # Step 1: Check for event queries
        event_intent = self.event_classifier.classify(question)
        
        if event_intent.is_event_query:
            print(f"✓ Detected EVENT query (confidence: {event_intent.confidence:.2f})")
            print(f"  Type: {event_intent.query_type}")
            print(f"  Temporal: {event_intent.temporal_filter}")
            
            result = self.event_handler.handle_query(question, event_intent)
            
            if result.success and result.events:
                print(f"✓ Found {len(result.events)} events via structured data")
                return result.formatted_response
            else:
                print(f"⚠ No events found, falling back to RAG")
                # Fall through to RAG
        
        # Step 2: Use standard RAG for non-event queries
        print("→ Using standard RAG pipeline")
        
        # Classify intent
        intent = self.intent_classifier.classify(question)
        print(f"  Category: {intent.category}")
        print(f"  Temporal: {intent.is_temporal} ({intent.temporal_mode})")
        
        # Retrieve documents
        docs = self._retrieve_documents(question, k)
        
        # Build context
        context = self._build_context(docs, intent)
        
        # Generate answer with CoT
        answer = self._generate_answer(question, context, intent, docs)
        
        # Validate
        validated_answer = self._validate_response(answer, docs, intent)
        
        return validated_answer
    
    def _retrieve_documents(self, question: str, k: int) -> List[Document]:
        """Retrieve relevant documents"""
        
        print(f"🔍 Retrieving top {k} documents...")
        
        # Vector search
        docs = self.vector_store.similarity_search(question, k=k*2)
        
        # Rerank if available
        if self.reranker and len(docs) > k:
            print("🎯 Reranking...")
            pairs = [[question, doc.page_content] for doc in docs]
            scores = self.reranker.predict(pairs)
            
            # Sort by score
            scored_docs = list(zip(docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            docs = [doc for doc, _ in scored_docs[:k]]
        else:
            docs = docs[:k]
        
        print(f"✓ Retrieved {len(docs)} documents")
        return docs
    
    def _build_context(self, docs: List[Document], intent: QueryIntent) -> str:
        """Build context from documents"""
        
        context_parts = []
        
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            
            # Fix text issues
            content = normalize_spacing(content)
            
            # Add to context
            context_parts.append(f"[Source {i}]\n{content}\n")
        
        context = "\n".join(context_parts)
        
        print(f"📄 Built context from {len(docs)} sources ({len(context)} chars)")
        
        return context
    
    def _generate_answer(
        self,
        question: str,
        context: str,
        intent: QueryIntent,
        docs: List[Document]
    ) -> str:
        """Generate answer using LLM or fallback with improved prompting"""
        
        if not self.llm.available:
            return self._extract_from_context(context, intent)
        
        # Build prompt using improved builder with CoT and ReAct
        current_date = datetime.now().strftime("%B %d, %Y")
        
        # Use strict strategy for critical queries
        strategy = "strict" if intent.category in ["events", "membership"] else "standard"
        
        prompt = ImprovedPromptBuilder.build(
            question=question,
            context=context,
            intent=intent,
            current_date=current_date,
            strategy=strategy
        )
        
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
            "Answer (Specific Year:",
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
                # Skip reasoning steps
                if "Let's think" in answer or "STEP" in answer:
                    if "\n\n" in answer:
                        answer = answer.split("\n\n")[-1].strip()
                if "\n" in answer:
                    answer = answer.split("\n", 1)[-1].strip()
        
        # Clean up any remaining artifacts
        answer = answer.replace("</context>", "").strip()
        answer = answer.replace("<context>", "").strip()
        answer = re.sub(r'^(Step \d+:|STEP|ANALYZE|SEARCH|VALIDATE|FORMULATE):.*?\n', '', answer, flags=re.MULTILINE)
        
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
        hallucination_patterns = [
            r'"[^"]*said[^"]*"',  # Fake quotes
            r'according to [A-Z][a-z]+ [A-Z][a-z]+',  # Fake attributions
            r'testimonial',
            r'review',
        ]
        
        for pattern in hallucination_patterns:
            if re.search(pattern, answer, re.I):
                context_text = " ".join(doc.page_content for doc in docs)
                if not re.search(pattern, context_text, re.I):
                    print("⚠ Potential hallucination detected")
        
        return answer
    
    def _extract_from_context(self, context: str, intent: QueryIntent) -> str:
        """Fallback extraction when LLM unavailable"""
        
        sentences = re.split(r'[.!?]+', context)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        return " ".join(sentences[:3]) + "..."


# ==================== CLI ====================

def main():
    """Test the RAG engine"""
    
    # Initialize
    engine = ACNRAGEngine()
    
    # Test queries
    test_queries = [
        "When is the Dallas Summit?",
        "What are ACN membership benefits?",
        "Show me upcoming summits",
        "What events happened in 2024?",
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        answer = engine.query(query)
        print(f"\n📝 ANSWER:\n{answer}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()