#!/usr/bin/env python3
"""
Fixed ACN RAG System - Corrected Retrieval Logic
Key fixes:
- Proper ChromaDB collection retrieval
- Correct embedding model initialization
- Fixed similarity search
- Better error handling
"""

import re
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Core libraries
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import CrossEncoder, SentenceTransformer

# ChromaDB
import chromadb
from chromadb.config import Settings

# Local config
from config import RAGConfig
from intelligent_text_fixer import IntelligentTextFixer, fix_text

# Import event query handler (if available)
try:
    import sys
    sys.path.insert(0, '..')
    from event_query_handler import EventQueryHandler, EventStore
    HAS_EVENT_HANDLER = True
except ImportError:
    HAS_EVENT_HANDLER = False


# ==================== TEXT NORMALIZATION ====================

def normalize_spacing(text: str) -> str:
    """Fix spacing issues in text"""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([.,!?])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ==================== QUERY CLASSIFICATION ====================

@dataclass
class QueryIntent:
    """Structured query intent"""
    category: str  # learning, membership, events, howto, resources, about, general
    is_temporal: bool
    temporal_mode: Optional[str]  # upcoming, past, specific_year, all
    year: Optional[int] = None
    confidence: float = 0.0


class QueryClassifier:
    """Intelligent query classification"""
    
    def __init__(self):
        self.patterns = {
            "learning": [
                r"cours[e]?s?|learning|training(?! session)|education|curriculum|class|lesson|tutorial|webinar|workshop|certification|study|learn|teach|instruction|epic (?:course|training|tip|technique)|applied epic|learning (?:center|resource|material)|deep dive|recorded on",
            ],
            "membership": [
                r"member(?:ship)?|join|benefit|cost|price|fee|renew|subscription|sign up|register",
            ],
            "events": [
                r"event|summit|conference|(?:live )?training session|calendar|schedule",
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
        
        # Check for specific year
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
        documents: List[Dict],
        top_k: int = 5
    ) -> List[Tuple[Dict, float]]:
        """Rerank documents using cross-encoder"""
        
        if not documents:
            return []
        
        # Prepare pairs for reranking
        pairs = [[query, doc['content']] for doc in documents]
        
        # Get relevance scores
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Combine documents with scores
        doc_score_pairs = list(zip(documents, scores))
        
        # Sort by score (descending)
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return doc_score_pairs[:top_k]


# ==================== IMPROVED RETRIEVER ====================

class ImprovedRetriever:
    """Fixed retriever with proper ChromaDB integration"""
    
    def __init__(
        self,
        chroma_client: chromadb.PersistentClient,
        collection_name: str,
        embedding_model: SentenceTransformer,
        reranker: Optional[GPUReranker] = None,
        config: RAGConfig = None
    ):
        self.client = chroma_client
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.config = config or RAGConfig()
        
        # Get collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
            print(f"✓ Collection '{self.collection_name}' loaded with {self.collection.count()} documents")
        except Exception as e:
            raise ValueError(f"Failed to load collection '{self.collection_name}': {e}")
    
    def retrieve(
        self,
        query: str,
        intent: QueryIntent,
        k: int = 20
    ) -> List[Dict]:
        """Retrieve relevant documents"""
        
        # Step 1: Generate query embedding
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True).tolist()
        
        # Step 2: Search ChromaDB
        initial_k = k * 2 if self.reranker else k
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(initial_k, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return []
        
        # Step 3: Convert to document format
        documents = []
        if results['documents'] and results['documents'][0]:
            for i, doc_content in enumerate(results['documents'][0]):
                doc = {
                    'content': doc_content,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 1.0
                }
                documents.append(doc)
        
        # Step 4: Filter by metadata
        documents = self._filter_by_metadata(documents, intent)
        
        # Step 5: Rerank if available
        if self.reranker and documents:
            doc_score_pairs = self.reranker.rerank(query, documents, top_k=k)
            documents = [doc for doc, score in doc_score_pairs]
        
        # Step 6: Deduplicate by source
        documents = self._deduplicate(documents)
        
        return documents[:k]
    
    def _filter_by_metadata(
        self,
        docs: List[Dict],
        intent: QueryIntent
    ) -> List[Dict]:
        """Filter documents by metadata"""
        
        if intent.category == "general":
            return docs
        
        # Filter by section
        filtered = []
        for doc in docs:
            section = doc['metadata'].get('section', '').lower()
            
            if intent.category == "learning" and "learning" in section:
                filtered.insert(0, doc)  # Prioritize
            elif intent.category == "membership" and "membership" in section:
                filtered.insert(0, doc)  # Prioritize
            elif intent.category == "events" and "events" in section:
                filtered.insert(0, doc)  # Prioritize
            else:
                filtered.append(doc)
        
        return filtered if filtered else docs
    
    def _deduplicate(self, docs: List[Dict]) -> List[Dict]:
        """Remove duplicate sources"""
        seen = set()
        unique = []
        
        for doc in docs:
            source = doc['metadata'].get('source', '')
            if source not in seen:
                seen.add(source)
                unique.append(doc)
        
        return unique


# ==================== LLM WRAPPER ====================

class LocalLLM:
    """GPU-optimized local LLM"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.available = False
        
        device = "cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu"
        
        print(f"Loading LLM: {config.LLM_MODEL}")
        print(f"Device: {device}")
        
        try:
            # Quantization config
            if config.USE_4BIT and device == "cuda":
                print("Using 4-bit quantization...")
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                quant_config = None
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                config.LLM_MODEL,
                quantization_config=quant_config,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
            
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.device = device
            self.available = True
            
            # Print memory usage
            if device == "cuda":
                memory_mb = torch.cuda.max_memory_allocated() / 1024**3
                print(f"✓ LLM loaded successfully")
                print(f"  Model: {config.LLM_MODEL}")
                print(f"  Device: {self.model.device}")
                print(f"  Memory: {memory_mb:.2f}GB")
            else:
                print(f"✓ LLM loaded on CPU")
            
        except Exception as e:
            print(f"✗ Failed to load LLM: {e}")
            print("  Continuing without LLM (using extraction fallback)")
            self.available = False
    
    def generate(self, prompt: str) -> str:
        """Generate response"""
        
        if not self.available:
            return "LLM not available"
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.MAX_NEW_TOKENS,
                    temperature=self.config.TEMPERATURE,
                    top_p=self.config.TOP_P,
                    top_k=self.config.TOP_K,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt from response
            if prompt in response:
                response = response.split(prompt)[-1].strip()
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Error generating response"


# ==================== PROMPT BUILDER ====================

class PromptBuilder:
    """Builds optimized prompts"""
    
    @staticmethod
    def build(question: str, context: str, intent: QueryIntent, current_date: str) -> str:
        """Build prompt for LLM"""
        
        if intent.category == "events":
            temporal_instruction = {
                "upcoming": f"Today is {current_date}. List ONLY events happening AFTER today.",
                "past": f"Today is {current_date}. List ONLY events that happened BEFORE today.",
                "specific_year": f"List ONLY events from {intent.year}.",
            }.get(intent.temporal_mode, "")
            
            prompt = f"""<s>[INST]
You are an ACN (Applied Client Network) events assistant.

Use ONLY the context below.
List information in a structured format.

Follow this structure:
1. Event Name
2. Date & Time
3. Location
4. Description
5. Registration Details (if available)

If multiple events exist, list each separately.
If no events match, clearly state that.

{temporal_instruction}

Context:
{context}

Question:
{question}
[/INST]

Answer:""" 
        elif intent.category == "membership":
            prompt = f"""<s>[INST]
        You are an ACN (Applied Client Network) membership assistant.

        Use ONLY the provided context.

        Format the response using the following structure:
        1. Overview
        2. Membership Benefits
        3. Eligibility
        4. How to Join
        5. Summary

        Use bullet points.
        Avoid paragraphs longer than 3 lines.

        Context:
        {context}

        Question:
        {question}
        [/INST]

        Answer:"""


        else:
            prompt = f"""<s>[INST]
You are an ACN (Applied Client Network) knowledge assistant.

Use ONLY the information provided in the context below.
Do NOT add assumptions or external knowledge.

Format your response in a clear, structured, and professional manner.

Follow this structure EXACTLY:
1. Overview
2. Key Details
3. Additional Information (if applicable)
4. Summary

Use bullet points where appropriate.
Use clear section headings.
Do not write long paragraphs.

Context:
{context}

Question:
{question}
[/INST]

Answer:"""
        
        return prompt


# ==================== MAIN RAG ENGINE ====================

class ACNRAGEngine:
    """Complete RAG engine with fixed retrieval"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        print("Initializing ACN RAG Engine...")
        
        # Initialize embedding model
        print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        print("✓ Embedding model loaded")
        
        # Initialize ChromaDB
        print(f"Loading ChromaDB from: {config.CHROMA_DIR}")
        self.chroma_client = chromadb.PersistentClient(
            path=config.CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        print("✓ ChromaDB loaded")
        
        # Initialize reranker
        device = "cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu"
        self.reranker = GPUReranker(config.RERANKER_MODEL, device) if config.USE_RERANKER else None
        
        # Initialize retriever
        self.retriever = ImprovedRetriever(
            chroma_client=self.chroma_client,
            collection_name="acn_knowledge_base",
            embedding_model=self.embedding_model,
            reranker=self.reranker,
            config=config
        )
        
        # Initialize LLM
        self.llm = LocalLLM(config)
        
        # Initialize other components
        self.classifier = QueryClassifier()
        self.prompt_builder = PromptBuilder()
        self.text_fixer = IntelligentTextFixer()
        
        # Initialize event handler if available
        self.event_handler = None
        if HAS_EVENT_HANDLER:
            try:
                event_store = EventStore()
                self.event_handler = EventQueryHandler(event_store)
                print("✓ Event query handler loaded")
            except Exception as e:
                print(f"⚠ Event query handler not available: {e}")
        
        print("\n" + "="*80)
        print("✓ ACN RAG Engine Ready!")
        print("✓ Intelligent text fixer loaded")
        print("="*80 + "\n")
    
    def query(self, question: str, k: int = 5) -> Dict:
        """Process a query and return answer"""
        
        start_time = datetime.now()
        
        # Step 1: Try event handler first if available
        if self.event_handler and hasattr(self.event_handler, 'query'):
            try:
                event_result = self.event_handler.query(question)
                # Check if this was an event query with results
                if event_result.ui and event_result.events:
                    # Return with UI data for event cards
                    return {
                        "answer": event_result.formatted_answer,
                        "confidence": event_result.confidence,
                        "sources": [e.source_url for e in event_result.events[:3]],
                        "intent": QueryIntent(category="events", is_temporal=False),
                        "num_docs": len(event_result.events),
                        "processing_time": (datetime.now() - start_time).total_seconds(),
                        "ui": event_result.ui  # Pass UI data to API
                    }
            except Exception as e:
                print(f"Event handler error: {e}")
                pass  # Fall through to standard RAG
        
        # Step 2: Classify query
        intent = self.classifier.classify(question)
        
        print("="*80)
        print(f"Query: {question}")
        print(f"Category: {intent.category} | Temporal: {intent.temporal_mode if intent.is_temporal else 'N/A'}")
        print("="*80)
        
        # Step 2: Retrieve documents
        print("Retrieving documents...")
        docs = self.retriever.retrieve(question, intent, k=k)
        print(f"✓ Retrieved {len(docs)} documents")
        
        # Return early if no documents found
        if not docs:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "confidence": 0.0,
                "sources": [],
                "intent": intent.category,
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
        
        # Step 3: Build context
        context = self._build_context(docs, intent)
        
        # Step 4: Generate answer
        print("Generating answer...")
        answer = self._generate_answer(question, context, intent, docs)
        
        # Step 5: Clean and fix answer
        if intent.category in ["events", "membership"]:
            answer = self.text_fixer.fix(answer)
        answer = normalize_spacing(answer)
        
        # Extract sources
        sources = [doc['metadata'].get('source', 'Unknown') for doc in docs[:3]]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        print(f"✓ Completed in {processing_time:.2f}s")
        
        return {
            "answer": answer,
            "confidence": 0.85,
            "sources": sources,
            "intent": intent.category,
            "num_docs": len(docs),
            "processing_time": processing_time
        }
    
    def _build_context(self, docs: List[Dict], intent: QueryIntent) -> str:
        """Build context from documents"""
        
        context_parts = []
        for doc in docs:
            title = doc['metadata'].get('title', 'Untitled')
            source = doc['metadata'].get('source', 'Unknown')
            content = doc['content']
            
            context_parts.append(
                f"Title: {title}\n"
                f"Source: {source}\n"
                f"Content: {content}\n"
            )
        
        return "\n\n".join(context_parts)[:6000]
    
    def _generate_answer(
        self,
        question: str,
        context: str,
        intent: QueryIntent,
        docs: List[Dict]
    ) -> str:
        """Generate answer using LLM"""
        
        if not self.llm.available:
            return self._extract_from_context(context, intent)
        
        # Build prompt
        current_date = datetime.now().strftime("%B %d, %Y")
        prompt = self.prompt_builder.build(question, context, intent, current_date)
        
        # Generate
        raw_output = self.llm.generate(prompt)
        
        # Clean output
        answer = raw_output.strip()
        
        # Remove answer markers
        for marker in ["Answer:", "Answer (", "Question:"]:
            if marker in answer:
                answer = answer.split(marker)[-1].strip()
                break
        
        # Remove incomplete sentences
        if answer and not answer.endswith(('.', '!', '?', ':', '"', "'")):
            last_period = max(
                answer.rfind('.'),
                answer.rfind('!'),
                answer.rfind('?'),
                answer.rfind(':')
            )
            
            if last_period > len(answer) * 0.5:
                answer = answer[:last_period + 1]
        
        return answer.strip()
    
    def _extract_from_context(self, context: str, intent: QueryIntent) -> str:
        """Fallback: extract info without LLM"""
        
        lines = context.split('\n')
        relevant = []
        
        keywords = {
            "learning": ["course", "training", "education", "lesson", "workshop", "webinar", "epic", "recorded"],
            "membership": ["benefit", "member", "join", "cost", "access"],
            "events": ["event", "date", "register", "summit", "conference"],
            "howto": ["step", "how", "guide", "instruction"],
        }.get(intent.category, [])
        
        for line in lines:
            if any(kw in line.lower() for kw in keywords):
                relevant.append(line.strip())
        
        return "\n".join(relevant[:20]) if relevant else context[:1500]


# ==================== MAIN ====================

def main():
    """Test the RAG system"""
    
    config = RAGConfig()
    engine = ACNRAGEngine(config)
    
    test_queries = [
        "What courses does ACN offer?",
        "Tell me about ACN training",
        "Show me Epic courses",
        "What are the membership benefits of ACN?",
        "How do I join Applied Client Network?",
        "What events does ACN organize?",
        "Show upcoming events at ACN",
        "Tell me about Applied Net conference",
    ]
    
    print("="*80)
    print("TESTING ACN RAG SYSTEM")
    print("="*80 + "\n")
    
    for query in test_queries:
        result = engine.query(query)
        
        print(f"\n{'='*80}")
        print(f"Q: {query}")
        print(f"{'='*80}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources:")
        for source in result['sources']:
            print(f"  - {source}")
        print(f"\nProcessing Time: {result['processing_time']:.2f}s")
        print()


if __name__ == "__main__":
    main()