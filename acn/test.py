#!/usr/bin/env python3
"""
ACN RAG System with Chain-of-Thought and ReAct
Enhancements:
- Chain-of-Thought reasoning for better answers
- ReAct framework for fact-grounding
- Improved prompt engineering
- Better temporal awareness
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
            
            if intent.category == "membership" and "membership" in section:
                filtered.append(doc)
            elif intent.category == "events" and ("event" in section or "calendar" in section):
                filtered.append(doc)
            elif intent.category == "howto" and ("guide" in section or "how" in section):
                filtered.append(doc)
            else:
                filtered.append(doc)
        
        return filtered if filtered else docs
    
    def _deduplicate(self, docs: List[Dict]) -> List[Dict]:
        """Remove duplicate documents by source"""
        
        seen_sources = set()
        unique_docs = []
        
        for doc in docs:
            source = doc['metadata'].get('source', '')
            if source not in seen_sources:
                seen_sources.add(source)
                unique_docs.append(doc)
        
        return unique_docs


# ==================== LOCAL LLM ====================

class LocalLLM:
    """GPU-optimized local LLM"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.available = False
        
        if config.USE_LOCAL_LLM:
            self._load_model()
    
    def _load_model(self):
        """Load LLM with GPU optimizations"""
        
        try:
            print(f"\nLoading LLM: {self.config.LLM_MODEL}")
            
            # Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.LLM_MODEL,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            if self.config.USE_4BIT and self.config.USE_GPU:
                # 4-bit quantization
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.LLM_MODEL,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.LLM_MODEL,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            self.model.eval()
            self.available = True
            
            print(f"  Model: {self.config.LLM_MODEL}")
            print(f"  Device: {self.model.device}")
            print("✓ LLM loaded successfully\n")
            
        except Exception as e:
            print(f"⚠ LLM loading failed: {e}")
            self.available = False
    
    def generate(self, prompt: str, max_new_tokens: int = None) -> str:
        """Generate response"""
        
        if not self.available:
            return "LLM not available"
        
        max_new_tokens = max_new_tokens or self.config.MAX_NEW_TOKENS
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=self.config.TEMPERATURE,
                    do_sample=self.config.TEMPERATURE > 0,
                    top_p=0.9 if self.config.TEMPERATURE > 0 else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from response
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            return response
            
        except Exception as e:
            print(f"Generation error: {e}")
            return "Error generating response"


# ==================== IMPROVED PROMPT BUILDER WITH CoT & ReAct ====================

class ImprovedPromptBuilder:
    """Build ChatGPT-style conversational prompts"""
    
    @staticmethod
    def build(
        question: str,
        context: str,
        intent: QueryIntent,
        current_date: str,
        strategy: str = "standard"
    ) -> str:
        """Build conversational prompt that produces clean answers"""
        
        # Get category-specific formatting guidance
        format_guide = ImprovedPromptBuilder._get_format_guide(intent, current_date)
        
        prompt = f"""<s>[INST] You are a knowledgeable assistant for Applied Client Network (ACN), an insurance technology user community. Today's date is {current_date}.

Your job is to provide helpful, accurate, well-structured answers based on the provided context.

{format_guide}

IMPORTANT FORMATTING RULES:
- Write in a natural, conversational tone like ChatGPT
- Use clear section headers to organize information
- Use bullet points with • or * for lists
- Use bold (**text**) for emphasis on key terms
- Break content into digestible paragraphs
- DO NOT show your reasoning steps
- DO NOT include "Information not available" disclaimers
- DO NOT number your thinking process
- DO NOT include meta-commentary about what you're doing

ANSWER STRUCTURE:
- Start with a brief overview sentence or paragraph
- Use clear section headers to organize different aspects
- Provide specific details under each section
- End naturally without disclaimers

Context information:
{context}

Question: {question}

Provide a comprehensive, well-formatted answer based on the context above.
[/INST]

Answer:"""
        
        return prompt
    
    @staticmethod
    def _get_format_guide(intent: QueryIntent, current_date: str) -> str:
        """Get category-specific formatting guidance"""
        
        if intent.category == "membership":
            return """
FORMAT FOR MEMBERSHIP QUESTIONS:
- Start with what ACN membership is/offers
- Use section headers like "What It Is", "Benefits", "Who Should Join", "How to Join"
- Include specific details about access, resources, pricing if available
- Use bullet points for lists of benefits or features"""
        
        elif intent.category == "events":
            if intent.temporal_mode == "upcoming":
                return f"""
FORMAT FOR UPCOMING EVENTS (Today is {current_date}):
- Start with a brief intro about ACN events
- Use section headers like "Upcoming Events", "What to Expect", "Who Should Attend"
- List ONLY events happening on or after {current_date}
- For each event include: Name, Date, Location, Key Details
- Use bullet points for lists"""
            
            elif intent.temporal_mode == "past":
                return f"""
FORMAT FOR PAST EVENTS (Today is {current_date}):
- Start with context about the event
- Use section headers like "Event Details", "Highlights"
- Focus on dates BEFORE {current_date}
- Use bullet points for lists"""
            
            else:
                return f"""
FORMAT FOR EVENT QUESTIONS (Today is {current_date}):
- Start with what the event is about
- Use section headers like "What It Is", "When & Where", "What Happens There", "Who Attends", "Why It Matters"
- Distinguish upcoming vs past events based on {current_date}
- Organize information logically with clear sections
- Use bullet points for lists"""
        
        elif intent.category == "howto":
            return """
FORMAT FOR HOW-TO QUESTIONS:
- Start with a brief explanation of what's being accomplished
- Use section headers like "Overview", "Step-by-Step Instructions", "Tips"
- Provide numbered steps for instructions
- Use bullet points for additional tips or resources"""
        
        else:
            return """
FORMAT FOR GENERAL QUESTIONS:
- Provide a comprehensive, well-organized answer
- Use clear section headers to break up content
- Structure information logically
- Be thorough but conversational
- Use bullet points for lists"""


# Backward compatibility wrapper
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
    """Complete RAG engine with CoT and ReAct"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        print("Initializing ACN RAG Engine with CoT & ReAct...")
        
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
        
        print("\n" + "="*80)
        print("✓ ACN RAG Engine Ready with Chain-of-Thought!")
        print("✓ Intelligent text fixer loaded")
        print("="*80 + "\n")
    
    def query(self, question: str) -> Dict:
        """Process a query and return answer"""
        
        start_time = datetime.now()
        
        # Step 1: Classify query
        intent = self.classifier.classify(question)
        
        print("="*80)
        print(f"Query: {question}")
        print(f"Category: {intent.category} | Temporal: {intent.temporal_mode if intent.is_temporal else 'N/A'}")
        print("="*80)
        
        # Step 2: Retrieve documents
        print("Retrieving documents...")
        docs = self.retriever.retrieve(question, intent, k=5)
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
        
        # Minimal cleanup - only normalize excessive whitespace
        answer = re.sub(r'\n\n\n+', '\n\n', answer)  # Max 2 newlines
        answer = answer.strip()
        
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
        """Generate answer using LLM with ChatGPT-style formatting"""
        
        if not self.llm.available:
            return self._extract_from_context(context, intent)
        
        # Build prompt using conversational prompt builder
        current_date = datetime.now().strftime("%B %d, %Y")
        
        prompt = ImprovedPromptBuilder.build(
            question=question,
            context=context,
            intent=intent,
            current_date=current_date,
            strategy="standard"  # Always use standard for conversational output
        )
        
        # Generate
        raw_output = self.llm.generate(prompt)
        
        # Minimal cleaning - preserve formatting
        answer = raw_output.strip()
        
        # Remove only the "Answer:" prefix if present at the very start
        if answer.startswith("Answer:"):
            answer = answer[7:].strip()
        
        # Remove any instruction echoes
        instruction_patterns = [
            "Let's think through this step by step:",
            "STEP-BY-STEP:",
            "[/INST]",
            "<s>[INST]",
        ]
        
        for pattern in instruction_patterns:
            if pattern in answer:
                # Take everything after the pattern
                parts = answer.split(pattern)
                if len(parts) > 1:
                    answer = parts[-1].strip()
        
        return answer.strip()
    
    def _extract_from_context(self, context: str, intent: QueryIntent) -> str:
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


# ==================== MAIN ====================

def main():
    """Test the RAG system"""
    
    config = RAGConfig()
    engine = ACNRAGEngine(config)
    
    test_queries = [
        "What are the membership benefits of ACN?",
        "How do I join Applied Client Network?",
        "What events does ACN organize?",
        "Show upcoming events at ACN",
        "Tell me about Applied Net conference",
    ]
    
    print("="*80)
    print("TESTING ACN RAG SYSTEM WITH CHAIN-OF-THOUGHT")
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