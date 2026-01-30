"""
Configuration for ACN RAG System
FIXED VERSION - Increased tokens for complete answers
"""

from dataclasses import dataclass
from typing import List

@dataclass
class RAGConfig:
    """Complete configuration for the RAG system"""
    
    # ========== MODEL CONFIGURATION ==========
    # Choose your LLM (larger is better with GPU)
    LLM_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.2"  # Recommended for GPU
    # Alternatives:
    # "meta-llama/Llama-3-8B-Instruct"  # Excellent quality
    # "microsoft/phi-2"  # Faster but less capable
    
    # Model settings - FIXED FOR COMPLETE ANSWERS
    USE_4BIT: bool = True  # CHANGED: Enable for 50% speed boost
    MAX_NEW_TOKENS: int = 500  # FIXED: Increased from 250 to prevent cutoffs
    TEMPERATURE: float = 0.3  # FIXED: Increased from 0.1 for better completion
    
    # ========== EMBEDDING CONFIGURATION ==========
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"  # Best quality
    # Alternative: "sentence-transformers/all-MiniLM-L6-v2"  # Faster
    
    # ========== RERANKER CONFIGURATION ==========
    USE_RERANKER: bool = True  # CRITICAL for accuracy
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    RERANK_TOP_K: int = 10  # OPTIMIZED: Reduced from 20 for speed
    FINAL_TOP_K: int = 5  # Return top 5 after reranking
    
    # ========== RETRIEVAL CONFIGURATION ==========
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    SIMILARITY_THRESHOLD: float = 0.5  # Minimum similarity score
    
    # ========== DATA PATHS ==========
    DATA_DIR: str = "./acn_data"
    CHROMA_DIR: str = "./acn_data/chroma_db"
    CACHE_DIR: str = "./acn_data/cache"
    LOGS_DIR: str = "./acn_data/logs"
    
    # ========== QUERY SETTINGS ==========
    ENABLE_QUERY_EXPANSION: bool = True  # Generate query variations
    ENABLE_TEMPORAL_FILTERING: bool = True  # For event queries
    ENABLE_RESPONSE_VALIDATION: bool = True  # Prevent hallucinations
    
    # ========== PERFORMANCE ==========
    BATCH_SIZE: int = 32  # For embedding generation
    USE_GPU: bool = True  # Auto-detect available
    
    def __post_init__(self):
        """Create necessary directories"""
        from pathlib import Path
        Path(self.DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.CHROMA_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.CACHE_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.LOGS_DIR).mkdir(parents=True, exist_ok=True)