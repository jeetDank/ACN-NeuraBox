"""
Configuration for ACN RAG System
Production-ready configuration with optimal settings
"""

from dataclasses import dataclass
from typing import List
from pathlib import Path

@dataclass
class RAGConfig:
    """Complete configuration for the RAG system"""
    
    # ========== MODEL CONFIGURATION ==========
    # LLM settings - optimized for quality and speed
    LLM_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.2"
    USE_4BIT: bool = True  # Enable 4-bit quantization for speed
    MAX_NEW_TOKENS: int = 512  # Increased for complete answers
    TEMPERATURE: float = 0.2
    TOP_P: float = 0.95
    TOP_K: int = 50
    
    # ========== EMBEDDING CONFIGURATION ==========
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    
    USE_LOCAL_LLM: bool = True 
    # ========== RERANKER CONFIGURATION ==========
    USE_RERANKER: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    RERANK_TOP_K: int = 10
    FINAL_TOP_K: int = 5
    
    # ========== RETRIEVAL CONFIGURATION ==========
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    SIMILARITY_THRESHOLD: float = 0.5
    
    # ========== DATA PATHS ==========
    DATA_DIR: str = "./acn_data"
    RAW_DIR: str = "./acn_data/raw"
    CHROMA_DIR: str = "./acn_data/chroma_db"
    CACHE_DIR: str = "./acn_data/cache"
    LOGS_DIR: str = "./acn_data/logs"
    
    # ========== QUERY SETTINGS ==========
    ENABLE_QUERY_EXPANSION: bool = True
    ENABLE_TEMPORAL_FILTERING: bool = True
    ENABLE_RESPONSE_VALIDATION: bool = True
    
    # ========== PERFORMANCE ==========
    BATCH_SIZE: int = 32
    USE_GPU: bool = True
    
    # ========== SCRAPING CONFIGURATION ==========
    BASE_URL: str = "https://www.appliedclientnetwork.org"
    MAX_PAGES: int = 100
    CRAWL_DELAY: float = 1.0  # Delay between requests
    TIMEOUT: int = 30
    
    # ========== LEARNING CENTER SCRAPING ==========
    LEARNING_BASE_URL: str = "https://learning.appliedclientnetwork.org"
    LEARNING_MAX_PAGES: int = 130
    
    # Pages to scrape (relative URLs)
    SEED_URLS: List[str] = None
    
    def __post_init__(self):
        """Initialize directories and default URLs"""
        # Create directories
        for dir_path in [self.DATA_DIR, self.RAW_DIR, self.CHROMA_DIR, 
                         self.CACHE_DIR, self.LOGS_DIR]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Default seed URLs if not provided
        if self.SEED_URLS is None:
            self.SEED_URLS = [
                "/",
                "/About",
                "/Membership",
                "/Membership/Why-ACN",
                "/Membership/Join-Renew",
                "/Membership/Access-Benefits",
                "/Events",
                "/Events/Summits",
                "/Events/Event-Calendar",
                "/Community",
                "/Community/Product-Forums",
                "/Community/Member-Alliance-Communities",
                "/Resources",
                "/Resources/Epic-Member-Hub",
                "/Resources/EZLynx-Member-Hub",
                "/Get-Involved",
                "/Get-Involved/Volunteer",
            ]