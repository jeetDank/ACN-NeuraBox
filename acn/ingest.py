#!/usr/bin/env python3
"""
ACN Data Ingestion Script
Processes scraped JSON files and builds vector database
Handles both regular content and structured learning center data
"""

import json
import os
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import logging

# Vector database and embeddings
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Progress bar
from tqdm import tqdm

# Local config
from config import RAGConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ACNDataIngester:
    """
    Ingests scraped ACN data into ChromaDB vector database
    Handles both regular content and structured learning center data
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.raw_dir = Path(config.RAW_DIR)
        self.chroma_dir = Path(config.CHROMA_DIR)
        
        # Create directories if they don't exist
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing ACN Data Ingester...")
        logger.info(f"Raw data directory: {self.raw_dir}")
        logger.info(f"ChromaDB directory: {self.chroma_dir}")
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info(f"✓ Embedding model loaded")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection_name = "acn_knowledge_base"
        try:
            self.collection = self.client.get_collection(self.collection_name)
            logger.info(f"✓ Using existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "ACN website content"}
            )
            logger.info(f"✓ Created new collection: {self.collection_name}")
    
    def load_json_files(self) -> List[Dict]:
        """Load all JSON files from raw directory"""
        logger.info("Loading JSON files...")
        
        all_documents = []
        section_stats = {}
        
        # Get all section directories
        sections = [d for d in self.raw_dir.iterdir() if d.is_dir()]
        
        for section_dir in sections:
            section_name = section_dir.name
            
            # Skip summary files
            if section_name.startswith('_'):
                continue
            
            logger.info(f"Processing section: {section_name}")
            
            # Load all JSON files in this section
            json_files = list(section_dir.glob("*.json"))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Check if this is learning center format (has 'items' array)
                    if section_name == 'learning' and 'items' in data:
                        # Process learning center structured data
                        docs = self._process_learning_center_data(data, json_file)
                        all_documents.extend(docs)
                        section_stats[section_name] = section_stats.get(section_name, 0) + len(docs)
                    else:
                        # Regular format - single document
                        data['section'] = section_name
                        data['file_path'] = str(json_file)
                        all_documents.append(data)
                        section_stats[section_name] = section_stats.get(section_name, 0) + 1
                    
                except Exception as e:
                    logger.error(f"Error loading {json_file}: {e}")
                    continue
        
        # Log statistics
        logger.info(f"\n✓ Loaded {len(all_documents)} documents from {len(section_stats)} sections:")
        for section, count in sorted(section_stats.items()):
            logger.info(f"  • {section}: {count} documents")
        
        return all_documents
    
    def _process_learning_center_data(self, data: Dict, json_file: Path) -> List[Dict]:
        """
        Process learning center structured data into individual documents
        Each item in the 'items' array becomes a separate document
        """
        documents = []
        
        source_url = data.get('source', 'https://learning.appliedclientnetwork.org')
        page_num = data.get('page', 'unknown')
        scraped_at = data.get('scraped_at', '')
        
        items = data.get('items', [])
        
        for idx, item in enumerate(items):
            # Create a document for each learning item
            item_type = item.get('type', 'learning_resource')
            title = item.get('title', 'Untitled')
            components = item.get('components', '')
            recorded_on = item.get('recorded_on', '')
            description = item.get('description', '')
            
            # Build content from available fields
            content_parts = []
            
            # Add title
            content_parts.append(f"Title: {title}")
            
            # Add type
            content_parts.append(f"Type: {item_type.replace('_', ' ').title()}")
            
            # Add components if available
            if components:
                content_parts.append(f"Components: {components}")
            
            # Add recorded date if available
            if recorded_on:
                content_parts.append(f"Recorded on: {recorded_on}")
            
            # Add description if available
            if description:
                content_parts.append(f"\nDescription: {description}")
            
            content = "\n".join(content_parts)
            
            # Create document
            doc = {
                'url': source_url,
                'title': title,
                'section': 'learning',
                'content': content,
                'scraped_at': scraped_at,
                'file_path': str(json_file),
                'item_type': item_type,
                'page': page_num,
                'item_index': idx
            }
            
            # Add optional fields
            if components:
                doc['components'] = components
            if recorded_on:
                doc['recorded_on'] = recorded_on
            
            documents.append(doc)
        
        return documents
    
    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        if not text or len(text) < chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to find a sentence boundary
            if end < len(text):
                # Look for period, question mark, or exclamation within last 100 chars
                for i in range(end, max(start + chunk_size - 100, start), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def process_documents(self, documents: List[Dict]) -> tuple:
        """Process documents into chunks with metadata"""
        logger.info("Processing documents into chunks...")
        
        all_chunks = []
        all_metadatas = []
        all_ids = []
        section_chunk_stats = {}
        
        chunk_id = 0
        
        for doc in tqdm(documents, desc="Chunking documents"):
            # Extract content
            content = doc.get('content', '')
            
            if not content or len(content.strip()) < 50:
                continue  # Skip empty or very short content
            
            # Clean content
            content = content.strip()
            
            # Split into chunks
            chunks = self.chunk_text(
                content,
                chunk_size=self.config.CHUNK_SIZE,
                overlap=self.config.CHUNK_OVERLAP
            )
            
            section = doc.get('section', 'general')
            section_chunk_stats[section] = section_chunk_stats.get(section, 0) + len(chunks)
            
            # Create metadata for each chunk
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                
                metadata = {
                    'source': doc.get('url', 'unknown'),
                    'title': doc.get('title', 'Untitled'),
                    'section': section,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'scraped_at': doc.get('scraped_at', ''),
                }
                
                # Add learning-specific metadata
                if section == 'learning':
                    if 'item_type' in doc:
                        metadata['item_type'] = doc['item_type']
                    if 'recorded_on' in doc:
                        metadata['recorded_on'] = doc['recorded_on']
                    if 'components' in doc:
                        metadata['components'] = doc['components']
                
                # Add discovered_via if present
                if 'discovered_via' in doc:
                    metadata['discovered_via'] = doc['discovered_via']
                
                all_metadatas.append(metadata)
                all_ids.append(f"doc_{chunk_id}")
                
                chunk_id += 1
        
        logger.info(f"\n✓ Created {len(all_chunks)} chunks from {len(documents)} documents")
        logger.info(f"Chunks per section:")
        for section, count in sorted(section_chunk_stats.items()):
            logger.info(f"  • {section}: {count} chunks")
        
        return all_chunks, all_metadatas, all_ids
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for texts"""
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            embeddings.extend(batch_embeddings.tolist())
        
        logger.info(f"✓ Generated {len(embeddings)} embeddings")
        
        return embeddings
    
    def store_in_chromadb(self, chunks: List[str], metadatas: List[Dict], ids: List[str], embeddings: List[List[float]]):
        """Store chunks in ChromaDB"""
        logger.info("Storing data in ChromaDB...")
        
        # ChromaDB has a batch size limit
        batch_size = 5000
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Storing in ChromaDB"):
            batch_chunks = chunks[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            self.collection.add(
                documents=batch_chunks,
                metadatas=batch_metadatas,
                ids=batch_ids,
                embeddings=batch_embeddings
            )
        
        logger.info(f"✓ Stored {len(chunks)} chunks in ChromaDB")
    
    def verify_ingestion(self):
        """Verify data was ingested correctly"""
        logger.info("Verifying ingestion...")
        
        count = self.collection.count()
        logger.info(f"✓ Collection contains {count} documents")
        
        # Test query - use our embedding model, not ChromaDB's default
        test_query = "What is ACN?"
        query_embedding = self.embedding_model.encode([test_query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=3
        )
        
        if results['documents']:
            logger.info(f"✓ Test query successful")
            logger.info(f"Sample result: {results['documents'][0][0][:100]}...")
        else:
            logger.warning("⚠ Test query returned no results")
        
        # Verify learning section
        learning_results = self.collection.get(
            where={"section": "learning"},
            limit=5
        )
        
        if learning_results['ids']:
            logger.info(f"✓ Learning section verified: {len(learning_results['ids'])} samples found")
        else:
            logger.warning("⚠ No learning documents found")
    
    def run(self, clear_existing: bool = False):
        """Run complete ingestion pipeline"""
        logger.info("=" * 80)
        logger.info("STARTING ACN DATA INGESTION")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Clear existing data if requested
        if clear_existing:
            logger.info("Clearing existing data...")
            try:
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "ACN website content"}
                )
                logger.info("✓ Existing data cleared")
            except Exception as e:
                logger.warning(f"Could not clear existing data: {e}")
        
        # Step 1: Load JSON files
        documents = self.load_json_files()
        
        if not documents:
            logger.error("❌ No documents found! Please run scraper first.")
            return False
        
        # Step 2: Process into chunks
        chunks, metadatas, ids = self.process_documents(documents)
        
        if not chunks:
            logger.error("❌ No chunks created! Check document content.")
            return False
        
        # Step 3: Generate embeddings
        embeddings = self.generate_embeddings(chunks)
        
        # Step 4: Store in ChromaDB
        self.store_in_chromadb(chunks, metadatas, ids, embeddings)
        
        # Step 5: Verify
        self.verify_ingestion()
        
        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        
        logger.info("=" * 80)
        logger.info("INGESTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Documents processed: {len(documents)}")
        logger.info(f"Chunks created: {len(chunks)}")
        logger.info(f"Embeddings generated: {len(embeddings)}")
        logger.info(f"Time elapsed: {elapsed:.2f}s")
        logger.info(f"Vector database: {self.chroma_dir}")
        logger.info("=" * 80)
        
        return True


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest ACN scraped data into vector database")
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear existing data before ingesting'
    )
    
    args = parser.parse_args()
    
    # Initialize and run
    config = RAGConfig()
    ingester = ACNDataIngester(config)
    
    success = ingester.run(clear_existing=args.clear)
    
    if success:
        print("\n✓ Data ingestion successful!")
        print(f"\nNext steps:")
        print(f"1. Test queries: python query.py")
        print(f"2. Run diagnostics: python diagnose.py")
        print(f"3. Start API: python api.py")
    else:
        print("\n❌ Data ingestion failed!")
        print("\nTroubleshooting:")
        print("1. Check if scraped data exists: ls acn_data/raw/")
        print("2. Check logs above for errors")
        print("3. Try clearing existing data: python ingest.py --clear")


if __name__ == "__main__":
    main()