#!/usr/bin/env python3
"""
Quick test script to verify your ACN database
Tests common queries to ensure data quality
"""

try:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
CHROMA_DIR = "./acn_data/chroma_db"

# Test queries from your requirements
TEST_QUERIES = [
    "What is the Applied Client Network?",
    "What are the membership benefits?",
    "When is the next ACN Summit?",
    "Tell me about the Dallas Summit",
    "What is the Calgary Summit schedule?",
    "How do I join ACN?",
    "What is Applied Net?",
    "Who is Brian Langerman?",
    "What are AMA sessions?",
    "What events does ACN offer?",
]

def test_database():
    """Test the database with sample queries"""
    print("="*80)
    print("ACN DATABASE QUALITY TEST")
    print("="*80)
    
    # Load database
    print("\nLoading database...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    print("✓ Database loaded")
    
    # Test queries
    print("\n" + "="*80)
    print("Testing Sample Queries")
    print("="*80 + "\n")
    
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[Query {i}/{len(TEST_QUERIES)}] {query}")
        print("-" * 80)
        
        results = db.similarity_search(query, k=3)
        
        if results:
            print(f"✓ Found {len(results)} results\n")
            
            for j, doc in enumerate(results, 1):
                print(f"Result {j}:")
                print(f"  Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"  Title: {doc.metadata.get('title', 'No title')}")
                
                # Show snippet (check for quality)
                content = doc.page_content[:300].replace('\n', ' ')
                print(f"  Content: {content}...")
                
                # Quality checks
                issues = []
                if 'As' in content and 'ACN' not in content:
                    issues.append("Possible 'ACN' corruption to 'As'")
                if 'hanson' in content.lower():
                    issues.append("'hands-on' corrupted to 'hanson'")
                if len(re.findall(r'[A-Z]{10,}', content)) > 2:
                    issues.append("Too many all-caps sequences")
                
                if issues:
                    print(f"  ⚠ Quality Issues: {', '.join(issues)}")
                else:
                    print(f"  ✓ Quality check passed")
                print()
        else:
            print("✗ No results found!")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nIf you see quality issues above, re-run the scraper.")
    print("If results look good, your database is ready to use!")

if __name__ == "__main__":
    import re
    test_database()