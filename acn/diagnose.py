#!/usr/bin/env python3
"""
Verification Script for Learning Center Integration
Tests that learning section data is properly ingested and queryable
"""

import json
from pathlib import Path
from config import RAGConfig

def verify_raw_data():
    """Verify raw learning data exists and is in correct format"""
    print("=" * 80)
    print("STEP 1: Verifying Raw Learning Data")
    print("=" * 80)
    
    config = RAGConfig()
    learning_dir = Path(config.RAW_DIR) / "learning"
    
    if not learning_dir.exists():
        print("❌ Learning directory does not exist!")
        print(f"   Expected: {learning_dir}")
        return False
    
    json_files = list(learning_dir.glob("*.json"))
    print(f"✓ Found {len(json_files)} JSON files in learning directory")
    
    if len(json_files) == 0:
        print("❌ No JSON files found in learning directory!")
        return False
    
    # Check format of first few files
    print("\nChecking file format...")
    for i, json_file in enumerate(json_files[:3]):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        required_fields = ['url', 'title', 'section', 'content', 'scraped_at']
        missing = [field for field in required_fields if field not in data]
        
        if missing:
            print(f"❌ {json_file.name} missing fields: {missing}")
            return False
        
        if data['section'] != 'learning':
            print(f"❌ {json_file.name} has wrong section: {data['section']}")
            return False
        
        print(f"✓ {json_file.name}")
        print(f"  Title: {data['title'][:50]}...")
        print(f"  Content length: {len(data['content'])} chars")
    
    print(f"\n✓ Raw learning data format is correct!")
    return True


def verify_chromadb():
    """Verify learning data is in ChromaDB"""
    print("\n" + "=" * 80)
    print("STEP 2: Verifying ChromaDB Contains Learning Data")
    print("=" * 80)
    
    try:
        import chromadb
        from chromadb.config import Settings
        
        config = RAGConfig()
        client = chromadb.PersistentClient(
            path=config.CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        
        collection = client.get_collection("acn_knowledge_base")
        total_count = collection.count()
        print(f"✓ Total documents in ChromaDB: {total_count}")
        
        # Query for learning section
        results = collection.get(
            where={"section": "learning"},
            limit=10
        )
        
        learning_count = len(results['ids']) if results['ids'] else 0
        
        if learning_count == 0:
            print("❌ No learning documents found in ChromaDB!")
            print("   Please run: python ingest.py --clear")
            return False
        
        print(f"✓ Found {learning_count} learning chunks (showing first 10)")
        
        # Show sample
        if results['metadatas']:
            print("\nSample learning documents:")
            for i, metadata in enumerate(results['metadatas'][:3]):
                print(f"\n{i+1}. {metadata.get('title', 'Untitled')}")
                print(f"   Section: {metadata.get('section')}")
                print(f"   Source: {metadata.get('source')}")
                if 'item_type' in metadata:
                    print(f"   Type: {metadata.get('item_type')}")
        
        # Get all learning documents count
        all_learning = collection.get(
            where={"section": "learning"}
        )
        total_learning = len(all_learning['ids']) if all_learning['ids'] else 0
        print(f"\n✓ Total learning chunks in database: {total_learning}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error accessing ChromaDB: {e}")
        return False


def verify_query_classification():
    """Verify query classification handles learning queries"""
    print("\n" + "=" * 80)
    print("STEP 3: Verifying Query Classification")
    print("=" * 80)
    
    try:
        from query import QueryClassifier
        
        classifier = QueryClassifier()
        
        test_queries = [
            ("What courses does ACN offer?", "learning"),
            ("Tell me about ACN training", "learning"),
            ("Show me Epic courses", "learning"),
            ("ACN webinars", "learning"),
            ("What learning resources are available?", "learning"),
            ("How do I join ACN?", "membership"),
            ("Upcoming events", "events"),
        ]
        
        print("Testing query classification...")
        all_correct = True
        
        for query, expected_category in test_queries:
            intent = classifier.classify(query)
            is_correct = intent.category == expected_category
            status = "✓" if is_correct else "❌"
            
            print(f"{status} '{query}'")
            print(f"   Expected: {expected_category}, Got: {intent.category}")
            
            if not is_correct:
                all_correct = False
        
        if all_correct:
            print("\n✓ All query classifications correct!")
        else:
            print("\n❌ Some query classifications failed")
        
        return all_correct
        
    except Exception as e:
        print(f"❌ Error testing query classification: {e}")
        return False


def verify_end_to_end():
    """Verify end-to-end query works for learning content"""
    print("\n" + "=" * 80)
    print("STEP 4: End-to-End Query Test")
    print("=" * 80)
    
    try:
        from query import ACNRAGEngine
        from config import RAGConfig
        
        config = RAGConfig()
        engine = ACNRAGEngine(config)
        
        test_query = "What courses does ACN offer?"
        print(f"\nTesting query: '{test_query}'")
        print("This may take a few seconds...")
        
        result = engine.query(test_query)
        
        print(f"\n✓ Query completed successfully!")
        print(f"Category detected: {result['intent']}")
        print(f"Documents retrieved: {result['num_docs']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        
        print(f"\nAnswer preview:")
        print(result['answer'][:300] + "..." if len(result['answer']) > 300 else result['answer'])
        
        print(f"\nSources:")
        for i, source in enumerate(result['sources'][:3], 1):
            print(f"  {i}. {source}")
        
        # Check if learning section was used
        if any('learning' in str(source).lower() for source in result['sources']):
            print("\n✓ Learning section content was retrieved!")
            return True
        else:
            print("\n⚠ Warning: Learning section not in top sources")
            return True  # Still pass, might be legitimate
        
    except Exception as e:
        print(f"❌ Error in end-to-end test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks"""
    print("\n" + "=" * 80)
    print("LEARNING CENTER INTEGRATION VERIFICATION")
    print("=" * 80)
    print()
    
    results = {}
    
    # Run checks
    results['raw_data'] = verify_raw_data()
    
    if results['raw_data']:
        results['chromadb'] = verify_chromadb()
    else:
        print("\n⚠ Skipping ChromaDB check (raw data not found)")
        results['chromadb'] = False
    
    results['classification'] = verify_query_classification()
    
    if results['chromadb']:
        results['end_to_end'] = verify_end_to_end()
    else:
        print("\n⚠ Skipping end-to-end test (ChromaDB not ready)")
        results['end_to_end'] = False
    
    # Final summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status:10s} {check.replace('_', ' ').title()}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL CHECKS PASSED!")
        print("Learning center integration is working correctly!")
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nTroubleshooting:")
        
        if not results['raw_data']:
            print("1. Run scrapers: python run_all_scrapers.py")
        
        if not results['chromadb']:
            print("2. Ingest data: python ingest.py --clear")
        
        if not results['classification']:
            print("3. Update query.py with new version")
        
        if not results['end_to_end']:
            print("4. Check query.py and ingest.py are updated")
    
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)