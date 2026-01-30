#!/usr/bin/env python3
"""
Interactive Test Script for ACN RAG System
Allows you to run your own queries and see detailed results
"""

import sys
from datetime import datetime
from pathlib import Path

# Local imports
from query import ACNRAGEngine, QueryClassifier
from config import RAGConfig


def print_separator(char="=", length=80):
    """Print a separator line"""
    print(char * length)


def print_section(title):
    """Print a section header"""
    print()
    print_separator()
    print(f"  {title}")
    print_separator()
    print()


def display_result(result: dict, query: str):
    """Display query result in a formatted way"""
    
    print_section("QUERY RESULT")
    
    print(f"📝 Query: {query}")
    print(f"🏷️  Category: {result['intent']}")
    print(f"📊 Documents Retrieved: {result['num_docs']}")
    print(f"⏱️  Processing Time: {result['processing_time']:.2f}s")
    print(f"🎯 Confidence: {result['confidence']:.2f}")
    
    print("\n" + "─" * 80)
    print("💬 ANSWER:")
    print("─" * 80)
    print()
    print(result['answer'])
    print()
    
    if result['sources']:
        print("─" * 80)
        print("📚 SOURCES:")
        print("─" * 80)
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. {source}")
        print()


def test_classification():
    """Test query classification"""
    
    print_section("QUERY CLASSIFICATION TEST")
    
    classifier = QueryClassifier()
    
    test_queries = [
        "What courses does ACN offer?",
        "Tell me about ACN training",
        "Show me Epic courses",
        "ACN webinars",
        "What learning resources are available?",
        "How do I join ACN?",
        "Upcoming events",
        "What is ACN?",
        "What are membership benefits?",
        "Applied Epic tips and techniques",
    ]
    
    print("Testing query classification...\n")
    
    for query in test_queries:
        intent = classifier.classify(query)
        print(f"Query: '{query}'")
        print(f"  → Category: {intent.category}")
        print(f"  → Temporal: {intent.temporal_mode if intent.is_temporal else 'N/A'}")
        print(f"  → Confidence: {intent.confidence:.2f}")
        print()


def test_predefined_queries(engine):
    """Test with predefined queries"""
    
    print_section("PREDEFINED QUERY TESTS")
    
    queries = [
        "What courses does ACN offer?",
        "Tell me about Applied Epic training",
        "Show me learning resources for Epic",
        "What are the membership benefits?",
        "When are upcoming ACN events?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'═' * 80}")
        print(f"TEST {i}/{len(queries)}")
        print(f"{'═' * 80}\n")
        
        result = engine.query(query)
        display_result(result, query)
        
        if i < len(queries):
            input("\nPress Enter to continue to next test...")


def interactive_mode(engine):
    """Interactive query mode"""
    
    print_section("INTERACTIVE MODE")
    
    print("Enter your queries below. Type 'quit', 'exit', or 'q' to stop.")
    print("Type 'classify' to test classification without full query.")
    print("Type 'help' for more options.\n")
    
    classifier = QueryClassifier()
    
    while True:
        try:
            query = input("🔍 Your query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if query.lower() == 'help':
                print("\nAvailable commands:")
                print("  - Enter any question to get an answer")
                print("  - 'classify' - Test classification only")
                print("  - 'test' - Run predefined tests")
                print("  - 'quit'/'exit'/'q' - Exit")
                print()
                continue
            
            if query.lower() == 'classify':
                test_query = input("  Enter query to classify: ").strip()
                if test_query:
                    intent = classifier.classify(test_query)
                    print(f"\n  Category: {intent.category}")
                    print(f"  Temporal: {intent.temporal_mode if intent.is_temporal else 'N/A'}")
                    print(f"  Confidence: {intent.confidence:.2f}\n")
                continue
            
            if query.lower() == 'test':
                test_predefined_queries(engine)
                continue
            
            # Process the query
            print()
            result = engine.query(query)
            display_result(result, query)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def quick_query(engine, query):
    """Quick single query"""
    result = engine.query(query)
    display_result(result, query)


def main():
    """Main function"""
    
    print_separator("═")
    print("  ACN RAG SYSTEM - INTERACTIVE TESTER")
    print_separator("═")
    print()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="Test ACN RAG system with custom queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test.py                              # Interactive mode
  python test.py --classify                   # Test classification only
  python test.py --test                       # Run predefined tests
  python test.py --query "What courses does ACN offer?"  # Single query
  python test.py -q "Tell me about Epic training"        # Short form
        """
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Run a single query'
    )
    
    parser.add_argument(
        '--classify',
        action='store_true',
        help='Test query classification only'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run predefined test queries'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Start interactive mode (default)'
    )
    
    args = parser.parse_args()
    
    # Classification test only
    if args.classify:
        test_classification()
        return
    
    # Initialize RAG engine for query tests
    print("Initializing RAG engine...")
    print("(This may take a moment to load models)\n")
    
    try:
        config = RAGConfig()
        engine = ACNRAGEngine(config)
    except Exception as e:
        print(f"❌ Failed to initialize RAG engine: {e}")
        print("\nMake sure you have:")
        print("1. Run: python ingest.py --clear")
        print("2. ChromaDB is populated with data")
        print("3. All required models are downloaded")
        return
    
    # Single query mode
    if args.query:
        quick_query(engine, args.query)
        return
    
    # Predefined tests mode
    if args.test:
        test_predefined_queries(engine)
        return
    
    # Default: Interactive mode
    interactive_mode(engine)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)