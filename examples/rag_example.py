"""
Comprehensive example showcasing RAG (Retrieval-Augmented Generation) capabilities.
"""

import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.settings import settings
from src.chains.rag_chain import RAGChain
from src.langsmith.setup import langsmith_setup


def main():
    """Main function to run the RAG example."""
    print("📚 RAG Example - Document-Based Question Answering")
    print("=" * 70)

    try:
        # Validate API keys
        validation = settings.validate_api_keys()
        if not validation["valid"]:
            print(f"❌ Missing API keys: {', '.join(validation['missing_keys'])}")
            print("Please set OPENAI_API_KEY in your .env file.")
            return

        # Initialize RAG chain
        print("🔄 Initializing RAG chain...")
        rag_chain = RAGChain()

        # Load sample documents
        rag_chain.load_sample_documents()

        # Interactive Q&A session
        print("\n🎯 Interactive Q&A Session")
        print("Ask questions about LangChain, LangGraph, and LangSmith!")
        print("Type 'quit' to exit.\n")

        while True:
            question = input("❓ Your question: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("👋 Thanks for using the RAG example!")
                break

            if not question:
                continue

            # Get answer from RAG
            result = rag_chain.ask_question(question)

            print(f"\n🤖 Answer: {result['answer']}")

            if result["sources"]:
                print("\n📖 Sources:")
                for i, source in enumerate(result["sources"], 1):
                    print(f"   {i}. {source['metadata'].get('source', 'Unknown')}")

            print("-" * 50)

        # Show LangSmith status
        print(f"\n📊 LangSmith Tracing: {'✅ Enabled' if langsmith_setup.is_enabled() else '❌ Disabled'}")

    except Exception as e:
        print(f"❌ RAG example failed: {e}")


if __name__ == "__main__":
    main()
