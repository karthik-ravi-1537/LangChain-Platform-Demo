"""
RAG (Retrieval-Augmented Generation) chain for document-based question answering.
"""
import os
import sys
from typing import List, Dict, Optional

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.settings import settings
from src.langsmith.setup import langsmith_setup


class RAGChain:
    """RAG chain for document-based question answering."""

    def __init__(self, documents_path: Optional[str] = None):
        """Initialize the RAG chain."""
        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API key required for RAG chain")

        self.llm = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            temperature=0.3
        )

        self.embeddings = OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        self.vectorstore = None
        self.qa_chain = None

        # Load documents if path provided
        if documents_path:
            self.load_documents(documents_path)

    def create_sample_documents(self) -> List[Document]:
        """Create sample documents for demonstration."""
        sample_docs = [
            Document(
                page_content="""
                LangChain is a framework for developing applications powered by language models. 
                It provides tools for connecting LLMs to data sources, building chains of operations,
                and creating agents that can use tools. Key features include:
                - Chain composition for complex workflows
                - Memory management for conversation context
                - Tool integration for external data access
                - Agent frameworks for autonomous decision making
                """,
                metadata={"source": "langchain_intro", "topic": "framework"}
            ),
            Document(
                page_content="""
                LangGraph is a library for building stateful, multi-actor applications with LLMs.
                It extends LangChain with graph-based workflows that can handle complex state management.
                Key capabilities include:
                - State management across workflow steps
                - Conditional routing and branching
                - Parallel execution of tasks
                - Error handling and recovery mechanisms
                - Human-in-the-loop interactions
                """,
                metadata={"source": "langgraph_intro", "topic": "workflow"}
            ),
            Document(
                page_content="""
                LangSmith is a platform for debugging, testing, and monitoring LLM applications.
                It provides observability into LangChain applications with features like:
                - Trace visualization for debugging
                - Performance monitoring and analytics
                - Dataset management for testing
                - Evaluation metrics and feedback collection
                - Collaboration tools for team development
                """,
                metadata={"source": "langsmith_intro", "topic": "monitoring"}
            ),
            Document(
                page_content="""
                Vector databases are essential for RAG (Retrieval-Augmented Generation) systems.
                They store document embeddings and enable similarity search. Popular options include:
                - FAISS: Facebook's library for efficient similarity search
                - Pinecone: Managed vector database service
                - Weaviate: Open-source vector database
                - Chroma: Lightweight embedding database
                Vector databases enable finding relevant documents based on semantic similarity.
                """,
                metadata={"source": "vector_db_guide", "topic": "storage"}
            ),
            Document(
                page_content="""
                Prompt engineering is crucial for effective LLM applications. Best practices include:
                - Clear and specific instructions
                - Providing examples (few-shot learning)
                - Breaking complex tasks into steps
                - Using consistent formatting
                - Testing with diverse inputs
                - Iterative refinement based on results
                Good prompts lead to more reliable and useful AI responses.
                """,
                metadata={"source": "prompt_engineering", "topic": "optimization"}
            )
        ]
        return sample_docs

    def load_documents(self, documents_path: str):
        """Load documents from a directory or file."""
        try:
            if os.path.isdir(documents_path):
                # Load from directory
                loader = DirectoryLoader(
                    documents_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                documents = loader.load()
            elif os.path.isfile(documents_path):
                # Load single file
                loader = TextLoader(documents_path)
                documents = loader.load()
            else:
                print(f"⚠️  Path not found: {documents_path}. Using sample documents.")
                documents = self.create_sample_documents()

            if not documents:
                print("ℹ️  No documents found. Using sample documents.")
                documents = self.create_sample_documents()

            self._create_vectorstore(documents)

        except Exception as e:
            print(f"❌ Error loading documents: {e}")
            print("ℹ️  Using sample documents instead.")
            documents = self.create_sample_documents()
            self._create_vectorstore(documents)

    def load_sample_documents(self):
        """Load sample documents for demonstration."""
        print("📚 Loading sample documents...")
        documents = self.create_sample_documents()
        self._create_vectorstore(documents)

    def _create_vectorstore(self, documents: List[Document]):
        """Create vector store from documents."""
        print(f"🔄 Processing {len(documents)} documents...")

        # Split documents
        splits = self.text_splitter.split_documents(documents)
        print(f"📄 Created {len(splits)} document chunks")

        # Create vector store
        self.vectorstore = FAISS.from_documents(
            splits,
            self.embeddings
        )

        # Create QA chain
        self._create_qa_chain()
        print("✅ RAG chain ready!")

    def _create_qa_chain(self):
        """Create the QA chain with custom prompt."""
        custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are an AI assistant that answers questions based on the provided context.
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer based on the context, say "I don't have enough information in the provided context to answer this question."
            
            Context:
            {context}
            
            Question: {question}
            
            Answer: Let me help you with that based on the information provided.
            """
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": custom_prompt},
            return_source_documents=True,
            verbose=True
        )

    def ask_question(self, question: str) -> Dict[str, str]:
        """Ask a question using the RAG chain."""
        if not self.qa_chain:
            return {
                "answer": "❌ RAG chain not initialized. Please load documents first.",
                "sources": []
            }

        print(f"❓ Question: {question}")

        try:
            result = self.qa_chain.invoke({"query": question})

            # Extract source information
            sources = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    source_info = {
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    sources.append(source_info)

            return {
                "answer": result["result"],
                "sources": sources
            }

        except Exception as e:
            return {
                "answer": f"❌ Error processing question: {str(e)}",
                "sources": []
            }

    def similarity_search(self, query: str, k: int = 3) -> List[Dict]:
        """Perform similarity search on the vector store."""
        if not self.vectorstore:
            return []

        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            results = []

            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })

            return results

        except Exception as e:
            print(f"❌ Similarity search error: {e}")
            return []

    def demonstrate_rag(self):
        """Demonstrate RAG functionality."""
        print("📚 RAG Chain Demo")
        print("=" * 60)

        # Load sample documents
        self.load_sample_documents()

        # Test questions
        test_questions = [
            "What is LangChain and what are its key features?",
            "How does LangGraph differ from LangChain?",
            "What monitoring capabilities does LangSmith provide?",
            "What are the benefits of using vector databases?",
            "What are some best practices for prompt engineering?",
            "How do I integrate multiple AI tools together?"
        ]

        for i, question in enumerate(test_questions, 1):
            print(f"\n--- Question {i} ---")
            result = self.ask_question(question)

            print(f"🤖 Answer: {result['answer']}")

            if result['sources']:
                print(f"\n📖 Sources ({len(result['sources'])}):")
                for j, source in enumerate(result['sources'], 1):
                    print(f"   {j}. {source['metadata'].get('source', 'Unknown')}")
                    print(f"      Topic: {source['metadata'].get('topic', 'N/A')}")

            print("-" * 40)

        # Show similarity search
        print(f"\n🔍 Similarity Search Demo")
        similar_docs = self.similarity_search("LangChain tools and agents", k=2)
        for i, doc in enumerate(similar_docs, 1):
            print(f"{i}. {doc['content'][:150]}...")

        # Show LangSmith status
        print(f"\n📊 LangSmith Tracing: {'✅ Enabled' if langsmith_setup.is_enabled() else '❌ Disabled'}")


def main():
    """Main function to run the RAG demo."""
    try:
        # Validate API keys
        validation = settings.validate_api_keys()
        if not validation["valid"]:
            print(f"❌ Missing API keys: {', '.join(validation['missing_keys'])}")
            print("Please set the required API keys in your .env file.")
            return

        # Create and run the RAG demo
        rag_chain = RAGChain()
        rag_chain.demonstrate_rag()

    except Exception as e:
        print(f"❌ RAG demo failed: {e}")
        print("\n💡 Make sure you have:")
        print("1. Set OPENAI_API_KEY in your .env file")
        print("2. Installed all dependencies: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
