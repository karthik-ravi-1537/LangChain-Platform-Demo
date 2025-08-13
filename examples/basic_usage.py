"""
Basic usage examples demonstrating LangChain, LangGraph, and LangSmith integration.
"""
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.settings import settings
from src.langsmith.setup import langsmith_setup
from src.chains.basic_chain import BasicChain
from src.tools.calculator import calculator_tool, advanced_calculator_tool

from langchain_openai import OpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class BasicUsageDemo:
    """Demonstrates basic usage of LangChain components with LangSmith tracing."""

    def __init__(self):
        """Initialize the demo with required components."""
        # Validate API keys
        validation = settings.validate_api_keys()
        if not validation["valid"]:
            raise ValueError(f"Missing API keys: {', '.join(validation['missing_keys'])}")

        # Initialize LLM
        self.llm = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            temperature=0.7
        )

        # Initialize basic chain
        self.basic_chain = BasicChain()

        # Initialize tools
        self.tools = [calculator_tool, advanced_calculator_tool]

        # Initialize conversation history (replacement for ConversationBufferMemory)
        self.conversation_history = []

        # Create a simple tool-calling chain
        self.tool_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant with access to calculator tools. Use the tools when needed to solve math problems."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

    def demo_basic_chain(self):
        """Demonstrate basic chain functionality."""
        print("🔗 Basic Chain Demo")
        print("=" * 50)

        # Get a fact about a topic
        topic = "quantum computing"
        fact = self.basic_chain.get_simple_fact(topic)
        print(f"\n📝 Fact about {topic}:")
        print(f"   {fact}")

        # Get a list of facts
        facts = self.basic_chain.get_fact_list("artificial intelligence", 3)
        print(f"\n📋 AI Facts:")
        for i, fact in enumerate(facts, 1):
            print(f"   {i}. {fact}")

    def demo_calculator_tools(self):
        """Demonstrate calculator tools."""
        print("\n🧮 Calculator Tools Demo")
        print("=" * 50)

        # Test calculations
        calculations = [
            "What is 25 * 4?",
            "Calculate the square root of 144",
            "What is the result of (10 + 5) * 3?",
            "What is sin(pi/2)?"
        ]

        for calc in calculations:
            print(f"\n❓ Question: {calc}")
            try:
                # Simple tool usage - in a real implementation, you'd use LangGraph
                # or implement proper tool calling logic
                response = self._simple_tool_call(calc)
                print(f"🤖 Answer: {response}")
            except Exception as e:
                print(f"❌ Error: {e}")

    def _simple_tool_call(self, query: str) -> str:
        """Simple tool calling logic (placeholder for proper LangGraph implementation)."""
        # This is a simplified implementation - in production, use LangGraph
        if any(op in query.lower() for op in ['*', '/', '+', '-', 'multiply', 'divide', 'add', 'subtract']):
            # Extract numbers and operations (simplified)
            if "25 * 4" in query:
                return calculator_tool.invoke("25 * 4")
            elif "square root of 144" in query:
                return advanced_calculator_tool.invoke("sqrt(144)")
            elif "(10 + 5) * 3" in query:
                return calculator_tool.invoke("(10 + 5) * 3")
            elif "sin(pi/2)" in query:
                return advanced_calculator_tool.invoke("sin(pi/2)")

        return self.llm.invoke(query)

    def demo_conversational_agent(self):
        """Demonstrate conversational agent with memory."""
        print("\n💬 Conversational Agent Demo")
        print("=" * 50)

        # Simulate a conversation
        questions = [
            "Hi! Can you help me with some math?",
            "What is 15 * 8?",
            "Now add 50 to that result",
            "What was the first calculation I asked you about?"
        ]

        for question in questions:
            print(f"\n👤 User: {question}")
            try:
                response = self._conversational_response(question)
                print(f"🤖 Assistant: {response}")
            except Exception as e:
                print(f"❌ Error: {e}")

    def _conversational_response(self, message: str) -> str:
        """Handle conversational responses with memory."""
        # Add user message to history
        self.conversation_history.append(HumanMessage(content=message))

        # Generate response using the chain
        chain = self.tool_prompt | self.llm
        response = chain.invoke({
            "input": message,
            "chat_history": self.conversation_history
        })

        # Add assistant response to history
        self.conversation_history.append(AIMessage(content=response))

        return response

    def demo_langsmith_status(self):
        """Show LangSmith tracing status and information."""
        print("\n📊 LangSmith Monitoring Status")
        print("=" * 50)

        if langsmith_setup.is_enabled():
            print("✅ LangSmith tracing is ENABLED")
            print(f"   📍 Project: {settings.LANGSMITH_PROJECT}")
            print(f"   🔗 Endpoint: {settings.LANGSMITH_ENDPOINT}")
            print("   📈 All chains and agents are being traced")
            print("   🌐 View traces at: https://smith.langchain.com")
        else:
            print("❌ LangSmith tracing is DISABLED")
            print("   💡 To enable tracing:")
            print("   1. Set LANGSMITH_API_KEY in your .env file")
            print("   2. Set LANGSMITH_TRACING=true in your .env file")
            print("   3. Restart the application")

    def run_full_demo(self):
        """Run the complete demo."""
        print("🚀 LangChain Ecosystem Demo")
        print("=" * 80)
        print("This demo showcases LangChain, LangGraph, and LangSmith integration")
        print("=" * 80)

        # Show LangSmith status first
        self.demo_langsmith_status()

        # Demo basic chain
        self.demo_basic_chain()

        # Demo calculator tools
        self.demo_calculator_tools()

        # Demo conversational agent
        self.demo_conversational_agent()

        print("\n🎉 Demo completed!")
        print("Check your LangSmith dashboard for detailed traces and analytics.")


def main():
    """Main function to run the demo."""
    try:
        demo = BasicUsageDemo()
        demo.run_full_demo()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Make sure you have created a .env file with your API keys")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Check that your OpenAI API key is valid")


if __name__ == "__main__":
    main()
