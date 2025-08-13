"""
Agent workflow example demonstrating multi-agent coordination and LangGraph workflows.
"""
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.settings import settings
from src.langsmith.setup import langsmith_setup
from src.graphs.workflow_graph import MathWorkflowGraph
from src.graphs.agent_graph import MultiAgentSystem


def main():
    """Main function to run the agent workflow example."""
    print("🤖 Agent Workflow Example")
    print("=" * 70)

    try:
        # Validate API keys
        validation = settings.validate_api_keys()
        if not validation["valid"]:
            print(f"❌ Missing API keys: {', '.join(validation['missing_keys'])}")
            print("Please set OPENAI_API_KEY in your .env file.")
            return

        print("🔄 Initializing agent systems...")

        # Initialize workflow graph
        print("\n1️⃣ Math Workflow Graph Demo")
        print("-" * 40)
        math_workflow = MathWorkflowGraph()

        # Demo workflow with a few examples
        workflow_queries = [
            "What is 15 * 8 + 12?",
            "Calculate sqrt(64) + sin(pi/2)"
        ]

        for query in workflow_queries:
            print(f"\n🔍 Processing: {query}")
            result = math_workflow.process_query(query)
            print(f"✅ Result: {result}")

        print("\n2️⃣ Multi-Agent System Demo")
        print("-" * 40)
        multi_agent = MultiAgentSystem()

        # Demo multi-agent with complex queries
        agent_queries = [
            "Calculate 25 squared and find recent AI news",
            "What is LangGraph and how does it work?"
        ]

        for query in agent_queries:
            print(f"\n🤖 Multi-Agent Processing: {query}")
            result = multi_agent.process_query(query)
            print(f"📋 Final Response: {result}")

        print("\n🎯 Interactive Mode")
        print("Try your own queries! Type 'quit' to exit.")

        while True:
            user_query = input("\n❓ Your query: ").strip()

            if user_query.lower() in ['quit', 'exit', 'q']:
                print("👋 Thanks for using the agent workflow example!")
                break

            if not user_query:
                continue

            # Choose system based on query complexity
            if any(word in user_query.lower() for word in ['calculate', 'math', 'sqrt', 'sin', 'cos']):
                print("🧮 Using Math Workflow...")
                result = math_workflow.process_query(user_query)
            else:
                print("🤖 Using Multi-Agent System...")
                result = multi_agent.process_query(user_query)

            print(f"✅ Response: {result}")
            print("-" * 50)

        # Show LangSmith status
        print(f"\n📊 LangSmith Tracing: {'✅ Enabled' if langsmith_setup.is_enabled() else '❌ Disabled'}")

    except Exception as e:
        print(f"❌ Agent workflow example failed: {e}")
        print("\n💡 Make sure you have:")
        print("1. Set OPENAI_API_KEY in your .env file")
        print("2. Installed all dependencies: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
