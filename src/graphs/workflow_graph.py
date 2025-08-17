"""
LangGraph workflow examples demonstrating state management and conditional routing.
"""

import os
import sys
from typing import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.settings import settings
from src.langsmith.setup import langsmith_setup
from src.tools.calculator import advanced_calculator_tool, calculator_tool


class WorkflowState(TypedDict):
    """State for the workflow graph."""

    messages: list[BaseMessage]
    current_task: str
    results: dict[str, str]
    next_action: str
    user_input: str
    error_count: int


class MathWorkflowGraph:
    """A workflow that processes mathematical queries with validation and retry logic."""

    def __init__(self):
        """Initialize the math workflow graph."""
        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API key required for LangGraph workflows")

        self.llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0.1)

        # Create the workflow graph
        self.workflow = StateGraph(WorkflowState)
        self._build_workflow()

        # Add memory for persistence
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)

    def _build_workflow(self):
        """Build the workflow graph with nodes and edges."""
        # Add nodes
        self.workflow.add_node("analyze_query", self._analyze_query)
        self.workflow.add_node("simple_calculation", self._simple_calculation)
        self.workflow.add_node("advanced_calculation", self._advanced_calculation)
        self.workflow.add_node("validate_result", self._validate_result)
        self.workflow.add_node("format_response", self._format_response)
        self.workflow.add_node("handle_error", self._handle_error)

        # Set entry point
        self.workflow.set_entry_point("analyze_query")

        # Add conditional edges
        self.workflow.add_conditional_edges(
            "analyze_query",
            self._route_calculation,
            {"simple": "simple_calculation", "advanced": "advanced_calculation", "error": "handle_error"},
        )

        self.workflow.add_edge("simple_calculation", "validate_result")
        self.workflow.add_edge("advanced_calculation", "validate_result")

        self.workflow.add_conditional_edges(
            "validate_result",
            self._check_validation,
            {"success": "format_response", "retry": "analyze_query", "error": "handle_error"},
        )

        self.workflow.add_edge("format_response", END)
        self.workflow.add_edge("handle_error", END)

    def _analyze_query(self, state: WorkflowState) -> WorkflowState:
        """Analyze the mathematical query and determine complexity."""
        print(f"🔍 Analyzing query: {state['user_input']}")

        # Use LLM to analyze the query
        analysis_prompt = f"""
        Analyze this mathematical query and determine if it needs basic or advanced calculation:
        Query: {state['user_input']}

        Consider:
        - Basic: Simple arithmetic (+ - * / parentheses)
        - Advanced: Functions like sqrt, sin, cos, log, pi, e

        Respond with either "basic" or "advanced" followed by the mathematical expression to evaluate.
        Format: <type>|<expression>

        Example: "basic|2 + 3 * 4" or "advanced|sqrt(16) + sin(pi/2)"
        """

        try:
            response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
            analysis = response.content.strip()

            if "|" in analysis:
                query_type, expression = analysis.split("|", 1)
                state["current_task"] = query_type.strip()
                state["results"]["expression"] = expression.strip()
                state["next_action"] = query_type.strip()
            else:
                state["current_task"] = "error"
                state["results"]["error"] = "Could not parse query analysis"
                state["next_action"] = "error"

        except Exception as e:
            state["current_task"] = "error"
            state["results"]["error"] = f"Analysis failed: {str(e)}"
            state["next_action"] = "error"

        state["messages"].append(AIMessage(content=f"Analysis: {state['current_task']}"))
        return state

    def _route_calculation(self, state: WorkflowState) -> str:
        """Route to appropriate calculation method."""
        task = state["current_task"]
        if task == "basic":
            return "simple"
        elif task == "advanced":
            return "advanced"
        else:
            return "error"

    def _simple_calculation(self, state: WorkflowState) -> WorkflowState:
        """Perform simple calculation using basic calculator."""
        print("🧮 Performing simple calculation")

        expression = state["results"].get("expression", "")
        result = calculator_tool._run(expression)

        state["results"]["calculation"] = result
        state["messages"].append(AIMessage(content=f"Simple calculation: {result}"))
        return state

    def _advanced_calculation(self, state: WorkflowState) -> WorkflowState:
        """Perform advanced calculation using advanced calculator."""
        print("🔬 Performing advanced calculation")

        expression = state["results"].get("expression", "")
        result = advanced_calculator_tool._run(expression)

        state["results"]["calculation"] = result
        state["messages"].append(AIMessage(content=f"Advanced calculation: {result}"))
        return state

    def _validate_result(self, state: WorkflowState) -> WorkflowState:
        """Validate the calculation result."""
        print("✅ Validating result")

        calculation = state["results"].get("calculation", "")

        if "Error:" in calculation:
            state["error_count"] = state.get("error_count", 0) + 1
            state["results"]["validation"] = "failed"
            state["next_action"] = "retry" if state["error_count"] < 3 else "error"
        else:
            state["results"]["validation"] = "passed"
            state["next_action"] = "success"

        return state

    def _check_validation(self, state: WorkflowState) -> str:
        """Check validation results and route accordingly."""
        return state["next_action"]

    def _format_response(self, state: WorkflowState) -> WorkflowState:
        """Format the final response."""
        print("📝 Formatting response")

        expression = state["results"].get("expression", "")
        calculation = state["results"].get("calculation", "")

        formatted_response = f"""
        📊 Mathematical Calculation Complete

        Query: {state['user_input']}
        Expression: {expression}
        Result: {calculation}

        ✅ Calculation completed successfully!
        """

        state["results"]["final_response"] = formatted_response
        state["messages"].append(AIMessage(content=formatted_response))
        return state

    def _handle_error(self, state: WorkflowState) -> WorkflowState:
        """Handle errors in the workflow."""
        print("❌ Handling error")

        error_msg = state["results"].get("error", "Unknown error occurred")
        error_response = f"""
        ❌ Error Processing Mathematical Query

        Query: {state['user_input']}
        Error: {error_msg}

        Please try rephrasing your mathematical question.
        """

        state["results"]["final_response"] = error_response
        state["messages"].append(AIMessage(content=error_response))
        return state

    def process_query(self, query: str) -> str:
        """Process a mathematical query through the workflow."""
        print(f"🚀 Starting workflow for query: {query}")

        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "current_task": "",
            "results": {},
            "next_action": "",
            "user_input": query,
            "error_count": 0,
        }

        # Run the workflow
        try:
            config = {"configurable": {"thread_id": "math_workflow"}}
            final_state = self.app.invoke(initial_state, config)
            return final_state["results"].get("final_response", "No response generated")
        except Exception as e:
            return f"❌ Workflow failed: {str(e)}"

    def demonstrate_workflow(self):
        """Demonstrate the workflow with sample queries."""
        print("🔄 LangGraph Workflow Demo")
        print("=" * 60)

        # Sample queries
        test_queries = [
            "What is 25 + 17 * 3?",
            "Calculate the square root of 144",
            "What is sin(pi/2) + cos(0)?",
            "Find 2 * pi * 5",
            "What is the result of (10 + 5) / 3?",
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test {i}: {query} ---")
            result = self.process_query(query)
            print(f"Result: {result}")
            print("-" * 40)

        # Show LangSmith status
        print(f"\n📊 LangSmith Tracing: {'✅ Enabled' if langsmith_setup.is_enabled() else '❌ Disabled'}")


def main():
    """Main function to run the workflow demo."""
    try:
        # Validate API keys
        validation = settings.validate_api_keys()
        if not validation["valid"]:
            print(f"❌ Missing API keys: {', '.join(validation['missing_keys'])}")
            print("Please set the required API keys in your .env file.")
            return

        # Create and run the workflow demo
        workflow = MathWorkflowGraph()
        workflow.demonstrate_workflow()

    except Exception as e:
        print(f"❌ Workflow demo failed: {e}")
        print("\n💡 Make sure you have:")
        print("1. Set OPENAI_API_KEY in your .env file")
        print("2. Installed all dependencies: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
