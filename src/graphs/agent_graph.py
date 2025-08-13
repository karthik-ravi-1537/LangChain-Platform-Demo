"""
Agent Graph demonstrating multi-agent coordination using LangGraph.
"""
import json
import os
import sys
from typing import Dict, List, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.settings import settings
from src.langsmith.setup import langsmith_setup
from src.tools.calculator import calculator_tool, advanced_calculator_tool
from src.tools.web_search import web_search_tool
from src.chains.rag_chain import RAGChain


class AgentState(TypedDict):
    """State for the multi-agent system."""
    messages: List[BaseMessage]
    current_agent: str
    task_type: str
    user_query: str
    agent_results: Dict[str, str]
    final_response: str
    next_action: str


class MultiAgentSystem:
    """Multi-agent system using LangGraph for coordination."""

    def __init__(self):
        """Initialize the multi-agent system."""
        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API key required for multi-agent system")

        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model="gpt-3.5-turbo",
            temperature=0.3
        )

        # Initialize specialized agents
        self._initialize_agents()

        # Create the coordination graph
        self.workflow = StateGraph(AgentState)
        self._build_coordination_graph()

        # Add memory for persistence
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)

    def _initialize_agents(self):
        """Initialize specialized agents with different tools."""
        # Math Agent - simplified for compatibility
        from langchain.agents import create_react_agent
        from langchain.agents import AgentExecutor
        from langchain import hub

        # Get the react prompt
        try:
            react_prompt = hub.pull("hwchase17/react")
        except:
            # Fallback prompt if hub is not available
            from langchain.prompts import PromptTemplate
            react_prompt = PromptTemplate.from_template("""
            Answer the following questions as best you can. You have access to the following tools:
            
            {tools}
            
            Use the following format:
            
            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question
            
            Begin!
            
            Question: {input}
            Thought: {agent_scratchpad}
            """)

        # Math Agent
        math_agent = create_react_agent(self.llm, [calculator_tool, advanced_calculator_tool], react_prompt)
        self.math_agent = AgentExecutor(agent=math_agent, tools=[calculator_tool, advanced_calculator_tool],
                                        verbose=True)

        # Research Agent - simplified to avoid multi-input issues
        research_agent = create_react_agent(self.llm, [web_search_tool], react_prompt)
        self.research_agent = AgentExecutor(agent=research_agent, tools=[web_search_tool], verbose=True)

        # Knowledge Agent (RAG)
        try:
            self.rag_chain = RAGChain()
            self.rag_chain.load_sample_documents()
        except Exception as e:
            print(f"⚠️  RAG chain initialization failed: {e}")
            self.rag_chain = None

    def _build_coordination_graph(self):
        """Build the coordination graph with agents and routing."""
        # Add agent nodes
        self.workflow.add_node("coordinator", self._coordinate_task)
        self.workflow.add_node("math_agent", self._run_math_agent)
        self.workflow.add_node("research_agent", self._run_research_agent)
        self.workflow.add_node("knowledge_agent", self._run_knowledge_agent)
        self.workflow.add_node("synthesizer", self._synthesize_results)

        # Set entry point
        self.workflow.set_entry_point("coordinator")

        # Add conditional edges from coordinator
        self.workflow.add_conditional_edges(
            "coordinator",
            self._route_to_agent,
            {
                "math": "math_agent",
                "research": "research_agent",
                "knowledge": "knowledge_agent",
                "synthesize": "synthesizer"
            }
        )

        # Connect agents to synthesizer
        self.workflow.add_edge("math_agent", "synthesizer")
        self.workflow.add_edge("research_agent", "synthesizer")
        self.workflow.add_edge("knowledge_agent", "synthesizer")

        # End at synthesizer
        self.workflow.add_edge("synthesizer", END)

    def _coordinate_task(self, state: AgentState) -> AgentState:
        """Coordinate and route tasks to appropriate agents."""
        print(f"🎯 Coordinating task: {state['user_query']}")

        # Analyze the query to determine task type
        analysis_prompt = f"""
        Analyze this user query and determine the best approach:
        Query: {state['user_query']}
        
        Task types:
        - math: Mathematical calculations, equations, formulas
        - research: Current events, facts, web information
        - knowledge: Questions about LangChain, LangGraph, LangSmith, AI concepts
        - synthesize: Complex queries requiring multiple agents
        
        If the query needs multiple agents, respond with "synthesize".
        Otherwise, respond with the single most appropriate task type.
        
        Response format: <task_type>
        """

        try:
            response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
            task_type = response.content.strip().lower()

            # Map task types to valid routing options
            valid_tasks = ["math", "research", "knowledge", "synthesize"]
            if task_type not in valid_tasks:
                task_type = "synthesize"  # Default to synthesize for complex queries

            state["task_type"] = task_type
            state["next_action"] = task_type

            print(f"📋 Task classified as: {task_type}")

        except Exception as e:
            print(f"❌ Coordination error: {e}")
            state["task_type"] = "synthesize"
            state["next_action"] = "synthesize"

        state["messages"].append(AIMessage(content=f"Task routed to: {state['task_type']}"))
        return state

    def _route_to_agent(self, state: AgentState) -> str:
        """Route to the appropriate agent based on task type."""
        return state["next_action"]

    def _run_math_agent(self, state: AgentState) -> AgentState:
        """Run the math agent for calculations."""
        print("🧮 Math Agent processing...")

        try:
            result = self.math_agent.invoke(state["user_query"])
            state["agent_results"]["math"] = result
            state["current_agent"] = "math"

            print(f"✅ Math result: {result}")

        except Exception as e:
            error_msg = f"Math agent error: {str(e)}"
            state["agent_results"]["math"] = error_msg
            print(f"❌ {error_msg}")

        state["messages"].append(AIMessage(content=f"Math Agent: {state['agent_results']['math']}"))
        return state

    def _run_research_agent(self, state: AgentState) -> AgentState:
        """Run the research agent for web information."""
        print("🔍 Research Agent processing...")

        try:
            result = self.research_agent.invoke(state["user_query"])
            state["agent_results"]["research"] = result
            state["current_agent"] = "research"

            print(f"✅ Research result: {result[:200]}...")

        except Exception as e:
            error_msg = f"Research agent error: {str(e)}"
            state["agent_results"]["research"] = error_msg
            print(f"❌ {error_msg}")

        state["messages"].append(AIMessage(content=f"Research Agent: {state['agent_results']['research']}"))
        return state

    def _run_knowledge_agent(self, state: AgentState) -> AgentState:
        """Run the knowledge agent using RAG."""
        print("📚 Knowledge Agent processing...")

        try:
            if self.rag_chain:
                result = self.rag_chain.ask_question(state["user_query"])
                state["agent_results"]["knowledge"] = result["answer"]
                state["current_agent"] = "knowledge"

                print(f"✅ Knowledge result: {result['answer'][:200]}...")
            else:
                state["agent_results"]["knowledge"] = "Knowledge agent not available"
                print("⚠️  Knowledge agent not available")

        except Exception as e:
            error_msg = f"Knowledge agent error: {str(e)}"
            state["agent_results"]["knowledge"] = error_msg
            print(f"❌ {error_msg}")

        state["messages"].append(AIMessage(content=f"Knowledge Agent: {state['agent_results']['knowledge']}"))
        return state

    def _synthesize_results(self, state: AgentState) -> AgentState:
        """Synthesize results from multiple agents."""
        print("🔄 Synthesizing results...")

        # Collect all agent results
        results = state["agent_results"]

        if state["task_type"] == "synthesize":
            # For complex queries, run multiple agents
            if "math" not in results:
                math_result = self._get_math_result(state["user_query"])
                results["math"] = math_result

            if "research" not in results:
                research_result = self._get_research_result(state["user_query"])
                results["research"] = research_result

            if "knowledge" not in results:
                knowledge_result = self._get_knowledge_result(state["user_query"])
                results["knowledge"] = knowledge_result

        # Synthesize final response
        synthesis_prompt = f"""
        Synthesize the following agent results into a comprehensive response:
        
        User Query: {state['user_query']}
        
        Agent Results:
        {json.dumps(results, indent=2)}
        
        Provide a clear, helpful response that integrates relevant information from the agents.
        If some agents didn't provide useful information, focus on the most relevant results.
        """

        try:
            response = self.llm.invoke([HumanMessage(content=synthesis_prompt)])
            state["final_response"] = response.content

            print(f"✅ Synthesis complete")

        except Exception as e:
            state["final_response"] = f"Synthesis error: {str(e)}"
            print(f"❌ Synthesis error: {e}")

        state["messages"].append(AIMessage(content=state["final_response"]))
        return state

    def _get_math_result(self, query: str) -> str:
        """Get math result for synthesis."""
        try:
            return self.math_agent.invoke(query)
        except:
            return "No mathematical computation needed"

    def _get_research_result(self, query: str) -> str:
        """Get research result for synthesis."""
        try:
            return self.research_agent.invoke(query)
        except:
            return "No web research needed"

    def _get_knowledge_result(self, query: str) -> str:
        """Get knowledge result for synthesis."""
        try:
            if self.rag_chain:
                result = self.rag_chain.ask_question(query)
                return result["answer"]
            return "Knowledge base not available"
        except:
            return "No knowledge base query needed"

    def process_query(self, query: str) -> str:
        """Process a query through the multi-agent system."""
        print(f"🚀 Multi-Agent System processing: {query}")

        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "current_agent": "",
            "task_type": "",
            "user_query": query,
            "agent_results": {},
            "final_response": "",
            "next_action": ""
        }

        # Run the workflow
        try:
            config = {"configurable": {"thread_id": "multi_agent_system"}}
            final_state = self.app.invoke(initial_state, config)
            return final_state["final_response"]
        except Exception as e:
            return f"❌ Multi-agent system failed: {str(e)}"

    def demonstrate_multi_agent_system(self):
        """Demonstrate the multi-agent system."""
        print("🤖 Multi-Agent System Demo")
        print("=" * 70)

        # Test queries that demonstrate different agent types
        test_queries = [
            "What is the square root of 144 plus 25?",
            "What are the latest developments in quantum computing?",
            "How does LangGraph handle state management?",
            "Calculate the area of a circle with radius 5 and tell me about recent AI breakthroughs",
            "What is 2 + 2 and what does LangChain do?"
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Query {i}: {query} ---")
            result = self.process_query(query)
            print(f"\n🤖 Final Response:")
            print(result)
            print("-" * 50)

        # Show LangSmith status
        print(f"\n📊 LangSmith Tracing: {'✅ Enabled' if langsmith_setup.is_enabled() else '❌ Disabled'}")


def main():
    """Main function to run the multi-agent demo."""
    try:
        # Validate API keys
        validation = settings.validate_api_keys()
        if not validation["valid"]:
            print(f"❌ Missing API keys: {', '.join(validation['missing_keys'])}")
            print("Please set the required API keys in your .env file.")
            return

        # Create and run the multi-agent system demo
        multi_agent = MultiAgentSystem()
        multi_agent.demonstrate_multi_agent_system()

    except Exception as e:
        print(f"❌ Multi-agent demo failed: {e}")
        print("\n💡 Make sure you have:")
        print("1. Set OPENAI_API_KEY in your .env file")
        print("2. Installed all dependencies: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
