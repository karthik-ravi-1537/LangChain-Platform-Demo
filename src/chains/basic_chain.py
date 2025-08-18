"""
Basic LangChain examples demonstrating chains and prompt templates.
"""

import os
import sys

from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain_openai import OpenAI

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.settings import settings
from src.langsmith.setup import langsmith_setup


class ListOutputParser(BaseOutputParser[list[str]]):
    """Parse output into a list of strings."""

    def parse(self, text: str) -> list[str]:
        """Parse the LLM output into a list."""
        return [item.strip() for item in text.split("\n") if item.strip()]


class BasicChain:
    """Demonstrates basic LangChain functionality."""

    def __init__(self):
        """Initialize the basic chain with OpenAI LLM."""
        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY in your .env file.")

        # Initialize the LLM
        self.llm = OpenAI(api_key=settings.OPENAI_API_KEY, temperature=0.7)

        # Create prompt templates
        self.simple_prompt = PromptTemplate(
            input_variables=["topic"], template="Tell me an interesting fact about {topic}."
        )

        self.list_prompt = PromptTemplate(
            input_variables=["topic", "count"],
            template="List {count} interesting facts about {topic}. Put each fact on a new line.",
        )

        # Create chains using RunnableSequence (prompt | llm)
        self.simple_chain = self.simple_prompt | self.llm

        self.list_chain = self.list_prompt | self.llm | ListOutputParser()

    def get_simple_fact(self, topic: str) -> str:
        """Get a simple fact about a topic."""
        print(f"🔍 Getting fact about: {topic}")

        try:
            result = self.simple_chain.invoke({"topic": topic})
            print(f"✅ Generated fact about {topic}")
            return result
        except Exception as e:
            print(f"❌ Error generating fact: {e}")
            return f"Sorry, I couldn't generate a fact about {topic}."

    def get_fact_list(self, topic: str, count: int = 3) -> list[str]:
        """Get a list of facts about a topic."""
        print(f"🔍 Getting {count} facts about: {topic}")

        try:
            result = self.list_chain.invoke({"topic": topic, "count": count})
            print(f"✅ Generated {len(result)} facts about {topic}")
            return result
        except Exception as e:
            print(f"❌ Error generating facts: {e}")
            return [f"Sorry, I couldn't generate facts about {topic}."]

    def demonstrate_basic_usage(self):
        """Demonstrate basic chain functionality."""
        print("🚀 Basic Chain Demo")
        print("=" * 50)

        # Simple fact generation
        print("\n1. Simple Fact Generation:")
        fact = self.get_simple_fact("artificial intelligence")
        print(f"Result: {fact}")

        # List generation
        print("\n2. List Generation:")
        facts = self.get_fact_list("space exploration", 3)
        for i, fact in enumerate(facts, 1):
            print(f"   {i}. {fact}")

        # Show LangSmith status
        print(f"\n📊 LangSmith Tracing: {'✅ Enabled' if langsmith_setup.is_enabled() else '❌ Disabled'}")


if __name__ == "__main__":
    # Validate API keys
    validation = settings.validate_api_keys()
    if not validation["valid"]:
        print(f"❌ Missing API keys: {', '.join(validation['missing_keys'])}")
        print("Please set the required API keys in your .env file.")
        exit(1)

    # Create and run the basic chain demo
    chain = BasicChain()
    chain.demonstrate_basic_usage()
