"""
Web search tool for LangChain agents using multiple search providers.
"""
import os
import sys
from typing import Type, List

import requests
from bs4 import BeautifulSoup
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.settings import settings


class SearchInput(BaseModel):
    """Input for web search tool."""
    query: str = Field(description="Search query to find information on the web")
    num_results: int = Field(default=5, description="Number of search results to return")


class WebSearchTool(BaseTool):
    """Web search tool using DuckDuckGo (no API key required)."""

    name: str = "web_search"
    description: str = "Search the web for current information. Use this when you need up-to-date facts, news, or information not in your training data."
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str, num_results: int = 5) -> str:
        """Execute web search."""
        try:
            # Use DuckDuckGo instant answer API (no key required)
            return self._duckduckgo_search(query, num_results)
        except Exception as e:
            return f"Error: Web search failed - {str(e)}"

    def _duckduckgo_search(self, query: str, num_results: int) -> str:
        """Search using DuckDuckGo."""
        try:
            # DuckDuckGo instant answer API
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []

            # Get instant answer
            if data.get('Abstract'):
                results.append(f"Summary: {data['Abstract']}")

            # Get related topics
            if data.get('RelatedTopics'):
                for topic in data['RelatedTopics'][:num_results]:
                    if isinstance(topic, dict) and topic.get('Text'):
                        results.append(f"Related: {topic['Text']}")

            # If no results, try web scraping approach
            if not results:
                return self._fallback_search(query, num_results)

            return "\n".join(results[:num_results])

        except Exception as e:
            return f"DuckDuckGo search failed: {str(e)}"

    def _fallback_search(self, query: str, num_results: int) -> str:
        """Fallback search method."""
        try:
            # Simple web scraping approach
            search_url = f"https://duckduckgo.com/html/?q={query}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            results = []

            # Find search result snippets
            for result in soup.find_all('a', class_='result__snippet')[:num_results]:
                if result.get_text().strip():
                    results.append(result.get_text().strip())

            return "\n".join(results) if results else "No search results found"

        except Exception as e:
            return f"Fallback search failed: {str(e)}"

    async def _arun(self, query: str, num_results: int = 5) -> str:
        """Async version of web search."""
        return self._run(query, num_results)


class SerpAPISearchTool(BaseTool):
    """Web search tool using SerpAPI (requires API key)."""

    name: str = "serpapi_search"
    description: str = "Advanced web search using SerpAPI with Google results. Requires API key."
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str, num_results: int = 5) -> str:
        """Execute SerpAPI search."""
        if not settings.SERPAPI_API_KEY:
            return "Error: SerpAPI key not configured. Using fallback search."

        try:
            url = "https://serpapi.com/search"
            params = {
                'q': query,
                'api_key': settings.SERPAPI_API_KEY,
                'engine': 'google',
                'num': num_results
            }

            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            results = []

            # Extract organic results
            if 'organic_results' in data:
                for result in data['organic_results'][:num_results]:
                    title = result.get('title', 'No title')
                    snippet = result.get('snippet', 'No snippet')
                    results.append(f"{title}: {snippet}")

            return "\n".join(results) if results else "No search results found"

        except Exception as e:
            return f"SerpAPI search failed: {str(e)}"

    async def _arun(self, query: str, num_results: int = 5) -> str:
        """Async version of SerpAPI search."""
        return self._run(query, num_results)


class NewsSearchTool(BaseTool):
    """News search tool for current events."""

    name: str = "news_search"
    description: str = "Search for recent news articles and current events."
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str, num_results: int = 5) -> str:
        """Execute news search."""
        try:
            # Use NewsAPI-like approach with DuckDuckGo
            news_query = f"{query} news recent"

            url = "https://api.duckduckgo.com/"
            params = {
                'q': news_query,
                'format': 'json',
                'no_html': '1'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []

            # Get news results
            if data.get('RelatedTopics'):
                for topic in data['RelatedTopics'][:num_results]:
                    if isinstance(topic, dict) and topic.get('Text'):
                        results.append(f"📰 {topic['Text']}")

            return "\n".join(results) if results else f"No recent news found for: {query}"

        except Exception as e:
            return f"News search failed: {str(e)}"

    async def _arun(self, query: str, num_results: int = 5) -> str:
        """Async version of news search."""
        return self._run(query, num_results)


# Create tool instances
web_search_tool = WebSearchTool()
serpapi_search_tool = SerpAPISearchTool()
news_search_tool = NewsSearchTool()


def demonstrate_search_tools():
    """Demonstrate the search tools."""
    print("🔍 Web Search Tools Demo")
    print("=" * 60)

    # Test queries
    test_queries = [
        "latest developments in artificial intelligence",
        "current weather in New York",
        "recent news about space exploration",
        "Python programming best practices",
        "quantum computing breakthrough 2024"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: {query} ---")

        # Test web search
        print("\n🌐 Web Search Results:")
        web_result = web_search_tool._run(query, 3)
        print(f"   {web_result}")

        # Test news search
        print("\n📰 News Search Results:")
        news_result = news_search_tool._run(query, 2)
        print(f"   {news_result}")

        print("-" * 40)

    # Show API key status
    print(f"\n🔑 SerpAPI Status: {'✅ Configured' if settings.SERPAPI_API_KEY else '❌ Not configured (using fallback)'}")


def get_available_search_tools() -> List[BaseTool]:
    """Get list of available search tools."""
    tools = [web_search_tool, news_search_tool]

    # Add SerpAPI tool if key is available
    if settings.SERPAPI_API_KEY:
        tools.append(serpapi_search_tool)

    return tools


if __name__ == "__main__":
    demonstrate_search_tools()
