"""
Configuration settings for the LangChain demo project.
"""
import os
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

    # LangSmith Configuration
    LANGSMITH_TRACING: bool = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    LANGSMITH_ENDPOINT: str = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    LANGSMITH_API_KEY: Optional[str] = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "langchain-demo")

    # Other LLM providers
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")

    # Search API
    SERPAPI_API_KEY: Optional[str] = os.getenv("SERPAPI_API_KEY")

    # Application Settings
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def validate_api_keys(self) -> dict:
        """Validate that required API keys are present."""
        missing_keys = []

        if not self.OPENAI_API_KEY:
            missing_keys.append("OPENAI_API_KEY")

        if self.LANGSMITH_TRACING and not self.LANGSMITH_API_KEY:
            missing_keys.append("LANGSMITH_API_KEY")

        return {
            "valid": len(missing_keys) == 0,
            "missing_keys": missing_keys
        }


# Global settings instance
settings = Settings()
