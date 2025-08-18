"""
Environment-specific configuration settings.
"""

from typing import Any

from .settings import Settings


class DevelopmentConfig(Settings):
    """Development environment configuration."""

    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    LANGCHAIN_PROJECT: str = "langchain-demo-dev"


class ProductionConfig(Settings):
    """Production environment configuration."""

    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    LANGCHAIN_PROJECT: str = "langchain-demo-prod"


class TestingConfig(Settings):
    """Testing environment configuration."""

    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    LANGCHAIN_PROJECT: str = "langchain-demo-test"
    # Override API keys for testing
    OPENAI_API_KEY: str = "test-key"
    LANGCHAIN_API_KEY: str = "test-key"


# Environment configuration mapping
CONFIG_MAP: dict[str, Any] = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}


def get_config(env: str = "development") -> Settings:
    """Get configuration based on environment."""
    return CONFIG_MAP.get(env, DevelopmentConfig)()
