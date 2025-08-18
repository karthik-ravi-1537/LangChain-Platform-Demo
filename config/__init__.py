"""
Configuration package for the LangChain demo project.

This package provides centralized configuration management with environment
variable support and validation.
"""

from .environments import DevelopmentConfig, ProductionConfig, TestingConfig, get_config
from .settings import Settings, settings

__all__ = ["Settings", "settings", "get_config", "DevelopmentConfig", "ProductionConfig", "TestingConfig"]

# Version information
__version__ = "1.0.0"
