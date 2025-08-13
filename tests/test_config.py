"""
Test suite for the configuration package.
"""
import os
import sys

import pytest

# Add the project root to the path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import settings, Settings, get_config, DevelopmentConfig, ProductionConfig, TestingConfig


class TestSettings:
    """Test the Settings class."""

    def test_settings_instance_creation(self):
        """Test that settings instance is created correctly."""
        assert isinstance(settings, Settings)

    def test_settings_has_required_attributes(self):
        """Test that settings has all required attributes."""
        required_attrs = [
            'OPENAI_API_KEY',
            'LANGSMITH_TRACING',
            'LANGSMITH_ENDPOINT',
            'LANGSMITH_API_KEY',
            'LANGSMITH_PROJECT'
        ]

        for attr in required_attrs:
            assert hasattr(settings, attr), f"Settings missing attribute: {attr}"

    def test_validate_api_keys_with_no_keys(self):
        """Test API key validation when no keys are set."""
        test_settings = Settings()
        test_settings.OPENAI_API_KEY = None
        test_settings.LANGCHAIN_API_KEY = None

        validation = test_settings.validate_api_keys()
        assert not validation["valid"]
        assert "OPENAI_API_KEY" in validation["missing_keys"]

    def test_validate_api_keys_with_openai_key(self):
        """Test API key validation with OpenAI key only."""
        test_settings = Settings()
        test_settings.OPENAI_API_KEY = "test-key"
        test_settings.LANGSMITH_API_KEY = None
        test_settings.LANGSMITH_TRACING = False  # Fixed: was LANGCHAIN_TRACING_V2

        validation = test_settings.validate_api_keys()
        assert validation["valid"]
        assert len(validation["missing_keys"]) == 0


class TestEnvironmentConfigs:
    """Test environment-specific configurations."""

    def test_development_config(self):
        """Test development configuration."""
        dev_config = get_config("development")
        assert isinstance(dev_config, DevelopmentConfig)
        assert dev_config.DEBUG == True
        assert dev_config.LOG_LEVEL == "DEBUG"
        assert "dev" in dev_config.LANGCHAIN_PROJECT

    def test_production_config(self):
        """Test production configuration."""
        prod_config = get_config("production")
        assert isinstance(prod_config, ProductionConfig)
        assert prod_config.DEBUG == False
        assert prod_config.LOG_LEVEL == "INFO"
        assert "prod" in prod_config.LANGCHAIN_PROJECT

    def test_testing_config(self):
        """Test testing configuration."""
        test_config = get_config("testing")
        assert isinstance(test_config, TestingConfig)
        assert test_config.DEBUG == True
        assert test_config.OPENAI_API_KEY == "test-key"
        assert test_config.LANGCHAIN_API_KEY == "test-key"

    def test_default_config(self):
        """Test that invalid environment returns development config."""
        default_config = get_config("invalid-env")
        assert isinstance(default_config, DevelopmentConfig)


class TestConfigPackage:
    """Test the config package structure."""

    def test_package_imports(self):
        """Test that all expected items can be imported from config package."""
        from config import (
            Settings, settings, get_config,
            DevelopmentConfig, ProductionConfig, TestingConfig
        )

        # Check that all imports are successful
        assert Settings is not None
        assert settings is not None
        assert get_config is not None
        assert DevelopmentConfig is not None
        assert ProductionConfig is not None
        assert TestingConfig is not None

    def test_package_version(self):
        """Test that package has version information."""
        import config
        assert hasattr(config, '__version__')
        assert config.__version__ == "1.0.0"

    def test_package_all_attribute(self):
        """Test that package has proper __all__ attribute."""
        import config
        assert hasattr(config, '__all__')
        expected_exports = [
            "Settings", "settings", "get_config",
            "DevelopmentConfig", "ProductionConfig", "TestingConfig"
        ]
        for item in expected_exports:
            assert item in config.__all__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
