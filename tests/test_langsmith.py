"""
Test suite for the LangSmith setup and monitoring.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add the project root to the path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.langsmith.setup import LangSmithSetup, langsmith_setup


class TestLangSmithSetup:
    """Test the LangSmith setup functionality."""

    def test_langsmith_setup_initialization(self):
        """Test that LangSmith setup can be initialized."""
        assert isinstance(langsmith_setup, LangSmithSetup)

    def test_is_enabled_method(self):
        """Test the is_enabled method."""
        result = langsmith_setup.is_enabled()
        assert isinstance(result, bool)

    @patch("src.langsmith.setup.settings")
    def test_setup_with_no_api_key(self, mock_settings):
        """Test setup behavior when no API key is provided."""
        mock_settings.LANGCHAIN_API_KEY = None
        mock_settings.LANGCHAIN_TRACING_V2 = False

        setup = LangSmithSetup()
        assert not setup.is_enabled()

    @patch("src.langsmith.setup.settings")
    @patch("src.langsmith.setup.Client")
    def test_setup_with_api_key(self, mock_client, mock_settings):
        """Test setup behavior when API key is provided."""
        # Use the correct attribute names that match our settings class
        mock_settings.LANGSMITH_API_KEY = "test-key"
        mock_settings.LANGSMITH_TRACING = True
        mock_settings.LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
        mock_settings.LANGSMITH_PROJECT = "test-project"

        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        setup = LangSmithSetup()
        assert setup.is_enabled()
        mock_client.assert_called_once()

    def test_create_dataset_without_client(self):
        """Test dataset creation when client is not initialized."""
        setup = LangSmithSetup()
        setup.client = None

        result = setup.create_dataset("test-dataset", "test description")
        assert result is None

    def test_log_feedback_without_client(self):
        """Test feedback logging when client is not initialized."""
        setup = LangSmithSetup()
        setup.client = None

        # Should not raise an exception
        setup.log_feedback("test-run-id", "test-key", 0.8, "test comment")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
