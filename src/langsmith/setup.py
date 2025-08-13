"""
LangSmith setup and configuration for monitoring and tracing.
"""
import os
from typing import Optional

from langsmith import Client

from config.settings import settings


class LangSmithSetup:
    """Setup and manage LangSmith tracing and monitoring."""

    def __init__(self):
        self.client: Optional[Client] = None
        self._setup_tracing()

    def _setup_tracing(self):
        """Initialize LangSmith tracing if API key is available."""
        if settings.LANGSMITH_API_KEY and settings.LANGSMITH_TRACING:
            try:
                # Set environment variables for LangChain tracing
                os.environ["LANGSMITH_TRACING"] = "true"
                os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
                os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
                os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT

                # Initialize LangSmith client
                self.client = Client(
                    api_url=settings.LANGSMITH_ENDPOINT,
                    api_key=settings.LANGSMITH_API_KEY
                )
                print(f"✅ LangSmith tracing enabled for project: {settings.LANGSMITH_PROJECT}")

            except Exception as e:
                print(f"⚠️  LangSmith setup failed: {e}")
                self.client = None
        else:
            print("ℹ️  LangSmith tracing disabled (no API key or tracing disabled)")

    def create_dataset(self, dataset_name: str, description: str = ""):
        """Create a new dataset in LangSmith."""
        if not self.client:
            print("❌ LangSmith client not initialized")
            return None

        try:
            dataset = self.client.create_dataset(
                dataset_name=dataset_name,
                description=description
            )
            print(f"✅ Dataset '{dataset_name}' created successfully")
            return dataset
        except Exception as e:
            print(f"❌ Failed to create dataset: {e}")
            return None

    def log_feedback(self, run_id: str, key: str, score: float, comment: str = ""):
        """Log feedback for a specific run."""
        if not self.client:
            print("❌ LangSmith client not initialized")
            return

        try:
            self.client.create_feedback(
                run_id=run_id,
                key=key,
                score=score,
                comment=comment
            )
            print(f"✅ Feedback logged for run {run_id}")
        except Exception as e:
            print(f"❌ Failed to log feedback: {e}")

    def is_enabled(self) -> bool:
        """Check if LangSmith tracing is enabled."""
        return self.client is not None


# Global LangSmith setup instance
langsmith_setup = LangSmithSetup()
