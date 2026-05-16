import logging
import os
from typing import Any, Dict, List

from openai import OpenAI

from .openai_base import OpenAIBaseAdapter, _filter_api_kwargs

logger = logging.getLogger(__name__)


class LMStudioAdapter(OpenAIBaseAdapter):
    """Adapter for LM Studio's local OpenAI-compatible server."""

    def init_client(self):
        """Initialize an OpenAI client pointed at the local LM Studio server.

        LM Studio exposes an OpenAI-compatible API (default http://localhost:1234/v1).
        No API key is required, but the OpenAI SDK still wants a non-empty string.
        Override the endpoint with LMSTUDIO_BASE_URL if running on a different host/port.
        """
        base_url = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:4444/v1")
        api_key = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")

        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=1800,
        )

    def _chat_completion(self, messages: List[Dict[str, str]]) -> Any:
        api_kwargs = _filter_api_kwargs(self.model_config.kwargs)
        logger.debug(
            f"Calling LM Studio API with model: {self.model_config.model_name} and kwargs: {api_kwargs}"
        )
        return self.client.chat.completions.create(
            model=self.model_config.model_name, messages=messages, **api_kwargs
        )
