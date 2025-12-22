"""LLM configuration and management."""

from typing import Literal, Optional

from langchain.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from autodnd.config import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_NUM_CTX,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_OLLAMA_API_KEY,
)


class LLMConfig(BaseModel):
    """LLM configuration model."""

    provider: Literal["openai", "ollama"] = Field(
        default=DEFAULT_LLM_PROVIDER, description="LLM provider"
    )
    api_key: Optional[str] = Field(default=None, description="API key for OpenAI")
    base_url: Optional[str] = Field(
        default=DEFAULT_LLM_BASE_URL, description="Base URL (for Ollama or custom OpenAI endpoints)"
    )
    model: str = Field(default=DEFAULT_LLM_MODEL, description="Model name")
    temperature: float = Field(
        default=DEFAULT_LLM_TEMPERATURE, ge=0.0, le=2.0, description="Temperature"
    )
    max_tokens: Optional[int] = Field(default=None, description="Max tokens")
    timeout: int = Field(default=DEFAULT_LLM_TIMEOUT, description="Timeout in seconds")


class LLMConfigManager:
    """Manages LLM configuration and hot-reload."""

    def __init__(self, initial_config: Optional[LLMConfig] = None) -> None:
        """Initialize with optional config."""
        self._config = initial_config or LLMConfig()
        self._llm_instance: Optional[BaseChatModel] = None
        # Don't initialize LLM at creation time - wait until first use
        # This allows the app to start without API keys

    @property
    def config(self) -> LLMConfig:
        """Get current config."""
        return self._config

    def update_config(self, new_config: LLMConfig) -> None:
        """Update configuration and recreate LLM instance."""
        self._config = new_config
        self._update_llm()

    def get_llm(self) -> ChatOpenAI:
        """Get current LLM instance."""
        if self._llm_instance is None:
            self._update_llm()
        return self._llm_instance

    def _update_llm(self) -> None:
        """Update LLM instance based on current config."""

        kwargs = {
            "model": self._config.model,
            "temperature": self._config.temperature,
            "timeout": self._config.timeout,
            "num_ctx": DEFAULT_LLM_NUM_CTX,
        }

        if self._config.max_tokens:
            kwargs["max_tokens"] = self._config.max_tokens

        if self._config.provider == "ollama":
            kwargs["base_url"] = self._config.base_url or DEFAULT_LLM_BASE_URL
            kwargs["api_key"] = DEFAULT_OLLAMA_API_KEY  # Ollama doesn't require real API key
        else:
            # For OpenAI, only set api_key if provided (allows environment variable fallback)
            if self._config.api_key:
                kwargs["api_key"] = self._config.api_key
            if self._config.base_url:
                kwargs["base_url"] = self._config.base_url

        try:
            if self._config.provider == "ollama":
                self._llm_instance = ChatOllama(**kwargs)
            else:
                self._llm_instance = ChatOpenAI(**kwargs)
        except Exception as e:
            # Log error but don't fail - LLM will be created when needed
            import logging
            logging.warning(f"Failed to initialize LLM: {e}. LLM will be created when first used.")
            self._llm_instance = None

