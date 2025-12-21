"""LLM configuration and management."""

from typing import Literal, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM configuration model."""

    provider: Literal["openai", "ollama"] = Field(default="ollama", description="LLM provider")
    api_key: Optional[str] = Field(default=None, description="API key for OpenAI")
    base_url: Optional[str] = Field(default="http://localhost:11434/v1", description="Base URL (for Ollama or custom OpenAI endpoints)")
    model: str = Field(default="gpt-oss:20b", description="Model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature")
    max_tokens: Optional[int] = Field(default=None, description="Max tokens")
    timeout: int = Field(default=60, description="Timeout in seconds")


class LLMConfigManager:
    """Manages LLM configuration and hot-reload."""

    def __init__(self, initial_config: Optional[LLMConfig] = None) -> None:
        """Initialize with optional config."""
        self._config = initial_config or LLMConfig()
        self._llm_instance: Optional[ChatOpenAI] = None
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
        }

        if self._config.max_tokens:
            kwargs["max_tokens"] = self._config.max_tokens

        if self._config.provider == "ollama":
            kwargs["base_url"] = self._config.base_url or "http://localhost:11434/v1"
            kwargs["api_key"] = "ollama"  # Ollama doesn't require real API key
        else:
            # For OpenAI, only set api_key if provided (allows environment variable fallback)
            if self._config.api_key:
                kwargs["api_key"] = self._config.api_key
            if self._config.base_url:
                kwargs["base_url"] = self._config.base_url

        try:
            self._llm_instance = ChatOpenAI(**kwargs)
        except Exception as e:
            # Log error but don't fail - LLM will be created when needed
            import logging
            logging.warning(f"Failed to initialize LLM: {e}. LLM will be created when first used.")
            self._llm_instance = None

