"""Security layer configuration."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from autodnd.api.llm_config import LLMConfig
from autodnd.config import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_SECURITY_ENABLED,
    DEFAULT_SECURITY_LLM_MODEL,
    DEFAULT_SECURITY_LLM_TEMPERATURE,
    DEFAULT_SECURITY_LLM_TIMEOUT,
    DEFAULT_SECURITY_MAX_INPUT_LENGTH,
    DEFAULT_SECURITY_USE_LLM_VALIDATION,
)


class SecurityConfig(BaseModel):
    """Security layer configuration."""

    enabled: bool = Field(
        default=DEFAULT_SECURITY_ENABLED, description="Whether security layer is enabled"
    )
    use_llm_validation: bool = Field(
        default=DEFAULT_SECURITY_USE_LLM_VALIDATION,
        description="Whether to use LLM-based validation (cheaper model)",
    )
    max_input_length: int = Field(
        default=DEFAULT_SECURITY_MAX_INPUT_LENGTH,
        ge=1,
        le=10000,
        description="Maximum input length",
    )
    security_llm_config: Optional[LLMConfig] = Field(
        default=None,
        description="LLM config for security agent (uses cheaper model by default)",
    )

    def get_security_llm_config(self) -> LLMConfig:
        """Get security LLM config, with defaults if not set."""
        if self.security_llm_config:
            return self.security_llm_config

        # Default to cheaper model
        return LLMConfig(
            provider="ollama",
            model=DEFAULT_SECURITY_LLM_MODEL,  # Cheaper model for security
            temperature=DEFAULT_SECURITY_LLM_TEMPERATURE,
            timeout=DEFAULT_SECURITY_LLM_TIMEOUT,
            base_url=DEFAULT_OLLAMA_BASE_URL,
        )


class SecurityConfigManager:
    """Manages security configuration."""

    def __init__(self, initial_config: Optional[SecurityConfig] = None) -> None:
        """Initialize with optional config."""
        self._config = initial_config or SecurityConfig()

    @property
    def config(self) -> SecurityConfig:
        """Get current config."""
        return self._config

    def update_config(self, new_config: SecurityConfig) -> None:
        """Update configuration."""
        self._config = new_config

