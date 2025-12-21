"""Security layer configuration."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from autodnd.api.llm_config import LLMConfig


class SecurityConfig(BaseModel):
    """Security layer configuration."""

    enabled: bool = Field(default=True, description="Whether security layer is enabled")
    use_llm_validation: bool = Field(
        default=True, description="Whether to use LLM-based validation (cheaper model)"
    )
    max_input_length: int = Field(default=1000, ge=1, le=10000, description="Maximum input length")
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
            model="qwen2.5:8b",  # Cheaper model for security
            temperature=0.1,
            timeout=30,
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

