"""Smoke tests for LLM configuration in autodnd/config.py."""

import pytest

from autodnd.api.llm_config import LLMConfig, LLMConfigManager
from autodnd.config import (
    DEFAULT_ACTION_VALIDATOR_TEMPERATURE,
    DEFAULT_GAME_MASTER_TEMPERATURE,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_NUM_CTX,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_NPC_AGENT_TEMPERATURE,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_RAG_AGENT_TEMPERATURE,
    DEFAULT_SECURITY_LLM_MODEL,
    DEFAULT_SECURITY_LLM_TEMPERATURE,
    DEFAULT_SECURITY_LLM_TIMEOUT,
)


class TestConfigSmoke:
    """Smoke tests to validate LLM configuration is valid and can be used."""

    def test_config_imports_successfully(self):
        """Test that all config constants can be imported without errors."""
        # If we get here, imports succeeded
        assert DEFAULT_LLM_PROVIDER is not None
        assert DEFAULT_LLM_MODEL is not None
        assert DEFAULT_LLM_TEMPERATURE is not None
        assert DEFAULT_LLM_TIMEOUT is not None
        assert DEFAULT_LLM_NUM_CTX is not None


    def test_config_manager_get_llm_does_not_raise_immediately(self):
        """Test that get_llm() doesn't raise immediately (lazy initialization)."""
        manager = LLMConfigManager()
        manager._update_llm()
        # get_llm() will try to create LLM instance, but should handle errors gracefully
        # We expect it might fail if Ollama isn't running, but shouldn't raise unhandled exceptions
        llm = manager.get_llm()
        # If successful, should return a BaseChatModel instance

        assert llm.invoke("Hello, world! Just say yes, nothing else!") is not None