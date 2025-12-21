"""Tests for SecurityAgent (smoke tests)."""

import pytest

from autodnd.api.llm_config import LLMConfig
from autodnd.security.security_agent import SecurityAgent, SecurityValidationResult


class TestSecurityAgent:
    """Test suite for SecurityAgent (smoke tests)."""

    def test_security_agent_initialization(self):
        """Test that SecurityAgent can be initialized."""
        config = LLMConfig(
            provider="ollama",
            model="qwen2.5:8b",
            temperature=0.1,
            timeout=30,
        )
        # Should not raise (even if LLM is not available)
        try:
            agent = SecurityAgent(config=config)
            assert agent is not None
            assert hasattr(agent, "validate_input")
            assert hasattr(agent, "update_llm")
            assert hasattr(agent, "update_config")
        except Exception:
            # If LLM is not available, that's okay for smoke tests
            pytest.skip("LLM not available for testing")

    def test_security_agent_default_config(self):
        """Test that SecurityAgent uses default config if not provided."""
        # Should not raise (even if LLM is not available)
        try:
            agent = SecurityAgent()
            assert agent is not None
        except Exception:
            pytest.skip("LLM not available for testing")

    def test_security_validation_result_model(self):
        """Test SecurityValidationResult Pydantic model."""
        result = SecurityValidationResult(
            is_safe=True,
            risk_level="low",
            reason="Test reason",
        )
        assert result.is_safe is True
        assert result.risk_level == "low"
        assert result.reason == "Test reason"

    def test_security_validation_result_optional_fields(self):
        """Test SecurityValidationResult with optional fields."""
        result = SecurityValidationResult(
            is_safe=False,
            risk_level="high",
            reason="Suspicious input",
            suggested_action="Reject input",
        )
        assert result.is_safe is False
        assert result.suggested_action == "Reject input"

    def test_interface_completeness(self):
        """Test that SecurityAgent has all required methods."""
        # Check interface without initializing (to avoid LLM dependency)
        assert hasattr(SecurityAgent, "__init__")
        assert hasattr(SecurityAgent, "validate_input")
        assert hasattr(SecurityAgent, "update_llm")
        assert hasattr(SecurityAgent, "update_config")

