"""Tests for SecurityConfig."""

from autodnd.api.llm_config import LLMConfig
from autodnd.api.security_config import SecurityConfig, SecurityConfigManager


class TestSecurityConfig:
    """Test suite for SecurityConfig."""

    def test_security_config_defaults(self):
        """Test SecurityConfig with defaults."""
        config = SecurityConfig()
        assert config.enabled is True
        assert config.use_llm_validation is True
        assert config.max_input_length == 1000

    def test_security_config_custom(self):
        """Test SecurityConfig with custom values."""
        config = SecurityConfig(
            enabled=False,
            use_llm_validation=False,
            max_input_length=500,
        )
        assert config.enabled is False
        assert config.use_llm_validation is False
        assert config.max_input_length == 500

    def test_get_security_llm_config_default(self):
        """Test getting default security LLM config."""
        config = SecurityConfig()
        llm_config = config.get_security_llm_config()
        assert isinstance(llm_config, LLMConfig)
        assert llm_config.model == "qwen2.5:8b"
        assert llm_config.temperature == 0.1

    def test_get_security_llm_config_custom(self):
        """Test getting custom security LLM config."""
        custom_llm_config = LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            temperature=0.2,
        )
        config = SecurityConfig(security_llm_config=custom_llm_config)
        llm_config = config.get_security_llm_config()
        assert llm_config.model == "gpt-3.5-turbo"
        assert llm_config.temperature == 0.2

    def test_security_config_manager(self):
        """Test SecurityConfigManager."""
        manager = SecurityConfigManager()
        assert manager.config is not None
        assert isinstance(manager.config, SecurityConfig)

    def test_security_config_manager_update(self):
        """Test updating SecurityConfigManager."""
        manager = SecurityConfigManager()
        new_config = SecurityConfig(enabled=False)
        manager.update_config(new_config)
        assert manager.config.enabled is False

    def test_interface_completeness(self):
        """Test that SecurityConfig has all required methods."""
        config = SecurityConfig()
        assert hasattr(config, "get_security_llm_config")
        assert callable(config.get_security_llm_config)

        manager = SecurityConfigManager()
        assert hasattr(manager, "config")
        assert hasattr(manager, "update_config")
        assert callable(manager.update_config)

