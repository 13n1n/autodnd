"""Tests for PromptBuilder."""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from autodnd.security.prompt_builder import PromptBuilder


class TestPromptBuilder:
    """Test suite for PromptBuilder."""

    def test_create_system_prompt(self):
        """Test creation of system prompt."""
        template = "You are a helpful assistant."
        prompt = PromptBuilder.create_system_prompt(template)
        assert prompt is not None

    def test_create_chat_prompt(self):
        """Test creation of chat prompt."""
        system_template = "You are a helpful assistant."
        prompt = PromptBuilder.create_chat_prompt(system_template)
        assert isinstance(prompt, ChatPromptTemplate)

    def test_create_chat_prompt_with_user_template(self):
        """Test creation of chat prompt with user template."""
        system_template = "You are a helpful assistant."
        user_template = "User said: {user_input}"
        prompt = PromptBuilder.create_chat_prompt(system_template, user_template)
        assert isinstance(prompt, ChatPromptTemplate)

    def test_create_simple_prompt(self):
        """Test creation of simple prompt."""
        template = "Hello {name}"
        prompt = PromptBuilder.create_simple_prompt(template)
        assert isinstance(prompt, PromptTemplate)

    def test_format_prompt_chat(self):
        """Test formatting of chat prompt."""
        system_template = "You are a helpful assistant."
        prompt = PromptBuilder.create_chat_prompt(system_template)
        formatted = PromptBuilder.format_prompt(prompt, "Hello")
        assert isinstance(formatted, str)
        assert "Hello" in formatted

    def test_format_prompt_simple(self):
        """Test formatting of simple prompt."""
        template = "Hello {name}"
        prompt = PromptBuilder.create_simple_prompt(template)
        formatted = PromptBuilder.format_prompt(prompt, "test", name="World")
        assert isinstance(formatted, str)
        assert "World" in formatted

    def test_interface_completeness(self):
        """Test that PromptBuilder has all required methods."""
        assert hasattr(PromptBuilder, "create_system_prompt")
        assert hasattr(PromptBuilder, "create_chat_prompt")
        assert hasattr(PromptBuilder, "create_simple_prompt")
        assert hasattr(PromptBuilder, "format_prompt")
        assert callable(PromptBuilder.create_system_prompt)
        assert callable(PromptBuilder.create_chat_prompt)
        assert callable(PromptBuilder.create_simple_prompt)
        assert callable(PromptBuilder.format_prompt)

