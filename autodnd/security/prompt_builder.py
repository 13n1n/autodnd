"""Secure prompt building using LangChain templates."""

from typing import Any, Optional

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate


class PromptBuilder:
    """Builds secure prompts using LangChain templates."""

    def __init__(self) -> None:
        """Initialize prompt builder."""
        pass

    @staticmethod
    def create_system_prompt(template: str, **kwargs: Any) -> SystemMessagePromptTemplate:
        """
        Create a system prompt template.
        System prompts should NEVER include user input directly.
        """
        return SystemMessagePromptTemplate.from_template(template)

    @staticmethod
    def create_chat_prompt(
        system_template: str,
        user_template: Optional[str] = None,
        **kwargs: Any,
    ) -> ChatPromptTemplate:
        """
        Create a chat prompt with separate system and user messages.
        User content is inserted via template variables, not concatenation.
        """
        messages = []

        # System message (never includes user input)
        messages.append(("system", system_template))

        # User message (if template provided)
        if user_template:
            messages.append(("user", user_template))
        else:
            # Default user template that safely includes user input
            messages.append(("user", "{user_input}"))

        return ChatPromptTemplate.from_messages(messages)

    @staticmethod
    def create_simple_prompt(template: str) -> PromptTemplate:
        """Create a simple prompt template (for non-chat models)."""
        return PromptTemplate.from_template(template)

    @staticmethod
    def format_prompt(
        template: ChatPromptTemplate | PromptTemplate,
        user_input: str,
        **kwargs: Any,
    ) -> str:
        """
        Format a prompt template with user input and other variables.
        This ensures user input is properly escaped via template substitution.
        """
        # Extract user_input from kwargs if not provided
        if "user_input" not in kwargs:
            kwargs["user_input"] = user_input

        # Format the prompt
        if isinstance(template, ChatPromptTemplate):
            # For chat prompts, format returns a list of messages
            formatted = template.format_messages(**kwargs)
            # Convert to string representation
            return "\n".join(str(msg.content) for msg in formatted)
        else:
            # For simple prompts, format returns a string
            return template.format(**kwargs)

