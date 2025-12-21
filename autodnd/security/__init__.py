"""Security and input sanitization module for AutoDnD."""

from autodnd.security.input_sanitizer import InputSanitizer
from autodnd.security.output_validator import OutputValidator
from autodnd.security.prompt_builder import PromptBuilder

# Lazy imports to avoid circular dependencies
__all__ = [
    "InputSanitizer",
    "PromptBuilder",
    "OutputValidator",
    "SecurityAgent",
    "SecurityValidationResult",
]


def __getattr__(name: str):
    """Lazy import for SecurityAgent to avoid circular imports."""
    if name == "SecurityAgent":
        from autodnd.security.security_agent import SecurityAgent
        return SecurityAgent
    if name == "SecurityValidationResult":
        from autodnd.security.security_agent import SecurityValidationResult
        return SecurityValidationResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
