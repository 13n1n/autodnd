"""Input sanitization for player actions."""

import re
import unicodedata
from typing import Optional

from autodnd.models.actions import ActionType


class InputSanitizer:
    """Sanitizes player input to prevent prompt injection."""

    # Special tokens that could be used for prompt injection
    DANGEROUS_TOKENS = [
        "{",
        "}",
        "<|",
        "|>",
        "[INST]",
        "[/INST]",
        "<|im_start|>",
        "<|im_end|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<<SYS>>",
        "<</SYS>>",
        "[SYSTEM]",
        "[/SYSTEM]",
        "```",
        "```python",
        "```javascript",
        "```json",
    ]

    # Maximum input length (characters)
    MAX_INPUT_LENGTH = 1000

    # Allowed action types (whitelist)
    ALLOWED_ACTION_TYPES = {action_type.value for action_type in ActionType}

    def __init__(
        self,
        max_length: int = MAX_INPUT_LENGTH,
        allowed_action_types: Optional[set[str]] = None,
    ) -> None:
        """Initialize sanitizer with configurable limits."""
        self.max_length = max_length
        self.allowed_action_types = allowed_action_types or self.ALLOWED_ACTION_TYPES

    def sanitize(self, input_text: str) -> str:
        """
        Sanitize input text by:
        1. Normalizing unicode
        2. Stripping dangerous tokens
        3. Truncating to max length
        4. Stripping whitespace
        """
        if not isinstance(input_text, str):
            raise TypeError(f"Input must be a string, got {type(input_text)}")

        # Normalize unicode (NFKC: compatibility decomposition + composition)
        normalized = unicodedata.normalize("NFKC", input_text)

        # Strip dangerous tokens
        sanitized = normalized
        for token in self.DANGEROUS_TOKENS:
            sanitized = sanitized.replace(token, "")

        # Remove any remaining control characters except newlines and tabs
        sanitized = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", sanitized)

        # Truncate to max length
        if len(sanitized) > self.max_length:
            sanitized = sanitized[: self.max_length]

        # Strip leading/trailing whitespace
        sanitized = sanitized.strip()

        return sanitized

    def validate_action_type(self, action_type: str) -> bool:
        """Validate that action type is in whitelist."""
        return action_type in self.allowed_action_types

    def is_safe(self, input_text: str) -> tuple[bool, Optional[str]]:
        """
        Check if input is safe.
        Returns (is_safe, error_message).
        """
        if not input_text:
            return False, "Input is empty"

        if len(input_text) > self.max_length:
            return False, f"Input exceeds maximum length of {self.max_length} characters"

        # Check for dangerous patterns
        for token in self.DANGEROUS_TOKENS:
            if token in input_text:
                return False, f"Input contains forbidden token: {token}"

        # Check for suspicious patterns (multiple consecutive special chars)
        if re.search(r"[{}]{3,}", input_text):
            return False, "Input contains suspicious pattern"

        return True, None

