"""Validates agent outputs against Pydantic schemas."""

import logging
import re
from typing import Any, Optional, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class OutputValidator:
    """Validates agent outputs against Pydantic schemas."""

    def __init__(self) -> None:
        """Initialize output validator."""
        self.suspicious_patterns: list[str] = []

    def validate(
        self,
        output: Any,
        schema: type[T],
        strict: bool = True,
    ) -> tuple[bool, Optional[T], Optional[str]]:
        """
        Validate output against Pydantic schema.
        Returns (is_valid, parsed_output, error_message).
        """
        try:
            if strict and not isinstance(output, dict):
                # Try to parse as JSON if it's a string
                if isinstance(output, str):
                    import json

                    try:
                        output = json.loads(output)
                    except json.JSONDecodeError:
                        return False, None, f"Output is not valid JSON: {output[:100]}"

            # Validate against schema
            parsed = schema.model_validate(output)
            return True, parsed, None

        except ValidationError as e:
            error_msg = f"Validation failed: {e.errors()}"
            logger.warning(f"Output validation failed: {error_msg}")
            return False, None, error_msg

        except Exception as e:
            error_msg = f"Unexpected error during validation: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, None, error_msg

    def check_suspicious_patterns(self, output: str) -> list[str]:
        """
        Check output for suspicious patterns that might indicate prompt injection.
        Returns list of detected patterns.
        """
        detected: list[str] = []

        # Check for common injection patterns
        suspicious = [
            (r"ignore\s+(previous|all|above)", "Ignore instruction pattern"),
            (r"system\s*:\s*", "System instruction pattern"),
            (r"assistant\s*:\s*", "Assistant instruction pattern"),
            (r"<\|.*?\|>", "Special token pattern"),
            (r"```.*?```", "Code block pattern"),
        ]

        for pattern, description in suspicious:
            if re.search(pattern, output, re.IGNORECASE):
                detected.append(description)
                if description not in self.suspicious_patterns:
                    self.suspicious_patterns.append(description)
                    logger.warning(f"Suspicious pattern detected: {description}")

        return detected

    def log_suspicious_activity(self, output: str, context: Optional[dict[str, Any]] = None) -> None:
        """Log suspicious activity for monitoring."""
        patterns = self.check_suspicious_patterns(output)
        if patterns:
            logger.warning(
                f"Suspicious patterns detected: {patterns}",
                extra={"output": output[:200], "context": context, "patterns": patterns},
            )

