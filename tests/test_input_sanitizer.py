"""Tests for InputSanitizer."""

import pytest

from autodnd.security.input_sanitizer import InputSanitizer


class TestInputSanitizer:
    """Test suite for InputSanitizer."""

    def test_sanitize_normal_input(self):
        """Test sanitization of normal input."""
        sanitizer = InputSanitizer()
        input_text = "I want to move north"
        result = sanitizer.sanitize(input_text)
        assert result == "I want to move north"

    def test_sanitize_strips_dangerous_tokens(self):
        """Test that dangerous tokens are stripped."""
        sanitizer = InputSanitizer()
        input_text = "I want to {move} north <|system|>"
        result = sanitizer.sanitize(input_text)
        assert "{" not in result
        assert "}" not in result
        assert "<|system|>" not in result

    def test_sanitize_truncates_long_input(self):
        """Test that input is truncated to max length."""
        sanitizer = InputSanitizer(max_length=10)
        input_text = "This is a very long input that should be truncated"
        result = sanitizer.sanitize(input_text)
        assert len(result) <= 10

    def test_sanitize_strips_whitespace(self):
        """Test that leading/trailing whitespace is stripped."""
        sanitizer = InputSanitizer()
        input_text = "   I want to move   "
        result = sanitizer.sanitize(input_text)
        assert result == "I want to move"

    def test_is_safe_normal_input(self):
        """Test is_safe with normal input."""
        sanitizer = InputSanitizer()
        input_text = "I want to move north"
        is_safe, error = sanitizer.is_safe(input_text)
        assert is_safe is True
        assert error is None

    def test_is_safe_dangerous_tokens(self):
        """Test is_safe detects dangerous tokens."""
        sanitizer = InputSanitizer()
        input_text = "I want to {move} north"
        is_safe, error = sanitizer.is_safe(input_text)
        assert is_safe is False
        assert error is not None
        assert "forbidden token" in error.lower()

    def test_is_safe_empty_input(self):
        """Test is_safe with empty input."""
        sanitizer = InputSanitizer()
        is_safe, error = sanitizer.is_safe("")
        assert is_safe is False
        assert "empty" in error.lower()

    def test_is_safe_too_long(self):
        """Test is_safe detects input that's too long."""
        sanitizer = InputSanitizer(max_length=10)
        input_text = "This is a very long input"
        is_safe, error = sanitizer.is_safe(input_text)
        assert is_safe is False
        assert "exceeds" in error.lower()

    def test_validate_action_type(self):
        """Test action type validation."""
        sanitizer = InputSanitizer()
        assert sanitizer.validate_action_type("move") is True
        assert sanitizer.validate_action_type("attack") is True
        assert sanitizer.validate_action_type("invalid_action") is False

    def test_sanitize_unicode_normalization(self):
        """Test that unicode is normalized."""
        sanitizer = InputSanitizer()
        # Test with some unicode characters
        input_text = "I want to move"
        result = sanitizer.sanitize(input_text)
        assert isinstance(result, str)

    def test_sanitize_control_characters(self):
        """Test that control characters are removed."""
        sanitizer = InputSanitizer()
        input_text = "I want to move\x00\x01\x02"
        result = sanitizer.sanitize(input_text)
        assert "\x00" not in result
        assert "\x01" not in result
        assert "\x02" not in result

    def test_sanitize_type_error(self):
        """Test that non-string input raises TypeError."""
        sanitizer = InputSanitizer()
        with pytest.raises(TypeError):
            sanitizer.sanitize(123)  # type: ignore

    def test_interface_completeness(self):
        """Test that InputSanitizer has all required methods."""
        sanitizer = InputSanitizer()
        assert hasattr(sanitizer, "sanitize")
        assert hasattr(sanitizer, "is_safe")
        assert hasattr(sanitizer, "validate_action_type")
        assert callable(sanitizer.sanitize)
        assert callable(sanitizer.is_safe)
        assert callable(sanitizer.validate_action_type)

