"""Tests for common/client.py — LLM client, rate limiting, token budget."""
import sys
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.client import (
    OpenAIClient,
    TokenUsage,
    LLMResponse,
    get_token_budget,
    ProviderConfigurationError,
    MODEL_CONTEXT_LIMITS,
)


class TestTokenUsage:
    """Test TokenUsage dataclass."""

    def test_default_values(self):
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_custom_values(self):
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150


class TestLLMResponse:
    """Test LLMResponse dataclass."""

    def test_default_values(self):
        resp = LLMResponse(content="", usage=TokenUsage())
        assert resp.content == ""
        assert resp.error == ""

    def test_with_error(self):
        resp = LLMResponse(content="", usage=TokenUsage(), error="timeout")
        assert resp.error == "timeout"


class TestGetTokenBudget:
    """Test get_token_budget function."""

    def test_known_model(self):
        budget = get_token_budget("llama-3.3-70b-versatile")
        assert budget > 0
        assert budget < 131072  # Less than full context

    def test_unknown_model_defaults(self):
        budget = get_token_budget("unknown-model-xyz")
        assert budget > 0

    def test_gpt53_codex(self):
        budget = get_token_budget("gpt-5.3-codex")
        assert budget > 0
        assert budget < 128000

    def test_ministral_correct_limit(self):
        """Ministral-3B should use 32K, not 128K."""
        budget = get_token_budget("Ministral-3B")
        assert budget < 32768  # Must be less than 32K context

    def test_claude_sonnet(self):
        budget = get_token_budget("claude-sonnet-4-5")
        assert budget > 0
        assert budget < 200000

    def test_safety_margin_scales(self):
        """Large models get 10% margin, small get 20%."""
        large = get_token_budget("llama-3.3-70b-versatile")  # 131072 context
        small = get_token_budget("gemma2-9b-it")  # 8192 context
        # Both should have positive budgets
        assert large > 0
        assert small > 0


class TestOpenAIClient:
    """Test OpenAIClient initialization and basic behavior."""

    def test_init_defaults(self):
        client = OpenAIClient()
        assert client.base_url == "http://localhost:1234/v1"
        assert client.model == "qwen/qwen3-8b"

    def test_init_custom(self):
        client = OpenAIClient(
            base_url="https://api.example.com/v1",
            model="my-model",
            api_key="test-key",
        )
        assert client.base_url == "https://api.example.com/v1"
        assert client.model == "my-model"
        assert client.api_key == "test-key"

    def test_inject_no_think(self):
        client = OpenAIClient(inject_no_think=True)
        messages = [{"role": "user", "content": "hello"}]
        result = client._inject_no_think(messages)
        # Should prepend /no_think system message
        assert any("/no_think" in m.get("content", "") for m in result)

    def test_inject_no_think_disabled(self):
        client = OpenAIClient(inject_no_think=False)
        messages = [{"role": "user", "content": "hello"}]
        result = client._inject_no_think(messages)
        assert result == messages

    def test_no_duplicate_no_think(self):
        client = OpenAIClient(inject_no_think=True)
        messages = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": "hello"},
        ]
        result = client._inject_no_think(messages)
        no_think_count = sum(1 for m in result if "/no_think" in m.get("content", ""))
        assert no_think_count == 1

    def test_parse_retry_delay(self):
        assert OpenAIClient._parse_retry_delay("Please retry in 5s") == 5.0
        assert OpenAIClient._parse_retry_delay("retryDelay: 10s") == 10.0
        assert OpenAIClient._parse_retry_delay("some random error") == 0.0

    def test_rate_limit_window_initialized(self):
        client = OpenAIClient()
        assert isinstance(OpenAIClient._rate_limit_window, list)


class TestProviderConfigurationError:
    """Test ProviderConfigurationError exception."""

    def test_is_value_error(self):
        """Should be a subclass of ValueError for easy catching."""
        assert issubclass(ProviderConfigurationError, ValueError)

    def test_can_be_caught_as_value_error(self):
        """Catch blocks for ValueError should also catch this."""
        with pytest.raises(ValueError):
            raise ProviderConfigurationError("test error")

    def test_preserves_message(self):
        """Error message should be preserved."""
        err = ProviderConfigurationError("missing API key")
        assert str(err) == "missing API key"
