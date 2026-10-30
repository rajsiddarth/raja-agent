"""Unit tests for costs.py — token totals and dollar cost."""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from costs import add_response_usage, cost_usd, format_cost, new_usage


def test_add_response_usage_sums_calls_and_ignores_missing_fields():
    total = new_usage()
    add_response_usage(total, SimpleNamespace(input_tokens=1000, output_tokens=100, cache_read_input_tokens=None))
    add_response_usage(total, SimpleNamespace(input_tokens=2000, output_tokens=50))
    assert total == {"api_calls": 2, "input_tokens": 3000, "output_tokens": 150,
                     "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


def test_cost_usd_uses_model_prices_and_cache_multipliers():
    usage = {"input_tokens": 1_000_000, "output_tokens": 100_000}
    assert cost_usd(usage, "claude-sonnet-4-6") == pytest.approx(3.00 + 1.50)
    assert cost_usd(usage, "claude-opus-5-5") == pytest.approx(4.00 + 2.00)
    cached = {"cache_creation_input_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000}
    assert cost_usd(cached, "claude-sonnet-4-6") == pytest.approx(3.00 * 1.25 + 3.00 * 0.1)
    assert cost_usd(usage, "some-unknown-model") is None


def test_format_cost():
    usage = {"api_calls": 3, "input_tokens": 15195, "output_tokens": 1134, "cost_usd": 0.0626}
    assert format_cost(usage) == "$0.063 · 3 API calls · 15,195 in / 1,134 out tokens"
    assert format_cost({"api_calls": 1, "cost_usd": None}) == "cost unknown · 1 API call · 0 in / 0 out tokens"
