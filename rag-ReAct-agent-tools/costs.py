"""Token usage and dollar cost of Claude API calls.

One agent turn makes several API calls (think, tools, final answer), and each call resends the whole
conversation so far, so a turn's cost is the sum over its calls. Embeddings (local MiniLM), SEC EDGAR and
yfinance are free; only Claude calls cost money.

Prices are list prices in $ per million tokens; check https://platform.claude.com/docs/en/about-claude/pricing
when switching models.
"""

# model: ($ per million input tokens, $ per million output tokens)
PRICES = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-5-5": (4.00, 20.00),
}
# Prompt caching: writing the cache costs 1.25x the input price (5-minute TTL), reading it 0.1x
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.1

USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


def new_usage() -> dict:
    """An empty usage total: API call count plus each token field."""
    return {"api_calls": 0, **{field: 0 for field in USAGE_FIELDS}}


def add_response_usage(total: dict, usage) -> None:
    """Add one API response's `usage` to a running total (missing or non-integer fields count as 0)."""
    total["api_calls"] += 1
    for field in USAGE_FIELDS:
        value = getattr(usage, field, None)
        total[field] += value if isinstance(value, int) else 0


def cost_usd(usage: dict, model: str) -> float | None:
    """Dollar cost of a usage total for a model, or None if the model's price is unknown."""
    if model not in PRICES:
        return None
    input_price, output_price = PRICES[model]
    return (
        usage.get("input_tokens", 0) * input_price
        + usage.get("cache_creation_input_tokens", 0) * input_price * CACHE_WRITE_MULTIPLIER
        + usage.get("cache_read_input_tokens", 0) * input_price * CACHE_READ_MULTIPLIER
        + usage.get("output_tokens", 0) * output_price
    ) / 1_000_000


def format_cost(usage: dict) -> str:
    """One line for the UI, e.g. '$0.048 · 3 API calls · 15,195 in / 1,134 out tokens'."""
    cost = usage.get("cost_usd")
    price = f"${cost:.3f}" if cost is not None else "cost unknown"
    calls = usage.get("api_calls", 0)
    line = (
        f"{price} · {calls} API call{'s' if calls != 1 else ''} · "
        f"{usage.get('input_tokens', 0):,} in / {usage.get('output_tokens', 0):,} out tokens"
    )
    cached = usage.get("cache_read_input_tokens", 0)
    return line + (f" · {cached:,} cached" if cached else "")
