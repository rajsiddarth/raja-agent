"""Unit tests for tools.py — mocks yfinance, tests data transformation logic."""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools import get_quote, get_fundamentals, get_history, get_options, get_news


# ---------------------------------------------------------------------------
# get_quote
# ---------------------------------------------------------------------------

def make_ticker(info: dict):
    ticker = MagicMock()
    ticker.info = info
    ticker.history.return_value = pd.DataFrame()
    return ticker


@patch("tools.yf.Ticker")
def test_get_quote_calculates_change(mock_ticker):
    mock_ticker.return_value = make_ticker({
        "currentPrice": 150.0,
        "previousClose": 140.0,
        "longName": "Apple Inc.",
        "volume": 1000000,
        "marketCap": 2_000_000_000,
        "currency": "USD",
    })
    result = get_quote({"symbol": "AAPL"})
    assert result["price"] == 150.0
    assert result["change"] == 10.0
    assert result["change_pct"] == pytest.approx(7.14, abs=0.01)


@patch("tools.yf.Ticker")
def test_get_quote_handles_missing_price(mock_ticker):
    mock_ticker.return_value = make_ticker({})
    result = get_quote({"symbol": "AAPL"})
    assert result["price"] is None
    assert result["change"] is None
    assert result["change_pct"] is None


@patch("tools.yf.Ticker")
def test_get_quote_symbol_uppercased(mock_ticker):
    mock_ticker.return_value = make_ticker({"currentPrice": 100.0, "previousClose": 100.0})
    result = get_quote({"symbol": "aapl"})
    assert result["symbol"] == "AAPL"


# ---------------------------------------------------------------------------
# get_history
# ---------------------------------------------------------------------------

def make_hist_df():
    index = pd.to_datetime(["2024-01-01", "2024-01-02"])
    return pd.DataFrame({
        "Open":   [100.0, 102.0],
        "High":   [105.0, 107.0],
        "Low":    [99.0,  101.0],
        "Close":  [103.0, 106.0],
        "Volume": [500000, 600000],
    }, index=index)


@patch("tools.yf.Ticker")
def test_get_history_returns_rows(mock_ticker):
    ticker = MagicMock()
    ticker.history.return_value = make_hist_df()
    mock_ticker.return_value = ticker

    result = get_history({"symbol": "AAPL", "period": "5d"})
    assert len(result["data"]) == 2
    assert result["data"][0]["close"] == 103.0
    assert result["data"][1]["close"] == 106.0


@patch("tools.yf.Ticker")
def test_get_history_empty_returns_error(mock_ticker):
    ticker = MagicMock()
    ticker.history.return_value = pd.DataFrame()
    mock_ticker.return_value = ticker

    result = get_history({"symbol": "AAPL", "period": "5d"})
    assert "error" in result


# ---------------------------------------------------------------------------
# get_options
# ---------------------------------------------------------------------------

@patch("tools.yf.Ticker")
def test_get_options_no_expiration_returns_list(mock_ticker):
    ticker = MagicMock()
    ticker.options = ("2024-01-19", "2024-02-16")
    mock_ticker.return_value = ticker

    result = get_options({"symbol": "AAPL"})
    assert "expirations" in result
    assert "2024-01-19" in result["expirations"]


@patch("tools.yf.Ticker")
def test_get_options_no_options_returns_error(mock_ticker):
    ticker = MagicMock()
    ticker.options = ()
    mock_ticker.return_value = ticker

    result = get_options({"symbol": "AAPL"})
    assert "error" in result


@patch("tools.yf.Ticker")
def test_get_options_invalid_expiration_returns_error(mock_ticker):
    ticker = MagicMock()
    ticker.options = ("2024-01-19",)
    mock_ticker.return_value = ticker

    result = get_options({"symbol": "AAPL", "expiration": "2099-01-01"})
    assert "error" in result
    assert "available" in result


# ---------------------------------------------------------------------------
# get_news
# ---------------------------------------------------------------------------

@patch("tools.yf.Ticker")
def test_get_news_returns_articles(mock_ticker):
    ticker = MagicMock()
    ticker.news = [
        {"content": {"title": "Apple hits record", "provider": {"displayName": "Reuters"}, "canonicalUrl": {"url": "https://example.com"}}},
        {"content": {"title": "iPhone sales up",   "provider": {"displayName": "Bloomberg"}, "canonicalUrl": {"url": "https://example2.com"}}},
    ]
    mock_ticker.return_value = ticker

    result = get_news({"symbol": "AAPL"})
    assert len(result["articles"]) == 2
    assert result["articles"][0]["title"] == "Apple hits record"


@patch("tools.yf.Ticker")
def test_get_news_handles_no_news(mock_ticker):
    ticker = MagicMock()
    ticker.news = None
    mock_ticker.return_value = ticker

    result = get_news({"symbol": "AAPL"})
    assert result["articles"] == []
