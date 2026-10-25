"""Finance tools — real-time data via yfinance."""

import yfinance as yf

# ---------------------------------------------------------------------------
# Portfolio (your positions — edit these to match your real holdings)
# ---------------------------------------------------------------------------

PORTFOLIO = {
    "AAPL": {"shares": 10, "avg_cost": 150.00},
    "GOOGL": {"shares": 5,  "avg_cost": 2800.00},
    "TSLA": {"shares": 8,   "avg_cost": 700.00},
    "MSFT": {"shares": 15,  "avg_cost": 300.00},
    "AMZN": {"shares": 3,   "avg_cost": 3400.00},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_price(ticker: str) -> float | None:
    """Return the latest price for a ticker, or None if unavailable."""
    try:
        info = yf.Ticker(ticker).fast_info
        return round(float(info["last_price"]), 2)
    except Exception:
        return None


def _fetch_news(ticker: str) -> list[dict]:
    """Return up to 3 recent news items for a ticker."""
    try:
        items = yf.Ticker(ticker).news or []
        return [
            {"title": n.get("content", {}).get("title", ""), "url": n.get("content", {}).get("canonicalUrl", {}).get("url", "")}
            for n in items[:3]
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def get_stock_price(args: dict) -> dict:
    """Return the current real-time price for a ticker symbol."""
    ticker = args["ticker"].upper()
    price = _fetch_price(ticker)
    if price is None:
        return {"error": f"Could not fetch price for {ticker}. Check the ticker symbol."}
    return {"ticker": ticker, "price": price, "currency": "USD"}


def get_portfolio(args: dict) -> dict:
    """Return the user's portfolio with real-time values and gain/loss."""
    rows = []
    total_cost = 0.0
    total_value = 0.0

    for ticker, pos in PORTFOLIO.items():
        price = _fetch_price(ticker)
        if price is None:
            rows.append({"ticker": ticker, "error": "price unavailable"})
            continue

        cost = pos["shares"] * pos["avg_cost"]
        value = pos["shares"] * price
        gain = value - cost
        gain_pct = (gain / cost * 100) if cost else 0

        rows.append({
            "ticker":    ticker,
            "shares":    pos["shares"],
            "avg_cost":  pos["avg_cost"],
            "price":     price,
            "value":     round(value, 2),
            "gain_loss": round(gain, 2),
            "gain_pct":  round(gain_pct, 2),
        })
        total_cost  += cost
        total_value += value

    return {
        "positions":   rows,
        "total_cost":  round(total_cost, 2),
        "total_value": round(total_value, 2),
        "total_gain":  round(total_value - total_cost, 2),
    }


def get_news(args: dict) -> dict:
    """Return recent news headlines for a ticker."""
    ticker = args["ticker"].upper()
    articles = _fetch_news(ticker)
    if not articles:
        return {"error": f"No news found for {ticker}."}
    return {"ticker": ticker, "news": articles}


def compare_stocks(args: dict) -> dict:
    """Compare real-time prices of two or more stocks side by side."""
    tickers = [t.upper() for t in args["tickers"]]
    result = []
    for ticker in tickers:
        price = _fetch_price(ticker)
        if price is not None:
            result.append({"ticker": ticker, "price": price})
        else:
            result.append({"ticker": ticker, "error": "price unavailable"})
    return {"comparison": result}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "get_stock_price",
        "description": "Get the current real-time price of a stock by ticker symbol (e.g. AAPL, TSLA, MSFT).",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. 'AAPL'."}
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_portfolio",
        "description": "Get the user's stock portfolio with real-time prices, current values, and total gain/loss.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_news",
        "description": "Get recent news headlines for a stock ticker.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol."}
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "compare_stocks",
        "description": "Compare current real-time prices of two or more stocks side by side.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ticker symbols to compare, e.g. ['AAPL', 'MSFT'].",
                }
            },
            "required": ["tickers"],
        },
    },
]

TOOL_HANDLERS = {
    "get_stock_price": get_stock_price,
    "get_portfolio":   get_portfolio,
    "get_news":        get_news,
    "compare_stocks":  compare_stocks,
}
