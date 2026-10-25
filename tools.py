"""Finance tools — fake data, no API key needed. Great for learning."""

# ---------------------------------------------------------------------------
# Fake data
# ---------------------------------------------------------------------------

PORTFOLIO = {
    "AAPL": {"shares": 10, "avg_cost": 150.00},
    "GOOGL": {"shares": 5,  "avg_cost": 2800.00},
    "TSLA": {"shares": 8,   "avg_cost": 700.00},
    "MSFT": {"shares": 15,  "avg_cost": 300.00},
    "AMZN": {"shares": 3,   "avg_cost": 3400.00},
}

PRICES = {
    "AAPL":  182.50,
    "GOOGL": 3100.00,
    "TSLA":  245.80,
    "MSFT":  415.20,
    "AMZN":  3850.00,
    "NVDA":  875.00,
    "META":  520.00,
}

NEWS = {
    "AAPL":  "Apple announces record iPhone sales in Q4.",
    "GOOGL": "Google DeepMind releases new AI model beating benchmarks.",
    "TSLA":  "Tesla misses delivery targets; stock drops 8%.",
    "MSFT":  "Microsoft Azure revenue up 28% year-over-year.",
    "AMZN":  "Amazon expands same-day delivery to 50 new cities.",
    "NVDA":  "Nvidia reports 200% revenue growth driven by AI chip demand.",
    "META":  "Meta's ad revenue rebounds strongly after privacy changes.",
}

# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def get_stock_price(args: dict) -> dict:
    """Return the current price for a ticker symbol."""
    ticker = args["ticker"].upper()
    if ticker not in PRICES:
        return {"error": f"Unknown ticker: {ticker}. Available: {list(PRICES.keys())}"}
    return {"ticker": ticker, "price": PRICES[ticker], "currency": "USD"}


def get_portfolio(args: dict) -> dict:
    """Return the user's full portfolio with current values."""
    rows = []
    total_cost = 0.0
    total_value = 0.0

    for ticker, pos in PORTFOLIO.items():
        price = PRICES.get(ticker, 0)
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
        "positions": rows,
        "total_cost":  round(total_cost, 2),
        "total_value": round(total_value, 2),
        "total_gain":  round(total_value - total_cost, 2),
    }


def get_news(args: dict) -> dict:
    """Return the latest news headline for a ticker."""
    ticker = args["ticker"].upper()
    if ticker not in NEWS:
        return {"error": f"No news for {ticker}. Available: {list(NEWS.keys())}"}
    return {"ticker": ticker, "headline": NEWS[ticker]}


def compare_stocks(args: dict) -> dict:
    """Compare two or more stocks by current price."""
    tickers = [t.upper() for t in args["tickers"]]
    result = []
    for ticker in tickers:
        if ticker in PRICES:
            result.append({"ticker": ticker, "price": PRICES[ticker]})
        else:
            result.append({"ticker": ticker, "error": "unknown ticker"})
    return {"comparison": result}


# ---------------------------------------------------------------------------
# Registry — maps tool name → (definition, handler)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "get_stock_price",
        "description": "Get the current price of a stock by ticker symbol (e.g. AAPL, TSLA).",
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
        "description": "Get the user's stock portfolio: positions, current values, and total gain/loss.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_news",
        "description": "Get the latest news headline for a stock ticker.",
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
        "description": "Compare current prices of two or more stocks side by side.",
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
