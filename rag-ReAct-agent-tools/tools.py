"""Finance tools powered by yfinance, recall_memory for long-term memory, and search_filings for RAG over SEC filings."""

import yfinance as yf
import edgar
import filings_store
import memory_store

# ---------------------------------------------------------------------------
# Tool definitions (what Claude sees)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "get_quote",
        "description": "Get the current price and basic market data for a stock or ETF.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol (e.g. AAPL, TSLA, SPY)."}
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_fundamentals",
        "description": "Get fundamental data for a stock: market cap, P/E ratio, 52-week range, dividend yield, sector, and more.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol."}
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_history",
        "description": "Get historical OHLCV price data for a stock.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol."},
                "period": {
                    "type": "string",
                    "description": "Time period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y.",
                    "enum": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
                },
            },
            "required": ["symbol", "period"],
        },
    },
    {
        "name": "get_news",
        "description": "Get recent news headlines for a stock.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol."}
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "search_ticker",
        "description": "Search for a ticker symbol by company name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Company name to search for (e.g. 'Apple', 'Tesla')."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_options",
        "description": "Get the options chain (calls and puts) for a stock on a given expiration date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol."},
                "expiration": {
                    "type": "string",
                    "description": "Expiration date in YYYY-MM-DD format. Leave blank to get available expiration dates.",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "recall_memory",
        "description": (
            "Search long-term memory of past conversations with this user, across all sessions. "
            "Returns the most relevant past questions and answers, with session name and date. "
            "Use when the user refers to something discussed before (e.g. 'my concern about AMD', "
            "'what did we find last time') or when earlier context would help. "
            "Do not use for fresh market data — call the data tools for that."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, e.g. 'AMD valuation concern'."},
                "n_results": {
                    "type": "integer",
                    "description": "Number of past turns to return (1-10, default 5).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_filings",
        "description": (
            "Search the company's most recent SEC filing (10-K annual report or 10-Q quarterly report) "
            "and return the most relevant passages, with section names and filing date. "
            "Use for questions answered in the company's own words: risk factors, business description, "
            "segments, competition, supply chain, legal proceedings, management's discussion (MD&A), "
            "guidance commentary, accounting policies. Not for prices or ratios — use the data tools for those. "
            "The first search of a filing downloads and indexes it, which can take up to a minute."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol of a US-listed company (e.g. AAPL)."},
                "query": {
                    "type": "string",
                    "description": "What to look for, phrased like the filing would, e.g. 'supply chain concentration risk'.",
                },
                "form": {
                    "type": "string",
                    "enum": ["10-K", "10-Q"],
                    "description": "10-K for the annual report (default, most detailed), 10-Q for the latest quarter.",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of passages to return (1-8, default 5).",
                },
            },
            "required": ["symbol", "query"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def get_quote(args: dict) -> dict:
    """Return current price and market data."""
    ticker = yf.Ticker(args["symbol"].upper())
    info = ticker.info
    hist = ticker.history(period="2d")

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
    change = round(price - prev_close, 2) if price and prev_close else None
    pct = round((change / prev_close) * 100, 2) if change and prev_close else None

    return {
        "symbol": args["symbol"].upper(),
        "name": info.get("longName") or info.get("shortName"),
        "price": price,
        "change": change,
        "change_pct": pct,
        "volume": info.get("volume") or info.get("regularMarketVolume"),
        "market_cap": info.get("marketCap"),
        "currency": info.get("currency", "USD"),
    }


def get_fundamentals(args: dict) -> dict:
    """Return fundamental data."""
    info = yf.Ticker(args["symbol"].upper()).info
    return {
        "symbol": args["symbol"].upper(),
        "name": info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "eps": info.get("trailingEps"),
        "dividend_yield": info.get("dividendYield"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "50d_avg": info.get("fiftyDayAverage"),
        "200d_avg": info.get("twoHundredDayAverage"),
        "revenue": info.get("totalRevenue"),
        "profit_margin": info.get("profitMargins"),
        "beta": info.get("beta"),
        "description": (info.get("longBusinessSummary") or "")[:400],
    }


def get_history(args: dict) -> dict:
    """Return OHLCV history."""
    hist = yf.Ticker(args["symbol"].upper()).history(period=args["period"])
    if hist.empty:
        return {"error": "No historical data found."}
    rows = [
        {
            "date": str(idx.date()),
            "open": round(row["Open"], 2),
            "high": round(row["High"], 2),
            "low": round(row["Low"], 2),
            "close": round(row["Close"], 2),
            "volume": int(row["Volume"]),
        }
        for idx, row in hist.iterrows()
    ]
    return {"symbol": args["symbol"].upper(), "period": args["period"], "data": rows}


def get_news(args: dict) -> dict:
    """Return recent news headlines."""
    ticker = yf.Ticker(args["symbol"].upper())
    articles = ticker.news or []
    return {
        "symbol": args["symbol"].upper(),
        "articles": [
            {
                "title": a.get("content", {}).get("title") or a.get("title"),
                "publisher": a.get("content", {}).get("provider", {}).get("displayName") or a.get("publisher"),
                "url": (a.get("content", {}).get("canonicalUrl") or {}).get("url") or a.get("link"),
            }
            for a in articles[:8]
        ],
    }


def search_ticker(args: dict) -> dict:
    """Search for a ticker by company name."""
    results = yf.Search(args["query"]).quotes
    return {
        "results": [
            {
                "symbol": r.get("symbol"),
                "name": r.get("longname") or r.get("shortname"),
                "type": r.get("quoteType"),
                "exchange": r.get("exchDisp"),
            }
            for r in (results or [])[:5]
        ]
    }


def get_options(args: dict) -> dict:
    """Return options chain or available expiration dates."""
    ticker = yf.Ticker(args["symbol"].upper())
    expirations = ticker.options
    if not expirations:
        return {"error": "No options data found."}

    expiration = args.get("expiration")
    if not expiration:
        return {"symbol": args["symbol"].upper(), "expirations": list(expirations)}

    if expiration not in expirations:
        return {"error": f"Expiration {expiration} not found.", "available": list(expirations)}

    chain = ticker.option_chain(expiration)
    def fmt(df):
        return df[["strike", "lastPrice", "bid", "ask", "impliedVolatility", "openInterest"]].head(10).to_dict("records")

    return {
        "symbol": args["symbol"].upper(),
        "expiration": expiration,
        "calls": fmt(chain.calls),
        "puts": fmt(chain.puts),
    }


def recall_memory(args: dict) -> dict:
    """Return the past turns most relevant to the query, from any session."""
    n_results = max(1, min(int(args.get("n_results") or 5), 10))
    memories = memory_store.search_memory(args["query"], n_results)
    if not memories:
        return {"memories": [], "note": "No past conversations stored yet."}
    return {"memories": memories}


def search_filings(args: dict) -> dict:
    """Return the passages of the latest 10-K/10-Q most relevant to the query."""
    form = args.get("form") if args.get("form") in edgar.SUPPORTED_FORMS else "10-K"
    n_results = max(1, min(int(args.get("n_results") or 5), 8))
    try:
        filing = filings_store.ensure_latest_indexed(args["symbol"], form)
    except edgar.EdgarError as exc:
        return {"error": str(exc)}
    return {
        "filing": {
            "symbol": filing["symbol"],
            "form": filing["form"],
            "filing_date": filing["filing_date"],
            "period": filing["period"],
            "url": filing["url"],
        },
        "passages": filings_store.search_filing(args["query"], filing["accession"], n_results),
    }


TOOL_HANDLERS = {
    "get_quote": get_quote,
    "get_fundamentals": get_fundamentals,
    "get_history": get_history,
    "get_news": get_news,
    "search_ticker": search_ticker,
    "get_options": get_options,
    "recall_memory": recall_memory,
    "search_filings": search_filings,
}
