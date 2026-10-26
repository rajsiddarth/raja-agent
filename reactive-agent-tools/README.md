# Reactive Finance Agent

A Claude-powered finance agent that uses **tool use** to fetch real-time market data via yfinance.

This is step 1 of the learning roadmap — a simple agent loop with no planning or reasoning step.

## How It Works

```
User message
    ↓
Claude decides which tool(s) to call   ← stop_reason == "tool_use"
    ↓
Tool handler runs locally via yfinance
    ↓
Result returned to Claude
    ↓
Claude generates final answer           ← stop_reason == "end_turn"
```

Claude picks tools based on the question. For "What is Apple's price?" it calls `get_quote`. For "Compare Apple and Microsoft" it calls `get_fundamentals` twice. No explicit planning step — Claude reacts directly to the question.

## Project Structure

```
reactive-agent-tools/
├── agent.py          # Agent loop
├── tools.py          # Tool definitions and yfinance handlers
├── app.py            # Streamlit chat UI
├── requirements.txt  # Dependencies
├── tests/
│   ├── test_agent.py # Agent loop tests
│   └── test_tools.py # Tool handler tests
└── .env              # API key (gitignored)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo ANTHROPIC_API_KEY=your-key-here > .env
```

## Running

### CLI
```bash
source venv/bin/activate
python3 agent.py
```

### Streamlit UI
```bash
source venv/bin/activate
streamlit run app.py
```

### Tests
```bash
source venv/bin/activate
pytest tests/ -v
```

## Available Tools

| Tool | What it fetches |
|---|---|
| `get_quote` | Price, change %, volume, market cap |
| `get_fundamentals` | P/E, EPS, 52-week range, profit margins |
| `get_history` | OHLCV bars (1d to 5y) |
| `get_news` | Recent headlines |
| `search_ticker` | Find ticker by company name |
| `get_options` | Calls/puts chain |

Data source: **yfinance** (Yahoo Finance) — no API key required.

## Model

`claude-opus-5-5` — change the `MODEL` constant in `agent.py` to switch.

## What's Next

See [`ReAct-agent-tools/`](../ReAct-agent-tools/) — adds a `think` tool so Claude reasons step-by-step before calling data tools.
