# raja-agent-tools

A learning project for building Claude-powered AI agents with tool use.
Starts simple (agent + tools), grows toward RAG and multi-agent patterns.

## Project Structure

- `agent.py` — Claude agent loop with yfinance tools
- `tools.py` — Tool definitions and handlers (yfinance)
- `app.py` — Streamlit chat UI
- `requirements.txt` — Dependencies
- `.env` — API key (gitignored)

## Running the Agent

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

### First-time setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo ANTHROPIC_API_KEY=your-key-here > .env
```

## How the Agent Loop Works

```
User message
    ↓
Claude decides which tool(s) to call  ← stop_reason == "tool_use"
    ↓
Tool handler runs locally via yfinance
    ↓
Result returned to Claude
    ↓
Claude generates final answer          ← stop_reason == "end_turn"
```

## Data Source

Real-time market data via **yfinance** (Yahoo Finance). No auth required.

| Tool | Data |
|---|---|
| `get_quote` | Price, change, volume, market cap |
| `get_fundamentals` | P/E, EPS, 52-week range, margins |
| `get_history` | OHLCV bars (1d to 5y) |
| `get_news` | Recent headlines |
| `search_ticker` | Find symbol by company name |
| `get_options` | Calls/puts chain |

## Model

Uses `claude-opus-5-5`. Change the `MODEL` constant in `agent.py` to switch.

## Coding Conventions

- Type hints on all function parameters and return values
- Tool definitions in `tools.py` (TOOLS + TOOL_HANDLERS)
- Agent loop stays in `agent.py`
- Handle errors defensively; surface clear messages to the user

## Roadmap

1. **Agent + yfinance tools** ← you are here
2. **Agent + memory** — persist conversation history across sessions
3. **Agent + RAG** — add vector search over earnings reports / filings
4. **Multi-agent** — orchestrator + specialist agents (research, risk, news)

Be formal and professional
