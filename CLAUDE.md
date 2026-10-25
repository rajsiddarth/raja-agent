# raja-agent-tools

A learning project for building Claude-powered AI agents with tool use.
Starts simple (agent + tools), grows toward RAG and multi-agent patterns.

## Project Structure

- `agent.py` — Claude agent loop (manual `while stop_reason == "tool_use"` pattern)
- `tools.py` — Tool definitions (Anthropic schema format) + handler functions + fake finance data

## Running the Agent

```bash
pip install anthropic
export ANTHROPIC_API_KEY=your-key-here
python3 agent.py
```

## How the Agent Loop Works

```
User message
    ↓
Claude decides which tool(s) to call  ← stop_reason == "tool_use"
    ↓
We execute the tool functions locally
    ↓
Send results back to Claude
    ↓
Claude generates final answer          ← stop_reason == "end_turn"
```

## Available Tools

| Tool | Description |
|---|---|
| `get_stock_price` | Current price for a ticker symbol |
| `get_portfolio` | Full portfolio with gain/loss per position |
| `get_news` | Latest headline for a ticker |
| `compare_stocks` | Side-by-side price comparison for multiple tickers |

> **Note:** All finance data is currently fake/hardcoded in `tools.py`. Next step is wiring in `yfinance` for real prices.

## Model

Uses `claude-opus-5-5`. Change the `MODEL` constant in `agent.py` to switch.

## Coding Conventions

- Type hints on all function parameters and return values
- Tool handler functions take a single `args: dict` and return a `dict`
- Tool definitions follow the Anthropic JSON schema format (`name`, `description`, `input_schema`)
- Keep tool functions pure — no side effects, no global state mutation
- Handle missing/unknown inputs defensively and return `{"error": "..."}` rather than raising

## Roadmap

1. **Agent + tools** ← you are here
2. **Agent + memory** — add conversation history persistence
3. **Agent + RAG** — add vector search over finance docs
4. **Multi-agent** — orchestrator + specialist agents
