# raja-agent-tools

A learning project for building Claude-powered AI agents with tool use.
Starts simple (agent + tools), grows toward RAG and multi-agent patterns.

## Project Structure

- `agent.py` — Claude agent loop connected to Robinhood via MCP

## Running the Agent

```bash
pip install anthropic>=2.3.0
export ANTHROPIC_API_KEY=your-key-here
python3 agent.py
```

Requires an active Robinhood session authenticated via the Claude connector.

## How the Agent Loop Works

```
User message
    ↓
Claude decides which Robinhood tool(s) to call  ← stop_reason == "tool_use"
    ↓
Robinhood MCP server executes the tool server-side
    ↓
Result returned to Claude
    ↓
Claude generates final answer                    ← stop_reason == "end_turn"
```

## Data Source

Real-time data via the **Robinhood MCP server** (`https://agent.robinhood.com`).
Claude has access to 87 Robinhood tools including:

| Category | Example tools |
|---|---|
| Quotes | `get_equity_quotes`, `get_crypto_quotes` |
| Portfolio | `get_portfolio`, `get_equity_positions` |
| Orders | `get_equity_orders`, `get_option_orders` |
| Research | `get_equity_fundamentals`, `get_earnings_calendar` |
| Watchlists | `get_watchlists`, `get_watchlist_items` |

## Model

Uses `claude-opus-5-5`. Change the `MODEL` constant in `agent.py` to switch.

## Coding Conventions

- Type hints on all function parameters and return values
- Keep the agent loop in `agent.py`; do not add local tool handlers
- MCP tools execute server-side — no local `tools.py` needed
- Handle errors defensively; surface clear messages to the user

## Roadmap

1. **Agent + MCP tools** ← you are here
2. **Agent + memory** — persist conversation history across sessions
3. **Agent + RAG** — add vector search over earnings reports / filings
4. **Multi-agent** — orchestrator + specialist agents (research, risk, news)

Be formal and professional
