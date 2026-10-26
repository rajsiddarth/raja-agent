# ReAct Finance Agent with Conversation Memory

A finance agent that uses the **ReAct pattern** + **conversation history persistence** — Claude remembers prior turns across sessions.

## Project Structure

- `agent.py` — ReAct agent loop with think tool + memory load/save
- `memory.py` — JSON-based session persistence (load, save, list, delete)
- `tools.py` — Tool definitions and handlers (yfinance)
- `app.py` — Streamlit chat UI with session sidebar
- `requirements.txt` — Dependencies
- `sessions/` — Saved session files (gitignored)
- `.env` — API key (gitignored)

## Running the Agent

### CLI
```bash
source ../venv/bin/activate
python3 agent.py
```

### Streamlit UI
```bash
source ../venv/bin/activate
streamlit run app.py
```

## How Memory Works

```
Session start
    ↓
Load history from sessions/<name>.json
    ↓
User message → agent turn (ReAct loop)
    ↓
Save updated history to sessions/<name>.json
    ↓
Next session: history is restored, Claude has full context
```

Each session is stored as a JSON file in `sessions/`. The full message list (including tool calls and results) is saved after every turn.

## How ReAct Works

```
User message
    ↓
Claude calls think() — reasons about what data it needs
    ↓
Claude calls data tools (get_quote, get_fundamentals, etc.)
    ↓
Claude synthesizes all data into a final answer
```

## Data Tools (via yfinance)

| Tool | Data |
|---|---|
| `get_quote` | Price, change, volume, market cap |
| `get_fundamentals` | P/E, EPS, 52-week range, margins |
| `get_history` | OHLCV bars (1d to 5y) |
| `get_news` | Recent headlines |
| `search_ticker` | Find symbol by company name |
| `get_options` | Calls/puts chain |

## Model

Uses `claude-sonnet-4-6`. Change the `MODEL` constant in `agent.py` to switch.

Note: `claude-opus-5-5` refuses financial data questions — use `claude-sonnet-4-6` or later.

## Coding Conventions

- Type hints on all function parameters and return values
- `on_think` callback surfaces reasoning to the UI
- `on_tool_call` callback surfaces tool calls to the UI
- History saved after every turn — never lose a conversation
- Handle errors defensively; surface clear messages to the user

## Roadmap

1. ✅ **Reactive agent** — agent + tools (`reactive-agent-tools/`)
2. ✅ **ReAct agent** — reason + act (`ReAct-agent-tools/`)
3. ✅ **Agent + memory** — conversation history persistence ← you are here
4. **Agent + RAG** — vector search over earnings reports / filings
5. **Multi-agent** — orchestrator + specialist agents

Be formal and professional
