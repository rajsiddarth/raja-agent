# ReAct Finance Agent with Conversation Memory

A Claude-powered finance agent using the **ReAct pattern** + **JSON-based conversation memory** — sessions persist across restarts.

This is step 3 of the learning roadmap, built on top of the [ReAct agent](../ReAct-agent-tools/).

## What's New vs ReAct Agent

| Feature | ReAct Agent | This Agent |
|---|---|---|
| Conversation history | In-memory only | Saved to `sessions/*.json` |
| Cross-session context | ❌ Forgotten on restart | ✅ Fully restored |
| Session management | None | Load / New / Delete via sidebar |
| CLI | Single session | Named sessions, resumable |

## How Memory Works

```
Session start
    ↓
Load history from sessions/<name>.json  ← full message list restored
    ↓
User message → ReAct loop (think → tools → answer)
    ↓
Save updated history to sessions/<name>.json  ← after every turn
    ↓
Next session: Claude has full prior context
```

The entire message list — including tool calls and results — is saved as JSON after every turn. When you resume a session, Claude sees exactly what was said and what data was fetched before.

## Project Structure

```
memory-convhistory-ReAct-agent-tools/
├── agent.py          # ReAct loop + memory load/save
├── memory.py         # JSON session persistence
├── tools.py          # yfinance tool definitions and handlers
├── app.py            # Streamlit UI with session sidebar
├── requirements.txt  # Dependencies
├── sessions/         # Saved session files (gitignored)
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
# Enter a session name (or press Enter for "default")
# Type "new" to start a fresh session
# Type "quit" to save and exit
```

### Streamlit UI
```bash
source venv/bin/activate
streamlit run app.py
```

Use the sidebar to load a saved session, start a new one, or delete one.

## The memory Module

```python
from memory import load_history, save_history, list_sessions, delete_session

history = load_history("my-session")   # [] if none exists
save_history("my-session", history)    # writes sessions/my-session.json
sessions = list_sessions()             # [{name, turns, updated_at}, ...]
delete_session("my-session")           # removes the file
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

`claude-sonnet-4-6` — change the `MODEL` constant in `agent.py` to switch.

## What's Next

Step 4: **Agent + RAG** — vector search over earnings reports and filings so the agent can answer questions about documents, not just live market data.
