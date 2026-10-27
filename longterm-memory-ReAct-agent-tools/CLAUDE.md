# ReAct Finance Agent with Long-Term Memory

A finance agent that uses the **ReAct pattern** + **long-term memory** — each turn (question + final answer) is stored in a vector database and searchable from any session via the `recall_memory` tool.

## Project Structure

- `agent.py` — ReAct agent loop with think tool + session load/save + stores each turn
- `memory.py` — JSON-based session persistence (load, save, list, delete)
- `memory_store.py` — ChromaDB long-term memory (extract turns, store, search)
- `tools.py` — Tool definitions and handlers (yfinance + `recall_memory`)
- `backfill.py` — Loads existing session JSON files into long-term memory
- `app.py` — Streamlit chat UI with session sidebar
- `requirements.txt` — Dependencies
- `sessions/` — Saved session files (gitignored)
- `memory_db/` — Vector database (gitignored)
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
User message → agent turn (ReAct loop, may call recall_memory)
    ↓
Save full history to sessions/<name>.json        (archive, current session context)
    ↓
Store "Q + final answer" in memory_db/            (long-term, all sessions)
```

- Current session: full history is loaded, as in step 3.
- Past sessions: Claude calls `recall_memory(query)` to retrieve the most relevant past turns.
- Records drop think steps and raw tool results; tickers are kept as metadata.
- Record ID is `session:turn_index`, so re-storing a turn overwrites it (backfill is re-runnable).
- Embeddings: ChromaDB default (`all-MiniLM-L6-v2`, local). Tests inject an offline embedding via `tests/conftest.py`.

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
| `recall_memory` | Search past conversations across all sessions |

## Model

Uses `claude-sonnet-4-6`. Change the `MODEL` constant in `agent.py` to switch.

Note: `claude-opus-5-5` refuses financial data questions — use `claude-sonnet-4-6` or later.

## Coding Conventions

- Type hints on all function parameters and return values
- `on_think` callback surfaces reasoning to the UI
- `on_tool_call` callback surfaces tool calls to the UI
- History saved after every turn — never lose a conversation
- A long-term memory failure must never break the chat — warn and continue
- Handle errors defensively; surface clear messages to the user

## Roadmap

1. ✅ **Reactive agent** — agent + tools (`reactive-agent-tools/`)
2. ✅ **ReAct agent** — reason + act (`ReAct-agent-tools/`)
3. ✅ **Agent + memory** — conversation history persistence (`memory-convhistory-ReAct-agent-tools/`)
4. ✅ **Agent + long-term memory** — vector search over past sessions ← you are here
5. **Agent + RAG** — vector search over earnings reports / filings
6. **Multi-agent** — orchestrator + specialist agents

Be formal and professional
