# ReAct Finance Agent with Long-Term Memory

A Claude-powered finance agent using the **ReAct pattern** + **long-term memory in a vector database**. Past conversations are searchable from any session, so a new session does not start from nothing.

This is step 4 of the learning roadmap, built on top of the [memory agent](../memory-convhistory-ReAct-agent-tools/).

## What's New vs the Memory Agent

| Feature | Memory Agent (step 3) | This Agent |
|---|---|---|
| Current session | Full history loaded | Full history loaded (unchanged) |
| Other sessions | ❌ Invisible unless you load them by name | ✅ Searchable via `recall_memory` |
| What is remembered long-term | Whole transcript (JSON) | Question + final answer per turn (vector DB) |
| Tokens for past context | Every past message, every request | Only the few relevant turns, only when needed |

## How Long-Term Memory Works

```
Turn finishes
    ↓ save full transcript to sessions/<name>.json     (archive — as in step 3)
    ↓ store "Q + final answer" in memory_db/            (long-term memory)

Later, in any session
    ↓ user: "What was my concern about AMD?"
    ↓ think() — the user refers to an earlier conversation
    ↓ recall_memory("AMD concern")  → most similar past turns, with session + date
    ↓ answer (and fetch fresh data if current values are needed)
```

- **One record per turn.** `think` steps and raw tool results are dropped; the final answer already holds the key figures. Tickers looked up during the turn are kept as metadata.
- **Embeddings run locally** (`all-MiniLM-L6-v2` via ChromaDB, ~80 MB downloaded to `~/.cache/chroma` on first use). No extra API key.
- **Claude decides when to recall.** `recall_memory` is a tool, so plain lookups like "AAPL price" cost no extra tokens.
- **Re-storing is safe.** Each record's ID is `session:turn_index`, so saving the same turn again overwrites it.

## Project Structure

```
longterm-memory-ReAct-agent-tools/
├── agent.py          # ReAct loop + session load/save + store each turn
├── memory.py         # JSON session persistence (unchanged from step 3)
├── memory_store.py   # Vector memory: extract turns, store, search
├── tools.py          # yfinance tools + recall_memory
├── backfill.py       # Load existing session JSON files into memory
├── app.py            # Streamlit UI with session sidebar + memory count
├── evals/            # Live eval suite, incl. a cross-session recall case
├── tests/            # Unit tests (offline embedding, no model download)
├── requirements.txt
├── sessions/         # Saved session files (gitignored)
├── memory_db/        # Vector database (gitignored)
└── .env              # API key (gitignored)
```

## Setup

```bash
source ../venv/bin/activate
pip install -r requirements.txt
echo ANTHROPIC_API_KEY=your-key-here > .env
```

To bring in the sessions you already have from step 3:

```bash
python backfill.py ../memory-convhistory-ReAct-agent-tools/sessions
```

## Running

### CLI
```bash
python agent.py
```

### Streamlit UI
```bash
streamlit run app.py
```

The sidebar shows how many turns long-term memory holds across all sessions.

### Tests and evals
```bash
python -m pytest -q tests     # offline, fast
python evals/run_evals.py     # live agent; memory goes to a temp folder
```

## The memory_store Module

```python
from memory_store import remember_last_turn, search_memory, extract_turns, memory_count

remember_last_turn("my-session", history)   # store the latest completed turn
search_memory("AMD valuation", n_results=5) # [{session, timestamp, tickers, text, relevance}, ...]
extract_turns(history)                      # [{question, answer, tickers}, ...]
memory_count()                              # turns stored across all sessions
```

## Available Tools

| Tool | What it does |
|---|---|
| `get_quote` | Price, change %, volume, market cap |
| `get_fundamentals` | P/E, EPS, 52-week range, profit margins |
| `get_history` | OHLCV bars (1d to 5y) |
| `get_news` | Recent headlines |
| `search_ticker` | Find ticker by company name |
| `get_options` | Calls/puts chain |
| `recall_memory` | Search past conversations across all sessions |

## Model

`claude-sonnet-4-6` — change the `MODEL` constant in `agent.py` to switch.

## Known Limitations

- The **current** session's history is still loaded in full, so a very long single session still grows in cost. The next improvement is to send only the last few turns and rely on `recall_memory` for older ones.
- Recalled figures are historical; the agent is instructed to say when they were recorded and fetch fresh data when needed.
- Deleting a session in the sidebar removes its JSON file but not its memory records.

## What's Next

Step 5: **Agent + RAG over filings** — reuse `memory_store`'s approach with a second collection holding chunks of 10-K/10-Q filings.
