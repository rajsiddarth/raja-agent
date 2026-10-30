# ReAct Finance Agent with RAG over SEC Filings

A finance agent that uses the **ReAct pattern** + **long-term memory** + **RAG over SEC filings** — `search_filings` retrieves passages from a company's 10-K/10-Q (latest, or a given fiscal year), and `recall_memory` searches past sessions.

## Project Structure

- `agent.py` — ReAct agent loop with think tool + session load/save + stores each turn
- `memory.py` — JSON-based session persistence (load, save, list, delete)
- `memory_store.py` — ChromaDB long-term memory (extract turns, store, search)
- `edgar.py` — SEC EDGAR client (ticker → CIK, find a 10-K/10-Q by recency or fiscal year, HTML → text)
- `filings_store.py` — Filing RAG: split sections, chunk, index (ChromaDB), search
- `tools.py` — Tool definitions and handlers (yfinance + `recall_memory` + `search_filings`)
- `backfill.py` — Loads existing session JSON files into long-term memory
- `app.py` — Streamlit chat UI with session sidebar
- `requirements.txt` — Dependencies
- `sessions/` — Saved session files (gitignored)
- `memory_db/` — Long-term memory vector DB (gitignored)
- `filings_db/` — Filings vector DB (gitignored)
- `edgar_cache/` — Cached SEC ticker list (gitignored)
- `.env` — `ANTHROPIC_API_KEY` and `SEC_USER_AGENT` (gitignored)

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

## How Filing RAG Works

```
search_filings(symbol, query, form="10-K", fiscal_year=None)
    ↓ filings_store.ensure_indexed — EDGAR lookup cached per (symbol, form, fiscal_year) for 24h
    ↓ edgar.find_filing — ticker → CIK → most recent filing of that form, or the one whose period ended in fiscal_year
    ↓ not indexed yet? download → html_to_text → split_sections → chunk_text → embed → filings_db/
    ↓ filings_store.search — strip_company removes the company name and ticker from the query
    ↓ query the chunks of that filing only (where accession = …), top n by cosine similarity
    ↓ return {filing: {form, filing_date, period, url}, passages: [{section, text, relevance}]}
```

- Fiscal year = year of the period end date (`reportDate`). Older filings live in extra submissions pages,
  fetched only when the scan reaches them; the scan stops once filing dates are before the requested year.
- 10-K also matches pre-2003 `10-K405`; amendments (`10-K/A`) are skipped.
- The company name comes from SEC's ticker list (`filing["company"]`, e.g. "NVIDIA CORP"). A name is stripped only where it is capitalised, and a ticker only in upper case with 2+ letters (so "revenue target" and the ticker "A" are left alone). Removing the name beat replacing it with "the Company" in the retrieval eval.
- `LOOKUP_TTL_S` (24h) bounds how long a running app can miss a newly filed 10-K/10-Q.
- Chunks: ~1,000 chars (MiniLM reads ~256 tokens), ~200 char overlap, never cross a Part/Item section.
- The section heading is embedded with each chunk and stripped from returned text.
- Chunk IDs are `accession:index`; re-indexing overwrites. One filing is indexed once, shared by all sessions.
- EDGAR needs `SEC_USER_AGENT` (name + email) and ≤10 requests/s; `edgar._get` enforces both.
- `EdgarError` messages are user-safe; the tool returns them as `{"error": ...}`.
- Tests mock HTTP and inject the offline embedding (`filings_db` fixture in `tests/conftest.py`).

## Evals

- `evals/run_evals.py` runs the live agent and checks the think step, the tools called and expected strings (it calls the Claude API).
- `evals/run_retrieval_evals.py` measures retrieval only, with no Claude calls. The gold set is `evals/filings_gold.json`, and the scoring lives in `evals/retrieval_metrics.py` (unit-tested).
- Gold cases are pinned to a filing by `fiscal_year`. Snippets are verbatim from the filing, and questions are paraphrased.
- `evals/run_answer_evals.py` runs the live agent plus an LLM judge (`evals/judge.py`, `claude-opus-5-5`, structured JSON output, `fallbacks: "default"`). Code-checked metrics are in `evals/answer_metrics.py`, and judge calibration is in `evals/calibrate_judge.py`. Each case costs ~$0.10. Gold questions are pinned to their filing by appending "(Use <SYMBOL>'s <form> for fiscal year <year>.)".
- The answer runner aborts on account errors (no credit, bad key) and refuses to save a partial run as results.
- Run the retrieval eval before and after any change to chunking or search. Use `--compare` against the latest file in `evals/results/` and save each accepted change as a new results file.

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
| `search_filings` | Passages from a 10-K/10-Q, latest or by fiscal year (SEC EDGAR) |

## Model

Uses `claude-sonnet-4-6`. Change the `MODEL` constant in `agent.py` to switch.

Note: `claude-opus-5-5` refuses financial data questions — use `claude-sonnet-4-6` or later.

## Coding Conventions

- Type hints on all function parameters and return values
- `on_think` callback surfaces reasoning to the UI
- `on_tool_call` callback surfaces tool calls to the UI
- History saved after every turn — never lose a conversation
- A long-term memory failure must never break the chat — warn and continue
- Filing answers must come from retrieved passages and cite form, filing date and section
- Handle errors defensively; surface clear messages to the user

## Roadmap

1. ✅ **Reactive agent** — agent + tools (`reactive-agent-tools/`)
2. ✅ **ReAct agent** — reason + act (`ReAct-agent-tools/`)
3. ✅ **Agent + memory** — conversation history persistence (`memory-convhistory-ReAct-agent-tools/`)
4. ✅ **Agent + long-term memory** — vector search over past sessions (`longterm-memory-ReAct-agent-tools/`)
5. ✅ **Agent + RAG** — vector search over SEC filings ← you are here
6. **Multi-agent** — orchestrator + specialist agents

Be formal and professional
