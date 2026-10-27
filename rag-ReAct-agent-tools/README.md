# ReAct Finance Agent with RAG over SEC Filings

A Claude-powered finance agent using the **ReAct pattern**, **long-term memory**, and **RAG (retrieval-augmented generation) over SEC filings**. It can answer questions from a company's own 10-K or 10-Q, such as risk factors, competition and supply chain, citing the section each point comes from.

This is step 5 of the learning roadmap, built on top of the [long-term memory agent](../longterm-memory-ReAct-agent-tools/).

## What's New vs the Long-Term Memory Agent

| Feature | Long-Term Memory Agent (step 4) | This Agent |
|---|---|---|
| Market data (yfinance) | ✅ | ✅ |
| Memory across sessions | ✅ `recall_memory` | ✅ `recall_memory` |
| Company filings | ❌ | ✅ `search_filings` over the latest 10-K / 10-Q |
| Answers cite sources | — | Form, filing date and section |

## How RAG Works Here

```
"What supply chain risks does Apple cite?"
    ↓ think() — this is in the company's own words, not market data
    ↓ search_filings(symbol="AAPL", query="supply chain risks", form="10-K")
        ↓ EDGAR: ticker → CIK → most recent 10-K               (edgar.py)
        ↓ first time only: download, convert HTML to text,
          split into Part/Item sections, chunk, embed, store    (filings_store.py → filings_db/)
        ↓ embed the query, return the 5 most similar chunks from that filing
    ↓ answer from those passages, citing "10-K filed 2025-10-31, Item 1A. Risk Factors"
```

- **Indexed once, reused everywhere.** A filing is downloaded and embedded the first time it is searched, which can take up to a minute. It is then stored in `filings_db/` and reused by every session.
- **Always the latest filing.** Each search checks EDGAR for the most recent filing of that form. A new 10-K is indexed automatically when it appears; older ones stay in the index.
- **Chunks follow the filing's structure.** Chunks are about 1,000 characters, which is what the embedding model reads in full. They overlap by about 200 characters and never cross a section boundary. Each chunk is tagged with its section, e.g. `Part I — Item 1A. Risk Factors`, and the heading is embedded with the text, so a search for "risk factors" favours risk chunks.
- **Hidden XBRL data is removed.** Modern filings embed machine-readable tags; these are stripped so only the readable document is indexed.
- **Same local embedding model** as long-term memory (`all-MiniLM-L6-v2`). No extra API key or cost.

## Project Structure

```
rag-ReAct-agent-tools/
├── agent.py          # ReAct loop + session load/save + stores each turn
├── edgar.py          # SEC EDGAR client: ticker → CIK, latest filing, HTML → text
├── filings_store.py  # Sections, chunking, indexing and search over filings
├── memory.py         # JSON session persistence
├── memory_store.py   # Long-term memory across sessions
├── tools.py          # yfinance tools + recall_memory + search_filings
├── backfill.py       # Load existing session JSON files into memory
├── app.py            # Streamlit UI: sessions, memory count, indexed filings
├── evals/            # Live eval suite, incl. memory and filing cases
├── tests/            # Unit tests (no network, offline embedding)
├── requirements.txt
├── sessions/         # Saved session files (gitignored)
├── memory_db/        # Long-term memory vector DB (gitignored)
├── filings_db/       # Filings vector DB (gitignored)
├── edgar_cache/      # Cached SEC ticker list (gitignored)
└── .env              # API key + SEC user agent (gitignored)
```

## Setup

```bash
source ../venv/bin/activate
pip install -r requirements.txt
```

`.env` needs two lines:

```
ANTHROPIC_API_KEY=your-key-here
SEC_USER_AGENT=Your Name you@example.com
```

SEC requires every EDGAR request to identify the caller with a name and email ([SEC guidance](https://www.sec.gov/os/accessing-edgar-data)). Without it, `search_filings` returns a clear error and the rest of the agent keeps working.

## Running

```bash
python agent.py          # CLI
streamlit run app.py     # Streamlit UI
```

The sidebar lists the filings indexed so far.

### Tests and evals
```bash
python -m pytest -q tests     # offline, fast
python evals/run_evals.py     # live agent; memory and filings go to temp folders
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
| `search_filings` | Search the latest 10-K or 10-Q for relevant passages |

## Model

`claude-sonnet-4-6` — change the `MODEL` constant in `agent.py` to switch.

## Known Limitations

- **US SEC filers only.** ETFs and most non-US companies (which file 20-F/40-F) have no 10-K.
- **Text only.** Financial tables become flattened rows of text, so exact figures are better fetched with the data tools.
- **Latest filing per form.** Questions about older years need a filing-date option, which is not built yet.
- **Retrieval can miss.** If the best passages do not answer the question, the agent is told to say so and to search again with different wording, not to guess.
- The current session's history is still sent in full with every request, as in step 4.

## What's Next

Step 6: **Multi-agent** — an orchestrator that delegates to specialist agents (market data, filings, memory).
