# ReAct Finance Agent with RAG over SEC Filings

A Claude-powered finance agent using the **ReAct pattern**, **long-term memory**, and **RAG (retrieval-augmented generation) over SEC filings**. It can answer questions from a company's own 10-K or 10-Q, such as risk factors, competition and supply chain, citing the section each point comes from.

This is step 5 of the learning roadmap, built on top of the [long-term memory agent](../longterm-memory-ReAct-agent-tools/).

## What's New vs the Long-Term Memory Agent

| Feature | Long-Term Memory Agent (step 4) | This Agent |
|---|---|---|
| Market data (yfinance) | ✅ | ✅ |
| Memory across sessions | ✅ `recall_memory` | ✅ `recall_memory` |
| Company filings | ❌ | ✅ `search_filings` over a 10-K / 10-Q, latest or any fiscal year |
| Answers cite sources | — | Form, filing date and section |

## How RAG Works Here

```
"What supply chain risks does Apple cite?"
    ↓ think() — this is in the company's own words, not market data
    ↓ search_filings(symbol="AAPL", query="supply chain risks", form="10-K")
        ↓ EDGAR: ticker → CIK → most recent 10-K               (edgar.py)
          (or the 10-K for a fiscal_year, e.g. 2021)
        ↓ first time only: download, convert HTML to text,
          split into Part/Item sections, chunk, embed, store    (filings_store.py → filings_db/)
        ↓ embed the query, return the 5 most similar chunks from that filing
    ↓ answer from those passages, citing "10-K filed 2025-10-31, Item 1A. Risk Factors"
```

- **Indexed once, reused everywhere.** A filing is downloaded and embedded the first time it is searched, which can take up to a minute. It is then stored in `filings_db/` and reused by every session.
- **Latest filing by default, any year on request.** Without `fiscal_year`, a search uses the most recent filing of that form. EDGAR's answer is cached for 24 hours, so even a long-running app picks up a new 10-K within a day; older ones stay in the index.
- **Older filings by fiscal year.** `fiscal_year` is the year the reporting period ended (Apple's year ending September 2021 is 2021; NVIDIA's year ending January 2025 is 2025). To compare years, the agent searches each year with the same query and says which period each point comes from.
- **Chunks follow the filing's structure.** Chunks are about 1,000 characters, which is what the embedding model reads in full. They overlap by about 200 characters and never cross a section boundary. Each chunk is tagged with its section, e.g. `Part I — Item 1A. Risk Factors`, and the heading is embedded with the text, so a search for "risk factors" favours risk chunks.
- **The company's name is removed from the search.** The search only covers that company's filing, so its name adds nothing to the query. It would also pull results toward boilerplate that repeats the name. The name and ticker are removed first, e.g. "What risks does NVIDIA cite?" becomes "What risks does cite?".
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

### Retrieval eval

`evals/run_retrieval_evals.py` measures retrieval alone: whether `search_filings` returns the right passages. It makes no Claude calls.

```bash
python evals/run_retrieval_evals.py                                          # report, k=5
python evals/run_retrieval_evals.py --save evals/results/retrieval_run.json  # also save as JSON
```

- **Gold set:** `evals/retrieval_gold.json` holds 37 paraphrased questions over five pinned filings (Apple 10-Ks for fiscal 2008, 2021 and 2025, NVIDIA's fiscal 2025 10-K, and an Apple 10-Q), plus 3 lookups that should fail. Each question lists verbatim snippets from the filing and the expected sections.
- **Metrics:** filing lookup, hit@1/3/k, MRR@k and section@1, shown overall, per question type and per filing. A passage counts as a hit if it contains a gold snippet.
- **Gold check:** every snippet is checked against the indexed chunks before scoring. A typo in the gold set is reported as invalid gold, not counted as a retrieval miss.
- **Index:** the filings are indexed once into `evals/filings_db/` (gitignored). Delete that folder after changing chunking.
- **Compare runs:** `--compare <saved json>` shows each metric's change and lists the questions whose rank moved.
- **Results so far** (`evals/results/`):

  | Run | hit@1 | hit@5 | MRR@5 | NVIDIA hit@5 |
  |---|---|---|---|---|
  | `retrieval_baseline.json` | 0.35 | 0.57 | 0.43 | 0.10 |
  | `retrieval_strip_company.json`: company name removed from queries | 0.32 | **0.92** | **0.53** | **0.90** |

  The company's name in a question pulled the search toward boilerplate that repeats it ("All references to NVIDIA ..."), so the name and ticker are now removed before searching. Ranking the best passage first (hit@1) is the next gap to close.

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
| `search_filings` | Search a 10-K or 10-Q (latest, or by fiscal year) for relevant passages |

## Model

`claude-sonnet-4-6` — change the `MODEL` constant in `agent.py` to switch.

## Known Limitations

- **US SEC filers only.** ETFs and most non-US companies (which file 20-F/40-F) have no 10-K.
- **Text only.** Financial tables become flattened rows of text, so exact figures are better fetched with the data tools.
- **Fiscal year means the year the period ended.** A few companies (e.g. some retailers whose year ends in late January) name their fiscal year after the previous calendar year; the agent checks the returned period.
- **One 10-Q per year.** With `fiscal_year`, a 10-Q search uses the latest quarter ending in that year; picking a specific quarter is not built yet.
- **Retrieval can miss.** If the best passages do not answer the question, the agent is told to say so and to search again with different wording, not to guess.
- The current session's history is still sent in full with every request, as in step 4.

## What's Next

Step 6: **Multi-agent** — an orchestrator that delegates to specialist agents (market data, filings, memory).
