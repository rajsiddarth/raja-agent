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
│
├── ── ENTRY POINTS ─────────────────────────────────────────────
├── agent.py          # run_agent(): the ReAct loop (think → tools → answer), system prompt, model
│                     #   callbacks on_think / on_tool_call / on_usage; main() is the CLI
├── app.py            # Streamlit UI: chat, sessions, indexed filings, cost per answer + session total
│
├── ── TOOLS CLAUDE CAN CALL ────────────────────────────────────
├── tools.py          # TOOLS (schemas Claude sees) + TOOL_HANDLERS (the Python that runs them)
│                     #   market data (yfinance), recall_memory, search_filings
│
├── ── RAG OVER SEC FILINGS ─────────────────────────────────────
├── edgar.py          # SEC EDGAR: ticker → CIK → filing (latest or by fiscal_year) → plain text
├── filings_store.py  # split_sections → chunk_text → index_filing (Chroma); strip_company → search
│
├── ── MEMORY AND COST ──────────────────────────────────────────
├── memory.py         # Current session: sessions/<name>.json
├── memory_store.py   # Long-term memory: question + answer per turn in Chroma
├── backfill.py       # Load existing session JSON files into long-term memory
├── costs.py          # Prices, token totals across a turn, cost_usd
│
├── ── QUALITY ──────────────────────────────────────────────────
├── tests/            # Offline unit tests (no network; fake embedding in conftest.py)
├── evals/
│   ├── filings_gold.json        # Gold set: questions, snippets, relevant passages, key facts
│   ├── run_retrieval_evals.py   # Retrieval eval (offline, free)
│   ├── retrieval_metrics.py     #   hit@k, MRR, recall@k, precision@k, MAP, section@1
│   ├── run_answer_evals.py      # Answer eval (live agent + LLM judge)
│   ├── answer_metrics.py        #   code checks: citations, exact facts, attribution
│   ├── judge.py                 #   claude-opus-5-5 judge
│   ├── calibrate_judge.py       #   judge vs. your own labels
│   ├── run_evals.py             # Original quick eval: tools called + expected words
│   └── results/                 # Saved runs, used with --compare
│
├── requirements.txt, README.md, CLAUDE.md
├── .env              # API key + SEC user agent (gitignored)
└── generated data (gitignored): filings_db/, memory_db/, sessions/, edgar_cache/, evals/filings_db/
```

### How a question flows through the files

```
"What supply chain risks did Apple cite in fiscal 2021?"
   │
   ▼
app.py / agent.main()  ──►  agent.run_agent()  ──►  Claude (claude-sonnet-4-6)
                                 │                    │ decides: think, then search_filings
                                 │◄───────────────────┘
                                 ▼
                          tools.search_filings(symbol="AAPL", query=..., fiscal_year=2021)
                                 │
                                 ▼
                          filings_store.ensure_indexed()
                                 ├─► edgar.find_filing()               which filing? (EDGAR)
                                 └─► not indexed yet? edgar.fetch_filing_text()
                                       → split_sections → chunk_text → index_filing (filings_db/)
                                 ▼
                          filings_store.search() → strip_company → search_filing (top 5 chunks)
                                 │
                                 ▼
                          passages back to Claude → answer with citations
   │
   ▼
costs: usage summed over every API call → on_usage(total + cost_usd)
memory.save_history (sessions/)  +  memory_store.remember_last_turn (memory_db/)
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

### Answer eval (live agent + LLM judge)

`evals/run_answer_evals.py` runs the live agent on the same gold set and grades its final answers. It costs real money: about $0.10 per case, so roughly $4–5 for the full set.

```bash
python evals/run_answer_evals.py --limit 3                                    # smoke test
python evals/run_answer_evals.py --save evals/results/answers_baseline.json   # full run
python evals/run_answer_evals.py --compare evals/results/answers_baseline.json
```

| Metric | Question it answers | Graded by |
|---|---|---|
| `correct`, `key_fact_coverage` | Does the answer contain the key facts? | Judge |
| `exact_fact_rate` | Do exact figures and names appear (e.g. `$89.3 billion`)? | Code |
| `faithfulness`, `fully_grounded` | Is every claim supported by what the tools returned? | Judge |
| `citation_present`, `filing_identified`, `item_validity` | Does it cite the form, the filing and only sections it retrieved? | Code |
| `citation_support` | Does each cited section back the claim attributed to it? | Judge |
| `correct_abstention`, `answered_unanswerable` | On 8 unanswerable questions, does it say so instead of answering? | Judge |
| `false_abstention` | Does it wrongly refuse questions it could answer? | Judge |

- **The judge** is `claude-opus-5-5`, a different, stronger model than the agent (`claude-sonnet-4-6`). It returns schema-validated JSON and sees only what the agent's tools returned that turn, so "supported" means "supported by what the agent read". Refusals fall back to another model server-side.
- **Attribution:** each answerable case is classified by whether a gold passage was retrieved and whether the answer was correct, which shows whether a failure came from retrieval or from the answer step.
- **Calibrate the judge before trusting it.** `--export-labels evals/results/judge_labels.json` writes answers with the judge's labels and blank fields for your own. Label about 20 answers, then run `python evals/calibrate_judge.py evals/results/judge_labels.json` to see agreement and every disagreement.
- `--runs 3` repeats the set and shows the min–max spread, because agent answers vary from run to run.

### Retrieval eval

`evals/run_retrieval_evals.py` measures retrieval alone: whether `search_filings` returns the right passages. It makes no Claude calls.

```bash
python evals/run_retrieval_evals.py                                          # report, k=5
python evals/run_retrieval_evals.py --save evals/results/retrieval_run.json  # also save as JSON
```

- **Gold set:** `evals/filings_gold.json` holds 37 paraphrased questions over five pinned filings: Apple 10-Ks for fiscal 2008, 2021 and 2025, NVIDIA's fiscal 2025 10-K, and an Apple 10-Q. There are also 3 lookups that should fail. Each question lists:
  - `gold_snippets`: verbatim text, any one of which marks a right passage.
  - `relevant`: for the 10 multi-part questions, every distinct passage a complete answer needs.
  - The expected sections, and the key facts the answer eval checks.

  The answer eval also uses 8 unanswerable questions: answers that aren't in the filing, false premises, and a filing that doesn't exist.
- **Gold check:** every snippet and every relevant passage is checked against the indexed chunks before scoring. A typo in the gold set is reported as invalid gold, not counted as a retrieval miss.
- **Index:** the filings are indexed once into `evals/filings_db/` (gitignored). Delete that folder after changing chunking.
- **Compare runs:** `--compare <saved json>` shows each metric's change and lists the questions whose rank moved.

#### Retrieval metrics

Search returns a **ranked list** of passages: rank 1 is the passage it thinks is most relevant. For each question, the eval records where the right passages landed in that list, scores it, and then averages over all questions.

| Metric | Question it answers | Per question |
|---|---|---|
| **hit@k** | Was a right passage anywhere in the top k? | 1 if yes, 0 if no |
| **MRR@k** (mean reciprocal rank) | How high was the first right passage? | 1 ÷ its rank: 1st = 1.0, 2nd = 0.5, 5th = 0.2, missing = 0 |
| **recall@k** | What share of **all** the passages a complete answer needs were in the top k? | parts found ÷ parts needed |
| **precision@k** | What share of the top k passages were relevant? | relevant passages ÷ k |
| **MAP@k** (mean average precision) | Like MRR, but rewards ranking **every** needed passage high, not just the first | average of the precision at the rank where each part is first found (0 for parts not found) |
| **section@1** | Is the top passage from the expected section, e.g. Item 1A? | 1 or 0 |

**Worked example.** *"What supply chain risks did Apple cite in fiscal 2021?"* needs 3 parts: A (single-source partners), B (limited sources) and C (shortages and price increases). Suppose search returns:

| Rank | Passage | Relevant? |
|---|---|---|
| 1 | Part A | ✓ |
| 2 | Unrelated | ✗ |
| 3 | Part B | ✓ |
| 4 | Unrelated | ✗ |
| 5 | Unrelated | ✗ |

- hit@5 = **1**: a right passage is in the list.
- MRR = 1 ÷ 1 = **1.0**: the first right passage is at rank 1.
- recall@5 = 2 of 3 parts = **0.67**: part C is missing, so the answer will be incomplete.
- precision@5 = 2 of 5 = **0.40**.
- MAP: part A is found at rank 1, where precision is 1/1; part B at rank 3, where precision is 2/3; part C is never found, so it scores 0. That gives (1 + 0.67 + 0) ÷ 3 = **0.56**.

Hit rate and MRR call this search perfect. Recall shows a third of the answer is missing.

How to read them:
- **hit@k and MRR assume one right passage per question.** For single-part questions, recall@k equals hit@k and MAP equals MRR.
- **recall@k matters for multi-part questions.** A missing part means an incomplete answer, even when hit@k is 1.
- **precision@k measures noise and cost.** Every irrelevant passage is roughly 250 tokens the agent pays for and has to ignore. It's a lower bound: a relevant passage that isn't labelled in the gold set counts as irrelevant.

#### Results so far (`evals/results/`)

| Run | hit@1 | hit@5 | MRR@5 | recall@5 | precision@5 | MAP@5 | NVIDIA hit@5 |
|---|---|---|---|---|---|---|---|
| `retrieval_baseline.json` | 0.35 | 0.57 | 0.43 | – | – | – | 0.10 |
| `retrieval_strip_company.json`: company name removed from queries | 0.32 | **0.92** | **0.53** | – | – | – | **0.90** |
| `retrieval_recall_precision.json`: same search, new metrics | 0.32 | 0.92 | 0.53 | 0.82 | 0.21 | 0.47 | 0.90 |

- **Company name:** the company's name in a question pulled the search toward boilerplate that repeats it ("All references to NVIDIA ..."). The name and ticker are now removed before searching.
- **Multi-part questions are the weak spot.** For the 10 multi-part questions, hit@5 is 0.90 but **recall@5 is only 0.52**, and MAP is 0.22: search usually finds one part of the answer and misses about half of the rest. In `aaplq_legal`, the top passage was "Other Legal Proceedings" (a hit), but none of the three cases asked about (Epic, DOJ, DMA) were retrieved, so recall was 0. For the 27 single-part questions, recall@5 = hit@5 = 0.93.
- **Precision@5 is about 0.21:** roughly one passage in five is relevant.
- **Next gaps:** ranking the best passage first (hit@1, MAP), and retrieving every part of a multi-part answer (recall). Options include a reranker, keyword search alongside vector search, or more passages per search for broad questions.

## Cost per Query

Each answer shows what it cost, e.g. `💲 $0.048 · 3 API calls · 15,195 in / 1,134 out tokens`. The CLI also prints the running total for the session, and the Streamlit sidebar shows it.

- **Only Claude calls cost money.** Embeddings (local MiniLM), SEC EDGAR and yfinance are free.
- **One question means several API calls.** The `think` step, each tool call and the final answer are separate calls, and each one resends the whole conversation so far: system prompt, tool definitions, history and tool results. So input tokens dominate the cost.
- **Typical cost.** In the answer eval, a filing question cost a median of about $0.05: $0.03 for a single lookup, $0.09 for questions needing several searches, and up to $0.17 for one that needed 6 tool calls.
- **Cost grows over a session.** Each new question resends the earlier ones in that session.
- **Where it's computed:** `costs.py` holds the prices. `run_agent(..., on_usage=callback)` reports the turn's token totals and `cost_usd` once the turn ends.

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
