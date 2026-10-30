# raja-agent: building a finance agent step by step

A learning project that builds a Claude-powered finance agent one step at a time. Each folder is a complete, runnable agent that adds one new idea to the step before it: tools, then reasoning, then memory, then long-term memory, then retrieval over SEC filings.

All the agents answer questions about stocks using market data from [yfinance](https://github.com/ranaroussi/yfinance). Each step has a command-line chat (`agent.py`) and a Streamlit web UI (`app.py`).

## Roadmap

| Step | Folder | What it adds | Status |
|---|---|---|---|
| 1 | [`reactive-agent-tools/`](reactive-agent-tools/) | **Tool use.** Claude calls yfinance tools (quote, fundamentals, history, news, options) in a simple loop | ✅ |
| 2 | [`ReAct-agent-tools/`](ReAct-agent-tools/) | **ReAct.** A `think` tool makes Claude reason before acting; first eval suite | ✅ |
| 3 | [`memory-convhistory-ReAct-agent-tools/`](memory-convhistory-ReAct-agent-tools/) | **Conversation memory.** Named sessions saved as JSON, so they can be resumed after a restart | ✅ |
| 4 | [`longterm-memory-ReAct-agent-tools/`](longterm-memory-ReAct-agent-tools/) | **Long-term memory.** Past turns stored in a Chroma vector DB; `recall_memory` searches every session | ✅ |
| 5 | [`rag-ReAct-agent-tools/`](rag-ReAct-agent-tools/) | **RAG over SEC filings.** `search_filings` retrieves cited passages from a company's 10-K/10-Q (latest or any fiscal year) | ✅ |
| 6 | — | **Multi-agent.** An orchestrator that hands work to specialist agents (market data, filings, memory) | Planned |

Each folder has a `README.md` covering how that step works and a `CLAUDE.md` with notes for Claude Code.

## How the steps build on each other

```
1 Reactive    user → Claude → tool calls → answer
2 ReAct       + think() before acting
3 Memory      + session history saved to sessions/*.json
4 Long-term   + every turn stored in memory_db/ (Chroma); recall_memory(query)
5 RAG         + SEC EDGAR filings chunked by section into filings_db/; search_filings(symbol, query, fiscal_year)
```

## Setup

One virtual environment at the top level is shared by every step.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r rag-ReAct-agent-tools/requirements.txt
```

Step 5's requirements are a superset of the earlier steps' requirements, so this one install covers every folder.

Each step reads its own `.env` file (gitignored):

```
ANTHROPIC_API_KEY=your-key-here
SEC_USER_AGENT=Your Name you@example.com    # step 5 only; SEC requires a name and email
```

## Running a step

```bash
cd rag-ReAct-agent-tools        # or any step folder
source ../venv/bin/activate
python agent.py                 # command-line chat
streamlit run app.py            # web UI
```

## Tests and evals

Every step has offline unit tests that make no network calls. Steps 2 to 5 also have a live eval suite that calls the Claude API.

```bash
python -m pytest -q tests       # offline, fast
python evals/run_evals.py       # live agent evals (uses the Claude API)
```

Step 5 also has a **retrieval eval**. It makes no Claude calls and measures whether filing search returns the right passages: hit@k, MRR and section accuracy against a 37-question gold set. The results are saved in `rag-ReAct-agent-tools/evals/results/` so you can compare runs:

```bash
cd rag-ReAct-agent-tools
python evals/run_retrieval_evals.py --compare evals/results/retrieval_strip_company.json
```

Step 5 also has an **answer eval**. It runs the live agent and has a second model, `claude-opus-5-5`, judge each answer. It measures:
- **Correctness:** whether the answer covers the key facts.
- **Faithfulness:** whether every claim is supported by what the agent retrieved.
- **Citations:** whether the answer cites the right form, filing and sections.
- **Abstention:** on unanswerable questions, whether the agent says so instead of answering.

A full run costs about $4–5 in API usage:

```bash
python evals/run_answer_evals.py --limit 3                                    # smoke test
python evals/run_answer_evals.py --save evals/results/answers_baseline.json   # full run
```

See [`rag-ReAct-agent-tools/README.md`](rag-ReAct-agent-tools/README.md) for the metrics, how the judge works and how to calibrate it.

## Generated data (gitignored)

| Folder | Created by | Contents |
|---|---|---|
| `sessions/` | steps 3–5 | Saved chat sessions (JSON) |
| `memory_db/` | steps 4–5 | Long-term memory vector DB |
| `filings_db/`, `evals/filings_db/` | step 5 | Indexed SEC filings (the app's index and the eval's index) |
| `edgar_cache/` | step 5 | Cached SEC ticker list |

## Model

Each step sets its Claude model in the `MODEL` constant in `agent.py`. Steps 2–5 use `claude-sonnet-4-6`.
