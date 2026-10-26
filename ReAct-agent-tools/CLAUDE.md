# ReAct Finance Agent

A finance agent that uses the **ReAct pattern** (Reason + Act) — Claude thinks step-by-step before calling tools.

## Project Structure

- `agent.py` — ReAct agent loop with think tool + yfinance tools
- `tools.py` — Tool definitions and handlers (yfinance)
- `app.py` — Streamlit chat UI showing reasoning steps
- `requirements.txt` — Dependencies
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

Compared to the reactive agent, the ReAct agent:
- Explains its reasoning before acting
- Plans which tools to call upfront
- Handles complex multi-step questions better (comparisons, analysis)

## The think Tool

```python
{
    "name": "think",
    "description": "Reason step-by-step before taking action.",
    "input_schema": {
        "reasoning": "Your step-by-step plan for answering the question."
    }
}
```

Claude's reasoning is visible in the UI — you can see exactly how it decides what to fetch.

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
- Handle errors defensively; surface clear messages to the user

## Testing

### Unit tests
```bash
source ../venv/bin/activate
pytest tests/test_agent.py -v
pytest tests/test_tools.py -v
```

9 agent tests (includes think tool and refusal handling) + 10 tool tests.

### Eval suite
```bash
source ../venv/bin/activate
python evals/run_evals.py
```

6 rule-based eval cases — checks think-tool usage, correct tool calls, and answer content against the live API.

## Roadmap

1. ✅ **Reactive agent** — agent + tools (`reactive-agent-tools/`)
2. ✅ **ReAct agent** — reason + act (`ReAct-agent-tools/`) ← you are here
3. **Agent + memory** — persist conversation history across sessions
4. **Agent + RAG** — vector search over earnings reports / filings
5. **Multi-agent** — orchestrator + specialist agents

Be formal and professional
