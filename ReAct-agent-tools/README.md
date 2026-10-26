# ReAct Finance Agent

A Claude-powered finance agent using the **ReAct pattern** (Reason + Act) — Claude thinks step-by-step before calling tools.

This is step 2 of the learning roadmap, built on top of the [reactive agent](../reactive-agent-tools/).

## How It Works

```
User message
    ↓
Claude calls think() — reasons about what data it needs
    ↓
Claude calls data tools (get_quote, get_fundamentals, etc.)
    ↓
Claude synthesizes all data into a final answer
```

The `think` tool is a no-op — it returns nothing to Claude. Its only purpose is to force Claude to externalize its reasoning before acting. That reasoning is captured and shown to the user in the UI.

## ReAct vs Reactive

| | Reactive | ReAct |
|---|---|---|
| Planning | None — reacts directly | Thinks before acting |
| Multi-tool questions | May miss steps | Plans all tools upfront |
| Transparency | Shows tool calls | Shows reasoning + tool calls |
| Auditability | Hard to trace | Reasoning is visible |

## Project Structure

```
ReAct-agent-tools/
├── agent.py          # ReAct agent loop with think tool
├── tools.py          # Tool definitions and yfinance handlers
├── app.py            # Streamlit UI with reasoning expander
├── requirements.txt  # Dependencies
├── tests/
│   ├── test_agent.py # 9 unit tests (loop, think tool, refusal handling)
│   └── test_tools.py # 10 tool handler tests
├── evals/
│   └── run_evals.py  # Rule-based eval suite (6 live cases)
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
```

### Streamlit UI
```bash
source venv/bin/activate
streamlit run app.py
```

### Unit Tests
```bash
source venv/bin/activate
pytest tests/ -v
```

### Eval Suite (hits live API)
```bash
source venv/bin/activate
python evals/run_evals.py
```

## The think Tool

```python
THINK_TOOL = {
    "name": "think",
    "description": "Reason step-by-step before taking action.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Your step-by-step plan for answering the question."
            }
        },
        "required": ["reasoning"]
    }
}
```

Claude writes its reasoning into the `reasoning` argument. The agent captures it via the `on_think` callback and surfaces it in the UI. Claude gets back `"Reasoning noted. Proceed with your plan."` — no data, just an acknowledgement.

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

> Note: `claude-opus-5-5` refuses financial data questions. Use `claude-sonnet-4-6` or later.

## Eval Suite

`evals/run_evals.py` runs 6 rule-based cases against the live agent and grades each on:

- **think usage** — did the agent reason before acting?
- **tool calls** — did it call the right tools?
- **answer content** — does the response contain expected strings (tickers, `$`, `P/E`)?

## What's Next

Step 3: **Agent + Memory** — persist conversation history across sessions so the agent remembers prior context.
