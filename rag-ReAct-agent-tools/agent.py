"""ReAct Finance Agent with long-term memory and RAG over SEC filings (10-K / 10-Q)."""

import json
import anthropic
from dotenv import load_dotenv
from tools import TOOLS, TOOL_HANDLERS
from memory import load_history, save_history
from memory_store import remember_last_turn

load_dotenv()

MODEL = "claude-sonnet-4-6"

client = anthropic.Anthropic()

THINK_TOOL = {
    "name": "think",
    "description": (
        "Use this tool to reason step-by-step before taking action. "
        "Think about what data you need, which tools to call, and in what order. "
        "This reasoning is shown to the user but does not fetch any data."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Your step-by-step reasoning about how to answer the question.",
            }
        },
        "required": ["reasoning"],
    },
}

ALL_TOOLS = [THINK_TOOL] + TOOLS

SYSTEM_PROMPT = """You are a financial data assistant. Your job is to fetch and present market data — prices, fundamentals, news, and historical performance.

Follow the ReAct pattern on every question:
1. THINK first — use the think tool to reason about what data you need and which tools to call
2. ACT — call the appropriate data tools (get_quote, get_fundamentals, etc.)
3. ANSWER — present the data in a clear, structured format

Always think before acting. For data lookups, one think step is enough.
For comparisons, think about all the data needed before fetching any of it.

You can also read companies' SEC filings with the search_filings tool (RAG over a 10-K or 10-Q).
Use it when the answer is in the company's own words rather than in market data: risk factors, business
model, segments, competition, supply chain, legal proceedings, management's discussion. When you use it:
- search with specific wording, and search again with different wording if the passages do not answer it
- it searches the latest filing by default; pass fiscal_year for an earlier one (e.g. "Apple's 2021 10-K")
- to compare years, search each year with the same query, and say which period each point comes from
- fiscal_year is the year the period ended; if a company labels its fiscal years differently, check the
  returned period and say which period you used
- answer only from the returned passages; if they do not cover the question, say so rather than guessing
- cite the filing (form and filing date) and the section each point comes from
- combine with data tools when a question needs both (e.g. "how has NVDA done, and what risks does it cite?")

You present factual market data only. You do not give investment advice or recommendations.
When users ask to "compare" stocks, fetch the data for each and present it side by side.
When users ask "what is the outlook", fetch recent news and present the headlines with context.

You have memory of the current session's conversation. If the user refers to a stock mentioned earlier
(e.g. "what about its P/E?" after discussing Apple), use context from prior turns to resolve it.

You also have long-term memory of past sessions via the recall_memory tool. If the user refers to
something not in the current conversation (e.g. "my concern about AMD", "what did we look at last time"),
call recall_memory before answering. Think before recalling, as with any other tool. Recalled figures
are historical: say when they were recorded, and fetch fresh data with the data tools if the user needs
current values.

Be formal and professional. Format numbers with $ and % signs."""


def run_agent(
    user_message: str,
    history: list,
    on_tool_call=None,
    on_think=None,
) -> tuple[str, list]:
    """Run one ReAct agent turn. Returns (reply, updated_history).

    Callbacks:
        on_tool_call(name): fired when a data tool is called
        on_think(reasoning): fired when the think tool is used
    """
    history = history + [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=ALL_TOOLS,
            messages=history,
        )

        # Store plain dicts (not SDK objects) so history can be saved as JSON
        content = [b.model_dump(exclude_none=True) for b in response.content]
        history = history + [{"role": "assistant", "content": content}]

        if response.stop_reason == "end_turn":
            reply = next(
                (b.text for b in response.content if hasattr(b, "text")),
                "(no response)",
            )
            return reply, history

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                if block.name == "think":
                    reasoning = block.input.get("reasoning", "")
                    print(f"  💭 {reasoning}")
                    if on_think:
                        on_think(reasoning)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Reasoning noted. Proceed with your plan.",
                    })
                else:
                    print(f"  → {block.name}({json.dumps(block.input)})")
                    if on_tool_call:
                        on_tool_call(block.name)

                    handler = TOOL_HANDLERS.get(block.name)
                    if handler is None:
                        result = {"error": f"Unknown tool: {block.name}"}
                        is_error = True
                    else:
                        try:
                            result = handler(block.input)
                            is_error = False
                        except Exception as exc:
                            result = {"error": str(exc)}
                            is_error = True

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                        "is_error": is_error,
                    })

            history = history + [{"role": "user", "content": tool_results}]

        elif response.stop_reason == "refusal":
            return (
                "I can only retrieve and present market data. "
                "Please ask me for stock prices, fundamentals, news, or historical data."
            ), history
        else:
            text = next(
                (b.text for b in response.content if hasattr(b, "text")),
                f"Unexpected stop: {response.stop_reason}",
            )
            return text, history


def _remember_or_warn(session_name: str, history: list) -> None:
    """Store the latest turn — a long-term memory failure must never break the chat."""
    try:
        remember_last_turn(session_name, history)
    except Exception as exc:
        print(f"  (warning: could not store turn in long-term memory: {exc})")


def main() -> None:
    """CLI entry point with session-based memory."""
    print("ReAct Finance Agent with Memory + SEC Filings  (type 'quit' to exit, 'new' for new session)")
    print("=" * 60)

    session_name = input("Session name (press Enter for 'default'): ").strip() or "default"
    history = load_history(session_name)

    if history:
        turn_count = sum(1 for m in history if m.get("role") == "user")
        print(f"Resumed session '{session_name}' — {turn_count} prior turn(s)\n")
    else:
        print(f"New session '{session_name}'\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            save_history(session_name, history)
            print("\nSession saved. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            save_history(session_name, history)
            print("Session saved. Goodbye!")
            break
        if user_input.lower() == "new":
            save_history(session_name, history)
            session_name = input("New session name: ").strip() or "default"
            history = []
            print(f"Started new session '{session_name}'\n")
            continue

        print()
        reply, history = run_agent(user_input, history)
        save_history(session_name, history)
        print(f"Agent: {reply}\n")
        _remember_or_warn(session_name, history)


if __name__ == "__main__":
    main()
