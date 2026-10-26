"""ReAct Finance Agent — Reason + Act pattern with yfinance tools."""

import json
import anthropic
from dotenv import load_dotenv
from tools import TOOLS, TOOL_HANDLERS

load_dotenv()

MODEL = "claude-sonnet-4-6"

client = anthropic.Anthropic()

# The think tool lets Claude reason before acting — core of ReAct pattern
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

You present factual market data only. You do not give investment advice or recommendations.
When users ask to "compare" stocks, fetch the data for each and present it side by side.
When users ask "what is the outlook", fetch recent news and present the headlines with context.

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

        history = history + [{"role": "assistant", "content": response.content}]

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


def main() -> None:
    print("ReAct Finance Agent  (type 'quit' to exit)")
    print("=" * 50)
    print("Try: 'Should I buy Apple or Microsoft right now?'")
    print("     'Compare Tesla and Ford fundamentals'")
    print("     'What is the outlook for NVDA based on recent news?'\n")

    history: list = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        print()
        reply, history = run_agent(user_input, history)
        print(f"Agent: {reply}\n")


if __name__ == "__main__":
    main()
