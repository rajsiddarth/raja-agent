"""Finance Agent — Claude + yfinance tools. Run: python agent.py"""

import json
import anthropic
from dotenv import load_dotenv
from tools import TOOLS, TOOL_HANDLERS

load_dotenv()

MODEL = "claude-opus-5-5"

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a professional personal finance assistant with access to real-time market data.
Use the available tools to fetch accurate, up-to-date information before answering — never guess prices or figures.
Be formal and professional in all responses. Format numbers with $ and % signs where appropriate."""


def run_agent(user_message: str, history: list, on_tool_call=None) -> tuple[str, list]:
    """Run one agent turn. Returns (reply, updated_history)."""
    history = history + [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
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

        else:
            return f"(stopped: {response.stop_reason})", history


def main() -> None:
    print("Finance Agent  (type 'quit' to exit)")
    print("=" * 50)
    print("Try: 'What is Apple's current price?'")
    print("     'Show me Tesla fundamentals'")
    print("     'What are the latest news for NVDA?'\n")

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
