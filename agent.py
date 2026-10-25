"""Finance Agent — Claude + tool use. Run: python agent.py"""

import json
import anthropic
from tools import TOOL_DEFINITIONS, TOOL_HANDLERS

MODEL = "claude-opus-5-5"

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a professional personal finance assistant.
You have access to the user's stock portfolio and real-time stock data.
Use tools to fetch accurate data before answering — never guess prices.
Be formal and professional in all responses. Format numbers with $ and % signs."""


def run_agent(user_message: str, history: list) -> tuple[str, list]:
    """Run one agent turn. Returns (reply, updated_history)."""
    history = history + [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=history,
        )

        # Always append the full response (preserves tool_use blocks)
        history = history + [{"role": "assistant", "content": response.content}]

        if response.stop_reason == "end_turn":
            reply = next(
                (b.text for b in response.content if b.type == "text"),
                "(no response)",
            )
            return reply, history

        if response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                print(f"  → calling {block.name}({json.dumps(block.input)})")

                handler = TOOL_HANDLERS.get(block.name)
                if handler:
                    result = handler(block.input)
                    is_error = "error" in result
                else:
                    result = {"error": f"Unknown tool: {block.name}"}
                    is_error = True

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     json.dumps(result),
                    "is_error":    is_error,
                })

            # Send all results back in one user message, then loop
            history = history + [{"role": "user", "content": tool_results}]

        else:
            return f"(stopped: {response.stop_reason})", history


def main() -> None:
    print("Finance Agent  (type 'quit' to exit)")
    print("=" * 45)
    print("Try: 'What's Apple's stock price?'")
    print("     'Show my portfolio'")
    print("     'Any news on Tesla?'")
    print("     'Compare AAPL and MSFT'\n")

    history: list = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Bye!")
            break

        print()
        reply, history = run_agent(user_input, history)
        print(f"Agent: {reply}\n")


if __name__ == "__main__":
    main()
