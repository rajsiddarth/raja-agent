"""Finance Agent — Claude + Robinhood MCP. Run: python agent.py"""

import anthropic

MODEL = "claude-opus-5-5"

# Robinhood MCP server — Claude calls these tools directly, no local handlers needed
MCP_SERVERS = [
    {
        "type": "url",
        "url":  "https://agent.robinhood.com",
        "name": "robinhood",
    }
]

# Tell Claude to use the Robinhood MCP toolset
TOOLS = [
    {
        "type":            "mcp_toolset",
        "mcp_server_name": "robinhood",
    }
]

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a professional personal finance assistant with access to the user's Robinhood account.
Use the available Robinhood tools to fetch accurate, real-time data before answering — never guess prices or balances.
Be formal and professional in all responses. Format numbers with $ and % signs.
Only perform read operations (quotes, portfolio, positions, news). Do not place or cancel orders."""


def run_agent(user_message: str, history: list) -> tuple[str, list]:
    """Run one agent turn. Returns (reply, updated_history)."""
    history = history + [{"role": "user", "content": user_message}]

    while True:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            mcp_servers=MCP_SERVERS,
            betas=["mcp-client-2025-11-20"],
            messages=history,
        )

        # Append full response content (preserves tool_use blocks)
        history = history + [{"role": "assistant", "content": response.content}]

        if response.stop_reason == "end_turn":
            reply = next(
                (b.text for b in response.content if hasattr(b, "text")),
                "(no response)",
            )
            return reply, history

        if response.stop_reason == "tool_use":
            # MCP tools execute server-side — we just collect results from the response
            tool_results = []
            for block in response.content:
                if not hasattr(block, "type") or block.type != "tool_use":
                    continue
                print(f"  → {block.name}")
                # For MCP tools, results come back automatically in the next response
                # We still need to send a tool_result acknowledging each tool_use block
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     "",  # MCP server handles execution; result is in next response
                })

            if tool_results:
                history = history + [{"role": "user", "content": tool_results}]

        else:
            return f"(stopped: {response.stop_reason})", history


def main() -> None:
    print("Finance Agent — Robinhood MCP  (type 'quit' to exit)")
    print("=" * 50)
    print("Try: 'What is Apple's current price?'")
    print("     'Show my portfolio'")
    print("     'What are my current positions?'\n")

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
