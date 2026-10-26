"""
Eval suite for the ReAct Finance Agent.

Runs a fixed set of questions against the live agent and grades each response on:
  - used_think   : did the agent call the think tool at least once?
  - tools_called : did the agent call all the expected data tools?
  - answer_check : does the final answer contain all expected strings?

Usage:
    cd ReAct-agent-tools
    source ../venv/bin/activate
    python evals/run_evals.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent import run_agent

# ---------------------------------------------------------------------------
# Eval cases
# ---------------------------------------------------------------------------
# Each case has:
#   question        : the user's input
#   expect_tools    : tool names the agent should call (order doesn't matter)
#   expect_in_answer: strings that must appear in the final answer (case-insensitive)

EVALS = [
    {
        "id": "single_quote",
        "question": "What is Apple's current stock price?",
        "expect_tools": ["get_quote"],
        "expect_in_answer": ["aapl", "$"],
    },
    {
        "id": "fundamentals_single",
        "question": "Get the fundamentals for Microsoft",
        "expect_tools": ["get_fundamentals"],
        "expect_in_answer": ["msft", "p/e"],
    },
    {
        "id": "fundamentals_compare",
        "question": "Get Apple and Microsoft fundamentals",
        "expect_tools": ["get_fundamentals"],   # called at least once
        "expect_in_answer": ["aapl", "msft"],
    },
    {
        "id": "news",
        "question": "Get recent NVDA news headlines",
        "expect_tools": ["get_news"],
        "expect_in_answer": ["nvda"],
    },
    {
        "id": "history",
        "question": "Get S&P 500 performance over the last month",
        "expect_tools": ["get_history"],
        "expect_in_answer": [],   # just check it doesn't error
    },
    {
        "id": "multi_tool",
        "question": "Get the price and recent news for Tesla",
        "expect_tools": ["get_quote", "get_news"],
        "expect_in_answer": ["tsla"],
    },
]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_eval(case: dict) -> dict:
    tools_called: list[str] = []
    thoughts: list[str] = []

    start = time.time()
    try:
        reply, _ = run_agent(
            case["question"],
            [],
            on_tool_call=lambda name: tools_called.append(name),
            on_think=lambda r: thoughts.append(r),
        )
        elapsed = round(time.time() - start, 1)
        error = None
    except Exception as exc:
        reply = ""
        elapsed = round(time.time() - start, 1)
        error = str(exc)

    # Grade
    used_think = len(thoughts) > 0

    tools_ok = all(t in tools_called for t in case["expect_tools"])

    answer_lower = reply.lower()
    missing_strings = [s for s in case["expect_in_answer"] if s not in answer_lower]
    answer_ok = len(missing_strings) == 0

    passed = used_think and tools_ok and answer_ok and error is None

    return {
        "id": case["id"],
        "question": case["question"],
        "passed": passed,
        "used_think": used_think,
        "tools_called": tools_called,
        "tools_ok": tools_ok,
        "answer_ok": answer_ok,
        "missing_strings": missing_strings,
        "reply_preview": reply[:120].replace("\n", " "),
        "elapsed_s": elapsed,
        "error": error,
    }


def print_result(r: dict) -> None:
    status = "PASS" if r["passed"] else "FAIL"
    print(f"\n[{status}] {r['id']}  ({r['elapsed_s']}s)")
    print(f"  Q: {r['question']}")
    print(f"  think={r['used_think']}  tools={r['tools_called']}  tools_ok={r['tools_ok']}  answer_ok={r['answer_ok']}")
    if r["missing_strings"]:
        print(f"  missing from answer: {r['missing_strings']}")
    if r["error"]:
        print(f"  ERROR: {r['error']}")
    else:
        print(f"  reply: {r['reply_preview']}...")


def main() -> None:
    print("=" * 60)
    print("ReAct Finance Agent — Eval Suite")
    print("=" * 60)

    results = []
    for case in EVALS:
        print(f"\nRunning: {case['id']} ...", end="", flush=True)
        result = run_eval(case)
        results.append(result)
        print_result(result)

    # Summary
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    total_time = sum(r["elapsed_s"] for r in results)

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} passed  ({total_time:.1f}s total)")

    think_rate = sum(1 for r in results if r["used_think"]) / total * 100
    print(f"Think tool usage: {think_rate:.0f}% of questions")

    if passed < total:
        print("\nFailing cases:")
        for r in results:
            if not r["passed"]:
                issues = []
                if not r["used_think"]:
                    issues.append("no think")
                if not r["tools_ok"]:
                    issues.append(f"missing tools (got {r['tools_called']})")
                if not r["answer_ok"]:
                    issues.append(f"missing strings {r['missing_strings']}")
                if r["error"]:
                    issues.append(f"error: {r['error']}")
                print(f"  {r['id']}: {', '.join(issues)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
