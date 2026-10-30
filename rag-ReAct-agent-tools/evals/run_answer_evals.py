"""
Answer eval for RAG over SEC filings — runs the live agent and grades its final answers.

Uses evals/filings_gold.json: the 37 answerable cases (with key_facts and exact_facts) and the
unanswerable_cases (not in the filing, false premise, no such filing). Each question is pinned to its
filing by appending "(Use <SYMBOL>'s <form> for fiscal year <year>.)", so new filings never change the
expected answer.

Metrics
  Answerable questions
    - correct            : share of answers covering >= 60% of their key facts (judge)
    - key_fact_coverage  : mean share of key facts covered (judge)
    - exact_fact_rate    : share of exact figures and names that appear in the answer (code)
    - faithfulness       : mean share of the answer's claims supported by what the tools returned (judge)
    - fully_grounded     : share of answers with no unsupported claim (judge)
    - citation_present   : answer names the form and at least one Item (code)
    - filing_identified  : answer gives the filing date, period or fiscal year of a filing it retrieved (code)
    - item_validity      : share of cited Items the agent actually retrieved passages from (code)
    - citation_support   : share of section citations backed by a passage from that section (judge)
    - false_abstention   : share of answerable questions the agent declined (judge; should be ~0)
    - filing_ok / retrieval_hit : the agent retrieved the right filing / a gold passage (code)
  Unanswerable questions
    - correct_abstention : share where the agent said the filing doesn't support it (judge; target 1.0)
    - answered_unanswerable : count of confident answers to unanswerable questions (the worst failure)
  Attribution: answerable cases split by (gold passage retrieved?) x (answer correct?)

Costs real money: every case is a full agent turn (claude-sonnet-4-6) plus a judge call (claude-opus-5-5).
The run prints its token usage and estimated cost. --no-judge runs the code checks only.

Usage:
    cd rag-ReAct-agent-tools
    source ../venv/bin/activate
    python evals/run_answer_evals.py --limit 3                     # smoke test
    python evals/run_answer_evals.py --save evals/results/answers_baseline.json
    python evals/run_answer_evals.py --runs 3                      # repeat to see run-to-run spread
    python evals/run_answer_evals.py --compare evals/results/answers_baseline.json
    python evals/run_answer_evals.py --export-labels evals/results/labels.json   # for judge calibration
"""

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

EVALS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(EVALS_DIR))
sys.path.insert(0, EVALS_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(EVALS_DIR), ".env"))

import anthropic

import agent
import filings_store
import memory_store
from answer_metrics import (
    attribution, citation_checks, evidence_text, exact_fact_rate, filing_searches, retrieval_hit, summarize,
    tool_calls,
)
from judge import JUDGE_MODEL, JudgeError, judge_answer

GOLD_PATH = os.path.join(EVALS_DIR, "filings_gold.json")
EVAL_FILINGS_DIR = os.path.join(EVALS_DIR, "filings_db")

# $ per million tokens (input, output); used only for the cost estimate printed at the end
PRICES = {"claude-sonnet-4-6": (3.00, 15.00), "claude-opus-5-5": (4.00, 20.00)}

HEADLINE = [
    "correct", "key_fact_coverage", "exact_fact_rate", "faithfulness", "fully_grounded", "citation_present",
    "filing_identified", "item_validity", "citation_support", "false_abstention", "filing_ok", "retrieval_hit",
    "correct_abstention",
]

_local = threading.local()
# set when the API rejects the account itself (no credit, bad key): later cases are skipped, not run
_abort = threading.Event()


def _is_account_error(exc: Exception) -> bool:
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return True
    return isinstance(exc, anthropic.BadRequestError) and "credit balance" in str(exc)


def _track_agent_usage() -> None:
    """Wrap the agent's client so each case can count the tokens its agent turn used."""
    create = agent.client.messages.create

    def counting_create(**kwargs):
        response = create(**kwargs)
        usage = getattr(_local, "usage", None)
        if usage is not None:
            usage["input_tokens"] += response.usage.input_tokens
            usage["output_tokens"] += response.usage.output_tokens
        return response

    agent.client.messages.create = counting_create


def load_cases(gold: dict, args: argparse.Namespace) -> list[dict]:
    cases = [dict(c, kind="answerable") for c in gold["cases"]]
    cases += [dict(c, kind="unanswerable", key_facts=[], exact_facts=[], gold_snippets=[])
              for c in gold.get("unanswerable_cases", [])]
    if args.type:
        cases = [c for c in cases if c["type"] == args.type]
    if args.ids:
        wanted = set(args.ids.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    return cases[: args.limit] if args.limit else cases


def agent_prompt(case: dict, filings: dict) -> str:
    """The question as sent to the agent, pinned to the gold filing."""
    if case.get("filing") is None:
        return case["question"]  # the question already names the (nonexistent) filing
    spec = filings[case["filing"]]
    return f"{case['question']} (Use {spec['symbol']}'s {spec['form']} for fiscal year {spec['fiscal_year']}.)"


def grade_case(case: dict, filings: dict, use_judge: bool, judge_model: str) -> dict:
    """Run the agent on one case and grade the answer."""
    prompt = agent_prompt(case, filings)
    result = {"id": case["id"], "kind": case["kind"], "type": case["type"], "prompt": prompt}
    if _abort.is_set():
        result["error"] = "skipped: run aborted after an account error"
        return result
    _local.usage = {"input_tokens": 0, "output_tokens": 0}
    start = time.time()
    try:
        reply, history = agent.run_agent(prompt, [])
    except Exception as exc:  # one failing case must not stop the run, unless the account itself is rejected
        if _is_account_error(exc):
            _abort.set()
        result.update(error=f"agent failed: {exc}", seconds=round(time.time() - start, 1), agent_usage=_local.usage)
        return result
    result["seconds"] = round(time.time() - start, 1)
    result["agent_usage"] = _local.usage
    result["answer"] = reply

    calls = tool_calls(history)
    searches = filing_searches(calls)
    expected_period = filings[case["filing"]]["period"] if case.get("filing") else None
    result.update(
        tools_called=[c["name"] for c in calls],
        filings_retrieved=[f"{s['filing']['symbol']} {s['filing']['form']} {s['filing']['period']}" for s in searches],
        filing_ok=any(s["filing"]["period"] == expected_period for s in searches) if expected_period else None,
        retrieval_hit=retrieval_hit(searches, case["gold_snippets"]),
        citations=citation_checks(reply, searches),
        exact_fact_rate=exact_fact_rate(reply, case["exact_facts"]),
    )
    result["evidence"] = evidence_text(calls)

    if use_judge:
        try:
            grade = judge_answer(case["question"], reply, result["evidence"], case["key_facts"], judge_model)
            if len(grade["key_facts"]) != len(case["key_facts"]):
                raise JudgeError(f"judge graded {len(grade['key_facts'])} key facts, expected {len(case['key_facts'])}")
            result["judge"] = grade
        except Exception as exc:  # JudgeError or an API error: record it, keep the code-checked metrics
            if _is_account_error(exc):
                _abort.set()
            result["judge_error"] = str(exc)
    return result


def run_once(cases: list[dict], filings: dict, args: argparse.Namespace) -> list[dict]:
    done = 0
    lock = threading.Lock()

    def task(case: dict) -> dict:
        nonlocal done
        r = grade_case(case, filings, not args.no_judge, args.judge_model)
        with lock:
            done += 1
        status = "ERROR" if r.get("error") else ("judge error" if r.get("judge_error") else "ok")
        print(f"  [{done}/{len(cases)}] {case['id']:<32} {r.get('seconds', 0):>5}s  {status}", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        return list(pool.map(task, cases))


def cost(results: list[dict], judge_model: str) -> dict:
    agent_in = sum(r.get("agent_usage", {}).get("input_tokens", 0) for r in results)
    agent_out = sum(r.get("agent_usage", {}).get("output_tokens", 0) for r in results)
    judge_in = sum(r["judge"]["usage"]["input_tokens"] for r in results if r.get("judge"))
    judge_out = sum(r["judge"]["usage"]["output_tokens"] for r in results if r.get("judge"))
    a_in, a_out = PRICES.get(agent.MODEL, (0, 0))
    j_in, j_out = PRICES.get(judge_model, (0, 0))
    return {
        "agent_tokens": [agent_in, agent_out],
        "judge_tokens": [judge_in, judge_out],
        "usd": round((agent_in * a_in + agent_out * a_out + judge_in * j_in + judge_out * j_out) / 1e6, 2),
    }


def fmt(value) -> str:
    return "  –  " if value is None else f"{value:.2f}"


def print_report(runs: list[list[dict]], before: dict, judge_model: str) -> dict:
    summaries = [summarize(r) for r in runs]
    results = runs[0]
    print("\nAnswer quality" + (f" (mean of {len(runs)} runs, min–max)" if len(runs) > 1 else "") + ":")
    for name in HEADLINE:
        values = [s.get(name) for s in summaries if s.get(name) is not None]
        if not values:
            continue
        line = f"  {name:<22} {fmt(sum(values) / len(values))}"
        if len(values) > 1:
            line += f"  ({min(values):.2f}–{max(values):.2f})"
        old = before.get("summary", {}).get(name)
        if old is not None:
            line += f"  ({sum(values) / len(values) - old:+.2f} vs compare)"
        print(line)
    s = summaries[0]
    if "answered_unanswerable" in s:
        print(f"  {'answered_unanswerable':<22} {s['answered_unanswerable']} of {s['n_unanswerable']}  (should be 0)")
    print(f"  cases: {s['n_answerable']} answerable, {s['n_unanswerable']} unanswerable, {s['errors']} errors")

    print("\nWhere answerable cases fail (gold passage retrieved? x answer correct?):")
    for label, count in attribution(results).items():
        print(f"  {label:<38} {count}")

    by_type = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)
    print("\nBy question type (first run):  correct  faithful  cited")
    for name, group in sorted(by_type.items()):
        g = summarize(group)
        main = g.get("correct_abstention") if group[0]["kind"] == "unanswerable" else g.get("correct")
        print(f"  {name:<22} n={len(group):<3} {fmt(main):>6}  {fmt(g.get('faithfulness')):>8}  {fmt(g.get('citation_present')):>5}")

    problems = []
    for r in results:
        j = r.get("judge")
        if r.get("error") or r.get("judge_error"):
            problems.append(f"{r['id']}: {r.get('error') or r.get('judge_error')}")
        elif j and r["kind"] == "unanswerable" and j["response_type"] != "abstained":
            problems.append(f"{r['id']}: should abstain, judged '{j['response_type']}'")
        elif j and r["kind"] == "answerable":
            missing = [f["fact"] for f in j["key_facts"] if not f["covered"]]
            unsupported = [c["claim"] for c in j["claims"] if not c["supported"]]
            if j["response_type"] == "abstained":
                problems.append(f"{r['id']}: abstained on an answerable question")
            if missing:
                problems.append(f"{r['id']}: missing facts: {'; '.join(missing)}")
            if unsupported:
                problems.append(f"{r['id']}: unsupported claims: {'; '.join(unsupported)[:300]}")
    if problems:
        print("\nProblems (first run):")
        for p in problems:
            print(f"  - {p}")

    spend = cost([r for run in runs for r in run], judge_model)
    seconds = [r["seconds"] for r in results if "seconds" in r]
    print(
        f"\nCost: ~${spend['usd']} (agent {spend['agent_tokens'][0]:,} in / {spend['agent_tokens'][1]:,} out, "
        f"judge {spend['judge_tokens'][0]:,} in / {spend['judge_tokens'][1]:,} out); "
        f"avg agent turn {sum(seconds) / max(len(seconds), 1):.0f}s"
    )
    return {"summaries": summaries, "cost": spend}


def export_labels(results: list[dict], gold_facts: dict, path: str) -> None:
    """Write answers with the judge's labels and blank human fields, for calibrating the judge."""
    items = []
    for r in results:
        if not r.get("judge"):
            continue
        items.append({
            "id": r["id"],
            "kind": r["kind"],
            "question": r["prompt"],
            "answer": r["answer"],
            "evidence": r["evidence"],
            "judge": {
                "response_type": r["judge"]["response_type"],
                "key_facts_covered": [f["covered"] for f in r["judge"]["key_facts"]],
                "has_unsupported_claim": any(not c["supported"] for c in r["judge"]["claims"]),
            },
            "key_facts": gold_facts[r["id"]],
            "human": {"response_type": None, "key_facts_covered": [None] * len(gold_facts[r["id"]]),
                      "has_unsupported_claim": None},
        })
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"\nExported {len(items)} answers for labelling to {path}. Fill in each 'human' block, then run:")
    print(f"  python evals/calibrate_judge.py {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=1, help="repeat the whole set N times (default 1)")
    parser.add_argument("--workers", type=int, default=4, help="cases run in parallel (default 4)")
    parser.add_argument("--type", help="only cases of this question type")
    parser.add_argument("--ids", help="comma-separated case ids")
    parser.add_argument("--limit", type=int, help="only the first N cases (smoke test)")
    parser.add_argument("--no-judge", action="store_true", help="code-checked metrics only, no judge calls")
    parser.add_argument("--judge-model", default=JUDGE_MODEL, help=f"judge model (default {JUDGE_MODEL})")
    parser.add_argument("--save", help="write full results as JSON to this path")
    parser.add_argument("--compare", help="a saved results JSON to show changes against")
    parser.add_argument("--export-labels", help="write answers + judge labels for hand-labelling to this path")
    args = parser.parse_args()

    with open(GOLD_PATH) as f:
        gold = json.load(f)
    cases = load_cases(gold, args)
    before = {}
    if args.compare:
        with open(args.compare) as f:
            saved = json.load(f)
        before = {"summary": saved["summaries"][0]} if saved.get("summaries") else {}

    # an isolated memory store, the eval's filing index, and no agent console output
    memory_store.MEMORY_DIR = tempfile.mkdtemp(prefix="answer_eval_memory_")
    memory_store.reset()
    filings_store.FILINGS_DIR = EVAL_FILINGS_DIR
    filings_store.reset()
    filings_store.indexed_filings()  # open the collection before worker threads share it
    agent.print = lambda *a, **k: None
    _track_agent_usage()

    print("=" * 72)
    print(f"Answer Eval — agent {agent.MODEL}, judge {'off' if args.no_judge else args.judge_model}")
    print("=" * 72)
    runs = []
    for i in range(args.runs):
        print(f"\nRun {i + 1}/{args.runs}: {len(cases)} cases")
        runs.append(run_once(cases, gold["filings"], args))
        if _abort.is_set():
            print("\nRUN ABORTED: the API rejected the account (out of credit or invalid key). "
                  "Scores below cover only the cases graded before that; do not save them as a baseline.")
            break

    report = print_report(runs, before, args.judge_model)

    if _abort.is_set() and (args.save or args.export_labels):
        print("Not saving or exporting results from an aborted run.")
        return

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        with open(args.save, "w") as f:
            json.dump({
                "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "agent_model": agent.MODEL,
                "judge_model": None if args.no_judge else args.judge_model,
                **report,
                "runs": [[{k: v for k, v in r.items() if k != "evidence"} for r in run] for run in runs],
            }, f, indent=2, ensure_ascii=False)
        print(f"Saved results to {args.save}")

    if args.export_labels:
        gold_facts = {c["id"]: c["key_facts"] for c in cases}
        export_labels(runs[0], gold_facts, args.export_labels)


if __name__ == "__main__":
    main()
