"""
Retrieval eval for RAG over SEC filings — measures whether search_filing returns the right passages.

No Claude calls: it runs filings_store directly with the real embedding model, so it is cheap enough
to run after every change to chunking or search. The gold set is evals/filings_gold.json.

Metrics (per question, then averaged overall, per question type and per filing):
  - filing lookup : did fiscal_year pick the expected filing (period end date)?
  - hit@k         : is a passage containing a gold snippet among the top k results?
  - MRR@k         : 1 / rank of the first such passage (0 if not in the top k)
  - section@1     : is the top passage from the expected section (e.g. Item 1A)?
  - recall@k      : share of the relevant passages (the "relevant" groups) found in the top k
  - precision@k   : share of the top k passages that are relevant (a lower bound: only labelled
                    passages count as relevant)
  - MAP@k         : mean average precision; like MRR, but rewards ranking every relevant passage high
  - error cases   : do lookups that should fail (ETF, foreign filer, future year) fail cleanly?

Before scoring, every gold snippet is checked against the indexed chunks of its filing. A case whose
snippets are not in any chunk is reported as INVALID GOLD and left out of the scores, so a typo in the
gold set is never mistaken for a retrieval miss.

The filings are indexed once into evals/filings_db/ (gitignored) and reused on later runs; delete that
folder after changing chunking so the filings are re-indexed with the new chunks.

Usage:
    cd rag-ReAct-agent-tools
    source ../venv/bin/activate
    python evals/run_retrieval_evals.py                      # k=5
    python evals/run_retrieval_evals.py --k 8 --type risk
    python evals/run_retrieval_evals.py --save evals/results/retrieval_baseline.json
    python evals/run_retrieval_evals.py --compare evals/results/retrieval_baseline.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

EVALS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(EVALS_DIR))
sys.path.insert(0, EVALS_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(EVALS_DIR), ".env"))

import edgar
import filings_store
from retrieval_metrics import (
    contains_any, first_hit_rank, relevance_groups, score_passages, section_matches, summarize, summarize_by,
)

GOLD_PATH = os.path.join(EVALS_DIR, "filings_gold.json")
EVAL_FILINGS_DIR = os.path.join(EVALS_DIR, "filings_db")


def index_filings(specs: dict) -> dict:
    """Look up and index each gold filing. Returns {key: {ok, period, accession, chunks, error}}."""
    filings = {}
    for key, spec in specs.items():
        try:
            filing = filings_store.ensure_indexed(spec["symbol"], spec["form"], spec["fiscal_year"])
        except edgar.EdgarError as exc:
            filings[key] = {"ok": False, "error": str(exc)}
            continue
        filings[key] = {
            "ok": filing["period"] == spec["period"],
            "period": filing["period"],
            "accession": filing["accession"],
            "filing": filing,
            "chunks": filings_store.filing_chunks(filing["accession"]),
        }
    return filings


def check_gold(case: dict, chunks: list[dict]) -> tuple[str | None, list[str]]:
    """Return (problem or None, sections of the chunks that contain a gold snippet).

    Every relevance group must also appear in some chunk, or recall would count a passage that cannot exist.
    """
    sections = sorted({c["section"] for c in chunks if contains_any(c["text"], case["gold_snippets"])})
    if not sections:
        return "no gold snippet appears in any indexed chunk", sections
    for i, group in enumerate(relevance_groups(case), start=1):
        if not any(contains_any(c["text"], group) for c in chunks):
            return f"relevant group {i} appears in no indexed chunk", sections
    return None, sections


def run_case(case: dict, filing: dict, k: int) -> dict:
    start = time.time()
    passages = filings_store.search(filing["filing"], case["question"], k)  # same path as the search_filings tool
    elapsed_ms = (time.time() - start) * 1000
    rank = first_hit_rank(passages, case["gold_snippets"])
    groups = relevance_groups(case)
    return {
        "id": case["id"],
        "type": case["type"],
        "filing": case["filing"],
        "parts": "multi-part" if len(groups) > 1 else "single",
        "rank": rank,
        **score_passages(passages, groups, k),
        "top_section": passages[0]["section"] if passages else "",
        "top_section_ok": bool(passages) and section_matches(passages[0]["section"], case["gold_sections"]),
        "top_text": passages[0]["text"][:160].replace("\n", " ") if passages else "",
        "ms": round(elapsed_ms),
    }


def run_error_case(case: dict) -> dict:
    try:
        filing = edgar.find_filing(case["symbol"], case["form"], case["fiscal_year"])
    except edgar.EdgarError as exc:
        return {"id": case["id"], "passed": True, "detail": str(exc)}
    return {"id": case["id"], "passed": False, "detail": f"unexpectedly found {filing['form']} {filing['period']}"}


def print_metrics(label: str, metrics: dict, k: int, before: dict | None = None) -> None:
    """One line of metrics; with `before` (same k), each value is followed by its change, e.g. 0.57 (+0.22)."""
    if not metrics.get("n"):
        print(f"  {label:<22} (no cases)")
        return

    def fmt(name: str) -> str:
        value = f"{name}={metrics[name]:.2f}"
        if before and name in before:
            value += f" ({metrics[name] - before[name]:+.2f})"
        return value

    names = ["hit@1", "hit@3", f"hit@{k}", f"mrr@{k}", "section@1", f"recall@{k}", f"precision@{k}", f"map@{k}"]
    print(f"  {label:<22} n={metrics['n']:<3} " + "  ".join(fmt(name) for name in names if name in metrics))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=5, help="passages retrieved per question (default 5)")
    parser.add_argument("--type", help="only run cases of this question type, e.g. risk")
    parser.add_argument("--save", help="write the full results as JSON to this path")
    parser.add_argument("--compare", help="a saved results JSON (e.g. the baseline) to show changes against")
    args = parser.parse_args()

    with open(GOLD_PATH) as f:
        gold = json.load(f)
    cases = [c for c in gold["cases"] if not args.type or c["type"] == args.type]

    filings_store.FILINGS_DIR = EVAL_FILINGS_DIR
    filings_store.reset()

    print("=" * 72)
    print("Retrieval Eval — RAG over SEC filings")
    print("=" * 72)

    print("\nFiling lookup (first run downloads and indexes each filing):")
    start = time.time()
    filings = index_filings({key: gold["filings"][key] for key in {c["filing"] for c in cases}})
    for key, filing in sorted(filings.items()):
        if "error" in filing:
            print(f"  ❌ {key:<16} {filing['error']}")
        else:
            mark = "✅" if filing["ok"] else "❌"
            expected = gold["filings"][key]["period"]
            print(f"  {mark} {key:<16} period {filing['period']} (expected {expected}), {len(filing['chunks'])} chunks")
    print(f"  ({time.time() - start:.0f}s)")

    results, invalid, gold_warnings = [], [], []
    for case in cases:
        filing = filings[case["filing"]]
        if not filing.get("accession"):
            invalid.append((case["id"], "filing could not be indexed"))
            continue
        problem, sections = check_gold(case, filing["chunks"])
        if problem:
            invalid.append((case["id"], problem))
            continue
        if not any(section_matches(s, case["gold_sections"]) for s in sections):
            gold_warnings.append((case["id"], sections))
        results.append(run_case(case, filing, args.k))

    overall = summarize(results, args.k)
    by_type = summarize_by(results, "type", args.k)
    by_filing = summarize_by(results, "filing", args.k)
    by_parts = summarize_by(results, "parts", args.k)

    before = {}
    if args.compare:
        with open(args.compare) as f:
            before = json.load(f)
        if before.get("k") != args.k:
            print(f"\n(--compare ignored: {args.compare} used k={before.get('k')}, this run k={args.k})")
            before = {}

    print("\nRetrieval" + (f" (change vs {os.path.basename(args.compare)})" if before else "") + ":")
    print_metrics("OVERALL", overall, args.k, before.get("overall"))
    print("\n  By question type:")
    for name, metrics in by_type.items():
        print_metrics(name, metrics, args.k, before.get("by_type", {}).get(name))
    print("\n  By filing:")
    for name, metrics in by_filing.items():
        print_metrics(name, metrics, args.k, before.get("by_filing", {}).get(name))
    print("\n  By number of relevant passages (recall < hit means a part of the answer was not retrieved):")
    for name, metrics in by_parts.items():
        print_metrics(name, metrics, args.k, before.get("by_parts", {}).get(name))

    if before:
        old_ranks = {c["id"]: c["rank"] for c in before.get("cases", [])}
        changed = [(r["id"], old_ranks[r["id"]], r["rank"]) for r in results
                   if r["id"] in old_ranks and old_ranks[r["id"]] != r["rank"]]
        if changed:
            print("\n  Rank changes (None = not in top k):")
            for case_id, old, new in changed:
                worse = new is None or (old is not None and new > old)
                print(f"    {'▼' if worse else '▲'} {case_id}: {old} → {new}")
    if results:
        print(f"\n  Avg search latency: {sum(r['ms'] for r in results) / len(results):.0f} ms")

    misses = [r for r in results if r["rank"] is None]
    if misses:
        print(f"\nMisses (no gold passage in top {args.k}):")
        for r in misses:
            print(f"  - {r['id']}: top passage from [{r['top_section']}] \"{r['top_text'][:100]}…\"")

    wrong_section = [r for r in results if r["rank"] is not None and not r["top_section_ok"]]
    if wrong_section:
        print("\nFound, but the top passage is from an unexpected section:")
        for r in wrong_section:
            print(f"  - {r['id']}: rank {r['rank']}, top passage from [{r['top_section']}]")

    if gold_warnings:
        print("\nGold snippet found, but only in chunks labelled with an unexpected section (chunking issue?):")
        for case_id, sections in gold_warnings:
            print(f"  - {case_id}: {'; '.join(sections)}")

    if invalid:
        print("\nINVALID GOLD (left out of the scores — fix the gold set):")
        for case_id, reason in invalid:
            print(f"  - {case_id}: {reason}")

    error_results = [run_error_case(c) for c in gold.get("error_cases", [])] if not args.type else []
    if error_results:
        print("\nError cases (should fail cleanly):")
        for r in error_results:
            print(f"  {'✅' if r['passed'] else '❌'} {r['id']}: {r['detail']}")

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        with open(args.save, "w") as f:
            json.dump({
                "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "k": args.k,
                "chunk_chars": filings_store.CHUNK_CHARS,
                "overlap_chars": filings_store.OVERLAP_CHARS,
                "filings": {
                    key: {field: value for field, value in info.items() if field not in ("chunks", "filing")}
                    for key, info in filings.items()
                },
                "overall": overall,
                "by_type": by_type,
                "by_filing": by_filing,
                "by_parts": by_parts,
                "cases": results,
                "invalid_gold": invalid,
                "gold_section_warnings": gold_warnings,
                "error_cases": error_results,
            }, f, indent=2)
        print(f"\nSaved results to {args.save}")


if __name__ == "__main__":
    main()
