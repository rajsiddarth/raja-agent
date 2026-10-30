"""
Measure how often the LLM judge agrees with your own labels — run before trusting judge-based scores.

1. Export answers with the judge's labels:
       python evals/run_answer_evals.py --export-labels evals/results/labels.json
2. Open the file and fill in each "human" block without looking at the "judge" block:
       response_type          "answered", "abstained" or "partial"
       key_facts_covered      true/false for each key fact, in order
       has_unsupported_claim  true if any claim in the answer is not backed by the evidence
   Leave a field as null to skip it. About 20 answers is enough for a first check.
3. Run:
       python evals/calibrate_judge.py evals/results/labels.json

Agreement below ~90% on a label means the judge prompt (evals/judge.py) needs work before that metric
can be trusted; the disagreements are listed so you can see which way the judge errs.
"""

import json
import sys


def agreement(items: list[dict]) -> dict[str, tuple[int, int, list[str]]]:
    """{label: (agreed, compared, [disagreement descriptions])} over the human-labelled fields."""
    stats = {"response_type": [0, 0, []], "key_facts_covered": [0, 0, []], "has_unsupported_claim": [0, 0, []]}
    for item in items:
        human, judge = item["human"], item["judge"]
        for label in ("response_type", "has_unsupported_claim"):
            if human[label] is None:
                continue
            stats[label][1] += 1
            if human[label] == judge[label]:
                stats[label][0] += 1
            else:
                stats[label][2].append(f"{item['id']}: you {human[label]}, judge {judge[label]}")
        for i, (h, j) in enumerate(zip(human["key_facts_covered"], judge["key_facts_covered"])):
            if h is None:
                continue
            stats["key_facts_covered"][1] += 1
            if h == j:
                stats["key_facts_covered"][0] += 1
            else:
                fact = item["key_facts"][i]
                stats["key_facts_covered"][2].append(f"{item['id']}: \"{fact}\": you {h}, judge {j}")
    return {label: (s[0], s[1], s[2]) for label, s in stats.items()}


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    with open(sys.argv[1]) as f:
        items = json.load(f)

    print("Judge agreement with your labels")
    for label, (agreed, compared, disagreements) in agreement(items).items():
        if not compared:
            print(f"  {label:<24} (no labels yet)")
            continue
        print(f"  {label:<24} {agreed}/{compared} = {agreed / compared:.0%}")
        for d in disagreements:
            print(f"      - {d}")


if __name__ == "__main__":
    main()
