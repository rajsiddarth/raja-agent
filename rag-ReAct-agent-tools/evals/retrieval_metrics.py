"""Scoring for the retrieval eval — pure functions, no network or vector DB, so they are unit-tested offline."""

import re

_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def normalize(text: str) -> str:
    """Lowercase, straighten curly quotes and collapse whitespace, so gold snippets match across line breaks."""
    return re.sub(r"\s+", " ", text.translate(_QUOTES)).strip().lower()


def contains_any(text: str, snippets: list[str]) -> bool:
    norm = normalize(text)
    return any(normalize(s) in norm for s in snippets)


def first_hit_rank(passages: list[dict], snippets: list[str]) -> int | None:
    """1-based rank of the first passage containing a gold snippet, or None if none does."""
    for rank, passage in enumerate(passages, start=1):
        if contains_any(passage["text"], snippets):
            return rank
    return None


def relevance_groups(case: dict) -> list[list[str]]:
    """Every distinct passage a complete answer needs, each as a group of alternative snippets.

    Cases without a "relevant" field have one group, their gold_snippets; for them recall@k equals hit@k
    and average precision equals the reciprocal rank.
    """
    return case.get("relevant") or [case["gold_snippets"]]


def score_passages(passages: list[dict], groups: list[list[str]], k: int) -> dict:
    """recall@k, precision@k and average precision@k for one question's ranked passages.

    - recall    : share of the relevant groups found in the top k
    - precision : share of the top k passages that match any relevant group
    - ap        : mean, over the relevant groups, of precision at the rank where each group is first found
                  (0 for groups not found), divided by min(len(groups), k) so a perfect ranking scores 1.0
    """
    top = passages[:k]
    first_rank: dict[int, int] = {}
    relevant_flags = []
    for rank, passage in enumerate(top, start=1):
        matched = [g for g, snippets in enumerate(groups) if contains_any(passage["text"], snippets)]
        relevant_flags.append(bool(matched))
        for g in matched:
            first_rank.setdefault(g, rank)

    def precision_at(rank: int) -> float:
        return sum(relevant_flags[:rank]) / rank

    return {
        "recall": len(first_rank) / len(groups),
        "precision": sum(relevant_flags) / len(top) if top else 0.0,
        "ap": sum(precision_at(r) for r in first_rank.values()) / min(len(groups), k),
    }


def section_matches(section: str, gold_sections: list[str]) -> bool:
    """True if the section label contains one of the expected labels, e.g. 'Item 1A. Risk Factors'."""
    norm = normalize(section)
    return any(normalize(g) in norm for g in gold_sections)


def summarize(results: list[dict], k: int) -> dict:
    """Aggregate per-case results into hit@1/3/k, MRR@k, section@1 and, when present, recall/precision/MAP@k."""
    n = len(results)
    if n == 0:
        return {"n": 0}
    ranks = [r["rank"] for r in results]
    summary = {
        "n": n,
        "hit@1": sum(1 for r in ranks if r is not None and r <= 1) / n,
        "hit@3": sum(1 for r in ranks if r is not None and r <= 3) / n,
        f"hit@{k}": sum(1 for r in ranks if r is not None and r <= k) / n,
        f"mrr@{k}": sum(1 / r for r in ranks if r is not None and r <= k) / n,
        "section@1": sum(1 for r in results if r["top_section_ok"]) / n,
    }
    if all("recall" in r for r in results):
        summary.update({
            f"recall@{k}": sum(r["recall"] for r in results) / n,
            f"precision@{k}": sum(r["precision"] for r in results) / n,
            f"map@{k}": sum(r["ap"] for r in results) / n,
        })
    return summary


def summarize_by(results: list[dict], key: str, k: int) -> dict[str, dict]:
    """summarize() for each value of results[key], e.g. per question type or per filing."""
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(r[key], []).append(r)
    return {name: summarize(group, k) for name, group in sorted(groups.items())}
