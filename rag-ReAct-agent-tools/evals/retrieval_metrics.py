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


def section_matches(section: str, gold_sections: list[str]) -> bool:
    """True if the section label contains one of the expected labels, e.g. 'Item 1A. Risk Factors'."""
    norm = normalize(section)
    return any(normalize(g) in norm for g in gold_sections)


def summarize(results: list[dict], k: int) -> dict:
    """Aggregate per-case results ({rank, top_section_ok}) into hit@1/3/k, MRR@k and section@1."""
    n = len(results)
    if n == 0:
        return {"n": 0}
    ranks = [r["rank"] for r in results]
    return {
        "n": n,
        "hit@1": sum(1 for r in ranks if r is not None and r <= 1) / n,
        "hit@3": sum(1 for r in ranks if r is not None and r <= 3) / n,
        f"hit@{k}": sum(1 for r in ranks if r is not None and r <= k) / n,
        f"mrr@{k}": sum(1 / r for r in ranks if r is not None and r <= k) / n,
        "section@1": sum(1 for r in results if r["top_section_ok"]) / n,
    }


def summarize_by(results: list[dict], key: str, k: int) -> dict[str, dict]:
    """summarize() for each value of results[key], e.g. per question type or per filing."""
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(r[key], []).append(r)
    return {name: summarize(group, k) for name, group in sorted(groups.items())}
