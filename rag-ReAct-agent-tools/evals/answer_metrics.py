"""Code-checked answer metrics — pure functions over the agent's history and reply, unit-tested offline.

The LLM judge (evals/judge.py) grades what code cannot (is a claim supported, is a fact covered, did the agent
abstain); everything here is deterministic: which filings and passages the agent actually retrieved, whether
the answer cites a form, a filing and real sections, and whether exact figures appear in the answer.
"""

import json
import re
from datetime import datetime

from retrieval_metrics import contains_any, normalize

# A case counts as "correct" when the answer covers at least this share of its key facts
CORRECT_THRESHOLD = 0.6

_ITEM_RE = re.compile(r"\bItem\s+(\d{1,2}[A-C]?)\b", re.I)
_ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_LONG_DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s+(\d{4})\b"
)


# ---------------------------------------------------------------------------
# What the agent saw
# ---------------------------------------------------------------------------

def tool_calls(history: list) -> list[dict]:
    """Every non-think tool call in the history: [{name, input, result, is_error}], in call order."""
    calls: dict[str, dict] = {}
    order: list[str] = []
    for message in history:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "tool_use" and block.get("name") != "think":
                calls[block["id"]] = {"name": block["name"], "input": block.get("input", {}), "result": None, "is_error": False}
                order.append(block["id"])
            elif block.get("type") == "tool_result" and block.get("tool_use_id") in calls:
                raw = block.get("content")
                try:
                    result = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    result = raw
                calls[block["tool_use_id"]].update(result=result, is_error=bool(block.get("is_error")))
    return [calls[i] for i in order]


def filing_searches(calls: list[dict]) -> list[dict]:
    """The successful search_filings results: [{filing: {...}, passages: [...]}]."""
    return [
        c["result"] for c in calls
        if c["name"] == "search_filings" and isinstance(c["result"], dict) and "passages" in c["result"]
    ]


def evidence_text(calls: list[dict]) -> str:
    """Everything the agent's tools returned this turn, formatted for the judge."""
    parts = []
    for i, call in enumerate(calls, start=1):
        header = f"[Tool call {i}] {call['name']}({json.dumps(call['input'])})"
        result = call["result"]
        if call["name"] == "search_filings" and isinstance(result, dict) and "passages" in result:
            f = result["filing"]
            lines = [f"{header}\nFiling: {f['symbol']} {f['form']}, period {f['period']}, filed {f['filing_date']}"]
            for j, p in enumerate(result["passages"], start=1):
                lines.append(f"  Passage {j} — section: {p['section']}\n  {p['text']}")
            parts.append("\n".join(lines))
        else:
            parts.append(f"{header}\nResult: {json.dumps(result)[:6000]}")
    return "\n\n".join(parts) if parts else "(the agent called no tools)"


# ---------------------------------------------------------------------------
# Citations and exact facts
# ---------------------------------------------------------------------------

def section_item(section: str) -> str | None:
    """'Part I — Item 1A. Risk Factors' -> '1A'."""
    match = _ITEM_RE.search(section)
    return match.group(1).upper() if match else None


def cited_items(answer: str) -> set[str]:
    return {m.upper() for m in _ITEM_RE.findall(answer)}


def named_sections(answer: str, searches: list[dict]) -> list[str]:
    """Retrieved sections the answer cites by name ("Risk Factors", "MD&A") rather than by Item number.

    A section counts when the first two words of its title appear in the answer; one-word titles such as
    "Business" are skipped because the word appears in ordinary text.
    """
    norm = normalize(answer)
    named = set()
    for section in {p["section"] for s in searches for p in s["passages"]}:
        title = normalize(re.sub(r"^.*?Item\s+\d{1,2}[A-C]?\.\s*", "", section))
        words = title.split()
        if len(words) >= 2 and " ".join(words[:2]) in norm:
            named.add(section)
        elif title.startswith("management's discussion") and re.search(r"\bmd&a\b", norm):
            named.add(section)
    return sorted(named)


def mentioned_dates(answer: str) -> set[str]:
    """Dates written as 2025-10-31 or October 31, 2025, normalised to ISO."""
    dates = set(_ISO_DATE_RE.findall(answer))
    for month, day, year in _LONG_DATE_RE.findall(answer):
        dates.add(datetime.strptime(f"{month} {day} {year}", "%B %d %Y").strftime("%Y-%m-%d"))
    return dates


def citation_checks(answer: str, searches: list[dict]) -> dict:
    """Does the answer cite a form, identify the filing it used, and cite only sections it actually retrieved?"""
    items = cited_items(answer)
    retrieved_items = {section_item(p["section"]) for s in searches for p in s["passages"]} - {None}
    dates = mentioned_dates(answer)
    norm = normalize(answer)

    def identifies(filing: dict) -> bool:
        fiscal_year = filing["period"][:4]
        return (
            filing["filing_date"] in dates
            or filing["period"] in dates
            or any(f in norm for f in (f"fiscal {fiscal_year}", f"fiscal year {fiscal_year}", f"fy{fiscal_year}", f"fy {fiscal_year}"))
        )

    return {
        "form_cited": bool(re.search(r"\b10-[KQ]\b", answer, re.I)),
        "items_cited": sorted(items),
        "sections_named": named_sections(answer, searches),
        "filing_identified": any(identifies(s["filing"]) for s in searches) if searches else None,
        # share of cited Items that the agent actually retrieved passages from (None if it cited none)
        "item_validity": (len(items & retrieved_items) / len(items)) if items and searches else None,
    }


def exact_fact_rate(answer: str, exact_facts: list[str]) -> float | None:
    """Share of exact strings (figures, names) that appear in the answer; None if the case has none."""
    if not exact_facts:
        return None
    norm = normalize(answer).replace(",", "")
    return sum(1 for f in exact_facts if normalize(f).replace(",", "") in norm) / len(exact_facts)


def retrieval_hit(searches: list[dict], gold_snippets: list[str]) -> bool:
    """Did any passage the agent retrieved contain a gold snippet?"""
    return any(contains_any(p["text"], gold_snippets) for s in searches for p in s["passages"])


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _rate(flags: list) -> float | None:
    flags = [f for f in flags if f is not None]
    return sum(1 for f in flags if f) / len(flags) if flags else None


def summarize(results: list[dict]) -> dict:
    """Headline metrics over per-case results (see run_answer_evals.grade_case for the fields)."""
    answerable = [r for r in results if r["kind"] == "answerable" and not r.get("error")]
    unanswerable = [r for r in results if r["kind"] == "unanswerable" and not r.get("error")]
    judged = [r for r in answerable if r.get("judge")]
    judged_u = [r for r in unanswerable if r.get("judge")]

    def coverage(r: dict) -> float | None:
        facts = r["judge"]["key_facts"]
        return sum(1 for f in facts if f["covered"]) / len(facts) if facts else None

    def faithfulness(r: dict) -> float | None:
        claims = r["judge"]["claims"]
        return sum(1 for c in claims if c["supported"]) / len(claims) if claims else None

    def citation_support(r: dict) -> float | None:
        cites = r["judge"]["citations"]
        return sum(1 for c in cites if c["supported_by_cited_section"]) / len(cites) if cites else None

    summary = {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "errors": sum(1 for r in results if r.get("error")),
        # retrieval inside the live agent
        "filing_ok": _rate([r["filing_ok"] for r in answerable]),
        "retrieval_hit": _rate([r["retrieval_hit"] for r in answerable]),
        # code-checked
        "exact_fact_rate": _mean([r["exact_fact_rate"] for r in answerable]),
        # names the form and at least one section, by Item number or by title
        "citation_present": _rate([
            r["citations"]["form_cited"] and bool(r["citations"]["items_cited"] or r["citations"]["sections_named"])
            for r in answerable
        ]),
        "filing_identified": _rate([r["citations"]["filing_identified"] for r in answerable]),
        "item_validity": _mean([r["citations"]["item_validity"] for r in answerable]),
    }
    if judged:
        coverages = [coverage(r) for r in judged]
        summary.update({
            "key_fact_coverage": _mean(coverages),
            "correct": _rate([c is not None and c >= CORRECT_THRESHOLD for c in coverages]),
            "faithfulness": _mean([faithfulness(r) for r in judged]),
            "fully_grounded": _rate([all(c["supported"] for c in r["judge"]["claims"]) for r in judged]),
            "citation_support": _mean([citation_support(r) for r in judged]),
            "false_abstention": _rate([r["judge"]["response_type"] == "abstained" for r in judged]),
        })
    if judged_u:
        summary.update({
            "correct_abstention": _rate([r["judge"]["response_type"] == "abstained" for r in judged_u]),
            "answered_unanswerable": sum(1 for r in judged_u if r["judge"]["response_type"] == "answered"),
        })
    return summary


def attribution(results: list[dict]) -> dict[str, int]:
    """Split answerable cases by (gold passage retrieved?, answer correct?) to locate failures."""
    table = {"retrieved & correct": 0, "retrieved & wrong (answer step)": 0,
             "missed & wrong (retrieval)": 0, "missed & correct (check grounding)": 0}
    for r in results:
        if r["kind"] != "answerable" or r.get("error") or not r.get("judge") or not r["judge"]["key_facts"]:
            continue
        facts = r["judge"]["key_facts"]
        correct = sum(1 for f in facts if f["covered"]) / len(facts) >= CORRECT_THRESHOLD
        if r["retrieval_hit"]:
            table["retrieved & correct" if correct else "retrieved & wrong (answer step)"] += 1
        else:
            table["missed & correct (check grounding)" if correct else "missed & wrong (retrieval)"] += 1
    return table
