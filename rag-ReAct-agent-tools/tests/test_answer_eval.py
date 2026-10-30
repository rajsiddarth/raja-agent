"""Unit tests for the answer eval's code checks, aggregation, judge wrapper and calibration — no network."""

import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evals"))
import judge
from answer_metrics import (
    attribution, citation_checks, evidence_text, exact_fact_rate, filing_searches, retrieval_hit, summarize,
    tool_calls,
)
from calibrate_judge import agreement

GOLD_PATH = os.path.join(os.path.dirname(__file__), "..", "evals", "filings_gold.json")

SEARCH_RESULT = {
    "filing": {"symbol": "AAPL", "form": "10-K", "filing_date": "2025-10-31", "period": "2025-09-27", "url": "u"},
    "passages": [
        {"section": "Part I — Item 1A. Risk Factors", "text": "The Company depends on outsourcing partners in China mainland.", "relevance": 0.7},
        {"section": "Part II — Item 7. Management's Discussion", "text": "Products gross margin percentage decreased.", "relevance": 0.6},
    ],
}

HISTORY = [
    {"role": "user", "content": "What supply chain risks does Apple cite?"},
    {"role": "assistant", "content": [
        {"type": "tool_use", "id": "t1", "name": "think", "input": {"reasoning": "search filings"}},
    ]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Reasoning noted."}]},
    {"role": "assistant", "content": [
        {"type": "tool_use", "id": "t2", "name": "search_filings", "input": {"symbol": "AAPL", "query": "supply chain"}},
        {"type": "tool_use", "id": "t3", "name": "get_quote", "input": {"symbol": "AAPL"}},
    ]},
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t2", "content": json.dumps(SEARCH_RESULT), "is_error": False},
        {"type": "tool_result", "tool_use_id": "t3", "content": json.dumps({"price": 250.0}), "is_error": False},
    ]},
    {"role": "assistant", "content": [{"type": "text", "text": "Answer"}]},
]


# ---------------------------------------------------------------------------
# What the agent saw
# ---------------------------------------------------------------------------

def test_tool_calls_pairs_results_and_skips_think():
    calls = tool_calls(HISTORY)
    assert [c["name"] for c in calls] == ["search_filings", "get_quote"]
    assert calls[0]["result"]["filing"]["period"] == "2025-09-27"
    assert calls[1]["result"] == {"price": 250.0}


def test_evidence_text_lists_passages_with_sections():
    text = evidence_text(tool_calls(HISTORY))
    assert "AAPL 10-K, period 2025-09-27, filed 2025-10-31" in text
    assert "section: Part I — Item 1A. Risk Factors" in text
    assert '"price": 250.0' in text


def test_retrieval_hit_checks_retrieved_passages():
    searches = filing_searches(tool_calls(HISTORY))
    assert retrieval_hit(searches, ["outsourcing partners in China mainland"])
    assert not retrieval_hit(searches, ["not in any passage"])


# ---------------------------------------------------------------------------
# Citations and exact facts
# ---------------------------------------------------------------------------

def test_citation_checks_valid_citation():
    answer = "Per Apple's 10-K filed October 31, 2025 (Item 1A. Risk Factors), it depends on partners in China."
    checks = citation_checks(answer, [SEARCH_RESULT])
    assert checks == {"form_cited": True, "items_cited": ["1A"], "sections_named": ["Part I — Item 1A. Risk Factors"],
                      "filing_identified": True, "item_validity": 1.0}


def test_citation_checks_flags_items_not_retrieved_and_accepts_fiscal_year():
    answer = "Apple's fiscal 2025 10-K discusses this in Item 1A and Item 3."
    checks = citation_checks(answer, [SEARCH_RESULT])
    assert checks["filing_identified"] is True
    assert checks["item_validity"] == 0.5  # Item 3 was never retrieved


def test_citation_checks_without_citations_or_searches():
    checks = citation_checks("Apple faces supply risks.", [])
    assert checks == {"form_cited": False, "items_cited": [], "sections_named": [], "filing_identified": None,
                      "item_validity": None}


def test_citation_checks_counts_sections_cited_by_name():
    searches = [{**SEARCH_RESULT, "passages": SEARCH_RESULT["passages"] + [
        {"section": "Part I — Item 1. Business", "text": "Apple designs products.", "relevance": 0.5}]}]
    answer = "Per the 10-K's MD&A and the Risk Factors section, the business depends on partners."
    checks = citation_checks(answer, searches)
    assert checks["items_cited"] == []
    assert checks["sections_named"] == ["Part I — Item 1A. Risk Factors", "Part II — Item 7. Management's Discussion"]


def test_exact_fact_rate_ignores_case_and_thousands_separators():
    assert exact_fact_rate("Apple had about 166000 full-time employees in China", ["166,000", "china", "India"]) == pytest.approx(2 / 3)
    assert exact_fact_rate("anything", []) is None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _result(kind="answerable", covered=(True, True), supported=(True,), response_type="answered", hit=True):
    return {
        "id": "x", "kind": kind, "type": "risk", "filing_ok": True, "retrieval_hit": hit, "exact_fact_rate": 1.0,
        "citations": {"form_cited": True, "items_cited": ["1A"], "sections_named": [], "filing_identified": True,
                      "item_validity": 1.0},
        "judge": {
            "response_type": response_type,
            "claims": [{"claim": "c", "supported": s, "evidence": ""} for s in supported],
            "key_facts": [{"fact": "f", "covered": c} for c in covered],
            "citations": [{"citation": "Item 1A", "attributed_claim": "c", "supported_by_cited_section": True}],
        },
    }


def test_summarize_answerable_and_unanswerable():
    results = [
        _result(),
        _result(covered=(True, False), supported=(True, False), hit=False),
        _result(kind="unanswerable", covered=(), response_type="abstained"),
        _result(kind="unanswerable", covered=(), response_type="answered"),
        {"id": "e", "kind": "answerable", "type": "risk", "error": "agent failed"},
    ]
    s = summarize(results)
    assert s["n_answerable"] == 2 and s["n_unanswerable"] == 2 and s["errors"] == 1
    assert s["correct"] == 0.5             # 2/2 facts passes, 1/2 is below the 60% threshold
    assert s["key_fact_coverage"] == 0.75
    assert s["faithfulness"] == 0.75       # (1.0 + 0.5) / 2
    assert s["fully_grounded"] == 0.5
    assert s["retrieval_hit"] == 0.5
    assert s["correct_abstention"] == 0.5
    assert s["answered_unanswerable"] == 1
    assert s["false_abstention"] == 0.0


def test_attribution_splits_retrieval_and_answer_failures():
    table = attribution([
        _result(),
        _result(covered=(False, False)),
        _result(covered=(False, False), hit=False),
        _result(hit=False),
    ])
    assert table == {"retrieved & correct": 1, "retrieved & wrong (answer step)": 1,
                     "missed & wrong (retrieval)": 1, "missed & correct (check grounding)": 1}


# ---------------------------------------------------------------------------
# Judge wrapper (mocked client)
# ---------------------------------------------------------------------------

def _response(text=None, stop_reason="end_turn", category=None):
    content = [SimpleNamespace(type="text", text=text)] if text is not None else []
    return SimpleNamespace(
        content=content, stop_reason=stop_reason, model="claude-opus-5-5",
        stop_details=SimpleNamespace(category=category) if category else None,
        usage=SimpleNamespace(input_tokens=1200, output_tokens=300),
    )


def test_judge_answer_parses_grade_and_usage():
    grade = {"response_type": "answered", "claims": [], "key_facts": [{"fact": "f", "covered": True}],
             "citations": [], "notes": ""}
    client = MagicMock()
    client.beta.messages.create.return_value = _response(json.dumps(grade))
    with patch.object(judge, "_client", client):
        result = judge.judge_answer("q", "a", "evidence", ["f"])
    kwargs = client.beta.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["output_config"]["format"]["schema"] == judge.JUDGE_SCHEMA
    assert kwargs["fallbacks"] == "default"
    assert "<key_facts>\n- f\n</key_facts>" in kwargs["messages"][0]["content"]
    assert result["key_facts"] == [{"fact": "f", "covered": True}]
    assert result["usage"] == {"input_tokens": 1200, "output_tokens": 300}


@pytest.mark.parametrize("response, message", [
    (_response(stop_reason="refusal", category="cyber"), "refused"),
    (_response("{not json"), "invalid JSON"),
    (_response(stop_reason="max_tokens"), "max_tokens"),
])
def test_judge_answer_raises_on_unusable_output(response, message):
    client = MagicMock()
    client.beta.messages.create.return_value = response
    with patch.object(judge, "_client", client), pytest.raises(judge.JudgeError, match=message):
        judge.judge_answer("q", "a", "evidence", [])


# ---------------------------------------------------------------------------
# Calibration and gold set
# ---------------------------------------------------------------------------

def test_calibration_agreement_counts_only_labelled_fields():
    items = [{
        "id": "a", "key_facts": ["f1", "f2"],
        "judge": {"response_type": "answered", "key_facts_covered": [True, False], "has_unsupported_claim": False},
        "human": {"response_type": "answered", "key_facts_covered": [True, True], "has_unsupported_claim": None},
    }]
    stats = agreement(items)
    assert stats["response_type"][:2] == (1, 1)
    assert stats["key_facts_covered"][:2] == (1, 2)
    assert stats["has_unsupported_claim"][:2] == (0, 0)
    assert "f2" in stats["key_facts_covered"][2][0]


def test_gold_set_has_answer_fields():
    with open(GOLD_PATH) as f:
        gold = json.load(f)
    for case in gold["cases"]:
        assert case["key_facts"], case["id"]
        assert isinstance(case["exact_facts"], list), case["id"]
    ids = [c["id"] for c in gold["cases"] + gold["unanswerable_cases"]]
    assert len(ids) == len(set(ids))
    for case in gold["unanswerable_cases"]:
        assert case["filing"] is None or case["filing"] in gold["filings"], case["id"]
        assert case["type"] in ("not_in_filing", "false_premise", "no_filing")
