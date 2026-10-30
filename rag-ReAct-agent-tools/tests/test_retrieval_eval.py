"""Unit tests for the retrieval eval's scoring and gold set — no network, offline embedding."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evals"))
from retrieval_metrics import (
    first_hit_rank, normalize, relevance_groups, score_passages, section_matches, summarize, summarize_by,
)

GOLD_PATH = os.path.join(os.path.dirname(__file__), "..", "evals", "filings_gold.json")


def test_normalize_ignores_case_whitespace_and_curly_quotes():
    assert normalize("The  Company’s\nfiscal “year”") == "the company's fiscal \"year\""


def test_first_hit_rank_finds_snippet_split_across_lines():
    passages = [
        {"text": "Unrelated passage about dividends."},
        {"text": "Sole-sourced third-party\nvendors in China perform final assembly of portable products."},
    ]
    assert first_hit_rank(passages, ["no match", "vendors in China perform final assembly"]) == 2
    assert first_hit_rank(passages, ["not there"]) is None


def test_section_matches_does_not_confuse_item_1_and_item_1a():
    assert section_matches("Part I — Item 1A. Risk Factors", ["Item 1A. Risk Factors"])
    assert not section_matches("Part I — Item 1A. Risk Factors", ["Item 1. Business"])
    assert section_matches("Part II — Item 1. Legal Proceedings", ["Part II — Item 1. Legal"])
    assert not section_matches("Part I — Item 1. Financial Statements", ["Part II — Item 1. Legal"])


def test_summarize_computes_hit_rates_and_mrr():
    results = [
        {"rank": 1, "top_section_ok": True, "type": "risk"},
        {"rank": 2, "top_section_ok": False, "type": "risk"},
        {"rank": 5, "top_section_ok": True, "type": "legal"},
        {"rank": None, "top_section_ok": False, "type": "legal"},
    ]
    metrics = summarize(results, k=5)
    assert metrics["n"] == 4
    assert metrics["hit@1"] == 0.25
    assert metrics["hit@3"] == 0.5
    assert metrics["hit@5"] == 0.75
    assert metrics["mrr@5"] == pytest.approx((1 + 0.5 + 0.2) / 4)
    assert metrics["section@1"] == 0.5
    by_type = summarize_by(results, "type", k=5)
    assert by_type["risk"]["hit@1"] == 0.5 and by_type["legal"]["hit@1"] == 0.0


def _passages(*texts):
    return [{"text": t} for t in texts]


def test_score_passages_single_group_matches_hit_and_reciprocal_rank():
    # one relevant passage, found at rank 2: recall 1 (= hit@5), AP 1/2 (= reciprocal rank)
    scores = score_passages(_passages("noise", "TSMC makes our wafers", "noise", "noise", "noise"), [["TSMC"]], k=5)
    assert scores == {"recall": 1.0, "precision": 0.2, "ap": 0.5}


def test_score_passages_multi_part_question():
    # three relevant parts; parts A and B found at ranks 1 and 3, part C never found
    groups = [["single-source"], ["limited sources"], ["price increases"]]
    passages = _passages("single-source partners", "noise", "limited sources of supply", "noise", "noise")
    scores = score_passages(passages, groups, k=5)
    assert scores["recall"] == pytest.approx(2 / 3)
    assert scores["precision"] == pytest.approx(2 / 5)
    # precision at rank 1 = 1/1, at rank 3 = 2/3; the missing part contributes 0; divided by 3 parts
    assert scores["ap"] == pytest.approx((1 + 2 / 3) / 3)


def test_score_passages_counts_a_part_once_and_handles_alternatives():
    groups = [["Item 1 wording", "Item 8 wording"], ["other part"]]
    passages = _passages("Item 8 wording here", "Item 1 wording here", "other part", "x", "y")
    scores = score_passages(passages, groups, k=5)
    assert scores["recall"] == 1.0
    assert scores["precision"] == pytest.approx(3 / 5)  # both copies of part 1 are relevant passages
    assert scores["ap"] == pytest.approx((1 + 1) / 2)  # part 1 first at rank 1 (P=1), part 2 at rank 3 (P=3/3)


def test_score_passages_with_nothing_found():
    assert score_passages(_passages("a", "b"), [["zzz"]], k=5) == {"recall": 0.0, "precision": 0.0, "ap": 0.0}


def test_relevance_groups_defaults_to_gold_snippets():
    assert relevance_groups({"gold_snippets": ["a", "b"]}) == [["a", "b"]]
    assert relevance_groups({"gold_snippets": ["a"], "relevant": [["a"], ["c"]]}) == [["a"], ["c"]]


def test_summarize_includes_recall_precision_map_when_scored():
    results = [
        {"rank": 1, "top_section_ok": True, "recall": 1.0, "precision": 0.4, "ap": 1.0},
        {"rank": None, "top_section_ok": False, "recall": 0.5, "precision": 0.2, "ap": 0.25},
    ]
    metrics = summarize(results, k=5)
    assert metrics["recall@5"] == 0.75
    assert metrics["precision@5"] == pytest.approx(0.3)
    assert metrics["map@5"] == 0.625


def test_summarize_handles_no_cases():
    assert summarize([], k=5) == {"n": 0}


def test_gold_set_is_well_formed():
    with open(GOLD_PATH) as f:
        gold = json.load(f)
    ids = [c["id"] for c in gold["cases"]] + [c["id"] for c in gold["error_cases"]]
    assert len(ids) == len(set(ids)), "case ids must be unique"
    for case in gold["cases"]:
        assert case["filing"] in gold["filings"], case["id"]
        assert case["question"] and case["gold_snippets"] and case["gold_sections"], case["id"]
    for case in gold["cases"]:
        for group in case.get("relevant", []):
            assert group and all(isinstance(snippet, str) for snippet in group), case["id"]
    for spec in gold["filings"].values():
        assert spec["fiscal_year"] and spec["period"].startswith(str(spec["fiscal_year"]))


def test_filing_chunks_returns_text_without_heading(filings_db):
    filing = {
        "symbol": "EXM", "form": "10-K", "accession": "0000123-25-000001",
        "filing_date": "2025-11-01", "period": "2025-09-30", "url": "https://www.sec.gov/x.htm",
    }
    text = "Item 1A. Risk Factors\n" + "The Company depends on a small number of suppliers in Asia for components. " * 3
    filings_db.index_filing(filing, text)
    chunks = filings_db.filing_chunks(filing["accession"])
    assert chunks and all(c["section"] == "Item 1A. Risk Factors" for c in chunks)
    assert chunks[0]["text"].startswith("The Company depends")
