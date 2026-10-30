"""Unit tests for the retrieval eval's scoring and gold set — no network, offline embedding."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evals"))
from retrieval_metrics import first_hit_rank, normalize, section_matches, summarize, summarize_by

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
