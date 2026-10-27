"""Unit tests for edgar.py, filings_store.py and the search_filings tool — no network, offline embedding."""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import edgar
from filings_store import CHUNK_CHARS, chunk_text, split_sections
from tools import search_filings

FILING_TEXT = "\n".join([
    "UNITED STATES SECURITIES AND EXCHANGE COMMISSION",
    "Form 10-K for Example Corp, a maker of widgets and consumer devices.",
    "PART I",
    "Item 1.",
    "Business",
    "Example Corp designs and sells widgets to consumers and businesses worldwide through retail and online channels.",
    "Item 1A. Risk Factors",
    "The Company depends on a small number of suppliers in Asia for key components, and supply chain disruption could harm results.",
    "Competition in the widget market is intense and prices may decline over time, reducing gross margin.",
    "PART II",
    "Item 7. Management's Discussion and Analysis",
    "Net sales increased 12% year over year, driven by strong demand for premium widgets in the Americas.",
])

FILING = {
    "symbol": "EXM", "cik": 123, "form": "10-K", "accession": "0000123-25-000001",
    "filing_date": "2025-11-01", "period": "2025-09-30", "url": "https://www.sec.gov/Archives/x.htm",
}


# ---------------------------------------------------------------------------
# edgar.py
# ---------------------------------------------------------------------------

def test_html_to_text_drops_hidden_xbrl_and_scripts():
    html = """<html><head><title>t</title></head><body>
      <ix:header><ix:hidden>dei:EntityName secret</ix:hidden></ix:header>
      <div style="display:none">hidden facts</div>
      <script>var x = 1;</script>
      <p>Item 1A.   Risk   Factors</p><table><tr><td>Revenue</td><td>$100</td></tr></table>
    </body></html>"""
    text = edgar.html_to_text(html)
    assert "Item 1A. Risk Factors" in text
    assert "Revenue" in text and "$100" in text
    for hidden in ("secret", "hidden facts", "var x"):
        assert hidden not in text


def test_missing_user_agent_raises_clear_error(monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(edgar.EdgarError, match="SEC_USER_AGENT"):
        edgar._user_agent()


def test_lookup_cik_uses_cached_ticker_list(tmp_path, monkeypatch):
    monkeypatch.setattr(edgar, "CACHE_DIR", str(tmp_path))
    (tmp_path / "company_tickers.json").write_text(json.dumps({"0": {"cik_str": 320193, "ticker": "AAPL"}}))
    with patch("edgar._get") as get:
        assert edgar.lookup_cik("aapl") == 320193
        get.assert_not_called()
    with pytest.raises(edgar.EdgarError, match="No SEC registrant"):
        edgar.lookup_cik("NOPE")


def test_latest_filing_picks_most_recent_matching_form():
    submissions = {"filings": {"recent": {
        "form": ["8-K", "10-Q", "10-K", "10-K"],
        "accessionNumber": ["a-1", "0000320193-25-000010", "0000320193-24-000123", "old"],
        "filingDate": ["2025-12-01", "2025-08-01", "2024-11-01", "2023-11-01"],
        "reportDate": ["", "2025-06-30", "2024-09-28", "2023-09-30"],
        "primaryDocument": ["x.htm", "q.htm", "aapl-20240928.htm", "old.htm"],
    }}}
    response = MagicMock()
    response.json.return_value = submissions
    with patch("edgar.lookup_cik", return_value=320193), patch("edgar._get", return_value=response):
        filing = edgar.latest_filing("aapl", "10-K")
    assert filing["accession"] == "0000320193-24-000123"
    assert filing["filing_date"] == "2024-11-01"
    assert filing["url"] == "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"


def test_latest_filing_rejects_unsupported_form():
    with pytest.raises(edgar.EdgarError, match="Unsupported form"):
        edgar.latest_filing("AAPL", "8-K")


# ---------------------------------------------------------------------------
# Sections and chunks
# ---------------------------------------------------------------------------

def test_split_sections_tracks_part_and_item():
    names = [name for name, _ in split_sections(FILING_TEXT)]
    assert names == [
        "Cover",
        "Part I — Item 1. Business",
        "Part I — Item 1A. Risk Factors",
        "Part II — Item 7. Management's Discussion and Analysis",
    ]


def test_chunks_respect_size_and_never_cross_sections():
    long_risk = " ".join(f"Risk sentence number {i} describes a supplier dependency." for i in range(200))
    text = FILING_TEXT.replace("Item 1A. Risk Factors", "Item 1A. Risk Factors\n" + long_risk)
    chunks = chunk_text(text)
    assert all(len(c["text"]) <= CHUNK_CHARS for c in chunks)
    risk_chunks = [c for c in chunks if c["section"].endswith("Risk Factors")]
    assert len(risk_chunks) > 5
    assert not any("Net sales increased" in c["text"] for c in risk_chunks)


def test_consecutive_chunks_overlap():
    lines = [f"Line {i}: the company relies on outsourced manufacturing partners." for i in range(60)]
    chunks = chunk_text("Item 1A. Risk Factors\n" + "\n".join(lines))
    first_last_line = chunks[0]["text"].splitlines()[-1]
    assert first_last_line in chunks[1]["text"]


# ---------------------------------------------------------------------------
# Index and search
# ---------------------------------------------------------------------------

def test_index_and_search_finds_risk_section(filings_db):
    count = filings_db.index_filing(FILING, FILING_TEXT)
    assert count == len(chunk_text(FILING_TEXT))
    assert filings_db.is_indexed(FILING["accession"])
    results = filings_db.search_filing("suppliers supply chain disruption risk", FILING["accession"], 1)
    assert results[0]["section"] == "Part I — Item 1A. Risk Factors"
    assert "suppliers in Asia" in results[0]["text"]
    assert not results[0]["text"].startswith("Part I")  # heading is stripped from the returned text


def test_reindexing_does_not_duplicate(filings_db):
    filings_db.index_filing(FILING, FILING_TEXT)
    filings_db.index_filing(FILING, FILING_TEXT)
    assert filings_db.indexed_filings() == [
        {"symbol": "EXM", "form": "10-K", "filing_date": "2025-11-01", "chunks": len(chunk_text(FILING_TEXT))}
    ]


def test_ensure_latest_indexed_downloads_only_once(filings_db):
    with patch("edgar.latest_filing", return_value=dict(FILING)) as latest, \
         patch("edgar.fetch_filing_text", return_value=FILING_TEXT) as fetch:
        filings_db.ensure_latest_indexed("EXM")
        filings_db.ensure_latest_indexed("EXM")
    latest.assert_called_once()
    fetch.assert_called_once()


# ---------------------------------------------------------------------------
# search_filings tool
# ---------------------------------------------------------------------------

def test_search_filings_tool_returns_filing_and_passages(filings_db):
    with patch("edgar.latest_filing", return_value=dict(FILING)), \
         patch("edgar.fetch_filing_text", return_value=FILING_TEXT):
        result = search_filings({"symbol": "EXM", "query": "competition prices gross margin", "n_results": 2})
    assert result["filing"]["form"] == "10-K"
    assert result["filing"]["filing_date"] == "2025-11-01"
    assert len(result["passages"]) == 2
    assert "Competition" in result["passages"][0]["text"]


def test_search_filings_tool_reports_edgar_errors(filings_db):
    with patch("edgar.latest_filing", side_effect=edgar.EdgarError("No recent 10-K found for XYZ.")):
        result = search_filings({"symbol": "XYZ", "query": "risks"})
    assert result == {"error": "No recent 10-K found for XYZ."}


def test_search_filings_tool_defaults_bad_form_to_10k(filings_db):
    with patch("filings_store.ensure_latest_indexed", side_effect=edgar.EdgarError("stop")) as ensure:
        search_filings({"symbol": "EXM", "query": "risks", "form": "8-K"})
    ensure.assert_called_once_with("EXM", "10-K")
