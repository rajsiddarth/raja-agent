"""Unit tests for edgar.py, filings_store.py and the search_filings tool — no network, offline embedding."""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import edgar
from filings_store import CHUNK_CHARS, chunk_text, split_sections, strip_company
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


RECENT = {
    "form": ["8-K", "10-Q", "10-K", "10-K/A", "10-K"],
    "accessionNumber": ["a-1", "0000320193-25-000010", "0000320193-24-000123", "amend", "0000320193-23-000106"],
    "filingDate": ["2025-12-01", "2025-08-01", "2024-11-01", "2024-01-15", "2023-11-03"],
    "reportDate": ["", "2025-06-30", "2024-09-28", "2023-09-30", "2023-09-30"],
    "primaryDocument": ["x.htm", "q.htm", "aapl-20240928.htm", "amend.htm", "aapl-20230930.htm"],
}
OLDER_PAGE = {
    "form": ["10-K", "4", "10-K405"],
    "accessionNumber": ["0000320193-21-000105", "f4", "0000320193-01-000005"],
    "filingDate": ["2021-10-29", "2019-05-01", "2001-12-21"],
    "reportDate": ["2021-09-25", "", "2001-09-29"],
    "primaryDocument": ["aapl-20210925.htm", "f4.xml", "d10k405.txt"],
}


APPLE_ENTRY = {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}


def _submissions_get(pages: dict):
    """Fake edgar._get that serves the main submissions file and older pages by URL."""
    def get(url: str) -> MagicMock:
        response = MagicMock()
        response.json.return_value = pages[url.rsplit("/", 1)[1]]
        return response
    return get


SUBMISSIONS = {
    "CIK0000320193.json": {"filings": {"recent": RECENT, "files": [{"name": "CIK0000320193-submissions-001.json"}]}},
    "CIK0000320193-submissions-001.json": OLDER_PAGE,
}


def test_find_filing_picks_most_recent_matching_form():
    with patch("edgar._ticker_entry", return_value=APPLE_ENTRY), patch("edgar._get", side_effect=_submissions_get(SUBMISSIONS)) as get:
        filing = edgar.find_filing("aapl", "10-K")
    assert filing["accession"] == "0000320193-24-000123"
    assert filing["filing_date"] == "2024-11-01"
    assert filing["company"] == "Apple Inc."
    assert filing["url"] == "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
    assert get.call_count == 1  # older pages are not fetched when the recent list has the answer


def test_find_filing_by_fiscal_year_skips_amendments():
    with patch("edgar._ticker_entry", return_value=APPLE_ENTRY), patch("edgar._get", side_effect=_submissions_get(SUBMISSIONS)):
        filing = edgar.find_filing("AAPL", "10-K", fiscal_year=2023)
    assert filing["accession"] == "0000320193-23-000106"
    assert filing["period"] == "2023-09-30"


def test_find_filing_reads_older_pages_and_10k405():
    with patch("edgar._ticker_entry", return_value=APPLE_ENTRY), patch("edgar._get", side_effect=_submissions_get(SUBMISSIONS)) as get:
        assert edgar.find_filing("AAPL", "10-K", fiscal_year=2021)["accession"] == "0000320193-21-000105"
        filing = edgar.find_filing("AAPL", "10-K", fiscal_year=2001)
    assert filing["accession"] == "0000320193-01-000005"
    assert filing["form"] == "10-K"
    assert get.call_count == 4


def test_find_filing_stops_early_and_lists_years_seen():
    with patch("edgar._ticker_entry", return_value=APPLE_ENTRY), patch("edgar._get", side_effect=_submissions_get(SUBMISSIONS)) as get:
        with pytest.raises(edgar.EdgarError, match=r"fiscal year 2025 .*seen.*: 2024\."):
            edgar.find_filing("AAPL", "10-K", fiscal_year=2025)
    assert get.call_count == 1  # filings dated before 2025 cannot be for fiscal 2025


def test_find_filing_rejects_unsupported_form():
    with pytest.raises(edgar.EdgarError, match="Unsupported form"):
        edgar.find_filing("AAPL", "8-K")


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


@pytest.mark.parametrize("query, company, symbol, expected", [
    ("What risks does NVIDIA cite?", "NVIDIA CORP", "NVDA", "What risks does cite?"),
    ("Which companies make Nvidia's chips?", "NVIDIA CORP", "NVDA", "Which companies make chips?"),
    ("How does Apple describe AAPL competition in Apple’s 10-K?", "Apple Inc.", "AAPL", "How does describe competition in 10-K?"),
    ("Meta Platforms antitrust risk", "Meta Platforms, Inc.", "META", "antitrust risk"),
    ("Does NVIDIA Corp's 10-K mention Apple Inc. as a customer?", "NVIDIA CORP", "NVDA", "Does 10-K mention Apple Inc. as a customer?"),
    ("What does Apple Inc. say about tariffs?", "Apple Inc.", "AAPL", "What does say about tariffs?"),
    ("What does Coca-Cola say about sugar taxes?", "COCA COLA CO", "KO", "What does say about sugar taxes?"),
    # lower-case words that happen to be the company name, and one-letter tickers, are kept
    ("Did Target miss its revenue target?", "TARGET CORP", "TGT", "Did miss its revenue target?"),
    ("A summary of supply risks", "AGILENT TECHNOLOGIES, INC.", "A", "A summary of supply risks"),
    # a query that is only the name is left as it is rather than emptied
    ("NVIDIA", "NVIDIA CORP", "NVDA", "NVIDIA"),
])
def test_strip_company_removes_name_and_ticker(query, company, symbol, expected):
    assert strip_company(query, {"company": company, "symbol": symbol}) == expected


def test_strip_company_without_company_name_only_drops_ticker():
    assert strip_company("EXM supply chain risks", {"symbol": "EXM"}) == "supply chain risks"


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


def test_search_strips_company_name_before_querying(filings_db):
    filing = dict(FILING, company="EXAMPLE CORP")
    filings_db.index_filing(filing, FILING_TEXT)
    with patch.object(filings_db, "search_filing", wraps=filings_db.search_filing) as search_filing:
        results = filings_db.search(filing, "What supply chain risks does Example Corp describe?", 1)
    search_filing.assert_called_once_with("What supply chain risks does describe?", FILING["accession"], 1)
    assert results[0]["section"] == "Part I — Item 1A. Risk Factors"


def test_reindexing_does_not_duplicate(filings_db):
    filings_db.index_filing(FILING, FILING_TEXT)
    filings_db.index_filing(FILING, FILING_TEXT)
    assert filings_db.indexed_filings() == [
        {"symbol": "EXM", "form": "10-K", "filing_date": "2025-11-01", "period": "2025-09-30", "chunks": len(chunk_text(FILING_TEXT))}
    ]


def test_ensure_indexed_downloads_only_once(filings_db):
    with patch("edgar.find_filing", return_value=dict(FILING)) as find, \
         patch("edgar.fetch_filing_text", return_value=FILING_TEXT) as fetch:
        filings_db.ensure_indexed("EXM")
        filings_db.ensure_indexed("EXM")
    find.assert_called_once()
    fetch.assert_called_once()


def test_ensure_indexed_asks_edgar_again_after_ttl(filings_db, monkeypatch):
    newer = dict(FILING, accession="0000123-26-000001", filing_date="2026-11-01", period="2026-09-30")
    with patch("edgar.find_filing", side_effect=[dict(FILING), newer]) as find, \
         patch("edgar.fetch_filing_text", return_value=FILING_TEXT) as fetch:
        assert filings_db.ensure_indexed("EXM")["accession"] == FILING["accession"]
        monkeypatch.setattr(filings_db, "LOOKUP_TTL_S", 0)
        assert filings_db.ensure_indexed("EXM")["accession"] == newer["accession"]
    assert find.call_count == 2
    assert fetch.call_count == 2  # the new filing is downloaded and indexed
    assert [f["filing_date"] for f in filings_db.indexed_filings()] == ["2026-11-01", "2025-11-01"]


def test_ensure_indexed_caches_each_fiscal_year_separately(filings_db):
    older = dict(FILING, accession="0000123-22-000001", filing_date="2022-11-01", period="2022-09-30")
    with patch("edgar.find_filing", side_effect=[dict(FILING), older]) as find, \
         patch("edgar.fetch_filing_text", return_value=FILING_TEXT):
        filings_db.ensure_indexed("EXM")
        assert filings_db.ensure_indexed("EXM", "10-K", 2022)["accession"] == older["accession"]
        filings_db.ensure_indexed("EXM", "10-K", 2022)
    assert [c.args for c in find.call_args_list] == [("EXM", "10-K", None), ("EXM", "10-K", 2022)]


# ---------------------------------------------------------------------------
# search_filings tool
# ---------------------------------------------------------------------------

def test_search_filings_tool_returns_filing_and_passages(filings_db):
    with patch("edgar.find_filing", return_value=dict(FILING)), \
         patch("edgar.fetch_filing_text", return_value=FILING_TEXT):
        result = search_filings({"symbol": "EXM", "query": "competition prices gross margin", "n_results": 2})
    assert result["filing"]["form"] == "10-K"
    assert result["filing"]["filing_date"] == "2025-11-01"
    assert len(result["passages"]) == 2
    assert "Competition" in result["passages"][0]["text"]


def test_search_filings_tool_reports_edgar_errors(filings_db):
    with patch("edgar.find_filing", side_effect=edgar.EdgarError("No recent 10-K found for XYZ.")):
        result = search_filings({"symbol": "XYZ", "query": "risks"})
    assert result == {"error": "No recent 10-K found for XYZ."}


def test_search_filings_tool_defaults_bad_form_to_10k(filings_db):
    with patch("filings_store.ensure_indexed", side_effect=edgar.EdgarError("stop")) as ensure:
        search_filings({"symbol": "EXM", "query": "risks", "form": "8-K"})
    ensure.assert_called_once_with("EXM", "10-K", None)


def test_search_filings_tool_passes_fiscal_year(filings_db):
    with patch("filings_store.ensure_indexed", side_effect=edgar.EdgarError("stop")) as ensure:
        search_filings({"symbol": "EXM", "query": "risks", "fiscal_year": "2021"})
    ensure.assert_called_once_with("EXM", "10-K", 2021)


@pytest.mark.parametrize("year", ["last year", 1980])
def test_search_filings_tool_rejects_bad_fiscal_year(filings_db, year):
    with patch("filings_store.ensure_indexed") as ensure:
        result = search_filings({"symbol": "EXM", "query": "risks", "fiscal_year": year})
    assert "fiscal_year" in result["error"]
    ensure.assert_not_called()
