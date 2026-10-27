"""SEC EDGAR client — find a company's latest 10-K/10-Q and return it as plain text.

SEC requires every request to identify the caller: set SEC_USER_AGENT in .env, e.g.
    SEC_USER_AGENT=Jane Doe jane@example.com
See https://www.sec.gov/os/accessing-edgar-data
"""

import json
import os
import re
import time

import requests
from bs4 import BeautifulSoup

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

CACHE_DIR = os.path.join(os.path.dirname(__file__), "edgar_cache")
SUPPORTED_FORMS = ("10-K", "10-Q")

# SEC allows at most 10 requests per second
_MIN_INTERVAL_S = 0.15
_last_request = 0.0


class EdgarError(Exception):
    """A filing could not be found or fetched; the message is safe to show the user."""


def _user_agent() -> str:
    agent = os.getenv("SEC_USER_AGENT", "").strip()
    if not agent:
        raise EdgarError(
            "SEC_USER_AGENT is not set. Add a line like 'SEC_USER_AGENT=Your Name you@example.com' to .env "
            "(SEC requires a name and email on every request)."
        )
    return agent


def _get(url: str) -> requests.Response:
    """GET with SEC's required User-Agent and a small delay to respect the rate limit."""
    global _last_request
    wait = _MIN_INTERVAL_S - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.time()
    response = requests.get(url, headers={"User-Agent": _user_agent()}, timeout=30)
    if response.status_code != 200:
        raise EdgarError(f"SEC EDGAR returned HTTP {response.status_code} for {url}")
    return response


def lookup_cik(symbol: str) -> int:
    """Return the SEC CIK number for a ticker. The ticker list is cached on disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "company_tickers.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    else:
        data = _get(TICKERS_URL).json()
        with open(path, "w") as f:
            json.dump(data, f)

    symbol = symbol.upper()
    for entry in data.values():
        if entry["ticker"].upper() == symbol:
            return int(entry["cik_str"])
    raise EdgarError(f"No SEC registrant found for ticker {symbol}. ETFs and non-US companies may not file 10-Ks.")


def latest_filing(symbol: str, form: str = "10-K") -> dict:
    """Return metadata for the company's most recent filing of the given form."""
    if form not in SUPPORTED_FORMS:
        raise EdgarError(f"Unsupported form {form}. Use one of {', '.join(SUPPORTED_FORMS)}.")
    cik = lookup_cik(symbol)
    recent = _get(SUBMISSIONS_URL.format(cik=cik)).json()["filings"]["recent"]

    for i, filing_form in enumerate(recent["form"]):
        if filing_form == form:
            accession = recent["accessionNumber"][i]
            return {
                "symbol": symbol.upper(),
                "cik": cik,
                "form": form,
                "accession": accession,
                "filing_date": recent["filingDate"][i],
                "period": recent["reportDate"][i],
                "url": ARCHIVE_URL.format(
                    cik=cik, accession=accession.replace("-", ""), document=recent["primaryDocument"][i]
                ),
            }
    raise EdgarError(f"No recent {form} found for {symbol.upper()}.")


def html_to_text(html: str) -> str:
    """Convert a filing's HTML to plain text, one block per line, without hidden XBRL data."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    for tag in soup.find_all(re.compile(r"^ix:header$", re.I)):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        tag.decompose()

    text = soup.get_text("\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def fetch_filing_text(filing: dict) -> str:
    """Download a filing's primary document and return its plain text."""
    return html_to_text(_get(filing["url"]).text)
