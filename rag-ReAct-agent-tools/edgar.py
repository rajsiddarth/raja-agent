"""SEC EDGAR client — find a company's latest 10-K/10-Q and return it as plain text.

SEC requires every request to identify the caller: set SEC_USER_AGENT in .env, e.g.
    SEC_USER_AGENT=Jane Doe jane@example.com
See https://www.sec.gov/os/accessing-edgar-data
"""

import json
import os
import re
import time
import warnings

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
OLDER_SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

CACHE_DIR = os.path.join(os.path.dirname(__file__), "edgar_cache")
SUPPORTED_FORMS = ("10-K", "10-Q")
# Before 2003 some annual reports were filed as 10-K405; amendments (10-K/A) are left out on purpose
FORM_VARIANTS = {"10-K": ("10-K", "10-K405"), "10-Q": ("10-Q",)}

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


def _filing_pages(cik: int):
    """Yield the company's filing lists, newest first.

    The submissions file holds the most recent filings (at least a year, up to 1,000). Older filings
    are in extra pages that are only fetched if the caller keeps iterating.
    """
    submissions = _get(SUBMISSIONS_URL.format(cik=cik)).json()
    yield submissions["filings"]["recent"]
    for page in submissions["filings"].get("files", []):
        yield _get(OLDER_SUBMISSIONS_URL.format(name=page["name"])).json()


def find_filing(symbol: str, form: str = "10-K", fiscal_year: int | None = None) -> dict:
    """Return metadata for a filing of the given form: the most recent one, or the one for a fiscal year.

    The fiscal year is the year the reporting period ended (e.g. Apple's year ending 2024-09-28 is 2024,
    NVIDIA's year ending 2025-01-26 is 2025). For a 10-Q it selects the latest quarter ending in that year.
    """
    if form not in SUPPORTED_FORMS:
        raise EdgarError(f"Unsupported form {form}. Use one of {', '.join(SUPPORTED_FORMS)}.")
    cik = lookup_cik(symbol)
    years_seen: set[str] = set()

    for page in _filing_pages(cik):
        for i, filing_form in enumerate(page["form"]):
            if filing_form not in FORM_VARIANTS[form]:
                continue
            period = page["reportDate"][i]
            years_seen.add(period[:4])  # recorded before stopping, so the error can name the latest year
            if fiscal_year is not None and page["filingDate"][i][:4] < str(fiscal_year):
                break  # filed before the year ended, so this and every older filing is too old
            if fiscal_year is None or period[:4] == str(fiscal_year):
                accession = page["accessionNumber"][i]
                return {
                    "symbol": symbol.upper(),
                    "cik": cik,
                    "form": form,
                    "accession": accession,
                    "filing_date": page["filingDate"][i],
                    "period": period,
                    "url": ARCHIVE_URL.format(
                        cik=cik, accession=accession.replace("-", ""), document=page["primaryDocument"][i]
                    ),
                }
        else:
            continue
        break  # stopped early inside the page: older pages cannot match either

    if fiscal_year is None:
        raise EdgarError(f"No {form} found for {symbol.upper()}.")
    available = ", ".join(sorted(years_seen, reverse=True)) or "none"
    raise EdgarError(
        f"No {form} for fiscal year {fiscal_year} found for {symbol.upper()}. "
        f"Fiscal years seen (by period end date): {available}."
    )


def html_to_text(html: str) -> str:
    """Convert a filing's HTML to plain text, one block per line, without hidden XBRL data."""
    with warnings.catch_warnings():
        # inline-XBRL filings are XHTML starting with <?xml; the HTML parser handles them fine
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
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
