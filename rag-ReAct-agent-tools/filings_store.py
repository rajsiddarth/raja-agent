"""RAG over SEC filings — split a 10-K/10-Q into section-tagged chunks, embed them, and search.

A filing is indexed the first time it is searched and kept in filings_db/ afterwards,
so later questions about the same filing (from any session) are fast.
"""

import os
import re
import time

import edgar

FILINGS_DIR = os.path.join(os.path.dirname(__file__), "filings_db")
COLLECTION_NAME = "filings"

# all-MiniLM-L6-v2 reads at most ~256 tokens (~1,000 characters); longer text would be cut off
CHUNK_CHARS = 1000
OVERLAP_CHARS = 200
MIN_CHUNK_CHARS = 80

# Override in tests to avoid downloading the default embedding model
embedding_function = None

# How long an EDGAR "which filing is current?" answer is trusted before asking EDGAR again
LOOKUP_TTL_S = 24 * 60 * 60

_collection = None
# (symbol, form, fiscal_year) -> (looked_up_at, filing)
_lookup_cache: dict[tuple[str, str, int | None], tuple[float, dict]] = {}

_PART_RE = re.compile(r"^PART\s+(I{1,3}|IV)\b", re.I)
_ITEM_RE = re.compile(r"^ITEM\s+(\d{1,2}[A-C]?)\s*[.:\-—]?\s*(.*)$", re.I)


def _get_collection():
    """Open (or create) the persistent Chroma collection on first use."""
    global _collection
    if _collection is None:
        import chromadb
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        client = chromadb.PersistentClient(path=FILINGS_DIR)
        _collection = client.get_or_create_collection(
            COLLECTION_NAME,
            embedding_function=embedding_function or DefaultEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def reset() -> None:
    """Drop cached handles (used when FILINGS_DIR changes, e.g. in tests)."""
    global _collection
    _collection = None
    _lookup_cache.clear()


# ---------------------------------------------------------------------------
# Sections and chunks
# ---------------------------------------------------------------------------

def split_sections(text: str) -> list[tuple[str, list[str]]]:
    """Group a filing's lines under their 'Part / Item' headings, e.g. 'Part I — Item 1A. Risk Factors'."""
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = [("Cover", [])]
    part = ""
    for i, line in enumerate(lines):
        part_match = _PART_RE.match(line)
        if part_match and len(line) < 80:
            part = f"Part {part_match.group(1).upper()}"
            continue
        item_match = _ITEM_RE.match(line)
        if item_match and len(line) < 150:
            title = item_match.group(2).strip()
            if not title and i + 1 < len(lines) and len(lines[i + 1]) < 100:
                title = lines[i + 1]  # heading split across two cells: "Item 1A." / "Risk Factors"
            label = f"Item {item_match.group(1).upper()}. {title}".strip()
            sections.append((f"{part} — {label}" if part else label, []))
            continue
        sections[-1][1].append(line)
    return [(name, body) for name, body in sections if body]


def _pieces(line: str) -> list[str]:
    """Split a line longer than CHUNK_CHARS at sentence ends (or hard-cut as a last resort)."""
    if len(line) <= CHUNK_CHARS:
        return [line]
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.;])\s+", line):
        while len(sentence) > CHUNK_CHARS:
            pieces.append(sentence[:CHUNK_CHARS])
            sentence = sentence[CHUNK_CHARS:]
        if current and len(current) + len(sentence) + 1 > CHUNK_CHARS:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str) -> list[dict]:
    """Split filing text into ~CHUNK_CHARS chunks that never cross a section, with OVERLAP_CHARS of overlap."""
    chunks = []
    for section, body in split_sections(text):
        lines = [piece for line in body for piece in _pieces(line)]
        current: list[str] = []
        for line in lines:
            if current and sum(len(l) + 1 for l in current) + len(line) > CHUNK_CHARS:
                chunks.append({"section": section, "text": "\n".join(current)})
                # carry the last lines forward so an idea split across chunks stays findable
                overlap: list[str] = []
                for prev in reversed(current):
                    if sum(len(l) + 1 for l in overlap) + len(prev) > OVERLAP_CHARS:
                        break
                    overlap.insert(0, prev)
                fits = sum(len(l) + 1 for l in overlap) + len(line) <= CHUNK_CHARS
                current = overlap if fits else []
            current.append(line)
        if current:
            chunks.append({"section": section, "text": "\n".join(current)})
    return [c for c in chunks if len(c["text"]) >= MIN_CHUNK_CHARS]


# ---------------------------------------------------------------------------
# Index and search
# ---------------------------------------------------------------------------

def is_indexed(accession: str) -> bool:
    return bool(_get_collection().get(where={"accession": accession}, limit=1, include=[])["ids"])


def index_filing(filing: dict, text: str) -> int:
    """Chunk and store a filing. Returns the number of chunks. Re-indexing overwrites by ID."""
    chunks = chunk_text(text)
    collection = _get_collection()
    batch = 200
    for start in range(0, len(chunks), batch):
        part = chunks[start:start + batch]
        collection.upsert(
            ids=[f"{filing['accession']}:{start + i}" for i in range(len(part))],
            # the section heading is embedded with the text, so "risk factors" queries match risk chunks
            documents=[f"{c['section']}\n{c['text']}" for c in part],
            metadatas=[{
                "symbol": filing["symbol"],
                "form": filing["form"],
                "accession": filing["accession"],
                "filing_date": filing["filing_date"],
                "period": filing["period"],
                "section": c["section"],
                "url": filing["url"],
            } for c in part],
        )
    return len(chunks)


def ensure_indexed(symbol: str, form: str = "10-K", fiscal_year: int | None = None) -> dict:
    """Make sure the requested filing (latest, or for a fiscal year) is indexed; return its metadata.

    EDGAR lookups are cached for LOOKUP_TTL_S, so a long-running app still notices a new filing.
    """
    key = (symbol.upper(), form, fiscal_year)
    cached = _lookup_cache.get(key)
    if cached is not None and time.time() - cached[0] < LOOKUP_TTL_S:
        filing = cached[1]
    else:
        filing = edgar.find_filing(symbol, form, fiscal_year)
        _lookup_cache[key] = (time.time(), filing)
    if not is_indexed(filing["accession"]):
        filing["chunks"] = index_filing(filing, edgar.fetch_filing_text(filing))
    return filing


def search_filing(query: str, accession: str, n_results: int = 5) -> list[dict]:
    """Return the chunks of one filing most similar to the query."""
    result = _get_collection().query(query_texts=[query], n_results=n_results, where={"accession": accession})
    return [
        {
            "section": meta["section"],
            "text": doc.split("\n", 1)[1] if "\n" in doc else doc,
            "relevance": round(1 - dist, 3),
        }
        for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
    ]


def indexed_filings() -> list[dict]:
    """List the filings stored in the index: [{symbol, form, filing_date, period, chunks}], newest first."""
    metas = _get_collection().get(include=["metadatas"])["metadatas"]
    filings: dict[str, dict] = {}
    for meta in metas:
        entry = filings.setdefault(meta["accession"], {
            "symbol": meta["symbol"], "form": meta["form"], "filing_date": meta["filing_date"],
            "period": meta["period"], "chunks": 0,
        })
        entry["chunks"] += 1
    return sorted(filings.values(), key=lambda f: f["filing_date"], reverse=True)
