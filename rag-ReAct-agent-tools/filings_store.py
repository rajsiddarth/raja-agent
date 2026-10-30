"""RAG over SEC filings — split a 10-K/10-Q into section-tagged chunks, embed them, and search.

A filing is indexed the first time it is searched and kept in filings_db/ afterwards,
so later questions about the same filing (from any session) are fast.
"""

import math
import os
import re
import time
from collections import Counter

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

# Search: "vector" (embeddings only) or "hybrid" (embeddings + BM25 keyword search, merged with RRF).
# The defaults below are the variant that won on the retrieval eval's dev questions (see README):
# BM25 re-orders the vector search's own results, so it improves ranking without dropping any passage.
SEARCH_MODE = "hybrid"
# True: BM25 only re-orders the passages vector search returned. False: each method contributes its top
# CANDIDATES and the merged list can include passages vector search missed (ranks better, but dropped
# correct passages from the top 5 on the eval).
KEYWORD_RERANK_ONLY = True
CANDIDATES = 20     # passages each method contributes when KEYWORD_RERANK_ONLY is False
RRF_K = 60          # reciprocal rank fusion constant: score = sum of weight / (RRF_K + rank) over both lists
KEYWORD_WEIGHT = 1.0  # weight of the BM25 list relative to the vector list (1.0) in the fusion
DROP_QUESTION_WORDS = True  # leave question-framing words ("describe", "fiscal", years) out of the BM25 query
BM25_K1 = 1.5       # how quickly repeated query words stop adding score
BM25_B = 0.75       # how much long chunks are penalised

# accession -> BM25 index over that filing's chunks, built on first keyword search
_bm25_cache: dict[str, "BM25"] = {}

_collection = None
# (symbol, form, fiscal_year) -> (looked_up_at, filing)
_lookup_cache: dict[tuple[str, str, int | None], tuple[float, dict]] = {}

_NAME_SUFFIXES = r"inc|corp|corporation|co|company|ltd|limited|plc|holdings|group|n\.?v|s\.?a|lp|llc"
_NAME_SUFFIX_RE = re.compile(rf"[,\s]+({_NAME_SUFFIXES})\.?$", re.I)
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
    _bm25_cache.clear()


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
    _bm25_cache.pop(filing["accession"], None)  # rebuilt from the new chunks on the next keyword search
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


def _short_name(company: str) -> str:
    """'NVIDIA CORP' -> 'NVIDIA', 'Meta Platforms, Inc.' -> 'Meta Platforms'."""
    name = company.strip()
    while True:
        shorter = _NAME_SUFFIX_RE.sub("", name).strip()
        if shorter == name:
            return name
        name = shorter


def strip_company(query: str, filing: dict) -> str:
    """Remove the company's name and ticker from a query.

    The search is already limited to this company's filing, so the name adds nothing, and the embedding
    then favours boilerplate that repeats it ("All references to NVIDIA ...") over the passage asked about.
    A name only counts when capitalised ("Target's risks", not "revenue target"); a ticker only in upper
    case and at least two letters long (so "A" and "on" are left alone).
    """
    cleaned = query
    name = _short_name(filing.get("company", ""))
    if len(name) >= 2:
        words = r"[\s\-]+".join(re.escape(w) for w in name.split())
        cleaned = re.sub(
            # the name, then an optional legal suffix ("Corp", "Inc.") and possessive
            rf"\b{words}\b(?:,?\s+(?:{_NAME_SUFFIXES})\b\.?)?(?:['’]s\b)?",
            lambda m: "" if m.group(0)[0].isupper() else m.group(0),
            cleaned,
            flags=re.I,
        )
    symbol = filing.get("symbol", "")
    if len(symbol) >= 2:
        cleaned = re.sub(rf"\b{re.escape(symbol)}(?:['’]s)?\b", "", cleaned)
    cleaned = re.sub(r"\s+([?.,;:])", r"\1", re.sub(r"\s+", " ", cleaned)).strip(" ,")
    # a query that was only the company name keeps it rather than becoming empty
    return cleaned if len(re.findall(r"\w", cleaned)) >= 3 else query


def search(filing: dict, query: str, n_results: int = 5) -> list[dict]:
    """Search one filing for the query, with the company's own name removed from it first."""
    return search_filing(strip_company(query, filing), filing["accession"], n_results)


def search_filing(query: str, accession: str, n_results: int = 5) -> list[dict]:
    """Return the chunks of one filing most relevant to the query: [{section, text, relevance}].

    In "vector" mode relevance is the cosine similarity. In "hybrid" mode the vector and BM25 rankings are
    merged with reciprocal rank fusion, and relevance is the fused score scaled so the top passage is 1.0.
    """
    if SEARCH_MODE == "vector":
        return [_passage(hit, round(hit["similarity"], 3)) for hit in _vector_search(query, accession, n_results)]

    bm25 = _bm25_index(accession)
    candidates = min(n_results if KEYWORD_RERANK_ONLY else CANDIDATES, len(bm25.ids))
    if candidates == 0:
        return []  # nothing indexed for this filing
    vector_ids = [hit["id"] for hit in _vector_search(query, accession, candidates)]
    keyword_ids = bm25.top(query, candidates, among=vector_ids if KEYWORD_RERANK_ONLY else None)
    fused = reciprocal_rank_fusion([vector_ids, keyword_ids], weights=[1.0, KEYWORD_WEIGHT])[:n_results]
    if not fused:
        return []
    top_score = fused[0][1]
    return [_passage(bm25.docs[chunk_id], round(score / top_score, 3)) for chunk_id, score in fused]


def _passage(hit: dict, relevance: float) -> dict:
    doc = hit["document"]
    return {"section": hit["section"], "text": doc.split("\n", 1)[1] if "\n" in doc else doc, "relevance": relevance}


def _vector_search(query: str, accession: str, n_results: int) -> list[dict]:
    """Embedding search: [{id, document, section, similarity}], most similar first."""
    result = _get_collection().query(query_texts=[query], n_results=n_results, where={"accession": accession})
    return [
        {"id": chunk_id, "document": doc, "section": meta["section"], "similarity": 1 - dist}
        for chunk_id, doc, meta, dist in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = RRF_K, weights: list[float] | None = None
) -> list[tuple[str, float]]:
    """Merge ranked lists of ids: each id scores sum(weight / (k + rank)) over the lists it appears in.

    Only ranks matter, not the raw scores, so cosine similarities and BM25 scores (which are on different
    scales) can be combined. An id near the top of both lists beats one at the top of only one.
    """
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


# ---------------------------------------------------------------------------
# BM25 keyword search
# ---------------------------------------------------------------------------

# very common words that carry no meaning for matching; BM25's IDF already discounts them, this just saves work
_STOPWORDS = frozenset(
    "a an and are as at be by did do does for from has have how in is it its of on or that the this to "
    "was were what when where which who why will with".split()
)


# Words that frame a question about a filing rather than name its topic. The search is already limited to
# one filing, so they match boilerplate everywhere ("fiscal 2021" appears throughout a fiscal-2021 10-K).
_QUESTION_WORDS = frozenset(
    "describe described explain say says said cite cites mention mentions discuss disclose report reported "
    "latest filing filings fiscal year quarter 10 k q annual quarterly".split()
)


def query_terms(query: str) -> list[str]:
    """The words BM25 searches for: tokens minus question-framing words and years, if DROP_QUESTION_WORDS."""
    terms = tokenize(query)
    if DROP_QUESTION_WORDS:
        terms = [t for t in terms if t not in _QUESTION_WORDS and not re.fullmatch(r"(19|20)\d\d", t)]
    return terms


def tokenize(text: str) -> list[str]:
    """Lowercase words and numbers, e.g. "Taiwan Semiconductor's 10-K" -> ['taiwan', 'semiconductor', 's', '10', 'k']."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


class BM25:
    """Okapi BM25 over one filing's chunks.

    A chunk scores higher when it contains the query's words often (term frequency, with diminishing returns
    set by k1), when those words are rare across the filing (inverse document frequency, so "Epic" counts far
    more than "company"), and when the chunk is not unusually long (length normalisation, set by b).
    """

    def __init__(self, ids: list[str], docs: dict[str, dict], k1: float = BM25_K1, b: float = BM25_B) -> None:
        self.ids, self.docs, self.k1, self.b = ids, docs, k1, b
        self.term_freqs = [Counter(tokenize(docs[i]["document"])) for i in ids]
        self.lengths = [sum(tf.values()) for tf in self.term_freqs]
        self.avg_length = sum(self.lengths) / len(ids) if ids else 0.0
        doc_freq = Counter(term for tf in self.term_freqs for term in tf)
        n = len(ids)
        self.idf = {term: math.log((n - df + 0.5) / (df + 0.5) + 1) for term, df in doc_freq.items()}

    def scores(self, query: str) -> list[float]:
        terms = [t for t in query_terms(query) if t in self.idf]
        result = []
        for tf, length in zip(self.term_freqs, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_length)
            result.append(sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if tf[t]))
        return result

    def top(self, query: str, n: int, among: list[str] | None = None) -> list[str]:
        """Ids of the n highest-scoring chunks that contain at least one query word (optionally only `among`)."""
        allowed = set(among) if among is not None else None
        ranked = sorted(
            ((score, chunk_id) for score, chunk_id in zip(self.scores(query), self.ids)
             if allowed is None or chunk_id in allowed),
            key=lambda pair: pair[0], reverse=True,
        )
        return [chunk_id for score, chunk_id in ranked[:n] if score > 0]


def _bm25_index(accession: str) -> BM25:
    """The BM25 index for one filing, built from its stored chunks on first use."""
    if accession not in _bm25_cache:
        result = _get_collection().get(where={"accession": accession}, include=["documents", "metadatas"])
        docs = {
            chunk_id: {"id": chunk_id, "document": doc, "section": meta["section"]}
            for chunk_id, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
        }
        _bm25_cache[accession] = BM25(sorted(docs), docs)
    return _bm25_cache[accession]


def filing_chunks(accession: str) -> list[dict]:
    """Every stored chunk of one filing: [{section, text}] (used by the retrieval eval to check gold labels)."""
    result = _get_collection().get(where={"accession": accession}, include=["documents", "metadatas"])
    return [
        {"section": meta["section"], "text": doc.split("\n", 1)[1] if "\n" in doc else doc}
        for doc, meta in zip(result["documents"], result["metadatas"])
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
