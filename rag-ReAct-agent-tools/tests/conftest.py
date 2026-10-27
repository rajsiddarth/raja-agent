"""Shared fixtures — an isolated, offline long-term memory store for each test."""

import hashlib
import math
import os
import re
import sys

import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import filings_store
import memory_store


class WordHashEmbedding(EmbeddingFunction[Documents]):
    """Deterministic bag-of-words embedding: texts sharing words are close. No model download."""

    DIMS = 128

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            vec = [0.0] * self.DIMS
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIMS] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    @staticmethod
    def name() -> str:
        return "word-hash-test"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "WordHashEmbedding":
        return WordHashEmbedding()


@pytest.fixture
def memory_db(tmp_path, monkeypatch):
    """Point memory_store at a fresh temp folder with the offline embedding."""
    monkeypatch.setattr(memory_store, "MEMORY_DIR", str(tmp_path / "memory_db"))
    monkeypatch.setattr(memory_store, "embedding_function", WordHashEmbedding())
    memory_store.reset()
    yield memory_store
    memory_store.reset()


@pytest.fixture
def filings_db(tmp_path, monkeypatch):
    """Point filings_store at a fresh temp folder with the offline embedding."""
    monkeypatch.setattr(filings_store, "FILINGS_DIR", str(tmp_path / "filings_db"))
    monkeypatch.setattr(filings_store, "embedding_function", WordHashEmbedding())
    filings_store.reset()
    yield filings_store
    filings_store.reset()
