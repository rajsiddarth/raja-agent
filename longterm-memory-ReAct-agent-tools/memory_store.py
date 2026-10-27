"""Long-term memory — stores each turn (question + final answer) in a vector DB, searchable across sessions."""

import os
from datetime import datetime

MEMORY_DIR = os.path.join(os.path.dirname(__file__), "memory_db")
COLLECTION_NAME = "conversation_memory"

# Override in tests to avoid downloading the default embedding model
embedding_function = None

_collection = None


def _get_collection():
    """Open (or create) the persistent Chroma collection on first use."""
    global _collection
    if _collection is None:
        import chromadb
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        client = chromadb.PersistentClient(path=MEMORY_DIR)
        _collection = client.get_or_create_collection(
            COLLECTION_NAME,
            embedding_function=embedding_function or DefaultEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def reset() -> None:
    """Drop the cached collection handle (used when MEMORY_DIR changes, e.g. in tests)."""
    global _collection
    _collection = None


def _text_of(content) -> str:
    """Join the text blocks of a message's content (str or list of blocks)."""
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text").strip()


def extract_turns(history: list) -> list[dict]:
    """Split a message history into completed turns: the user's question and Claude's final answer.

    A turn starts at a user message with plain-text content (tool_result messages are skipped).
    think/tool calls and raw tool results are dropped; tickers looked up during the turn are kept.
    """
    turns = []
    current = None
    for msg in history:
        content = msg.get("content")
        if msg.get("role") == "user" and isinstance(content, str):
            if current and current["answer"]:
                turns.append(current)
            current = {"question": content, "answer": "", "tickers": []}
        elif msg.get("role") == "assistant" and current is not None:
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        symbol = (block.get("input") or {}).get("symbol")
                        if symbol and symbol.upper() not in current["tickers"]:
                            current["tickers"].append(symbol.upper())
            text = _text_of(content)
            if text:
                current["answer"] = text
    if current and current["answer"]:
        turns.append(current)
    return turns


def remember_turn(
    session_name: str,
    turn_index: int,
    question: str,
    answer: str,
    tickers: list[str] | None = None,
    timestamp: str | None = None,
) -> str:
    """Store one turn. The ID is session + turn index, so re-storing the same turn overwrites it."""
    memory_id = f"{session_name}:{turn_index}"
    _get_collection().upsert(
        ids=[memory_id],
        documents=[f"Q: {question}\nA: {answer}"],
        metadatas=[{
            "session": session_name,
            "turn": turn_index,
            "timestamp": timestamp or datetime.now().isoformat(),
            "tickers": ",".join(tickers or []),
        }],
    )
    return memory_id


def remember_last_turn(session_name: str, history: list) -> str | None:
    """Store the most recent completed turn of a session. Returns its ID, or None if there is none."""
    turns = extract_turns(history)
    if not turns:
        return None
    last = turns[-1]
    return remember_turn(session_name, len(turns) - 1, last["question"], last["answer"], last["tickers"])


def search_memory(query: str, n_results: int = 5) -> list[dict]:
    """Return the past turns most similar to the query, from any session."""
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []
    result = collection.query(query_texts=[query], n_results=min(n_results, count))
    return [
        {
            "session": meta["session"],
            "timestamp": meta["timestamp"],
            "tickers": [t for t in meta["tickers"].split(",") if t],
            "text": doc,
            "relevance": round(1 - dist, 3),
        }
        for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
    ]


def memory_count() -> int:
    """Number of turns stored in long-term memory."""
    return _get_collection().count()
