"""Unit tests for memory_store.py, recall_memory and backfill.py — uses an offline embedding."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from memory_store import extract_turns
from tools import recall_memory
from backfill import backfill


def nvda_turn() -> list:
    """One full ReAct turn as stored in history: think, tool call, tool result, final answer."""
    return [
        {"role": "user", "content": "Get recent NVDA news headlines"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "think", "input": {"reasoning": "Need news."}},
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Reasoning noted."}]},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t2", "name": "get_news", "input": {"symbol": "nvda"}},
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "{\"articles\": []}"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "No recent NVDA news is available."}]},
    ]


# ---------------------------------------------------------------------------
# extract_turns
# ---------------------------------------------------------------------------

def test_extract_turns_keeps_question_and_final_answer_only():
    turns = extract_turns(nvda_turn())
    assert turns == [{
        "question": "Get recent NVDA news headlines",
        "answer": "No recent NVDA news is available.",
        "tickers": ["NVDA"],
    }]


def test_extract_turns_splits_multiple_turns():
    history = nvda_turn() + [
        {"role": "user", "content": "And AMD?"},
        {"role": "assistant", "content": "AMD has two headlines."},
    ]
    turns = extract_turns(history)
    assert [t["question"] for t in turns] == ["Get recent NVDA news headlines", "And AMD?"]
    assert turns[1]["answer"] == "AMD has two headlines."


def test_extract_turns_skips_turn_without_answer():
    history = nvda_turn() + [{"role": "user", "content": "Unanswered question"}]
    assert len(extract_turns(history)) == 1


# ---------------------------------------------------------------------------
# remember / search
# ---------------------------------------------------------------------------

def test_search_empty_store_returns_nothing(memory_db):
    assert memory_db.search_memory("anything") == []


def test_remember_last_turn_stores_turn(memory_db):
    memory_id = memory_db.remember_last_turn("teste", nvda_turn())
    assert memory_id == "teste:0"
    results = memory_db.search_memory("NVDA news")
    assert results[0]["session"] == "teste"
    assert results[0]["tickers"] == ["NVDA"]
    assert "Q: Get recent NVDA news headlines" in results[0]["text"]


def test_remember_same_turn_twice_does_not_duplicate(memory_db):
    memory_db.remember_last_turn("teste", nvda_turn())
    memory_db.remember_last_turn("teste", nvda_turn())
    assert memory_db.memory_count() == 1


def test_search_finds_relevant_turn_across_sessions(memory_db):
    memory_db.remember_turn("session-a", 0, "Is AMD valuation too high?", "AMD trades at a high P/E valuation.", ["AMD"])
    memory_db.remember_turn("session-b", 0, "Apple price today", "Apple is at $150.", ["AAPL"])
    results = memory_db.search_memory("AMD valuation", n_results=2)
    assert results[0]["session"] == "session-a"
    assert results[0]["relevance"] > results[1]["relevance"]


# ---------------------------------------------------------------------------
# recall_memory tool
# ---------------------------------------------------------------------------

def test_recall_memory_empty_returns_note(memory_db):
    result = recall_memory({"query": "AMD"})
    assert result["memories"] == []
    assert "No past conversations" in result["note"]


def test_recall_memory_clamps_n_results(memory_db):
    for i in range(12):
        memory_db.remember_turn("s", i, f"question {i}", f"answer {i}")
    assert len(recall_memory({"query": "question", "n_results": 50})["memories"]) == 10
    assert len(recall_memory({"query": "question", "n_results": 0})["memories"]) == 5


# ---------------------------------------------------------------------------
# backfill
# ---------------------------------------------------------------------------

def test_backfill_loads_sessions_and_is_rerunnable(memory_db, tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "teste.json").write_text(json.dumps(
        {"session": "teste", "updated_at": "2026-10-26T20:21:20", "history": nvda_turn()}
    ))
    (sessions / "broken.json").write_text("{not json")

    assert backfill(str(sessions)) == 1
    assert backfill(str(sessions)) == 1
    assert memory_db.memory_count() == 1
    assert memory_db.search_memory("NVDA")[0]["timestamp"] == "2026-10-26T20:21:20"
