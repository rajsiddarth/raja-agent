"""Conversation history persistence — saves/loads per-session history to JSON."""

import json
import os
from datetime import datetime

HISTORY_DIR = os.path.join(os.path.dirname(__file__), "sessions")


def _session_path(session_name: str) -> str:
    os.makedirs(HISTORY_DIR, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_name)
    return os.path.join(HISTORY_DIR, f"{safe_name}.json")


def load_history(session_name: str) -> list:
    """Load conversation history for a session. Returns [] if none exists."""
    path = _session_path(session_name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("history", [])


def save_history(session_name: str, history: list) -> None:
    """Persist conversation history for a session.

    Writes to a temp file and swaps it in, so a failed write never leaves a truncated session file.
    """
    path = _session_path(session_name)
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump({"session": session_name, "updated_at": datetime.now().isoformat(), "history": history}, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def list_sessions() -> list[dict]:
    """Return all saved sessions sorted by most recently updated."""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    sessions = []
    for fname in os.listdir(HISTORY_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(HISTORY_DIR, fname)
        with open(path) as f:
            data = json.load(f)
        turn_count = sum(1 for m in data.get("history", []) if m.get("role") == "user")
        sessions.append({
            "name": data.get("session", fname[:-5]),
            "updated_at": data.get("updated_at", ""),
            "turns": turn_count,
        })
    return sorted(sessions, key=lambda s: s["updated_at"], reverse=True)


def delete_session(session_name: str) -> None:
    """Delete a saved session."""
    path = _session_path(session_name)
    if os.path.exists(path):
        os.remove(path)
