"""Load existing session JSON files into long-term memory.

Safe to re-run: each turn is stored under session + turn index, so repeats overwrite instead of duplicating.

Usage:
    cd longterm-memory-ReAct-agent-tools
    source ../venv/bin/activate
    python backfill.py                                                    # this folder's sessions/
    python backfill.py ../memory-convhistory-ReAct-agent-tools/sessions   # step 3's sessions
"""

import json
import os
import sys

from memory import HISTORY_DIR
from memory_store import extract_turns, remember_turn, memory_count


def backfill(sessions_dir: str) -> int:
    """Store every completed turn from every session file in sessions_dir. Returns turns stored."""
    if not os.path.isdir(sessions_dir):
        print(f"No sessions folder at {sessions_dir}")
        return 0

    stored = 0
    for fname in sorted(os.listdir(sessions_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(sessions_dir, fname)
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  skip {fname}: {exc}")
            continue

        session_name = data.get("session", fname[:-5])
        turns = extract_turns(data.get("history", []))
        for i, turn in enumerate(turns):
            remember_turn(session_name, i, turn["question"], turn["answer"], turn["tickers"], data.get("updated_at"))
        print(f"  {session_name}: {len(turns)} turn(s)")
        stored += len(turns)
    return stored


if __name__ == "__main__":
    sessions_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else HISTORY_DIR
    print(f"Backfilling from {sessions_dir}")
    stored = backfill(sessions_dir)
    print(f"Stored {stored} turn(s). Long-term memory now holds {memory_count()} turn(s).")
