"""
Conversation history per session_id. Kept in memory for fast reads, and
mirrored to a JSON file on the same persistent volume Chroma already uses
(SESSION_STORE_PATH, default /data/sessions.json) so history survives an
app container restart. Process-local (not shared across multiple app
replicas) — fine for a single-container local deployment.
History length is capped so prompts stay bounded.
"""
import json
import logging
import os
import threading

from app import config

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def _load() -> dict[str, list[dict]]:
    path = config.SESSION_STORE_PATH
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not load session store at %s: %s", path, e)
        return {}


def _save(data: dict[str, list[dict]]) -> None:
    path = config.SESSION_STORE_PATH
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except OSError as e:
        # Don't let a persistence failure break the in-memory chat flow —
        # history still works for the life of this process either way.
        logger.warning("Could not persist session store to %s: %s", path, e)


_sessions: dict[str, list[dict]] = _load()


def get_history(session_id: str) -> list[dict]:
    with _lock:
        return list(_sessions.get(session_id, []))


def append_turn(session_id: str, user_message: str, assistant_message: str) -> None:
    with _lock:
        history = _sessions.setdefault(session_id, [])
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": assistant_message})
        max_messages = config.SESSION_MAX_TURNS * 2
        if len(history) > max_messages:
            del history[: len(history) - max_messages]
        _save(_sessions)


def clear_session(session_id: str) -> None:
    with _lock:
        _sessions.pop(session_id, None)
        _save(_sessions)