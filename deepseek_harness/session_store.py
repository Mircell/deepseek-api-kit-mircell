"""Session persistence for the DeepSeek provider.

Mirrors the legacy harness behavior: a single DeepSeek web-chat session is
reused across requests and persisted to ``.session_data.json`` (plus the
``.session_id`` file kept for backward compatibility with
``send_with_session.py``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional


class SessionStore:
    """Tracks DeepSeek chat sessions and their last message id."""

    def __init__(self, data_file: Path, legacy_file: Path) -> None:
        self.data_file = data_file
        self.legacy_file = legacy_file
        self.sessions: Dict[str, dict] = {}

    # -- persistence ----------------------------------------------------
    def load(self) -> Optional[str]:
        """Load a persisted session (if any) and return its id."""
        if not self.data_file.exists():
            return None
        try:
            data = json.loads(self.data_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        session_id = data.get("session_id")
        if not session_id:
            return None

        self.sessions[session_id] = {
            "created": time.time(),
            "last_message_id": data.get("last_message_id"),
        }
        self._write_legacy(session_id)
        return session_id

    def save(self, session_id: str, last_message_id: Optional[str] = None) -> None:
        """Persist the active session id and its last message id."""
        self.data_file.write_text(
            json.dumps({"session_id": session_id, "last_message_id": last_message_id}),
            encoding="utf-8",
        )
        self._write_legacy(session_id)

    def _write_legacy(self, session_id: str) -> None:
        try:
            self.legacy_file.write_text(session_id, encoding="utf-8")
        except OSError:
            pass

    # -- lifecycle ------------------------------------------------------
    def resolve(self, api, requested_id: Optional[str]) -> str:
        """Return an existing session id or create a fresh one."""
        if requested_id and requested_id in self.sessions:
            return requested_id
        if self.sessions:
            # Reuse the first known session (legacy single-session behavior).
            return next(iter(self.sessions))
        new_id = api.create_chat_session()
        self.sessions[new_id] = {"created": time.time(), "last_message_id": None}
        self.save(new_id, None)
        return new_id

    def parent_message_id(self, session_id: str) -> Optional[str]:
        return self.sessions.get(session_id, {}).get("last_message_id")

    def update(self, session_id: str, last_message_id: Optional[str]) -> None:
        if session_id not in self.sessions:
            self.sessions[session_id] = {"created": time.time(), "last_message_id": None}
        self.sessions[session_id]["last_message_id"] = last_message_id
        self.save(session_id, last_message_id)

    def reset(self, api) -> str:
        """Drop all sessions and create a brand new one."""
        self.sessions.clear()
        new_id = api.create_chat_session()
        self.sessions[new_id] = {"created": time.time(), "last_message_id": None}
        self.save(new_id, None)
        return new_id