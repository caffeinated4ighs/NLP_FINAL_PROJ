from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from Code.core.schema import IndexedFile


class SessionStatus(str, Enum):
    CREATED = "created"
    UPLOADED = "uploaded"
    INDEXING = "indexing"
    READY = "ready"
    ERROR = "error"


@dataclass
class SessionState:
    session_id: str
    status: SessionStatus = SessionStatus.CREATED
    uploaded_files: list[Path] = field(default_factory=list)
    indexed_files: list[IndexedFile] = field(default_factory=list)
    pipeline: Any | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "uploaded_files": [str(path) for path in self.uploaded_files],
            "indexed_files": [item.to_dict() for item in self.indexed_files],
            "error": self.error,
            "ready": self.status == SessionStatus.READY,
        }


class SessionManager:
    """
    In-memory session registry for the MVP.

    This is enough for one EC2 backend process.
    Later, this can be replaced with Redis/SQLite without changing API routes much.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def create_session(self) -> SessionState:
        session_id = str(uuid4())
        state = SessionState(session_id=session_id)
        self._sessions[session_id] = state
        return state

    def get(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")
        return self._sessions[session_id]

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def set_uploaded_files(self, session_id: str, files: list[Path]) -> SessionState:
        state = self.get(session_id)
        state.uploaded_files = files
        state.status = SessionStatus.UPLOADED
        state.error = None
        return state

    def set_indexing(self, session_id: str) -> SessionState:
        state = self.get(session_id)
        state.status = SessionStatus.INDEXING
        state.error = None
        return state

    def set_ready(
        self,
        session_id: str,
        pipeline: Any,
        indexed_files: list[IndexedFile],
    ) -> SessionState:
        state = self.get(session_id)
        state.pipeline = pipeline
        state.indexed_files = indexed_files
        state.status = SessionStatus.READY
        state.error = None
        return state

    def set_error(self, session_id: str, error: str) -> SessionState:
        state = self.get(session_id)
        state.status = SessionStatus.ERROR
        state.error = error
        return state

    def require_ready(self, session_id: str) -> SessionState:
        state = self.get(session_id)

        if state.status != SessionStatus.READY or state.pipeline is None:
            raise RuntimeError(
                f"Session {session_id} is not ready. Current status: {state.status.value}"
            )

        return state


session_manager = SessionManager()