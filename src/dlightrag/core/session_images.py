# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Session-scoped query image memory."""

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from time import monotonic


@dataclass
class _SessionImages:
    images: OrderedDict[str, str] = field(default_factory=OrderedDict)
    last_seen: float = field(default_factory=monotonic)
    counter: int = 0


class SessionImageStore:
    """In-memory TTL/LRU store for user-uploaded query images."""

    def __init__(
        self,
        *,
        max_images_per_session: int = 50,
        max_sessions: int = 100,
        ttl_seconds: int = 3600,
    ) -> None:
        self._max_images = max(1, int(max_images_per_session))
        self._max_sessions = max(1, int(max_sessions))
        self._ttl = max(1, int(ttl_seconds))
        self._sessions: OrderedDict[str, _SessionImages] = OrderedDict()
        self._lock = threading.Lock()

    def store(self, session_id: str | None, images: list[str] | None) -> list[str]:
        """Store raw base64/data URI images and return assigned ids."""
        if not session_id or not images:
            return []
        with self._lock:
            self._evict_expired()
            session = self._sessions.get(session_id) or _SessionImages()
            session.last_seen = monotonic()
            ids: list[str] = []
            for image in images:
                image_id = f"img_{session.counter}"
                session.counter += 1
                session.images[image_id] = image
                session.images.move_to_end(image_id)
                ids.append(image_id)
                while len(session.images) > self._max_images:
                    session.images.popitem(last=False)
            self._sessions[session_id] = session
            self._sessions.move_to_end(session_id)
            self._evict_sessions()
            return ids

    def get(self, session_id: str | None, image_ids: list[str] | None) -> list[str]:
        """Return stored images for ids, skipping expired or missing entries."""
        if not session_id or not image_ids:
            return []
        with self._lock:
            self._evict_expired()
            session = self._sessions.get(session_id)
            if session is None:
                return []
            session.last_seen = monotonic()
            self._sessions.move_to_end(session_id)
            found: list[str] = []
            for image_id in image_ids:
                if image_id not in session.images:
                    continue
                session.images.move_to_end(image_id)
                found.append(session.images[image_id])
            return found

    def clear(self, session_id: str) -> None:
        """Remove one session."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def _evict_expired(self) -> None:
        now = monotonic()
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.last_seen > self._ttl
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)

    def _evict_sessions(self) -> None:
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)


__all__ = ["SessionImageStore"]
