# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Canonical typed ids for agent sessions, entries, effects, and projections.

Every durable identity is a small immutable string value type wrapping one
canonical UUID. Framework-issued ids use UUIDv7; deterministic Fast and stage
identities use UUIDv5 derived from the run id and a declared namespace, so the
same identity is reconstructed identically across processes and replays.
"""

import re
from dataclasses import dataclass
from typing import Self
from uuid import NAMESPACE_URL, UUID, uuid5, uuid7


def _require_canonical_uuid(value: str, label: str) -> None:
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a canonical UUID string") from exc
    if str(parsed) != value:
        raise ValueError(f"{label} must be a canonical UUID string")


def new_uuid7() -> str:
    """Return one framework-issued UUIDv7 identity in canonical string form."""
    return str(uuid7())


def deterministic_uuid(*, seed: str, name: str) -> str:
    """Return one deterministic UUIDv5 over a seed and a declared namespace name."""
    namespace = uuid5(NAMESPACE_URL, seed)
    return str(uuid5(namespace, name))


@dataclass(frozen=True, slots=True)
class SessionId:
    """One agent session identity, pinned at acceptance for Research runs."""

    value: str

    def __post_init__(self) -> None:
        _require_canonical_uuid(self.value, "SessionId")

    @classmethod
    def new(cls) -> Self:
        return cls(new_uuid7())

    @classmethod
    def deterministic(cls, *, run_id: str, name: str) -> Self:
        return cls(deterministic_uuid(seed=run_id, name=name))

    def __str__(self) -> str:
        return self.value


_LANE_ID_RE = re.compile(r"^main$|^[0-9a-f-]{36}$")


@dataclass(frozen=True, slots=True)
class LaneId:
    """One stable, non-reusable cursor identity within a Session Tree."""

    value: str

    def __post_init__(self) -> None:
        if not _LANE_ID_RE.fullmatch(self.value):
            raise ValueError("LaneId must be 'main' or a canonical UUID")
        if len(self.value) == 36:
            _require_canonical_uuid(self.value, "LaneId")

    @classmethod
    def main(cls) -> Self:
        return cls("main")

    @classmethod
    def new(cls) -> Self:
        return cls(new_uuid7())

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class EntryId:
    """One immutable Session Entry identity."""

    value: str

    def __post_init__(self) -> None:
        _require_canonical_uuid(self.value, "EntryId")

    @classmethod
    def new(cls) -> Self:
        return cls(new_uuid7())

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class OperationId:
    """One accepted Agent operation within a Session Lane."""

    value: str

    def __post_init__(self) -> None:
        _require_canonical_uuid(self.value, "OperationId")

    @classmethod
    def new(cls) -> Self:
        return cls(new_uuid7())

    @classmethod
    def deterministic(cls, *, idempotency_key: str) -> Self:
        return cls(deterministic_uuid(seed=idempotency_key, name="agent-operation"))

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class AttemptId:
    """One provider or Tool effect attempt identity."""

    value: str

    def __post_init__(self) -> None:
        _require_canonical_uuid(self.value, "AttemptId")

    @classmethod
    def new(cls) -> Self:
        return cls(new_uuid7())

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class IntentId:
    """One effect intent identity, unique within its session."""

    value: str

    def __post_init__(self) -> None:
        _require_canonical_uuid(self.value, "IntentId")

    @classmethod
    def new(cls) -> Self:
        return cls(new_uuid7())

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ProjectionId:
    """One immutable context projection identity."""

    value: str

    def __post_init__(self) -> None:
        _require_canonical_uuid(self.value, "ProjectionId")

    @classmethod
    def new(cls) -> Self:
        return cls(new_uuid7())

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class StageIntentId:
    """One durable Fast stage identity, deterministic over run and stage name."""

    value: str

    def __post_init__(self) -> None:
        _require_canonical_uuid(self.value, "StageIntentId")

    @classmethod
    def deterministic(cls, *, run_id: str, name: str) -> Self:
        return cls(deterministic_uuid(seed=run_id, name=name))

    def __str__(self) -> str:
        return self.value


__all__ = [
    "AttemptId",
    "EntryId",
    "IntentId",
    "LaneId",
    "OperationId",
    "ProjectionId",
    "SessionId",
    "StageIntentId",
    "deterministic_uuid",
    "new_uuid7",
]
