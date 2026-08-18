# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The bounded, immutable Prepared Answer Input and its canonical serialization.

Acceptance pins every reachable model/adapter/profile/capability/policy fact
into a Prepared Answer Input, projects history, and builds the resource
manifest before any run row exists (M3 acceptance ordering). Queued and running
rows store exactly one ``prepared_input_json``; terminal transitions clear it,
and recovery never re-resolves configuration from a stored prepared input.

The canonical encoding is bounded at 8 MiB. An oversized prepared input is
rejected as ``prepared_input_too_large`` before a run row is written.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dlightrag_agent.session.effects import canonical_json
from dlightrag_agent.session.ids import SessionId

MAX_PREPARED_INPUT_BYTES = 8 * 1024 * 1024
PREPARED_INPUT_SCHEMA_VERSION = 1


class PreparedInputTooLargeError(ValueError):
    """The canonical prepared input exceeds the 8 MiB durable bound."""

    def __init__(self, *, encoded_bytes: int) -> None:
        self.encoded_bytes = encoded_bytes
        super().__init__(
            "prepared_input_too_large: "
            f"{encoded_bytes} bytes exceed the {MAX_PREPARED_INPUT_BYTES} byte bound"
        )


@dataclass(frozen=True, slots=True)
class PreparedAnswerInput:
    """One immutable, bounded execution description accepted for a new Answer Run.

    ``session_id`` is the UUIDv7 Research session pinned at acceptance; the
    first journal transaction must create exactly that session (M3-D10).
    ``fingerprint`` is the public request fingerprint computed before any
    enrichment, retained so acceptance replay and idempotency never depend on
    prepared profile facts.
    """

    session_id: str
    fingerprint: str
    query: str
    workspaces: tuple[str, ...]
    history: tuple[Mapping[str, Any], ...] = ()
    profile_facts: tuple[Mapping[str, Any], ...] = ()
    resource_manifest: tuple[Mapping[str, Any], ...] = ()
    schema_version: int = PREPARED_INPUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        SessionId(self.session_id)  # validates the canonical UUID form
        if not self.query.strip():
            raise ValueError("prepared input query cannot be empty")
        if not self.workspaces:
            raise ValueError("prepared input requires at least one workspace")
        if len(self.fingerprint) != 64:
            raise ValueError("prepared input fingerprint must be a SHA-256 hex digest")
        if self.schema_version != PREPARED_INPUT_SCHEMA_VERSION:
            raise ValueError("prepared input schema version is not current")

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "fingerprint": self.fingerprint,
            "query": self.query,
            "workspaces": list(self.workspaces),
            "history": [dict(turn) for turn in self.history],
            "profile_facts": [dict(fact) for fact in self.profile_facts],
            "resource_manifest": [dict(resource) for resource in self.resource_manifest],
            "schema_version": self.schema_version,
        }

    def canonical_json(self) -> str:
        """Return the canonical UTF-8 JSON the durable store keeps."""
        return canonical_json(self.canonical_payload())


def prepared_input_bytes(prepared: PreparedAnswerInput) -> bytes:
    """Return the exact canonical UTF-8 encoding of one prepared input."""
    return prepared.canonical_json().encode("utf-8")


def encode_prepared_input(prepared: PreparedAnswerInput) -> bytes:
    """Encode one prepared input, enforcing the 8 MiB durable bound.

    Raises :class:`PreparedInputTooLargeError` before any run row exists; the
    caller never writes a run for an oversized prepared input.
    """
    encoded = prepared_input_bytes(prepared)
    if len(encoded) > MAX_PREPARED_INPUT_BYTES:
        raise PreparedInputTooLargeError(encoded_bytes=len(encoded))
    return encoded


def prepared_input_with_payload(
    prepared: PreparedAnswerInput,
    *,
    extra_bytes: int,
    field: str = "resource_manifest",
) -> PreparedAnswerInput:
    """Return a prepared input whose canonical size grew by ``extra_bytes``.

    A test/construction helper: pads the declared manifest field with inert
    text so boundary checks can pin exact byte sizes without hand-encoding.
    """
    if extra_bytes < 0:
        raise ValueError("extra_bytes cannot be negative")
    padding = "x" * extra_bytes
    if field == "resource_manifest":
        manifest = (
            *prepared.resource_manifest,
            {"kind": "padding", "bytes": padding},
        )
        return PreparedAnswerInput(
            session_id=prepared.session_id,
            fingerprint=prepared.fingerprint,
            query=prepared.query,
            workspaces=prepared.workspaces,
            history=prepared.history,
            profile_facts=prepared.profile_facts,
            resource_manifest=manifest,
        )
    if field == "profile_facts":
        return PreparedAnswerInput(
            session_id=prepared.session_id,
            fingerprint=prepared.fingerprint,
            query=prepared.query,
            workspaces=prepared.workspaces,
            history=prepared.history,
            profile_facts=(*prepared.profile_facts, {"kind": "padding", "bytes": padding}),
            resource_manifest=prepared.resource_manifest,
        )
    raise ValueError(f"unsupported prepared input padding field: {field}")


def minimal_prepared_input(
    *,
    query: str,
    workspaces: Sequence[str] = ("default",),
    session_id: str | None = None,
    fingerprint: str = "0" * 64,
) -> PreparedAnswerInput:
    """Build a minimal valid prepared input for tests and acceptance plumbing."""
    from dlightrag_agent.session.ids import SessionId

    return PreparedAnswerInput(
        session_id=session_id or SessionId.new().value,
        fingerprint=fingerprint,
        query=query,
        workspaces=tuple(workspaces),
    )


__all__ = [
    "MAX_PREPARED_INPUT_BYTES",
    "PREPARED_INPUT_SCHEMA_VERSION",
    "PreparedAnswerInput",
    "PreparedInputTooLargeError",
    "encode_prepared_input",
    "minimal_prepared_input",
    "prepared_input_bytes",
    "prepared_input_with_payload",
]
