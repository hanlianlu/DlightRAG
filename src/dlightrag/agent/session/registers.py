# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Closed typed current-state registers for one Agent Session.

Entries carry immutable conversation meaning. Registers carry only current
runtime state and are replaced with exact sequence-token compare-and-set.
The union grows only with a real writer; there is no custom JSON namespace.
"""

from dataclasses import dataclass
from typing import Any, Literal

from dlightrag.agent.session.ids import EntryId, LaneId

REGISTER_SCHEMA_VERSION = 1

RegisterKind = Literal["lane_head", "lane_state"]


@dataclass(frozen=True, slots=True)
class RegisterRef:
    """Stable identity of one closed register cell."""

    kind: RegisterKind
    key: str

    def __post_init__(self) -> None:
        if self.kind not in {"lane_head", "lane_state"}:
            raise ValueError("unknown Agent Session register kind")
        if not self.key:
            raise ValueError("register key cannot be empty")


@dataclass(frozen=True, slots=True)
class LaneHead:
    """The immutable-entry leaf where one Lane appends next."""

    lane_id: LaneId
    entry_id: EntryId | None

    @property
    def ref(self) -> RegisterRef:
        return RegisterRef("lane_head", self.lane_id.value)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": REGISTER_SCHEMA_VERSION,
            "lane_id": self.lane_id.value,
            "entry_id": self.entry_id.value if self.entry_id is not None else None,
        }


@dataclass(frozen=True, slots=True)
class LaneState:
    """The bounded mutable lifecycle of one Lane cursor."""

    lane_id: LaneId
    archived: bool = False
    active_operation_id: str | None = None

    def __post_init__(self) -> None:
        if self.archived and self.active_operation_id is not None:
            raise ValueError("an archived Lane cannot own an active Operation")

    @property
    def ref(self) -> RegisterRef:
        return RegisterRef("lane_state", self.lane_id.value)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": REGISTER_SCHEMA_VERSION,
            "lane_id": self.lane_id.value,
            "archived": self.archived,
            "active_operation_id": self.active_operation_id,
        }


type SessionRegister = LaneHead | LaneState


@dataclass(frozen=True, slots=True)
class RegisterRecord:
    """One current register value and its exact CAS sequence token."""

    value: SessionRegister
    sequence: int

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError("register sequence must be positive")

    @property
    def ref(self) -> RegisterRef:
        return self.value.ref


@dataclass(frozen=True, slots=True)
class SetRegister:
    value: SessionRegister

    @property
    def ref(self) -> RegisterRef:
        return self.value.ref


@dataclass(frozen=True, slots=True)
class DeleteRegister:
    ref: RegisterRef


type RegisterWrite = SetRegister | DeleteRegister


def decode_register(*, kind: str, payload: dict[str, Any]) -> SessionRegister:
    """Decode one value from the closed durable register union."""
    if int(payload.get("schema_version") or 0) != REGISTER_SCHEMA_VERSION:
        raise ValueError("Agent Session register schema version is not current")
    lane_id = LaneId(str(payload["lane_id"]))
    if kind == "lane_head":
        raw_entry_id = payload.get("entry_id")
        return LaneHead(
            lane_id=lane_id,
            entry_id=EntryId(str(raw_entry_id)) if raw_entry_id is not None else None,
        )
    if kind == "lane_state":
        return LaneState(
            lane_id=lane_id,
            archived=bool(payload.get("archived") or False),
            active_operation_id=(
                str(payload["active_operation_id"])
                if payload.get("active_operation_id") is not None
                else None
            ),
        )
    raise ValueError(f"unknown Agent Session register kind: {kind}")


__all__ = [
    "REGISTER_SCHEMA_VERSION",
    "DeleteRegister",
    "LaneHead",
    "LaneState",
    "RegisterRecord",
    "RegisterRef",
    "RegisterWrite",
    "SessionRegister",
    "SetRegister",
    "decode_register",
]
