# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Caller-facing durable Answer Run service and roster projections.

Answer domain contracts live under :mod:`dlightrag.engine.answer`; importing
this package does not eagerly load the AnswerService implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .child_roster import (
        CHILD_ROSTER_PAGE_DEFAULT_LIMIT,
        CHILD_ROSTER_PAGE_MAX_LIMIT,
        ChildRosterCursor,
        ChildRosterCursorCodec,
        ChildRosterCursorError,
        ChildRosterPage,
        ChildRosterPageRequest,
        ChildRosterRowPage,
    )
    from .service import (
        CHILD_CONTROL_SUCCESS_OUTCOMES,
        AgentControlReceipt,
        AgentTranscriptTail,
        AnswerHistoryResource,
        AnswerInputArtifact,
        AnswerRequest,
        AnswerRunAcceptor,
        AnswerRuntimeUnavailableError,
        AnswerService,
        ChildControlReceipt,
        ChildObservation,
        child_control_receipt_payload,
        child_control_succeeded,
    )

_SERVICE_EXPORTS = {
    "AgentControlReceipt",
    "AgentTranscriptTail",
    "AnswerHistoryResource",
    "AnswerInputArtifact",
    "AnswerRequest",
    "AnswerRunAcceptor",
    "AnswerRuntimeUnavailableError",
    "AnswerService",
    "CHILD_CONTROL_SUCCESS_OUTCOMES",
    "ChildControlReceipt",
    "ChildObservation",
    "child_control_receipt_payload",
    "child_control_succeeded",
}

_CONTRACT_EXPORTS = {
    "CHILD_ROSTER_PAGE_DEFAULT_LIMIT",
    "CHILD_ROSTER_PAGE_MAX_LIMIT",
    "ChildRosterCursor",
    "ChildRosterCursorCodec",
    "ChildRosterCursorError",
    "ChildRosterPage",
    "ChildRosterPageRequest",
    "ChildRosterRowPage",
}

__all__ = [
    "AgentControlReceipt",
    "AgentTranscriptTail",
    "AnswerHistoryResource",
    "AnswerInputArtifact",
    "AnswerRequest",
    "AnswerRunAcceptor",
    "AnswerRuntimeUnavailableError",
    "AnswerService",
    "CHILD_CONTROL_SUCCESS_OUTCOMES",
    "ChildControlReceipt",
    "ChildObservation",
    "child_control_receipt_payload",
    "child_control_succeeded",
    "CHILD_ROSTER_PAGE_DEFAULT_LIMIT",
    "CHILD_ROSTER_PAGE_MAX_LIMIT",
    "ChildRosterCursor",
    "ChildRosterCursorCodec",
    "ChildRosterCursorError",
    "ChildRosterPage",
    "ChildRosterPageRequest",
    "ChildRosterRowPage",
]


def __getattr__(name: str) -> Any:
    if name in _CONTRACT_EXPORTS:
        from .child_roster import (
            CHILD_ROSTER_PAGE_DEFAULT_LIMIT,
            CHILD_ROSTER_PAGE_MAX_LIMIT,
            ChildRosterCursor,
            ChildRosterCursorCodec,
            ChildRosterCursorError,
            ChildRosterPage,
            ChildRosterPageRequest,
            ChildRosterRowPage,
        )

        return {
            "CHILD_ROSTER_PAGE_DEFAULT_LIMIT": CHILD_ROSTER_PAGE_DEFAULT_LIMIT,
            "CHILD_ROSTER_PAGE_MAX_LIMIT": CHILD_ROSTER_PAGE_MAX_LIMIT,
            "ChildRosterCursor": ChildRosterCursor,
            "ChildRosterCursorCodec": ChildRosterCursorCodec,
            "ChildRosterCursorError": ChildRosterCursorError,
            "ChildRosterPage": ChildRosterPage,
            "ChildRosterPageRequest": ChildRosterPageRequest,
            "ChildRosterRowPage": ChildRosterRowPage,
        }[name]
    if name in _SERVICE_EXPORTS:
        from .service import (
            CHILD_CONTROL_SUCCESS_OUTCOMES,
            AgentControlReceipt,
            AgentTranscriptTail,
            AnswerHistoryResource,
            AnswerInputArtifact,
            AnswerRequest,
            AnswerRunAcceptor,
            AnswerRuntimeUnavailableError,
            AnswerService,
            ChildControlReceipt,
            ChildObservation,
            child_control_receipt_payload,
            child_control_succeeded,
        )

        return {
            "AgentControlReceipt": AgentControlReceipt,
            "AgentTranscriptTail": AgentTranscriptTail,
            "AnswerHistoryResource": AnswerHistoryResource,
            "AnswerInputArtifact": AnswerInputArtifact,
            "AnswerRequest": AnswerRequest,
            "AnswerRunAcceptor": AnswerRunAcceptor,
            "AnswerRuntimeUnavailableError": AnswerRuntimeUnavailableError,
            "AnswerService": AnswerService,
            "CHILD_CONTROL_SUCCESS_OUTCOMES": CHILD_CONTROL_SUCCESS_OUTCOMES,
            "ChildControlReceipt": ChildControlReceipt,
            "ChildObservation": ChildObservation,
            "child_control_receipt_payload": child_control_receipt_payload,
            "child_control_succeeded": child_control_succeeded,
        }[name]
    raise AttributeError(name)
