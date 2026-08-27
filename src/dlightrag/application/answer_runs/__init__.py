# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Caller-facing durable Answer Run use case and contracts.

Contract submodules are importable without loading AnswerService. The service
module pulls Research acceptance measurement, which must not run during
contract imports.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AgentControlReceipt",
    "AgentTranscriptTail",
    "AnswerHistoryResource",
    "AnswerInputArtifact",
    "AnswerRequest",
    "AnswerRunAcceptor",
    "AnswerRuntimeUnavailableError",
    "AnswerService",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .service import (
            AgentControlReceipt,
            AgentTranscriptTail,
            AnswerHistoryResource,
            AnswerInputArtifact,
            AnswerRequest,
            AnswerRunAcceptor,
            AnswerRuntimeUnavailableError,
            AnswerService,
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
        }[name]
    raise AttributeError(name)
