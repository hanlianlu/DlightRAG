# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Typed browser route request payloads."""

from uuid import UUID

from dlightrag.engine.answer.client_contracts import AnswerEffort, ClientContractModel
from dlightrag.engine.answer.mode import AnswerMode


class WebAnswerRequest(ClientContractModel):
    query: str = ""
    workspaces: list[str] | None = None
    conversation_id: UUID | None = None
    submission_id: UUID
    mode: AnswerMode | None = None
    requested_skill: str | None = None
    effort: AnswerEffort | None = None


__all__ = ["WebAnswerRequest"]
