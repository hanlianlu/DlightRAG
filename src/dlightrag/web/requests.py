# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Typed browser route request payloads."""

from uuid import UUID

from pydantic import Field

from dlightrag.core.client_contracts import ClientContractModel


class WebAnswerRequest(ClientContractModel):
    query: str = ""
    images: list[str] = Field(default_factory=list)
    workspaces: list[str] | None = None
    conversation_id: UUID
    submission_id: UUID


__all__ = ["WebAnswerRequest"]
