# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Typed browser route request payloads."""

from pydantic import Field

from dlightrag.core.client_contracts import ClientContractModel, ConversationMessage


class WebAnswerRequest(ClientContractModel):
    query: str = ""
    images: list[str] = Field(default_factory=list, max_length=3)
    workspaces: list[str] | None = None
    conversation_history: list[ConversationMessage] | None = None
    session_id: str = ""


__all__ = ["WebAnswerRequest"]
