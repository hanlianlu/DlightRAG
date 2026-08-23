# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Typed context contributions and deterministic model-message projection."""

from dataclasses import dataclass
from typing import Any, Literal

from dlightrag.ai.tokens import estimate_messages_tokens

type ContextAuthority = Literal[
    "system",
    "workspace",
    "conversation",
    "user",
    "working",
    "evidence",
    "profile",
    "reference",
]

_AUTHORITY_ORDER: dict[ContextAuthority, int] = {
    "system": 0,
    "workspace": 10,
    "conversation": 20,
    "user": 30,
    "working": 40,
    "evidence": 50,
    "profile": 60,
    "reference": 70,
}


@dataclass(frozen=True, slots=True)
class ContextContribution:
    """Messages from one authority with explicit prompt semantics.

    Storage and retrieval remain owned by the contributor. The Agent kernel
    receives only model-ready messages plus facts needed for safe projection.
    """

    source: str
    authority: ContextAuthority
    messages: tuple[dict[str, Any], ...]
    citable: bool = False
    compressible: bool = True

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("context contribution source cannot be empty")
        if self.citable and self.authority != "evidence":
            raise ValueError("only evidence contributions may be citable")

    @property
    def estimated_tokens(self) -> int:
        return estimate_messages_tokens(list(self.messages))


@dataclass(frozen=True, slots=True)
class ProjectedContext:
    messages: tuple[dict[str, Any], ...]
    sources: tuple[str, ...]
    estimated_tokens: int


class ContextProjector:
    """Order contributions by authority while preserving source-local order."""

    def project(self, contributions: list[ContextContribution]) -> ProjectedContext:
        ordered = sorted(
            enumerate(contributions),
            key=lambda item: (_AUTHORITY_ORDER[item[1].authority], item[0]),
        )
        messages = tuple(message for _, item in ordered for message in item.messages)
        return ProjectedContext(
            messages=messages,
            sources=tuple(item.source for _, item in ordered),
            estimated_tokens=estimate_messages_tokens(list(messages)),
        )


__all__ = [
    "ContextAuthority",
    "ContextContribution",
    "ContextProjector",
    "ProjectedContext",
]
