# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Typed context contributions and deterministic model-message projection."""

from dataclasses import dataclass
from typing import Any, Literal

from dlightrag.engine.ai.tokens import estimate_messages_tokens

type ContextAuthority = Literal[
    "system",
    "workspace",
    "conversation",
    "user",
    "working",
    "evidence",
    "profile",
    "reference",
    "visual",
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
    # The run-local visual lane re-renders every request, so it trails the
    # byte-stable per-Run tail instead of preceding it (ADR 0015).
    "visual": 80,
}

#: The authorities whose text or pixels a Citation may point at.
_CITABLE_AUTHORITIES: frozenset[ContextAuthority] = frozenset({"evidence", "visual"})


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
        if self.citable and self.authority not in _CITABLE_AUTHORITIES:
            raise ValueError("only evidence contributions, text or pixels, may be citable")

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
