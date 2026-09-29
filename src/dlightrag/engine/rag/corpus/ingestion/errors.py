# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Corpus ingestion execution errors independent of lifecycle persistence."""


class RetryOutcomeUncertainError(RuntimeError):
    """A retry may have committed, but its authoritative status is unavailable."""


class ParserInputPlacementError(RuntimeError):
    """A document's parser input could not be copied to where LightRAG reads it.

    Placement precedes every upstream effect of its batch, so nothing changed.
    """


__all__ = ["ParserInputPlacementError", "RetryOutcomeUncertainError"]
