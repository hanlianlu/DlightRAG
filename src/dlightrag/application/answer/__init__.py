# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Read-side Answer projections that are derived rather than declared.

A module lands here when a transport must say something about a settled Answer that the Answer
itself may not be trusted to state -- and that is therefore read back from the Run's own durable
records. ``provenance`` is the first: which Agent Skill's verifier, if any, actually ran behind
the Artifacts an Answer published.
"""

__all__: list[str] = []
