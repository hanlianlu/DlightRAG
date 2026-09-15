# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Starter endpoints the create form offers as one-click fills.

A Preset is product-owned copy plus an endpoint and the authentication choice that suits its
tier: a service that answers unauthenticated starts on ``none``, and one that cannot be used
without an account starts on the choice that account needs. Choosing a Preset never creates,
enables, or authorizes anything -- it only fills the create form, and the ordinary create
command remains the one authority that validates the endpoint and owns the Connection.

This list is small, static, code-owned, credential-free, and reviewed by hand. It is
deliberately not a marketplace, a registry lookup, or a discovery surface: the plan's exclusion
of a Connection marketplace still holds.
"""

from dataclasses import dataclass
from typing import Literal

Authentication = Literal["none", "bearer", "oauth"]


@dataclass(frozen=True)
class ConnectionPreset:
    """One starter endpoint and the authentication tab that fits it."""

    preset_id: str
    label: str
    endpoint: str
    default_authentication: Authentication


PRESETS: tuple[ConnectionPreset, ...] = (
    ConnectionPreset("notion", "Notion", "https://mcp.notion.com/mcp", "oauth"),
    ConnectionPreset("huggingface", "Hugging Face", "https://huggingface.co/mcp", "none"),
    ConnectionPreset("wolfram", "Wolfram", "https://agenttools.wolfram.com/mcp", "none"),
)
