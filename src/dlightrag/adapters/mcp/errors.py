# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Refusals the MCP adapter raises itself."""


class ToolRejection(Exception):
    """A refusal the MCP caller can act on; its message is shown as is.

    Tool code raises it for its own checks (unknown arguments, access denials,
    missing runs). Anything untyped stays behind the internal-failure text.
    """


__all__ = ["ToolRejection"]
