# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Inbound MCP tool handlers registered on the one server/lifespan."""


def register() -> None:
    from . import answer_runs, corpus_admin, memory, model_catalogue, retrieval  # noqa: F401
