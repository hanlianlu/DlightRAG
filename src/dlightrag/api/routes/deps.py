# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Common dependencies for API routes."""

from fastapi import Request

from dlightrag.config import get_config
from dlightrag.core.servicemanager import RAGServiceManager


def get_manager(request: Request) -> RAGServiceManager:
    return request.app.state.manager


def resolve_workspace(ws: str | None) -> str:
    from dlightrag.utils import normalize_workspace

    return normalize_workspace(ws or get_config().workspace)
