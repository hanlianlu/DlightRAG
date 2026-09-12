# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Finite product ceilings; no Run, storage, or execution dependencies."""

from pydantic import BaseModel, ConfigDict, Field


class ConnectionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    oauth_callback_url: str | None = Field(default=None, max_length=2048)
    oauth_timeout: float = Field(default=300, ge=30, le=600)
    max_connections: int = Field(default=20, ge=1, le=100)
    max_enabled_tools: int = Field(default=256, ge=1, le=1024)
    max_tools: int = Field(default=128, ge=1, le=256)
    max_pages: int = Field(default=16, ge=1, le=32)
    max_schema_bytes: int = Field(default=32768, ge=1, le=65536)
    max_description_bytes: int = Field(default=8192, ge=1, le=16384)
    max_catalogue_bytes: int = Field(default=524288, ge=1, le=1048576)
    discovery_concurrency: int = Field(default=4, ge=1, le=16)
    refresh_seconds: float = Field(default=300, ge=1, le=3600)
    discovery_timeout: float = Field(default=30, ge=1, le=120)
    call_timeout: float = Field(default=60, ge=1, le=120)
    call_concurrency: int = Field(default=8, ge=1, le=32)
    max_result_bytes: int = Field(default=262144, ge=1, le=1048576)
    max_call_argument_bytes: int = Field(default=65536, ge=1, le=262144)
    max_result_parts: int = Field(default=32, ge=1, le=128)
    connect_timeout: float = Field(default=10, ge=1, le=30)
    idle_timeout: float = Field(default=15, ge=1, le=60)
    max_response_bytes: int = Field(default=1048576, ge=1, le=4194304)
    allow_private_hosts: tuple[str, ...] = ()
    require_https: bool = True
