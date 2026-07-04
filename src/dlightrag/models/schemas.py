# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Pydantic schemas for answer output."""

from pydantic import BaseModel, Field, field_validator


class Reference(BaseModel):
    """A document-level reference cited in the answer."""

    id: str = Field(description="Reference id matching [n] in inline citations")
    title: str = Field(description="Document title/filename")

    @field_validator("id", mode="before")
    @classmethod
    def _coerce_id(cls, value: object) -> str:
        return str(value)


__all__ = ["Reference"]
