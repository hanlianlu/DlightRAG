# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Data source adapters for document ingestion."""

from typing import TYPE_CHECKING

from dlightrag.sourcing.base import AsyncDataSource

if TYPE_CHECKING:
    from dlightrag.sourcing.url import URLDataSource

__all__ = ["AsyncDataSource", "URLDataSource"]


def __getattr__(name: str):
    """Lazy import optional sourcing adapters."""
    if name == "AzureBlobDataSource":
        from dlightrag.sourcing.azure_blob import AzureBlobDataSource

        return AzureBlobDataSource
    if name == "S3DataSource":
        from dlightrag.sourcing.aws_s3 import S3DataSource

        return S3DataSource
    if name == "URLDataSource":
        from dlightrag.sourcing.url import URLDataSource

        return URLDataSource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
