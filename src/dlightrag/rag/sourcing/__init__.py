# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Data source adapters for document ingestion."""

from dlightrag.rag.sourcing.aws_s3 import S3DataSource
from dlightrag.rag.sourcing.azure_blob import AzureBlobDataSource
from dlightrag.rag.sourcing.base import AsyncDataSource, SourceDocument
from dlightrag.rag.sourcing.url import URLDataSource

__all__ = [
    "AsyncDataSource",
    "AzureBlobDataSource",
    "S3DataSource",
    "SourceDocument",
    "URLDataSource",
]
