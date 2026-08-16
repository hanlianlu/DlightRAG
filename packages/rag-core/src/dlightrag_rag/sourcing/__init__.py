# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Data source adapters for document ingestion."""

from dlightrag_rag.sourcing.aws_s3 import S3DataSource
from dlightrag_rag.sourcing.azure_blob import AzureBlobDataSource
from dlightrag_rag.sourcing.base import AsyncDataSource, SourceDocument
from dlightrag_rag.sourcing.url import URLDataSource

__all__ = [
    "AsyncDataSource",
    "AzureBlobDataSource",
    "S3DataSource",
    "SourceDocument",
    "URLDataSource",
]
