# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unified representational multimodal RAG module.

Provides visual page embedding pipeline as an alternative to
RAGAnything's caption-based approach. Each document page is rendered
as a high-DPI image, embedded via multimodal embedding model, and
stored alongside a LightRAG knowledge graph built from VLM-extracted text.
"""

from dlightrag.unifiedrepresent.engine import UnifiedRepresentEngine
from dlightrag.unifiedrepresent.renderer import PageRenderer

__all__ = ["UnifiedRepresentEngine", "PageRenderer"]
