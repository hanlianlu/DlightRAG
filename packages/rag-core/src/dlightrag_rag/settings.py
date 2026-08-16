# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Immutable settings for one workspace RAG pipeline."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TypedDict

from dlightrag_ai.settings import (
    EmbeddingSettings,
    ModelRoleSettings,
    ModelSettings,
    RerankSettings,
    freeze_settings_value,
)


class LightRAGPipelineKwargs(TypedDict):
    max_parallel_insert: int
    max_parallel_parse_native: int
    max_parallel_parse_mineru: int
    max_parallel_parse_docling: int
    max_parallel_analyze: int
    queue_size_parse: int
    queue_size_analyze: int
    queue_size_insert: int


@dataclass(frozen=True, slots=True)
class RagSettings:
    """Fully resolved settings owned by one workspace RAG pipeline."""

    model_roles: ModelRoleSettings
    embedding: EmbeddingSettings
    rerank: RerankSettings
    rerank_scoring_model: ModelSettings
    rag_pipeline_max_async: int
    embedding_func_max_async: int
    embedding_batch_num: int
    max_parallel_insert: int
    max_parallel_parse_native: int
    max_parallel_parse_mineru: int
    max_parallel_parse_docling: int
    max_parallel_analyze: int
    queue_size_parse: int
    queue_size_analyze: int
    queue_size_insert: int
    read_only: bool = False
    input_root: Path = Path("dlightrag_storage/inputs")
    parser_rules: str = "*:mineru-iteP"
    docling_active: bool = False
    docling_code_formula_preset: str | None = None
    parser_min_image_pixel: int = 80
    extraction_language: str = "English"
    entity_type_prompt_file: str | None = None
    entity_extraction_use_json: bool = True
    chunk_p_token_size: int = 2000
    kg_entity_types: tuple[str, ...] = ()
    kg_chunk_pick_method: Literal["VECTOR", "WEIGHT"] = "VECTOR"
    max_entity_tokens: int = 6000
    max_relation_tokens: int = 8000
    max_total_tokens: int = 40000
    direct_visual_top_k: int = 20
    rrf_k: int = 60
    thumb_cache_size: int = 256
    thumb_max_px: int = 300
    ingestion_replace_default: bool = False
    retain_remote_source_files: bool = False
    url_ingest_max_bytes: int = 100 * 1024 * 1024
    url_ingest_private_host_allowlist: tuple[str, ...] = ()
    blob_connection_string: str | None = None
    azure_sas_expiry: int = 3600
    s3_presign_expiry: int = 3600
    s3_region: str | None = None
    chunk_options: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        for field_name in (
            "rag_pipeline_max_async",
            "embedding_func_max_async",
            "embedding_batch_num",
            "max_parallel_insert",
            "max_parallel_parse_native",
            "max_parallel_parse_mineru",
            "max_parallel_parse_docling",
            "max_parallel_analyze",
            "queue_size_parse",
            "queue_size_analyze",
            "queue_size_insert",
            "parser_min_image_pixel",
            "chunk_p_token_size",
            "max_entity_tokens",
            "max_relation_tokens",
            "max_total_tokens",
            "rrf_k",
            "thumb_cache_size",
            "thumb_max_px",
            "url_ingest_max_bytes",
            "azure_sas_expiry",
            "s3_presign_expiry",
        ):
            if int(getattr(self, field_name)) < 1:
                raise ValueError(f"{field_name} must be positive")
        if self.direct_visual_top_k < 0:
            raise ValueError("direct_visual_top_k cannot be negative")
        object.__setattr__(
            self,
            "chunk_options",
            freeze_settings_value(self.chunk_options),
        )
        object.__setattr__(self, "input_root", Path(self.input_root))
        object.__setattr__(self, "kg_entity_types", tuple(self.kg_entity_types))
        object.__setattr__(
            self,
            "url_ingest_private_host_allowlist",
            tuple(self.url_ingest_private_host_allowlist),
        )

    def lightrag_pipeline_kwargs(self) -> LightRAGPipelineKwargs:
        """Return LightRAG's parser and insertion pipeline controls."""
        return {
            "max_parallel_insert": self.max_parallel_insert,
            "max_parallel_parse_native": self.max_parallel_parse_native,
            "max_parallel_parse_mineru": self.max_parallel_parse_mineru,
            "max_parallel_parse_docling": self.max_parallel_parse_docling,
            "max_parallel_analyze": self.max_parallel_analyze,
            "queue_size_parse": self.queue_size_parse,
            "queue_size_analyze": self.queue_size_analyze,
            "queue_size_insert": self.queue_size_insert,
        }

    def addon_params(self) -> dict[str, Any]:
        """Return LightRAG addon parameters from immutable RAG-owned facts."""
        params: dict[str, Any] = {
            "language": self.extraction_language,
            "chunker": {
                "paragraph_semantic": {
                    "chunk_token_size": self.chunk_p_token_size,
                }
            },
        }
        if self.entity_type_prompt_file:
            params["entity_type_prompt_file"] = self.entity_type_prompt_file
        if self.kg_entity_types:
            params["entity_types_guidance"] = (
                "Prioritize domain entities in these categories: "
                f"{', '.join(self.kg_entity_types)}."
            )
        return params


__all__ = ["LightRAGPipelineKwargs", "RagSettings"]
