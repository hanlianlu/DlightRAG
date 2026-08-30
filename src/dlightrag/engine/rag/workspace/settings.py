# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Canonical immutable corpus and workspace-RAG settings."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal, Self, TypedDict

from pydantic import Field, field_serializer, field_validator, model_validator

from dlightrag.engine.ai.settings import (
    FrozenSettings,
    ModelsSettings,
    freeze_settings_value,
    thaw_settings_value,
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


class ParserSettings(FrozenSettings):
    chunk_options: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("chunk_options", mode="after")
    @classmethod
    def _freeze(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return freeze_settings_value(value)

    @field_serializer("chunk_options")
    def _serialize_chunk_options(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return thaw_settings_value(value)


class ExtractionSettings(FrozenSettings):
    use_json: bool = True
    language: str = "English"
    entity_type_prompt_file: str | None = None

    @field_validator("entity_type_prompt_file")
    @classmethod
    def _validate_profile(cls, value: str | None) -> str | None:
        if value is None:
            return None
        name = value.strip()
        if not name:
            return None
        candidate = Path(name)
        if (
            "\\" in name
            or candidate.is_absolute()
            or candidate.name != name
            or ".." in candidate.parts
        ):
            raise ValueError(
                "entity_type_prompt_file must be a file name under PROMPT_DIR/entity_type"
            )
        if candidate.suffix.lower() not in {".yml", ".yaml"}:
            raise ValueError("entity_type_prompt_file must use a .yml or .yaml extension")
        return name


class VLMSidecarSettings(FrozenSettings):
    enabled: bool = True
    max_image_bytes: int = Field(default=5_242_880, ge=1)
    min_image_pixel: int = Field(default=80, ge=1)
    surrounding_leading_max_tokens: int | None = Field(default=256, ge=0)
    surrounding_trailing_max_tokens: int | None = Field(default=256, ge=0)
    _ENV_MAP: ClassVar[dict[str, str]] = {
        "enabled": "VLM_PROCESS_ENABLE",
        "max_image_bytes": "VLM_MAX_IMAGE_BYTES",
        "min_image_pixel": "VLM_MIN_IMAGE_PIXEL",
        "surrounding_leading_max_tokens": "SURROUNDING_LEADING_MAX_TOKENS",
        "surrounding_trailing_max_tokens": "SURROUNDING_TRAILING_MAX_TOKENS",
    }


class MinerUSidecarSettings(FrozenSettings):
    api_mode: Literal["local", "official"] = "local"
    api_token: str | None = None
    official_endpoint: str = "https://mineru.net"
    local_endpoint: str = "http://127.0.0.1:8210"
    language: Literal[
        "ch",
        "ch_server",
        "korean",
        "ta",
        "te",
        "ka",
        "th",
        "el",
        "arabic",
        "east_slavic",
        "cyrillic",
        "devanagari",
    ] = "ch"
    backend: Literal["pipeline", "vlm-engine", "hybrid-engine"] = "hybrid-engine"
    poll_interval_seconds: int = Field(default=5, ge=1)
    max_polls: int = Field(default=1440, ge=1)
    _ENV_MAP: ClassVar[dict[str, str]] = {
        "api_mode": "MINERU_API_MODE",
        "api_token": "MINERU_API_TOKEN",
        "official_endpoint": "MINERU_OFFICIAL_ENDPOINT",
        "local_endpoint": "MINERU_LOCAL_ENDPOINT",
        "language": "MINERU_LANGUAGE",
        "backend": "MINERU_LOCAL_BACKEND",
        "poll_interval_seconds": "MINERU_POLL_INTERVAL_SECONDS",
        "max_polls": "MINERU_MAX_POLLS",
    }


class DoclingSidecarSettings(FrozenSettings):
    endpoint: str = "http://127.0.0.1:5001"
    do_formula_enrichment: bool = True
    force_ocr: bool = True
    code_formula_preset: str | None = "granite_docling"
    poll_interval_seconds: int = Field(default=5, ge=1)
    max_polls: int = Field(default=1440, ge=1)
    _ENV_MAP: ClassVar[dict[str, str]] = {
        "endpoint": "DOCLING_ENDPOINT",
        "do_formula_enrichment": "DOCLING_DO_FORMULA_ENRICHMENT",
        "force_ocr": "DOCLING_FORCE_OCR",
        "poll_interval_seconds": "DOCLING_POLL_INTERVAL_SECONDS",
        "max_polls": "DOCLING_MAX_POLLS",
    }


class ParserSidecarsSettings(FrozenSettings):
    vlm: VLMSidecarSettings = Field(default_factory=VLMSidecarSettings)
    mineru: MinerUSidecarSettings | None = None
    docling: DoclingSidecarSettings | None = Field(default_factory=DoclingSidecarSettings)

    @model_validator(mode="before")
    @classmethod
    def _select_parser(cls, value: Any) -> Any:
        if isinstance(value, dict):
            value = dict(value)
            if value.get("mineru") is not None and "docling" not in value:
                value["docling"] = None
            elif value.get("docling") is not None and "mineru" not in value:
                value["mineru"] = None
            elif value.get("mineru") is None and value.get("docling") is None:
                value["docling"] = {}
        return value

    @property
    def active_parser(self) -> Literal["mineru", "docling"]:
        return "mineru" if self.mineru is not None else "docling"


class PipelineSettings(FrozenSettings):
    max_concurrency: int = Field(default=16, ge=1)
    max_parallel_insert: int = Field(default=3, ge=1)
    max_parallel_parse_native: int = Field(default=5, ge=1)
    max_parallel_parse_mineru: int = Field(default=2, ge=1)
    max_parallel_parse_docling: int = Field(default=2, ge=1)
    max_parallel_analyze: int = Field(default=5, ge=1)
    queue_size_parse: int = Field(default=20, ge=1)
    queue_size_analyze: int = Field(default=100, ge=1)
    queue_size_insert: int = Field(default=4, ge=1)

    def lightrag_kwargs(self) -> LightRAGPipelineKwargs:
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


class IngestionSettings(FrozenSettings):
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    chunk_token_size: int = Field(default=2000, ge=1)
    replace_default: bool = False
    retain_remote_source_files: bool = False
    url_max_bytes: int = Field(default=100 * 1024 * 1024, ge=1)
    url_private_host_allowlist: tuple[str, ...] = ()
    max_upload_bytes: int = Field(default=100 * 1024 * 1024, ge=1)
    timeout: float | None = Field(default=None, ge=0)


class BM25ProfileSettings(FrozenSettings):
    name: str
    text_config: str
    languages: tuple[str, ...] = ()
    fallback: bool = False

    @field_validator("name", "text_config")
    @classmethod
    def _identifier(cls, value: str) -> str:
        import re

        text = value.strip()
        pattern = r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$"
        if not re.fullmatch(pattern, text):
            raise ValueError("BM25 identifier must be a safe PostgreSQL identifier")
        return text

    @field_validator("languages", mode="before")
    @classmethod
    def _languages(cls, value: Any) -> tuple[str, ...]:
        return tuple(str(item).strip().lower() for item in value if str(item).strip())

    @model_validator(mode="after")
    def _routing(self) -> Self:
        if self.fallback and self.languages:
            raise ValueError("BM25 fallback profiles must not declare languages")
        if not self.fallback and len(self.languages) != 1:
            raise ValueError("BM25 language profiles must declare exactly one language")
        return self


def _bm25_profiles() -> tuple[BM25ProfileSettings, ...]:
    rows = (
        ("zh", "public.jiebacfg", ("zh",), False),
        ("en", "english", ("en",), False),
        ("de", "german", ("de",), False),
        ("sv", "swedish", ("sv",), False),
        ("es", "spanish", ("es",), False),
        ("fr", "french", ("fr",), False),
        ("it", "italian", ("it",), False),
        ("pt", "portuguese", ("pt",), False),
        ("nl", "dutch", ("nl",), False),
        ("ru", "russian", ("ru",), False),
        ("da", "danish", ("da",), False),
        ("fi", "finnish", ("fi",), False),
        ("simple", "simple", (), True),
    )
    return tuple(
        BM25ProfileSettings(name=a, text_config=b, languages=c, fallback=d) for a, b, c, d in rows
    )


class RetrievalSettings(FrozenSettings):
    top_k: int = Field(default=40, ge=1)
    chunk_top_k: int = Field(default=20, ge=1)
    timeout: int = Field(default=300, gt=0)
    bm25_enabled: bool = True
    bm25_profiles: tuple[BM25ProfileSettings, ...] = Field(default_factory=_bm25_profiles)
    bm25_k1: float = Field(default=1.2, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    rrf_k: int = Field(default=60, ge=1)
    direct_visual_top_k: int = Field(default=20, ge=0)
    metadata_filter_exact_vector_threshold: int = Field(default=8192, ge=0)
    max_entity_tokens: int = Field(default=6000, ge=1)
    max_relation_tokens: int = Field(default=8000, ge=1)
    max_total_tokens: int = Field(default=40000, ge=1)
    kg_chunk_pick_method: Literal["VECTOR", "WEIGHT"] = "VECTOR"
    kg_entity_types: tuple[str, ...] = ()


class WorkspacePromotionSettings(FrozenSettings):
    """Thresholds for moving one shared workspace into dedicated partitions.

    Commit 1 exposes tiny test overrides but deliberately ships no guessed
    production threshold. Commit 3 enables the worker after the scale release
    gate supplies benchmark-derived values.
    """

    doc_threshold: int | None = Field(default=None, ge=1)
    chunk_threshold: int | None = Field(default=None, ge=1)


class SourceSettings(FrozenSettings):
    blob_connection_string: str | None = None
    azure_sas_expiry: int = Field(default=3600, ge=1)
    s3_presign_expiry: int = Field(default=3600, ge=1)
    s3_region: str | None = None


class VisualAssetSettings(FrozenSettings):
    thumb_max_px: int = Field(default=300, ge=1)
    thumb_cache_size: int = Field(default=256, ge=1)


class CorpusSettings(FrozenSettings):
    parser: ParserSettings = Field(default_factory=ParserSettings)
    sidecars: ParserSidecarsSettings = Field(default_factory=ParserSidecarsSettings)
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    promotion: WorkspacePromotionSettings = Field(default_factory=WorkspacePromotionSettings)
    sources: SourceSettings = Field(default_factory=SourceSettings)
    visual_assets: VisualAssetSettings = Field(default_factory=VisualAssetSettings)

    @property
    def parser_rules(self) -> str:
        return f"*:{self.sidecars.active_parser}-iteP"


class RagSettings(FrozenSettings):
    """Runtime composition referencing canonical settings without copying fields."""

    models: ModelsSettings
    corpus: CorpusSettings
    input_root: Path = Path("dlightrag_storage/inputs")
    read_only: bool = False

    @property
    def model_roles(self):
        return self.models.chat

    @property
    def embedding(self):
        return self.models.embedding

    @property
    def rerank(self):
        return self.models.rerank

    @property
    def rerank_scoring_model(self):
        return self.models.rerank.scoring_model(self.models.chat.default)

    @property
    def rag_pipeline_max_async(self):
        return self.corpus.ingestion.pipeline.max_concurrency

    @property
    def embedding_func_max_async(self):
        return self.models.embedding.max_concurrency

    @property
    def embedding_batch_num(self):
        return self.models.embedding.batch_size

    @property
    def parser_rules(self):
        return self.corpus.parser_rules

    @property
    def docling_active(self):
        return self.corpus.sidecars.docling is not None

    @property
    def docling_code_formula_preset(self):
        return (
            self.corpus.sidecars.docling.code_formula_preset
            if self.corpus.sidecars.docling
            else None
        )

    @property
    def parser_min_image_pixel(self):
        return self.corpus.sidecars.vlm.min_image_pixel

    @property
    def chunk_options(self):
        return self.corpus.parser.chunk_options

    @property
    def extraction_language(self):
        return self.corpus.extraction.language

    @property
    def entity_type_prompt_file(self):
        return self.corpus.extraction.entity_type_prompt_file

    @property
    def entity_extraction_use_json(self):
        return self.corpus.extraction.use_json

    @property
    def chunk_p_token_size(self):
        return self.corpus.ingestion.chunk_token_size

    @property
    def kg_entity_types(self):
        return self.corpus.retrieval.kg_entity_types

    @property
    def kg_chunk_pick_method(self):
        return self.corpus.retrieval.kg_chunk_pick_method

    @property
    def max_entity_tokens(self):
        return self.corpus.retrieval.max_entity_tokens

    @property
    def max_relation_tokens(self):
        return self.corpus.retrieval.max_relation_tokens

    @property
    def max_total_tokens(self):
        return self.corpus.retrieval.max_total_tokens

    @property
    def direct_visual_top_k(self):
        return self.corpus.retrieval.direct_visual_top_k

    @property
    def rrf_k(self):
        return self.corpus.retrieval.rrf_k

    @property
    def thumb_cache_size(self):
        return self.corpus.visual_assets.thumb_cache_size

    @property
    def thumb_max_px(self):
        return self.corpus.visual_assets.thumb_max_px

    @property
    def ingestion_replace_default(self):
        return self.corpus.ingestion.replace_default

    @property
    def retain_remote_source_files(self):
        return self.corpus.ingestion.retain_remote_source_files

    @property
    def url_ingest_max_bytes(self):
        return self.corpus.ingestion.url_max_bytes

    @property
    def url_ingest_private_host_allowlist(self):
        return self.corpus.ingestion.url_private_host_allowlist

    @property
    def blob_connection_string(self):
        return self.corpus.sources.blob_connection_string

    @property
    def azure_sas_expiry(self):
        return self.corpus.sources.azure_sas_expiry

    @property
    def s3_presign_expiry(self):
        return self.corpus.sources.s3_presign_expiry

    @property
    def s3_region(self):
        return self.corpus.sources.s3_region

    @property
    def max_parallel_insert(self):
        return self.corpus.ingestion.pipeline.max_parallel_insert

    @property
    def max_parallel_parse_native(self):
        return self.corpus.ingestion.pipeline.max_parallel_parse_native

    @property
    def max_parallel_parse_mineru(self):
        return self.corpus.ingestion.pipeline.max_parallel_parse_mineru

    @property
    def max_parallel_parse_docling(self):
        return self.corpus.ingestion.pipeline.max_parallel_parse_docling

    @property
    def max_parallel_analyze(self):
        return self.corpus.ingestion.pipeline.max_parallel_analyze

    @property
    def queue_size_parse(self):
        return self.corpus.ingestion.pipeline.queue_size_parse

    @property
    def queue_size_analyze(self):
        return self.corpus.ingestion.pipeline.queue_size_analyze

    @property
    def queue_size_insert(self):
        return self.corpus.ingestion.pipeline.queue_size_insert

    def lightrag_pipeline_kwargs(self) -> LightRAGPipelineKwargs:
        return self.corpus.ingestion.pipeline.lightrag_kwargs()

    def addon_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "language": self.extraction_language,
            "chunker": {"paragraph_semantic": {"chunk_token_size": self.chunk_p_token_size}},
        }
        if self.entity_type_prompt_file:
            params["entity_type_prompt_file"] = self.entity_type_prompt_file
        if self.kg_entity_types:
            params["entity_types_guidance"] = (
                f"Prioritize domain entities in these categories: {', '.join(self.kg_entity_types)}."
            )
        return params


__all__ = ["CorpusSettings", "RagSettings"]
