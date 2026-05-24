# LightRAG Main Sidecar Unified Architecture

**Date:** 2026-05-24  
**Status:** Design spec for holistic refactor planning
**Scope:** Replace DlightRAG's two ingestion/retrieval paths with one LightRAG-main-based multimodal path, while preserving PostgreSQL-backed metadata, in-filtering, direct image embedding, and DlightRAG-level BM25 hybrid retrieval.

---

## 1. Decision Summary

DlightRAG should move to a single unified architecture built on the current `HKUDS/LightRAG` `main` branch, not on the latest PyPI release. The verified upstream target for this design pass is:

- `HKUDS/LightRAG` `origin/main`: `a9b9079` checked on `2026-05-24`
- `HKUDS/LightRAG` `v1.5.0rc2`: `b62c260`, reviewed as the release note checkpoint for parser routing, multimodal sidecars, role LLMs, task-aware embeddings, and pipeline status semantics.
- PostgreSQL major version: `18`; current official minor checked for this design is `18.4`, released on `2026-05-14`.
- MinerU upstream checked at `1d15485` on `origin/master`; current license is `LicenseRef-MinerU-Open-Source-License`, based on Apache 2.0 with additional terms, and no longer AGPL.

The new architecture deletes the `raganything` dependency entirely. ArtRAG is used as a research input because it has already learned how to work with the latest LightRAG sidecar model, but DlightRAG must not inherit ArtRAG domain concepts, copy ArtRAG class boundaries wholesale, or keep an ArtRAG-shaped artist/artwork hierarchy.

Hard decisions:

- There is no `caption` path and no old `unified` page-render path.
- There is one ingestion engine, one retrieval engine, and one LightRAG instance per workspace.
- PostgreSQL 18 is the only supported storage ecosystem for the core product path. Development, CI, Docker, and production docs should track the current PG18 minor release; as of this design pass, that is PostgreSQL 18.4.
- LightRAG query mode is always `mix`.
- DlightRAG hybrid retrieval means `LightRAG mix + BM25 + RRF`, not LightRAG's `hybrid` query mode.
- Embedding configuration is provider-aware and multimodal. Query/document asymmetric embedding defaults to `auto`: enable it for providers with documented task routing, otherwise use LightRAG's symmetric fallback.
- Embedding provider/model/dimension/asymmetric settings are a storage contract. Changing any vector-affecting setting requires clearing vector data and rebuilding indexes.
- Text-only embedding models are invalid for this architecture.
- LLM configuration is role-based and aligned to LightRAG's current `role_llm_configs` surface: `extract`, `keyword`, `query`, and `vlm`.
- Entity/relation extraction should use LightRAG's structured JSON output support by default where the chosen LLM binding supports it. `ENTITY_TYPES` must not survive; domain guidance moves to `ENTITY_TYPE_PROMPT_FILE`.
- Parser selection uses LightRAG's parser-routing model: `LIGHTRAG_PARSER`-style rules plus filename hints, resolved through upstream `lightrag.parser_routing`. DlightRAG should not keep a separate product-level `parser.engine` / `parser.process_options` branch.
- Document deduplication uses LightRAG's canonical basename and `content_hash` fields in `doc_status` / `full_docs`. A separate DlightRAG hash index should be deleted from the runtime path.
- Document tables and equations use LightRAG's own multimodal document handling.
- Document-extracted image sidecar assets use LightRAG `i` for VLM-generated text chunks and KG participation, plus DlightRAG direct multimodal image embedding for visual recall. Native images use DlightRAG direct multimodal image embedding plus a LightRAG text/KG projection because they are not primarily document-parser inputs.
- LightRAG sidecar files are the canonical parse artifacts. DlightRAG must not write a second filesystem sidecar format; it only stores adapter-normalized references, metadata, and chunk provenance in PostgreSQL.

PostgreSQL extension requirements are layered:

| Layer | Extension | Reason |
|---|---|---|
| LightRAG vector storage | `vector` / pgvector | `PGVectorStorage` vector columns and HNSW search. |
| LightRAG graph storage | `age` / Apache AGE | `PGGraphStorage` knowledge graph storage. |
| DlightRAG BM25 hybrid | `pg_textsearch` | BM25 index/operator used by DlightRAG lexical retrieval. |

The "two plugins plus PG18" memory applies to LightRAG's core PG storage (`vector` + `age`). DlightRAG's BM25 hybrid makes `pg_textsearch` a third required extension for our target runtime. Metadata filtering remains LLM-assisted and intent-aware, but the matching layer is deterministic: exact normalized fields, date ranges, JSONB containment, and explicit filename patterns only. DlightRAG should not require or install fuzzy metadata extensions.

---

## 2. Why This Changes the Architecture

The old DlightRAG split made sense when RAGAnything owned document multimodal parsing and DlightRAG's unified mode owned page-level visual embeddings. That split is now wrong for three reasons:

1. LightRAG `main` now includes parser routing, MinerU/Docling/native parsing, sidecar outputs, multimodal analysis, and sidecar-to-chunk construction.
2. MinerU is no longer blocked by AGPL concerns for our expected usage profile.
3. Maintaining both RAGAnything and DlightRAG page-render ingestion duplicates the same lifecycle: parse, artifact management, chunk provenance, metadata filtering, deletion, and retrieval fusion.

The replacement is not "use ArtRAG". The replacement is "use the LightRAG-main sidecar model, with the clean architectural lessons ArtRAG exposed."

---

## 3. Upstream Model We Depend On

LightRAG `main` currently provides the pieces DlightRAG should treat as upstream-owned:

- `apipeline_enqueue_documents()` and `apipeline_process_enqueue_documents()` for queued document processing.
- `parse_native`, `parse_mineru`, and `parse_docling` parser routes.
- `lightrag.parser_routing` parser-rule and filename-hint resolution, including strict startup validation for invalid engines/options and missing external parser endpoints.
- `lightrag.sidecar` writer output, `sidecar_location`, `*.blocks.jsonl`, per-modality JSON files, and extracted assets.
- Chunk-level `sidecar` references for LightRAG-built multimodal chunks.
- Multimodal process options where `i` targets images/drawings, `t` targets tables, `e` targets equations, `!` skips KG, and `P` selects paragraph semantic chunking.
- Per-document `chunk_options` snapshots stored in `full_docs`, so reprocessing uses the enqueue-time chunker parameters instead of whatever the environment later becomes.
- `doc_status` / `full_docs` parser provenance, canonical filename handling, and `content_hash` deduplication.
- `analyze_multimodal()` and sidecar chunk construction for LightRAG-owned multimodal text chunks.
- `LightRAG.aquery_data()` with `mode="mix"` for structured retrieval data.

DlightRAG should depend on these surfaces through one adapter module so upstream private storage changes fail fast and locally. The adapter is read-only with respect to LightRAG sidecar files: it resolves and validates sidecar references, but does not translate them into another sidecar format. The supported LightRAG storage configuration is PostgreSQL 18 based: `PGVectorStorage`, `PGGraphStorage`, `PGKVStorage`, and `PGDocStatusStorage`.

---

## 4. Target Module Boundaries

New or retained modules should be arranged around DlightRAG concerns, not historical path names:

| Module | Responsibility |
|---|---|
| `core/service.py` | Owns workspace lifecycle, LightRAG creation, config validation, storage initialization, and public APIs. |
| `core/lightrag_stores.py` | The only module allowed to touch LightRAG internals such as `chunks_vdb`, `text_chunks`, `full_docs`, and `doc_status`. It also owns the PG-only precomputed vector writer used for direct image embeddings. |
| `core/ingestion/engine.py` | Single ingestion orchestrator for documents, native images, metadata mirrors, and deletion hooks. Parser routing and document dedup authority stay upstream in LightRAG. |
| `core/ingestion/lightrag_sidecar.py` | `LightRAGSidecarAdapter`: resolves canonical LightRAG sidecar files and yields typed references for DlightRAG indexing/direct-image embedding. It does not write DlightRAG sidecar files. |
| `core/ingestion/direct_image.py` | Creates direct image embedding chunk specs for native images and extracted drawing/image assets. |
| `core/ingestion/visual_semantics.py` | Creates VLM-generated text projections for native images and for parser backends that cannot produce LightRAG `i` analysis, then inserts them through LightRAG so image entities/relationships enter the KG. |
| `storage/document_artifacts.py` | `DocumentArtifactRegistry`: PostgreSQL table for document-level LightRAG sidecar URI/path references and parser provenance. |
| `storage/chunk_provenance.py` | `ChunkProvenanceIndex`: PostgreSQL table for chunk-to-document, chunk-to-sidecar, modality, asset, page, and bbox provenance. |
| `core/retrieval/retriever.py` | Single retrieval orchestrator: metadata filter, LightRAG mix, BM25, direct image query, RRF, rerank, enrichment. |
| `core/retrieval/bm25.py` | PostgreSQL BM25 search and score normalization. |
| `core/retrieval/filtered_vdb.py` | In-filter wrapper around LightRAG vector queries with strict empty-filter semantics. |
| `core/retrieval/fusion.py` | RRF and post-merge score handling. |
| `models/embedding_inputs.py` | Typed text, image, and multimodal embedding input records independent of provider payload shape. |
| `models/multimodal_embedding.py` | Shared text/image embedder used by LightRAG text embeddings and DlightRAG direct-image retrieval. |
| `models/providers/embed_providers.py` | Provider-specific task, payload, endpoint, response, image-support, and dimension rules. |
| `models/llm_roles.py` or `models/llm.py` | Canonical DlightRAG-to-LightRAG role mapping and role-specific model callables. |

Modules to delete or collapse after migration:

- `captionrag/*`
- all `raganything` import/config glue
- old `rag_mode` branching in service/config/API layers
- the old page-render-as-primary-ingest path in `unifiedrepresent/*`
- the custom `visual_chunks` KV concept if its only purpose is page-render output storage
- `core/ingestion/hash_index.py` and related service wiring
- `storage/json_metadata_index.py` and any storage abstraction whose only product purpose is non-PostgreSQL fallback

Reusable provider code from `unifiedrepresent` can be moved, but the old mode boundary should not survive.

---

## 5. Unified Ingestion Flow

All ingest calls pass through `UnifiedIngestionEngine`.

```mermaid
flowchart TD
    A["Source file"] --> B["Hash + system metadata"]
    B --> C{"Source kind"}
    C -->|"Document-like"| D["LightRAG parser pipeline"]
    C -->|"Native image"| E["DlightRAG native-image artifact"]
    D --> F["LightRAG chunks, KG, doc_status"]
    D --> G["LightRAG sidecar adapter"]
    G --> H["Direct image chunks for drawing/image assets"]
    E --> H
    H --> K["Native/fallback visual semantic projection docs"]
    K --> F
    F --> I["Metadata + artifact stores"]
    H --> I
    I --> J["Workspace registration"]
```

### 5.1 Source Classification

The engine classifies source files into two high-level behaviors:

- **Document-like:** PDF, Office files, spreadsheets, presentations, HTML/Markdown/text where LightRAG parsing or normal text ingest is the primary path.
- **Native image:** PNG, JPEG, WEBP, GIF, BMP, JP2, and any source whose whole semantic identity is the image itself.

LightRAG supports some image formats through parser engines, but DlightRAG should still route native images through direct image embedding. The image is the source, not a page artifact to be converted into document text first.

### 5.2 Document-Like Files

Default document parser rules should use LightRAG's own routing syntax and be resolved with upstream `lightrag.parser_routing`, not with a DlightRAG parser switch:

```text
*:native-iteP,*:mineru-iteP,*:legacy-R
```

Meaning:

- `native-iteP`: prefer native structured parsing where LightRAG supports it, currently strongest for DOCX, with image/table/equation VLM analysis enabled.
- `mineru-iteP`: use MinerU for parser-supported document formats that native does not cover, with image/table/equation VLM analysis enabled.
- `legacy-R`: retain LightRAG's legacy text extractor and recursive chunking fallback for formats not handled by native/MinerU.
- `i`: let LightRAG analyze document-extracted images/drawings into text chunks for KG, BM25, and citations.
- `t`: let LightRAG handle tables.
- `e`: let LightRAG handle equations.
- `P`: use paragraph semantic chunking by default.
- no `!`: KG extraction remains enabled.

This default can be configurable as parser rules, but the default should express the product decision: document-extracted tables/equations/images get LightRAG semantic text/KG treatment, while drawings/images also receive DlightRAG direct image vectors for visual recall. File-level overrides should use LightRAG filename hints such as `report.[mineru-R].pdf` or `memo.[-!].docx`.

The direct image collector must read parser-emitted drawing/image assets, not LightRAG image-analysis chunks. In LightRAG v1.5.0rc2, `i/t/e` gates VLM analysis, not parser extraction or embedding: sidecar files are written by the parser when content exists. Therefore DlightRAG should enable `i` for document-like ingestion so LightRAG creates image semantic text chunks, and still read `drawings.json` to direct-embed the original image assets.

This means document-extracted images have two intentional representations:

1. LightRAG `i` uses the `vlm` role to analyze drawings/images and produces text chunks that participate in BM25, entity extraction, relationship extraction, and citations.
2. DlightRAG direct image embedding indexes the original image bytes in the same multimodal vector space for visual similarity search.
3. `chunk_provenance` links both representations back to the same sidecar item.

The direct image vector remains the visual identity used for image similarity search. LightRAG `i` output is the KG/BM25/citation bridge; it must not replace the image vector.

LightRAG's current public vector-store flow is text-first: `PGVectorStorage.upsert()` reads row `content` and calls `embedding_func(batch, context="document")`; it does not expose a stable public upsert API for a caller-supplied per-row image vector. Querying is more flexible: `PGVectorStorage.query()` accepts a precomputed `query_embedding`. Therefore DlightRAG must own direct image indexing through the `LightRAGStores` adapter:

1. Compute image vectors through `MultimodalEmbedder.embed_index_images()` from image bytes/PIL images, not from VLM captions.
2. Upsert a companion display/citation text row into LightRAG `text_chunks`.
3. Upsert the same chunk id into LightRAG's PostgreSQL chunk vector table with `content_vector` set to the precomputed image vector.
4. Never call LightRAG's normal `chunks_vdb.upsert()` for direct image vectors, because that would embed the text `content` instead of the image.
5. For image queries, compute `embed_query_images()` and pass the vector through LightRAG's `chunks_vdb.query(query_embedding=...)`.

Document ingest steps:

1. Write a pending metadata skeleton keyed by the intended source document id; do not treat DlightRAG as the document dedup authority.
2. Resolve parser directives with LightRAG parser-rule syntax plus filename hints, then enqueue/process through LightRAG with `docs_format=FULL_DOCS_FORMAT_PENDING_PARSE`, the resolved `parse_engine`, and resolved `process_options`.
3. Let LightRAG write its normal document chunks, KG records, and `doc_status`.
4. Read parser sidecar locations, `parse_engine`, `process_options`, `chunk_options`, `content_hash`, and canonical basename from LightRAG `doc_status` / `full_docs`.
5. Persist a document artifact registry row that records source path, canonical basename, parse engine, process options, chunk options snapshot, content hash, LightRAG sidecar URI/directory, blocks path, drawings path, tables path, equations path, and LightRAG full document id.
6. Use `LightRAGSidecarAdapter` to resolve LightRAG sidecar items into typed references.
7. For drawing/image sidecar references only, write direct image embedding chunk rows through the LightRAG store adapter. LightRAG `i` has already produced the corresponding semantic text chunks when image analysis succeeds.
8. Persist chunk provenance for LightRAG-owned document chunks, LightRAG-owned multimodal image/table/equation text chunks, and DlightRAG-owned direct image chunks.
9. For parser backends or retry cases where LightRAG `i` analysis is unavailable, create DlightRAG visual semantic projection documents as a fallback so image-derived entities and relationships still enter the KG.
10. Finalize metadata only after LightRAG writes and DlightRAG registry/provenance/direct-image/visual-semantic writes succeed.

If upstream LightRAG later exposes stable sidecar ids on its own chunks, DlightRAG should update direct image chunks in place when possible. If that mapping is not stable, DlightRAG-owned direct image chunk ids are the canonical fallback:

```text
{workspace}:{full_doc_id}:sidecar:{sidecar_type}:{sidecar_id}
```

This keeps deletion, in-filtering, and retrieval enrichment deterministic.

### 5.3 Native Images

Native image ingest does not use LightRAG document multimodal analysis. It creates a native-image artifact/provenance record, not a DlightRAG sidecar file:

```json
{
  "source_kind": "native_image",
  "source_path": "...",
  "asset_path": "...",
  "page": null,
  "bbox": null,
  "mime_type": "image/png"
}
```

The direct image path writes:

- a multimodal image vector into LightRAG's chunk vector store;
- a direct-image text chunk containing either user-provided description, configured VLM caption, or a concise file-derived fallback;
- a visual semantic projection document inserted through LightRAG with KG extraction enabled, using the `vlm` role for image understanding and the `extract` role for entity/relation extraction;
- a `doc_status`/`full_docs` record so deletion and retrieval provenance remain uniform;
- `document_artifacts` and `chunk_provenance` rows.

The vector identity is always the image itself. Captions and visual semantic projections are retrieval support material for BM25, KG, display, and citation context; they are not a substitute for image embedding.

---

## 6. LightRAG Sidecar + Metadata Model

DlightRAG keeps metadata management as a first-class feature. LightRAG sidecars remain upstream-owned parse artifacts; DlightRAG extends them with PostgreSQL artifact references and chunk provenance instead of replacing document metadata or duplicating sidecar files.

The boundary is strict:

- LightRAG writes and owns `*.parsed/`, `*.blocks.jsonl`, modality JSON files, assets, and sidecar refs on LightRAG chunks.
- DlightRAG reads those artifacts through `LightRAGSidecarAdapter`.
- DlightRAG stores only relational references, normalized metadata, and direct-image chunk provenance.
- DlightRAG may create direct-image chunk records for native images and extracted assets, but those records point back to native source files or LightRAG sidecar items.

### 6.1 Document Metadata

Existing `PGMetadataIndex` remains the document-level filter source. It continues to store:

- system metadata such as filename, canonical basename, extension, size, title, author, dates, and LightRAG content hash;
- user metadata;
- parser metadata such as `parse_engine`, `process_options`, `artifact_status`, and source kind.

The LLM may infer structured filters from natural language, but storage executes them deterministically. String filters use normalized case-insensitive exact matching backed by `LOWER(field)` btree expression indexes; partial file references must be represented explicitly as filename patterns.

DlightRAG should expose a user metadata contract, but it should not delegate that contract to LightRAG:

- Ingest APIs accept `metadata: Mapping[str, Any]` plus optional typed system fields such as `title`, `author`, and dates.
- Config may declare metadata fields with `type`, `normalizer`, `filter_ops`, and `indexed` settings. Supported matching operators are exact, `IN`, range, JSONB containment, and explicit pattern for selected string fields.
- Unknown user metadata may be stored in JSONB for enrichment. It is not filterable through the normal exact/range/pattern path until declared in the field registry.
- LLM intent-aware detection may mention an unknown metadata key, but it cannot make that key filterable by itself. The planner must either ignore it as an untrusted filter with a traceable warning, ask the caller to declare the field, or use an explicitly declared `metadata_json` field whose `filter_ops` includes `contains`.
- Reserved namespaces are immutable: `sys.*` for DlightRAG system metadata, `lightrag.*` for mirrored LightRAG provenance, and `user.*` for caller-provided metadata.
- Metadata updates affect filtering and enrichment immediately, but do not mutate LightRAG chunks, KG, or vectors. If a metadata field should become semantic content, the caller must request re-ingest or explicit KG projection.

The LightRAG bridge is by identifier and provenance, not by making LightRAG own user metadata. DlightRAG reads LightRAG `doc_status` / `full_docs` operational fields such as `sidecar_location`, `parse_engine`, `process_options`, `chunk_options`, `content_hash`, and `chunks_list`, then mirrors them into PostgreSQL. Retrieval resolves user metadata filters to document ids, maps them to chunk ids through `ChunkProvenanceIndex`, and applies those candidate ids to LightRAG vector search, BM25, and direct visual search. DlightRAG should not write arbitrary user metadata into LightRAG `doc_status.metadata` or depend on it for filtering, because that field is used by LightRAG pipeline lifecycle and is not a stable filter API.

The old `rag_mode` metadata should be removed. If a field is needed for observability, use `ingest_strategy="lightrag_sidecar_unified"`.

A protocol may remain for test doubles, but not as a product promise for non-PostgreSQL metadata backends.

### 6.2 Document Artifacts

`DocumentArtifactRegistry` stores LightRAG sidecar references and parser provenance:

```sql
CREATE TABLE dlightrag_document_artifacts (
    workspace TEXT NOT NULL,
    full_doc_id TEXT NOT NULL,
    source_uri TEXT,
    local_source_path TEXT,
    canonical_basename TEXT,
    source_kind TEXT NOT NULL,
    parse_format TEXT,
    parse_engine TEXT,
    process_options TEXT,
    chunk_options JSONB DEFAULT '{}',
    content_hash TEXT,
    sidecar_location TEXT,
    artifact_dir TEXT,
    lightrag_document_path TEXT,
    blocks_path TEXT,
    drawings_path TEXT,
    tables_path TEXT,
    equations_path TEXT,
    duplicate_kind TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (workspace, full_doc_id)
);
```

There is no file-backed alternative target for this table. Development and production both use PostgreSQL 18 so metadata, chunk provenance, vector search, KG storage, and BM25 share one consistent transactional substrate.

### 6.3 Chunk Provenance

`ChunkProvenanceIndex` is the bridge from document filters to retrieval filters:

```sql
CREATE TABLE dlightrag_chunk_provenance (
    workspace TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    full_doc_id TEXT NOT NULL,
    embedding_input_kind TEXT NOT NULL, -- text | image | multimodal
    sidecar_type TEXT,                  -- block | drawing | table | equation | native_image
    sidecar_id TEXT,
    asset_path TEXT,
    page_number INTEGER,
    bbox JSONB,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (workspace, chunk_id)
);
```

Rules:

- LightRAG-owned blocks/tables/equations are indexed as `embedding_input_kind="text"` unless upstream provides a true multimodal vector for that chunk.
- DlightRAG-owned native images and drawing/image sidecar assets are indexed as `embedding_input_kind="image"`.
- LightRAG `i` image-analysis chunks and DlightRAG fallback visual semantic projection chunks are indexed as `embedding_input_kind="text"` and carry `semantic_projection_of_chunk_id` in provenance metadata when they correspond to a direct-image chunk.
- A chunk may carry sidecar provenance even when the visible retrieval text came from LightRAG.
- Metadata filters resolve document ids first, then chunk ids through this table.

---

## 7. Provider-Aware Multimodal Embedding

Startup validation must reject configurations where the provider/model pair cannot embed both text and images into one shared vector space. This must be enforced by provider metadata plus a startup probe, not by a loose user-supplied boolean.

### 7.1 Normalized Inputs and Context

DlightRAG needs a provider-neutral input layer:

```python
TextEmbeddingInput(text: str)
ImageEmbeddingInput(data_uri: str | None = None, path: str | None = None, url: str | None = None)
MultimodalEmbeddingInput(parts: list[TextEmbeddingInput | ImageEmbeddingInput])
```

The public provider protocol should accept context:

```python
class EmbedProvider(Protocol):
    endpoint: str
    supports_images: bool
    supports_asymmetric: bool  # provider capability, not the active runtime mode
    default_dim: int | None
    known_dims: frozenset[int] | None

    def build_payload(
        self,
        model: str,
        inputs: list[EmbeddingInput],
        *,
        context: Literal["query", "document"],
        asymmetric: bool = False,
        output_dimension: int | None = None,
    ) -> dict: ...
```

Context propagation is mandatory, but asymmetric behavior is not:

- LightRAG document indexing calls text embedding with `context="document"`.
- DlightRAG direct-image indexing calls image embedding with `context="document"`.
- Text queries call embedding with `context="query"`.
- Image queries call image embedding with `context="query"`.
- LightRAG receives `EmbeddingFunc(..., supports_asymmetric=True)` when DlightRAG's asymmetric mode resolves active: `auto` plus a capable provider, or `require` plus a capable provider.
- Otherwise DlightRAG relies on LightRAG's built-in symmetric fallback: `EmbeddingFunc.__call__()` strips the `context` parameter when `supports_asymmetric=False`, so the same provider payload shape is used for indexed content and queries.

This context is an embedding-generation hint, not a vector-store query flag. The stored vectors are already produced with document/index semantics; query-time retrieval first produces a query vector with query semantics, then the vector store compares vectors normally. DlightRAG APIs should therefore avoid ambiguous names such as `embed_pages()` and instead expose separate indexing/query call sites such as `embed_index_images(..., context="document")` and `embed_query_images(..., context="query")`.

### 7.2 Provider Matrix

| Provider | Target models | Context mapping | Image payload | Dimension policy |
|---|---|---|---|---|
| `voyage` | `voyage-multimodal-3.5` | `auto`/`require` -> `query`/`document` `input_type`; `disable` -> omit task hint | `/multimodalembeddings` `image_base64` content | default `1024`; validate returned dim. |
| `dashscope_qwen` | `qwen3-vl-embedding` cloud API | `auto` falls back to symmetric until provider docs confirm task routing; `require` needs an explicit adapter | DashScope multimodal `contents` with text/image items | cloud default `2560`; allow configured dimensions such as `2048`, `1536`, `1024`, `768`, `512`, `256`. |
| `qwen_openai_compatible` | LM Studio/vLLM-style `qwen3-vl-embedding-2b` and `qwen3-vl-embedding-8b` | no asymmetric mapping by default; use provider-specific instruction or explicit task mode only when the server documents it | OpenAI-compatible `input` using text and image data URIs | validate returned dim; expected local model dims include `2048` for 2B and `4096` for 8B unless the serving stack documents projection. |
| `gemini` | `gemini-embedding-2` | `auto`/`require` maps query/document to native retrieval task behavior; `disable` stays symmetric | native Google multimodal embedding request | default `3072`; common configured dims `1536` or `768`. |
| `jina` | Jina multimodal embeddings v4 | `auto`/`require`: `query` -> `retrieval.query`, `document` -> `retrieval.passage`; `disable` stays symmetric | Jina text/image input items | default dense `2048`; validate returned dim. |
| `openai_compatible` | explicit local or proxy provider | only asymmetric if configured/probed as task-aware | data URI passthrough if the server accepts images | no hard default; startup probe and returned-dim validation are required. |
| `ollama` | local embedding servers | text-only unless a specific server proves image support | provider-specific | not valid for this architecture unless image embedding probe passes. |

Provider detection may use model-name heuristics as a convenience, but production config should prefer an explicit `embedding.provider`. The matching layer must never infer multimodal safety only from a string containing `vl` or `multimodal`.

### 7.3 Startup Validation

Validation policy:

- `embedding.provider`, `embedding.model`, and `embedding.dim` are required for production config.
- The provider must report `supports_images=True` or pass an image embedding probe.
- Known text-only models fail startup before LightRAG initialization.
- The returned vector length must equal `embedding.dim`.
- `embedding.asymmetric` is a three-state mode: `auto`, `require`, or `disable`.
- `auto` is the default and quality-first path: activate asymmetric routing for provider adapters with documented task parameters; otherwise use LightRAG's symmetric fallback.
- `require` activates asymmetric routing and fails startup if the selected provider cannot honor it.
- `disable` forces symmetric behavior even for capable providers.
- Active `supports_asymmetric=True` is allowed only when the resolved mode is active and the provider has native task parameters or an explicit configured query/document instruction strategy. It must not be inferred from OpenAI compatibility alone.
- If asymmetric embedding is active, both query and document probes should be exercised where provider cost allows; unit tests can mock this.
- A provider/model/dimension/asymmetric-setting change after indexing must fail fast unless the workspace is explicitly reset or vector indexes are rebuilt.

Symmetric fallback only applies to query/document task routing. Text-only embedding fallback is still not supported because it would silently break native images and parser-extracted drawing/image assets.

---

## 8. Role-Based LLM Configuration

DlightRAG should clean up its current model configuration surface and align with LightRAG `main` role-based LLM configuration. The canonical shape is:

```yaml
llm:
  default:
    provider: openai
    model: gpt-4.1
    temperature: 0.5
  roles:
    extract:
      model: gpt-4.1-mini
    keyword:
      model: gpt-4.1-mini
    query:
      model: gpt-4.1
    vlm:
      provider: gemini
      model: gemini-2.5-flash
```

Role responsibilities:

| Role | Owner | Uses |
|---|---|---|
| `extract` | LightRAG + DlightRAG | entity extraction, document metadata normalization, structured metadata extraction. |
| `keyword` | LightRAG + DlightRAG | LightRAG keyword extraction and DlightRAG intent-aware metadata filter detection. |
| `query` | LightRAG + DlightRAG | answer generation, query planning, citation-aware responses. |
| `vlm` | LightRAG + DlightRAG | visual/multimodal analysis, parser/VLM calls, image descriptions, and structured visual semantic projections for KG insertion. |

Cleanup rules:

- Remove product-level `ingest` as a separate LLM role; ingest-time calls map to `extract` or `vlm`.
- Rename plural `keywords` to singular `keyword` to match LightRAG.
- Pass `extract`, `keyword`, `query`, and `vlm` through LightRAG `role_llm_configs` when configured.
- Keep `embedding` and `rerank` separate from LLM roles. The `vlm` role is not a reranker.
- DlightRAG local metadata filter intent detection should use the `keyword` role, while deterministic metadata matching still happens in PostgreSQL.
- Runtime role updates can later call LightRAG `aupdate_llm_role_config()`, but the first refactor milestone only needs clean init-time configuration.

---

## 9. Retrieval Architecture

Retrieval has one orchestrator and three candidate-producing signals:

1. LightRAG semantic/KG retrieval using `QueryParam(mode="mix")`.
2. DlightRAG BM25 lexical retrieval over chunk text.
3. Direct visual retrieval when query images are present.

```mermaid
flowchart TD
    A["Query text + optional images + metadata filters"] --> B["Metadata filter to candidate chunk ids"]
    B --> C{"Candidate set empty?"}
    C -->|"yes"| Z["Return empty result"]
    C -->|"no / no filter"| D["Install chunks_vdb in-filter"]
    D --> E["LightRAG aquery_data(mode=mix)"]
    D --> F["BM25 chunk search"]
    D --> G{"Query has images?"}
    G -->|"yes"| H["Direct image embedding search"]
    G -->|"no"| I["Skip visual path"]
    E --> J["RRF fusion"]
    F --> J
    H --> J
    J --> K["Optional rerank"]
    K --> L["Sidecar + metadata enrichment"]
```

### 8.1 Strict In-Filtering

Metadata filters must never degrade to unfiltered retrieval by accident.

Semantics:

- `candidate_ids is None`: no metadata filter is active.
- `candidate_ids == set()`: a metadata filter is active and matched nothing; return empty immediately.
- non-empty `candidate_ids`: all vector and BM25 paths must filter to those ids.

This fixes the old ambiguity where an empty candidate set could behave like no filter.

### 8.2 LightRAG Path

The LightRAG path always calls `aquery_data()` with `mode="mix"`. Public DlightRAG APIs may accept retrieval settings, but they must not expose LightRAG query mode as a user-facing downgrade path.

The filtered vector wrapper applies candidate chunk ids inside `chunks_vdb.query()` so LightRAG's own mix retrieval only sees permitted chunks.

### 8.3 BM25 Path

BM25 is a DlightRAG retrieval signal, not a LightRAG query mode.

Implementation target:

- Use a PostgreSQL BM25-capable index over LightRAG document chunks, scoped by `workspace`.
- Prefer `pg_textsearch`/ParadeDB-style BM25 when available.
- If the BM25 extension is unavailable, fail startup when BM25 is enabled rather than silently degrading to a different backend.
- All BM25 DDL and extension checks target PostgreSQL 18.4 semantics first.

Fusion uses reciprocal rank fusion:

```text
score = sum(1 / (rrf_k + rank + 1))
```

Default `rrf_k` should be configurable, with `60` as a standard starting point.

BM25 must honor metadata filters by applying `candidate_ids` before ranking.

### 8.4 Direct Visual Path

When query images are supplied:

1. Embed each query image with the same multimodal embedding provider.
2. Query LightRAG's chunk vector store with the precomputed image vector.
3. Apply the same candidate id filter.
4. Prefer direct-image chunks but do not exclude LightRAG multimodal chunks if they share the same embedding space.
5. Feed results into the same RRF/rerank/enrichment pipeline.

When no query images are supplied, this path is skipped.

### 8.5 Result Enrichment

Every returned chunk should be enriched from:

- `chunk_provenance`: sidecar type, asset path, page number, bounding box, embedding input kind;
- document metadata: filename, user metadata, source fields;
- `document_artifacts`: sidecar directory and original parser provenance;
- LightRAG result data: entity/relation/chunk context from `aquery_data()`.

This preserves existing DlightRAG metadata UX while adding sidecar-specific citations.

---

## 10. Config Changes

Remove:

- `rag_mode`
- all `raganything` configuration
- caption-mode parser settings that only existed for RAGAnything
- old page rendering settings that only existed to create page images for unified mode
- user-selectable non-PostgreSQL LightRAG storage backends and their optional config blocks
- legacy top-level `chat`, `ingest`, `extract`, `keywords`, `query`, and `vlm` fields

Keep or add:

```yaml
storage:
  postgres_major: 18
  postgres_min_minor: "18.4"
  vector: PGVectorStorage
  graph: PGGraphStorage
  kv: PGKVStorage
  doc_status: PGDocStatusStorage

parser:
  rules: "*:native-iteP,*:mineru-iteP,*:legacy-R"
  native_images_bypass_lightrag: true
  chunk_options: {}   # optional per-workspace overrides passed to LightRAG enqueue; normally empty

extraction:
  use_json: true
  entity_type_prompt_file: null  # optional LightRAG profile file; replaces ENTITY_TYPES

llm:
  default:
    provider: openai
    model: gpt-4.1
    temperature: 0.5
  roles:
    extract:
      model: gpt-4.1-mini
    keyword:
      model: gpt-4.1-mini
    query:
      model: gpt-4.1
    vlm:
      provider: gemini
      model: gemini-2.5-flash

embedding:
  provider: voyage
  model: voyage-multimodal-3.5
  dim: 1024
  asymmetric: auto   # enable provider task routing when supported; otherwise LightRAG fallback
  startup_probe: true
  model_kwargs: {}

# Other valid embedding providers include dashscope_qwen,
# qwen_openai_compatible, gemini, jina, openai_compatible, and ollama
# only when image support is explicitly proven.
embedding_local_qwen_example:
  provider: qwen_openai_compatible
  model: qwen3-vl-embedding-2b
  base_url: http://host.docker.internal:1234/v1
  dim: 2048
  asymmetric: auto   # currently resolves to symmetric unless the local server documents a task adapter

retrieval:
  lightrag_mode: mix   # internal invariant; not a public mode selector
  bm25:
    enabled: true
    top_k: 40
    rrf_k: 60
  direct_visual:
    enabled: true
    top_k: 20

metadata:
  enabled: true
  allow_ad_hoc_json: true
  unknown_filter_policy: ignore_with_warning  # reject | ignore_with_warning | allow_declared_json_contains
  fields:
    title:
      type: string
      normalizer: casefold_trim
      filter_ops: [exact, pattern]
      indexed: true
    author:
      type: string
      normalizer: casefold_trim
      filter_ops: [exact]
      indexed: true
    published_at:
      type: date
      filter_ops: [range]
      indexed: true
    tags:
      type: string_array
      normalizer: casefold_trim
      filter_ops: [contains]
      indexed: true
    metadata_json:
      type: json
      filter_ops: [contains]
      indexed: false   # explicit opt-in; enable only if broad JSONB containment is acceptable
```

The storage values are operational facts, not product-mode switches. Config validation should reject PostgreSQL servers older than major version 18, and should reject `Neo4JStorage`, `MilvusVectorDBStorage`, `QdrantVectorDBStorage`, `JsonKVStorage`, `NetworkXStorage`, and other non-PostgreSQL storage choices in the core path.

Dependency policy:

```toml
"lightrag-hku @ git+https://github.com/HKUDS/LightRAG.git@main"
```

The exact dependency syntax can follow the package manager in use, but it must target GitHub `main`, not a release specifier such as `>=1.5.0rc1`. `raganything` must be removed from runtime and optional dependencies.

---

## 11. Deletion and Re-Ingest Semantics

Deletion must cascade through all layers:

1. metadata index row;
2. document artifact row;
3. chunk provenance rows;
4. DlightRAG-owned direct image chunks in LightRAG stores;
5. LightRAG-owned document chunks, entities, relationships, full docs, doc status, and parser artifact directory cleanup through the LightRAG lifecycle;
6. DlightRAG-owned native-image temp/copy artifacts, if any.

Re-ingest with `replace=True` should delete the old full document first, then insert fresh LightRAG and DlightRAG records. This avoids stale sidecar ids and stale candidate filters.

Existing stored data from the old `caption` and `unified` paths should be treated as requiring re-ingestion unless a later migration tool is explicitly requested.

---

## 12. ArtRAG Lessons Adopted

Adopted:

- A narrow LightRAG adapter is required because LightRAG storage internals are not a stable public API.
- Sidecar provenance must be explicit and queryable.
- Metadata filtering must resolve to candidate chunk ids before retrieval.
- Empty filters must return empty results, not unfiltered results.
- BM25 and semantic retrieval should be fused with RRF.
- Direct image embedding needs the same vector space as text query embedding.
- Tests should use golden sidecar fixtures rather than depending on full parser execution for every unit test.

Not adopted:

- Artist/artwork/document hierarchy.
- ArtRAG chunk mapping semantics.
- ArtRAG evaluation gateway.
- ArtRAG preference scoring.
- ArtRAG service names, class names, or domain-specific tables.

The point is to reuse architectural lessons, not to make DlightRAG a genericized ArtRAG.

---

## 13. Test Strategy

Required test groups:

1. **Dependency cleanup**
   - `raganything` is absent from dependencies and imports.
   - LightRAG dependency points to GitHub `main`.

2. **Config validation**
   - text-only embedding config fails startup.
   - multimodal embedding config passes startup.
   - provider/model/dimension mismatch fails startup.
   - LightRAG embedding wrapper declares `supports_asymmetric=True` by default for providers where `embedding.asymmetric=auto` resolves to a documented task-aware route.
   - LightRAG embedding wrapper declares `supports_asymmetric=False` when `auto` falls back for an unsupported provider or when `embedding.asymmetric=disable`.
   - `embedding.asymmetric=require` fails startup for providers without a task-aware route.
   - LightRAG query mode cannot be changed away from `mix`.
   - PostgreSQL server version below 18 fails startup.
   - non-PostgreSQL LightRAG storage choices fail startup.
   - `llm.roles.extract`, `llm.roles.keyword`, `llm.roles.query`, and `llm.roles.vlm` map to LightRAG `role_llm_configs`.
   - legacy `ingest` and plural `keywords` fields are rejected.

3. **Embedding provider matrix**
   - Voyage maps context to `input_type` in `auto`/`require`, and omits `input_type` in `disable`.
   - Qwen/DashScope passes image content and configured dimension.
   - Qwen OpenAI-compatible validates returned dimensions for local models.
   - Gemini passes output dimension and uses retrieval task behavior when asymmetric resolves active.
   - Jina maps context to `retrieval.query` / `retrieval.passage` when asymmetric resolves active.
   - Direct image indexing uses document context and image query uses query context at DlightRAG call sites; provider payloads use task routing only when asymmetric resolves active.

4. **Document ingest golden fixture**
   - LightRAG parser directives are resolved from `*:native-iteP,*:mineru-iteP,*:legacy-R` plus filename hints.
   - `chunk_options` is snapshotted and persisted with document artifacts.
   - document dedup uses LightRAG canonical basename and `content_hash`, not a DlightRAG hash index.
   - table/equation sidecars become LightRAG-owned text chunks.
   - drawing/image sidecars become LightRAG-owned image-analysis text chunks through `i` and DlightRAG-owned direct image chunks through direct embedding.
   - `document_artifacts` and `chunk_provenance` are written.

5. **Native image ingest**
   - native image bypasses LightRAG document multimodal analysis.
   - image bytes are embedded directly.
   - VLM-generated visual semantic projection enters LightRAG KG extraction.
   - uniform metadata, artifact, doc status, and chunk provenance are written.

6. **Metadata in-filtering**
   - no filter means no restriction.
   - matching filter restricts all retrieval paths to candidate ids.
   - empty filter returns empty results before LightRAG/BM25/vector calls.

7. **BM25 hybrid retrieval**
   - BM25 path ranks lexical matches.
   - BM25 applies workspace and candidate filters.
   - RRF merges BM25, LightRAG mix, and direct visual results deterministically.

8. **Deletion lifecycle**
   - delete removes LightRAG records, DlightRAG direct image chunks, artifact rows, chunk provenance rows, and metadata rows.

9. **Compatibility guard**
   - missing LightRAG parser/storage surfaces fail with a clear startup error.

---

## 14. Implementation Order After Review

After this design is reviewed, the implementation plan should be split into small vertical slices:

1. Dependency/PostgreSQL-only config cleanup and startup validation.
2. Role-based LLM config cleanup aligned to LightRAG `role_llm_configs`.
3. Provider-aware multimodal embedding contract, task context mapping, and dimension probes.
4. LightRAG store adapter and compatibility guard.
5. Artifact and chunk provenance storage.
6. Unified document ingest using LightRAG parser sidecars.
7. Native image and extracted-image direct embedding.
8. Strict in-filter retrieval.
9. BM25 + RRF hybrid retrieval.
10. Deletion lifecycle and old module removal.

The first implementation milestone should prove one document fixture containing text, table, equation, and extracted image sidecars plus one native image fixture. That is the smallest testable slice that exercises the new architecture instead of only renaming the old paths.
