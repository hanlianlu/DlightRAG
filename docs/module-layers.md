# Module Layering

Code-organisation view of `src/dlightrag/`. Complements the runtime
data-flow diagram in [`architecture.drawio`](architecture.drawio): that file
shows request-time calls; this one shows import boundaries.

## Layered Structure

Modules sit on a monotonically decreasing dependency stack: a module at layer
N may import from layers 0..N, never from a higher layer. DlightRAG has one
runtime RAG path based on LightRAG main; interfaces and orchestration depend
on the shared lower layers, not on each other.

```text
L9  api, mcp, web                                  interface adapters
L8  core.servicemanager                            multi-workspace coordinator
L7  core.{service, reset, ingest_tasks}            per-workspace facade
L6  core.{answer, answer_context, answer_images, federation,
          query_images, query_planner,
          session_images, visual_assets}           query/answer/visual orchestration
    core.retrieval.{retriever, fusion,
                    lightrag_backend}
    core.ingestion.engine
L5  core.{lightrag_stores, _lightrag_patches,
          compat_guard}
    core.ingestion.{lightrag_sidecar,
                    direct_image,
                    visual_semantics,
                    parser_hygiene}
    core.retrieval.{filtered_vdb, bm25}
L4  models.{embedding, multimodal_embedding,
            llm, llm_roles, rerank}
    core.retrieval.{metadata_path}
    core.ingestion.{cleanup}
L3  models.providers                               provider implementations
    storage                                        PostgreSQL domain stores
    sourcing, citations
L2  config
    core.retrieval.{protocols, models,
                    metadata_fields}
    models.{schemas, embedding_inputs}
L1  observability                                  cross-cutting wrappers
L0  prompts, utils                                 pure helpers
```

## Why This Matters

- Lower layers do not know about higher ones. `config` does not import
  `core.service`, and storage primitives do not depend on API routes.
- LightRAG integration is centralized. Parser sidecars, KG ingestion, vector
  retrieval, direct image retrieval, BM25, and RRF are assembled by the core
  orchestration layer instead of leaking into interface code.
- PostgreSQL is the only supported core storage ecosystem. Backend-specific
  behavior lives under `storage/` and the LightRAG store adapter, not in
  request handlers or model providers.

## Per-Module Quick Reference

| Module | Layer | Role |
|---|---:|---|
| `prompts/`, `utils/` | L0 | Pure helpers and prompt text |
| `observability` | L1 | Langfuse wrappers; no-op when disabled |
| `config` | L2 | `DlightragConfig` plus env/yaml loader |
| `models.{schemas,embedding_inputs}` | L2 | Shared data types |
| `core.retrieval.{protocols,models,metadata_fields}` | L2 | Retrieval contracts and metadata registry |
| `models.providers` | L3 | OpenAI/Anthropic/Gemini and embedding provider wrappers |
| `storage` | L3 | PostgreSQL pools, metadata index, version/replication helpers |
| `sourcing` | L3 | Local, Azure Blob, and S3 source readers |
| `citations` | L3 | Inline citation validation, source projection, optional highlights, streaming normalization |
| `models.{embedding,multimodal_embedding,llm,llm_roles,rerank}` | L4 | LightRAG-compatible model callables |
| `core.retrieval.metadata_path` | L4 | Metadata-to-candidate resolver |
| `core.ingestion.cleanup` | L4 | Cascade cleanup |
| `core.lightrag_stores` | L5 | LightRAG store and sidecar adapter |
| `core.ingestion.{lightrag_sidecar,direct_image,visual_semantics,parser_hygiene}` | L5 | Parser sidecars, direct image embedding, visual KG projection, LightRAG parser hygiene |
| `core.retrieval.{filtered_vdb,bm25}` | L5 | PostgreSQL in-filtering and pg_textsearch BM25 |
| `core.ingestion.engine` | L6 | Unified ingest orchestration |
| `core.retrieval.{retriever,fusion,lightrag_backend}` | L6 | LightRAG mix retrieval, BM25 fusion, direct image path |
| `core.{answer,answer_context,answer_images,federation,query_images,query_planner,session_images,visual_assets}` | L6 | Answering, answer-context packing, image budgeting, query-image semantics, session image memory, visual asset serving, federation, filter planning |
| `core.{service,reset,ingest_tasks}` | L7 | Workspace lifecycle and public service facade |
| `core.servicemanager` | L8 | Multi-workspace pool |
| `api/`, `mcp/`, `web/` | L9 | Interface adapters |

## Verification

The layering is audited by the same checks used in CI. A violation means a
lower layer imported a higher layer or a request interface leaked into core
runtime code.

```bash
uv run lint-imports
uv run python scripts/ci/check-architecture.py
```

## Adding A Module

Decide its layer first by asking which existing layer it must import. Place it
one layer above that dependency. If a helper is shared by ingestion, retrieval,
and interfaces, keep it in a lower shared layer instead of attaching it to one
orchestration path.
