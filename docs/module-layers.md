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
L6  core.{answer, federation, query_planner}       query/answer orchestration
    core.retrieval.{retriever, fusion,
                    lightrag_backend}
    core.ingestion.engine
L5  core.{lightrag_stores, _lightrag_patches,
          compat_guard}
    core.ingestion.{lightrag_sidecar,
                    direct_image,
                    visual_semantics}
    core.retrieval.{filtered_vdb, bm25}
L4  models.{embedding, multimodal_embedding,
            llm, llm_roles, rerank}
    core.retrieval.{metadata_path}
    core.ingestion.{cleanup, policy}
    core.vlm_ocr
L3  models.providers                               provider implementations
    storage                                        PostgreSQL domain stores
    sourcing, converters, citations
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
| `storage` | L3 | PostgreSQL pools, metadata index, artifact/provenance stores |
| `sourcing` | L3 | Local, Azure Blob, and S3 source readers |
| `converters` | L3 | Local document conversion helpers |
| `citations` | L3 | Inline citation validation, source projection, optional highlights, streaming normalization |
| `models.{embedding,multimodal_embedding,llm,llm_roles,rerank}` | L4 | LightRAG-compatible model callables |
| `core.retrieval.metadata_path` | L4 | Metadata-to-candidate resolver |
| `core.ingestion.{cleanup,policy}` | L4 | Cascade cleanup and ingest policy |
| `core.vlm_ocr` | L4 | VLM OCR block helpers |
| `core.lightrag_stores` | L5 | LightRAG store and sidecar adapter |
| `core.ingestion.{lightrag_sidecar,direct_image,visual_semantics}` | L5 | Parser sidecars, direct image embedding, visual KG projection |
| `core.retrieval.{filtered_vdb,bm25}` | L5 | PostgreSQL in-filtering and pg_textsearch BM25 |
| `core.ingestion.engine` | L6 | Unified ingest orchestration |
| `core.retrieval.{retriever,fusion,lightrag_backend}` | L6 | LightRAG mix retrieval, BM25 fusion, direct image path |
| `core.{answer,federation,query_planner}` | L6 | Answering, federation, filter planning |
| `core.{service,reset,ingest_tasks}` | L7 | Workspace lifecycle and public service facade |
| `core.servicemanager` | L8 | Multi-workspace pool |
| `api/`, `mcp/`, `web/` | L9 | Interface adapters |

## Verification

The layering is intended to be audited with an AST import check before large
releases. A violation means a lower layer imported a higher layer or a request
interface leaked into core runtime code.

```bash
uv run python - <<'PY'
import ast
from collections import defaultdict
from pathlib import Path

src_root = Path("src/dlightrag")
graph: dict[str, set[str]] = defaultdict(set)
for py in src_root.rglob("*.py"):
    if "__pycache__" in py.parts:
        continue
    rel = py.relative_to(src_root.parent.parent / "src").with_suffix("")
    mod = ".".join(rel.parts).removesuffix(".__init__")
    tree = ast.parse(py.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("dlightrag"):
            graph[mod].add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("dlightrag"):
                    graph[mod].add(alias.name)

print(f"audited {len(graph)} modules")
PY
```

## Adding A Module

Decide its layer first by asking which existing layer it must import. Place it
one layer above that dependency. If a helper is shared by ingestion, retrieval,
and interfaces, keep it in a lower shared layer instead of attaching it to one
orchestration path.
