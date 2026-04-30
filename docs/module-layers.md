# Module Layering

Code-organisation view of `src/dlightrag/`. Complements the runtime
data-flow diagram in [`architecture.drawio`](architecture.drawio) — that
file shows what calls what at request time; this one shows what *imports*
what at module load.

## Layered structure

Modules sit on a single monotonically-decreasing dependency stack: a
module at layer **N** may import from layers **0..N**, never from
**N+1..9**. Cross-mode imports between `captionrag/` and
`unifiedrepresent/` are forbidden — both pull from the shared layers
below.

```
L9  api, mcp, web                                    ← interfaces
L8  core.servicemanager                              ← multi-workspace coordinator
L7  core.{service, reset, ingest_tasks}              ← per-workspace facade
L6  captionrag.pipeline,                             ← per-mode top engines
    unifiedrepresent.{engine, lifecycle},
    core.{federation, answer, query_planner}
L5  captionrag.{vlm_parser, retrieval},              ← mode-specific stages
    unifiedrepresent.{extractor, retriever,
                      multimodal_query},
    core._lightrag_patches, core.compat_guard
L4  models.{embedding, llm, rerank},                 ← model factories +
    core.retrieval.{filtered_vdb,                      retrieval/ingestion impl
                    metadata_path},
    core.ingestion.{hash_index, cleanup,
                    page_metadata},
    captionrag.chunking,
    unifiedrepresent.embedder,
    core.vlm_ocr
L3  models.providers,                                ← backend impls + side-effect
    storage, sourcing, converters, citations,         layers
    core.retrieval.path_resolver,
    core.ingestion.{policy, docling_postprocess,
                    metadata_extract}
L2  config,                                          ← settings + pure data types
    core.retrieval.{protocols, models,
                    metadata_fields}
L1  observability,                                   ← cross-cutting, schema types
    models.{schemas, prompts}
L0  utils, prompts                                   ← pure helpers, no internal deps
```

## Why this matters

- **Lower layers don't know about higher ones.** `config` (L2) cannot
  import `core.service` (L7). A breaking change in `service` cannot ripple
  down to settings parsing.
- **Two RAG modes are independent.** Caption and unified each consume
  shared infrastructure (`models/`, `storage/`, `core.retrieval/`,
  `converters/`) without seeing each other. You can delete one mode
  without touching the other.
- **The interface layer (L9) is leaf.** Nothing else imports from `api/`,
  `mcp/`, or `web/` — they're consumers of the engine, not part of it.
  Replacing or adding interfaces is a localised change.

## Per-module quick reference

| Module | Layer | Role |
|---|---|---|
| `prompts/`, `utils/` | L0 | Pure helpers, no internal deps |
| `observability` | L1 | Langfuse wrappers; no-op when disabled |
| `models/{schemas,prompts}` | L1 | Pydantic types, prompt strings |
| `config` | L2 | `DlightragConfig` + env/yaml loader |
| `core.retrieval.{protocols,models,metadata_fields}` | L2 | Retrieval data types + metadata schema registry |
| `models.providers/` | L3 | Native SDK wrappers (OpenAI/Anthropic/Gemini) |
| `storage/` | L3 | PG JSONB KV, metadata index, pool |
| `sourcing/` | L3 | Local / Azure Blob / S3 data sources |
| `converters/` | L3 | LibreOffice + page renderer (shared by both modes) |
| `citations/` | L3 | Index, parser, processor, highlight, streaming |
| `core.ingestion.{policy,docling_postprocess,metadata_extract}` | L3 | Pure transforms on parsed content |
| `models/{embedding,llm,rerank}` | L4 | Build LightRAG-compatible callables from `ModelConfig` |
| `core.retrieval.{filtered_vdb,metadata_path}` | L4 | VDB filter wrapper + metadata→chunk_ids resolver |
| `core.ingestion.{hash_index,cleanup,page_metadata}` | L4 | Dedup, cascade delete, page-idx backfill |
| `captionrag.chunking` | L4 | Docling HybridChunker for LightRAG |
| `unifiedrepresent.embedder` | L4 | Visual embedding HTTP client |
| `core.vlm_ocr` | L4 | Shared VLM OCR helpers |
| `captionrag.{vlm_parser,retrieval}` | L5 | Caption mode stages |
| `unifiedrepresent.{extractor,retriever,multimodal_query}` | L5 | Unified mode stages |
| `core.{_lightrag_patches,compat_guard}` | L5 | LightRAG hardening |
| `captionrag.pipeline` | L6 | Caption ingestion orchestrator |
| `unifiedrepresent.{engine,lifecycle}` | L6 | Unified ingestion + retrieval orchestrator |
| `core.{federation,answer,query_planner}` | L6 | Cross-cutting orchestrators |
| `core.{service,reset,ingest_tasks}` | L7 | `RAGService` + workspace lifecycle |
| `core.servicemanager` | L8 | Multi-workspace pool, federation entry |
| `api/`, `mcp/`, `web/` | L9 | Interface adapters |

## Verification

The layering is enforced by an AST audit, not runtime guards. To re-run
it (e.g. before a release, or after large refactors):

```bash
uv run python - <<'PY'
import ast
from collections import defaultdict
from pathlib import Path

# Layer table — keep in sync with this doc when adding modules.
LAYER = {
    # ... see commit 2b83c99 for the full table used during the audit
}

src_root = Path("src/dlightrag")
graph: dict[str, set[str]] = defaultdict(set)
for py in src_root.rglob("*.py"):
    if "__pycache__" in py.parts:
        continue
    rel = py.relative_to(src_root.parent.parent / "src").with_suffix("")
    mod = ".".join(rel.parts).removesuffix(".__init__")
    tree = ast.parse(py.read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("dlightrag"):
            graph[mod].add(n.module)
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("dlightrag"):
                    graph[mod].add(a.name)

# ... resolve LAYER for src and dep, flag where src_layer < dep_layer ...
PY
```

Expected output: **0 layer violations, 0 cross-mode imports
(`captionrag` ↔ `unifiedrepresent`)**, no top-level circular imports.

## When adding a module

Decide its layer first by asking: *what's the highest layer this would
need to import from?* Place the new module **one layer above** that.

Common pitfalls:
- Putting a "data type" file inside an impl directory (e.g.
  `storage/metadata_fields.py` was a schema registry, not storage —
  belongs in L2).
- Putting a "wrapper that needs L3" into L1 (e.g. `models/streaming.py`
  imported `citations/` — belongs in L3 next to citations).
- A helper used by both `captionrag/` and `unifiedrepresent/` shouldn't
  live in either; lift it to a shared layer (e.g. `converters/`,
  `core.retrieval/`, `models/`).
