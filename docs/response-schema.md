# API Reference

Request and response structures for `ingest`, `retrieve`, and `answer` — shared across Python SDK, REST API, MCP server, and Web UI.


## Ingestion

### Python SDK

```python
from dlightrag import RAGServiceManager, DlightragConfig

manager = await RAGServiceManager.create(DlightragConfig())
try:
    # Local files or directory
    result = await manager.aingest("default", source_type="local", path="./docs")

    # Azure Blob Storage
    result = await manager.aingest(
        "default",
        source_type="azure_blob",
        container_name="documents",
        prefix="reports/",       # or blob_path="reports/q1.pdf"
    )

    # AWS S3
    result = await manager.aingest(
        "default",
        source_type="s3",
        bucket="my-bucket",
        key="docs/q1.pdf",
    )
finally:
    await manager.close()
```

### REST API

```bash
curl -X POST http://localhost:8100/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type": "local", "path": "/data/docs"}'
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `source_type` | `string` | yes | `local`, `azure_blob`, `s3` |
| `path` | `string` | local | File or directory path |
| `container_name` | `string` | azure_blob | Blob container name |
| `blob_path` | `string` | — | Specific blob (mutually exclusive with `prefix`) |
| `prefix` | `string` | — | Blob/key prefix filter |
| `bucket` | `string` | s3 | S3 bucket name |
| `key` | `string` | s3 | S3 object key |
| `replace` | `boolean` | — | Replace existing documents with same content hash (cascade-purges prior record after the new ingest succeeds) |
| `workspace` | `string` | — | Target workspace (default: `default`) |
| `title` | `string` | — | User-declared document title stored in metadata |
| `author` | `string` | — | User-declared document author stored in metadata |
| `metadata` | `object` | — | Declared/custom ingest metadata |
| `metadata_policy` | `string` | — | `validate`, `reject_unknown`, or `store_only` |

REST also supports one-file multipart upload at `POST /ingest/blob`. Fields are
`file` plus optional `workspace`, `title`, `author`, `metadata` (JSON string),
and `metadata_policy`. The file is staged under DlightRAG's managed input
directory and ingested through the same local pipeline.

### MCP Server

MCP `ingest` exposes the same source and metadata arguments as REST `/ingest`,
passed as tool arguments.

### Metadata Schema And Policy

The filterable metadata schema is service configuration, not an ingest-time
API payload. Declare custom filter fields in `config.yaml` under
`metadata.fields`; REST, MCP, and SDK ingest calls then pass values for those
fields through `metadata`.

```yaml
metadata:
  allow_ad_hoc_json: true
  default_ingest_policy: validate
  fields:
    department:
      type: string
      filter_ops: ["exact"]
```

For string fields with `exact` filtering, the default normalizer is
`casefold_trim`; set `normalizer: identity` only for case-sensitive identifiers.
There is no separate `indexed` flag: declaring a field with filter operations
is the product signal that it is filterable.

System metadata such as `filename`, `filename_stem`, `file_extension`,
`doc_title`, `doc_author`, parser details, and ingest strategy is extracted or
mapped by DlightRAG. User metadata follows the configured policy:

| Policy | Behavior |
|---|---|
| `validate` | Default. Declared filterable fields are normalized and promoted to `custom_metadata`; undeclared fields are stored as JSON enrichment when `allow_ad_hoc_json` is true. |
| `reject_unknown` | Rejects undeclared user metadata fields. |
| `store_only` | Stores user metadata as JSON enrichment but does not promote declared fields for filtering. |

Query filters use the declared schema. For custom metadata, pass
`filters.custom`, for example `{"custom": {"department": "finance"}}`.
Undeclared JSON enrichment is retained for display/debugging, but is not part
of the supported filter surface.

### Ingestion Response

Single-file ingestion returns the concrete file result from the unified
LightRAG path:

```json
{
  "doc_id": "file-doc-abc123",
  "source_kind": "document",
  "chunks": ["chunk-a", "chunk-b"],
  "ingest_strategy": "lightrag_sidecar_unified",
  "parse_engine": "mineru",
  "process_options": "iteP"
}
```

Directory, Web upload, or remote prefix ingestion uses LightRAG's staged batch
pipeline and wraps the per-file results:

```json
{
  "processed": 2,
  "errors": [],
  "results": [
    {
      "doc_id": "file-doc-abc123",
      "source_kind": "document",
      "chunks": ["chunk-a", "chunk-b"],
      "ingest_strategy": "lightrag_sidecar_unified",
      "parse_engine": "mineru",
      "process_options": "iteP"
    },
    {
      "doc_id": "file-doc-def456",
      "source_kind": "document",
      "chunks": ["chunk-c"],
      "ingest_strategy": "lightrag_sidecar_unified",
      "parse_engine": "mineru",
      "process_options": "iteP"
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `doc_id` | `string` | Canonical document id for a single ingested file |
| `source_kind` | `string` | `document` or `skipped` |
| `status` | `string` | Present for unsupported-format skips (`skipped`) |
| `reason` | `string` | Skip reason when a file is not ingested |
| `chunks` | `list[string]` | LightRAG chunk IDs created or reused |
| `ingest_strategy` | `string` | Ingestion path used for successful files |
| `parse_engine` | `string` | Parser selected for document files |
| `process_options` | `string` | LightRAG parser process options, for example `iteP` |
| `processed` | `int` | Files represented in a directory/upload/prefix batch result |
| `errors` | `list[string]` | Compatibility field for batch wrappers; batch-level failures raise instead |
| `results` | `list[object]` | Per-file results |
| `replica_replay_lsn` | `string` | Present when `read_after_write_mode: wait_for_replay` waited for replica WAL replay |


## Retrieval & Answer

### Quick Reference

| Interface | `retrieve` | `answer` | Streaming |
|---|---|---|---|
| Python SDK | `RetrievalResult` | `RetrievalResult` | `(contexts, token_iter)` |
| REST API | JSON object | JSON object | SSE (`stream: true`) |
| MCP Server | JSON text | JSON text | N/A |
| Web UI | — | SSE (HTML) | Built-in |

### Python SDK

```python
# Retrieve: contexts only, no LLM answer
result = await manager.aretrieve(query="What are the key findings?")
result.answer     # None
result.contexts   # RetrievalContexts: {"chunks": [...], "entities": [...], "relationships": [...]}

# Answer: contexts + LLM-generated answer
result = await manager.aanswer(query="What are the key findings?")
result.answer      # "The key findings are... [1-1] [2-3]"
result.contexts    # same structure as retrieve, packed to what the answer model saw
result.references  # validated cited documents, derived from inline citations

# Streaming answer
contexts, token_iter = await manager.aanswer_stream(query="What are the key findings?")
# contexts (answer-packed RetrievalContexts) available immediately
async for token in token_iter:
    print(token, end="")
```

**Parameters** (shared by all three methods):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | required | Search query |
| `workspace` | `str \| None` | config default | Target workspace |
| `workspaces` | `list[str] \| None` | `None` | Federated search across multiple workspaces |
| `top_k` | `int \| None` | config default | LightRAG KG breadth: entities in local retrieval and relationships in global retrieval. |
| `chunk_top_k` | `int \| None` | config default | Chunk-level vector results for `/retrieve`; for `/answer`, `answer_candidate_top_k` is mapped to this unless explicitly overridden. |
| `answer_candidate_top_k` | `int \| None` | `answer.candidate_top_k` | `/answer` only. Chunk/visual candidates fetched before answer-stage packing; maps to LightRAG `QueryParam.chunk_top_k`, not `QueryParam.top_k`. |
| `answer_context_top_k` | `int \| None` | `answer.context_top_k` | `/answer` only. Maximum chunks included in the final answer prompt after image-budget packing and backfill. |
| `stream` | `bool` | `true` for REST `/answer` | `true` returns SSE; pass `false` to opt into one JSON response |
| `multimodal_content` | `list[dict]` | `None` | Raw direct visual-retrieval inputs. Use for programmatic image embedding when the answer model does not need to see the image. |
| `query_images` | `list[str \| dict]` | `None` | User-attached images. They are described by the VLM for semantic/BM25 retrieval, embedded directly for visual retrieval, stored in session memory when `session_id` is present, and bounded before being sent to the answer LLM. Capped at 10. |
| `session_id` | `str \| None` | `None` | Conversation/session key for reusing uploaded query images. |
| `referenced_image_ids` | `list[str] \| None` | `None` | Image IDs from a previous `image_meta` event or JSON response to include again in retrieval and answer generation. |
| `filters` | `MetadataFilter \| None` | `None` | Structured metadata filter (also auto-detected from query); supports declared metadata fields such as filename, extension, title, author, dates, and custom fields |

### REST API

```bash
# Retrieve
curl -X POST http://localhost:8100/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "key findings"}'

# Answer as one JSON response
curl -X POST http://localhost:8100/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "key findings", "stream": false}'

# Streaming answer (default)
curl -X POST http://localhost:8100/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "key findings"}'

# Create an empty workspace
curl -X POST http://localhost:8100/workspaces \
  -H "Content-Type: application/json" \
  -d '{"workspace": "Research Notes"}'

# Delete/reset a workspace
curl -X DELETE "http://localhost:8100/workspaces/research_notes?keep_files=false"
```

**Workspace list response:**

```json
{
  "workspaces": ["default", "research_notes"],
  "records": [
    {
      "workspace": "default",
      "display_name": "default",
      "embedding_model": "voyage-multimodal-3.5",
      "created_at": "2026-05-25T19:22:22.788620+00:00",
      "updated_at": "2026-05-25T19:42:08.781671+00:00"
    }
  ]
}
```

**Non-streaming response:**

```json
{
  "answer": "The key findings are... [1-1] [2-3]",
  "contexts": { "chunks": [...], "entities": [...], "relationships": [...] },
  "references": [{"id": "1", "title": "report.pdf"}, {"id": "2", "title": "spec.pdf"}],
  "sources": [...],
  "trace": {...},
  "image_descriptions": ["Image 1: a line chart about revenue"],
  "current_image_ids": ["img_0"]
}
```

**SSE streaming** (`stream: true`): events are newline-delimited JSON.

| Event | Payload | Description |
|---|---|---|
| `context` | `{type, data}` | Full contexts, sent first |
| `token` | `{type, content}` | LLM answer token (repeats) |
| `sources` | `{type, data}` | Validated cited sources, after all tokens and before done |
| `trace` | `{type, data}` | Retrieval trace counts and planner/filter decisions |
| `image_meta` | `{type, current_image_ids, image_descriptions}` | Session image IDs and VLM image descriptions |
| `done` | `{type, answer}` | Stream complete; `answer` is the final normalized answer body after citation validation |
| `error` | `{type, message}` | Error mid-stream |

```
data: {"type":"context","data":{"chunks":[...],"entities":[...],"relationships":[...]},"sources":[...]}

data: {"type":"token","content":"The key findings"}

data: {"type":"token","content":" are..."}

data: {"type":"sources","data":[{"id":"1","title":"report.pdf","path":"/data/report.pdf","chunks":[...]}]}

data: {"type":"trace","data":{"bm25_enabled":true,"fused_chunk_count":8}}

data: {"type":"image_meta","current_image_ids":["img_0"],"image_descriptions":["Image 1: a line chart"]}

data: {"type":"done","answer":"The key findings are..."}
```

REST uses the same fields as the Python manager methods. `retrieve` and
`answer` both accept `filters`, `query_images`, and `multimodal_content`.

### MCP Server

MCP tools return JSON text with `sources` at top level:

```json
{
  "answer": "The key findings are... [1-1]",
  "contexts": { "chunks": [...], "entities": [...], "relationships": [...] },
  "references": [{"id": "1", "title": "report.pdf"}],
  "sources": [...]
}
```


## Contexts Object

All modes return `contexts` as a `RetrievalContexts` TypedDict with three arrays. Chunks are the primary retrieval unit; entities and relationships come from the knowledge graph.

REST and Web responses never expose inline base64 page/image payloads. When a
retrieved chunk has a visual sidecar, DlightRAG projects it to
`image_url`/`thumbnail_url` routes. Python manager internals may still carry
`image_data` inside contexts so answer generation and reranking can use bounded
multimodal payloads without a second database read.

```python
from dlightrag.core.retrieval.protocols import RetrievalContexts, ChunkContext, EntityContext, RelationshipContext
```

### chunks

```json
{
  "chunk_id": "abc123",
  "reference_id": "1",
  "file_path": "/data/report.pdf",
  "content": "Page text content...",
  "page_idx": 2,
  "bbox": {"x0": 0.1, "y0": 0.2, "x1": 0.8, "y1": 0.6},
  "image_url": "/images/default/abc123?size=full",
  "thumbnail_url": "/images/default/abc123?size=thumb",
  "image_mime_type": "image/png",
  "relevance_score": 0.87
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `chunk_id` | string | yes | Unique chunk identifier |
| `reference_id` | string | yes | Document-level ID (groups chunks from the same file) |
| `file_path` | string | yes | Source file path |
| `content` | string | yes | Chunk text content |
| `page_idx` | int \| null | no | **1-based** page number |
| `bbox` | object \| null | no | Visual block bounding box when LightRAG/MinerU sidecar provenance provides one |
| `image_url` | string \| null | no | Full image route for visual chunks in public REST/Web responses |
| `thumbnail_url` | string \| null | no | Thumbnail route for source-panel rendering |
| `image_mime_type` | string \| null | no | MIME type for the visual asset |
| `relevance_score` | float \| null | no | 0–1 relevance score (when reranking is enabled) |
| `metadata` | object | no | Extra metadata (`file_name`, `file_type`, etc.) |
| `_workspace` | string | no | Source workspace (federated queries only) |

### entities

```json
{
  "entity_name": "PostgreSQL",
  "entity_type": "TECHNOLOGY",
  "description": "An open-source relational database",
  "source_id": "abc123"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `entity_name` | string | yes | Entity name/label |
| `entity_type` | string | yes | Category (Person, Organization, Technology, etc.) |
| `description` | string | yes | Summary description |
| `source_id` | string | yes | Comma-separated `chunk_id` values linking to source chunks |
| `reference_id` | string | no | Document reference (inferred from source_id) |

### relationships

```json
{
  "src_id": "PostgreSQL",
  "tgt_id": "pgvector",
  "description": "extension for vector similarity search",
  "source_id": "abc123"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `src_id` | string | yes | Source entity name |
| `tgt_id` | string | yes | Target entity name |
| `description` | string | yes | Relationship description |
| `source_id` | string | yes | Comma-separated `chunk_id` values linking to source chunks |
| `reference_id` | string | no | Document reference (inferred from source_id) |


## Sources

Sources are document-level groupings derived from chunks via `build_sources()`.
They appear in REST/MCP responses and drive the Web UI's source panel. Cited
answer paths use the same citation indexer as answer validation, so chunk order
matches `[ref_id-chunk_idx]` markers instead of page sorting.

```json
{
  "id": "1",
  "title": "report.pdf",
  "path": "/data/dlightrag_storage/docs/report.pdf",
  "type": "file",
  "url": "/api/files/docs/report.pdf",
  "cited_chunk_ids": ["abc123", "def456"],
  "chunks": [
    {
      "chunk_id": "abc123",
      "chunk_idx": 1,
      "page_idx": 2,
      "bbox": null,
      "content": "First 200 characters of content...",
      "image_url": null,
      "thumbnail_url": null
    },
    {
      "chunk_id": "def456",
      "chunk_idx": 2,
      "page_idx": 5,
      "bbox": {"x0": 0.2, "y0": 0.1, "x1": 0.7, "y1": 0.5},
      "content": "Another chunk...",
      "image_url": "/images/default/def456?size=full",
      "thumbnail_url": "/images/default/def456?size=thumb"
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `id` | string | Reference ID (matches `reference_id` in chunks) |
| `title` | string \| null | Document title (filename or metadata) |
| `path` | string | Source file path |
| `type` | string \| null | File type |
| `url` | string \| null | Resolved download URL (via PathResolver) |
| `cited_chunk_ids` | list \| null | Cited chunk IDs for answer responses; null when returning all retrieved sources |
| `chunks` | list | Chunk snippets in citation-index order |

Each **chunk snippet** within a source:

| Field | Type | Description |
|---|---|---|
| `chunk_id` | string | Unique chunk identifier |
| `chunk_idx` | int | 1-based position within this source; matches `[ref_id-chunk_idx]` citations |
| `page_idx` | int \| null | 1-based page number |
| `bbox` | object \| null | Visual block bounding box when available |
| `content` | string | Filtered display content |
| `image_url` | string \| null | Full visual asset route |
| `thumbnail_url` | string \| null | Thumbnail visual asset route |
| `highlight_phrases` | list \| null | Semantic highlight phrases (when available) |


## References

The `answer` response includes a `references` array containing document-level
references cited in the answer. DlightRAG derives this from validated inline
citations, not from provider-specific structured output or generated
`### References` tails. The richer `sources` array is the same cited subset
with paths, chunks, pages, images, and optional highlights.

```json
{
  "id": "1",
  "title": "report.pdf"
}
```

| Field | Type | Description |
|---|---|---|
| `id` | string | Reference ID matching `[n]` in inline citations |
| `title` | string | Document title/filename |

**Relationship to `sources`:** `retrieve` returns all retrieved sources. `answer`
returns answer-packed contexts plus only cited sources after citation
validation. `references` is a compact document-level projection of that cited
`sources` list. Answer packing removes pure visual chunks whose image could not
fit the answer image budget, while preserving text from mixed text+image
chunks.


## Citations

When using `answer`, the LLM response may contain inline citations in two formats:

| Format | Example | Meaning |
|---|---|---|
| `[ref_id-chunk_idx]` | `[1-2]` | Chunk-level: document 1, chunk 2 |
| `[n]` | `[3]` | Document-level: all chunks from document 3 |

- `ref_id` maps to `reference_id` in chunks and `id` in sources
- `chunk_idx` is **1-based**, matching the chunk's position within its document

### Resolving a citation

To trace `[1-2]` back to source material:

1. Find chunks where `reference_id == "1"` — these are all chunks from that document
2. The 2nd chunk (1-based) in that group is the cited chunk
3. Use `chunk_id` to look up the source in `sources` (by matching `id`)
4. Use `page_idx` on the chunk for the page number
5. Use `page_idx` and `bbox` when present for page/block localization
6. Use source `url` or `GET /api/files/{path}` for the original file; use
   `image_url`/`thumbnail_url` for retrieved visual chunks


## Multimodal Queries

Upload images alongside a text query for visual similarity search and answer
reasoning:

```python
# Python SDK
with open("photo.png", "rb") as f:
    img_bytes = f.read()

result = await manager.aanswer(
    query="What does this diagram show?",
    query_images=[base64.b64encode(img_bytes).decode("ascii")],
    session_id="chat-123",
)
```

```bash
# REST API — base64-encoded image
curl -X POST http://localhost:8100/answer \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does this diagram show?",
    "stream": false,
    "query_images": ["<base64>"],
    "session_id": "chat-123"
  }'

# REST API — lower-level direct visual payload
curl -X POST http://localhost:8100/answer \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does this diagram show?",
    "stream": false,
    "multimodal_content": [{"type": "image", "data": "<base64>"}]
  }'
```

`query_images` is the user-facing chat path: DlightRAG stores bounded session
images, asks the VLM for concise semantic descriptions, embeds the raw image
for direct visual retrieval, and sends a bounded image preview to the answer
model. `multimodal_content` remains the lower-level direct visual retrieval
input for programmatic callers.

Answer-model previews are quality-preserving. Budgeted JPEG, PNG, and WebP
payloads pass through unchanged; oversized images are recompressed only down to
the configured `answer.image_min_quality` and `answer.image_min_px` floors. If
an image still cannot fit, DlightRAG skips it rather than sending a degraded
preview that could hurt visual understanding. Pure visual retrieved chunks
whose image is skipped are also removed from the answer context; later text or
sendable visual chunks in the retrieved set remain available to the answer
model.
