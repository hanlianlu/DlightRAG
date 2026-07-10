# Interfaces

This page is for SDK, REST, MCP, and Web consumers. It owns request and response
contracts for ingestion, jobs, retrieval, answers, contexts, sources, citations,
and multimodal payloads. Security posture lives in [security.md](security.md);
runtime retrieval behavior lives in [retrieval-answer.md](retrieval-answer.md);
configuration fields live in [configuration.md](configuration.md).

## Interface Overview

| Interface | Primary use | Ingestion behavior |
|---|---|---|
| Python SDK | In-process applications and custom connectors | Foreground via `aingest`; background via `astart_ingest_job` |
| REST API | Web clients, services, and remote callers | Durable ingest jobs |
| MCP Server | Agent tools over stdio or streamable HTTP | Durable ingest jobs |
| Web UI | Browser upload and chat | Durable ingest jobs behind the Files panel |

## Ingestion

### Python SDK

```python
from collections.abc import AsyncIterator
from pathlib import Path

from dlightrag import DlightragConfig, IngestSpec, RAGServiceManager
from dlightrag.sourcing import AsyncDataSource, SourceDocument


class BynderSource(AsyncDataSource):
    """Adapter around your own Bynder client; not a built-in DlightRAG connector."""

    def __init__(self, bynder_client) -> None:
        self.bynder_client = bynder_client

    async def aiter_documents(self, prefix: str | None = None) -> AsyncIterator[SourceDocument]:
        async for asset in self.bynder_client.iter_assets(prefix=prefix):
            key = f"{asset['id']}/{asset['filename']}"
            yield SourceDocument(
                key=key,
                source_uri=f"bynder://assets/{asset['id']}",
                display_filename=asset["filename"],
                title=asset.get("title"),
                metadata={
                    "asset_id": asset["id"],
                    "collection": asset.get("collection"),
                },
            )

    async def amaterialize_document(self, document: SourceDocument, destination: Path) -> None:
        # User-owned client method. Stream or write bytes to the supplied path.
        asset_id = document.key.split("/", 1)[0]
        await self.bynder_client.download_asset_to_file(
            asset_id,
            destination,
        )


manager = await RAGServiceManager.create(DlightragConfig())
try:
    # Local files or directory
    result = await manager.aingest("default", IngestSpec(source_type="local", path="./docs"))

    # Azure Blob Storage
    result = await manager.aingest(
        "default",
        IngestSpec(
            source_type="azure_blob",
            container_name="documents",
            prefix="reports/",       # or blob_path="reports/q1.pdf"
        ),
    )

    # AWS S3
    result = await manager.aingest(
        "default",
        IngestSpec(
            source_type="s3",
            bucket="my-bucket",
            region="us-east-1",      # optional; credentials come from AWS env/config/IAM
            key="docs/q1.pdf",       # or prefix="docs/"
        ),
    )

    # Custom SDK connector; useful for Bynder/SaaS clients that handle auth
    # and write document bytes into DlightRAG's parser staging path.
    result = await manager.aingest_source(
        "default",
        BynderSource(bynder_client),
        source_type="bynder",
        metadata={"source_system": "bynder"},
    )

    # Explicit non-blocking ingest
    job = await manager.astart_ingest_job(
        "default",
        IngestSpec(source_type="s3", bucket="my-bucket", prefix="docs/"),
    )
    status = await manager.get_ingest_job(job["job_id"])
finally:
    await manager.close()
```

DlightRAG calls `aiter_documents()` to discover `SourceDocument` descriptors and
`amaterialize_document(document, destination)` to write each document into parser
staging without loading the whole object into memory. Ingest-call `metadata` is
the batch default; `SourceDocument.metadata` overlays it for that document.

### REST API

```bash
curl -X POST http://localhost:8100/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type": "local", "path": "docs"}'

curl -X POST http://localhost:8100/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type": "url", "url": "https://api.bynder.com/docs/getting-started", "filename": "getting-started.html"}'

curl -X POST http://localhost:8100/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type": "url", "url": "https://cdn.example.com/download?id=asset-1&signature=secret", "filename": "asset.pdf", "source_uri": "bynder://asset/asset-1"}'
```

All ingest operations are represented internally as jobs. REST and MCP ingest
return `202 Accepted`; poll `GET /ingest/jobs/{job_id}` for progress and final
result. `source_type="url"` is intentionally limited to public or signed HTTPS
URLs; authenticated SaaS APIs should fetch through a caller-owned SDK
`AsyncDataSource` connector and use `aingest_source()`. S3 credentials are read
from the standard AWS credential chain (environment, shared config, or IAM
role); ingest payloads do not carry access keys.

For URL ingest, `url`/`urls` are fetch endpoints. Persisted provenance defaults
to the same URL without query or fragment so signed tokens are not stored.
Pass `source_uri` for one URL or `source_uris` for a URL batch when the durable
source identity is a SaaS asset id, CMS URI, or another stable reference.

Remote source files are transient by default. Enabling retention with
`retain_remote_source_files` in config or per-call `retain_source_file` keeps
fetched files under the workspace input root and points stored metadata
`file_path` at that retained local file.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `source_type` | `string` | yes | `local`, `azure_blob`, `s3`, `url` |
| `path` | `string` | local | File or directory path relative to DlightRAG's managed `input_dir/<workspace>` |
| `container_name` | `string` | azure_blob | Blob container name |
| `blob_path` | `string` | — | Specific blob (mutually exclusive with `prefix`) |
| `prefix` | `string` | — | Blob/key prefix filter for `azure_blob`/`s3` batches; mutually exclusive with `blob_path`/`key`. Omit (or pass `""`) to ingest the whole container/bucket. |
| `bucket` | `string` | s3 | S3 bucket name. With neither `key` nor `prefix`, ingests the whole bucket. |
| `region` | `string` | — | Optional S3 region for this ingest; falls back to `s3_region` or the AWS SDK environment/config defaults |
| `key` | `string` | — | S3 object key for a single object; mutually exclusive with `prefix` |
| `url` | `string` | url | Single public or signed HTTPS document URL |
| `urls` | `list[string]` | url | Multiple public or signed HTTPS document URLs; mutually exclusive with `url` |
| `filename` | `string` | — | Parser filename for a single URL, useful when the URL path has no extension |
| `source_uri` | `string` | — | Stable stored source URI for a single URL, independent of the signed fetch URL |
| `source_uris` | `list[string]` | — | Stable stored source URIs for URL batches; must match `urls` length |
| `documents` | `list[object]` | — | Explicit document manifest. Local documents use `path`, S3/Azure use `key`, URL documents use `url`; per-document metadata overlays request metadata. |
| `retain_source_file` | `boolean` | — | Per-call remote source retention override. `true` keeps fetched remote files under the workspace input root; `false` keeps them transient even if config defaults to retention. |
| `replace` | `boolean` | — | Replace existing documents by cascade-purging the prior LightRAG record before enqueueing the new ingest |
| `workspace` | `string` | — | Target workspace (default: `default`) |
| `title` | `string` | — | User-declared document title stored in metadata |
| `author` | `string` | — | User-declared document author stored in metadata |
| `metadata` | `object` | — | Declared/custom ingest metadata |
| `metadata_policy` | `string` | — | `validate`, `reject_unknown`, or `store_only` |
REST also supports one-file multipart upload at `POST /ingest/blob`. Fields are
`file` plus optional `workspace`, `title`, `author`, `metadata` (JSON string),
and `metadata_policy`. The file is staged under DlightRAG's managed input
workspace directory and returns an ingest job.

For per-document metadata, pass a manifest instead of prefix discovery:

```json
{
  "source_type": "s3",
  "bucket": "my-bucket",
  "metadata": {"source_system": "s3-prod"},
  "documents": [
    {"key": "docs/a.pdf", "metadata": {"department": "legal", "asset_id": "a"}}
  ]
}
```

### MCP Server

MCP `ingest` exposes the same source and metadata arguments as REST `/ingest`,
passed as tool arguments. Local path arguments are relative to the managed
`input_dir/<workspace>`. Calls return a background job; call
`ingest_job_status` with the returned `job_id` to read progress.

### Metadata At Call Time

The filterable metadata schema is service configuration, not an ingest-time API
payload. Declare custom filter fields in `config.yaml` under `metadata.fields`;
REST, MCP, and SDK ingest calls then pass values for those fields through
`metadata`. Configuration fields and defaults live in
[configuration.md](configuration.md).

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

Directory and Web upload ingestion use LightRAG's staged batch pipeline and
wrap the per-file results:

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
| `errors` | `list[string]` | Per-file ingest errors collected by the batch result; batch-level failures raise instead |
| `results` | `list[object]` | Per-file results |

Background ingestion through REST or MCP returns a job first:

```json
{
  "job_id": "8f3b7c1d9d9a4e6e8e5f6a7b8c9d0e1f",
  "workspace": "default",
  "source_type": "s3",
  "status": "queued",
  "total_items": 0,
  "processed_items": 0,
  "failed_items": 0,
  "current_window": 0,
  "errors": [],
  "request": {
    "workspace": "default",
    "source_type": "s3",
    "kwargs": {"bucket": "my-bucket", "prefix": "docs/"}
  },
  "result": {},
  "status_url": "/ingest/jobs/8f3b7c1d9d9a4e6e8e5f6a7b8c9d0e1f"
}
```

`GET /ingest/jobs/{job_id}` and MCP `ingest_job_status` return the same job row.
`status` is one of `queued`, `running`, `succeeded`, or `failed`. When the job
succeeds, `result` contains the same single-file or staged batch response shown
above. If a synchronous REST/MCP ingest exceeds `ingest_timeout`, the job keeps
running and the transport returns the same `202` job shape instead of cancelling
the ingest. On service startup, recent `queued`/`running` rows are recovered
automatically. Remote prefix jobs resume from `current_window`, so completed
source windows are not downloaded again; already processed documents are still
deduplicated by LightRAG's document status and DlightRAG's content-hash guard.


## Retrieval And Answer

### Quick Reference

| Interface | `retrieve` | `answer` | Streaming |
|---|---|---|---|
| Python SDK | `RetrievalResult` | `RetrievalResult` | `(contexts, token_iter)` |
| REST API | JSON object | JSON object | SSE (`stream: true`) |
| MCP Server | JSON text | JSON text | N/A |
| Web UI | — | SSE (HTML) | Built-in |
| CLI (`scripts/cli.py`) | JSON object printed to stdout | Terminal text; `answer_blocks` image refs render as image URL lines | N/A |

### Contract Terms

| Term | Meaning |
|---|---|
| `contexts` | Evidence package. `/retrieve` returns the broader retrieved set; `/answer` returns the packed evidence that the answer model actually saw. |
| `sources` | Document-level source objects with chunks, pages, optional visual routes, and optional highlights. `/retrieve` returns all retrieved sources; `/answer` returns only cited sources. |
| `references` | Compact document-level citation summary for answers, derived from validated inline citations. |
| `answer_images` | Registry of cited visual assets available for rendering. Entries reference image routes, not inline document image bytes. |
| `answer_blocks` | Display plan for answers: markdown text blocks plus `image_ref` blocks that point into `answer_images`. |

### Python SDK

```python
# Retrieve: contexts only, no LLM answer
result = await manager.aretrieve(query="What are the key findings?")
result.answer     # None
result.contexts   # RetrievalContexts: {"chunks": [...], "entities": [...], "relationships": [...]}

# Answer: contexts + LLM-generated answer
result = await manager.aanswer(
    query="What are the key findings?",
    semantic_highlights=True,  # optional; default false outside Web
)
result.answer      # "The key findings are... [1-1] [2-3]"
result.contexts    # same structure as retrieve, packed to what the answer model saw
result.references  # validated cited documents, derived from inline citations
result.answer_images  # cited visual assets available for rendering
result.answer_blocks  # markdown/image_ref blocks for structured display

# Streaming answer
contexts, token_iter = await manager.aanswer_stream(query="What are the key findings?")
# contexts (answer-packed RetrievalContexts) available immediately
async for token in token_iter:
    print(token, end="")
```

**Parameters**:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | required | Search query |
| `workspace` | `str \| None` | config default | Target workspace |
| `workspaces` | `list[str] \| None` | `None` | Federated search across multiple workspaces |
| `top_k` | `int \| None` | config default | LightRAG KG breadth: entities in local retrieval and relationships in global retrieval. |
| `chunk_top_k` | `int \| None` | config default | Explicit chunk/visual candidates fetched for `/retrieve` and before `/answer` packing. Maps to LightRAG `QueryParam.chunk_top_k`, not `QueryParam.top_k`. |
| `answer_context_top_k` | `int \| None` | `answer.context_top_k` | `/answer` only. Maximum chunks included in the final answer prompt after image-budget packing and backfill. |
| `stream` | `bool` | `true` for REST `/answer` | `true` returns SSE; pass `false` to opt into one JSON response |
| `multimodal_content` | `list[dict]` | `None` | Raw direct visual-retrieval inputs. Use for programmatic image embedding when the answer model does not need to see the image. |
| `query_images` | `list[QueryImage]` | `None` | User-attached OpenAI-style `image_url` blocks. They are described by the VLM for semantic/BM25 retrieval, embedded directly for visual retrieval, stored in session memory when `session_id` is present, and bounded by the user-image answer budget before being sent to the answer LLM. Capped at 3. |
| `session_id` | `str \| None` | `None` | Conversation/session key for reusing uploaded query images. |
| `referenced_image_ids` | `list[str] \| None` | `None` | Image IDs from a previous `image_meta` event or JSON response to include again in retrieval and answer generation. |
| `semantic_highlights` | `bool` | `false` | `/answer` only. When true and `citations.highlights.enabled` is true, fills `sources[].chunks[].highlight_phrases` with answer-aware phrase highlights. |
| `filters` | `MetadataFilter \| None` | `None` | Structured metadata filter (also auto-detected from query); supports declared metadata fields such as filename, extension, title, author, dates, and custom fields |

### REST API

DlightRAG does not expose a public `/query` route. Use `/retrieve` when a
client needs contexts only, and `/answer` when it needs generated text,
validated citations, and structured answer media.

```bash
# Retrieve
curl -X POST http://localhost:8100/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "key findings"}'

# Answer as one JSON response
curl -X POST http://localhost:8100/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "key findings", "stream": false, "semantic_highlights": true}'

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

Workspace reset results include `ingest_jobs_cancelled`, the number of active
in-process ingest jobs cancelled before reset, and `ingest_jobs_deleted`, the
number of durable ingest job rows removed for that workspace. Dry-run reset
reports `0` for both fields and does not cancel jobs or mutate the job table.
`DELETE /files` accepts `dry_run: true` to report matched documents and source
paths without deleting LightRAG rows, metadata, or local files.

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
  "answer_images": [
    {
      "id": "fig-1",
      "chunk_id": "fig-1",
      "source_ref": "1-1",
      "url": "/images/default/fig-1?size=full",
      "thumbnail_url": "/images/default/fig-1?size=thumb",
      "label": "report.pdf"
    }
  ],
  "answer_blocks": [
    {"type": "markdown", "text": "The diagram shows... [1-1]."},
    {"type": "image_ref", "image_id": "fig-1"}
  ],
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
| `sources` | `{type, data}` | Validated cited sources, after all tokens and before done. Includes `highlight_phrases` when `semantic_highlights` is true and enrichment succeeds. |
| `trace` | `{type, data}` | Retrieval trace counts and planner/filter decisions |
| `image_meta` | `{type, current_image_ids, image_descriptions}` | Session image IDs and VLM image descriptions |
| `done` | `{type, answer, answer_images, answer_blocks}` | Stream complete; `answer` is the final normalized answer body after citation validation |
| `error` | `{type, message}` | Error mid-stream |

```
data: {"type":"context","data":{"chunks":[...],"entities":[...],"relationships":[...]}}

data: {"type":"token","content":"The key findings"}

data: {"type":"token","content":" are..."}

data: {"type":"sources","data":[{"id":"1","title":"report.pdf","path":"/data/report.pdf","chunks":[...]}]}

data: {"type":"trace","data":{"bm25_enabled":true,"fused_chunk_count":8}}

data: {"type":"image_meta","current_image_ids":["img_0"],"image_descriptions":["Image 1: a line chart"]}

data: {"type":"done","answer":"The key findings are...","answer_images":[],"answer_blocks":[{"type":"markdown","text":"The key findings are..."}]}
```

REST uses the same fields as the Python manager methods.

Web streaming uses the same answer pipeline but always attempts semantic
highlights when citation highlighting is enabled. If phrases are found, Web
emits a `highlights` SSE event with the updated source panel HTML.

### MCP Server

MCP tools return JSON text with `sources` at top level:

```json
{
  "answer": "The key findings are... [1-1]",
  "contexts": { "chunks": [...], "entities": [...], "relationships": [...] },
  "references": [{"id": "1", "title": "report.pdf"}],
  "sources": [...],
  "answer_images": [...],
  "answer_blocks": [...]
}
```

Pass `semantic_highlights: true` to the MCP `answer` tool to include
`highlight_phrases` in cited source chunks when highlight enrichment is enabled.


## Contexts Object

All modes return `contexts` as a `RetrievalContexts` TypedDict with three arrays. Chunks are the primary retrieval unit; entities and relationships come from the knowledge graph.

REST and Web responses never expose inline base64 page/image payloads. When a
retrieved chunk has a visual sidecar, DlightRAG projects it to
`image_url`/`thumbnail_url` routes. Python manager internals may still carry
`image_data` inside contexts so answer generation and reranking can use bounded
multimodal payloads without a second database read.

### Images Are References, Not Inline Payloads

Retrieved document images are exposed as route references, not embedded bytes:

| Interface | Image reference shape | Byte access |
|---|---|---|
| REST | `/images/{workspace}/{chunk_id}?size=thumb\|full` in `image_url`, `thumbnail_url`, and `answer_images` | Authenticated REST image route |
| Web | `/web/images/{workspace}/{chunk_id}?size=thumb\|full` in rendered HTML/SSE payloads | Same-origin Web image route |
| MCP | Same JSON `image_url`/`thumbnail_url` references as REST when a REST image route is reachable | No separate MCP binary stream today |
| SDK | `answer_images` render references; internal `contexts` may still include `image_data` | In-process caller can inspect internals, but renderers should prefer `answer_images` |

User-supplied `query_images` are different: they can arrive as data URIs, are
bounded before model use, and may be stored temporarily in session memory so a
later request can refer to `current_image_ids`.

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
| `url` | string \| null | Resolved download URL (via SourceUrlResolver) |
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
fit the retrieved-context image budget, while preserving text from mixed
text+image chunks.


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
import base64

with open("photo.png", "rb") as f:
    img_bytes = f.read()

data_uri = "data:image/png;base64," + base64.b64encode(img_bytes).decode("ascii")

result = await manager.aanswer(
    query="What does this diagram show?",
    query_images=[{"type": "image_url", "image_url": {"url": data_uri}}],
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
    "query_images": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,<base64>"}}
    ],
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
