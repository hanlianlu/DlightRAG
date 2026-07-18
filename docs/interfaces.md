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

### Choosing an interface

Pick by **where the engine runs**, not by language preference:

- **Engine runs as a separate service** (Docker, a shared host, an internal
  deployment): use the **REST API**, **MCP**, or **Web UI**. Remote callers talk
  to a running server over HTTP and should not import the `dlightrag` package.
  This is the common case.
- **Engine runs inside your own process** (your application *is* the RAG service
  and owns its PostgreSQL, parser endpoint, and model providers): use the
  **Python SDK** (`RAGServiceManager`). This is a power-user surface — the REST
  and MCP servers are themselves built on it.

### Configuration is one-time; the runtime surface is small

Configuring models, providers, credentials, PostgreSQL, and the parser is a
**one-time setup step**, not something callers repeat per request. Values are
resolved from (highest precedence first): constructor args › environment
variables › `.env` › `config.yaml` › defaults (see
[configuration.md](configuration.md)).

- **Deployment / repo:** edit `config.yaml` (app settings) + `.env` (secrets), or
  run `uv run prerequisite_setup.py` to generate both.
- **Programmatic:** build `DlightragConfig(...)` in code and pass overrides
  directly; no files required.

Once configured, the SDK runtime is a small create-once / call / close lifecycle:

```python
manager = await RAGServiceManager.acreate(config)  # start: warms the default workspace
# per request:
await manager.aingest(...)   # or aretrieve(...) / aanswer(...) / aanswer_stream(...)
await manager.aclose()       # stop
```

`DlightragConfig` ships a curated default model stack, but it still needs the
matching provider credentials to run. Supply credentials, model choices, and
provider overrides from any configuration source above.

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
                download_uri=f"https://cdn.example.com/assets/{asset['id']}.pdf",
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


manager = await RAGServiceManager.acreate(DlightragConfig())
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
            s3_region="us-east-1",   # optional; credentials come from AWS env/config/IAM
            s3_key="docs/q1.pdf",    # or prefix="docs/"
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
    status = await manager.aget_ingest_job(job["job_id"])
finally:
    await manager.aclose()
```

DlightRAG calls `aiter_documents()` to discover `SourceDocument` descriptors and
`amaterialize_document(document, destination)` to write each document into parser
staging without loading the whole object into memory. Ingest-call `metadata` is
the batch default; `SourceDocument.metadata` overlays it for that document.
`SourceDocument.source_uri` is stable identity. For a non-retained connector,
`SourceDocument.download_uri` is the durable original-byte locator. A connector
that keeps this mapping outside its descriptors may omit the field and pass a
`download_uri_for_key` callback to `aingest_source()` instead. Per-document
`SourceDocument.download_uri` takes precedence over that callback. If retention
is disabled and neither is available, that document is rejected before
`amaterialize_document()` runs; DlightRAG never silently retains it.

### REST API

```bash
curl -X POST http://localhost:8100/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type": "local", "path": "docs"}'

curl -X POST http://localhost:8100/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type": "url", "url": "https://api.bynder.com/docs/getting-started", "filename": "getting-started.html"}'

# Queryless URL batch.
curl -X POST http://localhost:8100/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type": "url", "urls": ["https://cdn.example.com/a.pdf", "https://cdn.example.com/b.pdf"], "download_uris": ["https://cdn.example.com/a.pdf", "https://cdn.example.com/b.pdf"]}'

# Signed fetch retained by DlightRAG.
curl -X POST http://localhost:8100/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type": "url", "url": "https://fetch.example.com/download?signature=secret", "filename": "asset.pdf", "source_uri": "bynder://asset/asset-1", "retain_source_file": true}'

# Signed fetch with a separate queryless durable locator.
curl -X POST http://localhost:8100/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type": "url", "url": "https://fetch.example.com/download?signature=secret", "filename": "asset.pdf", "source_uri": "bynder://asset/asset-1", "download_uri": "https://cdn.example.com/assets/asset-1.pdf"}'
```

All ingest operations are represented internally as jobs. REST returns `202 Accepted`
with the job object; MCP `ingest` returns the same job object as a tool result.
Poll `GET /ingest/jobs/{job_id}` or call MCP `get_ingest_job` for progress and
the final result. `source_type="url"` is intentionally limited to public or signed HTTPS
URLs; authenticated SaaS APIs should fetch through a caller-owned SDK
`AsyncDataSource` connector and use `aingest_source()`. S3 credentials are read
from the standard AWS credential chain (environment, shared config, or IAM
role); ingest payloads do not carry access keys.

For URL ingest, `url`/`urls` are fetch endpoints. `source_uri`/`source_uris` are
stable identities and never act as download addresses. `download_uri` or
`download_uris` supplies a supported durable locator for the original bytes.
Queryless public HTTPS fetch URLs can be used implicitly; query- or
fragment-bearing signed URLs cannot. Signed fetches therefore require either
`retain_source_file=true` or a separate queryless locator. Invalid documents
are rejected before fetch/materialization, and DlightRAG never silently changes
the requested retention policy.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `source_type` | `string` | yes | `local`, `azure_blob`, `s3`, `url` |
| `path` | `string` | local | File or directory path relative to DlightRAG's managed `input_dir/<workspace>` |
| `container_name` | `string` | azure_blob | Blob container name |
| `blob_path` | `string` | — | Specific blob (mutually exclusive with `prefix`) |
| `prefix` | `string` | — | Blob/key prefix filter for `azure_blob`/`s3` batches; mutually exclusive with `blob_path`/`s3_key`. Omit (or pass `""`) to ingest the whole container/bucket. |
| `bucket` | `string` | s3 | S3 bucket name. With neither `s3_key` nor `prefix`, ingests the whole bucket. |
| `s3_region` | `string` | — | Optional S3 region for this ingest; falls back to the `s3_region` config setting or the AWS SDK environment/config defaults |
| `s3_key` | `string` | — | S3 object key for a single object; mutually exclusive with `prefix` |
| `url` | `string` | url | Single public or signed HTTPS document URL |
| `urls` | `list[string]` | url | Multiple public or signed HTTPS document URLs; mutually exclusive with `url` |
| `filename` | `string` | — | Parser filename for a single URL, useful when the URL path has no extension |
| `source_uri` | `string` | — | Stable stored source URI for a single URL, independent of the signed fetch URL |
| `source_uris` | `list[string]` | — | Stable stored source URIs for URL batches; must match `urls` length |
| `download_uri` | `string` | — | Durable S3, Azure, or queryless public HTTPS locator for one URL; independent of the fetch endpoint |
| `download_uris` | `list[string]` | — | Durable locators for a URL batch; must match `urls` length |
| `documents` | `list[object]` | — | Explicit document manifest. Local documents use `path`, S3/Azure use `key`, URL documents use `url` and may set `source_uri`/`download_uri`; per-document metadata overlays request metadata. |
| `retain_source_file` | `boolean` | — | Per-call remote source retention override. `true` keeps fetched bytes as the download source; `false` requires a durable remote locator. |
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
`get_ingest_job` with the returned `job_id` to read progress. For URL sources,
the tool description distinguishes fetch `url`/`urls`, stable
`source_uri`/`source_uris`, and durable `download_uri`/`download_uris`; a signed
fetch must use retention or a separate queryless locator under the same
fail-closed contract as REST.

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

`GET /ingest/jobs/{job_id}` and MCP `get_ingest_job` return the same job row.
`status` is one of `queued`, `running`, `succeeded`, or `failed`. When the job
succeeds, `result` contains the same single-file or staged batch response shown
above. REST, Web, and MCP start the job immediately and do not wait on
`ingest_timeout`. The SDK convenience method `RAGServiceManager.aingest()` starts
the same durable job, waits up to `ingest_timeout`, and returns either the
completed result or the still-running job row without cancelling it. On service
startup, recent `queued`/`running` rows are recovered automatically. Remote
prefix jobs resume from `current_window`, so completed source windows are not
downloaded again; already processed documents are still deduplicated by
LightRAG's document status and DlightRAG's content-hash guard.


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

### Web Conversation Boundary

The Web-only conversation lifecycle is server-owned and principal-scoped. The
browser creates, lists, selects, renames, deletes, and reloads conversations
through `/web/conversations`; it sends only `conversation_id`, the current query,
current images, and the selected search workspaces to `/web/answer`. Conversation
IDs are server-generated UUIDs and are never credentials. History and image
reads always filter by both the authenticated principal and conversation ID, so
another principal receives the same 404 as a missing conversation.

Web conversations retain up to 100 complete turns with 30-day inactivity
retention. Current Web images are admitted using `query_images.max_current_images`
and `query_images.max_upload_bytes`; the defaults are three images and 15 MiB per
decoded image, and the browser upload entry is gated by the query-role answer
model's discovered image capability (unsupported or unknown disables uploads).
Current-turn images always have priority. Referenced historical images are
resolved by the planner and share one adaptive `answer.max_images` transport
budget with current and RAG images.

REST, MCP, and Python answer/retrieve calls remain stateless. Answer calls
accept an optional caller-supplied `history` of prior `role`/`content` turns for
multi-turn follow-ups, but DlightRAG never persists it: the client owns
conversation storage and re-sends the turns it wants on each request. They do
not accept a server `conversation_id` or durable historical-image IDs, and
`query_images` belong only to the current request and are never persisted.

The REST API uses resource-oriented verbs (for example `POST /workspaces`,
`DELETE /workspaces/{workspace}`), while the `/web/*` surface uses htmx action
endpoints that return HTML fragments (for example `POST /web/workspaces/create`,
`POST /web/workspaces/delete`). This split is intentional: HTML forms cannot
issue `DELETE`, and Web responses are markup rather than JSON. Prefer REST or the
SDK for programmatic access.

Image support is a deployment capability, not a per-request negotiation, so callers
discover it up front. REST `GET /health` returns `answer_image_capability`
(`status`, `effective_max_images`, `configured_ceiling`, `model`); the MCP
`get_capabilities` tool returns the same summary; and the Python SDK exposes it as
`manager.answer_image_capability`. When `status` is not `supported`, attaching
`query_images` is rejected fail-closed with a stable `error_kind`
(`CURRENT_IMAGES_UNSUPPORTED` or `ANSWER_IMAGE_CAPABILITY_UNKNOWN`): REST returns
HTTP 400 (or a classified SSE `error` event carrying `error_kind` when streaming),
MCP returns the error text, and the SDK raises `AnswerImageError`.

The Web shell defaults answer scope to `Search in: All authorized workspaces`.
That authorization-relative multi-workspace selection is independent from
`Files in`, which continues to name one workspace for file management and
ingestion.

### Python SDK

```python
# Retrieve: contexts only, no LLM answer
result = await manager.aretrieve(query="What are the key findings?")
result.answer     # None
result.contexts   # RetrievalContexts: {"chunks": [...], "entities": [...], "relationships": [...]}

# Query every registered workspace from the trusted in-process SDK
all_contexts = await manager.aretrieve(
    query="What are the key findings?",
    all_workspaces=True,
)

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
| `all_workspaces` | `bool` | `false` | Query every workspace visible to the current caller. For REST/MCP this is the existing `workspace.query`-authorized set; for the in-process SDK it is every registered workspace. Mutually exclusive with a non-empty `workspace`/`workspaces` selection. |
| `top_k` | `int \| None` | config default | LightRAG KG breadth: entities in local retrieval and relationships in global retrieval. |
| `chunk_top_k` | `int \| None` | config default | Explicit chunk/visual candidates fetched for `/retrieve` and before `/answer` packing. Maps to LightRAG `QueryParam.chunk_top_k`, not `QueryParam.top_k`. |
| `answer_context_top_k` | `int \| None` | `answer.context_top_k` | `/answer` only. Maximum chunks included in the final answer prompt after image-budget packing and backfill. |
| `stream` | `bool` | `true` for REST `/answer` | `true` returns SSE; pass `false` to opt into one JSON response |
| `query_images` | `list[QueryImage]` | `None` | Current-request OpenAI-style `image_url` blocks. They are described by the VLM for semantic/BM25 retrieval, embedded directly for visual retrieval, and bounded before being sent to the answer LLM. Capped at 3. |
| `semantic_highlights` | `bool` | `false` | `/answer` only. When true and `citations.highlights.enabled` is true, fills `sources[].chunks[].highlight_phrases` with answer-aware phrase highlights. |
| `history` | `list[ConversationMessage] \| None` | `None` | `/answer` only. Optional caller-supplied prior turns as `role` (`user`/`assistant`) + `content` messages for multi-turn follow-ups. Stateless: never persisted, so the caller re-sends the turns it wants each request. Folded into the planner's standalone-query rewrite and answer generation. Capped at 100 messages. |
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

# Retrieve across every workspace authorized for this caller
curl -X POST http://localhost:8100/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "key findings", "all_workspaces": true}'

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
  "image_descriptions": ["Image 1: a line chart about revenue"]
}
```

**SSE streaming** (`stream: true`): events are newline-delimited JSON.

| Event | Payload | Description |
|---|---|---|
| `context` | `{type, data}` | Full contexts, sent first |
| `token` | `{type, content}` | LLM answer token (repeats) |
| `sources` | `{type, data}` | Validated cited sources, after all tokens and before done. Includes `highlight_phrases` when `semantic_highlights` is true and enrichment succeeds. |
| `trace` | `{type, data}` | Retrieval trace counts and planner/filter decisions |
| `image_meta` | `{type, image_descriptions}` | VLM descriptions for current-request images |
| `done` | `{type, answer, answer_images, answer_blocks}` | Stream complete; `answer` is the final normalized answer body after citation validation |
| `error` | `{type, message}` | Error mid-stream |

```
data: {"type":"context","data":{"chunks":[...],"entities":[...],"relationships":[...]}}

data: {"type":"token","content":"The key findings"}

data: {"type":"token","content":" are..."}

data: {"type":"sources","data":[{"id":"1","title":"report.pdf","source_uri":"local://default/report.pdf","download_url":"/files/raw/doc-a1b2c3?workspace=default","chunks":[...]}]}

data: {"type":"trace","data":{"bm25_enabled":true,"fused_chunk_count":8}}

data: {"type":"image_meta","image_descriptions":["Image 1: a line chart"]}

data: {"type":"done","answer":"The key findings are...","answer_images":[],"answer_blocks":[{"type":"markdown","text":"The key findings are..."}]}
```

REST uses the same answer and context shapes, while its HTTP adapter projects
each source's authorized `download_url`. Transport-neutral manager/MCP payloads
keep `download_url` null.

`all` is authorization-relative, not deployment-global. If 14 workspaces are
registered and the current caller may query 10, `all_workspaces: true` queries
those 10. `None` and `[]` remain omission; omitting both selectors still uses
the configured default workspace. A caller with no queryable workspaces receives
an authorization error. Ingest remains single-workspace and does not support
broadcast ingestion. The strings `"*"` and `"all"` are ordinary workspace names,
not selector aliases.

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

Pass `all_workspaces: true` to MCP `retrieve` or `answer` to query every
workspace visible to the current MCP caller:

```json
{"query": "key findings", "all_workspaces": true}
```


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

User-supplied `query_images` are different: they can arrive as data URIs and are
bounded before model use. Public answer/retrieve requests do not persist them or
return durable image identifiers.

```python
from dlightrag.core.retrieval.protocols import RetrievalContexts, ChunkContext, EntityContext, RelationshipContext
```

### chunks

```json
{
  "chunk_id": "abc123",
  "reference_id": "1",
  "file_path": "report.pdf",
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
| `file_path` | string | yes | Display-only source basename; never use it as provenance or a download locator |
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
matches `[ref_id-chunk_idx]` markers instead of page sorting. `source_uri` is
stable provenance. HTTP adapters project the internal document ID and source
workspace to an authorized `download_url`, then look up the locator server-side;
raw storage locators and workspace-routing fields are never public.
REST links use `/files/raw/{document_id}`; Web links use the Web-authenticated
`/web/files/raw/{document_id}`. Transport-neutral SDK/MCP payloads leave
`download_url` null.

```json
{
  "id": "1",
  "title": "report.pdf",
  "type": "file",
  "source_uri": "local://default/docs/report.pdf",
  "download_url": "/files/raw/doc-a1b2c3?workspace=default",
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
| `type` | string \| null | File type |
| `source_uri` | string | Stable source identity; may use a connector-specific scheme |
| `download_url` | string \| null | Authorized HTTP download route; null on transport-neutral SDK/MCP payloads and required by Web rendering |
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
with identity, authorized downloads, chunks, pages, images, and optional
highlights.

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
6. Use the source's `download_url` for the original file; use
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
    ]
  }'
```

`query_images` is the user-facing current-request image path. REST, MCP, CLI,
and Python calls ask the VLM for concise semantic descriptions, embed the raw
image for direct visual retrieval, and send a bounded image preview to the
answer model without persistence. The Web route additionally commits validated
current-turn images with the complete successful turn in its principal-scoped
conversation.

Answer-model previews are quality-preserving. Budgeted JPEG, PNG, and WebP
payloads pass through unchanged; oversized images are recompressed only down to
the configured `answer.image_min_quality` and `answer.image_min_px` floors. If
an image still cannot fit, DlightRAG skips it rather than sending a degraded
preview that could hurt visual understanding. Pure visual retrieved chunks
whose image is skipped are also removed from the answer context; later text or
sendable visual chunks in the retrieved set remain available to the answer
model.
