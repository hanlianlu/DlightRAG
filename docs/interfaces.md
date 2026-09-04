# Interfaces

This document owns public REST, MCP, Web, and in-process request/response
contracts. Configuration belongs in [Configuration](configuration.md), runtime
behavior in [Retrieval and Answer](retrieval-answer.md), durable lifecycle rules
in [Durable Answer Runs](durable-answer-runs.md), and authorization in
[Security](security.md).

## Choosing An Interface

| Interface | Use when | Ingestion |
|---|---|---|
| REST | DlightRAG runs as a service | Durable jobs |
| MCP | An agent connects over stdio or streamable HTTP | Durable jobs |
| Web | A browser user uploads and chats | Durable jobs |
| In-process Application | Your process owns DlightRAG and its dependencies | Foreground convenience or durable jobs |

Remote clients should not import `dlightrag`; use REST, MCP, or Web. Configure
models, PostgreSQL, credentials, and the parser once, then reuse the service or
Application instance.

```python
from dlightrag import DlightragConfig, create_application

application = await create_application(DlightragConfig())
try:
    ...
finally:
    await application.aclose()
```

## Common Answer Terms

| Term | Meaning |
|---|---|
| `contexts` | Retrieved evidence. `/answer` returns only what the answer model saw. |
| `sources` | Document-level groups containing chunks, pages, and media routes. `/answer` returns only cited sources. |
| `references` | Compact cited-document projection derived from validated inline citations. |
| `evidence_images` | Cited visual evidence available for rendering. |
| `parts` | Ordered Markdown, Artifact, and explicitly inline Evidence Image parts. |
| `usage` | Root, child, and inclusive provider usage when available. |
| `evidence` | Counts of admitted chunks, entities, relationships, and cited sources. |
| `parent_run_id`, `continuation_kind` | Follow-up/fork lineage. |

Answer **attachments** are files or HTTP(S) references used only by one answer.
They never become workspace documents or appear in `/retrieve`. The separate
`query_images` input belongs only to `/retrieve` and performs knowledge-base
visual search.

## Ingestion

### REST

`POST /ingest` accepts JSON and returns `202 Accepted` with a durable ingest job.
`POST /ingest/blob` accepts one multipart `file` plus optional `workspace`,
`title`, `author`, and JSON-string `metadata` fields.

```bash
curl -X POST http://localhost:8100/ingest \
  -H 'Content-Type: application/json' \
  -d '{"source_type":"local","path":"docs"}'

curl -X POST http://localhost:8100/ingest \
  -H 'Content-Type: application/json' \
  -d '{"source_type":"s3","bucket":"my-bucket","prefix":"docs/"}'

curl -X POST http://localhost:8100/ingest \
  -H 'Content-Type: application/json' \
  -d '{"source_type":"url","url":"https://cdn.example.com/report.pdf"}'
```

| Field | Required for | Description |
|---|---|---|
| `source_type` | all | `local`, `azure_blob`, `s3`, or `url` |
| `path` | local | File/directory relative to managed `input_dir/<workspace>` |
| `container_name` | Azure | Container name |
| `blob_path` | Azure single | Object path; exclusive with `prefix` |
| `bucket` | S3 | Bucket name |
| `s3_key` | S3 single | Object key; exclusive with `prefix` |
| `s3_region` | S3 optional | Per-request region override |
| `prefix` | Azure/S3 batch | Prefix; omit or use `""` for the whole container/bucket |
| `url` / `urls` | URL | One URL or a batch; mutually exclusive |
| `filename` | URL optional | Parser filename when the URL path lacks an extension |
| `source_uri` / `source_uris` | URL optional | Stable provenance, independent of the fetch URL |
| `download_uri` / `download_uris` | URL optional | Durable S3, Azure, or queryless public HTTPS locator |
| `documents` | optional | Explicit manifest with per-document metadata |
| `retain_source_file` | remote optional | Keep fetched bytes; otherwise a durable locator is required |
| `replace` | optional | Purge an existing document before enqueueing replacement |
| `workspace` | optional | Target, default `default` |
| `title`, `author` | optional | Built-in document metadata |
| `metadata` | optional | Custom metadata object |

`source_uri` is identity, never a download address. A signed URL containing a
query or fragment is not durable; use `retain_source_file: true` or supply a
separate queryless locator. S3 uses the standard AWS credential chain. Payloads
never carry access keys.

Per-document metadata uses a manifest:

```json
{
  "source_type": "s3",
  "bucket": "my-bucket",
  "metadata": {"source_system": "s3-prod"},
  "documents": [
    {"key": "docs/a.pdf", "metadata": {"department": "legal"}}
  ]
}
```

### In-process And MCP

```python
from dlightrag.application.corpus_admin import IngestSpec

result = await application.corpora.ingest(
    "default", IngestSpec(source_type="local", path="./docs")
)
job = await application.corpora.start_ingest_job(
    "default", IngestSpec(source_type="s3", bucket="my-bucket", prefix="docs/")
)
status = await application.corpora.get_ingest_job(job["job_id"])
```

MCP `ingest` exposes the REST arguments and returns the same job object. Use
`get_ingest_job` to poll and `cancel_ingest_job` to stop unfinished work.

### Jobs And Results

A job has `job_id`, `workspace`, `source_type`, `status`, item counters,
`current_window`, bounded `errors`, `errors_truncated`, the accepted `request`,
optional `result`, and `status_url`. Status is `queued`, `running`, `succeeded`,
`partial`, or `failed`. `succeeded` and `partial` carry a result; `partial` means
some items landed. At most 200 error messages are retained, while
`failed_items` remains authoritative.

`GET /ingest/jobs/{job_id}` returns the current row.
`POST /ingest/jobs/{job_id}/cancel` requests cancellation and leaves completed
documents intact. Recent queued/running rows recover at startup; remote-prefix
jobs resume at `current_window`.

`CorpusAdmin.ingest()` waits up to `corpus.ingestion.timeout` and returns either
the completed result or the still-running job without cancelling it. REST, MCP,
and Web never wait on that timeout.

Single-file result:

```json
{
  "doc_id": "file-doc-abc123",
  "source_kind": "document",
  "chunks": ["chunk-a", "chunk-b"],
  "parse_engine": "mineru",
  "process_options": "iteP"
}
```

Batch result:

```json
{"processed": 2, "errors": [], "results": [{"doc_id": "file-doc-abc123"}]}
```

Unsupported files use `source_kind: "skipped"` with `status` and `reason`.
Batch-level failures raise; per-file failures appear in `errors`.

### Metadata

Custom keys need no declaration and are immediately filterable through
`filters.custom`. Matching is case-insensitive without rewriting stored values
or keys. Built-ins such as `filename`, `filename_stem`, `file_extension`,
`title`, and `author` are reserved. Set title/author through their fields.
`creation_date` is the one built-in accepted under `metadata`; it must be ISO
8601 and is filtered with `creation_date_from`/`creation_date_to`.

`POST /metadata/search` returns document IDs ordered by `doc_id`, with `limit`
(1–100, default 50) and a signed opaque `cursor`. The cursor is bound to the
workspace, request filters, and filename match mode. Invalid or cross-workspace
cursors return 422 before storage access.

## Retrieval And Answer

### Request Fields

| Field | Endpoint | Default | Description |
|---|---|---|---|
| `query` | both | required | Query text |
| `mode` | answer | `auto` | `auto`, `fast`, or `research` |
| `workspace` | both | configured default | One workspace |
| `workspaces` | both | unset | Explicit federated workspace list |
| `all_workspaces` | both | `false` | Every workspace visible to the caller; exclusive with explicit selection |
| `top_k` | both | config | KG entity/relationship breadth |
| `chunk_top_k` | both | config | Text/visual candidate breadth |
| `bm25_query` | retrieve | query-derived | Optional lexical override; REST/MCP cap it at 1,024 characters |
| `query_images` | retrieve | none | Up to three current images for visual search |
| `attachments` | answer | none | Link descriptors or multipart files used only by this answer |
| `semantic_highlights` | answer | `false` | Add answer-aware source highlights when globally enabled |
| `history` | answer | none | Up to 100 caller-supplied user/assistant messages |
| `filters` | both | none | Built-in and custom metadata filters |

`all_workspaces` is authorization-relative. `None` and `[]` mean omission;
`"*"` and `"all"` are ordinary workspace names. Ingestion remains
single-workspace.

### REST

```bash
curl -X POST http://localhost:8100/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"key findings","all_workspaces":true}'

# Accept a durable answer run.
curl -X POST http://localhost:8100/answer \
  -H 'Content-Type: application/json' \
  -d '{"query":"key findings","semantic_highlights":true}'

# Read status/result and follow events.
curl http://localhost:8100/answer/$RUN_ID
curl -N -H 'Last-Event-ID: 12' http://localhost:8100/answer/$RUN_ID/events

# Attach an HTTPS resource.
curl -X POST http://localhost:8100/answer \
  -H 'Content-Type: application/json' \
  -d '{"query":"summarize this","attachments":[{"url":"https://cdn.example.com/report.pdf","filename":"report.pdf"}]}'

# Upload resources: one JSON request part plus repeated attachments.
curl -X POST http://localhost:8100/answer \
  -F 'request={"query":"summarize this"};type=application/json' \
  -F 'attachments=@report.pdf' \
  -F 'attachments=@figure.png'
```

There is no public `/query` route and no ephemeral answer mode. `/retrieve`
returns contexts immediately. `POST /answer` always persists a run and returns
HTTP 202:

```json
{
  "run_id": "019…",
  "status": "queued",
  "status_url": "/answer/019…",
  "events_url": "/answer/019…/events",
  "cancel_url": "/answer/019…"
}
```

### Answer Run Endpoints

| Operation | Contract |
|---|---|
| `POST /answer` | Accept a run. Optional `Idempotency-Key` replays the same normalized request; conflicting reuse returns 409. |
| `GET /answer` | List this owner's runs oldest-first; `after` + `limit` (1–100, default 50). |
| `GET /answer/{run_id}` | Return status, cancellation flag, phase, progress version, terminal error, and canonical result when succeeded. |
| `GET /answer/{run_id}/events` | Reconnectable SSE; resume with `Last-Event-ID` or integer `after`. |
| `GET /answer/{run_id}/artifacts` | List stored-result Artifact descriptors/outcome; 409 before a result exists. |
| `GET /answer/{run_id}/artifacts/{resource_id}` | Stream Artifact bytes with Range support; `download=true` forces attachment. |
| `GET /answer/{run_id}/artifacts/{resource_id}/presentation` | Project an available Markdown Artifact as typed `AnswerResponse`, including that Artifact's validated citation sources. |
| `DELETE /answer/{run_id}` | Idempotent cancellation; 200 if terminal, otherwise 202. |
| `POST /answer/{run_id}/steer` | Queue an instruction for live Research. |
| `POST /answer/{run_id}/follow-up` | Create a child run using the selected terminal answer as context. |
| `POST /answer/{run_id}/fork` | Create a sibling branch from accepted context. |
| `POST /answer/{run_id}/resume` | Return current state before event reattachment. |
| `GET /answer/{run_id}/transcript` | Return bounded canonical ancestry. |
| `GET /answer/{run_id}/children` | Newest-first keyset page (`limit` 1–100, default 50). |

Run status is `queued`, `running`, `succeeded`, `failed`, or `cancelled`; phase is
`routing`, `planning`, `searching`, `researching`, or `generating`. Unknown,
pruned, foreign-owner runs all return 404. A retained run whose event log was
trimmed returns 410 from the event endpoint; status/result remains readable.
Disconnecting a client never cancels a run.

SSE event types are exactly:

| Event | Payload |
|---|---|
| `progress` | Current `phase` |
| `token` | Coalesced answer text |
| `reset` | Invalidate all previously streamed draft text before continuation, replacement, or terminal projection |
| `tool_start`, `tool_progress`, `tool_end` | Safe metadata only; no raw stdout/stderr |
| `done` | Terminal success with full `result`, or cancellation without one |
| `error` | Terminal `{kind, message}` failure |

Each durable sequence is the SSE `id`. Supplying conflicting header/query
cursors returns 400. Without a cursor, replay starts at sequence 1. Ten-second
comment keepalives consume no sequence. Exactly one terminal event is committed.

Canonical successful result:

```json
{
  "answer": "The key findings are... [1-1] [2-3]",
  "contexts": {"chunks": [], "entities": [], "relationships": []},
  "references": [{"id": "1", "title": "report.pdf"}],
  "sources": [],
  "evidence_images": [],
  "parts": [{"type": "markdown", "text": "The key findings are... [1-1]"}],
  "artifacts": [],
  "artifact_outcome": {"status": "complete", "issues": []},
  "usage": {},
  "evidence": {},
  "trace": {},
  "image_descriptions": []
}
```

`trace.bm25_enabled` reports lexical-lane participation. If one retrieval lane
fails and the other succeeds, the result continues with `bm25_error_type` or
`lightrag_error_type`; `lightrag_mix_chunk_count` records the pre-fusion
LightRAG count.

### In-process Application

```python
from dlightrag.application.access import DEPLOYMENT_OWNER_ID
from dlightrag.application.answer_runs import AnswerRequest
from dlightrag.application.retrieval import RetrieveRequest

retrieved = await application.retrieval.retrieve(
    RetrieveRequest(query="What changed?", workspaces=("default",))
)

answer = await application.answers.answer(
    AnswerRequest(
        query="What changed?",
        workspaces=("default",),
        semantic_highlights=True,
    ),
    owner_id=DEPLOYMENT_OWNER_ID,
)

async for event in application.answers.answer_stream(
    AnswerRequest(query="What changed?", workspaces=("default",)),
    owner_id=DEPLOYMENT_OWNER_ID,
):
    print(event.event_type, event.payload)
```

Files/URLs become `ResourceInput` values through
`AnswerAttachment.from_path/from_bytes/from_url` and
`resource_inputs_from_attachments`. `AnswerService` also exposes status,
subscription, cancellation, steering, continuation, transcript, and roster
methods. There is no separate public Python SDK for remote callers.

### MCP Server

MCP `answer` returns only the durable descriptor; poll `get_answer_run` for the
canonical result. A tool result puts typed JSON in `structuredContent` and
formatted equivalent JSON in its first text block. Expected validation or
authorization failures set `isError: true`; protocol failures remain JSON-RPC
errors.

Registered public tool names are:

- query/run: `retrieve`, `answer`, `get_answer_run`, `cancel_answer_run`,
  `steer_answer_run`, `follow_up_answer_run`, `fork_answer_run`,
  `resume_answer_run`, `get_answer_transcript`, `list_answer_children`,
  `list_answer_runs`, `list_answer_artifacts`, `read_answer_artifact`
- corpus: `list_workspaces`, `get_capabilities`,
  `get_workspace_storage_status`, `create_workspace`, `delete_workspace`,
  `ingest`, `get_ingest_job`, `cancel_ingest_job`, `list_files`, `delete_files`
- model catalogue: `get_model_catalogue`, `upsert_model_catalogue_entry`,
  `remove_model_catalogue_entry`
- memory: `list_memories`, `remember_memory`, `forget_memory`,
  `undo_memory_change`, `get_memory_settings`, `set_memory_enabled`,
  `clear_memory`

### Web

Web routes under `/web/api/*` are browser contracts, not compatibility aliases
for REST. `GET /web/api/bootstrap` returns authorized workspace state, Files
target, attachment limits, and image capability—never bearer or edge tokens.
Route families cover:

- `/conversations`, `/conversations/{id}/history`, and
  `/runs/{run_id}/attachments/{ordinal}` (plus `/thumbnail`);
- `/answer`, submission reconciliation, status/resume/steer/children,
  follow-up/fork/cancel, Artifacts/presentation, and events; and
- Files/upload/ingest status, workspaces, images, Memory, and model catalogue.

`/web/` is unpersisted New Chat;
`/web/conversations/{conversation_id}` selects a durable owner-scoped
conversation. The URL is authoritative for reload and browser history.

`POST /web/api/answer` accepts an optional conversation ID, query, attachments,
and search workspaces. It returns HTTP 202 with canonical `{conversation, turn}`.
For a first submission, the server creates conversation, turn, blobs, and run in
one transaction from the owner-scoped `submission_id`. On an ambiguous result,
use `GET /web/api/answer-submissions/{submission_id}`; the browser must not
blindly repeat the POST.

The Web event stream follows the same durable sequence as REST but projects a
typed `AnswerPresentation` (`answer_text`, `parts`, `sources`,
`evidence_images`, `artifacts`, and `artifact_outcome`). Conversation history
uses the same shape. Pending, failed, and cancelled turns remain visible;
only succeeded turns become model history.

History defaults to the newest 40 turns and accepts a signed cursor plus a limit
up to 100. Attachments are owner-scoped, content-addressed run blobs and are
re-registered lazily for follow-ups. Count, per-file, and total-byte limits are
validated before acceptance; read failures after acceptance produce a terminal
error rather than silent omission. Lifecycle details are centralized in
[Durable Answer Runs](durable-answer-runs.md).

## Contexts

`contexts` always contains `chunks`, `entities`, and `relationships`. Public
REST/Web responses use image routes rather than inline base64. In-process
internals may carry bounded `image_data` for model use.

### Chunk

```json
{
  "chunk_id": "abc123",
  "reference_id": "1",
  "file_path": "report.pdf",
  "content": "Page text...",
  "page_number": 2,
  "image_url": "/images/default/abc123?size=full",
  "thumbnail_url": "/images/default/abc123?size=thumb",
  "image_mime_type": "image/png",
  "relevance_score": 0.87
}
```

| Field | Meaning |
|---|---|
| `chunk_id` | Unique chunk ID |
| `reference_id` | Document-level citation ID |
| `file_path` | Display basename, not provenance or download authority |
| `content` | Chunk text |
| `page_number` | Optional 1-based display page |
| `image_url`, `thumbnail_url`, `image_mime_type` | Optional visual route metadata |
| `relevance_score` | Optional 0–1 rerank score |
| `metadata` | Extra metadata |
| `_workspace` | Source workspace for federated retrieval |

### Entity And Relationship

Entity rows contain `entity_name`, `entity_type`, `description`, `source_id`,
and optional `reference_id`. Relationship rows contain `src_id`, `tgt_id`,
`description`, `source_id`, and optional `reference_id`. `source_id` is a
comma-separated list of supporting chunk IDs.

## Sources

A source groups one document's chunks in citation-index order:

```json
{
  "id": "1",
  "title": "report.pdf",
  "type": "file",
  "source_uri": "local://default/docs/report.pdf",
  "download_url": "/files/raw/doc-a1b2c3?workspace=default",
  "cited_chunk_ids": ["abc123"],
  "chunks": [{
    "chunk_id": "abc123",
    "chunk_idx": 1,
    "page_number": 2,
    "content": "Page text...",
    "image_url": null,
    "thumbnail_url": null,
    "highlight_phrases": null
  }]
}
```

`source_uri` is stable provenance. `download_url` is an authorized projection
(`/files/raw/{document_id}` for REST,
`/web/api/files/raw/{document_id}` for Web); MCP transport-neutral payloads
leave it null. `retrieve` returns all retrieved sources. `answer` returns cited
sources only and sets `cited_chunk_ids`.

## Citations

`references` is the compact `{id, title}` projection of validated cited sources.
Inline citations accept:

| Format | Meaning |
|---|---|
| `[1-2]` | Document/reference 1, its second chunk (1-based) |
| `[3]` | Document/reference 3 |

Resolve `[1-2]` by finding source `id: "1"`, then chunk `chunk_idx: 2`.
`page_number` helps navigation but does not affect citation validity.

Visual bytes are read through authenticated routes:

| Interface | Reference |
|---|---|
| REST | `/images/{workspace}/{chunk_id}?size=thumb|full` |
| Web | `/web/api/images/{workspace}/{chunk_id}?size=thumb|full` |
| MCP | REST-style URL when a reachable REST route exists; no MCP binary stream |
| Application | Render references; internals may also expose `image_data` |

## Multimodal Inputs

Use answer attachments for question-local documents/images:

```bash
curl -X POST http://localhost:8100/answer \
  -F 'request={"query":"What does this show?"};type=application/json' \
  -F 'attachments=@photo.png'
```

Use `query_images` only for visual retrieval against the knowledge base:

```json
{
  "query": "diagrams like this",
  "query_images": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,<base64>"}}
  ]
}
```

Image support is a deployment capability. Discover it through REST
`GET /health`, MCP `get_capabilities`, or
`await application.answers.capabilities()`. Unsupported/unknown image input
fails closed with `CURRENT_IMAGES_UNSUPPORTED` or
`ANSWER_IMAGE_CAPABILITY_UNKNOWN`.

## Workspace And File Management

| Route | Contract |
|---|---|
| `GET /workspaces` | Page the authorized workspace catalogue. |
| `POST /workspaces` | Create an empty workspace (201; duplicate 409). |
| `DELETE /workspaces/{workspace}` | Reset/delete one workspace; supports `keep_files` and `dry_run`. |
| `GET /workspaces/{workspace}/storage` | Read operator storage/promotion state. |
| `POST /reset` | Reset one workspace while retaining its registry identity. |
| `GET /files` | Page processed files for one workspace. |
| `DELETE /files` | Delete by paths/names; supports `dry_run`. |
| `GET /files/failed` | Page failed documents. |
| `POST /files/retry` | Re-ingest failed documents from stored source metadata. |
| `GET /files/raw/{document_id:path}` | Stream or redirect one authorized source. |
| `POST /metadata/search` | Page matching document IDs. |
| `GET /metadata/{doc_id}` | Read one document's metadata. |
| `POST /metadata/{doc_id}` | Merge nonempty custom metadata. |

REST uses resource-oriented operations:

```bash
curl -X POST http://localhost:8100/workspaces \
  -H 'Content-Type: application/json' \
  -d '{"workspace":"Research Notes"}'

curl -X DELETE 'http://localhost:8100/workspaces/research_notes?keep_files=false'
```

`GET /workspaces` orders by workspace ID and pages with `limit` (default 50,
maximum 100) plus a signed cursor. Access filtering happens after catalog
paging. The response contains `workspaces`, `records`, and `next_cursor`. MCP
`list_workspaces` returns only the first 50 plus `has_more`.

`DELETE /files` supports `dry_run: true`. Workspace reset reports
`ingest_jobs_cancelled` and `ingest_jobs_deleted`; dry-run reports zero and does
not mutate jobs. Web Files uses a workspace-bound signed keyset cursor, defaults
to 50 files (maximum 100), and orders by `updated_at DESC, id ASC`.

## Model Catalogue And Profile Memory

| REST route | Contract |
|---|---|
| `GET /models/catalogue` | Effective runtime overlay with revision. |
| `PUT /models/catalogue` | Upsert one complete endpoint profile under optimistic revision; requires `model_catalogue.write`. |
| `DELETE /models/catalogue` | Remove one overlay entry under optimistic revision; requires `model_catalogue.write`. |
| `GET /memory` | Newest-first active records with `limit` 1–100 and signed cursor. |
| `POST /memory` | Remember one owner-scoped Profile Memory record. |
| `DELETE /memory/{memory_id}` | Forget a record. |
| `POST /memory/changes/{change_id}/undo` | Apply the compensating undo. |
| `GET|PUT /memory/settings` | Read/change the owner capability switch. |
| `POST /memory/clear` | Clear owner records; returns 204. |

Memory cursors are signed and owner-independent as tokens; owner scope remains
an authenticated query predicate. Invalid cursors return 422 before storage.
When Memory is disabled, mutation/recall operations are unavailable except
reading/changing the setting.

## Health And Errors

`GET /health` is liveness: it returns in-process health, startup warnings,
storage backend names, and `answer_image_capability` without touching
PostgreSQL. Degraded state remains HTTP 200. `GET /ready` checks traffic
readiness and the injected database adapter, returning only fixed-detail 503
errors. Readiness checks are single-flighted and memoized for two seconds.

General errors are `{detail, error_type, error_kind?}` where `error_type` is
`validation`, `auth`, `unavailable`, `configuration`, or `internal`. Stable
answer error kinds are:

- `CURRENT_IMAGES_UNSUPPORTED`, `CURRENT_IMAGE_LIMIT_EXCEEDED`,
  `CURRENT_DOCUMENT_PARSE_FAILED`, `ANSWER_IMAGE_CAPABILITY_UNKNOWN`,
  `ANSWER_INPUT_OVERFLOW`, `MODEL_CAPABILITY_UNAVAILABLE`,
  `unsupported_resource_capability`, and `ANSWER_RESOURCE_INVALID`;
- `invalid_tool_configuration`, `unsupported_answer_mode`, `routing_failed`,
  `tool_contract_changed`, `run_abandoned`, and `run_execution_failed`; and
- `ANSWER_STREAM_FAILED`.

Internal exception text and schema detail are not public.

Accepted answer runs queue when saturated; there is no application queue timeout
or capacity rejection. Attachment total-byte overflow returns HTTP 413 before
buffering. Generic rate, connection, and volumetric controls belong at ingress;
see [Security](security.md#ingress-responsibilities).
