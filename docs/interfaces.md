# Interfaces

This page is for SDK, REST, MCP, and Web consumers. It owns request and response
contracts for ingestion, jobs, retrieval, answers, contexts, sources, citations,
and multimodal payloads. Security posture lives in [security.md](security.md);
runtime retrieval behavior lives in [retrieval-answer.md](retrieval-answer.md);
configuration fields live in [configuration.md](configuration.md).

## Interface Overview

| Interface | Primary use | Ingestion behavior |
|---|---|---|
| Python SDK | In-process applications | Foreground via `corpora.ingest`; background via `corpora.start_ingest_job` |
| REST API | Web clients, services, and remote callers | Durable ingest jobs |
| MCP Server | Agent tools over stdio or streamable HTTP | Durable ingest jobs |
| Web UI | Browser upload and chat | Durable ingest jobs behind the Files panel |

Answer requests take one input contract on every channel: a query plus optional
**attachments**. Attachments are files (images, PDFs, DOCX, PPTX, XLSX, HTML, CSV)
or HTTPS references that become request-local resources read on demand for the
lifetime of one answer. REST and MCP JSON bodies carry HTTPS link descriptors;
REST multipart bodies additionally carry uploaded files; the Python SDK adds
path/bytes/url conveniences; and the Web UI uploads the same files. Attachments
never become workspace documents, never enter LightRAG storage, BM25, vectors,
or KG, and never appear on `/retrieve`. The separate `/retrieve` path keeps its
own `query_images` current-image inputs for knowledge-base visual search.

### Choosing an interface

Pick by **where the engine runs**, not by language preference:

- **Engine runs as a separate service** (Docker, a shared host, an internal
  deployment): use the **REST API**, **MCP**, or **Web UI**. Remote callers talk
  to a running server over HTTP and should not import the `dlightrag` package.
  This is the common case.
- **Engine runs inside your own process** (your application *is* the RAG service
  and owns its PostgreSQL, parser endpoint, and model providers): use the
  **Python SDK** (`Application`). This is a power-user surface — the REST
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
application = await Application.acreate(config)  # start: warms the default workspace
# per request:
await application.corpora.ingest(...)
await application.retrieval.retrieve(RetrieveRequest(...))
await application.answers.answer(AnswerRequest(...), owner_id=owner_id)
await application.aclose()  # stop
```

`DlightragConfig` ships a curated default model stack, but it still needs the
matching provider credentials to run. Supply credentials, model choices, and
provider overrides from any configuration source above.

## Ingestion

### Python SDK

```python
from dlightrag import DlightragConfig, Application
from dlightrag.services.corpora import IngestSpec

application = await Application.acreate(DlightragConfig())
try:
    # Local files or directory
    result = await application.corpora.ingest(
        "default",
        IngestSpec(source_type="local", path="./docs"),
    )

    # Azure Blob Storage
    result = await application.corpora.ingest(
        "default",
        IngestSpec(
            source_type="azure_blob",
            container_name="documents",
            prefix="reports/",  # or blob_path="reports/q1.pdf"
        ),
    )

    # AWS S3
    result = await application.corpora.ingest(
        "default",
        IngestSpec(
            source_type="s3",
            bucket="my-bucket",
            s3_region="us-east-1",  # optional; credentials come from AWS env/config/IAM
            s3_key="docs/q1.pdf",  # or prefix="docs/"
        ),
    )

    # Explicit non-blocking ingest
    job = await application.corpora.start_ingest_job(
        "default",
        IngestSpec(source_type="s3", bucket="my-bucket", prefix="docs/"),
    )
    status = await application.corpora.get_ingest_job(job["job_id"])
finally:
    await application.aclose()
```

  `CorpusAdmin` accepts local, Azure Blob, S3, and public HTTPS sources. SaaS APIs
  that require custom authorization or pagination must stage their content through
  one of those supported sources before ingestion.

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
URLs; authenticated SaaS APIs must stage content through a supported local,
Azure Blob, or S3 source. S3 credentials are read
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
| `metadata` | `object` | — | Custom ingest metadata |
REST also supports one-file multipart upload at `POST /ingest/blob`. Fields are
`file` plus optional `workspace`, `title`, `author`, and `metadata` (JSON
string). The file is staged under DlightRAG's managed input
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

Custom metadata needs no declaration. Any key passed through `metadata` on a
REST, MCP, or SDK ingest call is stored verbatim and is immediately filterable.

System metadata such as `filename`, `filename_stem`, `file_extension`,
`title`, and `author` is extracted or mapped by DlightRAG. These names are
reserved: supplying one under `metadata` is rejected rather than stored where no
filter would read it. Set `title` and `author` through their ingest parameters;
`creation_date` is the one such column a caller sets through `metadata`.

For custom metadata, filter with `filters.custom`, for example
`{"custom": {"department": "finance"}}`. Matching is case-insensitive, the same
rule the named fields use, and it is applied to the comparison rather than to
the stored value — so what you sent is what you read back. Keys are matched by
the same rule, so `Department` and `department` are one key.

`creation_date` is the document's own date and is the one built-in column a
caller sets through `metadata`, since `title` and `author` have their own ingest
parameters. Pass an ISO 8601 date or timestamp, with or without an offset; an
offset is converted to UTC and a bare value is read as UTC. Anything else is
rejected at ingest rather than stored unfiltered. Filter it with
`creation_date_from` / `creation_date_to`.

### Ingestion Response

Single-file ingestion returns the concrete file result from the unified
LightRAG path:

```json
{
  "doc_id": "file-doc-abc123",
  "source_kind": "document",
  "chunks": ["chunk-a", "chunk-b"],
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
      "parse_engine": "mineru",
      "process_options": "iteP"
    },
    {
      "doc_id": "file-doc-def456",
      "source_kind": "document",
      "chunks": ["chunk-c"],
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
  "errors_truncated": false,
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
`status` is one of `queued`, `running`, `succeeded`, `partial`, or `failed`.
`partial` means some items landed and some did not, so `result` is present but
incomplete; `succeeded` and `partial` are the two states that carry a `result`,
containing the same single-file or staged batch response shown above. At most
200 error messages are retained per job; `failed_items` remains
the authoritative failed-item count and `errors_truncated` reports whether
additional messages were omitted. REST, Web, and MCP start the job immediately and do not wait on
`ingest_timeout`. The in-process `CorpusAdmin.ingest()` convenience method starts
the same durable job, waits up to `ingest_timeout`, and returns either the
completed result or the still-running job row without cancelling it.

`POST /ingest/jobs/{job_id}/cancel` and MCP `cancel_ingest_job` stop a running
job and return the job row. Whatever the job already ingested stays ingested;
unfinished documents are parked so a later run picks them up. Both require the
`job.cancel` action.

On service
startup, recent `queued`/`running` rows are recovered automatically. Remote
prefix jobs resume from `current_window`, so completed source windows are not
downloaded again; already processed documents are still deduplicated by
LightRAG's document status and DlightRAG's content-hash guard.


## Retrieval And Answer

### Quick Reference

| Interface | `retrieve` | `answer` | Following a run |
|---|---|---|---|
| In-process Application | `application.retrieval.retrieve()` | `application.answers.answer()` | `application.answers.subscribe()` yields durable events |
| Python SDK | — | `AnswerRunClient.answer()` waits for the result | `AnswerRunClient.events()` yields reconnectable durable events |
| REST API | JSON object | HTTP 202 run descriptor | reconnectable SSE at `/answer/{run_id}/events` |
| MCP Server | JSON text | descriptor-only, returns immediately | `get_answer_run` / `cancel_answer_run` tools |
| Web UI | — | HTTP 202 run descriptor | rendered events at `/web/api/answer/{run_id}/events` |
| CLI (`scripts/cli.py`) | JSON object printed to stdout | Terminal text; `answer_blocks` image refs render as image URL lines | follows the run, then falls back to status |

### Contract Terms

| Term | Meaning |
|---|---|
| `contexts` | Evidence package. `/retrieve` returns the broader retrieved set; `/answer` returns the packed evidence that the answer model actually saw. |
| `sources` | Document-level source objects with chunks, pages, optional visual routes, and optional highlights. `/retrieve` returns all retrieved sources; `/answer` returns only cited sources. |
| `references` | Compact document-level citation summary for answers, derived from validated inline citations. |
| `answer_images` | Registry of cited visual assets available for rendering. Entries reference image routes, not inline document image bytes. |
| `answer_blocks` | Display plan for answers: markdown text blocks plus `image_ref` blocks that point into `answer_images`. |

### Durable Answer Runs

Every answer is one durable run with one identifier and one lifecycle, shared by
REST, MCP, Web, the Python application, the CLI, and evaluation. There is no
ephemeral answer mode and no `stream` request field.

`POST /answer` validates, persists, and returns **HTTP 202**:

```json
{
  "run_id": "019…",
  "status": "queued",
  "status_url": "/answer/019…",
  "events_url": "/answer/019…/events",
  "cancel_url": "/answer/019…"
}
```

| Operation | Contract |
|---|---|
| `POST /answer` | 202 with the descriptor. An idempotent replay returns 202 with the run's current status. Reusing a key with different normalized input returns 409. |
| `GET /answer/{run_id}` | `queued` / `running` / `succeeded` / `failed` / `cancelled`, whether cancellation was requested, current phase (`routing` \| `planning` \| `searching` \| `researching` \| `generating`), `durable_progress_version`, the canonical result once succeeded, and one public `error_kind` + `error_message` for a terminal failure. |
| `GET /answer/{run_id}/events` | Reconnectable SSE. Each durable sequence is the SSE `id`; resume with `Last-Event-ID` or the integer `after` query parameter. Supplying both with different values returns 400. Without a cursor, replay starts at sequence 1. A quiet run sends a comment keepalive every 10s, which consumes no sequence. |
| `DELETE /answer/{run_id}` | Requests cancellation. 200 when it completed or found a terminal transition, 202 while a running worker must still observe it. Cancelling a terminal run is an idempotent no-op. |

Durable event types are exactly `progress`, `token`, `reset`, `done`, and
`error`. `progress` carries the core phases `routing`, `planning`, `searching`,
`researching`, and `generating`. Successful `done` embeds the complete canonical
result (answer, contexts, references, sources, answer-image metadata, trace,
image descriptions); cancelled `done` carries `status="cancelled"` with no
result; `error` is used only for `failed`. Exactly one terminal event is
committed per run and the stream closes after replaying it. `reset` means a
partial draft must be cleared before regenerated output.

Idempotency uses one optional key per owner: the `Idempotency-Key` header for
REST, an `idempotency_key` argument for MCP and Python, and the browser's
`submission_id` for Web. Creation without a key always creates a new run.

Owner scope is uniform: unknown runs, pruned runs, and another owner's runs are
indistinguishable and all return 404 from status, events, and cancellation. An
authorized terminal run whose 30-day event log was trimmed returns **410** from
the events endpoint; its canonical result stays available from the status
endpoint. Stored events and results carry transport-neutral source identities;
each authenticated read projects fresh download URLs without modifying an event.

Client disconnect never cancels a run. Closing an event subscriber or cancelling
a waiting convenience call detaches that caller only; explicit run cancellation
is the sole client action that requests a terminal `cancelled`.

### Web Conversation Boundary

The authenticated browser starts from one typed `GET /web/api/bootstrap`
snapshot. It contains only the authorized workspace records and selected scope,
the primary Files target, answer-attachment limits, and the current image-input
capability; it never contains bearer credentials or edge identity tokens. A
Vite-owned static document renders `<dl-app>` immediately; the Lit root stays
inert until this snapshot succeeds and exposes an explicit retry on failure.

Browser navigation has two explicit page routes: `/web/` is an unpersisted New
Chat, while `/web/conversations/{conversation_id}` identifies one durable
conversation. The URL, not local storage, is the active-conversation authority;
direct reload and browser Back/Forward therefore reopen the same owner-scoped
history. A missing, malformed, or foreign id stays on its URL and renders the
same unavailable state. Navigation detaches any local SSE reader without
cancelling its run, closes conversation-scoped Sources/Report panels, and keeps
the workspace-scoped Files panel. An unsent draft guards click, programmatic,
and browser-history navigation through the same confirmation.

The Web-only conversation lifecycle is server-owned and principal-scoped. The
browser creates, lists, selects, renames, deletes, and reloads conversations
through `/web/api/conversations`; it sends the optional `conversation_id`, the
current query, current answer attachments, and the selected search workspaces to
`/web/api/answer`. Omitting `conversation_id` denotes the first submission from
`/web/`: the server derives a stable UUID from the owner-wide submission key and
creates the conversation, turn, uploaded artifacts, and durable run in one
transaction. Admission failure leaves no empty conversation, and retrying the
same submission resolves the same conversation. Conversation IDs are
server-generated UUIDs and are never credentials. History
and attachment reads always filter by both the authenticated principal and
conversation ID, so another principal receives the same 404 as a missing
conversation.

`POST /web/api/answer` creates a core run and returns its 202 descriptor, including
the authoritative conversation summary; the browser then subscribes to its own
owner-scoped `GET /web/api/answer/{run_id}/events`. That
stream follows the same durable event log as the REST stream, with the same
sequence, `Last-Event-ID` resume, 410-on-trim, and detach semantics, and differs
only in projection: a browser `done` frame carries rendered presentation
(`html`, `answer`, `answer_images`, optional `primary_report` handle), not the
canonical result payload REST serves. The Web thread shows the terminal answer;
a Primary Report opens in the document panel. The run and its conversation turn are inserted in one transaction before
the 202 response, so no subscriber, finalizer, or reconnect commits history
afterwards. Disconnecting the browser closes that subscriber only, and
reconnecting resumes from the durable event sequence. Conversation reads return
every linked turn in order: queued and running turns are pending entries carrying
`answer_run_id`, status, and cancellation state, so a reloaded tab can
resubscribe without remembering the original 202. Failed and cancelled turns stay
visible with their public terminal error until their run is pruned; only
succeeded turns become model history.

Answer attachment admission completes before `/web/api/answer` returns. Unsupported,
empty, unsafe-name, per-attachment oversized, and over-count uploads return HTTP
4xx before the run is accepted: a request exceeding `answer.max_attachments`,
`answer.max_attachment_bytes`, or `answer.max_total_attachment_bytes` is rejected
with a stable limit message. Once a run is accepted, a resource that cannot be
read produces a classified terminal `error` event; the answer does not silently
drop evidence.

Web conversations retain up to 100 complete turns with 30-day inactivity
retention. Uploaded answer attachments are stored once as owner-scoped
content-addressed blobs owned by the run, not by a Web-owned table. Historical
attachments are re-registered lazily as request-local resources when a follow-up
is answered, newest first up to the available attachment-count limit. An
attachment-bearing conversation therefore remains on the research path, and
browser thumbnails are derived on demand. There is no parsed-chunk table and no
vector cache: the research path reads each attachment fresh from its stored
bytes. Manual delete, TTL pruning, and window trimming delete the linked runs,
which cascades their events and references and releases blobs no surviving run
still references.

`AnswerOrchestrator` owns every answer. Callers set `mode` to `auto`, `fast`, or
`research` (omitted means `auto`). Capability resolves a Valid Mode Set; routing
writes a durable Resolved Mode. Fast plans, retrieves, and synthesizes with no
Agent Session. Research runs `AgentLoop` until the model emits no tool call; the
last silent turn is the answer. Optional `artifacts/report.md` publishes as a
Primary Report handle.
Evidence-producing Exa result URLs are registered as opaque request-local
resources, so the same `read` tool can deepen a search result without
accepting an arbitrary model-supplied URL. Reading performs no login, cookie
session, or Playwright interaction; callers provide authenticated bytes or
screenshots as attachments when those are required.
After generation, citation/source/media finalization is deterministic. Provider
and resource lifetimes are request-local; nothing is shared across turns except
the durable attachment blobs the run owns.

REST, MCP, and Python answer/retrieve calls remain stateless. Answer calls
accept an optional caller-supplied `history` of prior `role`/`content` turns for
multi-turn follow-ups, but DlightRAG never persists it: the client owns
conversation storage and re-sends the turns it wants on each request. They do not
accept a server `conversation_id`; historical files are re-sent as current
`attachments` when a follow-up depends on them. Public answer calls take no
`query_images`; that field belongs to `/retrieve` only.

The REST API uses resource-oriented verbs (for example `POST /workspaces`,
`DELETE /workspaces/{workspace}`), while the internal `/web/api/*` surface serves
the browser (for example `POST /web/api/workspaces/create`,
`POST /web/api/workspaces/delete`) and answers with whatever that page needs —
temporarily an HTML fragment for legacy panels, otherwise JSON for state the
browser owns. These browser routes have no compatibility aliases at their old
`/web/*` paths. Prefer REST or the SDK for programmatic access.

Image support is a deployment capability, not a per-request negotiation, so callers
discover it up front. REST `GET /health` returns `answer_image_capability`
(`status`, `effective_max_images`, `configured_ceiling`, `model`); the MCP
`get_capabilities` tool returns the same summary; and the Python SDK exposes it as
`await application.answers.capabilities()`. When `status` is not `supported`, attaching
image resources is rejected fail-closed with a stable `error_kind`
(`CURRENT_IMAGES_UNSUPPORTED` or `ANSWER_IMAGE_CAPABILITY_UNKNOWN`): REST returns
HTTP 422 (or a classified SSE `error` event carrying `error_kind` when streaming),
MCP returns the error text, and the SDK raises `AnswerImageError`.

`GET /health` is liveness only: it answers from `ApplicationHealth` in-process
state (degraded state, startup warnings, and the projected
`answer_image_capability`) plus configured storage backend names, stays HTTP 200
when the process is degraded, and never touches PostgreSQL, so an unauthenticated
poll loop cannot become database load. Unauthenticated `GET /ready` is the
traffic-readiness probe: HTTP 200 only after `ApplicationHealth` is marked ready
and its injected readiness adapter confirms the DlightRAG domain session is
writable. A reader additionally proves its corpus session is still read-only and
still resolves the corpus. `ApplicationHealth` single-flights and memoizes that
adapter verdict for two seconds; ready, degraded, and closed transitions
invalidate the memo so a startup or schema transition is never answered from a
stale verdict. The status route imports neither the application nor PostgreSQL. Any
failed condition returns a minimal HTTP 503 with a fixed detail string and no
exception text.

Error responses are `{detail, error_type, error_kind?}`. `error_type` is one of
`validation` (400/413/422), `auth` (401/403), `unavailable` (503),
`configuration` (a server tool-composition failure, carrying
`error_kind: invalid_tool_configuration`), or `internal`. A durable schema that
is incompatible with the running revision answers HTTP 503 `unavailable` with a
fixed detail and no schema detail. Terminal run failures carry one stable
`error_kind`: an answer-input kind (`CURRENT_IMAGES_UNSUPPORTED`,
`CURRENT_IMAGE_LIMIT_EXCEEDED`, `CURRENT_DOCUMENT_PARSE_FAILED`,
`ANSWER_INPUT_OVERFLOW`, `ANSWER_IMAGE_CAPABILITY_UNKNOWN`,
`MODEL_CAPABILITY_UNAVAILABLE`, `ANSWER_RESOURCE_INVALID`),
`invalid_tool_configuration`, `unsupported_answer_mode`, `routing_failed`,
`tool_contract_changed`, `run_abandoned` when a run exceeds its crash-recovery
bound, `run_execution_failed` when Runtime catches an unclassified runtime or
executor failure outside the Answer taxonomy, or `ANSWER_STREAM_FAILED`. There
are no checkpoint error kinds.

A saturated service does not refuse an answer: accepted runs queue until an
execution slot frees or the caller cancels the run, with no queue timeout, queue
depth cap, or capacity error. An answer request whose attachments exceed
`answer.max_total_attachment_bytes` returns HTTP 413 before the route buffers the
body. Generic request-rate, connection, and volumetric limits are an ingress
responsibility, not an application one — see
[security.md](security.md#ingress-responsibilities).

The Web shell defaults answer scope to `Search in: All authorized workspaces`.
That authorization-relative multi-workspace selection is independent from
`Files in`, which continues to name one workspace for file management and
ingestion.

### Python SDK

```python
# Retrieve: contexts only, no LLM answer
from dlightrag.access import DEPLOYMENT_OWNER_ID
from dlightrag.services.answers import AnswerRequest
from dlightrag.services.retrieval import RetrieveRequest

result = await application.retrieval.retrieve(
  RetrieveRequest(
    query="What are the key findings?",
    workspaces=("default",),
  )
)
result.contexts  # RetrievalContexts: {"chunks": [...], "entities": [...], "relationships": [...]}
result.sources  # client-safe source projections

# Query a concrete, already-authorized workspace set
all_contexts = await application.retrieval.retrieve(
  RetrieveRequest(
    query="What are the key findings?",
    workspaces=("finance", "legal"),
  )
)

# Answer: contexts + LLM-generated answer
result = await application.answers.answer(
  AnswerRequest(
    query="What are the key findings?",
    workspaces=("default",),
    semantic_highlights=True,  # optional; default false outside Web
  ),
  owner_id=DEPLOYMENT_OWNER_ID,
)
result.answer  # "The key findings are... [1-1] [2-3]"
result.contexts  # same structure as retrieve, packed to what the answer model saw
result.references  # validated cited documents, derived from inline citations
result.answer_images  # cited visual assets available for rendering
result.answer_blocks  # markdown/image_ref blocks for structured display

# Answer with attachments: files or HTTPS references become request-local
# resources read on demand. The SDK builds ResourceInput objects from the
# AnswerAttachment path/bytes/url conveniences owned by the resource domain.
from dlightrag.answer.resources.attachments import (
    AnswerAttachment,
    resource_inputs_from_attachments,
)

result = await application.answers.answer(
  AnswerRequest(
    query="Summarize the attached report and figure.",
    workspaces=("default",),
    resources=tuple(
      resource_inputs_from_attachments(
        [
          AnswerAttachment.from_path("report.pdf"),
          AnswerAttachment.from_url("https://cdn.example.com/figure.png"),
        ]
      )
    ),
    ),
  owner_id=DEPLOYMENT_OWNER_ID,
)

# Streaming answer
async for event in application.answers.answer_stream(
  AnswerRequest(query="What are the key findings?", workspaces=("default",)),
  owner_id=DEPLOYMENT_OWNER_ID,
):
    print(event.event_type, event.payload)
```

**Parameters**:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | required | Search query |
| `mode` | `auto \| fast \| research` | `auto` | Answer Mode. Omitted hashes as `auto`. |
| `workspace` | `str \| None` | config default | Target workspace |
| `workspaces` | `list[str] \| None` | `None` | Federated search across multiple workspaces |
| `all_workspaces` | `bool` | `false` | Query every workspace visible to the current caller. For REST/MCP this is the existing `workspace.query`-authorized set; for the in-process SDK it is every registered workspace. Mutually exclusive with a non-empty `workspace`/`workspaces` selection. |
| `top_k` | `int \| None` | config default | LightRAG KG breadth: entities in local retrieval and relationships in global retrieval. |
| `chunk_top_k` | `int \| None` | config default | Explicit chunk/visual candidates fetched for `/retrieve` and before `/answer` packing. Maps to LightRAG `QueryParam.chunk_top_k`, not `QueryParam.top_k`. |
| `bm25_query` | `str \| None` | `None` | `retrieve` only. Optional workspace BM25 query override; when omitted, RetrievalPlanner supplies lexical terms or retrieval uses the main query. REST and MCP inputs are capped at 1,024 characters. |
| `query_images` | `list[QueryImage]` | `None` | `retrieve` only. Current-request OpenAI-style `image_url` blocks for knowledge-base visual search: described by the VLM for semantic/BM25 retrieval and embedded directly for visual retrieval. Capped at 3. Answer calls do not accept this field. |
| `attachments` | `list[AnswerAttachmentLink]` (SDK: `list[AnswerAttachment]` via `resources`) | `None` | `/answer` only. Files or HTTPS references read as request-local resources for one answer. JSON/MCP bodies carry HTTPS link descriptors (`{url, filename?}`, HTTPS-only, no credentials); REST multipart adds uploaded files; the SDK uses `AnswerAttachment.from_path/from_bytes/from_url`. Bounded by `answer.max_attachments` (6), `answer.max_attachment_bytes` (100 MiB), and `answer.max_total_attachment_bytes` (128 MiB). |
| `semantic_highlights` | `bool` | `false` | `/answer` only. When true and `citations.highlights.enabled` is true, fills `sources[].chunks[].highlight_phrases` with answer-aware phrase highlights. |
| `history` | `list[ConversationMessage] \| None` | `None` | `/answer` only. Optional caller-supplied prior turns as `role` (`user`/`assistant`) + `content` messages for multi-turn follow-ups. Stateless: never persisted, so the caller re-sends the turns it wants each request. Fast retrieval uses it for standalone-query rewrite and final generation uses the same bounded turns; research control and final generation see it, while agent-selected KB queries stay unchanged. Capped at 100 messages. |
| `filters` | `MetadataFilter \| None` | `None` | Structured metadata filter (also auto-detected from query); supports `filename`, `file_extension`, `title`, `author`, `creation_date_from`/`creation_date_to`, and any `custom` key |

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

# Create an answer run; the 202 descriptor carries its status and events URLs
curl -X POST http://localhost:8100/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "key findings", "semantic_highlights": true}'

# Read the run's status, and its canonical result once it succeeded
curl http://localhost:8100/answer/$RUN_ID

# Follow its durable events, resuming after the last sequence you saw
curl -N -H "Last-Event-ID: 12" http://localhost:8100/answer/$RUN_ID/events

# Request cancellation (idempotent)
curl -X DELETE http://localhost:8100/answer/$RUN_ID

# Answer with an HTTPS attachment reference (JSON body)
curl -X POST http://localhost:8100/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "summarize this", "attachments": [{"url": "https://cdn.example.com/report.pdf", "filename": "report.pdf"}]}'

# Answer with uploaded files (multipart): exactly one JSON `request` part plus
# repeated `attachments` file parts; uploaded files and JSON links may mix.
curl -X POST http://localhost:8100/answer \
  -F 'request={"query": "summarize this"};type=application/json' \
  -F 'attachments=@report.pdf' \
  -F 'attachments=@figure.png'

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

**Canonical answer result** (the `result` field of a succeeded run's status
response, and the payload embedded in its terminal `done` event):

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

**Durable run events** (`GET /answer/{run_id}/events`): each frame carries the
durable sequence as the SSE `id` and the event type as `event:`.

| Event | Payload | Description |
|---|---|---|
| `progress` | `{"phase": "routing" \| "planning" \| "searching" \| "researching" \| "generating"}` | Last-writer-wins phase notification, not a monotonic history |
| `token` | `{"text": "..."}` | Coalesced answer text batch (repeats) |
| `reset` | `{}` | Clear the partial draft; regenerated output follows |
| `done` | `{"status": "succeeded", "result": {...}}` or `{"status": "cancelled"}` | Terminal; succeeded embeds the complete canonical result |
| `error` | `{"kind": "...", "message": "..."}` | Terminal failure with one public error kind |

```
id: 1
event: progress
data: {"phase":"searching"}

id: 2
event: token
data: {"text":"The key findings"}

id: 3
event: token
data: {"text":" are..."}

id: 4
event: done
data: {"status":"succeeded","result":{"answer":"The key findings are...","sources":[...],"contexts":{...}}}
```

Intermediate contexts are never published: research may still change them, so
clients receive the authoritative contexts and every other piece of metadata in
the successful `done` event and from the status endpoint. A quiet run keeps its
connection alive with SSE comments, which consume no sequence number.

`bm25_enabled` reports whether the workspace PostgreSQL BM25 lane participated.
If that lane fails while semantic retrieval succeeds, retrieval continues with
semantic results and trace includes `bm25_error_type`. Conversely,
`lightrag_error_type` records semantic-lane degradation to BM25-only results.

REST uses the same answer and context shapes, while its HTTP adapter projects
each source's authorized `download_url`. Transport-neutral application/MCP payloads
keep `download_url` null.

`all` is authorization-relative, not deployment-global. If 14 workspaces are
registered and the current caller may query 10, `all_workspaces: true` queries
those 10. `None` and `[]` remain omission; omitting both selectors still uses
the configured default workspace. A caller with no queryable workspaces receives
an authorization error. Ingest remains single-workspace and does not support
broadcast ingestion. The strings `"*"` and `"all"` are ordinary workspace names,
not selector aliases.

Web follows the same durable run events as every other transport and always
attempts semantic highlights when citation highlighting is enabled. Previews are
derived from token text and highlights are requested after `done`; neither is
stored.

### MCP Server

MCP 2 tools return a `CallToolResult`: `structuredContent` contains the typed
JSON payload, while the first text content block carries equivalent formatted
JSON for clients that consume text-only tool results. Expected validation or
authorization failures set `isError: true`; protocol-level `MCPError` responses
remain JSON-RPC errors. The server exposes these tools:

`retrieve`, `answer`, `get_answer_run`, `cancel_answer_run`, `ingest`,
`get_ingest_job`, `cancel_ingest_job`, `list_files`, `delete_files`,
`list_workspaces`, `create_workspace`, `delete_workspace`, `list_memories`,
`forget_memory`, and `get_capabilities`.

MCP `answer` is deliberately descriptor-only: it creates the durable run and
returns immediately rather than holding one tool call open for a run that may
take tens of minutes. Poll `get_answer_run` for status and, once the run
succeeded, its canonical result; `cancel_answer_run` requests cancellation.
Answer payloads keep `sources` at top level:

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

All modes return `contexts` as a `RetrievalContexts` mapping with three arrays.
Each row is a `ContextRow` dictionary. Chunks are the primary retrieval unit;
entities and relationships come from the knowledge graph.

REST and Web responses never expose inline base64 page/image payloads. When a
retrieved chunk has a visual sidecar, DlightRAG projects it to
`image_url`/`thumbnail_url` routes. Python application internals may still carry
`image_data` inside contexts so answer generation and reranking can use bounded
multimodal payloads without a second database read.

### Images Are References, Not Inline Payloads

Retrieved document images are exposed as route references, not embedded bytes:

| Interface | Image reference shape | Byte access |
|---|---|---|
| REST | `/images/{workspace}/{chunk_id}?size=thumb\|full` in `image_url`, `thumbnail_url`, and `answer_images` | Authenticated REST image route |
| Web | `/web/api/images/{workspace}/{chunk_id}?size=thumb\|full` in rendered HTML/SSE payloads | Same-origin Web image route |
| MCP | Same JSON `image_url`/`thumbnail_url` references as REST when a REST image route is reachable | No separate MCP binary stream today |
| SDK | `answer_images` render references; internal `contexts` may still include `image_data` | In-process caller can inspect internals, but renderers should prefer `answer_images` |

User-supplied `/retrieve` `query_images` are different: they can arrive as data
URIs and are bounded before model use. Answer attachments are also different:
they are read as request-local resources and never returned as durable image
identifiers. Public answer/retrieve requests do not persist either.

```python
from dlightrag_rag.retrieval import ContextRow, RetrievalContexts
```

### chunks

```json
{
  "chunk_id": "abc123",
  "reference_id": "1",
  "file_path": "report.pdf",
  "content": "Page text content...",
  "page_number": 2,
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
| `page_number` | int \| null | no | Optional **1-based** display page number |
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
stable provenance and is returned consistently by REST, MCP, SDK, and Web. The
Web source panel renders it in a new tab when it is a credential-free public
HTTP(S) URL. HTTP adapters separately project the internal document ID and
source workspace to an authorized `download_url`, then look up the locator
server-side; raw storage locators and workspace-routing fields are never public.
REST links use `/files/raw/{document_id}`; Web links use the Web-authenticated
`/web/api/files/raw/{document_id}`. Transport-neutral SDK/MCP payloads leave
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
      "page_number": 2,
      "content": "First 200 characters of content...",
      "image_url": null,
      "thumbnail_url": null
    },
    {
      "chunk_id": "def456",
      "chunk_idx": 2,
      "page_number": 5,
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
| `source_uri` | string | Stable source identity; may use a connector-specific scheme. Public HTTP(S) values are linkable provenance. |
| `download_url` | string \| null | Authorized HTTP download route for retained files; null when no download permission/route exists. |
| `cited_chunk_ids` | list \| null | Cited chunk IDs for answer responses; null when returning all retrieved sources |
| `chunks` | list | Chunk snippets in citation-index order |

Each **chunk snippet** within a source:

| Field | Type | Description |
|---|---|---|
| `chunk_id` | string | Unique chunk identifier |
| `chunk_idx` | int | 1-based position within this source; matches `[ref_id-chunk_idx]` citations |
| `page_number` | int \| null | Optional 1-based display page number |
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
4. Use `page_number` when present for human page navigation; it does not
  participate in citation validity or semantic highlight grounding
5. Use a public HTTPS `source_uri` to visit external provenance, or
  `download_url` for an authorized retained file; use `image_url`/
  `thumbnail_url` for retrieved visual chunks


## Multimodal Queries

There are two distinct visual paths, and they belong to different endpoints.

**`/answer` uses attachments.** Send images (and documents) as answer
attachments; they become request-local resources that the orchestrator reads and
inspects on demand:

```python
# Python SDK — answer over an attached image
from dlightrag.answer.resources.attachments import (
    AnswerAttachment,
    resource_inputs_from_attachments,
)

result = await application.answers.answer(
  AnswerRequest(
    query="What does this diagram show?",
    workspaces=("default",),
    resources=tuple(
      resource_inputs_from_attachments(
        [AnswerAttachment.from_path("photo.png")]
      )
    ),
    ),
  owner_id=DEPLOYMENT_OWNER_ID,
)
```

```bash
# REST API — attach the image as a multipart upload
curl -X POST http://localhost:8100/answer \
  -F 'request={"query": "What does this diagram show?"};type=application/json' \
  -F 'attachments=@photo.png'
```

**`/retrieve` uses `query_images`.** For knowledge-base visual search, send
current-request images to `/retrieve`; they are VLM-described for semantic/BM25
retrieval and embedded directly for visual retrieval, without persistence:

```bash
# REST API — retrieve with a base64-encoded query image
curl -X POST http://localhost:8100/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "diagrams like this one",
    "query_images": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,<base64>"}}
    ]
  }'
```

Answer-model image previews are quality-preserving. Budgeted JPEG, PNG, and WebP
payloads pass through unchanged; oversized images are recompressed only down to
the configured `answer.image_min_quality` and `answer.image_min_px` floors. If
an image still cannot fit, DlightRAG skips it rather than sending a degraded
preview that could hurt visual understanding. Pure visual retrieved chunks
whose image is skipped are also removed from the answer context; later text or
sendable visual chunks in the retrieved set remain available to the answer
model.
