# Interfaces

This document owns public REST, MCP, Web, and in-process request/response
contracts. Configuration belongs in [Configuration](configuration.md), runtime
behavior in [Retrieval and Answer](retrieval-answer.md), durable lifecycle rules
in [RunRuntime and durable query execution](run-runtime.md), and authorization in
[Security](security.md).

## Choosing An Interface

| Interface | Use when | Ingestion |
|---|---|---|
| REST | An application or script integrates knowledge, research, or authorized management capabilities | Durable Corpus Mutation Runs |
| MCP | An external agent manages knowledge and delegates answers or research over stdio or streamable HTTP | Durable Corpus Mutation Runs |
| Web | A browser user works with knowledge, research, personal settings, and authorized administration | Durable Corpus Mutation Runs |
| In-process Application | Your process owns DlightRAG and its dependencies | Durable Corpus Mutation Runs |

Remote clients should not import `dlightrag`; use REST, MCP, or Web. Configure
models, PostgreSQL, credentials, and the parser once, then reuse the service or
Application instance.

The interfaces share Application services and authorization, while exposing
operations suited to their callers. Web's `/web/api/*` contracts support the
first-party experience. Public REST also supports custom clients, explicit
child-agent supervision, owner-scoped Memory management, and authorized service
administration. MCP exposes knowledge tasks and answer-level interaction; it
does not mirror every REST route or browser action.

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
| `evidence_images` | Visual evidence available for rendering: a cited visual chunk, or a Resource the Run itself holds and the Answer places. A corpus image carries `workspace` and `chunk_id` and is addressed under them; a held image carries only its address, which a transport publishes only where it serves run resources -- the Web does, and cookie-free REST and MCP calls leave `url` empty rather than inventing one. |
| `parts` | One complete Markdown document followed by its actual Artifact/Evidence Image placements. Each placement carries its parsed `target` and zero-based `slot`; these replace occurrences inside the document, not additional trailing content. |
| `artifact_bindings` | Validated parsed destination → resource id bindings for this document. Available and unavailable resources share this resolution path. Each Artifact descriptor carries its own document's bindings. |
| `usage` | Root, child, and inclusive provider usage when available. |
| `evidence` | Counts of admitted chunks, entities, relationships, and cited sources. |
| `attestation` | Per-Artifact record derived from the Run's own durable records: the event log says which Tool calls ran, crossed with the Session transcript by `call_id` so only a call the log shows executed supplies its full arguments: `status` (`attested`, `unattested`, `unknown`), the `skill`, the declared `verifier`, a `reason` when not attested, the `basis` an attested record rests on, and `required_reads_missing` (evidence beside the verdict; `null` means the records cannot settle it). `attested` means a command recorded in this Run named the Skill's verifier; it is evidence the verifier ran, not proof it ran over this file. The REST and Web projections stamp it on each Artifact; MCP `get_run` does not stamp it today. The record never withholds an Artifact: a Run whose records name an unattested verifier is served with the record beside its files, and a verdict that names no verifier (no Skill, a Skill that declares none, an unreadable declaration, a log that cannot name the Skill) is served the same way. |
| `parent_run_id`, `continuation_kind` | Follow-up/fork lineage. A continuation injects no history: it derives it from its Session branch point. |

A Resource handle a later turn may name includes a Published Artifact of the same Agent
Session (`artifact-<hash of its path>`), because publication registers it as a Resource
([ADR 0023](adr/0023-a-published-product-is-a-resource.md)); the handle is what the
attaching Tool returns and what the compaction summary keeps naming.

Render the complete Markdown once and place resources at its actual link/image
tokens. Do not concatenate `parts` as alternating source fragments or scan the
source for resource-looking strings: code, math, link titles, and unused reference
definitions are inert. Repeated resource occurrences have distinct slots. An
`artifact:` stable id can also resolve directly against the returned descriptors;
the binding map covers path references and explicitly unavailable targets. The
map records terminal validation and grants no publication authority.

Images inside link labels are actual references too; links quoted inside image
alt text are not. The browser keeps a valid linked image passive inside its
outer link. An unavailable image or a resource needing its own controls replaces
the whole outer link with one resource placement, avoiding nested controls.

The plain-text CLI prints the complete Markdown once, then a separately labelled
Resources catalogue with each resource identity listed once. It does not attempt
an interactive inline layout or concatenate placement metadata into the prose.


Answer **attachments** are files or HTTP(S) references used only by one answer.
They never become workspace documents or appear in `/retrieve`. The separate
`query_images` input belongs only to `/retrieve` and performs knowledge-base
visual search.

An upload is admitted only when an Answer Run can read it
([admission rule](resource-reading.md#registration-and-acquisition)). REST and
the Web refuse any other upload at acceptance with HTTP 422 and
`error_kind: UNSUPPORTED_ATTACHMENT_TYPE`, naming its type; nothing is stored. A
link is not an upload: it is admitted whatever its address names and read under
the public Web acquisition contract. MCP accepts links only.

## Ingestion

### REST

`POST /runs/corpus/ingest` and `/runs/corpus/replace` accept JSON and
return `202 Accepted` with a durable `corpus_mutation` Run descriptor.
Single and batch multipart uploads use the corresponding `/upload` and
`/uploads` suffixes. Every surface stages uploads through the same Application
bounds: each file under `corpus.ingestion.max_upload_bytes`, the whole request
under `interfaces.max_upload_size_mb`, and at most 100 files. A single upload may
provide a 64-character hexadecimal
`content_sha256`; mismatch rejects acceptance and deletes the Run-exclusive
staging directory. Every REST mutation requires `Idempotency-Key` (422 without
it).

```bash
curl -X POST http://localhost:8100/runs/corpus/ingest \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: ingest-docs-1' \
  -d '{"source_type":"local","path":"docs"}'

curl -X POST http://localhost:8100/runs/corpus/ingest \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: ingest-s3-1' \
  -d '{"source_type":"s3","bucket":"my-bucket","prefix":"docs/"}'

curl -X POST http://localhost:8100/runs/corpus/ingest \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: ingest-url-1' \
  -d '{"source_type":"url","url":"https://cdn.example.com/report.pdf"}'
```

| Field | Required for | Description |
|---|---|---|
| `source_type` | all | `local`, `azure_blob`, `s3`, or `url` |
| `path` | local | File/directory relative to managed `input_dir/<workspace>`; exclusive with `documents` |
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

A remote document that cannot be fetched fails alone, however briefly its
server refused, dropped, timed out, or answered with an error. The Run's result
names it and says why (`the source could not be downloaded`), and the Run fails
with it rather than deferring; nothing is reported as a dependency outage in
`GET /health`. A remote document server is the caller's source, not a service
this deployment runs, and a Run that deferred on it would fetch every other
document again when it resumed. LightRAG never saw the document, so a retry Run
cannot reach it: submit it again once its source is back. A retry Run downloads
a remote document again unless its bytes were retained, and so does an ingest
that resumes after LightRAG processed a document DlightRAG had not yet
finished. When that download fails, the document fails the same way, and it
stays retryable: retry it once its source is back.

A `local` source is copied into the Run's own stage before it is accepted, so
later edits under `input_dir` never change what the Run ingests. `path` may name
a file or a folder; each `documents` entry must name a file. The copy never
follows a link: among the entries ingestion would read, a symlink or anything
but a regular file or folder refuses the source as the caller's to fix, even
when the link is swapped in while the copy runs. A folder is listed before
anything is copied; it may hold at most 100 files that ingestion would read,
nest folders at most 32 deep, and hold at most 10,000 entries in all. Entries
ingestion skips are not copied, and a link among them is ignored, but they
count toward the 10,000. A folder with nothing to ingest is refused too.

A document's parser input takes its file name in the Workspace's corpus
directory, so a name that directory keeps for itself cannot be a document's:
one starting with a dot (Run stages, temporary copies) or one of its folders
(`__parsed__`, `__local_sources__`, `__remote_sources__`). A
folder listing and an upload skip such an entry, with everything below it
(`.DS_Store`, `.git/`), and an upload left with nothing to ingest is refused
with 400; so is a local file of such a name, or a manifest entry naming one.

LightRAG names a document by its file name alone, so one request may not hold
two files with the same name (a parser hint such as `.[mineru]` aside), even
from different folders or uploads: it is refused with 400 before anything is
staged.

`input_dir` (`<working_dir>/inputs`) belongs to operators: DlightRAG reads local
sources from `input_dir/<workspace>` and never writes there. Every file
DlightRAG writes for a Workspace (Run stages, parser inputs, the copies
documents keep of their sources, and LightRAG's parser sidecars) lives in its
own `<working_dir>/corpus/<workspace>`, which is LightRAG's input directory.
Corpus Reset and Workspace Delete clear that directory; they never delete
operators' files in `input_dir`.

A Run's stage (`corpus/<workspace>/.runs/<run>/sources`) holds each accepted
file as `<n>/<file name>`, and the Run records exactly that list, in order, with
each file's size and digest. The Run ingests only the files it lists; nothing
scans a folder. Right before a file is enqueued it is copied to
`corpus/<workspace>/<file name>`, the one place LightRAG looks a document up,
with its page margin when it is an image (`corpus.ingestion.image_margin`);
LightRAG archives that copy, and it is removed once LightRAG has processed the
document. The document keeps the file as it was given in
`corpus/<workspace>/__local_sources__/<file name>`, which its download and a
retry read. A listed file missing or no longer of its recorded size when the
Run starts fails it as `corpus_source_unavailable` before any upstream effect,
and so does one that cannot be copied.

A stage belongs to its Run alone. The Run removes it once it ends: it succeeds,
fails, or is cancelled before its handoff. It stays while the Run is deferred or
waiting for repair, and when its worker stops or loses the Run to another, which
resumes from it. Cancelling a queued Run removes its stage at once, a full reset
removes the stage of the Run it supersedes, and Workspace Delete those of the
Runs it cancels. Corpus Reset and Workspace Delete clear the rest of the
Workspace's corpus directory but never another Run's stage, so an upload still
being staged is never cut short. The hourly Run maintenance of a `writer`
removes any stage left behind: one whose Run has ended, and one with no Run (its
request stopped before the Run was accepted) a day after the request began.

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
from dlightrag.application.corpus_admin.service import IngestSpec

creation = await application.corpus_mutations.create_ingest(
    workspace="default",
    spec=IngestSpec(source_type="s3", bucket="my-bucket", prefix="docs/"),
    submitted_by="operator-1",
    idempotency_key="ingest-s3-1",
)
run = await application.runs.get(owner_id="default", run_id=creation.run.run_id)
```

MCP `ingest` exposes the REST source arguments and returns the common Run
descriptor. MCP also exposes `retry_files` and `delete_files`; use `get_run` and
`cancel_run` for their shared lifecycle. Corpus reset, Workspace Delete, and explicit operator
repair/resume are available through Web and REST.

### Runs And Results

Corpus Mutation actions are `ingest`, `replace`, `delete`, `retry`, `reset`, and
`delete_workspace`.
They use the common `queued`, `running`, `succeeded`, `failed`, and `cancelled`
states and the common `GET|DELETE /runs/{run_id}` plus
`GET /runs/{run_id}/events` observation routes. Cancellation closes at the
durable upstream handoff. A multi-document mutation with any failed document is
`failed`, not partially successful. Acceptance is bounded by the Corpus Mutation
[admission limit](run-runtime.md#admission-limits): reaching it returns HTTP 503
before a Run is inserted, and accepted work is durable.

The terminal result carries the action, stable `track_id`, bounded per-document
outcomes, `document_count`, and `details_truncated`. An ingest's result names
each document that failed by `identifier`, its file name, with the `reason` it
failed. An ambiguous destructive
outcome remains `running` with `phase=waiting_for_repair` and exposes bounded
`repair_reason` and `repair_remedy` guidance. An authorized operator repairs the
upstream state and explicitly uses `POST /runs/{run_id}/resume`, or supersedes
the waiting Run with a Corpus Reset or Workspace Delete naming it as
`supersedes_run_id`.

```json
{
  "action": "ingest",
  "track_id": "dlightrag-corpus-0199a0a0-0000-7000-8000-000000000001",
  "document_count": 1,
  "details_truncated": false,
  "documents": [{"document_id": "file-doc-abc123", "status": "ready", "phase": "finalized"}]
}
```

### Metadata

Custom keys need no declaration and are immediately filterable through
`filters.custom`. Matching is case-insensitive without rewriting stored values
or keys. Built-ins such as `filename`, `filename_stem`, `file_extension`,
`title`, and `author` are reserved. Set title/author through their fields.
`creation_date` is the one built-in accepted under `metadata`; it must be ISO
8601 and is filtered with `creation_date_from`/`creation_date_to`.
`_dlightrag_finalization_complete` is internal and is rejected from caller
metadata.

Metadata GET, update, search, planner schema, and search statistics expose only
Product Documents whose internal finalization marker is exactly true. Pending,
failed-finalization, legacy-unproven, and direct-LightRAG-bypass rows behave as
not found on these surfaces.

`POST /metadata/search` returns document IDs ordered by `doc_id`, with `limit`
(1–100, default 50) and a sealed opaque `cursor`. The cursor is bound to the
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
| `federated_rerank` | retrieve | `false` | Rerank the merged multi-workspace candidate pool when a reranker is configured |
| `bm25_query` | retrieve | query-derived | Optional lexical override; REST/MCP cap it at 1,024 characters |
| `query_images` | retrieve | none | Up to three current images for visual search |
| `attachments` | answer | none | Link descriptors or multipart files used only by this answer |
| `semantic_highlights` | answer | `false` | Add answer-aware source highlights when globally enabled |
| `history` | answer | none | Up to 100 caller-supplied user/assistant messages. Independent `/answer` calls re-send whatever they need. A follow-up or fork injects none: the Agent Session fold at the branch point is the context. |
| `effort` | answer | unset | Agent effort for this Run's answering agent: `low`, `high`, or `max` |
| `filters` | both | none | Built-in and custom metadata filters |

`all_workspaces` is authorization-relative. `None` and `[]` mean omission;
`"*"` and `"all"` are ordinary workspace names. Ingestion remains
single-workspace.

`effort` re-levels the answering agent of one accepted Run and nothing else: a
subagent keeps the level configured for its own role, and a Run that omits the
field keeps the deployment default. The value is refused with a validation error
when it is not one of the three. A level below what the answering model can express is
clamped by the engine to the nearest supported one, preferring the higher neighbour, and
a choice the deployment cannot apply at all — a Fast answer, a model that names no
non-off level, or a role whose reasoning belongs to raw model kwargs — is accepted and
ignored rather than failed; the Run trace's `agent_effort` states the requested level and
what ran. The accepted Run records the choice in its prepared input, and the
submission's idempotency fingerprint distinguishes it, so reusing a submission
id with another effort is a conflict rather than a replay. [ADR 0014](adr/0014-caller-chosen-agent-effort.md)
owns the decision.

An administrator may also set a ceiling on how high a caller's Answers run (the
`agent_effort_ceiling` [Runtime Policy](#runtime-policies)). The Run is pinned to the ceiling
its caller had when it was accepted, as `effort_ceiling` beside `effort`, and every agent
of it runs no higher: the Run's own agent, whatever it chose, and each subagent, whatever its
role was configured with. A level above the ceiling is not refused: it runs at the highest level
the model offers at or below the ceiling, and a model that offers none runs at its lowest. The
ceiling is no part of the idempotency fingerprint, so an edit between a submission and its retry
leaves the retry a replay, and it never reaches a Run already accepted. The trace's
`agent_effort` is `{requested, ceiling, effective}`, with `ignored` (`fast`, `raw_kwargs` or
`no_level`) where the Run's agent could not be bounded and `above_ceiling: true` where the model's
lowest level is above the ceiling. [ADR 0042](adr/0042-runtime-policies-are-read-at-the-door-and-pinned-into-the-run.md)
owns the decision.

### REST

```bash
# Accept a durable Retrieval run.
RETRIEVAL_RUN=$(curl -sS -X POST http://localhost:8100/retrieve \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: retrieval-example-1' \
  -d '{"query":"key findings","all_workspaces":true}' | jq -r .run_id)
curl http://localhost:8100/runs/$RETRIEVAL_RUN

# Accept a durable Answer run.
ANSWER_RUN=$(curl -sS -X POST http://localhost:8100/answer \
  -H 'Content-Type: application/json' \
  -d '{"query":"key findings","semantic_highlights":true}' | jq -r .run_id)

# Read status/result and follow events.
curl http://localhost:8100/runs/$ANSWER_RUN
curl -N -H 'Last-Event-ID: 12' http://localhost:8100/runs/$ANSWER_RUN/events

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

There is no public `/query` route, inline remote Retrieval result, or ephemeral
Answer mode. `POST /retrieve` and `POST /answer` persist a Run and return HTTP
202. A Retrieval descriptor differs only in `run_kind: "retrieval"`:

```json
{
  "run_id": "019…",
  "run_kind": "answer",
  "lane": "query",
  "status": "queued",
  "status_url": "/runs/019…",
  "events_url": "/runs/019…/events",
  "cancel_url": "/runs/019…"
}
```

### Run Lifecycle and Answer Endpoints

| Operation | Contract |
|---|---|
| `POST /retrieve` | Accept a Retrieval Run and return its descriptor. Optional `Idempotency-Key` replays the same normalized request; conflicting reuse returns 409. |
| `POST /answer` | Accept an Answer Run. Optional `Idempotency-Key` replays the same normalized request; conflicting reuse returns 409. |
| `GET /runs` | List this owner's runs oldest-first; `after` + `limit` (1–100, default 50). |
| `GET /runs/{run_id}` | Return common kind, lane, status, cancellation, phase, progress, error, and terminal result fields. |
| `GET /runs/{run_id}/events` | Reconnectable SSE; resume with `Last-Event-ID` or integer `after`. |
| `DELETE /runs/{run_id}` | Idempotent cancellation; 200 if terminal, otherwise 202. |
| `POST /runs/{run_id}/resume` | Requeue an authorized Corpus Mutation Run from `phase=waiting_for_repair`; 409 for every other lifecycle state. |
| `GET /answer/{run_id}/artifacts` | List Answer Artifact descriptors/outcome; 409 before a result exists. |
| `GET /answer/{run_id}/artifacts/{resource_id}` | Stream Answer Artifact bytes with Range support; `download=true` forces attachment. |
| `GET /answer/{run_id}/artifacts/{resource_id}/presentation` | Project an available Markdown Artifact as typed `AnswerResponse`, including that Artifact's validated citation sources. |
| `POST /answer/{run_id}/steer` | Queue an instruction for live Research. |
| `POST /answer/{run_id}/follow-up` | Create a child run that appends to the selected terminal run's Lane. It injects no history; the fold at the tip is the context. |
| `POST /answer/{run_id}/fork` | Create a sibling branch from the state that run settled at. It injects no history; the fold at the Fork Point is the context. |
| `GET /answer/{run_id}/transcript` | Return bounded canonical ancestry. Each message carries its Session `sequence` and `at`, the moment its entry was written (UTC ISO 8601 with a `Z`), which is what a client times a step by: a Tool's own time is the gap between the message that called it and the result's. A message states what was said and done, never the model, parameters, prompt digests or timings of the call that made it: those stay in the Session ([ADR 0043](adr/0043-a-step-records-the-model-parameters-and-prompt-digests-it-ran-on.md)). |
| `GET /answer/{run_id}/transcript/search` | Return the messages of **every agent of that Run** that hold **every word** of `q` (1–200 characters). The words are split on whitespace and punctuation, a quoted `"…"` stays whole, and a run of Han is read **as written**; only when those words match nothing does a second pass ask for **any** of their characters, and the answer's `match` says which pass answered (`all` or `any`). Every word is matched with `ILIKE`, the shape a `pg_trgm` index over the words stored beside each entry serves — a word of one or two characters cannot use it and is scanned, which is stated where the index is declared. Answers are ordered the Run's own agent first and oldest first within an agent, `limit` bounds what one agent contributes (100), and `total` states how many messages the Run holds before that bound, so a shortened list is never silent. Every hit carries the words in their neighbourhood as `snippet` (up to 800 characters, centred on the first word that appears, so a reader can read past the match in place), names its agent (`agent`, `main` for the Run's own) and that agent's task (`agent_label`), and the answer states the words it read as `terms` — which is what a client marks, so a mark always agrees with the match. There is no cursor: one Run is the scope. |
| `GET /answer/{run_id}/overview` | Web only, under `/web/api`. The Inspector's Run-level overview: whole-Run facts only, never a page of steps. `{run_id, totals: {steps, kinds: {thinking, tool, say, answer, instruction, delivery}, decisions, marks: {failed, degraded, repeated, redirect}, tool_time_ms}, agents: [{agent, label, order, steps, tool_calls, marks, tool_time_ms, first_at, last_at}], findings: [{agent, sequence, kind, at, name, subject, outcome, marks, excerpt}], findings_total, usage}`. `agents` are the roster's agents in roster order, the Run's own first (`main`); every count covers the whole Run, not a loaded page. `findings` are the steps the whole Run's mechanical marks name plus its `delivery` (checkpoint) steps, in merge order, capped at 40 with the uncapped `findings_total`: each carries its mark words and the first line of its already-masked content (at most 160 characters), and a thinking step is never a finding, so no reasoning travels. `usage` is the Run result's own `{usage_details, child_usage_details, inclusive_usage_details}` record when the record states one, else `null` — never zeros, and never recomputed. Agent labels and finding excerpts pass the reader content gate (ADR 0061). `Cache-Control: private, no-cache`; the `ETag` is derived from the Run version the read was built under and a matching `If-None-Match` answers `304`. 404 for an unknown or foreign Run. What the panel renders, and why the Run grain is an overview and not a step index, is [ADR 0085](adr/0085-the-run-panel-is-an-overview-and-steps-belong-to-the-agent-page.md). |
| `GET /answer/{run_id}/steps/search` | Web only, under `/web/api`. The one step list the Run panel walks: the steps whose messages hold the search's words across **every agent of the Run** — the same match `/transcript/search` performs — paged newest page first. `q` is required (1–200 characters), `limit` is 1–200 (default 100), and `before` is the `next_before` the previous page stated. The answer is `{run_id, steps[], next_before, matched, terms, match}`: each step names its agent (`main` or a Child Session id), `agent_label`, `at`, `kind` (`tool`, `say`, `answer`, `instruction`, or `delivery` — a thinking step never appears, because a search row names the message's actionable step and the list states no reasoning), the Tool `name`/`subject`/`outcome`/`duration_ms` where it is one, its `sequence`, a bounded `excerpt`, and the mechanical `marks` the whole Run states about it; `matched` is how many matching steps the Run holds, and `terms`/`match` state the words the search read. The cursor names the step's kind as well as its agent and sequence, so a page boundary cannot split one message's thought from its words. `Cache-Control: private, no-store`. 404 for an unknown or foreign Run; 422 for a cursor this route did not state. |
| `GET /answer/{run_id}/feedback` | Web only, under `/web/api`. Whether the owner is offered feedback, and the judgements they gave the messages of that Run: `{run_id, enabled, feedback: [{agent, sequence, value, reasons, note, updated_at}]}`. The Run's own agent comes first, then each Child, and within an agent the oldest message first. `enabled` is false where the `answer_feedback` [Runtime Policy](#runtime-policies) turns feedback off for the caller, and `feedback` is then empty without being read. |
| `PUT /answer/{run_id}/feedback` | Web only, under `/web/api`. Keep the owner's judgement of one message, replacing an earlier one. Body `{agent, sequence, value, reasons, note}`; 200 with the judgement as kept, `{run_id, agent, sequence, value, reasons, note, updated_at}`; 404 for an unknown Run or a message that is not an assistant message that Run wrote; 403 where the `answer_feedback` policy turns feedback off for the caller; 422 for a judgement that cannot be kept. |
| `DELETE /answer/{run_id}/feedback` | Web only, under `/web/api`. Take the judgement of the message named by `agent` and `sequence` (query) back. 204 whether or not there was one, so 404 only for an unknown Run and never for the message. The `answer_feedback` policy does not gate it: taking a judgement back removes only the owner's own data, so it works for a caller the policy has since turned feedback off for. |
| `GET /answer/{run_id}/children` | Newest-first Child Session roster page (`limit` 1–100, default 50). Public status only: no host/plan/budget envelopes or provider-private reasoning. Each child carries `started_at` and `finished_at` (UTC ISO 8601 with a `Z`): when its current Operation began, and when it settled, null while the Child runs; a Child with no Operation reports its own creation time. `pending_questions` counts its questions to the parent that still await a reply and have not expired. `result_sequence` is the `sequence`, in the Child's own Session, of the message it answered with (null until it has), and `feedback` the owner's judgement of exactly that message as `{value, reasons, note}` (null while there is none), so a client names the message to judge without reading the transcript. The page states the Run's own row the same way: `run_status`, and `run_started_at` / `run_finished_at` (same form, null before it starts and while it runs), so a client draws the main agent from the facts a child's row is drawn from instead of inventing a clock for it. |
| `GET /answer/{run_id}/children/{child_session_id}` | Bounded Child Session observation: public status (with the roster's `started_at`, `finished_at`, `pending_questions`, `result_sequence`, and `feedback`), transcript tail, queued/consumed controls, questions, and Evidence handles. `limit` 1–100, default 20. |
| `POST /answer/{run_id}/children/{child_session_id}/control` | Steer, continue, or cancel one Child Session. Requires `Idempotency-Key`. Body `{action, content, reauthorize_user_cancelled}`. 202 for `queued` / `consumed` / `accepted` / `cancellation_requested`; 422 without `Idempotency-Key` or for invalid content or key; 409 with the explicit outcome (`terminal_child`, `run_terminal`, `reauthorization_required`, …); 404 if unknown. User-cancelled continuation requires `reauthorize_user_cancelled=true`. |
| `POST /answer/{run_id}/child-guidance/{request_id}/reply` | Reply to one correlated `ask_parent` request. Requires `Idempotency-Key`. 202 for `replied`; 422 without `Idempotency-Key` or for invalid content or key; 404 if the request is unknown; 409 with the outcome otherwise. |

Run status is `queued`, `running`, `succeeded`, `failed`, or `cancelled`. Phase is
an executor-owned string. Retrieval uses `planning` and `searching`; Answer uses
`routing`, `planning`, `searching`, `researching`, and `generating`. Unknown,
pruned, and foreign-owner Runs all return 404. A retained Run whose event log was
trimmed returns 410 from the event endpoint; status/result remains readable.
Disconnecting a client never cancels a Run.

The common SSE terminal and progress events are:

| Event | Payload |
|---|---|
| `progress` | Current executor-owned `phase` |
| `done` | Terminal success with full kind-specific `result`, or cancellation without one |
| `error` | Terminal `{kind, message}` failure |

Answer may additionally emit `token`, `reset`, `reasoning`, `tool_start`, `tool_progress`,
and `tool_end`. Retrieval emits only `progress` and one terminal event.

A `reasoning` frame carries a batch of the Run's root agent's own reasoning text while its model
is thinking. It is a reader-facing projection like `token`: the edge masks it with the same
safe-prefix carry, so a protected passage crossing a batch boundary is withheld rather than
released early. A `tool_start`, `tool_progress`, or `tool_end` frame closes the reasoning block:
the projection drops that block's held tail there rather than carrying it into the next block, and
the trace page is what states the block whole; the `token` draft's carry survives a Tool event. A
`reasoning` frame states no fact the trace pages hold, so a client that reads pages for facts does
not treat it as one. The same text is what the Inspector carries after the fact, bounded to 4000 characters: the
activity page draws it on the step that message produced — the words it said, the Tool calls it chose, or the answer it concluded (a message with
no words of its own stands as a Thinking step before its calls); the Run panel states no
reasoning and no step order, so the activity page is the only trace surface that reads it; REST and
MCP's transcript tail and the public Child observation do not carry it
([ADR 0065](adr/0065-the-models-reasoning-is-a-trace-surface-bounded-and-masked.md)).

Every answer-shaped payload here is a reader-facing projection, and a projection masks the words
of a protected Skill document: a verified reproduction becomes `[...]` in the answer, its retrieval
contexts, its source chunks, the transcript (including Tool arguments), Profile Memory, a Child's
conclusion and questions, and a conversation's goal text. A `token` frame carries only text the
reader may be shown, so a reproduction can be withheld for a batch and replaced once its end is
known; the `done` frame is projected through the same gate, so a finished answer and a reloaded
one agree ([ADR 0061](adr/0061-a-protected-skills-text-is-masked-at-the-readers-edge.md)).

Tool events carry metadata only; the browser projection adds fields of its own:

| Field | Events | Meaning |
|---|---|---|
| `tool_name` | all three | Transport-neutral tool identity |
| `call_id` | all three | The one call being started, updated, or settled |
| `source_index` | all three | Position of the call in its model turn |
| `outcome` | `tool_end` | `succeeded`, or a typed failure such as `failed`, `invalid_arguments`, `tool_contract_changed`, `outcome_unknown` |
| `duration_ms` | `tool_end` | Measured wall time of the attempt that settled; absent when no call ran |
| `object_label` | `tool_progress` | The bounded one-line Tool Subject a Tool reports for its live row (a query, a path, a command, a pattern, a skill name, an anonymous public URL, a Resource Handle). It is the Tool's own label derived from one of the call's inputs, never the raw argument payload; a Tool reports it at its own discretion: a call whose only candidate subject is prose rather than a name, or an identity no viewer could act on — a Child Session id, a Memory Record id, an Outbound MCP Tool's unknowable arguments — reports none |
| `text_chars`, `attachment_count`, `output_bytes`, `spill_state` | either | Bounded size and spill metadata when a producer reports it |
| `elapsed_ms` | `tool_start` | Browser projection only: how long a replayed start has already been running, so a reloaded trace keeps counting |
| `tool_label` | all three | Browser projection only: a human name for a pinned Connection tool, resolved from the Run's pins |
| `facts` | `tool_end` | The bounded structured facts a Tool reports about what it did: counts, statuses, and identifiers (a search's provider and hit count, the sources it returned, a read's line window). Never the result text: facts are the shape of a step. Their string leaves still pass the reader's content gate, because a query or a page title is a Tool's prose: the browser and REST frames mask them, and a frame rendered without a gate omits the field ([ADR 0061](adr/0061-a-protected-skills-text-is-masked-at-the-readers-edge.md)) |

No tool event carries stdout, stderr, arguments as such, or Tool output. The
producer-chosen facts that do cross are `object_label` and, on the settled event, `facts`: a bounded
Tool Subject the Tool derives from one of the call's inputs — a command, a query, a path — never the
argument payload itself, and a Tool reports it only when it names one thing; and a closed set of
bounded structured facts about the call (`ToolResult.facts`, at most 16 keys, 2 KiB, two levels), which
the Runtime validates and the transcript carries beside the result.

Each durable sequence is the SSE `id`. A cursor that is not a non-negative
integer, or header and query cursors that disagree, return 422. Without a
cursor, replay starts at sequence 1. Ten-second
comment keepalives consume no sequence. Exactly one terminal event is committed.

Canonical successful Retrieval result after reader projection:

```json
{
  "contexts": {"chunks": [], "entities": [], "relationships": []},
  "sources": [],
  "trace": {},
  "image_descriptions": []
}
```

Runtime storage omits authorization-dependent download/image URLs and query-image
bytes. `GET /runs/{run_id}` and terminal SSE reads project source and visual URLs
against the caller's current permissions.

Canonical successful Answer result:

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

In the REST and Web projections, each Artifact a result lists carries the record the platform
derives from the Run's own durable records
([ADR 0058](adr/0058-a-skill-declares-the-verifier-of-its-delivery.md)); the MCP `get_run` result does
not stamp it today. A Skill declares its verifier in `SKILL.md` frontmatter, and the record says
whether that verifier was recorded running and which declared reads the Run skipped. The record never
withholds the file: an unattested verdict is served beside the Artifacts it covers, and a verdict that
names no verifier (no Skill, a Skill that declares none, an unreadable declaration, a log that cannot
name the Skill) is served the same way. It never turns a succeeded run into a failed one.

`trace.agent_shell_confinement` states what the answering Agent's processes could be
confined to on the host that produced the Run, in the same closed shape `/health`
reports. A Research Run's `trace.agent_turns` counts its own model calls, and
`trace.prompt_cache` aggregates their billed prompt and cache hits (`turns`,
`prompt_tokens`, `cache_hit_tokens`, `cold_turns`); both count the attempt that finished
the Run, so a Run recovered after a crash reports only the calls made after its recovery.
`trace.bm25_enabled` reports lexical-lane participation. If one retrieval lane
fails and the other succeeds, the result continues with `bm25_error_type` or
`lightrag_error_type`; `lightrag_mix_chunk_count` records the pre-fusion
LightRAG count.

A Research Run's `trace.tool_observations` is one row per root-agent tool call: `tool`, `call_id`,
`outcome`, `cached`, `is_error`, `content_chars`, `capacity_tokens`, `ceiling_tokens`,
`observed_truncated`, and `subject` — the first subject the Tool reports in a progress update (later
updates, and a subject present only on the settled result, are ignored): a bounded, tool-reported line
naming what the call acted on (a search query, a read URL or path, a shell command, a browser action;
at most 64 characters, absent when the tool reports none, and never inferred from its arguments).
`trace.web_search_cost_dollars` is the web-search provider cost the search tool accumulates for the
run. The trace is a bounded summary of what the Run did, not its durable call log: the event log
carries call metadata and the subject only, while full tool arguments and model-visible (fitted) tool
results live in the Session transcript, and in the training export where that Session is exported with
the owner's consent ([ADR 0044](adr/0044-the-training-dataset-is-a-full-re-export-of-whole-consented-sessions.md)).

### In-process Application

```python
from dlightrag.application.access.principal import DEPLOYMENT_OWNER_ID
from dlightrag.application.answer_runs.service import AnswerRequest
from dlightrag.application.retrieval.service import RetrieveProjection, RetrieveRequest

retrieval_run = await application.retrieval.create(
    request=RetrieveRequest(query="What changed?", workspaces=("default",)),
    owner_id=DEPLOYMENT_OWNER_ID,
    idempotency_key="retrieval-example-1",
)
retrieved = await application.retrieval.wait(
    owner_id=DEPLOYMENT_OWNER_ID,
    run_id=retrieval_run.run.run_id,
    projection=RetrieveProjection(
        downloadable_workspaces=None,
        visual_workspaces=None,
    ),
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

`RetrievalService.retrieve()` is the create-and-wait convenience; cancelling that
await only detaches and does not cancel its accepted Run. `RunService` exposes
common status, listing, subscription, and cancellation.

`AnswerService` owns Answer-specific steering, continuation, transcript, answer
feedback, Child Session roster/observation, and child control/reply methods. There is no separate
public Python SDK for remote callers.

### MCP Server

MCP `retrieve` and `answer` return only durable descriptors; poll `get_run` for
the canonical result. A tool result puts typed JSON in `structuredContent` and
formatted equivalent JSON in its first text block. Expected validation,
authorization, conflict, and temporary-unavailability failures set
`isError: true` with their public text; an invalid argument is named with the
reason, never echoed. A Run the caller may not see reads as not found. Other
failures report only an internal tool failure. Protocol failures remain
JSON-RPC errors.

The 18 public tools form one fixed task interface. `answer(mode="research")`
retains autonomous research, including its internal tools, child agents, and
Memory under the existing owner policy. Run-level steering, continuation,
branching, transcript access, cancellation, and Artifact reads let the caller
direct and receive the work without taking over individual child agents.

Registered public tool names are:

- query/run: `retrieve`, `answer`, `get_run`, `cancel_run`, `list_runs`,
  `steer_answer_run`, `follow_up_answer_run`, `fork_answer_run`,
  `get_answer_transcript`, `list_answer_artifacts`, `read_answer_artifact`
- corpus: `list_workspaces`, `get_capabilities`,
  `create_workspace`, `ingest`, `retry_files`, `list_files`, `delete_files`

`read_answer_artifact` serves only an available Published Artifact from the
Run's stored result, the same rule as the REST and Web artifact routes; input
uploads and fetched resources are not artifacts.

Management operations are REST-only: Profile Memory, the model catalogue,
storage status, corpus reset and resume, and child supervision. MCP argument
objects are not REST payloads; owner identity, resource permissions, and required
idempotency keys still apply. The standalone Memory package's separately bound
MCP server keeps its own four-tool contract.

### Web

Web routes under `/web/api/*` are browser contracts, not compatibility aliases
for REST. `GET /web/api/bootstrap` (bootstrap contract version 6) returns
authorized workspace state, Files target, attachment limits, image capability,
the document types admitted by their extension (`answer_attachments.extensions`,
the Engine's list above, which is what the composer offers),
`agent_effort: {levels, default}` — the efforts this deployment applies to this
caller and, when it is one of them, the level their Answers run at when they
choose none (the deployment's own, lowered to the caller's `agent_effort_ceiling`)
— and `answer_feedback_enabled`,
whether the `answer_feedback` [Runtime Policy](#runtime-policies) offers this
caller the thumbs under an answer — never bearer or edge tokens. Where the
deployment configured both halves of the browser channel it also returns
`observability: {url, apiKey, service}`, the ClickStack ingestion address, key,
and service name the page's RUM and session replay report under; the object is
absent otherwise, and the page starts no SDK. Every bootstrap
reads the policies afresh, so a change holds on the next page load. A stored
document that no longer validates never takes the page down: the feedback
policy's leaves `answer_feedback_enabled` `false` and the rating routes answer
503, and the ceiling's leaves `agent_effort.levels` empty and a new submission
answers 503 (so does the active-run limit's), until an administrator replaces
the document.
Each workspace it lists, like each one `/web/api/workspaces` pages and the one
`/web/api/workspaces/create` returns, carries `changes`: the Corpus Mutations its
caller may request there, of `ingest`, `replace`, `delete`, `retry`, `reset`, and
`delete_workspace`. The default workspace never offers `delete_workspace`.
`POST /web/api/answer` accepts the same optional `effort` as REST and MCP, on both its JSON and its
multipart form.
Route families cover:

- `/conversations`, `/conversations/{id}/history`, and the one run-resource read
  surface `/runs/{run_id}/resources/{resource_id}` (plus `/thumbnail`), which
  serves an accepted upload, a publication, and an image the run fetched,
  rendered, or adopted under the id its registry recorded;
- `/runs/{run_id}/browser/live/frames`, `/control`, `/input`, and `/screenshot`, the
  live preview of a browser the Run has leased. `frames` is one WebSocket
  (`WS`/`WSS`, the scheme of the page that opened it), and it is accepted only for a
  caller whose session cookie authenticates and whose `Origin` is this application's
  own: a browser can set neither a header nor a token on an upgrade, so those two are
  the whole handshake. The binary opcode carries one frame — a fixed 16-byte header and
  then the JPEG bytes the page's browser encoded, handed on to the byte, never base64 and
  never wrapped in text. The header is `!Qd`, big-endian: bytes 0–7 are this session's own
  sequence number for that frame (a `uint64` starting at 1 that only ever goes up), and
  bytes 8–15 are the moment the page was captured as a `float64` of epoch seconds, which is
  CDP's own frame metadata `timestamp` handed on as it stands: the protocol states that
  moment in seconds since the epoch, and a frame whose metadata carries none is stamped with
  the moment it reached the server instead, a little later and never earlier. A reader drops
  a frame whose sequence is at or below the one it has already drawn rather than painting it
  over a newer one, subtracts the capture time from its clock to know how old the picture
  already was when it was drawn, and compares that frame's number with the newest the
  session has delivered — every text message states it — to know whether it is behind the
  stream or only looking at a page that has stopped changing. The header is fixed-width in
  front of the picture precisely so that a frame and what describes it cannot race each
  other. The text opcode carries the state it arrived in:
  `{"control": "agent"|"human", "driver": <label>|null, "attached": <bool>,
  "ended": <string>, "sequence": <newest frame int>, "frames_per_second": <number>|null,
  "jpeg_quality": <number>|null, "max_width": <number>|null, "at": <epoch seconds>}`.
  `driver` is a *display label* for
  whoever holds the seat, such as `"viewer-3f9a1c04"`, and never a credential: it is
  broadcast to every reader of the Run, so nothing that can drive may travel in it. A text
  message is sent for every
  update, whether or not a frame follows it, so a reader that missed a frame still reads
  who drives from the next one, and the three budget fields say what the session is
  delivering at now — a session whose viewers cannot keep up steps itself down, so the
  rung moves without anybody asking. Both are driven by the page, so a page that is quiet sends
  nothing at all, and the route keeps the socket alive itself: when nothing has gone out
  for 30 seconds it sends one heartbeat, `{"at": <epoch ms>, "sequence": <newest frame int>}`
  — a text message carrying no state field, which a reader applies as no state change and
  no repaint, and whose sequence is the number the last state message stated, repeated
  unchanged: level with the newest frame a reader has drawn it says the page has not
  changed, and ahead of it says a frame was delivered the reader has not drawn yet. That
  is also what holds a proxy's idle window (Cloudflare's is about 100 seconds) open, so it
  does not close a dock whose page has not moved. The 10-second comment keepalive the SSE
  routes use
  (`SSE_KEEPALIVE_SECONDS`) is another protocol's answer to another clock; this route's own
  window is the 30 seconds above. The socket is read-only: a viewer sends it nothing, and
  the budget in force paces every frame it is sent. It captures only while somebody is
  reading it, and a viewer that reads slowly is sent fewer frames rather than older ones.
  A reader whose page is hidden closes this socket and opens no other, which is the whole
  of the protocol for a preview nobody is looking at: the last reader leaving is what
  stops the page's capture, and there is no message for it. It closes `1000`
  when the session ends, and before any frame with `4401` (no credential), `4403` (the
  `Origin` is not this application's), `4404` (no such Run for this caller), or `4409`
  (the Run holds no live session, its close reason the browser failure's own word:
  `not_configured`, `unreachable`, `no_page`, or `expired`). `control` takes the driving
  seat
  (`{"state": "human"}`) or hands it back (`{"state": "agent"}`). A take carries no
  credential and answers the seat as `{"control", "driver", "token"}`: `token` is the
  credential that holds the seat, minted as that one request is answered, written down
  nowhere else and replaced whenever the seat changes hands. A hand-back presents it in
  `X-Live-Seat` and answers `{"control", "driver", "token": null}`; the credential is spent
  with the hand-back, and a hand-back that presents none is refused 409 as `taken_over` with
  the seat still held, because the credential *is* the authority to give the seat up. `input`
  sends one mouse, wheel, key or navigation event, carrying that same `X-Live-Seat`
  credential and the `Idempotency-Key` a retry resumes
  under, plus an optional `sequence` — the sender's own count of the events it has sent,
  which the session drops rather than runs when it is at or below one already run, so an
  event the wire overtook cannot land after the event that overtook it. A write with no
  credential is refused as an unprocessable request, and one presenting a credential that
  is not the seat's is refused 409 as `taken_over`: a second reader — which reads the whole
  broadcast, and the broadcast holds only the label — stays an observer. Nothing drives or
  types over the socket: it carries
  the read stream only. `screenshot` serves the last picture the Run's
  browser stored, or 404 where it stored none, so a dock with nothing attached is
  never blank. A refusal here is the browser command envelope, `{kind, message}`,
  with `kind` one of `not_configured`, `unreachable`, `no_page`, `taken_over`,
  `action_failed`, `expired`; every answer of the three HTTP routes is
  `Cache-Control: private, no-store`;
- `/answer`, submission reconciliation, status/resume/steer/children (whose
  page also carries the Run's own row — `run_status`, `run_started_at`,
  `run_finished_at` — so a client sees whether Child commands still apply and
  draws the main agent from the same facts every child's row carries),
  child observation/control/guidance reply, fork/cancel,
  Artifacts/presentation, answer feedback, and events;
- `/run-activity`, the reader's recent conversation-linked Answer Runs: what is
  running and what finished where, across conversations. `GET
  /web/api/run-activity?limit=` answers `{"items": [{"run_id",
  "conversation_id", "conversation_title", "status", "phase", "created_at",
  "started_at", "finished_at", "goal_round"}]}` for the authenticated owner, newest first,
  bounded to `limit` (1–100, 30 unless given). A row is a conversation-linked
  Answer Run only — a Run accepted through REST or MCP has no conversation and
  is absent — and `conversation_title` is null for an untitled conversation.
  `goal_round` is true when the Run is a goal's continuation round: the page keeps
  it as activity but never announces it as a completion. The list is a read of the
  durable Runs and never a queue: it creates nothing, and a page derives running
  and unread from it, keeping unread in its own storage and watching only while a
  Run is live ([ADR 0069](adr/0069-the-web-reads-run-activity-from-the-runs-it-keeps.md));
- Files/upload and same-origin `/corpus-runs/{run_id}`
  status/events/cancel/resume, workspaces, images, Memory, the owner's [Schedules](#schedules),
  the owner's Agent Accounts, the owner's Conversation Contribution, the Agent Skills the
  viewer can load and the owner's own, model catalogue, the
  [Runtime Policies](#runtime-policies), and `/client-errors`, which logs one uncaught browser
  error.

`/agent-accounts` is an owner's Agent Accounts in Settings. It is Web-only, like Connections:
owner Settings state has no REST counterpart. `GET /web/api/agent-accounts` answers
`{"available": <bool>, "registration": {"allowed": <bool>, "enabled": <bool>}, "accounts":
[{"site", "email", "username", "created_at", "last_used_at"}]}`, the accounts in the order of
their sites. `available` says the deployment composes Agent Accounts, which it does wherever
an Agent Browser is configured; `registration.allowed` is the deployment's allowance for the
Agent to register, which holds where an Agent Browser is configured and
[`account_registration`](configuration.md#agent-browser) is on, and `enabled` the owner's
switch, and what the Agent may do is both. `email`, `username`, and `last_used_at` may be null,
and times are UTC ISO 8601 with a `Z`. `PUT /web/api/agent-accounts/settings` takes
`{"registration_enabled": <bool>}` and is accepted even where the deployment does not allow
registration.
`DELETE /web/api/agent-accounts/{site}` removes the owner's account on a site, and answers 404
where this owner has none there, another owner's account being as unknown as none. Each
answers 200 with the fresh view, and none takes an `Idempotency-Key`, since both writes are
idempotent by key. No answer carries a password,
envelope, key id, or account id ([Security](security.md#agent-accounts)).

`/privacy/settings` is an owner's Conversation Contribution in Settings
([ADR 0041](adr/0041-a-conversation-improves-the-product-only-when-its-owner-asked.md)). It is
Web-only, like Agent Accounts, and no access action guards it: it is the owner's own state.
`GET /web/api/privacy/settings` answers `{"contribute": <bool>, "since": <string|null>}`.
`contribute` is whether the owner lets their conversations be used to evaluate and train models,
and `since` is when that permission began, UTC ISO 8601 with a `Z`, null while it is off. An owner
who has never chosen reads `{"contribute": false, "since": null}`. `PUT /web/api/privacy/settings`
takes `{"contribute": <bool>}`, a JSON boolean and nothing else (422 otherwise), and answers 200
with what is kept, in the same shape. It is idempotent by value: turning it on while it is on keeps
`since`, turning it off clears it, and turning it on again sets it to that moment. The operator's
trace export, `scripts/export_traces.py`, honours the setting with default deny and has no option
that includes more: it exports a Run only if its owner contributes then and the Run's Agent Session
started at or after `since`. A Session starts at the earlier of its own row's creation and the
acceptance of its first remaining Run, and the Run's Session is the one its routing row names,
never a Child's. A conversation and everything forked or followed up from it, on the Web or over
REST or MCP, share one Agent Session, and a Run submitted without a parent starts a new one, so the
setting covers no earlier conversation, even one that is continued or forked. A covered
conversation is covered with what it brings in: an attachment, the `history` a REST caller sends
with a request made after `since`, and a Profile Memory the agent recalls, whose body is a Tool
result in the Session. A Run with no Session, a Retrieval or a Corpus Mutation, is never exported.
In `none` and `simple` authentication every caller is the same owner, so one switch covers the
whole deployment, REST and MCP clients included. The setting changes nothing in monitoring
([ADR 0039](adr/0039-observability-is-always-on-and-the-seam-masks-private-data.md)).

The export is a host tool and the server has no surface for it. `make export-traces` runs
`scripts/export_traces.py` and passes `ARGS` on. It reads the deployment's PostgreSQL in one
read-only transaction, in one of two modes that share one record builder and the rule above:

- **Raw**, the default: one JSONL record per Run on stdout or in `--out`, ids as stored, for the
  operator to inspect. `--out` is written with mode 0600, replacing a file already there and
  leaving its directory as it is. `--since` and `--until` bound when a Run was created; `--limit`
  (1000 unless given), `--run-id` and `--owner` narrow; `--dedupe` keeps the newest Run of each
  request fingerprint.
- **Dataset**, `make export-traces ARGS="--dataset DIR"`
  ([ADR 0044](adr/0044-the-training-dataset-is-a-full-re-export-of-whole-consented-sessions.md)):
  `train.jsonl`, `eval.jsonl` and `datasheet.json`, written into a directory that does not exist
  or is empty (mode 0700, the files 0600) and refused for any other. It takes whole Sessions, every
  finished Run (succeeded, failed or cancelled) of each, and `--owner`, `--since` and `--until`
  apply to a Session as a unit, by when it began. `--limit`, `--run-id`, `--dedupe` and `--out` are
  refused, because each would cut a Session or leave the directory. Every owner, `submitted_by`,
  Session, Run, Entry and Child id, and the ids minted for a tool call's intent and attempt, a
  compaction, a steer and a fork's lane, is replaced, wherever it stands in a record, by its
  HMAC-SHA256 under the key `DLIGHTRAG_EXPORT_KEY` ([Configuration](configuration.md)), at least 32
  characters in `.env` or the environment, and the mode refuses to run without it; the same key
  gives the same pseudonyms, so datasets made with it link up. Text is not scrubbed. A Session goes
  to `eval` when a hash of its stored id is below `--eval-fraction` (0.10 unless given) and to
  `train` otherwise, so exporting more never moves one. Records are sorted by Session, creation and
  Run, so the same database, key and options write the same `train.jsonl` and `eval.jsonl` byte for
  byte. `datasheet.json`, written last, states the selection rule and consent basis, the counts
  (which are of the files), the time window, the label distribution, what is left out and why,
  the split, a fingerprint of the key, the SHA-256 of each file, how every time is measured, and the
  limits of the data.

A record is one Run, an episode: `schema_version` (5), `split` (null in raw mode), `run`, `input`,
`routing`, `session`, `steps`, `children`, `evidence`, `session_notes`, `result` and `labels`.

- `run` has the status, stop reason, error kind (never the error message), and `created_at`,
  `started_at`, `finished_at` and `duration_seconds`.
- `steps` are the entries the Run itself wrote, in Session order, each with `sequence`, `entry_id`,
  `parent_entry_id`, `type`, `schema_version` (that of the entry's payload format), `timestamp`,
  `payload` as stored but for `provider_state`, and `tool`. An assistant message or a compaction
  carries its `call` at `payload.call`: the model, parameters, prompt digests and timings
  (`queued_ms`, the wait for a model slot, and `duration_ms`) of the model call that made it, absent
  on a step written before it existed
  ([ADR 0043](adr/0043-a-step-records-the-model-parameters-and-prompt-digests-it-ran-on.md)). A tool
  result's `tool` has `started_at`, `ended_at` and `duration_ms`, from the `tool_start` and
  `tool_end` events of its call, each null where those events do not say it exactly.
- `session` has `id`, `head_entry_id` (the Run's last entry) and `carried`: the entries of the chain
  above the Run's first step which no exported Run wrote, whole, root first, because retention
  pruned their Run, its owner deleted it, or an option left it out. The Run saw them. The rest of
  the chain is steps of the Session's other records, so a consumer rebuilds all of it by following
  `parent_entry_id` from the first step through those steps and `carried`; no list of it is kept,
  since it would grow with the square of a conversation.
- `children` are nested episodes, each tied to the step that spawned it by `parent_call_id`, with
  its own `steps`, `started_at`, `finished_at` and `duration_seconds` (null while it has not
  finished).
- `evidence` lists the ledger snapshots the Run's tool calls left, by digest and locator and with
  no passage text; `result` is `{"sources": [...]}`, the sources the answer cites with the
  passages cited, or null; `labels` lists the owner's judgements of its answers, each `agent`
  (`main` or a Child), `sequence` of the step judged, `value`, `reasons`, `note` and `given_at`, and
  is never reduced to a score.

`GET /web/api/skills` is what `/skill:` completion offers: `{"skills": [{"name",
"description", "source"}]}` for the built-in, global and own Skills the viewer's agent can load.
An own Skill its owner turned off is not in it, and a same-named Skill of a lower tier is.
`/skills/mine` is the owner's own Skills in Settings, Web-only like Agent Accounts, and none of it
takes an `Idempotency-Key`, since each write is idempotent. `GET /web/api/skills/mine` answers
`{"skills": [{"name", "description", "enabled"}], "limit": <int>}`, the owner's own tier by name
with those turned off included, and `limit` the quota of Skills an owner may keep.
`PUT /web/api/skills/mine/{name}/enabled` takes `{"enabled": <bool>}` and answers 200 with the
updated row. `DELETE /web/api/skills/mine/{name}` answers 204, and
`GET /web/api/skills/mine/{name}/document` the Skill's `SKILL.md` as `text/plain` with
`X-Content-Type-Options: nosniff`, whether the Skill is on or off. The three that name a Skill
answer 404 in the general envelope (`error_type: "not_found"`) where this owner has none: a
built-in or global Skill, another owner's, and a name that is not kebab-case are as unknown as no
Skill at all. A `/skill:` request for a Skill the owner turned off, and that no lower tier serves,
is refused with `invalid_request` and says to turn it on in Settings.

An answer that cites a stored image is projected with that image addressed on
this origin: `<img src>` is rewritten to the run-resource address when the run
holds the bytes (matched on the URL its own row recorded), and left exactly as
the answer wrote it otherwise — no image is fetched to make a rewrite possible.
[ADR 0016](adr/0016-one-run-resource-read-surface.md) owns the decision.

`/web/` is unpersisted New Chat;
`/web/conversations/{conversation_id}` selects a durable owner-scoped
conversation. The URL is authoritative for reload and browser history.

A missing, expired, or rejected sign-in is answered by what the request is. A
browser's top-level navigation (`Sec-Fetch-Mode: navigate`, GET, no
`Authorization` header) is redirected to `/web/login?next=…` wherever it
points, a download link under `/web/api` included, and drops a pasted-token
cookie that stopped working. Every other request (`fetch`, a nested frame, a
bearer header) gets HTTP 401 in the general envelope (`error_type: "auth"`),
never a redirect or the login page, and its cookies are left alone. A client
that sends no `Sec-Fetch-Mode` is judged by the path: `/web/api/*` answers the
401 and any other `/web` page redirects. With an edge-asserted identity nothing
redirects; the edge owns sign-in. The browser app reads any 401 as the end of
its sign-in: it replaces the page with one prompt to sign in again that returns
to the current URL.

`POST /web/api/answer` accepts an optional conversation ID, query, attachments,
and search workspaces. It returns HTTP 202 with canonical `{conversation, turn}`.
For a first submission, the server creates conversation, turn, blobs, and run in
one transaction from the owner-scoped `submission_id`. On an ambiguous result,
use `GET /web/api/answer-submissions/{submission_id}`; the browser must not
blindly repeat the POST. Submit, lookup, and `POST /web/api/answer/{run_id}/fork`
refuse typed failures with `{kind, message, error_kind?}`, where `kind` is one
of `invalid_request`, `attachment_rejected`, `scope_forbidden`,
`conversation_missing`, `submission_conflict`, or `service_unavailable`, and
`error_kind` names the stable answer error kind when admission rejected the
input. `service_unavailable` can also follow an accepted submission, so the
browser reconciles through the lookup or retries with the same `submission_id`.
Request-shape validation, sign-in, and internal failures keep the general
envelope.

The browser reads an agent's activity a page at a time with
`GET /web/api/answer/{run_id}/transcript`: the Run's main agent, or the Child
Session named by `child`. An assistant message's `tool_calls` entries carry `id`, `name`, and —
where the call's `tool_progress` event carried one — the optional `subject`: the same bounded
`object_label` the live Tool row shows, joined from the Run's durable events. It is metadata about
what the call acted on, not the call's result. `limit` is 1-50 (default 30). Without a cursor it
returns the newest page; each page holds the messages older than the `before`
sequence, oldest first, and carries `next_before` (the cursor of the page before
it, null at the start) and `running` (whether that agent is still working, so a
Tool call without a result is in flight). An assistant message also carries
`reasoning`, the model's own thinking for that turn, when the provider returned
any: it is projected through the same reader content gate as the message's words,
bounded to 4000 characters, and omitted rather than sent empty. The live
`reasoning` frame is the same text while the Run runs; this is how the trace reads
it back, and REST and MCP's transcript tail do not carry the field
([ADR 0065](adr/0065-the-models-reasoning-is-a-trace-surface-bounded-and-masked.md)).
Each message carries `at`, the moment
its entry was written, so a pane times a Tool step by the gap between the message
that called it and the result's, and a call still in flight by the clock it
holds; a message built from a request rather than a stored entry leaves it out. A
page reads the Run's own messages: the Run's main agent is one Session across
every Run of its conversation, and what spans those Runs is what a conversation
projects.

A reader that already holds a transcript asks the other way round with `since`:
the oldest messages written after that sequence, so asking again from the last
message it holds appends what landed without skipping one. A delta carries
`next_before: null`, because walking back is the page read's business. `before`
and `since` are alternatives: naming both is a 422.

`GET /web/api/answer/{run_id}/children/{child_session_id}` observes a
Child without its transcript. The Inspector is a reading surface: it shows that observation (including
the controls a Run received) but offers no child control or guidance reply itself; the two POST routes
below keep their contract for clients that call them directly, and an owner who wants to act does so in
the conversation.

A reader searching those words asks `GET /web/api/answer/{run_id}/transcript/search`
with `q`, and gets back the messages of every agent of that Run that hold them, each naming
its agent and carrying the words in their neighbourhood, so the dock marks exactly what the
search read. That answer states how many matches the Run holds — a shortened list says so —
and whether the words as written answered or a looser reading of them did.

An owner judges one message of an answer with `PUT /web/api/answer/{run_id}/feedback`, body
`{agent, sequence, value, reasons, note}`. `agent` is `main` for the Run's own agent or a Child
Session's id, spelled as the transcript and the search name it (lower-case, canonical), and
`sequence` is the message's own `sequence` in that agent's transcript. `value` is `up` or `down`.
A `down` says why in `reasons` — any of `unsupported`, `factual_error`, `too_verbose`, `off_topic`,
and `irrelevant_citations`, kept once each in that order — or in a `note` of the owner's own words
of up to 2000 characters, trimmed and null when empty; one that says neither, with no reason and no
note, is refused with 422. An `up` carries neither, and one that does is refused with 422. Judging a
message again replaces the earlier judgement whole. The answer is 200
`{run_id, agent, sequence, value, reasons, note, updated_at}`, `updated_at` being when the
judgement was last given, in UTC ISO 8601 with a `Z`, as the roster states its times.

Only an assistant message that Run itself wrote can be judged. A user message, a tool result, a
message an earlier Run of the same conversation wrote, and a message of another Run's Child all
answer 404, because the statement that keeps a judgement also checks that its message is there:
a judgement for a message that is not there is never stored. `DELETE
/web/api/answer/{run_id}/feedback?agent=&sequence=` takes a judgement back and answers 204 whether
or not there was one, and `GET` answers what the owner has given the Run's messages. The Run has
to be the caller's own (404 otherwise, for all three). Rating asks for no access to the workspaces
the answer searched, because a judgement is its owner's own state
([Security](security.md#actions-and-presets)); whether the owner is offered it is the
`answer_feedback` policy below. Where it is off for the caller, `PUT` answers 403 "Answer
feedback is turned off for you" and `GET` answers `enabled: false`, but `DELETE` still takes a
judgement back, because that removes only the owner's own data. A judgement that cannot be
kept is refused with 422 before the message is looked up. A Run's judgements are
deleted with it. REST and MCP have no route to give, read, or take back a judgement; a
Child's row, which REST shares with the Web, still carries the owner's judgement of its answer.

The Web event stream follows the same durable sequence as REST but projects a
typed `AnswerPresentation` (`answer_text`, `parts`, `sources`,
`evidence_images`, `artifacts`, and `artifact_outcome`). Conversation history
uses the same shape. Pending, failed, and cancelled turns remain visible;
only succeeded turns become model history. A history turn also carries the
`steering_messages` its Run received, oldest first and at most the newest 50:
the owner's own steers of the parent Run, never a Child's targeted control or a
message the parent wrote itself. The Web draws them under the question, above
the Run's answer. Each turn also carries `child_count`, the
number of Child Sessions its Run spawned (0 for none, and for a turn just
accepted), so a settled turn can tell whether it has children to open. It carries
the two facts the thumbs under its answer need as well: `answer_sequence`, the
`sequence` of the message the Run answered with, and `feedback`, the owner's
judgement of exactly that message as `{value, reasons, note}`. The message a Run
answered with is the one its agent ended on: its last assistant message, when that
carries no tool calls and says something. `answer_sequence` is null until there is
one (and for a turn just accepted), `feedback` is null while the owner has given
none, and a judgement of any other message of the Run is not shown. The same two
facts come on a Child's row as `result_sequence` and `feedback`, in the Child's own
Session.

History defaults to the newest 40 turns and accepts a sealed cursor plus a limit
up to 100. Attachments are owner-scoped, content-addressed run blobs and are
re-registered lazily for continuations. Count, per-file, and total-byte limits are
validated before acceptance; read failures after acceptance produce a terminal
error rather than silent omission. Lifecycle details are centralized in
[RunRuntime and durable query execution](run-runtime.md).

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
sources only and sets `cited_chunk_ids`. Projecting a historical result does not
re-authorize its source: the live download and visual routes return 404 after a
document is hidden.

## Citations

`references` is the compact `{id, title}` projection of validated cited sources.
Inline citations accept:

| Format | Meaning |
|---|---|
| `[1-2]` | Document/reference 1, its second chunk (1-based) |
| `[3]` | Document/reference 3 |

Resolve `[1-2]` by finding source `id: "1"`, then chunk `chunk_idx: 2`.
`page_number` helps navigation but does not affect citation validity.

Markers are recognized in Markdown prose. Code, math, escaped brackets, existing
link/image contents and reference definitions stay literal. A real reference
link such as `[official][1]` with a `[1]: ...` definition takes priority over
the numeric citation form. Validation, source selection, highlights and browser
badges share this interpretation. Finalization preserves surrounding Markdown
and ordinary `References` sections; only actual invalid markers are removed.

Visual bytes are read through authenticated routes. Both full and thumbnail
reads return 404 when the chunk has no attributable, currently finalized
Product Document; cached thumbnails do not bypass that check:

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

The agent efforts a deployment applies to a caller are the same kind of fact. MCP
`get_capabilities` reports them as `answer_agent_effort_levels` and the Web bootstrap as
`agent_effort.levels`, for the caller asking: the levels above that caller's
`agent_effort_ceiling` are not in them. An `effort` outside that set is accepted and never a
validation failure: it runs at the nearest level that applies, or is ignored, and the Run trace's
`agent_effort` records the level that ran. A ceiling policy that cannot be read offers none.

## Workspace And File Management

| Route | Contract |
|---|---|
| `GET /workspaces` | Page the authorized workspace catalogue. |
| `POST /workspaces` | Create an empty workspace attributed to the caller (201; duplicate 409). |
| `GET /workspaces/{workspace}/storage` | Read operator storage/promotion state. |
| `POST /runs/corpus/reset` | Accept Corpus Reset while retaining Workspace identity and history. |
| `POST /runs/corpus/delete-workspace` | Accept Workspace Delete for an explicitly named, registered, non-default workspace (404 once it is gone). Like reset, it may name the Workspace's waiting mutation as `supersedes_run_id`. |
| `GET /files` | Page processed files for one workspace. |
| `POST /runs/corpus/delete` | Accept durable deletion by exact path, name, or document ID. |
| `GET /files/failed` | Page failed documents. |
| `POST /runs/corpus/retry` | Accept retry for exact document IDs or `all_retryable`. |
| `GET /files/raw/{document_id:path}` | Stream or redirect one authorized source. |
| `POST /metadata/search` | Page matching document IDs. |
| `GET /metadata/{doc_id}` | Read one document's metadata. |
| `POST /metadata/{doc_id}` | Merge nonempty custom metadata. |

REST uses resource-oriented operations:

```bash
curl -X POST http://localhost:8100/workspaces \
  -H 'Content-Type: application/json' \
  -d '{"workspace":"Research Notes"}'

curl -X POST http://localhost:8100/runs/corpus/reset \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: reset-research-notes-1' \
  -d '{"workspace":"research_notes"}'
```

`GET /workspaces` orders by workspace ID and pages with `limit` (default 50,
maximum 100) plus a sealed cursor. Access filtering happens after catalog
paging; the cursor is encrypted, so it names no row its caller cannot see. The
response contains `workspaces`, `records`, and `next_cursor`. MCP
`list_workspaces` returns only the first 50 plus `has_more`.

Corpus deletion, reset, and Workspace Delete are durable Run actions rather than
direct mutation routes. Every Corpus Mutation, ingest and uploads included,
needs a Workspace the catalogue lists: one never created, or deleted, is refused
before anything is staged, and an unreadable catalogue refuses with 503. REST
and MCP refuse it as not found (404). The Web files routes (listing, upload,
delete, and failed-file recovery) answer 409 from their own catalogue check,
which the page reports as a Workspace that is no longer available. A retried idempotency key
returns its Run first for a delete, retry, reset, Workspace Delete, or remote
ingest; a local source or upload is identified by the bytes it stages, so a
retry after its Workspace is gone is refused like a new submission. A Run
accepted while its Workspace was being deleted runs after that delete and
fails as `workspace_not_found` before any effect. Reset preserves the Workspace
registry, access scope, and history.
Workspace Delete also removes the Workspace from the catalogue and cancels
mutations queued behind it; Runs and Conversations stay. Either may supersede
one mutation waiting for repair. Web
Files uses a workspace-bound sealed keyset cursor, defaults
to 50 files (maximum 100), and orders by `updated_at DESC, id ASC`; processed
rows appear only after Product Document finalization. An ingest finalizes each
document as soon as LightRAG has settled it, so a multi-document ingest's rows
appear one by one while its other documents are still processing. The
failed-files view remains administrative and available for repair regardless of
publication.

## Model Catalogue And Profile Memory

| REST route | Contract |
|---|---|
| `GET /models/catalogue` | Effective runtime overlay with revision. |
| `PUT /models/catalogue` | Upsert one complete endpoint profile under optimistic revision; requires `model_catalogue.write`. |
| `DELETE /models/catalogue` | Remove one overlay entry under optimistic revision; requires `model_catalogue.write`. |
| `GET /memory` | Newest-first active records with `limit` 1–100 and sealed cursor; each names its `kind` and `origin_kind`. |
| `POST /memory` | Remember one owner-scoped Profile Memory record of kind `preference`, `fact` or `experience`; 422 for a `supersedes_id` that is not an id. |
| `POST /memory/fuse` | Replace 2–10 active records with one supplied sentence; `kind` is required when they differ; 422 for a `memory_ids` entry that is not an id. |
| `GET /memory/runs/{run_id}/changes` | What one Run's writers left in the owner's Profile Memory that is still active, oldest first: `{changes: [{memory_id, kind, body, action, change_id, writer}]}`. `action` is `add`, `update` (a remember that named a predecessor) or `fuse`, read from the operation journal row behind the record; `writer` is `agent` for a record the Agent's own memory tool wrote in the Run and `conversation` for one the Session Memory Pass wrote after it. A record since forgotten, corrected or merged away is not listed, nor is one an undo restored (an undo of an undo is refused), nor one whose journal row has aged out of the retention window, since nothing could undo it. Owner-scoped: another owner's records under the same `run_id` are never listed. Empty, not refused, while the owner's Profile Memory is off. |
| `DELETE /memory/{memory_id}` | Forget a record. 422 for a `memory_id` that is not an id. |
| `POST /memory/changes/{change_id}/undo` | Apply the compensating undo. 422 for a `change_id` that is not an id. |
| `GET|PUT /memory/settings` | Read/change the owner capability switch. |
| `POST /memory/clear` | Clear owner records; returns 204. |

A memory or change id is a UUID, and one that is not is refused with the shared 422 before storage,
on the Web routes as on the REST ones. A well-formed id no record has is not a 422 but a receipt
whose `outcome` is `unchanged` (forget) or `conflict` (undo, fuse, remember with a `supersedes_id`).

The Web adapter exposes the same paginated `GET /web/api/memory` contract for
the Settings memory list, which labels a record the Session Memory Pass wrote
(`origin_kind` `session_pass`) as learned from a conversation; per-record forget
and undo use the existing Web mutation routes and owner policy. Settings opens
independently of Memory reads. `GET /web/api/memory/runs/{run_id}/changes` is
the same read as the REST route above, for the notice under an answer: its
`change_id` is what `POST /web/api/memory/changes/{change_id}/undo` takes for an
`update` or a `fuse`, and an `add` is taken back with `DELETE /web/api/memory/{memory_id}`.
A record the pass wrote names the Run whose success scheduled it in its
provenance `run_id`, which is what lists it under that answer.

A fuse names its sources and the sentence that replaces them; the store never
words it. The receipt's `memory_ids` lists the new record first and the
superseded sources after it, and its `supersedes_id` is null. A malformed
request — fewer than two or more than ten ids, a repeated id, a body outside
1–500 characters, or mixed kinds without `kind` — is refused with 409; a source
that is not the owner's or not active, or a body another active record holds,
returns `outcome: "conflict"` and changes nothing.

Memory cursors are sealed and owner-independent as tokens; owner scope remains
an authenticated query predicate. Invalid cursors return 422 before storage.
When Memory is disabled, mutation/recall operations are unavailable except
reading/changing the setting.

A record's `kind` says whose statement it is and how it is recalled: a
`preference` is what the owner wants of an answer, a `fact` what the owner is
like, and an `experience` what the agent learned to be true about its own work.
Recall injects two labeled sections, every line naming its kind and when the
record was last written, to the hour in UTC (`- (preference, 2026-10-17 14:00) ...`);
the block's title says once that times are UTC. **Relevant to
this request** is filled first: the ten best records of any kind with evidence
that they bear on the question — a word-for-word restatement, a shared content
word, or embedding similarity at or above the model's calibrated relevance
floor — within half the budget. **Standing memories** apply to every answer:
the newest preferences and experiences, whatever the question, in what the
budget has left and leaving out any record already shown as relevant. A
standing record crowded out by newer ones is not deleted; a question that bears
on it brings it back as relevant, and the block ends by saying how many
standing memories it did not show. Records are indexed for BM25 the way
corpus chunks are: the corpus language classifier labels each fact, and that
language's profile analyzes it (jieba for Chinese, stemming and stopwords for
English, German and the other profile languages), so function words never count.
A question is searched in every language the owner's facts are in, so a Chinese
question naming Python meets an English fact. Without pg_jieba Chinese facts are
indexed under `simple`; without pg_textsearch there is no BM25 index, and facts
need an exact restatement or embedding similarity. Only `voyage-multimodal-3.5` has a floor today,
0.35, calibrated at 1,024 dimensions; with other models, and whenever the query
embedding misses its two-second deadline, facts are recalled by their words.
A record the Session Memory Pass wrote from the owner's own words carries them, a short
verbatim excerpt of at most 120 characters in the language the owner wrote, and recall shows
them on a second line under the memory, with the Agent's `memory_lookup` as with the injected
block: `  owner said: "…"`. Nothing else about the line changes, and the id a lookup adds stays on
the first. Both sections share an 8,000-character budget that counts each rendered line, its
kind and time label included, and for a record that has one its quote line, the words and 17
characters of framing: a record with a full quote costs up to 120 characters more, so fewer
records fit. The Agent's `memory_lookup` tool and the
dlightrag-memory MCP `memory_lookup` also list the ten newest other memories
with their ids, so an agent can correct a record that recall did not find
relevant; the Agent's lines carry the same time after the kind, and each MCP record
is `{memory_id, kind, body, written_at_utc, quote}`, `written_at_utc` being that time (`YYYY-MM-DD HH:00`,
null when the record has no timestamp) and `quote` the owner's words or null. Every section
lists oldest first. The quote is written by the pass alone: `memory_remember`, `POST /memory`,
`POST /memory/fuse` and the MCP write tools take none.

The Agent's memory tools are `memory_remember`, `memory_forget`, `memory_undo`
and `memory_lookup`; a Child holds only `memory_lookup`. `memory_forget` of an id no
memory has, or of a body no active memory has exactly, is an error that points to
`memory_lookup`; of a memory already forgotten or replaced it says so and is not.
`memory_undo` names an active record by id or exact body and compensates the change behind
its current form. Of an id no memory has, or a body no active memory has exactly, it is an
error that points to `memory_lookup`, and so is an undo of a memory already forgotten or
replaced, which it says is no longer active. A `memory_id` of `memory_forget` or `memory_undo`,
or a `supersedes_id` of `memory_remember`, that is not an id at all is refused as invalid
arguments before storage, with the same pointer to `memory_lookup`; the tool schemas stay
plain strings, because a Run's tool schemas are pinned when it is accepted. MCP adds
`memory_fuse`, which the Agent never holds, and its `memory_undo` names a change id.

## Runtime Policies

A Runtime Policy is a setting an administrator changes while the deployment runs. It is kept
in PostgreSQL and read on every request that depends on it, so a change holds on the next
request of every API process, with no restart and no rebuild. It is not in `config.yaml`: a
setting there is read when a process starts ([Configuration](configuration.md#configuration-ownership)).
The Web is its only surface.

| Web route | Contract |
|---|---|
| `GET /web/api/runtime-policies/{key}` | A policy as it stands: `{key, policy, updated_at, updated_by}`. `updated_at` (UTC ISO 8601 with a `Z`) and `updated_by` are null while nobody has written the policy and it holds its code default. A policy lists the people it targets, so reading one needs `runtime_policy.manage`. |
| `PUT /web/api/runtime-policies/{key}` | Replace the policy's whole document and answer what is kept, in the shape `GET` answers. Whoever writes last wins. Needs `runtime_policy.manage`; 404 for a key the application does not register, which is never stored; 422 for a document that does not fit the policy. |

`runtime_policy.manage` is deployment-wide, and `common_user` does not hold it: `admin`
over every workspace, or a rule that names the action over every workspace, does
([Security](security.md#actions-and-presets)). The caller's user id is kept as `updated_by`.
A stored document that no longer validates answers 503 on this `GET` and wherever its policy
decides a request, until an administrator `PUT`s a valid one: the rating routes for
`answer_feedback`, and a new Answer submission for the other two. A keyed replay of an Answer
that was already accepted is answered without reading a policy. The bootstrap keeps the page
loading, offering no thumbs and no effort.

Three policies are registered:

| Key | Value | Code default | Decides |
|---|---|---|---|
| `answer_feedback` | `true` or `false` | `true` | Whether an owner is offered the thumbs under an answered message ([Answer Feedback](domain-language.md)). |
| `agent_effort_ceiling` | `"low"`, `"high"` or `"max"` | `"max"` (no ceiling) | The highest agent effort an Answer's agents may run at, for the callers the policy reaches (see [Request Fields](#request-fields)). |
| `active_answer_runs` | a whole number of 1 or more, or `null` | `null` (no limit) | How many Answers one owner may have queued or running at once. |

A targeted policy's document is a default and ordered rules:

```json
{
  "default": true,
  "rules": [{"claim": "email", "value": "someone@example.com", "then": false}]
}
```

`then` has the policy's value type: a bool for `answer_feedback`, a level for
`agent_effort_ceiling`, and a count or `null` for `active_answer_runs`, where `null` is
unlimited for the callers the rule names (`{"default": 3, "rules": [{"claim": "email", "value":
"lead@example.com", "then": null}]}` limits everyone to three but one person). A rule applies to a
caller whose verified JWT claim `claim` is `value`, or holds it among several values, the way an
Access Rule matches. The first rule that matches decides, and the default holds for a caller no
rule matches, so a policy can be on for everyone but some people or off for everyone but some
people. A deployment that verifies no JWT (`none`, `simple`) cannot tell its callers apart, so only
the default applies there. A document holds at most 100 rules, a `claim` of 1-64 and a `value` of
1-256 characters (trimmed), and nothing else: a field this paragraph does not name is refused. A
policy nobody has written holds its code default, `{"default": true, "rules": []}` for
`answer_feedback`.

The two Answer policies are read when an Answer is accepted, once, and the Run keeps what they
said: a change reaches the next submission of every process and never a Run already accepted
([ADR 0042](adr/0042-runtime-policies-are-read-at-the-door-and-pinned-into-the-run.md)).

**The active-run limit** counts the owner's Answer Runs in status `queued` or `running`, a Run
deferred for a dependency included, under the lock that admits the Run. At the limit a new
submission is refused before anything is stored: HTTP 429 `{detail, error_type: "unavailable",
error_kind: "active_run_limit"}` on REST, the same error kind in the Web command envelope
(`kind: "service_unavailable"`, which the browser retries with the draft kept), and `Error
[active_run_limit]: …` on MCP. A keyed retry of a submission that was accepted is answered while its
owner is at the limit, a limit lowered below what an owner holds cancels nothing, and a Run that has
ended makes room for one more. The owner is the identity the deployment projects a caller to, so
under `none` and `simple` authentication, where every caller is the deployment's one owner, the limit
is the deployment's: how many Answers it may have in progress. It bounds Answers only, and not
Retrieval or Corpus Mutations, and it is separate from the lane's
[admission limit](run-runtime.md#admission-limits).

## Schedules

A Schedule is an owner's durable record that starts an Answer Run on the clock
([ADR 0103](adr/0103-a-schedule-starts-a-run-on-the-clock-and-the-database-is-its-queue.md)). The
owner creates, edits, pauses, resumes, runs and deletes it here. The Agent makes, changes, pauses,
resumes and deletes them through its own tools on a turn a person started, under the same rules and
never with a goal or a token ceiling. Every route is owner-scoped: a Schedule that is not the caller's is as unknown as one that
does not exist (404), and creating, editing or resuming one makes the caller the identity its
fires act as from then on.

| REST route | Contract |
|---|---|
| `GET /schedules` | `{schedules: [Schedule], default_max_tokens_per_day}`: the owner's Schedules, and the daily token ceiling a new one starts with. |
| `POST /schedules` | Create an active Schedule; 201 with the Schedule. |
| `GET /schedules/{schedule_id}` | One Schedule. |
| `PATCH /schedules/{schedule_id}` | Change the parts named; 200 with the Schedule. |
| `DELETE /schedules/{schedule_id}` | Delete the Schedule and its Fire history; 204. The conversations its Runs wrote stay. |
| `POST /schedules/{schedule_id}/pause` | Stop it firing until the owner resumes it. Pausing a paused Schedule changes nothing. |
| `POST /schedules/{schedule_id}/resume` | Let a paused Schedule fire again, as the caller. It replays no slot it slept through beyond the one late fire its grace window allows. |
| `POST /schedules/{schedule_id}/run` | Fire once now, under every rule a clock fire meets. Answers the Fire this made, whose `state` says what became of it. |
| `GET /schedules/{schedule_id}/fires` | `{fires: [Fire], next_cursor}`, newest slot first, with `limit` 1-200 (50 unless given) and the `next_cursor` of the page before. |
| `POST /schedules/preview` | `{zone, slots: [{at, local}]}`: the next five slots of a cadence, `at` in UTC and `local` on the zone's own clock with its offset. Nothing is stored. |

A Schedule reads `{schedule_id, name, prompt, mode, workspaces, target, conversation_id, zone,
cron, at, max_tokens_per_day, goal_max_rounds, state, pause_reason, next_fire_at, created_at,
updated_at, changed_by}`. `mode` is `research` or `fast`; `target` is `new_conversation` or
`same_conversation`, and `conversation_id` is the one conversation a `same_conversation`
Schedule keeps once its first fire has opened it. The cadence is `cron`, five fields read on the
wall clock of the IANA `zone`, or `at`, one UTC instant; the other is null. `state` is `active`,
`paused` or `completed` (a single time that has fired); `pause_reason` says why the platform paused
one, and is null when its owner did,
and `next_fire_at` is null once nothing is left to fire. `goal_max_rounds` is null unless each
fire starts a goal whose objective is the `prompt`. `changed_by` is `owner` or `agent`: who last
made, edited or resumed it, so what the Agent wrote can be told from what its owner did (pausing,
deleting and running now leave it). Every time is UTC ISO 8601 with a `Z`. A Schedule never shows
the identity it acts as.

`POST /schedules` takes `name`, `prompt`, `workspaces` (at least one), `zone`, and exactly one of
`cron` or `at` (an instant that carries its offset); `mode` (default `research`), `target`
(default `new_conversation`), `max_tokens_per_day` (default `default_max_tokens_per_day`) and
`goal_max_rounds` are optional, and a goal Schedule runs in `research`. `PATCH` takes any of
`name`, `prompt`, `mode`, `workspaces`, `zone`, `cron`, `at`, `max_tokens_per_day` and
`goal_max_rounds`. Naming `cron` or `at` replaces the cadence and naming only `zone` moves its
recurrence to the new zone; any change of cadence recomputes `next_fire_at` from now. Whether a
Schedule starts a goal, and its `target`, are fixed when it is made, and a completed Schedule
takes no edit. A field these routes do not name is refused.

A Fire reads `{fire_id, schedule_id, slot, state, reason, run_id, conversation_id,
missed_count, last_slot, manual, last_error, created_at, updated_at}`. `state` is `pending`, `started`, `succeeded`,
`failed`, `cancelled`, `refused`, `skipped_busy`, `skipped_limit` or `missed`; `reason` says why
a Fire that started nothing did not, or what a failed Run said; `run_id` and `conversation_id`
name what a started Fire began; and a `missed` Fire stands for `missed_count` slots from `slot`
to `last_slot`. `manual` marks a Fire the owner asked for with Run now; `last_error` is what
stopped the latest attempt to start a Fire that is still `pending`. A Schedule keeps its latest
`schedules.fires_kept` Fires (200 by default).

The workspaces a caller names are authorized for `workspace.query` as they are for an Answer: a
workspace the caller may not query is a 403 and nothing is stored, and access is read again at
every fire. A refusal of what the owner must change (a cadence that does not parse, is closer than
the minimum interval, or is not in the future; an unknown zone; a goal in `fast` mode; a field
outside its bounds) is a 422; a change the Schedule's state does not allow (editing, pausing or
resuming a completed one, or asking to run one while its previous Fire is still under way) and the per-owner
cap on active Schedules (`schedules.max_active_per_owner`, counted when a Schedule is created or
resumed) are a 409 with the reason. `POST /schedules/preview` holds a cadence to every rule saving
it would, so its 422 is the refusal the owner would meet there.

The Web adapter serves the same operations, answering the same shapes, under
`/web/api/schedules` for the sidebar's Scheduled row and its dialogs: `GET`/`POST /web/api/schedules`,
`GET`/`PATCH`/`DELETE /web/api/schedules/{schedule_id}`, `POST` `.../pause`, `.../resume` and
`.../run`, `GET .../fires`, and `POST /web/api/schedules/preview`. Its writes carry the Web's
double-submit token like every other owner write.

## Health And Errors

`GET /health` is liveness: it returns in-process state without model, parser,
corpus, or database I/O. Its bounded `components` map distinguishes `process`,
`operational_state`, `run_coordinator`, `cancellation_listener`,
`corpus_storage`, `parser`, `providers`, and `agent_workspace`; details and
warnings use fixed sanitized text. The four LightRAG storage class names and
`answer_image_capability` are also reported, as is `agent_shell_confinement`: what an
Agent's processes can be confined to on this host, one of `disabled` (no Agent
environment), `unavailable` (no kernel seam), or `landlock:abiN`. `agent_browser`
reports `{"state": "configured" | "disabled", "endpoints": <n>, "sandbox": <bool>,
"accounts": <bool>, "registration": <bool>, "mailbox": <bool>}` from configuration alone:
`sandbox` is the configured `chromium_sandbox`, whether browsers launch inside Chromium's own
sandbox, and not whether a host can start it; `accounts` and `mailbox` say whether Agent
Accounts and an Agent Mailbox are composed, which the first is wherever the browser is, and
not whether the bucket answers; `registration` is the deployment's allowance for the Agent to
register, the same answer as `registration.allowed` of the
[Agent Accounts view](#web), which no owner's own switch exceeds. It never
names an endpoint and never reaches the pool or the bucket, so a pool that is down is seen in
its containers' own health, not here ([Operations](operations.md#agent-browser-pool)), and a
mailbox that cannot be read is seen as the `inbox` call's reason
([Agent Mailbox](operations.md#agent-mailbox)). Degraded state remains HTTP 200.

`GET /ready` checks only the authority required to durably admit and coordinate
Runs: Application Operational State plus the injected writable Operational
State database probe. It does not probe the reader corpus or external vector,
parser, or model providers. The endpoint returns fixed-detail HTTP 503 when
that authority is unavailable. Readiness checks are single-flighted and
memoized for two seconds.

General errors are `{detail, error_type, error_kind?}` where `error_type` is
`validation`, `auth`, `not_found`, `conflict` (HTTP 409 and 412),
`unavailable`, `configuration`, or `internal`. A request that fails validation
answers 422 whose `detail` names each invalid field and why, as
`body.query: Field required`; submitted values are never
echoed back. That includes the parts a route reads by hand: multipart fields
(`body.file`, `body.metadata`), event cursors (`query.after`,
`header.Last-Event-ID`), and a missing `Idempotency-Key`
(`header.Idempotency-Key: Field required`). OpenAPI documents every 422 with
this `ErrorDetail` body, not FastAPI's list-shaped default. A refusal of
well-formed input keeps the status documented beside its route, such as the
Corpus Mutation refusals above (400). REST and MCP Run acceptance
answer a reused idempotency key with `Idempotency key was reused with a
different request` (HTTP 409 on REST); browser commands use their own envelope
below. Stable answer error kinds are:

- `CURRENT_IMAGES_UNSUPPORTED`, `CURRENT_IMAGE_LIMIT_EXCEEDED`,
  `ANSWER_IMAGE_CAPABILITY_UNKNOWN`, `ANSWER_INPUT_OVERFLOW`,
  `unsupported_resource_capability`, `ANSWER_RESOURCE_INVALID`, and
  `UNSUPPORTED_ATTACHMENT_TYPE`;
- `invalid_tool_configuration`, `unsupported_answer_mode`, `routing_failed`,
  `tool_contract_changed`, `run_abandoned`, `workspace_integrity_error`, and
  `run_execution_failed`;
- `ANSWER_STREAM_FAILED`; and
- `active_run_limit`, a refusal at acceptance, never the end of a Run: HTTP 429 on REST and the
  Web, where the caller already has as many Answers in progress as the `active_answer_runs`
  [Runtime Policy](#runtime-policies) lets them.

Stable top-level Retrieval terminal error kinds are `retrieval_timeout`,
`retrieval_failed`, `retrieval_model_changed`,
`retrieval_context_policy_changed`, `retrieval_input_incompatible`, and
`retrieval_input_missing`. `run_abandoned` is common to both durable Query kinds.
`dependency_unavailable` ends any Run, Corpus Mutations included, whose
dependency outages already deferred it ten times. A Corpus Mutation defers when
the document parser is unavailable, once every document it attempted has
settled; resumed, it retries only the documents that did not become ready (see
[Parser Services](operations.md#parser-services)).

Internal exception text and schema detail are not public.

A lane at its [admission limit](run-runtime.md#admission-limits) rejects a new
Run with HTTP 503 before inserting it; REST, MCP, and same-origin browser
commands report the retriable message
`Deployment-wide nonterminal admission limit reached`. (An owner at their own
`active_answer_runs` limit is refused with HTTP 429 and `active_run_limit`
instead, even when the lane has room.) Queue residence has no
application timeout. Top-level Retrieval applies
`corpus.retrieval.timeout` only during claimed
execution and reports `retrieval_timeout` terminally. Explicit transient corpus
or provider interruptions defer Retrieval and Answer, and an Agentspace
volume that does not answer defers Answer, with a durable bounded
retry checkpoint, release Query compute capacity, and later resume the same
Run; the eleventh would fail it as `dependency_unavailable` instead. Authentication, unsupported configuration/schema, invalid input,
deterministic model rejection, context overflow, and unknown exceptions remain
terminal. Attachment total-byte
overflow returns HTTP 413 before buffering. Generic rate, connection, and
volumetric controls belong at ingress;
see [Security](security.md#ingress-responsibilities).
