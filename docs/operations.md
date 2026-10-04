# Operations

This document owns maintenance and recovery runbooks. PostgreSQL tuning lives in
[PostgreSQL](postgresql.md); fields/defaults live in
[Configuration](configuration.md). These commands are not normal query traffic.

Deployments follow the
[configuration ownership and container contract](configuration.md#container-and-kubernetes-contract).

## Full Development Reset

`scripts/reset_development.py` erases the complete development environment:
PostgreSQL data/migration ledger, LightRAG corpus/KG/vector/status data,
DlightRAG Runs/answers/Web state, and local runtime/corpus files. It is never
exposed through REST, Web, or MCP.

```bash
# Read-only previews; safe while services run.
uv run scripts/reset_development.py --mode docker --dry-run
uv run scripts/reset_development.py --mode native --dry-run

# Docker: remove app volumes, start only empty PostgreSQL, verify extensions.
uv run scripts/reset_development.py --mode docker
# or
make dev-reset

# Native: replace the dedicated database public schema and empty working files.
uv run scripts/reset_development.py --mode native
```

Interactive runs require the exact database name. `--yes` skips only that
prompt, never target validation. Native mode refuses non-loopback hosts unless
`--allow-remote-reset` and other sessions unless `--force-disconnect`. Docker
reset leaves API/MCP/readers/writers stopped; start one writer separately so it
creates baseline schema.

Rebuilding the PostgreSQL image does not change an existing data volume:
`postgres/init.sql` creates the extensions only in a new one. After a rebuild
changes a pinned extension version, run the Docker reset.

This differs from `scripts/reset_workspace.py`, which resets authorized Corpus
Workspaces through Reset Runs of an application it starts itself. Neither
delegates to the other.

## RunRuntime And Durable Query And Corpus Mutation Runs

- Start a writer before readers: the writer creates the schema, and readers
  only validate it. Workers sharing a database must run the same model roles,
  Agent execution mode, Answer policy, and `answer.agent.connections` policy;
  each owner enables their own Connections.
- Shared mounts, worker capacity, and recovery after a crash or shutdown are in
  [Workers And Scaling](run-runtime.md#workers-and-scaling). Monitor
  `dlightrag_runs`, `dlightrag_run_events`, `dlightrag_blobs`, and
  `dlightrag_blob_chunks` alongside the
  [admission limits](run-runtime.md#admission-limits).
- Route traffic with `GET /ready`, and use `GET /health` for liveness and the
  degradation view ([Interfaces](interfaces.md#health-and-errors)). A corpus or
  provider outage does not remove readiness: acceptance continues up to the
  lane's admission limit, and accepted Runs defer durably. Nor does it stop
  startup: a default workspace that cannot be built because corpus storage, the
  model provider (its image embedding probe), or the parser is briefly
  unavailable starts the process with that component degraded in
  `GET /health`; any other failure refuses to start.
- A process whose Run cancellation listener is not ready within 30 seconds of
  startup (PostgreSQL LISTEN or the first cancel-pending rescan keeps failing)
  reports not ready, with `cancellation_listener` degraded, and keeps waiting:
  once the listener is ready it starts claiming Runs and becomes ready without a
  restart. It never claims a Run before then.

`make validate-runtime` runs the RunRuntime failure matrix (`runtime-faults`,
which needs a reachable PostgreSQL), the PostgreSQL 18 convergence gate
(`runtime-pg18`), and the [load campaign](run-runtime.md#load-evidence), all with
fake models and no external parser.

Retention needs no cron; what it removes and when is in
[RunRuntime](run-runtime.md#retention).

`dlightrag-workspace-audit` reports an Agent Workspace root without deleting
anything: it counts Run roots and names those whose Run row is gone, which the
runtime's hourly orphan sweep removes. Pass `--root` to audit a path an earlier
configuration left behind, and `--sample` to bound the report. A deployment
whose execution is `disabled` and which names no root has nothing to audit.

## Release Distribution

DlightRAG ships as a repository, not as a package: a deployment clones this
repository and runs Compose or a native process, or adds the clone as an
editable path dependency. No workflow publishes DlightRAG to a package index,
and `v*` tags only mark releases on GitHub. `make release-check`, which
`make ci` runs, enforces lockstep versions across `pyproject.toml`,
`packages/memory/pyproject.toml`, `frontend/package.json`, and the Memory
runtime. `make workspace-wheels` is a local packaging check that installs the
built wheels in isolation; it distributes nothing.

## Agent Chart Rendering

The built-in [`charts`](../src/dlightrag/engine/agent/builtin_skills/charts/SKILL.md)
Skill has the Research agent write an Apache ECharts option and pipe it to
`echarts-render`. The command draws the option as an SVG with ECharts on the
image's `node` and rasterizes the SVG with `resvg` into a PNG; `--svg` adds a
vector file and `--html` a self-contained interactive page. The house theme
always applies, and Noto Sans SC is the only font `resvg` loads, so a font name
an option writes still draws.

The image adds, all under `/usr/local`:

| Path | Content |
|---|---|
| `bin/echarts-render` | Symlink to `lib/echarts-render/echarts_render.py` |
| `lib/echarts-render/` | The renderer from `chart-render/` (`echarts_render.py`, `ssr.cjs`, `theme.json`), `echarts.min.js`, and the ECharts and zrender licenses |
| `bin/resvg` | The upstream resvg CLI, built from its crates.io release; its licenses are in `share/doc/resvg/` |
| `share/fonts/noto-sans-sc/` | Noto Sans SC Regular and Bold, with the OFL-1.1 license |

Rust exists only in the builder stage. The font covers Chinese and Latin text;
emoji, Korean and Arabic draw as boxes, and the command says so in a note.

The pins to bump are `RESVG_VERSION` and the `rust:` builder tag in the
`Dockerfile` (the tag must meet the crate's minimum Rust version); `echarts` in
`chart-render/package.json`, followed by `npm install --package-lock-only` in
that directory to refresh `package-lock.json`; and `NOTO_CJK`, one noto-cjk
commit, together with the three `sha256` checksums taken from it.

After a bump, build the image. Its smoke test renders a two-bar chart with a
Chinese title as the `app` user and fails the build when node, ECharts, `resvg`
or the font is missing or unreadable, so a broken renderer never reaches a Run.
CI builds no image: the test runs wherever the image is built. Then render a
bar chart, a horizontal bar chart with long Chinese category names and a line
chart from the new image and look at the PNGs. A new ECharts release can change
defaults, and `resvg` exits 0 after dropping text it cannot match to a font, so
`echarts-render` treats its `No match for` warning as a failure.

## Parser Services

The parser block is configured under
[Parser Sidecars](configuration.md#parser-sidecars). Host MinerU:

```bash
make mineru-install
make mineru-service-install
make mineru-service-status
make mineru-service-logs
make mineru-service-stop
```

Use `make mineru-api` for foreground operation where a background user service
is unavailable. MinerU title correction is best-effort: the launcher bounds each
provider attempt by `MINERU_TITLE_AIDED_ATTEMPT_TIMEOUT_SECONDS` (default 60)
and stops after `MINERU_TITLE_AIDED_MAX_ATTEMPTS` (default 2), then continues
parsing without corrected title levels. Set overrides in `.env.mineru` and
restart the service; diagnostics never log the title-aided API key.
`make mineru-title-aided` asks DlightRAG's model catalogue how the title
model's endpoint turns reasoning off and stores those request fields in
`~/mineru.json`; re-run it after changing the endpoint or model.

Optional Compose Docling CPU:

```bash
docker compose --profile docling up -d
```

Point the [`docling` block](configuration.md#docling) at it. It publishes only
`127.0.0.1:5001`; do not run it beside a host Docling service on the same port.
Independently managed Docling endpoints are also supported.

LightRAG's MinerU or Docling client makes every parser request, whichever parser
a document is routed to. When the parser's host name does not resolve (Compose
stops resolving a stopped or restarting service), the parser service or a proxy
in front of it refuses or resets the connection, a connect/read/write timeout
expires, the connection drops, or the service answers HTTP 408, 425, 429, 500,
502, 503, 504, 520-524, or 529, the document is recorded with the fixed error
`Document parser is temporarily unavailable` and the underlying client error is
logged as a warning. Only the response status and the kind of transport failure
decide this, never error text, so a file or endpoint name cannot turn an outage
into a rejection. Every other parser failure keeps LightRAG's own message: a 4xx
rejection, a conversion the parser reports as failed, an exhausted polling
budget or download deadline, an oversized or malformed result bundle, and a
misconfigured endpoint (a TLS certificate that fails verification, or a TLS
protocol mismatch such as an https URL for a plain-HTTP service).

Any other parser failure leaves its document failed; retry failed documents as
below once the cause is fixed. A parser outage instead defers the Corpus Mutation,
since the same document may well parse a minute later. Ingestion first settles
every document the Run attempts, a remote source's later windows included, so
LightRAG tracks all of them, and publishes each one that becomes ready as it
settles; then the Run defers with the usual backoff and
`GET /health` reports the `parser` component degraded. When it resumes, it
settles each tracked document from its durable state: a document that became
ready stays as it is, without being parsed again, and every other document is
retried, including one that failed for its own reason beside the outage. If that
one fails again once the parser is back, the Run fails with it; when the Run
completes, `parser` is reported healthy again. A retry Run stops at the outage
and resumes its cohort the same way. The Run's ten dependency deferrals bound
this, because a document that crashes or exhausts the parser service (an
out-of-memory kill, for example) looks like an outage on every attempt: the
eleventh fails the Run as `dependency_unavailable`.

## Product Document Finalization And Failed Ingestion Cleanup

A LightRAG `processed` status alone does not publish a Product Document: only a
true finalization marker makes it visible
([visibility](retrieval-answer.md#product-document-visibility-and-metadata-in-filtering)),
and how each mutation sets that marker is in
[Corpus Mutations](run-runtime.md#corpus-mutations). A failure in
metadata/source finalization, BM25 labeling, required retained source/sidecar
work, or enabled visual fusion leaves the marker false while LightRAG's status
stays `processed`. Re-ingest the same retained source to replay only the
idempotent same-ID finalizers; never set the marker by hand. Documents written
directly through LightRAG have no completion proof and stay excluded.

A retry, or an ingest that resumes after LightRAG processed a document
DlightRAG had not finished, fails such a document at once when it cannot be
replayed: its source file is gone (`the source file is no longer available`),
or its stored source metadata is incomplete or invalid (`source metadata
incomplete`, `source metadata invalid`). The document stays hidden; restore its
source and retry it, or delete it. An upload or a local source keeps its copy in
`<working_dir>/corpus/<workspace>/__local_sources__`, under its file name.

Failed documents are terminal and are not automatically retried. First inspect
the workspace:

```bash
curl 'http://127.0.0.1:8100/files/failed?workspace=personel'
```

If the stored source/download locator is still available, retry every failed
document with the currently configured parser:

```bash
curl -X POST http://127.0.0.1:8100/runs/corpus/retry \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: retry-personel-failed-1' \
  -d '{"workspace":"personel","selector":"all_retryable"}'
```

If retry is unwanted or the source is unavailable, accept exact durable
deletion by filename:

```bash
curl -X POST http://127.0.0.1:8100/runs/corpus/delete \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: delete-personel-failed-1' \
  -d '{"workspace":"personel","filenames":["failed.pdf"]}'
```

Deletion hides the document before LightRAG deletes it
([Delete](run-runtime.md#delete)), then removes its status, full document,
metadata, chunks, vectors, and graph entries where present, its source file,
and its `.parsed`/`.mineru_raw`/`.docling_raw` directories.

### Repairing An Ambiguous Mutation

If Run status reports `phase=waiting_for_repair`, do not submit a replacement
mutation and do not rewrite the Run row. Inspect `repair_reason` and
`repair_remedy`, repair or verify LightRAG's authoritative state, then resume
the same Run as a caller that holds the Run's action permission:

```bash
curl -X POST http://127.0.0.1:8100/runs/$RUN_ID/resume
```

If repair is inappropriate, accept a Corpus Reset or Workspace Delete that names
the waiting Run as `supersedes_run_id`. What resuming and superseding do is in
[RunRuntime](run-runtime.md#cancellation-repair-and-supersession).

## Workspace BM25 Rebuild

`dlightrag-rebuild-bm25` creates configured pg_textsearch indexes and refreshes
`dlightrag_bm25_language` for existing chunks. It neither parses, calls models,
rebuilds vectors, nor changes sources.

Run it after enabling BM25 on an existing corpus or changing BM25 profiles,
`k1`, or `b`:

```bash
# Stop every API, MCP, ingest, and reader process using the workspace.
uv run dlightrag-rebuild-bm25 --yes

# With an explicit environment file:
uv run dlightrag-rebuild-bm25 --env-file /absolute/path/to/.env --yes
```

The configured role must be `writer` and BM25 must be enabled. Restart services
only after completion. `--batch-size N` bounds language-label transactions.
With BM25 enabled, a successful `chunks` or `all` vector rebuild runs this
maintenance too.

`dlightrag-rebuild-bm25` and `dlightrag-rebuild-vdb` address
`deployment.workspace` by the canonical id the service stores that workspace
under (`deployment.workspace_id`), so the label `My Space` rebuilds
`my_space`. A label with no canonical id (empty, or longer than 64 characters
once normalized) is refused when the configuration loads, so neither command
opens any storage for it.

## Offline Vector Storage Rebuild

`dlightrag-rebuild-vdb` rebuilds LightRAG vectors from existing graph/chunk rows
using the configured workspace, embedding model, BM25 labels, and visual
alignment. It does not ingest/parse files or create document status.

Use it for missing or stale vector rows, a failed `check`, or an intentional
embedding change whose vector schema already supports the dimension. Use
failed-file retry—not this command—for ingestion failures.

| Target | Writes | Behavior |
|---|---:|---|
| `check` | no | Compare graph records with entity/relation vectors |
| `graph` | yes | Rebuild entity and relationship vectors |
| `chunks` | yes | Rebuild chunk vectors, labels, and fused visual alignment |
| `all` | yes | Run graph + chunks maintenance |

`graph`, `chunks`, and `all` require `--yes`. Stop every writer to the same
LightRAG storage before them.

### Native

```bash
uv run dlightrag-rebuild-vdb --target check
uv run dlightrag-rebuild-vdb --target all --yes

# With an explicit environment file:
uv run dlightrag-rebuild-vdb --env-file /absolute/path/to/.env --target check
uv run dlightrag-rebuild-vdb --env-file /absolute/path/to/.env --target all --yes
```

| Flag | Meaning |
|---|---|
| `--env-file PATH` | Load explicit environment file |
| `--batch-size N` | Source rows per rebuild batch |
| `--no-restore-sidecar-alignment` | Skip fused visual-vector restoration |

### Docker Compose

```bash
# Check is read-only.
docker compose run --rm dlightrag-api dlightrag-rebuild-vdb --target check

# Stop writers, rebuild in the app image, restart.
docker compose stop dlightrag-api dlightrag-mcp
docker compose run --rm dlightrag-api dlightrag-rebuild-vdb --target all --yes
docker compose up -d dlightrag-api dlightrag-mcp
```

After `chunks`/`all`, DlightRAG refreshes BM25 language labels and replaces
canonical drawing vectors with fused VLM-description+image vectors when direct
multimodal embedding is active. Skip alignment only for diagnosis or an
intentional text-only deployment.

With alignment on, `chunks`/`all` settle whether direct multimodal embedding is
active before writing any vector, running the same image/fusion probe as the
service when `startup_probe` is on. When the probe cannot settle it — the
embedding provider fails transiently, or `input_modality: multimodal` cannot be
honored — the command prints `Nothing was rebuilt: …` and exits 1 with every
vector untouched; run it again once the provider is reachable or the
configuration is fixed. A failure after writing began (a reported rebuild error,
or restoration interrupted by the provider) also exits nonzero, possibly
leaving drawing vectors text-only; rerun the same target, which rewrites the
chunk vectors and restores alignment again.

Before destructive production rebuilds, back up PostgreSQL and use the service's
own `.env`, `config.yaml`, workspace, and model. Inspect any nonzero exit before
restart.

## Connection Key Ring Rotation

Personal Connection credentials are sealed under
`<deployment.working_dir>/connection-keyring.json`, which the first writer
creates ([format](personal-mcp-connections.md#secret-handling-and-key-ring)).

1. Add a fresh 32-byte base64url key under a new ID, keeping the old IDs:

   ```bash
   python3 -c 'import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode().rstrip("="))'
   ```

2. Point `active` at the new ID and restart **all** workers, so none still
   encrypts with the old key. Writer maintenance then re-encrypts live Grants
   ([Operational lifecycle](personal-mcp-connections.md#operational-lifecycle)).
3. Wait until the old key's Grant and OAuth inbox counts reach zero. Inbox flows
   expire within `oauth_timeout` (at most 600 seconds) and are then collected.
   On the deployment database, count envelopes; never select or export them:

   ```sql
   SELECT key_id, count(*) FROM dlightrag_connection_grants
   WHERE encrypted_envelope IS NOT NULL GROUP BY key_id;
   SELECT envelope::jsonb->>'key_id' AS key_id, count(*)
   FROM dlightrag_connection_oauth_flows f
   CROSS JOIN LATERAL (VALUES (f.encrypted_result), (f.encrypted_credentials)) v(envelope)
   WHERE envelope IS NOT NULL GROUP BY 1;
   ```

4. Remove the old key from the ring. A Grant it still sealed would need
   authorization again, and backups stay readable to any retained copy of it.

## Local Langfuse Observability

The repository can run an isolated local Langfuse Compose project in
`../langfuse-local` (override with `LANGFUSE_LOCAL_DIR`).

```bash
make langfuse-up
```

Open <http://localhost:3300> with user `admin@localhost.local`; read the generated
password from `../langfuse-local/.env`. Traces appear after a model call.

| Target | Behavior |
|---|---|
| `make langfuse-stack` | Download/patch the official Compose file |
| `make langfuse-bootstrap` | Sync project credentials to both env files |
| `make langfuse-up` | Bootstrap and start |
| `make langfuse-down` | Stop |
| `make langfuse-restart` | Re-sync and recreate Web/worker |
| `make langfuse-status` | Show containers |
| `make langfuse-logs` | Follow Web/worker logs |
| `make langfuse-health` | Check host endpoint |
| `make langfuse-reset CONFIRM=1` | Delete all local Langfuse data: traces, users, project keys, and model prices |

### Connection And Keys

`scripts/langfuse/headless.py` writes one key pair to the Langfuse
`LANGFUSE_INIT_PROJECT_*` variables and DlightRAG's
`DLIGHTRAG_OBSERVABILITY__LANGFUSE_{PUBLIC,SECRET}_KEY`. Both are required;
without either, tracing is disabled. Initialization variables are read only when
Langfuse first creates its database; rotate established credentials in the UI or
reset the local stack.

Set `observability.langfuse_host` according to where DlightRAG runs:

| DlightRAG | Trace host |
|---|---|
| Docker Compose | `http://host.docker.internal:3300` |
| Native | `http://localhost:3300` |

Browsers and `make langfuse-health` always use `http://localhost:3300`.
`config.yaml` owns the nonsecret host; bootstrap writes only secrets.

Compose reads `.env` only when creating a container. After keys/host change:

```bash
docker compose up -d --force-recreate dlightrag-api dlightrag-mcp
docker compose logs dlightrag-api | grep -i 'langfuse tracing'
```

A log target of `https://cloud.langfuse.com` means the local host setting was
not loaded.

### Cost And Recovery

DlightRAG always reports tokens. Configure matching model prices in Langfuse to
compute cost. OpenRouter can instead return charged cost when enabled only on
its model block:

```yaml
models:
  chat:
    default:
      provider: openai
      base_url: https://openrouter.ai/api/v1
      model_kwargs:
        usage:
          include: true
```

Do not send that vendor-specific option to providers that reject it.

To disable tracing, clear both Langfuse keys and recreate app containers. If the
UI password is lost, read `LANGFUSE_INIT_USER_PASSWORD` from the local env. If
the env file was deleted, `make langfuse-bootstrap` recovers the API key pair
from DlightRAG's `.env`, but cannot change the established UI password.

Last-resort reset, which deletes all local Langfuse data but none of
DlightRAG's; `make langfuse-up` seeds the user and project keys again from the
env files:

```bash
make langfuse-bootstrap
make langfuse-reset CONFIRM=1
make langfuse-up
grep LANGFUSE_INIT_USER_PASSWORD ../langfuse-local/.env
```
