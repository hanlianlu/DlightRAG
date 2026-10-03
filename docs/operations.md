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
  Agent execution mode, Answer policy, `answer.agent.connections` policy, and
  `answer.agent.browser` endpoints (the pool's leases are shared); each owner
  enables their own Connections.
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

## Agent Browser Pool

The Agent Browser ([ADR 0032](adr/0032-the-agent-browser.md)) is a pool of Playwright
containers plus one Squid proxy. The bundled Compose stack runs two members
(`agent-browser-1`, `agent-browser-2`) and `agent-browser-egress`, and binds their
addresses into `dlightrag-api`, `dlightrag-mcp`, and `dlightrag-reader`
([fields](configuration.md#agent-browser); the boundary is in
[Security](security.md#agent-browser-boundary)). No service waits for them: a render
with no browser up fails as `unreachable` and the Run goes on.

```bash
# From the repository root, so the seccomp profile path in docker-compose.yml resolves.
docker compose up -d --build agent-browser-1 agent-browser-2 agent-browser-egress
docker compose ps
docker compose logs agent-browser-egress
```

- **Check.** `GET /health` shows `agent_browser` as `configured` with the endpoint
  count, from configuration alone, so it does not say the pool is up. A member is
  healthy when `docker compose ps` says so; its check asks the run-server for `/json`.
  Which Run holds which member is the `dlightrag_agent_browser_leases` table:
  `SELECT endpoint, run_id, updated_at FROM dlightrag_agent_browser_leases`. A row is
  free when it names no Run, or when its Run no longer holds its lease (not `running`,
  another lease owner or fencing epoch, or expired), whatever the row still says. The
  proxy logs every request to its stdout; a destination it refuses is `TCP_DENIED`.
- **Size.** A Run holds a member only while it renders and for
  `idle_release_seconds` after, so the pool's size bounds how many Runs render at the
  same moment across every process that runs Query workers. When every member is held,
  a render waits up to `lease_wait_seconds` and then fails as `busy`; the model reads
  that and works from the direct read. Add members, or lower `idle_release_seconds`,
  when that is frequent. Each member is capped at `COMPOSE_AGENT_BROWSER_MEM_LIMIT`
  (default `2g`) and 1024 processes.
- **Adding a member.** Add its service (`<<: *agent-browser`) on a network of its own,
  declare that network `internal: true`, add the network to `agent-browser-egress` and
  to `dlightrag-api`, `dlightrag-mcp`, and `dlightrag-reader`, and add the member's
  `ws://` URL to the `endpoints` binding. Never put two members on one network:
  `--unsafe` lets a client that reaches a member choose its browser's launch arguments.
  Every process must be restarted with the same endpoint URLs, spelled identically,
  because the lease table is keyed by the URL.
- **Upgrading.** The Python `playwright` package and the pool image are one version,
  and the server refuses a client of another major or minor version with HTTP 428.
  Bump every pin together: `pyproject.toml` (`playwright==X`), `uv.lock`, the
  `PLAYWRIGHT_VERSION` argument of `agent-browser/browser/Dockerfile`, and the
  `package.json` and `package-lock.json` beside it, plus the image tag in
  `docker-compose.yml`. `make release-check` (`scripts/verify_release_contract.py`)
  fails unless they agree. Rebuild the pool image and restart the pool with the
  application.
- **Troubleshooting.** The application logs `Agent Browser connect failed` at ERROR with
  the endpoint and the error type, never a page URL.
  - `unreachable`: the member is down, the endpoint is misspelled, the application
    service is not on the member's network, or the versions differ (HTTP 428).
  - `busy`: every member's row names a Run that holds its lease. Wait for one to
    finish rendering, or for the lease of a Run whose worker died to expire (about a
    minute).
  - A page that never loads: look for `TCP_DENIED` in the proxy's log. A private
    destination or a port other than 80 and 443 is refused by design.
  - `Chromium sandboxing failed`, or the WARNING that an endpoint cannot start
    Chromium's sandbox: the host restricts unprivileged user namespaces (Ubuntu 24.04
    sets `kernel.apparmor_restrict_unprivileged_userns=1`) or its container runtime
    ignores the seccomp profile. The endpoint then runs unsandboxed, once per process,
    and `trace.agent_browser_sandbox` says `unavailable`; lift the restriction for the
    sandboxed path.
  - A member refuses to start: the seccomp path did not resolve because Compose ran
    outside the repository root.
- **Development.** `tests/integration/test_agent_browser_pg.py` runs a real
  `playwright run-server` with Chromium. Install the browser once with
  `uv run playwright install chromium` (on Linux, `--with-deps`); on a host that
  restricts unprivileged user namespaces, also lift the restriction CI lifts for the
  sandboxed path (`sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`).

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
