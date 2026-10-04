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
| `bin/resvg` | The upstream resvg CLI, built from its crates.io release; the licenses of resvg and of every crate its locked build used are in `share/doc/resvg/`, one directory per crate |
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

## Agent Browser Pool

The Agent Browser ([ADR 0032](adr/0032-the-agent-browser.md)) is a pool of Playwright
containers plus one Squid proxy. The bundled Compose stack runs the members its
`members=N` declares, two by default (`agent-browser-1`, `agent-browser-2`; see
Resizing below), and `agent-browser-egress`, and binds their
addresses into `dlightrag-api`, `dlightrag-mcp`, and `dlightrag-reader`
([fields](configuration.md#agent-browser); the boundary is in
[Security](security.md#agent-browser-boundary)). No service waits for them: a render or a
page's first `navigate` with no browser up fails as `unreachable` and the Run goes on.

```bash
# From the repository root, so the seccomp profile path in docker-compose.yml resolves.
docker compose up -d --build $(docker compose config --services | grep '^agent-browser-')
docker compose ps
docker compose logs agent-browser-egress
```

- **Check.** `GET /health` is no check of the pool
  ([what it reports](interfaces.md#health-and-errors)), so look at the members: one is
  healthy when `docker compose ps` says so, and its check asks the run-server for
  `/json`. Which Run holds which member is the `dlightrag_agent_browser_leases` table:
  `SELECT endpoint, run_id, updated_at FROM dlightrag_agent_browser_leases`. A row that
  names a Run holds its member only while that Run's lease is live
  ([when](architecture.md#agent-browser)), whatever the row still says. The proxy logs
  every request to its stdout; a destination it refuses is `TCP_DENIED`.
- **Size.** A Run holds a member while it renders or has an Agent Page open, and for
  `idle_release_seconds` after the last of them ends, so the pool's size bounds how many
  Runs use a browser at the same moment across every process that runs Query workers. An
  Agent Page stays open until its Session ends
  ([when](architecture.md#agent-browser)), so `idle_release_seconds` frees a member only
  for a Run whose Agent Pages have all closed. When every member is held, a render or a
  first `navigate` waits up to `lease_wait_seconds` and then fails as `busy`; the model
  reads that and works from the direct read. Add members when that is frequent. Each
  member is capped at `COMPOSE_AGENT_BROWSER_MEM_LIMIT` (default `2g`) and 1024 processes.
- **Resizing.** The pool's size is the `members=N` in `docker-compose.yml`. After changing
  it, run `uv run python scripts/agent_browser_pool.py`, which writes the three
  `agent-browser-pool` blocks from it: each member's service on an internal network of its
  own, the networks that `agent-browser-egress`, `dlightrag-api`, `dlightrag-mcp`, and
  `dlightrag-reader` join, and the `endpoints` binding. Two members never share a network
  ([why](security.md#agent-browser-boundary)). Then apply it with
  `docker compose up -d --remove-orphans` plus the `--profile` flags the deployment runs
  with, such as `--profile reader`, since a service whose profile is left out keeps its old
  endpoints. Every process then restarts with the same endpoint URLs, which the lease table
  is keyed by, and a member the pool no longer has is removed rather than left running.
  `tests/unit/test_compose_agent_browser_pool.py` fails when the blocks disagree with
  `members=N`.
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
    service is not on the member's network, or the versions differ (HTTP 428). The
    lease store, PostgreSQL, can also be the one that cannot be reached; the
    application then logs `Agent Browser lease store failed` with the error type.
  - `busy`: every member's row names a Run that holds its lease. Wait for one to
    finish rendering or browsing, or for the lease of a Run whose worker died to expire
    (about a minute).
  - A page that never loads: look for `TCP_DENIED` in the proxy's log. A private
    destination or a port other than 80 and 443 is refused by design.
  - `unreachable` for every member while the members are healthy, with
    `Agent Browser connect failed` logged for each, typically as `TargetClosedError`:
    Chromium exited as it launched. The usual cause is its sandbox, which Playwright words
    as `Chromium sandboxing failed` in the error it returns; the application logs only the
    error's type. The host restricts unprivileged user namespaces (Ubuntu 24.04 sets
    `kernel.apparmor_restrict_unprivileged_userns=1`) or its container runtime ignores the
    seccomp profile, and `chromium_sandbox` (default `true`) makes every launch ask for the
    sandbox. Nothing falls back to running without it. Either relax the host (CI lifts the
    restriction with `sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`) or set
    `answer.agent.browser.chromium_sandbox: false` and restart the processes that run Query
    workers. With `false` the container and its network are the only isolation
    ([why](security.md#agent-browser-boundary)).
  - A member refuses to start: the seccomp path did not resolve because Compose ran
    outside the repository root.
- **Development.** `tests/integration/test_agent_browser_pg.py` runs a real
  `playwright run-server` with Chromium, launched inside its sandbox.
  `tests/integration/test_agent_browser_tool.py` drives the `browser` tool in that browser
  without a database, and `tests/integration/test_agent_browser_tool_pg.py` runs a Research
  Run through settlement, Child Sessions, and recovery. Install the browser
  once with `uv run playwright install chromium` (on Linux, `--with-deps`); on a host that
  restricts unprivileged user namespaces, also lift the restriction CI lifts
  (`sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`).

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

## Key Ring Rotation

The deployment key ring is `<deployment.working_dir>/connection-keyring.json`, which the
first writer creates ([format and consumers](personal-mcp-connections.md#secret-handling-and-key-ring)).
Rotating it moves what each consumer sealed to the new key.

1. Add a fresh 32-byte base64url key under a new ID, keeping the old IDs:

   ```bash
   python3 -c 'import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode().rstrip("="))'
   ```

2. Point `active` at the new ID and restart **all** workers, so none still
   encrypts with the old key. Writer maintenance then re-encrypts live Grants
   ([Operational lifecycle](personal-mcp-connections.md#operational-lifecycle)) and
   re-seals Agent Account envelopes, in a loop of its own that passes at startup and
   then once a minute, over every account still under an old key.
3. Wait until the old key's Grant, OAuth inbox, and Agent Account counts reach
   zero. Inbox flows expire within `oauth_timeout` (at most 600 seconds) and are then
   collected. On the deployment database, count envelopes; never select or export them:

   ```sql
   SELECT key_id, count(*) FROM dlightrag_connection_grants
   WHERE encrypted_envelope IS NOT NULL GROUP BY key_id;
   SELECT envelope::jsonb->>'key_id' AS key_id, count(*)
   FROM dlightrag_connection_oauth_flows f
   CROSS JOIN LATERAL (VALUES (f.encrypted_result), (f.encrypted_credentials)) v(envelope)
   WHERE envelope IS NOT NULL GROUP BY 1;
   SELECT key_id, count(*) FROM dlightrag_agent_accounts GROUP BY key_id;
   ```

4. Remove the old key from the ring. A Grant it still sealed would need
   authorization again, an Agent Account it still sealed is unusable until the
   site's password reset replaces it, and backups stay readable to any retained copy
   of it.

## Agent Mailbox

The Agent Mailbox ([ADR 0034](adr/0034-agent-accounts-and-the-agent-mailbox.md)) is
optional, and the repository holds no vendor code for it.
[Configuration](configuration.md#agent-mailbox) has the fields and the bucket's contract.
How mail reaches the bucket, and how long it stays, are the deployment's. The steps
below are one deployment's, a catch-all on its own domain through Cloudflare Email
Routing and an Email Worker to R2; the Worker is an example for that deployment, not
product code, and another one could use SES receipt rules that write to S3.

- **A catch-all needs the apex.** Cloudflare allows catch-all rules only on a zone's
  apex domain, and a subdomain gets literal rules only. The Agent mints a different
  address for every owner and site, so literal rules cannot work, and the catch-all and
  `alias_domain` are the apex. It then takes every address of that domain that has no
  literal rule, and writes it all to the bucket for the Agent to read. To keep the
  domain's own mail, give those addresses literal rules, which take precedence, or use a
  domain of its own for the Agent.
- **Check the domain's mail first.** `dig +short MX <domain>` and
  `dig +short TXT <domain>`, and look at its DNS in the dashboard. Enabling Email
  Routing replaces its MX records with Cloudflare's, so a domain that already receives mail
  stops receiving it: use another domain.
- **The bucket.** Create an R2 bucket (for example `dlightrag-agent-mail`; lower-case
  letters, digits, `-`, and `.`) in the default location. A jurisdiction changes the
  endpoint to `https://<account-id>.<jurisdiction>.r2.cloudflarestorage.com`.
- **The Worker.** Cloudflare accepts mail up to 25 MiB and gives the Worker the envelope
  recipient as `message.to`. Bind the bucket as `MAIL` and set `PREFIX` to the same value as
  `answer.agent.mailbox.prefix`, `mail` by default:

  ```js
  // One deployment's example: write every incoming message to R2 as it arrived.
  // key = <PREFIX>/<envelope recipient, lower case>/<ISO time>-<uuid>.eml
  export default {
    async email(message, env, ctx) {
      const recipient = message.to.toLowerCase();   // the envelope RCPT TO, not the To: header
      const prefix = env.PREFIX ? `${env.PREFIX}/` : "";
      const key = `${prefix}${recipient}/${new Date().toISOString()}-${crypto.randomUUID()}.eml`;
      const raw = await new Response(message.raw).arrayBuffer();   // the whole message
      await env.MAIL.put(key, raw, { httpMetadata: { contentType: "message/rfc822" } });
    },
  };
  ```
- **Routing.** Enable Email Routing for the zone, which adds Cloudflare's MX and SPF
  records. The wizard may ask for a destination address to verify; it is not used, and the
  Agent never uses any address of the owner's. Set the catch-all to **Send to a Worker**,
  choose the Worker, and check it is **Active**. Leave subaddressing off.
- **Retention is the deployment's, and DlightRAG never deletes.** Add an R2 lifecycle rule
  that deletes objects under the prefix after 30 days, so a mailbox alias stays within what a
  listing reads ([the bucket's contract](configuration.md#agent-mailbox)).
- **The key.** Create an R2 account API token with **Object Read only**, scoped to the
  bucket. DlightRAG only lists and gets. Note its Access Key ID and Secret Access Key when
  it is created; the Secret is shown once. The endpoint is
  `https://<account-id>.r2.cloudflarestorage.com`. Revoke and recreate the token, and update
  `.env`, if either key may have leaked.
- **Turning it on.** Put the settings in `.env`
  ([names and rules](configuration.md#agent-mailbox)) and restart `dlightrag-api`,
  `dlightrag-mcp`, and `dlightrag-reader`. `GET /health` then shows `"accounts": true` and
  `"mailbox": true` under `agent_browser` ([what it reports](interfaces.md#health-and-errors)).
  A wrong key or bucket answers at a Run's `inbox` as
  `The Agent Mailbox could not be read (AccessDenied)`, naming only the error code.
- **Test a message.** Send mail from any address to `probe-<anything>@<domain>`. An object
  named `<prefix>/probe-<anything>@<domain>/<time>-<id>.eml` appears in the bucket. If it does
  not, look at the Worker's logs and Email Routing's activity log.
- **Never log what the browser or the client sees.** Do not set `DEBUG=pw:*` for the pool
  or the application: Playwright's debug log prints each call's arguments, a filled password
  among them. Keep production at `log_level: info`, since the S3 client's debug log names the
  endpoint and the access key id. Neither the key nor an endpoint reaches an error or a log
  line DlightRAG writes.
- **Development.** `tests/integration/test_agent_accounts_browser.py` drives `register`,
  `login`, and `inbox` in a real Chromium over a proxy that terminates TLS, against a
  loopback S3 double, with no database, and `tests/integration/test_agent_accounts_pg.py`
  runs the account store, its re-sealing, and a later Run's login against PostgreSQL. Both
  need the browser installed as for the [pool's tests](#agent-browser-pool).

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
