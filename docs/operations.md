# Operations

This document owns maintenance and recovery runbooks. PostgreSQL tuning lives in
[PostgreSQL](postgresql.md); fields/defaults live in
[Configuration](configuration.md). These commands are not normal query traffic.

Deployments follow the
[configuration ownership and container contract](configuration.md#container-and-kubernetes-contract).

## Runbooks by task

| I want to | Section |
|---|---|
| Erase and rebuild my development environment | [Full Development Reset](#full-development-reset) |
| Recover a PostgreSQL container that will not start | [PostgreSQL startup recovery](postgresql.md#startup-recovery) |
| Diagnose or size Run workers and mutation lanes | [RunRuntime And Durable Query And Corpus Mutation Runs](#runruntime-and-durable-query-and-corpus-mutation-runs) |
| Cut a release | [Release Distribution](#release-distribution) |
| Update a dependency or a pinned image | [Updating Dependencies](#updating-dependencies) |
| Render a chart or an HTML report | [Agent Chart Rendering](#agent-chart-rendering) |
| Install, run, or switch a parser | [Parser Services](#parser-services) |
| Run the Agent Browser pool | [Agent Browser Pool](#agent-browser-pool) |
| Retry or clean up a failed ingestion | [Product Document Finalization And Failed Ingestion Cleanup](#product-document-finalization-and-failed-ingestion-cleanup) |
| Rebuild BM25 | [Workspace BM25 Rebuild](#workspace-bm25-rebuild) |
| Rebuild vectors offline | [Offline Vector Storage Rebuild](#offline-vector-storage-rebuild) |
| Rotate the key ring | [Key Ring Rotation](#key-ring-rotation) |
| Debug the Agent Mailbox | [Agent Mailbox](#agent-mailbox) |
| Bring up the ClickStack stack | [Local ClickStack Observability](#local-clickstack-observability) |

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
changes the `pg_textsearch` pin, the writer updates the installed extension at its next
start ([Updating An Installed Extension](#updating-an-installed-extension)); after one
changes any other pinned extension version, run the Docker reset.

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

`dlightrag-workspace-audit` reports an Agentspace root without deleting
anything: it counts Run roots and names those whose Run row is gone, which the
runtime's hourly orphan sweep removes. Pass `--root` to audit a path an earlier
configuration left behind, and `--sample` to bound the report. A deployment
whose execution is `disabled` and which names no root has nothing to audit.

## Updating Dependencies

Six planes carry versions, and only two are automated: Dependabot opens a weekly pull
request for GitHub Actions and for the frontend's npm tree. Every other plane is a
deliberate upgrade — change the source of truth, then run the gate that compares the
copies.

| Plane | Source of truth | How to move it | Gate |
|---|---|---|---|
| Python: the root distribution and the three workspace packages | `pyproject.toml` pins plus `uv.lock` | `uv lock --upgrade`, or `--upgrade-package <name>`; then `make sync-dev`, which checks the lock before syncing | `make platform-check`, `make ci` |
| Frontend npm | `frontend/package.json` and its lock | the Dependabot pull request, or `npm --prefix frontend update` | `make frontend-ci`, `make frontend-audit` |
| Chart renderer npm | `chart-render/package.json` and its lock | the Dependabot pull request, or `npm --prefix chart-render install echarts@latest` | `make chart-render-test` |
| Agent Browser | **one** Playwright version, in four literals a person edits: the client pin in `pyproject.toml`, `ARG PLAYWRIGHT_VERSION` in `agent-browser/browser/Dockerfile`, the server's `package.json`, and the `dlightrag-agent-browser:<version>` image tag in `docker-compose.yml`. `uv.lock` and `package-lock.json` are generated from those, the base image tag is interpolated from the ARG, and the Dockerfile asserts the installed npm copy against it | change the four together, then rebuild the pool image — the tag is what makes Compose build rather than reuse a stale 3.5 GB image | `make platform-check` refuses until they agree |
| Base images and toolchains | `Dockerfile` (`python:`, `node:`, `rust:`, `UV_VERSION`, `RESVG_VERSION`), `postgres/Dockerfile`, the observability stack's image tags in `packages/observability/compose.yaml` | edit the tag or `ARG`, rebuild | `make ci-full` for PostgreSQL, `make hyperdx-up` for the stack |
| MinerU sidecar | the pinned LightRAG client decides the range; `.env.mineru` overrides either end | `make mineru-install`, then `make mineru-service-start` | `curl http://127.0.0.1:8210/health` |

Two rules make this safe. **A version moves in one place, and a gate compares the
copies** — that is what `platform-check` is for. And **a root-level component owes a
gate** ([ADR 0054](adr/0054-mineru-is-a-workspace-package.md) §1): outside the Python
lock it has no release contract, so the gate that compares its copies is the only thing
keeping it from drifting.

A release is separate: `make release VERSION=…` moves the four manifests and the lock
together ([Release Distribution](#release-distribution)).

### Updating An Installed Extension

A rebuilt `dlightrag-postgres` image replaces only the shared library. A database created
under an older `pg_textsearch` keeps that release's SQL objects, and
`CREATE EXTENSION IF NOT EXISTS` never updates them. So at startup the writer, under the lock
that creates the extensions, compares the installed version with the one the server carries
(`pg_available_extensions`) and runs `ALTER EXTENSION pg_textsearch UPDATE` when the installed
one is lower. Profile Memory's setup does the same in its own database. A database already
at or above the server's release, without the extension, or on a version that is not a
dotted number is left alone.

Moving `pg_textsearch` from 1.4.0 to 1.5.1 needs nothing else. The update scripts only add
functions and the `tsquery` operator to the BM25 operator family. The segment format is
unchanged, and the index metapage moves from version 8 to 9 in place at the index's first
write, so no `REINDEX` is required (one would only make the index keep documents whose
tsvector is empty, which older indexes dropped). `dlightrag-rebuild-bm25` is not needed
either. The update script refuses to run unless `pg_textsearch` is in
`shared_preload_libraries`.

A refused update (a managed role that does not own the extension) is logged once as
`PostgreSQL extension pg_textsearch stays at X while the server carries Y` and startup goes
on: the installed objects keep working under the newer library. Run it as the extension's
owner in each database that has it:

```sql
SELECT installed_version, default_version FROM pg_available_extensions WHERE name = 'pg_textsearch';
ALTER EXTENSION pg_textsearch UPDATE;
```

## Release Distribution

DlightRAG ships as a repository, not as a package: a deployment clones this
repository and runs Compose or a native process, or adds the clone as an
editable path dependency. No workflow publishes DlightRAG to a package index,
no workflow publishes an image either — Compose and CI build
`dlightrag-postgres:pg18` from `postgres/`, so nothing depends on a registry,
a credential or a pinned digest — and `v*` tags only mark releases on GitHub.

A release is one command:

```bash
make release VERSION=2.2.0                        # writes, re-locks, and checks the contract
make release VERSION=2.2.0 ARGS=--dry-run         # say what would be written
make release VERSION=2.3.0 ARGS="--commit --tag"
make release VERSION=0.1.0 ARGS=--allow-downgrade # a deliberate reset
```

The version lives in four manifests — `pyproject.toml` (with the
`dlightrag-memory==` and `dlightrag-observability==` pins),
`packages/memory/pyproject.toml`, `packages/mineru/pyproject.toml`, and
`packages/observability/pyproject.toml`.
Everything else derives it: each workspace runtime reads its installed distribution,
the frontend names no version, and `uv.lock` follows the pins. A release stages those
four files and the lock and nothing else, so work in progress stays uncommitted; a
version reset is allowed only when it says so.

The configuration baseline in `config.yaml` (`# DlightRAG 2.0 canonical
configuration`) names the config schema a deployment writes against. It is not the
product version, and no release rewrites it.

Tagging and the GitHub Release are separate and manual: the tag is `v<version>` on
main (`ARGS=--tag` makes it), and the Releases page entry is whatever you say it is —
no workflow publishes anything and nothing reads the page back.

`make release-check` holds the version contract and runs at release time;
`make platform-check` holds the pins and configuration every deployment depends
on — the Playwright pin across the pool image, its manifests, the lock and
Compose, plus the canonical Agent defaults in `config.yaml` — and that is what
`make ci` runs on every push. `make workspace-wheels` is a local packaging check
that installs the built wheels in isolation; it distributes nothing.

## Agent Chart Rendering

The built-in [`charts`](../src/dlightrag/engine/agent/builtin_skills/charts/SKILL.md)
Skill has the Research agent write an Apache ECharts option and pipe it to
`echarts-render`. The command draws the option as an SVG with ECharts on the
image's `node` and rasterizes the SVG with `resvg` into a PNG; `--svg` adds a
vector file and `--html` a self-contained interactive page. The house theme
always applies, and Noto Sans SC is the only font `resvg` loads, so a font name
an option writes still draws. A native process gets both commands from
`make chart-render-link`, which installs ECharts and links them into the project
environment; rasterizing is what the image's `resvg` and Noto Sans SC add, so
without them `html-report` builds a report and `echarts-render` stops at the
missing rasterizer.

`html-report` is the interactive half. It turns an HTML fragment (prose,
slicers, pages, and one JSON block per chart) into one self-contained report
for an Artifact: the build checks the fragment, and writes a document that
carries the full ECharts build, the stylesheet and the runtime, so it needs no
network. The runtime lays each chart out for the width it has, applies the
slicers (filters, a metric switch, numeric sliders), and follows the reader's
light or dark scheme. The build refuses what
the Artifact sandbox would refuse anyway (a remote URL, a frame, a form,
`localStorage`, `fetch`) and warns about what makes a report worse: a chart
without a title or a subtitle that gives the unit and the source, more series
than colour can tell apart, and boilerplate the reader did not ask for (a
disclaimer, an AI credit, a privacy or copyright line, a footer credit, a call
to action). `--preview DIR` draws every chart to PNG at 360 and 900 px, and
again at one non-default state of each slicer it lists, with the code the
browser runs, so the author can look before the report ships — and see that the
controls move the charts they claim to ([ADR 0050](adr/0050-a-chart-on-a-filtered-field-says-what-it-does-about-the-filter.md)).
Every report also carries the fragment it was built from, deflated in its head,
which `html-report source report.html src.html` writes back byte for byte: a
later turn that must change a report reads the published report rather than an
Agentspace that no longer exists ([ADR 0051](adr/0051-a-report-carries-the-fragment-it-was-built-from.md)).
The fragment contract is the built-in `interactive-html` Skill's, and so are the
examples under its `references/`: a model reads them through `load_skill`, and
the browser tests build and open those same files.

Both commands draw in one palette, the Mineral palette of the product's own
design system. `palette.json` holds its chart colours once, for light and dark:
the roles (surface, text, axis and grid lines, tooltip) and four palettes.
`theme.json` is the structure of the house theme with every colour written as a
role reference, and `theme.js` builds the ECharts theme from the two; PNGs and
reports load the same files, so a chart cannot look different in one than in
the other. The categorical palette leads with the product's gold and holds
eight colours in a fixed order. Neighbouring colours differ in lightness and
hue and none is a red beside a green; every colour keeps 3:1 against the page
and the card in both modes; and the first six differ pairwise by at least 8
(OKLab x 100) under protanopia and deuteranopia, which is why the build warns
above six series: from the seventh, two colours can look the same to a reader
with colour-vision deficiency, so fold the rest into "其他" or highlight one
series. From four line series up each line also gets its own dash and marker.
An option chooses another palette with `"palette": "highlight"`, `"sequential"`
or `"diverging"`; colours an author writes win.
`tests/unit/test_chart_palette.py` pins these properties and the match with the
product's tokens.

The image adds, all under `/usr/local`:

| Path | Content |
|---|---|
| `bin/echarts-render`, `bin/html-report` | Symlinks to `lib/echarts-render/echarts_render.py` and `html_report.py` |
| `lib/echarts-render/` | The toolkit from `chart-render/`: the two commands, `ssr.cjs`, the shared `theme.js`, `theme.json` and `palette.json`, the report runtime in `report/` (`core.js`, `runtime.js`, `report.css` and `preview.cjs`), `echarts.min.js`, and the ECharts and zrender licenses |
| `bin/resvg` | The upstream resvg CLI, built from its crates.io release; the licenses of resvg and of every crate its locked build used are in `share/doc/resvg/`, one directory per crate |
| `share/fonts/dlightrag/` | The font set: Noto Sans SC, JP and KR Regular and Bold for those scripts, Source Sans 3 Regular, Bold, Italic and Bold-Italic for Latin, and both OFL-1.1 licenses |

Rust exists only in the builder stage. Between them the fonts cover Latin,
Chinese, Japanese and Korean — and not more: Source Sans 3 is what carries the
Latin the CJK faces stop short of (they hold Latin-1 and a third of Latin
Extended-A, so Turkish, Polish, Czech, Hungarian and Romanian would lose
letters), while emoji, Arabic, Hebrew, Thai and Devanagari are in none of them.
`echarts-render` names the characters no font has rather than drawing boxes in
silence. Source Sans 3 draws the same shapes as the Latin inside the CJK faces,
at 0.895 of their scale, so one line mixing the two sizes it 1.117 times or sets
the whole line in one family.

A chart is subtler than a PDF: `resvg` gives one text element one family and
boxes the characters that family lacks, without walking on to the next family
for them, so a field mixing scripts — a title holding both Latin and Chinese —
loses one of them while both languages are shipped and the PDF path renders the
same string whole. One script per field is what the chart Skill asks for, and the
image's chart smoke test draws Korean, Turkish and Japanese fields to hold it.

The pins to bump are `RESVG_VERSION` and the `rust:` builder tag in the
`Dockerfile` (the tag must meet the crate's minimum Rust version); `echarts` in
`chart-render/package.json`, followed by `npm install --package-lock-only` in
that directory to refresh `package-lock.json`; and `NOTO_CJK` and
`SOURCE_SANS`, one noto-cjk commit and one source-sans commit, together with the
twelve `sha256` checksums taken from them. `scripts/install-fonts.sh` reads the
font stage's own `/fonts/` `ADD` lines — names, checksums and the two base URLs —
so the font half of a bump changes no script; the resvg archive's `sha256` is the
one constant a `RESVG_VERSION` bump adds, in `scripts/install-chart-tools.sh`.

After a bump, build the image. Its smoke tests render a two-bar chart with a
Chinese title, build a one-chart report with its previews, and write one PDF
holding all eight languages as the `office-documents` Skill says to — register
each family, set the language's own, fall back to the rest — failing on fpdf2's
own missing-glyph warning, so a font that cannot draw a language fails the build
rather than dropping characters in a Run. All of them run as the `app` user and
fail the build when node, ECharts, `resvg`, a font or a runtime file is missing
or unreadable, so a broken toolkit never reaches a Run. CI builds no image: the
smoke tests run wherever the image is built. Then render a bar chart, a
horizontal bar chart with long Chinese category names and a line chart from the
new image, build the three examples and look at the PNGs. A new ECharts release
can change defaults, and `resvg` exits 0 after dropping text it cannot match to a
font, so `echarts-render` treats its `No match for` warning as a failure.

CI runs the toolkit's tests without the image. `make chart-render-install`
installs ECharts for them (`make test-unit` and `make test-e2e` call it), and
`scripts/install-chart-tools.sh` installs `resvg` and the font set at the
`Dockerfile`'s pins on Linux x86_64; the `fast` job puts them on `PATH` and points
`FONTS_DIR` at them. On a development machine without them the tests that
draw PNGs skip, and under `CI` they fail, so a gate cannot silently not run.

`make fonts` installs the font set alone, at the same pins and on any platform,
into `.chart-tools/fonts/dlightrag`, and prints the export line. That directory
is also where an Agent embeds a PDF from: `fpdf2`'s built-in fonts are Latin-1 and
it reads no environment of its own, so `office-documents` names both this variable
and the image's path, and a native Run without either has nothing to embed.

## Parser Services

The parser block is configured under
[Parser Sidecars](configuration.md#parser-sidecars). The host-side MinerU manager —
installer, launcher, and service units — is the
[`dlightrag-mineru`](../packages/mineru/README.md) package; copy its template once
with `cp packages/mineru/.env.example .env.mineru`. Host MinerU:

```bash
make mineru-install
make mineru-service-install
make mineru-service-status
make mineru-service-logs
make mineru-service-stop
```

Use `make mineru-api` for foreground operation where a background user service
is unavailable (containers, and Linux distributions without systemd); the tool
itself runs on macOS, Linux, and WSL2 only. MinerU title correction is best-effort: the launcher bounds each
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
- **The live browser preview.** The Inspector's `Live browser` dock shows the page a Run
  is driving and offers to take the driving seat
  ([knobs](configuration.md#agent-browser)). A turn whose Run is browsing carries a small
  indicator — a browser glyph and the page's host, taken from the `browser` tool's own
  events — and that indicator is the way in; nothing opens the dock by itself. Whether a
  preview is being watched is visible in two places: the lease row's `control_state` says
  who drives it (`SELECT endpoint, run_id, control_state, control_changed_at
  FROM dlightrag_agent_browser_leases WHERE run_id = '...'`), and a watched page costs
  frames while an unwatched one costs none, because the screencast only runs while a
  reader is subscribed — a hidden tab, a minimised window and a locked phone are all the
  same thing to it, because the dock closes its socket when the page goes away and asks
  for nothing while it is gone. What a watched page costs is one budget for the whole
  session, whoever drives: the seat is authority over the page, not an allowance of
  frames. What moves the budget is the session itself. Every frame a reader's slot still
  held when the next one arrived is a frame that arrived faster than it could be handed
  on, and when a quarter of a window's frames are those, for two windows in a row, the
  session steps down one rung: the JPEG quality first (70 → 52 → 45), because a page is
  read rather than watched, and then the frame rate (24 → 15 → 10), which is the floor.
  Two clear windows in a row step it back up, the rate before the quality, and a page
  that has gone quiet ends the run of windows rather than counting in it. Each rung is
  logged once, `Live browser session stepped down to 15 fps at quality 45`, and every
  state message carries the rung the reader is being served at, so the dock can say why
  the picture got lighter. A frame's age is bounded by design: the session acknowledges a frame when the next one supersedes it,
  so the browser cannot run ahead of the stream by more than the one encode it is next asked for. A page
  the browser cannot encode quickly is served more slowly, one frame in hand at a time — what a heavy page
  costs is bandwidth and the encoder's CPU — and a session whose readers cannot keep up says so through
  the step-down above.
  One seat holds one holder: taking it mints a credential that only the answer to that
  request carries, and every update every reader receives names the seat as held, and its
  holder by a display label rather than by that credential. A second reader is therefore an
  observer of a stream it may replay in full — and still cannot take the seat, type into the
  page, or hand it back — so two people
  never drive one page at once. Handing the seat back spends the credential, and so does
  letting it go idle or reaching the session cap: the seat taken next is a different one.
  Being looked at is what the seat does *not* depend on: closing the dock hands it back, and
  nothing else about the reader's own page does — a tab hidden mid-drive keeps the seat and
  the preview resumes with it still held, because the idle release above is what decides a
  holder has gone, not a lifecycle event the reader never asked for.
  Input events carry the sender's own sequence number, and one numbered at or below an event
  already run is dropped rather than replayed over the state the newer event left. Refusals
  reach the reader as
  the dock's own states: a Run with no browser lease says so instead of showing a blank
  frame, an unreachable browser keeps the last frame, a session past its hard cap says its
  time limit was reached rather than saying the Run has no browser, and a Run with nothing
  attached
  falls back to the last screenshot it stored. A reader who took control holds the page
  for `answer.agent.browser.live.control_idle_seconds` after their last click or key, and
  at most `session_seconds` in all; the agent's `browser` calls are refused as
  `taken_over` while they hold it, and the Run goes on. A session ends with the lease it
  sits on: cancelling the Run, closing its browser, or letting the lease expire ends the
  dock's stream and drops the indicator with the Run's own end, and the pool tells the
  live sessions directly rather than waiting for the next reader to notice. The dock
  reads it over one WebSocket (`WS /web/api/runs/{run_id}/browser/live/frames`, the
  [protocol](interfaces.md#web) in Interfaces), so a reverse proxy in front of the
  API must pass `Upgrade` and `Connection` for `/web` and must not close a quiet socket.
  Frames and state are both driven by the page, so a page that is not repainting produces
  neither, and Cloudflare — the front door these deployments sit behind — closes a proxied
  WebSocket after about 100 seconds without data. The route answers that window itself:
  when nothing has gone out for 30 seconds it sends a heartbeat text message,
  `{"at": <epoch ms>, "sequence": <newest frame>}`, which changes nothing for the reader
  and repeats the newest frame the session has delivered: level with what the dock has
  drawn, it is a page that has not changed rather than a stream that is behind. That
  leaves the socket with traffic every 30 seconds, a comfortable margin under the ~100
  second idle timeout, so a reader watching a page that has not moved keeps the dock
  instead of reconnecting roughly every hundred seconds. Give this path a long read
  timeout (an hour is the session cap `session_seconds` plus slack) and no response
  buffering — the same treatment the SSE answer streams need — and remember that a proxy
  that terminates TLS must be trusted for `X-Forwarded-Proto`, because the socket's
  `Origin` is checked against the front door ([why](security.md#per-surface-front-doors)).
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

## Local ClickStack Observability

The repository runs a self-hosted ClickStack — ClickHouse, the HyperDX UI/API, the
ClickStack OpenTelemetry collector, and MongoDB — from
`packages/observability/compose.yaml`. It is the destination the product's traces
are exported to, and the same ClickHouse holds the infrastructure metrics the
stack's own collector scrapes.

```bash
make hyperdx-up
```

Open <http://127.0.0.1:8300> and create the first administrator. ClickStack
provisions the collector's pipelines over OpAMP once the application has a team
and a data source, which happens on that first UI run; before it, the stack is up
but its collector does not accept OTLP yet.

| Target | Behavior |
|---|---|
| `make hyperdx-bootstrap` | Write `stack/.env` once (generated secrets), send a probe span, apply retention to every telemetry table, and report whether the probe landed |
| `make hyperdx-up` | Start the four containers, then bootstrap |
| `make hyperdx-down` | Stop |
| `make hyperdx-status` | Show containers |
| `make hyperdx-logs` | Follow the stack's logs |
| `make hyperdx-reset` | Stop and delete the stack's volumes: all telemetry. DlightRAG's own data is a separate Compose project and is not touched |

The CLI behind them is `dlightrag-observability` (`up`, `down`, `status`,
`logs`, `reset`, `bootstrap`, `alerts push`). It reads the stack from `stack_root()`:
`packages/observability` in a checkout, or whatever
`DLIGHTRAG_OBSERVABILITY_STACK_DIR` names. **The source distribution carries those
files and the wheel does not** — uv's build backend includes only the module root, so
`source-include` is the only route `compose.yaml` and `stack/**` have into an artifact.
A wheel-only install has the exporter and the classifier and nothing else: every command
that needs the stack exits 1 at `stack_root()`, with

> compose.yaml not found; set DLIGHTRAG_OBSERVABILITY_STACK_DIR to the directory holding the stack (packages/observability in a checkout)

so a wheel can export telemetry but cannot start, inspect, or configure the stack. Pointing
the variable at an unpacked sdist is what makes those commands work, and the directory has
to be writable, because `bootstrap` renders `stack/.env` into it. Nothing publishes either
artifact ([Release Distribution](#release-distribution)), so the checkout — where the files
sit beside the package — is what a deployment uses, and neither the variable nor the
unpacking is needed there.

### Pointing DlightRAG At It

The endpoint is not a secret and belongs in `config.yaml`:

| DlightRAG | `observability.otlp_endpoint` |
|---|---|
| Docker Compose | `http://host.docker.internal:4318` |
| Native | `http://127.0.0.1:4318` |

The ingestion key is a secret. ClickStack's OTLP receiver authenticates it as the
`authorization` header, and the key that works is the **team's** ingestion key:
copy it from the UI's Team Settings. The value `bootstrap` generated into
`stack/.env` is a seed for the stack's first run and can differ from the one the
application registers, so a 401 from the collector usually means the wrong one of
the two. Put it in `.env`:

```bash
DLIGHTRAG_OBSERVABILITY__OTLP_HEADERS={"authorization":"<team ingestion key>"}
```

### The Telemetry Erasure User

An owner erase (ADR 0049) removes that owner's spans at once instead of leaving
them to the table TTL, and the collector's own ClickHouse user is write-only, so
`dlightrag-owner-data erase` needs credentials that may run a delete. It
refuses — rather than writing a receipt for a half-erase — until they are set.
ClickHouse stores these users in its own access storage, so the account survives
a restart and goes with the stack's volumes: after `make hyperdx-reset`, create
it again. Password generation is the operator's; a hex string keeps `.env`
parsing trivial.

```bash
# 1. Create the account and grant it exactly what an erase runs.
PW=$(openssl rand -hex 24)
CH=http://127.0.0.1:8123/
curl -s "$CH" --data-binary "CREATE USER IF NOT EXISTS dlightrag_erasure
  IDENTIFIED WITH sha256_password BY '$PW' SETTINGS readonly = 0"
curl -s "$CH" --data-binary "GRANT SELECT, ALTER DELETE ON default.otel_traces TO dlightrag_erasure"
curl -s "$CH" --data-binary "GRANT SELECT, ALTER DELETE ON default.otel_logs TO dlightrag_erasure"

# 2. Point the deployment at it (the endpoint is the stack's loopback port, which
#    a container reaches through the same host gateway its OTLP endpoint uses).
cat >> .env <<EOF
DLIGHTRAG_OBSERVABILITY__TELEMETRY_ERASURE__URL=http://host.docker.internal:8123/
DLIGHTRAG_OBSERVABILITY__TELEMETRY_ERASURE__USER=dlightrag_erasure
DLIGHTRAG_OBSERVABILITY__TELEMETRY_ERASURE__PASSWORD=$PW
EOF

# 3. Recreate the API so it reads the new environment, then prove the path.
docker compose up -d --no-deps dlightrag-api
docker exec dlightrag-dlightrag-api-1 dlightrag-owner-data \
  --owner verify-not-a-real-owner erase --line b2c --region eu
```

Step 3 must print `telemetry=erased` in the receipt's counts. A deployment that
keeps no telemetry passes `--no-telemetry` instead, and its receipt says
`telemetry=skipped`, which is a different fact from erasing nothing. The owner id
in this recipe is invented, so it created nothing and the erase covers no Corpus
Workspace; run it with a real owner only when the customer's deletion is what is
being asked for.

`erase` takes `--line b2c|b2b` because the line decides who controls the Corpus
Workspaces the requester created. On `b2c` the creator is the data subject, so
those Workspaces are covered by default: each one becomes a `delete_workspace`
Corpus Mutation Run, the receipt records the Run ids under
`corpora_deletion_runs`, and the documents stay until that Run executes — the
30-day promise is read off the Run's terminal state, not off the erase's exit
code. Pass `--keep-corpora` to keep one instead; the receipt then names it. On
`b2b` the customer controls them, so `--customer-approval <ticket or email
reference>` is required and is recorded in the receipt beside the line. A
pre-flight that removes nothing and lists what would be covered:

```bash
docker exec dlightrag-dlightrag-api-1 dlightrag-owner-data \
  --owner verify-not-a-real-owner erase --line b2c --dry-run
```

The delivered Runs are ordinary Corpus Mutation Runs, so they are visible where
the browser sees them (`/web/api/corpus-runs/<run_id>`) and in the lane's rows:

```bash
docker exec dlightrag-postgres-1 psql -U dlightrag -d dlightrag -c \
  "SELECT run_id, owner_id, status FROM dlightrag_runs WHERE run_kind = 'corpus_mutation'"
```

Each Run's `owner_id` is the Workspace it deletes, which is how a receipt's
`corpora_deletion_runs` ids map back to Workspace names.

Compose reads `.env` only when it creates a container:

```bash
docker compose up -d --force-recreate dlightrag-api dlightrag-mcp dlightrag-reader
docker compose logs dlightrag-api | grep -i 'OTLP tracing'
```

`OTLP tracing enabled → http://host.docker.internal:4318/v1/traces` is the line
that says where spans go; `OTLP tracing off (no endpoint configured)` means the
setting did not load. The same endpoint starts the log channel one line later
(`OTLP log export enabled → http://host.docker.internal:4318/v1/logs`), so a
process that exports traces exports its own records too — there is no second
switch. Startup performs no authentication or reachability check, so a wrong key
or a stopped stack shows up as missing traces, not as a failed boot. A process
that names no endpoint exports nothing and retries nothing.

Set `DLIGHTRAG_OBSERVABILITY__SERVICE_NAME` per process
(`dlightrag-api`, `dlightrag-mcp`, `dlightrag-reader`) so the service map
separates them; `environment` and `release` separate deployments and versions
([field table](configuration.md#observability)).

### Where The Traces Are

Spans land in `default.otel_traces` in the stack's ClickHouse, over its HTTP
interface on loopback:

```bash
curl -sS 'http://127.0.0.1:8123/?default_format=TabSeparated' --data-binary \
  "SELECT ServiceName, SpanName, count() AS spans FROM default.otel_traces \
   WHERE Timestamp > now() - INTERVAL 1 HOUR GROUP BY 1, 2 ORDER BY spans DESC"
```

The product's vocabulary rides on the spans, not in the table's shape:
`SpanName` is the registry's name (such as `run-answer`),
`dlightrag.observation.kind` is its kind, `session.id` and `user.id` carry the
conversation and the owner, and the usage buckets are `gen_ai.usage.*` with
`dlightrag.usage.total` ([rendering](observability.md#rendering-on-the-wire)).
A failed Run also carries `dlightrag.error.kind`, the kind its executor
classified — `dependency_unavailable`, `workspace_integrity_error`, and the rest
of the coordinator's vocabulary. An alert prefers that attribute and falls back to
the status description, which is what a span recorded before the attribute existed
carries, and what a loss the durable store settled without a surviving worker has
instead.
`bootstrap` runs the same check as its probe, so a stack that just started either
reports one landed span or says explicitly that ingestion is not provisioned yet.

### Where The Logs Are

The process's own records land in `default.otel_logs`, one row per record, in the
same ClickHouse:

```bash
curl -sS 'http://127.0.0.1:8123/?default_format=TabSeparated' --data-binary \
  "SELECT Timestamp, ServiceName, SeverityText, Body, TraceId, SpanId \
   FROM default.otel_logs WHERE Timestamp > now() - INTERVAL 1 HOUR \
   ORDER BY Timestamp DESC LIMIT 20"
```

The shape to read: `Body` is the masked message, `SeverityText` is the
OpenTelemetry severity (`WARN`, `FATAL`), `LogAttributes` holds what the caller
attached with `extra` plus the code location the record came from, and
`TraceId`/`SpanId` are the observation that logged it — empty outside a Run. That
pair is the join: search a trace in the UI and its lines are the rows whose
`TraceId` matches, which is why a log line can be read next to the span that
produced it. A record whose field name carries a secret (`api_key`,
`authorization`, …) arrives as `[redacted]`; a message is bounded, and the
exporter's own records stay on stdout rather than being exported.

stdout is unchanged, and it is still the first place to look: the export is one
extra handler, it never blocks a logging call, and when the collector is down the
records are dropped from a bounded queue while the process keeps printing.

### Alerts And The Dashboard

The notification channel, the alerts, and the dashboard tiles are files, not
clicks: they ship under `packages/observability/stack/alerts/` (one JSON document
per alert, `channel.json` for the webhook) and `.../stack/dashboards/`, and one
command writes them through ClickStack's REST API. A file carries **no instance id**:
the connection and the sources a query reads are `${HYPERDX_CONNECTION_ID}`,
`${HYPERDX_SOURCE_TRACES}` and `${HYPERDX_SOURCE_METRICS}`, which are the ids *this*
stack's first seeding minted. `alerts push` asks the stack's own API for them
(`GET /api/v2/connections` and `/api/v2/sources`: the oldest connection, and the oldest
source of each kind) and uses what it gets; a value set in the environment or in
`stack/.env` is used as written and not asked for. A stack nobody has opened yet has
none, and the push stops and says to create the administrator and the sources in the UI
first, then run it again. A push finds what it wrote **by name**: it lists the collection once
(`GET /api/v2/webhooks`, `/alerts`, `/dashboards`, which ClickStack 2.42 serves), updates
the document that carries the file's name (`PUT`) and creates the rest (`POST`). The
channel goes first because every alert names it, as `${HYPERDX_WEBHOOK_ID}`, which the
push fills in with the id the channel just got. A file that changes is the change:

```bash
export DLIGHTRAG_OBSERVABILITY_API_KEY=<the access key the UI mints under Team Settings>
uv run dlightrag-observability alerts push
```

`alerts push --dry-run` lists what it would write and sends nothing, so it does not ask
the API for the stack's ids either; `DLIGHTRAG_OBSERVABILITY_API_URL` points the command at another stack
(`http://127.0.0.1:8000` by default). The webhook's URL carries a bot token and is
therefore a credential: `channel.json` names `${DLIGHTRAG_TELEGRAM_WEBHOOK_URL}`
and the push resolves it, so nothing secret lives in the repository. The operator
sets two variables, and the command reads `stack/.env` when the process environment
does not (`packages/observability/stack/.env.example` declares them):
`DLIGHTRAG_TELEGRAM_CHAT_ID` (the conversation the bot posts to) and
`DLIGHTRAG_TELEGRAM_WEBHOOK_URL` (which carries the token). The webhook has no id to
keep: the push creates it and finds it again by its name. The artifacts are checked with:

```bash
uv run pytest tests/unit/test_observability_stack.py -q
```


What the stack watches, and what it cannot: p95 answer duration, failing runs by
kind, runs lost to an outage or an abandoned lease, Postgres availability and
connection headroom, host disk pressure, collector refusals, and front-end errors
from the browser channel.

There is no product-api 5xx alert: the product api emits feature-level spans and no
request-level observation, so an alert on api request errors could not fire honestly
([ADR 0040](adr/0040-observability-lands-in-clickstack.md)). A deployment that adds
request instrumentation to the api can add one.

The Postgres alert reads the metrics `postgres-exporter` publishes, and nothing
else in the deployment reports that database: start the `metrics` profile (below)
or the alert stays quiet, because a window with no `pg_` metric is a deployment
that is not watching Postgres rather than one whose Postgres is down. It fires on
`pg_up` going to zero, or on the busiest single scrape in its window passing 80% of
the `pg_settings_max_connections` that scrape reported. That 80% is a ceiling, not
a level ordinary operation reaches: with a fifth of the server's connections still
free, the next burst of concurrent Runs is what spends the last of the headroom,
which is why the alert is a warning ahead of `FATAL: sorry, too many clients
already` rather than a report of one. The word *scrape* is load-bearing:
`pg_stat_activity_count` carries one row per `(datname, state)` label pair and one
per scrape, so a plain `sum` over a five-minute bucket measures label combinations
times scrapes — backs × states × scrapes — and not backends. The shipped query
deduplicates each scrape first (`SELECT DISTINCT TimeUnix AS scrape, MetricName,
Value, Attributes`), sums only that scrape's own series, and takes the busiest one
as the value; `max_connections` is server-global, so its aggregate is `max`, never
a `sum`. There is no bucket around the scrape, and no `$__timeInterval` in the
query, because an alert whose chart is a `number` is given no `intervalSeconds`:
the macro renders to `toStartOfInterval(scrape, INTERVAL {intervalSeconds:Int64}
second)` and every evaluation failed with `Substitution intervalSeconds is not set`
while the alert's state stayed `OK`. A `number` alert reads the worst row its query returns, so a bucket
between the scrape and that reading changed nothing but the ability to evaluate.

Front-end errors arrive as error spans from `dlightrag-web`: the browser SDK's
default `onerror` capture raises one per uncaught error. That alert also has a log
branch for `dlightrag-web` records, which is the right signal if console capture
is ever turned on; **no web log record has ever been exported**, so the span branch
is the one that fires, and the alert's message says so rather than implying two
live signals.

### Where The Metrics Are

The `metrics` profile adds a second collector process and the exporters it scrapes,
none of which the four required containers include:

```bash
docker compose --project-name dlightrag-observability \
  --env-file packages/observability/stack/.env \
  -f packages/observability/compose.yaml --profile metrics up -d
```

| Source | Metrics | Read by |
|---|---|---|
| `hostmetrics` | `system.cpu.*`, `system.memory.*`, `system.disk.*`, `system.filesystem.usage` | the host-disk alert |
| `docker_stats` | `container.cpu.utilization`, `container.memory.usage.total`, the network counters | an operator asking which container is eating memory |
| `prometheus` → `postgres-exporter` | `pg_up`, `pg_stat_activity_count`, `pg_settings_max_connections`, `pg_stat_database_numbackends`, `pg_database_size_bytes`, `pg_locks_count`, `pg_replication_lag_seconds`, `pg_stat_user_tables_*` | the postgres alert, and an operator |

They land in `otel_metrics_gauge` (gauges) and `otel_metrics_sum` (counters) with
`MetricName`, `TimeUnix`, `Value`, `Attributes` (the Prometheus labels), and
`ResourceAttributes`:

```bash
curl -sS 'http://127.0.0.1:8123/?default_format=TabSeparated' --data-binary \
  "SELECT MetricName, Attributes, max(Value) AS value FROM default.otel_metrics_gauge \
   WHERE startsWith(MetricName, 'pg_') AND TimeUnix > now() - INTERVAL 5 MINUTE \
   GROUP BY MetricName, Attributes ORDER BY MetricName LIMIT 20"
```

`postgres-exporter` reaches the application's database through the host gateway on
`127.0.0.1:5432`, with the application's own local defaults
(`dlightrag`/`dlightrag`/`dlightrag`). A deployment that changed them sets
`DLIGHTRAG_POSTGRES_USER`, `__PASSWORD`, `__HOST`, `__PORT`, and `__DATABASE` in
`stack/.env` — the observability stack is a different Compose project and cannot
read the application's `.env`. The exporter publishes its own exposition on
`127.0.0.1:9187`, which is how to check it without the collector in the path.

### Retention

`HYPERDX_OTEL_EXPORTER_TABLES_TTL` in `stack/.env` is the deployment's **one**
statement of retention: a TTL for every `otel_*` table the collector creates.
`dlightrag-observability bootstrap` reads the same variable and applies that
retention to every telemetry table it finds in `system.tables` — including the
ones the collector's variable cannot reach, a table a signal creates with its
first write and `hyperdx_sessions`, which session replay creates on its first
export:

```bash
uv run dlightrag-observability bootstrap
# retention: 168 hour(s) on hyperdx_sessions.Timestamp, otel_logs.Timestamp, …
```

The variable is the duration ClickHouse states (`168h`, `30d`, `2w`; a bare number
is seconds) and it defaults to seven days when neither the environment nor
`stack/.env` states one. `bootstrap` applies it **as stated**: a `36h` deployment
gets `toIntervalHour(36)`, not the two days a day count would have rounded it to,
because a restated duration is a second retention and this one decides when
telemetry is deleted. That is why `bootstrap` and the collector cannot come to
different answers, and why `.env.example` cannot promise a retention the CLI would
not apply. `--days N` overrides it for one run with a whole-day count — the
operator's own number rather than a reading of the variable. The pass is idempotent
(`MODIFY TTL` states the retention a table should have), it names the column it
keyed each table on, and it reports a table it will not guess at rather than
altering it. A deployment that has exported nothing yet says so, and one whose
browser SDK has not sent a replay says it is waiting for `hyperdx_sessions`;
neither is a failure. Change retention in `stack/.env` and recreate the collector,
or bootstrap again with another `--days`.

### Health And Outage Behavior

The collector answers on <http://127.0.0.1:13133> and its own metrics are on
8888; the UI answers on 8300. When the stack stops answering, nothing in the
product does: spans are dropped from a bounded in-process queue, a request, a Tool
call, and a log line never wait on the export, and shutdown flushes under a
timeout. An outage is reported once — one `WARNING` when the export starts
failing, one `INFO` line when it recovers — and the failures in between are
counted rather than logged, so turning the observing stack off gives a quiet
product instead of a log flood
([ADR 0039](adr/0039-observability-is-always-on-and-the-seam-masks-private-data.md)).

### Cost And Recovery

DlightRAG always reports tokens and never computes cost. The counters a price
table needs are on every span (`gen_ai.usage.input_tokens`, `output_tokens`,
`gen_ai.usage.cache_read.input_tokens`, and `dlightrag.usage.total`), keyed by
the model alias in `gen_ai.request.model`, so the operator builds that table
beside the traces it prices ([Observability](observability.md#cost)). The cache
read is a *subset* of `input_tokens`, not a sibling of it, so a table that prices
the uncached rate prices `input_tokens - cache_read.input_tokens` and one that
prices the cached rate prices the read on its own; adding the two would count the
hit twice. ClickStack's own LLM cost panel
reads a built-in catalogue and is not the deployment's money account. OpenRouter
can instead return the charge it applied, when enabled only on its model block:

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

Do not send that vendor-specific option to providers that reject it. A reported
charge becomes `gen_ai.usage.cost` and outranks any derived price.

Recovery is a restart, not a data migration. If the stack's volumes are gone,
`make hyperdx-up` starts it again, the UI asks for a first administrator, and
`make hyperdx-bootstrap` re-applies retention and proves ingestion with a
probe. The generated secrets stay in `stack/.env`; delete that file to have
bootstrap write a new pair, then restart the containers with the new
`HYPERDX_API_KEY` and update DlightRAG's `otlp_headers` to match. Last resort,
which deletes all telemetry and none of DlightRAG's data:

```bash
make hyperdx-reset
make hyperdx-up
```
