# DlightRAG

[![CI](https://github.com/hanlianlu/dlightrag/actions/workflows/ci.yml/badge.svg)](https://github.com/hanlianlu/dlightrag/actions/workflows/ci.yml)
[![Ask DeepWiki](https://img.shields.io/badge/Ask-DeepWiki-1f6feb)](https://deepwiki.com/hanlianlu/DlightRAG)

DlightRAG is a self-hosted, always-on grounded agent harness. It runs agents that
read your documents, cite what they read, and remember across conversations.
Retrieval (multimodal RAG) and memory are first-class modules of the harness, not
add-ons to it. PostgreSQL is its only database and state store; a deployment also
needs a filesystem every process shares, a document parser, and a model provider.

## What it gives you

- **Durable Runs.** A Run survives the disconnect that ends a chat, resumes where it
  stopped, and is claimed by whichever worker holds its lease
  ([Run Runtime](docs/run-runtime.md)).
- **Retrieval, multimodal.** Hybrid: knowledge graph, vectors, BM25, metadata
  filters, page images and rerankers. Every answer cites the passage it read and
  highlights it in the source ([Retrieval and Answer](docs/retrieval-answer.md)).
- **Memory.** A Session remembers what it concluded and a profile remembers what you
  prefer. It is the `dlightrag-memory` package, which also runs on its own as an MCP
  server.
- **Skills.** Agent Skills are packages of instructions, references and scripts
  (`SKILL.md`) the agent loads on demand. Charts, council, interactive HTML reports,
  Office documents and a Skill that writes Skills are built in; an operator adds
  Skills for everyone and an owner adds personal ones.
- **Subagents.** A Research Run delegates to child agents and they report back to it.
- **MCP, both ways.** An MCP server with 18 tools, over stdio and streamable HTTP,
  lets other agents use the knowledge base and Runs. Personal Connections go the other
  way: each owner connects remote MCP servers, with a bearer token, a header or OAuth,
  and their tools join every Research Run that owner starts
  ([Personal MCP Connections](docs/personal-mcp-connections.md)).
- **An agent browser.** Pages are rendered, and driven, in Playwright containers
  behind an egress proxy, never in the application's own process
  ([Architecture](docs/architecture.md#agent-browser)).
- **Observability.** One trace per unit of work, exported over OTLP, with a privacy
  mask in front of the exporter. The `dlightrag-observability` package ships a
  ClickStack (ClickHouse and HyperDX) stack with dashboards and alerts
  ([Observability](docs/observability.md)).
- **Many providers.** Chat through OpenAI-compatible (Chat Completions and Responses),
  Anthropic and Gemini endpoints; embeddings through OpenAI-compatible, Voyage, Jina,
  Gemini and Cohere; rerankers through Jina, Voyage, Cohere, Aliyun, Azure and any
  HTTP endpoint ([Configuration](docs/configuration.md)).
- **One runtime, four doors.** Web, REST, MCP and an in-process Python API share the
  same Application services and authorization rules.

**Runtime:** Python 3.14 (the checkout accepts any 3.14 — `.python-version` names the minor, so a
machine is not blocked on a patch it cannot install; CI and the image run 3.14.10, the newest patch,
and the MinerU sidecar venv uses 3.13, which
MinerU supports) · PostgreSQL 18 ecosystem · Apache-2.0

LightRAG's storage legs are PostgreSQL — KV, vector, graph and document status. The
one optional exception is the vector leg: a writer deployment may replace it with
Milvus (including Milvus-compatible Zilliz endpoints) by installing
`dlightrag[milvus]` ([Configuration](docs/configuration.md#postgresql-and-process-role)).

## Architecture

![DlightRAG system overview: browser and API clients, the API, MCP and reader
roles, the shared state they use, the Agent Browser pool and its egress proxy, the
parser, and the outside services a deployment reaches](docs/diagrams/system.svg)

LightRAG supplies graph and vector retrieval; DlightRAG owns product policy,
multimodal alignment, durable ingestion and answers, security, storage adapters,
and public interfaces. Fast and Research answers share one durable conversation
tree, and Research adds a per-run Agentspace, tools, memory, and child agents.
Module and storage ownership is in [Architecture](docs/architecture.md).

### Deployment paths

| Path | PostgreSQL | Parser | Security |
|---|---|---|---|
| Local Docker | Compose PG18 | Self-hosted MinerU by default, or Compose Docling | Loopback, `auth_mode: none` |
| Native API | Compose or external PG18 | Any reachable MinerU or Docling | Local or explicit auth |
| Shared service | Managed or self-hosted PG18 | Independently operated parser | `jwt`; `simple` is one owner |
| Enterprise | Managed PG18 | Independently operated parser | `jwt` with Access Rules and workspace creators |

The parser runs outside the DlightRAG application container, and every process of a
deployment mounts one shared `deployment.working_dir`
([Architecture](docs/architecture.md#deployment-and-storage),
[Workers and scaling](docs/run-runtime.md#workers-and-scaling)).

## Quick Start

Install [Docker + Compose](https://docs.docker.com/get-docker/),
[`uv`](https://docs.astral.sh/uv/), `git`, and `make`. The REST examples below pipe
through `jq`; install it, or drop the pipe.

### Interactive setup

```bash
git clone https://github.com/hanlianlu/dlightrag.git
cd dlightrag
uv run prerequisite_setup.py
```

The wizard configures models, parser, secrets, and the local stack. It defaults to
self-hosted MinerU and is safe to rerun.

### Manual setup

```bash
git clone https://github.com/hanlianlu/dlightrag.git
cd dlightrag
cp .env.example .env
mkdir -p "${HOME}/.dlightrag/skills"
```

The last command prepares the default read-only operator Skills bind source; when
`COMPOSE_GLOBAL_SKILLS_DIR` selects another host path, create that directory
instead. `docker compose` reads `.env` for the three services that need it, so the
copy is not optional. Add the keys required by your `config.yaml` model blocks:

```bash
DLIGHTRAG_MODELS__CHAT__DEFAULT__API_KEY=...
DLIGHTRAG_MODELS__EMBEDDING__API_KEY=...
DLIGHTRAG_MODELS__CHAT__ROLES__EXTRACT__API_KEY=...
DLIGHTRAG_MODELS__CHAT__ROLES__KEYWORD__API_KEY=...
DLIGHTRAG_MODELS__CHAT__ROLES__QUERY__API_KEY=...
DLIGHTRAG_MODELS__CHAT__ROLES__VLM__API_KEY=...
DLIGHTRAG_MODELS__RERANK__API_KEY=...
```

Choose one parser; DlightRAG reads exactly one `corpus.sidecars` block.

**MinerU on the host (the checked-in default; macOS, Linux, and WSL2 only):**

```bash
make mineru-install
make mineru-service-install  # installs and starts the background service
curl http://127.0.0.1:8210/health

cp packages/mineru/.env.example .env.mineru   # optional: overrides and title-aided
docker compose up -d
docker compose ps
curl http://127.0.0.1:8100/ready
```

`docker compose ps` says the containers are up; `/ready` is the application's own
verdict and it reflects parser and model reachability, so read it before the first
ingest — `/health` alone shows a degraded view rather than failing
([Parser Services](docs/operations.md#parser-services)).

If PostgreSQL is the container that never settles — it restarts forever with
`FATAL: bogus data in lock file` — an unclean Docker shutdown crashed it on a lock file,
not on its data: `make postgres-recover` repairs it without rebuilding the database
([PostgreSQL](docs/postgresql.md#startup-recovery)).

The first `up` builds the PostgreSQL image from `postgres/`: it compiles
`pg_textsearch` and `pg_jieba` against PostgreSQL 18, so it takes a few minutes
once per machine. The application and browser-pool images are multi-stage builds
(frontend, a Rust toolchain, fonts) and their first build is slow too.

Everything works without `.env.mineru`; it exists for overrides. The template is safe
to copy verbatim — it leaves the install extras unset so the installer derives them
(core,mlx on Apple Silicon, core elsewhere), and no active line in it changes an
existing setup. Only title correction needs it: fill `MINERU_TITLE_AIDED_API_KEY`,
`_BASE_URL`, and `_MODEL`, then run `make mineru-title-aided`, which points MinerU's
title correction at a model from your own catalogue
([Parser Services](docs/operations.md#parser-services)). The wizard writes the file for
you.

Use `make mineru-api` when the platform cannot install a background user service: the
MinerU commands run on macOS, Linux, and WSL2 (`make mineru-install` itself installs
into `.venv-mineru`, and `make mineru-service-*` manages a launchd or systemd --user
unit). Every other path here is platform-neutral, and a host without a user service
manager can still run Docling in Compose below.

**Docling in Compose (CPU, nothing installed on the host):** in `config.yaml`,
replace the `mineru` block with

```yaml
corpus:
  sidecars:
    docling:
      endpoint: http://docling:5001
      code_formula_preset: null
```

then start the stack with the `docling` profile:

```bash
docker compose --profile docling up -d
docker compose ps
```

Open <http://localhost:8100/web/>. The stack publishes:

| Service | Address |
|---|---|
| REST API and Web | `http://127.0.0.1:8100` |
| MCP streamable HTTP | `http://127.0.0.1:8101` |
| PostgreSQL | `127.0.0.1:5432` |

Parser changes affect only new parses. Configuration fields and parser operations
are documented in [Configuration](docs/configuration.md) and
[Operations](docs/operations.md).

### Native API

Run PostgreSQL in Docker and the API on the host:

```bash
docker compose up -d postgres
uv sync
DLIGHTRAG_CORPUS__SIDECARS__MINERU__LOCAL_ENDPOINT=http://127.0.0.1:8210 \
  uv run dlightrag-api
```

The checked-in config is Docker-first, so a native process overrides the parser
host alias with loopback. Put local sources under
`./dlightrag_storage/inputs/<workspace>`; DlightRAG only reads them and keeps
its own corpus files under `./dlightrag_storage/corpus/<workspace>`.

The `charts` and `interactive-html` Skills shell out to `echarts-render` and
`html-report`; the image puts both on `PATH`, and `make chart-render-link` does the
same for a native process. Rasterizing needs the image's `resvg` and Noto Sans SC
([Chart rendering](docs/operations.md#agent-chart-rendering) covers a source
install, including `scripts/install-chart-tools.sh` for Linux x86_64).

### Read-only replica

A `reader` serves every read surface against the same database and refuses
corpus writes with HTTP 503 at acceptance
([Service roles](docs/postgresql.md#service-roles-and-shared-artifacts)):

```bash
docker compose --profile reader up -d dlightrag-reader
curl -s localhost:8102/health   # service_role: reader
```

## Use DlightRAG

### Web

The Web UI supports workspace and file management, durable Fast and Research
conversations, answer attachments, citations, source highlights, child-agent
status, and typed Answer Artifacts. Research publishes only Agentspace roots it
explicitly attaches; prose links alone never publish files. English, Chinese,
and automatic browser language modes are available under Settings.

### REST

Ingestion, Retrieval, and Answer create durable Runs and return `202 Accepted`;
corpus mutations require a stable idempotency key:

```bash
INGEST_RUN=$(curl -sS -X POST http://localhost:8100/runs/corpus/ingest \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: ingest-report-1" \
  -d '{"source_type":"local","path":"report.pdf"}' | jq -r .run_id)
curl "http://localhost:8100/runs/$INGEST_RUN"
```

[Interfaces](docs/interfaces.md#ingestion) carries the same shape for retrieval and
answers, plus responses, pagination, SSE, attachments, citations, and every
transport contract.

### MCP

For a local stdio client:

```json
{
  "mcpServers": {
    "dlightrag": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/dlightrag", "dlightrag-mcp", "--env-file", "/absolute/path/to/dlightrag/.env"]
    }
  }
}
```

The Compose stack also exposes streamable HTTP on port 8101. MCP lets external
agents manage source-backed knowledge, delegate durable Retrieval and Answer
Runs, steer or continue research, and read its Artifacts. Its 18 tools cover
these tasks; personal Memory management, service administration, and direct
child-agent supervision are available through Web and REST. All interfaces reuse
the same Application services and authorization rules — see
[Interfaces](docs/interfaces.md#mcp-server) for the tool list.

### Python

There is no PyPI distribution: the runtime is consumed from the clone
`uv sync` prepared above, where `uv run` uses the locked environment. An
application in its own project can add the clone as an editable path dependency
instead (`uv add --editable /absolute/path/to/dlightrag`), and the Memory
workspace package resolves from the same clone.

Create an application with `create_application(config)`, use
`application.corpus_mutations` for durable corpus writes,
`application.retrieval` for durable Retrieval, and `application.answers` for
durable Answers, then call `application.aclose()`. Complete typed examples are in
[Interfaces](docs/interfaces.md#in-process-application).

## Security

Loopback development can use `access.auth_mode: none`. A deployment for one owner
uses `simple`, a bearer token; one shared by several people uses `jwt`, where an
issuer and an audience verify each person's token and Access Rules and workspace
creators decide what they may do. DlightRAG does not issue tokens or replace an
ingress WAF, rate limiter, TLS terminator, or identity provider. See
[Security](docs/security.md).

## Development

Node 26 (the pin in `Dockerfile` and CI) drives the frontend and chart tooling, and
`ci-full`/`ci-e2e` want a running PostgreSQL 18. `make ci` installs what it needs.

```bash
make sync-dev    # uv sync --locked --group dev
make hooks
make ci          # lint, security, format, types, architecture, frontend, unit
make ci-full     # plus integration tests
make ci-e2e      # plus E2E smoke
```

## Documentation

| Document | Owns |
|---|---|
| [Documentation map](docs/README.md) | Reading paths for each role, and where each topic is owned |
| [Architecture](docs/architecture.md) | Runtime ownership, deployment topology, flows, storage, layering |
| [Domain Language](docs/domain-language.md) | Canonical product vocabulary and the concept map |
| [Configuration](docs/configuration.md) | Configuration precedence, fields, defaults, examples |
| [Interfaces](docs/interfaces.md) | Python, REST, MCP, and Web contracts |
| [Retrieval and Answer](docs/retrieval-answer.md) | Retrieval, fusion, rerank, packing, citations, highlights |
| [Run Runtime](docs/run-runtime.md) | Run lifecycle, lanes, admission limits, repair, retention, scaling |
| [Personal MCP Connections](docs/personal-mcp-connections.md) | Owner-managed MCP Connections, authorization, and their binding into Research Runs |
| [Answer Resource Reading](docs/resource-reading.md) | `read`/`view` contracts for Answer attachments and Resources, conversion routes |
| [Security](docs/security.md) | Authentication, authorization, ingress and content boundaries |
| [Trust](docs/trust/data-map.md) | Data map, retention, subprocessors, and the [vulnerability disclosure](docs/trust/vulnerability-disclosure.md) at `/.well-known/security.txt` |
| [PostgreSQL](docs/postgresql.md) | PostgreSQL requirements, schema ownership, tuning |
| [Operations](docs/operations.md) | Executable runbooks: reset, rebuild, parsers, ClickStack, recovery |
| [Observability](docs/observability.md) | Trace structure, span vocabulary, attribution, redaction, deployment labels |
| [Web Theme Design](docs/web-theme-design.md) | Web appearance and interaction decisions |
| [Skills Routing Eval](docs/skills-routing-eval.md) | The built-in Skills' routing suite and its CI half |
| [Memory Pass Eval](docs/memory-pass-eval.md) | What the Session Memory Pass would remember, graded across models and prompts |

ADRs under `docs/adr/` record design decisions; they are not required reading for
operating DlightRAG.

## License

Apache License 2.0. See [LICENSE](LICENSE).

Built by HanlianLyu. Contributions welcome.
