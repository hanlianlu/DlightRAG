# DlightRAG

[![PyPI](https://img.shields.io/pypi/v/dlightrag)](https://pypi.org/project/dlightrag/)
[![CI](https://github.com/hanlianlu/dlightrag/actions/workflows/ci.yml/badge.svg)](https://github.com/hanlianlu/dlightrag/actions/workflows/ci.yml)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/hanlianlu/DlightRAG)

DlightRAG is a production-ready multimodal RAG service built on LightRAG. It offers superior context intelligence, great accuracy with citation / highlight grounding, and unified interfaces for REST, Web, MCP, and Python SDK clients. It is designed for developers, seasoned users and teams who need a reliable RAG core service with cutting edge features integrated into their workflows and products.

Status: Python 3.14. Storage: PostgreSQL 18 ecosystem. License: Apache-2.0.

## Architecture At A Glance

<p align="center">
  <img src="docs/architecture.svg" alt="DlightRAG system context showing callers, the service boundary, external integrations, PostgreSQL, and corpus artifacts" width="1080" />
</p>

This is intentionally a system-context view: it shows DlightRAG as one system
and does not mix internal modules, execution steps, package imports, or database
entities into the same arrows.

DlightRAG has one unified production RAG path: LightRAG provides fusional one-hop graph traversal and vector retrieval. DlightRAG adds product-layer metadata governance, hybrid BM25 sparse retrieval, fused visual-vector alignment, orchestration, citations, highlighting and standardized interfaces. Research mode is a full agent loop with seven Pi-parity filesystem tools (`read`, `bash`, `edit`, `write`, `grep`, `find`, `ls`), durable per-run workspaces, foreground child agents, streamed tool progress, and verified image snapshots attached straight into the model call. The full runtime, deployment, and code-layer views are in
[docs/architecture.md](docs/architecture.md).

The repository is one UV workspace with two lockstep distributions. The root
`dlightrag` distribution contains three internal deep modules: `dlightrag.ai`
owns model settings and provider lifecycles, `dlightrag.agent` owns generic tool
and turn mechanics, and `dlightrag.rag` owns LightRAG integration plus
storage-neutral retrieval and corpus contracts. The root composes those modules
into the REST, Web, MCP, SDK, and PostgreSQL-backed product. Owner Profile Memory
remains the independently installable `dlightrag-memory` distribution for both
DlightRAG and external stdio MCP hosts. CI builds and inspects the root and
Memory wheels outside the editable workspace. `dlightrag.runtime` owns
storage-neutral durable-run contracts and coordination;
`dlightrag.adapters.postgres` owns every concrete product PostgreSQL
implementation; and `ApplicationHealth` is the single process-health state
projected by status interfaces.

## Choose Your Deployment Path

| Path | Use this when | PostgreSQL | Parser endpoint | Security | Start here |
|---|---|---|---|---|---|
| Local Docker | Developer machine, Web UI, smoke tests | Compose PG18 | Host-native Docling service (default); optional bundled Docling or MinerU | `access.auth_mode: none` on loopback | [Quick Start](#quick-start) |
| Native API | API process runs on host, PostgreSQL stays in Docker | Compose PG18 | Any reachable configured Docling or MinerU endpoint | Local or explicit auth | [Native API Variant](#native-api-variant) |
| Shared service | Remote users, agents, team workspace | Managed or self-hosted PG18 | Independently operated parser service | `simple` or `jwt` | [PostgreSQL](docs/postgresql.md), [Configuration](docs/configuration.md), [Security](docs/security.md) |
| Enterprise | Multi-user internal product | Managed PG18 | Independently operated parser service | `jwt` + JWKS, optional claim access control | [Security](docs/security.md), [PostgreSQL](docs/postgresql.md), [Configuration](docs/configuration.md) |

Do not install a parser into the DlightRAG app container. The checked-in config
consumes an independently operated Docling service at
`http://host.docker.internal:5001`; it does not start the optional Compose
Docling container. On Apple Silicon, the matching host service is
[docling-serve-mps](https://github.com/hanlianlu/docling-serve-mps). The bundled
CPU Docling profile and MinerU infrastructure remain available as alternatives.

## Quick Start

**Prerequisites.** Install [Docker + Compose](https://docs.docker.com/get-docker/)
(runs the API and PostgreSQL), [`uv`](https://docs.astral.sh/uv/) (runs setup and
native development commands), plus `git` and `make`. Start a reachable Docling
service before ingesting documents. DlightRAG targets **Python 3.14**; `uv`
installs it for you (`uv python install 3.14`), so a system Python is not
required for the Docker path.

```bash
# Install uv — macOS/Linux (see the uv docs for the Windows PowerShell command)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install make if you don't have it (git is usually already present):
#   macOS          xcode-select --install       # or: brew install make
#   Debian/Ubuntu  sudo apt-get install -y make
#   Fedora/RHEL    sudo dnf install -y make
#   Windows        no POSIX make on Windows — use WSL2 (Ubuntu): install Docker
#                  Desktop with the WSL2 backend, then inside WSL2 follow the
#                  Debian/Ubuntu line above. uv, make, and Python 3.14 all live
#                  inside WSL2 (a Linux environment), not on Windows.
```

The Docker Quick Start does **not** require `uv sync` — DlightRAG itself runs
inside containers. Run `uv sync` only for the
[Native API Variant](#native-api-variant) or
[development](#operations-and-development).

### One-command setup (recommended)

From a fresh clone, an interactive wizard configures your models, recommends an
external Docling endpoint, retains bundled Docling and MinerU as alternatives,
brings up the stack, and ends with a clickable Web UI link:

```bash
git clone https://github.com/hanlianlu/dlightrag.git
cd dlightrag
uv run prerequisite_setup.py
```

It writes `config.yaml` and `.env` for you (with timestamped backups) and is safe
to re-run. The wizard does not preserve the checked-in model choices: the minimum
path writes only `models.chat.default` plus `models.embedding`, while the custom path replaces
role-specific LLM blocks with the roles you choose. Prefer the manual steps below
if you'd rather configure everything by hand.

### Manual setup

1. Clone the repo and create a secrets file:

```bash
git clone https://github.com/hanlianlu/dlightrag.git
cd dlightrag
cp .env.example .env
```

Fill secrets in `.env`:

```bash
DLIGHTRAG_MODELS__CHAT__DEFAULT__API_KEY=...
DLIGHTRAG_MODELS__EMBEDDING__API_KEY=...
DLIGHTRAG_MODELS__CHAT__ROLES__EXTRACT__API_KEY=...
DLIGHTRAG_MODELS__CHAT__ROLES__KEYWORD__API_KEY=...
DLIGHTRAG_MODELS__CHAT__ROLES__QUERY__API_KEY=...
DLIGHTRAG_MODELS__CHAT__ROLES__VLM__API_KEY=...
DLIGHTRAG_MODELS__RERANK__API_KEY=...
```

These match the checked-in `config.yaml`, which configures DeepSeek extract and
keyword roles, OpenRouter default, query, and VLM roles, plus Voyage reranking.
Role overrides are
atomic: an omitted or blank key falls back to the complete default model
configuration instead of combining its endpoint with the default key. If you
remove role-specific model blocks or switch rerank back to `chat_llm_reranker`,
the corresponding role/rerank keys can be omitted. Reserve `api_key: null` for
a genuinely unauthenticated endpoint.

Normal behavior lives in [config.yaml](config.yaml): model names, parser
sidecar settings, metadata schema, retrieval breadth, auth mode, Langfuse
behavior, and deployment endpoints. Deep config reference is in
[docs/configuration.md](docs/configuration.md).

2. Start the external Docling service selected by the checked-in config. On
Apple Silicon, the local default is:

```bash
uv tool install docling-serve-mps
docling-serve-mps start
curl http://127.0.0.1:5001/health
```

The service binds to host loopback, while DlightRAG's Docker containers consume
it through `http://host.docker.internal:5001`. Do not enable the Compose Docling
profile at the same time because both services use host port 5001.

For another independently operated Docling service, change the checked-in
`corpus.sidecars.docling.endpoint`. The optional official CPU image remains
available with `docker compose --profile docling up -d`; in that case use
`http://docling:5001` and set `code_formula_preset: null`. MinerU remains
supported: replace the Docling block with a MinerU block, then use the existing
`make mineru-install`, `make mineru-api`, or `make mineru-service-install`
tooling described in [configuration](docs/configuration.md). Changing this
selection does not reparse an existing corpus; reset and reingest only when you
intend to rebuild it with the new parser.

3. Start DlightRAG and PostgreSQL:

```bash
docker compose up -d
docker compose ps
```

This starts:

| Service | Purpose | Host port |
|---|---|---|
| `dlightrag-api` | REST API + Web UI | `127.0.0.1:8100` |
| `dlightrag-mcp` | MCP streamable HTTP server | `127.0.0.1:8101` |
| `postgres` | PG18 ecosystem | `5432` |

The upstream LightRAG graph browser is opt-in:
`docker compose -f docker-compose.yml -f docker-compose.gui.yml --profile gui up -d`
serves it on `127.0.0.1:9621`.

4. Open the Web UI:

```text
http://localhost:8100/web/
```

Upload documents or images from the Files panel, then ask a question.

### Native API Variant

Use this when the API process should run on the host while PostgreSQL stays in
Docker:

```bash
docker compose up -d postgres
uv sync
DLIGHTRAG_CORPUS__SIDECARS__DOCLING__ENDPOINT=http://127.0.0.1:5001 \
  uv run dlightrag-api
```

A native run's managed input root is the host `./dlightrag_storage/inputs/<workspace>`,
so files dropped there are ingested by name; paths outside it are rejected on
every surface. The checked config is Docker-first, so the
command overrides its Docker host alias with the native loopback endpoint.

## Use DlightRAG

### Web

The Web UI is served by the REST API at `/web/`. Vite owns its static document
and hashed assets, while light-DOM Lit components own browser presentation and
route-driven state. It supports workspace selection, file/folder upload,
durable principal-scoped conversations and answer attachments, citations,
source/report panels, and semantic highlights. Desktop panel resizing uses the
Web Awesome Split Panel component behind DlightRAG tokens and persisted-width
state; compact layouts retain the native modal overlay behavior. The Web-only
conversation lifecycle provides New chat, select, rename, delete, reload,
durable resume/cancel, Research steering and child status, and minimal
follow-up/fork controls. Its recent-turn read window is not retention: linked
runs follow the configured retention floor. `Search in: All authorized
workspaces` is the answer default; the independent `Files in` selector remains a
single-workspace file-management target.

REST, MCP, and Python answer/retrieve calls require no Web conversation ID.
Answer calls accept optional caller-supplied `history`; an independent request
re-sends the turns it wants, while the accepted run pins that bounded history
for recovery and server-owned follow-up/fork. All
channels take the same answer inputs: a query plus optional **attachments**
(images, PDFs, Office documents, HTML/CSV, or HTTPS references). Attachments
become request-local resources read on demand — deterministic text decoding and
conversion first, focused VLM inspection for figures — and are bounded by
`answer.generation.max_attachments` (default 6), a per-attachment size cap (100 MiB), and a
per-request total (128 MiB). Uploaded bytes are stored once as owner-scoped
content-addressed artifacts owned by the run, and historical ones are
re-registered lazily on follow-ups. The separate `/retrieve` path keeps its own
`query_images` current-image inputs for knowledge-base visual search.

### Durable answers

Every answer is one durable run with one identifier and one lifecycle, shared by
REST, MCP, Web, the SDK, and evaluation. `POST /answer` returns HTTP 202
with the run's status, events, and cancel URLs; the run outlives its creating
request, so a disconnected client only detaches. Events are reconnectable SSE
resumed by durable sequence, and a restart folds the selected Agent Session head
or restarts an unfinished Fast stage. Clients may inspect status, usage,
evidence, and child lineage; steer a live Research run; start follow-up or fork
continuations; resume observation; or explicitly cancel. Event logs and
terminal rows follow `answer.runtime.answer_run_retention_days` (default 365);
an expired event log returns 410 while an unpruned result remains readable.
See [docs/durable-answer-runs.md](docs/durable-answer-runs.md).

### REST

REST ingest starts durable background jobs. Poll the job endpoint for status.

```bash
curl -X POST http://localhost:8100/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type": "local", "path": "report.pdf"}'

curl http://localhost:8100/ingest/jobs/<job_id>

# Answers are durable runs: POST returns 202 with the run's URLs.
RUN=$(curl -sS -X POST http://localhost:8100/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?"}' | jq -r .run_id)

curl -N "http://localhost:8100/answer/$RUN/events"   # follow, resumable by id
curl "http://localhost:8100/answer/$RUN"             # status + canonical result
```

All SDK, REST, MCP, Web contracts and response shapes are in
[docs/interfaces.md](docs/interfaces.md).

### Python SDK

```bash
uv add dlightrag
```

```python
import asyncio
import os

from dlightrag import Application
from dlightrag.access import DEPLOYMENT_OWNER_ID
from dlightrag.ai.settings import EmbeddingSettings, ModelRoleSettings, ModelSettings, ModelsSettings
from dlightrag.config import DeploymentSettings, DlightragConfig
from dlightrag.services.answers import AnswerRequest
from dlightrag.services.corpora import IngestSpec


async def main() -> None:
    workspace = "research_notes"
    config = DlightragConfig(
        deployment=DeploymentSettings(
            workspace=workspace,
            working_dir="./dlightrag_storage/sdk_demo",
        ),
        models=ModelsSettings(
            chat=ModelRoleSettings(
                default=ModelSettings(
                    provider="openai",  # protocol family: openai | anthropic | gemini (vendor via base_url)
                    model="gpt-5.6-luna",
                    api_key=os.environ["OPENAI_API_KEY"],
                    temperature=1.0,
                ),
            ),
            embedding=EmbeddingSettings(
                provider="openai",
                model="text-embedding-3-large",
                api_key=os.environ["OPENAI_API_KEY"],
                base_url="https://api.openai.com/v1",
                dim=3072,
            ),
        ),
    )
    application = await Application.acreate(config)
    try:
        await application.corpora.ingest(
            workspace,
            IngestSpec(source_type="local", path="./docs"),
        )
        answer = await application.answers.answer(
            AnswerRequest(
                query="What are the key findings?",
                workspaces=(workspace,),
            ),
            owner_id=DEPLOYMENT_OWNER_ID,
        )
        print(answer.answer)
    finally:
        await application.aclose()


asyncio.run(main())
```

`config.yaml` is optional for SDK users; constructor values take precedence.

### MCP

Use stdio when an agent starts DlightRAG as a subprocess:

```json
{
  "mcpServers": {
    "dlightrag": {
      "command": "uvx",
      "args": ["dlightrag-mcp", "--env-file", "/absolute/path/to/.env"]
    }
  }
}
```

Use streamable HTTP when multiple clients connect to a running service:

```bash
DLIGHTRAG_INTERFACES__MCP__TRANSPORT=streamable-http \
DLIGHTRAG_INTERFACES__MCP__HOST=127.0.0.1 \
dlightrag-mcp
```

MCP tools include retrieval, durable Answer start/status/cancel, steer,
follow-up, fork, resume, transcript and child-roster operations, plus corpus and
ingest administration and `get_capabilities`. See
[docs/interfaces.md](docs/interfaces.md#mcp-server)
for the authoritative tool-result contract.

## Core Concepts

**Workspaces.** A workspace is the primitive isolation unit for indexed data,
metadata, jobs, files, and queries. Query calls can target one workspace or
federate across multiple workspaces.

**Ingestion sources.** Local files, Web uploads, S3, Azure Blob, public/signed
HTTPS URLs, and SDK `AsyncDataSource` connectors flow through the same ingest
contract. Web and REST uploads are staged under DlightRAG's managed
`deployment.working_dir/inputs/<workspace>/` tree, then copied into the workspace input
root as retained local sources. Upload batch staging under `__uploads__/` is
cleaned by the durable ingest job after the handoff.

**Source downloads.** Every successful ingest remains downloadable, whether or
not DlightRAG retains a local copy. `source_uri` is stable provenance;
`download_uri` is the durable S3, Azure, or queryless public HTTPS locator used
when `retain_source_file` is false. Signed HTTPS fetch URLs need either a
separate durable locator or retention. A non-retained custom SDK connector must
provide `SourceDocument.download_uri` (or `download_uri_for_key`). DlightRAG
rejects a document before materialization when this contract cannot be met; it
never silently changes the caller's retention choice.

**Runtime storage.** Docker Compose stores `deployment.working_dir` in the
`dlightrag_data` named volume mounted at `/app/dlightrag_storage`; the host
`./dlightrag_storage` directory is only used by native, non-Docker runs.

**Metadata.** Pass any custom fields through `metadata` on ingest; they are
stored as sent and are filterable without being declared first. Request-level
metadata is the batch default; manifest or `SourceDocument` metadata overlays it
per document.

**Retrieval and answers.** DlightRAG uses LightRAG `mix` as the base retrieval
mode, then adds metadata filtering, BM25, optional direct image retrieval, RRF
fusion, reranking, answer packing, citations, and optional semantic highlights.
Research mode turns the answer into an agent run: the model works in a durable
per-run workspace with the seven-tool filesystem set in Pi order (`read`,
`bash`, `edit`, `write`, `grep`, `find`, `ls`), knowledge-base and web search,
memory, skills, and spawnable child agents; tool progress streams to Web as
metadata-only events, and reading a verified image attaches the original
snapshot to the next model call. The detailed mechanism is in
[docs/retrieval-answer.md](docs/retrieval-answer.md).

**Observability.** Langfuse tracing is optional. Non-secret SDK behavior is set
in [docs/configuration.md](docs/configuration.md). To run the bundled local
Langfuse stack (`make langfuse-up`) and view traces, see
[docs/operations.md](docs/operations.md#local-langfuse-observability).

## Security Model

Local loopback development can use `access.auth_mode: none`. Shared or exposed
deployments should enable auth:

| Mode | Use case |
|---|---|
| `simple` | One shared bearer token |
| `jwt` | Externally issued signed tokens |
| `jwt` + JWKS | OIDC-style issuers with key rotation |
| `jwt` + `jwt_claims` access control | Workspace/action permissions from verified claims |

DlightRAG verifies bearer tokens and can enforce workspace/action access
control. It does not issue OAuth tokens or manage users. Use an external IdP or
gateway for login and token issuance. Generic request-rate limiting, WAF rules,
DDoS protection, TLS termination, and connection caps belong to your ingress
(Front Door, Application Gateway, APIM, NGINX); DlightRAG ships no in-process
rate limiter. Full guidance is in [docs/security.md](docs/security.md).

## Operations And Development

Use [docs/operations.md](docs/operations.md) for the full stop/rebuild/restart
sequence and maintenance safety notes.

### Development setup:

```bash
uv sync
cd frontend && npm ci && cd ..
make hooks
```

Verification:

```bash
make ci          # the fast gate: lint, security, format, types, architecture, shell, frontend, unit tests
make ci-full     # above + integration tests
make ci-e2e      # above + E2E smoke
```

Frontend checks after editing `frontend/`:

```bash
make frontend-ci  # install, typecheck, CSS lint, Node + browser tests, build, audit
```

For an individual check, run its script under `frontend/` (`typecheck`, `test`,
`test:browser`, `build`, or `lint:css`). `npm run build` writes Vite-owned HTML
and hashed browser assets to `src/dlightrag/web/static/app/`. For HMR, run
`npm run dev`; the Vite server proxies authenticated browser APIs and support
assets to FastAPI on `127.0.0.1:8100`.
That directory is gitignored and rebuilt by `make ci`; the wheel and source
distribution include it through their Hatch artifact settings.

Evaluation with RAGAS is documented in [docs/evaluation.md](docs/evaluation.md).

## Documentation Map

- [docs/architecture.md](docs/architecture.md) - runtime ownership, storage topology, and code layering.
- [docs/domain-language.md](docs/domain-language.md) - the project's shared vocabulary for domain terms.
- [docs/interfaces.md](docs/interfaces.md) - SDK, REST, MCP, and Web contracts.
- [docs/security.md](docs/security.md) - auth, JWT/JWKS, IdP boundaries, and access control.
- [docs/configuration.md](docs/configuration.md) - configuration precedence, fields, and defaults.
- [docs/retrieval-answer.md](docs/retrieval-answer.md) - retrieval, filters, BM25, fusion, rerank, answers, citations, and highlights.
- [docs/postgresql.md](docs/postgresql.md) - PostgreSQL requirements and tuning.
- [docs/durable-answer-runs.md](docs/durable-answer-runs.md) - the durable Answer run contract.
- [docs/operations.md](docs/operations.md) - maintenance commands and recovery workflows.
- [docs/evaluation.md](docs/evaluation.md) - RAGAS evaluation workflow.
- [docs/web-theme-design.md](docs/web-theme-design.md) - current Web appearance, semantic geometry, panel, interaction, and accessibility design.
- [LightRAG API Server docs](https://github.com/HKUDS/LightRAG/blob/main/docs/LightRAG-API-Server.md) - upstream parser routing and external parser contracts.
- [Docling Serve](https://github.com/docling-project/docling-serve) - official Docling HTTP service and container images.
- [Docling Serve MPS](https://github.com/hanlianlu/docling-serve-mps) - host-native Apple Silicon service used by the checked-in local configuration.
- [MinerU Docker deployment docs](https://opendatalab.github.io/MinerU/quick_start/docker_deployment/) - optional Linux/WSL2 deployment and macOS warning.

## License

Apache License 2.0. See [LICENSE](LICENSE).

Built by HanlianLyu. Contributions welcome.
