# DlightRAG

[![PyPI](https://img.shields.io/pypi/v/dlightrag)](https://pypi.org/project/dlightrag/)
[![CI](https://github.com/hanlianlu/dlightrag/actions/workflows/ci.yml/badge.svg)](https://github.com/hanlianlu/dlightrag/actions/workflows/ci.yml)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/hanlianlu/DlightRAG)

DlightRAG is a production ready multimodal RAG service built on LightRAG. It 
offers superior context intelligence, great accuracy with citation / highlight grounding, and unified interfaces for REST, Web, MCP, and Python SDK clients. It is designed for developers, seasoned users and teams who need a reliable RAG core service with cutting edge features integrated into their workflows and products. 

Status: Python 3.14. Storage: PostgreSQL 18 ecosystem. License: Apache-2.0.

## Architecture At A Glance

<p align="center">
  <img src="docs/architecture.svg" alt="DlightRAG Architecture" width="1080" />
</p>

```text
Clients
  -> REST / Web / MCP / SDK adapters
  -> Application (composition)
      -> WorkspaceRag -> LightRAG main -> corpus storage
      -> dlightrag.runtime RunCoordinator -> Answer executor + root PG adapter
      -> dlightrag-rag-core corpus ports -> root PG corpus adapter
      -> ApplicationHealth -> liveness/readiness projection
  -> PostgreSQL 18 storage ecosystem
```

DlightRAG has one unified production RAG path: LightRAG provides fusional one-hop graph traversal and vector retrieval. DlightRAG adds product-layer metadata governance, hybrid BM25 sparse retrieval, fused visual-vector alignment, orchestration, citations, highlighting and standardized interfaces. The full runtime and code-layer view is in
[docs/architecture.md](docs/architecture.md).

The repository is one UV workspace with a lockstep release train. Reusable
model settings, provider lifecycles, embedding/rerank execution, media, and
capability probing ship as `dlightrag-ai`; generic tool execution ships as
`dlightrag-agent-core`; LightRAG adapters, rerank orchestration, and
storage-neutral retrieval records/fusion ship as `dlightrag-rag-core`. The
`dlightrag` distribution maps product configuration and composes those cores
into the REST, Web, MCP, SDK, and PostgreSQL-backed product. CI builds and
inspects all four wheels outside the editable workspace. Inside that root
distribution, `dlightrag.runtime` owns storage-neutral durable-run contracts and
coordination; `dlightrag-rag-core` owns storage-neutral corpus coordination and
maintenance ports; and `dlightrag.adapters.postgres` owns every concrete
PostgreSQL implementation. `ApplicationHealth` is the single process-health
state projected by status interfaces.

## Choose Your Deployment Path

| Path | Use this when | PostgreSQL | Parser endpoint | Security | Start here |
|---|---|---|---|---|---|
| Local Docker | Developer machine, Web UI, smoke tests | Compose PG18 | Host-native MinerU (default) or optional Docling profile | `auth_mode: none` on loopback | [Quick Start](#quick-start) |
| Native API | API process runs on host, PostgreSQL stays in Docker | Compose PG18 | Any reachable configured MinerU or Docling endpoint | Local or explicit auth | [Native API Variant](#native-api-variant) |
| Shared service | Remote users, agents, team workspace | Managed or self-hosted PG18 | Official MinerU API or independent parser service | `simple` or `jwt` | [PostgreSQL](docs/postgresql.md), [Configuration](docs/configuration.md), [Security](docs/security.md) |
| Enterprise | Multi-user internal product | Managed PG18 | Independently operated parser service | `jwt` + JWKS, optional claim access control | [Security](docs/security.md), [PostgreSQL](docs/postgresql.md), [Configuration](docs/configuration.md) |

Do not install a parser into the DlightRAG app container. Configure one
`parser_sidecars.mineru` or `parser_sidecars.docling` block; DlightRAG derives
LightRAG routing automatically. If both blocks are present, MinerU takes
priority. On macOS, keep MinerU as a native host process for MLX/MPS, or use the
optional Docling Compose profile.

## Quick Start

**Prerequisites.** Install [Docker + Compose](https://docs.docker.com/get-docker/)
(runs the API and PostgreSQL), [`uv`](https://docs.astral.sh/uv/) (builds the
isolated MinerU sidecar environment), plus `git` and `make`. DlightRAG targets
**Python 3.14**; `uv` installs it for you (`uv python install 3.14`), so a system
Python is not required for the Docker path.

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
inside containers, and `make mineru-install` builds its own isolated
`.venv-mineru`. Run `uv sync` only for the [Native API Variant](#native-api-variant)
or [development](#operations-and-development).

### One-command setup (recommended)

From a fresh clone, an interactive wizard configures your models, selects
MinerU (local/official) or Docling (bundled/external), brings up the stack, and
ends with a clickable Web UI link:

```bash
git clone https://github.com/hanlianlu/dlightrag.git
cd dlightrag
uv run prerequisite_setup.py
```

It writes `config.yaml` and `.env` for you (with timestamped backups) and is safe
to re-run. The wizard does not preserve the checked-in model choices: the minimum
path writes only `llm.default` plus `embedding`, while the custom path replaces
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
DLIGHTRAG_LLM__DEFAULT__API_KEY=...
DLIGHTRAG_EMBEDDING__API_KEY=...
DLIGHTRAG_LLM__ROLES__EXTRACT__API_KEY=...
DLIGHTRAG_LLM__ROLES__KEYWORD__API_KEY=...
DLIGHTRAG_LLM__ROLES__QUERY__API_KEY=...
DLIGHTRAG_LLM__ROLES__VLM__API_KEY=...
DLIGHTRAG_RERANK__API_KEY=...
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

2. Choose one parser. The checked-in config uses a native MinerU sidecar:

```bash
cp .env.mineru.example .env.mineru
make mineru-install
make mineru-api
```

`make mineru-api` serves `http://127.0.0.1:8210` by default and blocks in the
current terminal. Its config block uses `http://host.docker.internal:8210`,
which is reachable from the DlightRAG containers.

To run MinerU in the background instead (launchd on macOS, `systemd --user` on
Linux/WSL2), use the service targets — one command starts **both** the API
backend and the Gradio WebUI. The WebUI reuses that same backend automatically,
so it needs no extra setup and never loads a second copy of the models:

```bash
make mineru-service-install   # install + start at login
make mineru-service-status    # also: -start / -stop / -logs / -uninstall
```

The WebUI opens at `http://127.0.0.1:7860` (unauthenticated — keep it on
loopback). Set `MINERU_GRADIO_ENABLE=false` in `.env.mineru` to manage the API
backend alone.

Alternatively, comment/remove the MinerU block in `config.yaml`, enable the
commented Docling block, and start the optional official CPU image:

```bash
docker compose --profile docling up -d
```

The image follows `quay.io/docling-project/docling-serve-cpu:latest`. Set a
different official image with `DOCLING_SERVE_IMAGE`; external Docling users set
the block's `endpoint` instead and do not enable the profile.

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

The upstream LightRAG graph browser is opt-in: `docker compose --profile gui up -d`
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
DLIGHTRAG_PARSER_SIDECARS__MINERU__LOCAL_ENDPOINT=http://127.0.0.1:8210 \
  uv run dlightrag-api
```

A native run's managed input root is the host `./dlightrag_storage/inputs/<workspace>`,
so files dropped there are ingested by name; paths outside it are rejected on
every surface. The checked config is Docker-first, so the
command overrides its Docker host alias with the native loopback endpoint. A
native Docling deployment likewise sets its active block endpoint to
`http://127.0.0.1:5001`.

## Use DlightRAG

### Web

The Web UI is served by the REST API at `/web/`. Vite owns its static document
and hashed assets, while light-DOM Lit components own browser presentation and
route-driven state. It supports workspace selection, file/folder upload,
durable principal-scoped conversations and answer attachments, citations,
source/report panels, and semantic highlights. Desktop panel resizing uses the
Web Awesome Split Panel component behind DlightRAG tokens and persisted-width
state; compact layouts retain the native modal overlay behavior. The
Web-only conversation lifecycle provides New chat, select, rename, delete, and
reload persistence with 30-day inactivity retention. `Search in: All authorized
workspaces` is the answer default; the independent `Files in` selector remains a
single-workspace file-management target.

REST, MCP, and Python answer/retrieve calls remain stateless. Answer calls
accept an optional caller-supplied `history` of prior turns for multi-turn
follow-ups, but never a Web conversation ID or server-stored history: the client
owns conversation storage and re-sends the turns it wants on each request. All
channels take the same answer inputs: a query plus optional **attachments**
(images, PDFs, Office documents, HTML/CSV, or HTTPS references). Attachments
become request-local resources read on demand — deterministic text decoding and
conversion first, focused VLM inspection for figures — and are bounded by
`answer.max_attachments` (default 6), a per-attachment size cap (100 MiB), and a
per-request total (128 MiB). Uploaded bytes are stored once as owner-scoped
content-addressed artifacts owned by the run, and historical ones are
re-registered lazily on follow-ups. The separate `/retrieve` path keeps its own
`query_images` current-image inputs for knowledge-base visual search.

### Durable answers

Every answer is one durable run with one identifier and one lifecycle, shared by
REST, MCP, Web, the SDK, the CLI, and evaluation. `POST /answer` returns HTTP 202
with the run's status, events, and cancel URLs; the run outlives its creating
request, so a disconnected client only detaches. Events are reconnectable SSE
resumed by durable sequence, a process restart resumes from the latest completed
control turn, and `DELETE /answer/{run_id}` is the only client action that
cancels. Event logs are always trimmed 30 days after a run finishes, and the
events endpoint then answers 410 while the result stays readable from the status
endpoint; the terminal run row is pruned at the same age, except a succeeded run
a Web conversation still shows, whose row survives while that conversation does.
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
from dlightrag.config import DlightragConfig, EmbeddingConfig, LLMConfig, ModelConfig
from dlightrag.services.answers import AnswerRequest
from dlightrag.services.corpora import IngestSpec


async def main() -> None:
    workspace = "research_notes"
    config = DlightragConfig(
        workspace=workspace,
        working_dir="./dlightrag_storage/sdk_demo",
        llm=LLMConfig(
            default=ModelConfig(
                provider="openai",  # protocol family: openai | anthropic | gemini (vendor via base_url)
                model="gpt-4.1-mini",
                api_key=os.environ["OPENAI_API_KEY"],
                temperature=0.2,
            )
        ),
        embedding=EmbeddingConfig(
            provider="openai_compatible",
            model="text-embedding-3-large",
            api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://api.openai.com/v1",
            dim=3072,
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
DLIGHTRAG_MCP_TRANSPORT=streamable-http \
DLIGHTRAG_MCP_HOST=127.0.0.1 \
dlightrag-mcp
```

MCP tools include `retrieve`, `answer`, `ingest`, `get_ingest_job`,
`cancel_ingest_job`, `list_files`, `delete_files`, `list_workspaces`,
`create_workspace`, and `delete_workspace`, plus `get_capabilities` for image
and metadata-filter capability discovery. See
[docs/interfaces.md](docs/interfaces.md#mcp-server)
for the authoritative tool-result contract.

## Core Concepts

**Workspaces.** A workspace is the primitive isolation unit for indexed data,
metadata, jobs, files, and queries. Query calls can target one workspace or
federate across multiple workspaces.

**Ingestion sources.** Local files, Web uploads, S3, Azure Blob, public/signed
HTTPS URLs, and SDK `AsyncDataSource` connectors flow through the same ingest
contract. Web and REST uploads are staged under DlightRAG's managed
`working_dir/inputs/<workspace>/` tree, then copied into the workspace input
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

**Runtime storage.** Docker Compose stores `working_dir` in the
`dlightrag_data` named volume mounted at `/app/dlightrag_storage`; the host
`./dlightrag_storage` directory is only used by native, non-Docker runs.

**Metadata.** Pass any custom fields through `metadata` on ingest; they are
stored as sent and are filterable without being declared first. Request-level
metadata is the batch default; manifest or `SourceDocument` metadata overlays it
per document.

**Retrieval and answers.** DlightRAG uses LightRAG `mix` as the base retrieval
mode, then adds metadata filtering, BM25, optional direct image retrieval, RRF
fusion, reranking, answer packing, citations, and optional semantic highlights.
The detailed mechanism is in [docs/retrieval-answer.md](docs/retrieval-answer.md).

**Observability.** Langfuse tracing is optional. Non-secret SDK behavior is set
in [docs/configuration.md](docs/configuration.md). To run the bundled local
Langfuse stack (`make langfuse-up`) and view traces, see
[docs/operations.md](docs/operations.md#local-langfuse-observability).

## Security Model

Local loopback development can use `auth_mode: none`. Shared or exposed
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
- [docs/interfaces.md](docs/interfaces.md) - SDK, REST, MCP, and Web contracts.
- [docs/security.md](docs/security.md) - auth, JWT/JWKS, IdP boundaries, and access control.
- [docs/roadmap.md](docs/roadmap.md) - decided follow-ups that are not a milestone yet.
- [docs/configuration.md](docs/configuration.md) - configuration precedence, fields, and defaults.
- [docs/retrieval-answer.md](docs/retrieval-answer.md) - retrieval, filters, BM25, fusion, rerank, answers, citations, and highlights.
- [docs/postgresql.md](docs/postgresql.md) - PostgreSQL requirements and tuning.
- [docs/durable-answer-runs.md](docs/durable-answer-runs.md) - the durable Answer run contract.
- [docs/operations.md](docs/operations.md) - maintenance commands and recovery workflows.
- [docs/evaluation.md](docs/evaluation.md) - RAGAS evaluation workflow.
- [docs/web-theme-design.md](docs/web-theme-design.md) - current Web appearance, semantic geometry, panel, interaction, and accessibility design.
- [LightRAG API Server docs](https://github.com/HKUDS/LightRAG/blob/main/docs/LightRAG-API-Server.md) - upstream parser routing and external parser contracts.
- [MinerU Docker deployment docs](https://opendatalab.github.io/MinerU/quick_start/docker_deployment/) - Linux/WSL2 Docker support and macOS warning.
- [Docling Serve](https://github.com/docling-project/docling-serve) - official Docling HTTP service and container images.

## License

Apache License 2.0. See [LICENSE](LICENSE).

Built by HanlianLyu. Contributions welcome.
