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
  -> RAGServiceManager
  -> RAGService
  -> LightRAG main
  -> PostgreSQL 18 storage ecosystem
```

DlightRAG has one unified production RAG path: LightRAG provides fusional one-hop graph traversal and vector retrieval. DlightRAG adds product-layer metadata governance, hybrid BM25 sparse retrieval, fused visual-vector alignment, orchestration, citations, highlighting and standardized interfaces. The full runtime and code-layer view is in
[docs/architecture.md](docs/architecture.md).

## Choose Your Deployment Path

| Path | Use this when | PostgreSQL | Parser endpoint | Security | Start here |
|---|---|---|---|---|---|
| Local Docker | Developer machine, Web UI, smoke tests | Compose PG18 | Host-native MinerU-compatible sidecar | `auth_mode: none` on loopback | [Quick Start](#quick-start) |
| Native API | API process runs on host, PostgreSQL stays in Docker | Compose PG18 | Any reachable MinerU-compatible endpoint | Local or explicit auth | [Native API Variant](#native-api-variant) |
| Shared service | Remote users, agents, team workspace | Managed or self-hosted PG18 | Official MinerU API or independent parser service | `simple` or `jwt` | [PostgreSQL](docs/postgresql.md), [Configuration](docs/configuration.md), [Security](docs/security.md) |
| Enterprise | Multi-user internal product | Managed PG18 | Independently operated parser service | `jwt` + JWKS, optional claim access control | [Security](docs/security.md), [PostgreSQL](docs/postgresql.md), [Configuration](docs/configuration.md) |

Do not install MinerU into the DlightRAG app container. DlightRAG consumes the
MinerU-compatible HTTP endpoint that LightRAG expects. On macOS, keep MinerU as
a native host process so MLX/MPS acceleration is available. On Linux GPU, run
MinerU as an independent service/router or use the official API.

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

From a fresh clone, an interactive wizard configures your models, sets up MinerU
(local or the official cloud API), brings up the stack, and ends with a clickable
Web UI link:

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
DLIGHTRAG_RERANK__API_KEY=...
```

These match the checked-in `config.yaml`, which configures DeepSeek extract and
keyword roles plus Voyage reranking. If you remove role-specific model blocks or
switch rerank back to `chat_llm_reranker`, the corresponding role/rerank keys can
be omitted.

Normal behavior lives in [config.yaml](config.yaml): model names, parser
sidecar settings, metadata schema, retrieval breadth, auth mode, Langfuse
behavior, and deployment endpoints. Deep config reference is in
[docs/configuration.md](docs/configuration.md).

2. Install and start a native MinerU sidecar if one is not already running:

```bash
cp .env.mineru.example .env.mineru
make mineru-install
make mineru-api
```

`make mineru-api` serves `http://127.0.0.1:8210` by default and blocks in the
current terminal. Docker Compose does not run MinerU; it maps the host-native
endpoint into DlightRAG containers as `http://host.docker.internal:8210`.

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
| `lightrag-gui` | Upstream LightRAG graph browser | `127.0.0.1:9621` |
| `postgres` | PG18 ecosystem | `5432` |

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
uv run dlightrag-api
```

Native runs can ingest host paths directly because the API process sees the
same filesystem as your shell.

## Use DlightRAG

### Web

The Web UI is served by the REST API at `/web/`. It supports workspace
selection, file/folder upload, durable principal-scoped conversations and
current-turn images, citations, source panels, and semantic highlights. The
Web-only conversation lifecycle provides New chat, select, rename, delete, and
reload persistence with 30-day inactivity retention. `Search in: All authorized
workspaces` is the answer default; the independent `Files in` selector remains a
single-workspace file-management target.

REST, MCP, and Python answer/retrieve calls remain stateless. Answer calls
accept an optional caller-supplied `history` of prior turns for multi-turn
follow-ups, but never a Web conversation ID or server-stored history: the client
owns conversation storage and re-sends the turns it wants on each request. Web
current-image admission is configurable and defaults to three
images at 15 MiB each. Durable current-turn images always have priority over any
historical image selection; historical images that miss a transport slot
contribute their stored text descriptions rather than raw pixels.

### REST

REST ingest starts durable background jobs. Poll the job endpoint for status.

```bash
curl -X POST http://localhost:8100/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type": "local", "path": "report.pdf"}'

curl http://localhost:8100/ingest/jobs/<job_id>

curl -X POST http://localhost:8100/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?", "stream": false}'
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

from dlightrag import IngestSpec, RAGServiceManager
from dlightrag.config import DlightragConfig, EmbeddingConfig, LLMConfig, ModelConfig


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
    manager = await RAGServiceManager.acreate(config)
    try:
        await manager.aingest(
            workspace,
            IngestSpec(source_type="local", path="./docs"),
        )
        answer = await manager.aanswer("What are the key findings?", workspace=workspace)
        print(answer.answer)
    finally:
        await manager.aclose()


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
`list_files`, `delete_files`, `list_workspaces`, `create_workspace`, and
`delete_workspace`.

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

**Metadata.** Declare filterable custom fields once in configuration. Ingest
calls pass values. Request-level metadata is the batch default; manifest or
`SourceDocument` metadata overlays it per document.

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
gateway for login and token issuance. Full guidance is in
[docs/security.md](docs/security.md).

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
make ci          # lint + type-check + architecture + unit tests
make ci-full     # above + integration tests
make ci-e2e      # above + E2E smoke
```

Frontend checks after editing `frontend/`:

```bash
cd frontend
npm run typecheck
npm run build
npm run lint:css
```

`npm run build` writes the browser bundle to `src/dlightrag/web/static/generated/`;
commit those generated files with frontend changes.

Evaluation with RAGAS is documented in [docs/evaluation.md](docs/evaluation.md).

## Documentation Map

- [docs/architecture.md](docs/architecture.md) - runtime ownership, storage topology, and code layering.
- [docs/interfaces.md](docs/interfaces.md) - SDK, REST, MCP, and Web contracts.
- [docs/security.md](docs/security.md) - auth, JWT/JWKS, IdP boundaries, and access control.
- [docs/configuration.md](docs/configuration.md) - configuration precedence, fields, and defaults.
- [docs/retrieval-answer.md](docs/retrieval-answer.md) - retrieval, filters, BM25, fusion, rerank, answers, citations, and highlights.
- [docs/postgresql.md](docs/postgresql.md) - PostgreSQL requirements and tuning.
- [docs/operations.md](docs/operations.md) - maintenance commands and recovery workflows.
- [docs/evaluation.md](docs/evaluation.md) - RAGAS evaluation workflow.
- [LightRAG API Server docs](https://github.com/HKUDS/LightRAG/blob/main/docs/LightRAG-API-Server.md) - upstream parser routing and MinerU official API contract.
- [MinerU Docker deployment docs](https://opendatalab.github.io/MinerU/quick_start/docker_deployment/) - Linux/WSL2 Docker support and macOS warning.

## License

Apache License 2.0. See [LICENSE](LICENSE).

Built by HanlianLyu. Contributions welcome.
