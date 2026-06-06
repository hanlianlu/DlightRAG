# DlightRAG

[![PyPI](https://img.shields.io/pypi/v/dlightrag)](https://pypi.org/project/dlightrag/)
[![CI](https://github.com/hanlianlu/dlightrag/actions/workflows/ci.yml/badge.svg)](https://github.com/hanlianlu/dlightrag/actions/workflows/ci.yml)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/hanlianlu/DlightRAG)

DlightRAG is a multimodal RAG service built on LightRAG main. LightRAG owns
document parsing, staged ingest, chunks, document status, vectors, and the
knowledge graph. DlightRAG adds product-layer metadata governance,
PostgreSQL BM25, direct image-vector alignment, answer orchestration,
citations, REST, Web, SDK, and MCP interfaces.

Status: Python 3.12+. Storage: PostgreSQL 18 with pgvector, Apache AGE,
pg_textsearch, and pg_jieba. License: Apache-2.0.

## Quick Start

Start by choosing the topology. The model, parser, and database decisions are
different for local development and cloud deployment.

| Mode | Use this when | PostgreSQL | MinerU parser endpoint | Auth |
|---|---|---|---|---|
| Local | Developer machine, Web UI, smoke tests | Docker Compose PG18 | Native host sidecar on macOS/M4, or any reachable MinerU API/router | `none` on loopback; `simple` if exposed |
| Cloud | Shared service, remote users, agents | Managed or self-hosted PG18 | MinerU official API, or an independent GPU/API service | `simple` or `jwt` |

Do not install MinerU into the DlightRAG app container. DlightRAG consumes the
MinerU-compatible HTTP endpoint that LightRAG expects. On macOS this should be
a native host process so MLX/MPS acceleration is available. On Linux GPU, run
MinerU as an independent service/router or use the official API.

### Local Setup

1. Clone the repo and create a secrets file:

```bash
git clone https://github.com/hanlianlu/dlightrag.git
cd dlightrag
cp .env.example .env
```

Fill only secrets and deployment-only overrides in `.env`:

```bash
DLIGHTRAG_LLM__DEFAULT__API_KEY=...
DLIGHTRAG_EMBEDDING__API_KEY=...
DLIGHTRAG_LLM__ROLES__EXTRACT__API_KEY=...
DLIGHTRAG_LLM__ROLES__KEYWORD__API_KEY=...
```

Normal behavior lives in [config.yaml](config.yaml): model names, parser
sidecar settings, metadata schema, retrieval breadth, auth mode, Langfuse
behavior, and deployment endpoints. Rare PostgreSQL and retrieval tuning
belongs in [docs/PG.md](docs/PG.md) and
[docs/config-reference.md](docs/config-reference.md).

2. Install and start a native MinerU sidecar if one is not already running.

MinerU is intentionally a native sidecar in local development. Docker Compose
does not run MinerU; it runs DlightRAG, MCP, and PostgreSQL, then connects the
DlightRAG containers back to the host-native MinerU endpoint.

Create MinerU's own env file and install the dedicated MinerU virtual
environment once:

```bash
cp .env.mineru.example .env.mineru
make mineru-install
```

This creates `.venv-mineru` so MinerU's ML dependencies stay out of the
DlightRAG runtime. Re-run `make mineru-install` only when upgrading MinerU or
changing `.env.mineru` package extras such as `MINERU_INSTALL_EXTRAS`.

Choose `MINERU_INSTALL_EXTRAS` in `.env.mineru` for the local machine:

- Apple Silicon local development: `core,mlx`
- Linux or WSL CPU fallback: `core`
- Linux GPU service, when supported by the target MinerU release:
  `core,vllm` or `core,lmdeploy`

For a temporary foreground process on any local OS, run:

```bash
make mineru-api
```

`make mineru-api` blocks in the current terminal and serves
`http://127.0.0.1:8210` by default.

On macOS, service-manage the same API through launchd:

```bash
make mineru-service-install
make mineru-service-status
make mineru-service-logs
```

`make mineru-service-install` writes the LaunchAgent plist and starts the API.
Use `make mineru-service-start` later to restart an already installed service.

On Linux or WSL with systemd, service-manage the same API with a user service:

```bash
repo="$(pwd)"
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/dlightrag-mineru.service <<EOF
[Unit]
Description=DlightRAG MinerU API sidecar

[Service]
WorkingDirectory=${repo}
Environment=MINERU_ENV_FILE=${repo}/.env.mineru
ExecStart=${repo}/scripts/mineru/api.sh
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now dlightrag-mineru.service
systemctl --user status dlightrag-mineru.service
journalctl --user -u dlightrag-mineru.service -f
```

On a headless Linux host where the user service must survive logout, enable
lingering once with `loginctl enable-linger "$USER"`.

Endpoint alignment:

- Native MinerU listens on `MINERU_API_HOST:MINERU_API_PORT` from
  `.env.mineru`, defaulting to `http://127.0.0.1:8210`.
- [config.yaml](config.yaml) defaults
  `parser_sidecars.mineru.local_endpoint` to `http://127.0.0.1:8210`.
- Docker Compose maps that host-native endpoint into DlightRAG containers as
  `http://host.docker.internal:8210` through `MINERU_DOCKER_LOCAL_ENDPOINT`.
  Override that value in `.env` only if the DlightRAG containers must reach a
  different externally managed MinerU endpoint.

3. Start DlightRAG and PostgreSQL:

```bash
docker compose up -d
docker compose ps
curl http://localhost:8100/health
```

This starts:

| Service | Purpose | Host port |
|---|---|---|
| `dlightrag-api` | REST API + Web UI | `8100` |
| `dlightrag-mcp` | MCP streamable HTTP server | `127.0.0.1:8101` |
| `postgres` | PG18 + pgvector + AGE + pg_textsearch + pg_jieba | `5432` |

When `dlightrag-api` runs in Docker and MinerU runs as a native host process,
Compose maps `MINERU_LOCAL_ENDPOINT` inside the app container to
`http://host.docker.internal:8210`. Override `MINERU_DOCKER_LOCAL_ENDPOINT`
only when the app container must reach a different host-native endpoint.

4. Open the Web UI:

```text
http://localhost:8100/web/
```

Upload documents or images from the Files panel, then ask a question. Web
uploads are staged under DlightRAG's managed `working_dir/inputs/<workspace>/`
tree, which is also the root LightRAG parser workers can see.

### Cloud Setup

Cloud deployments should make the parser and database endpoints explicit.

1. Provide PostgreSQL 18 with:

- pgvector
- Apache AGE
- pg_textsearch
- pg_jieba
- `shared_preload_libraries=age,pg_textsearch,pg_jieba`

Use [docs/PG.md](docs/PG.md) for extension, SSL, pool, HNSW, and sizing notes.

2. Choose one MinerU mode in `config.yaml`.

For the MinerU official API:

```yaml
parser_sidecars:
  mineru:
    api_mode: official
    official_endpoint: https://mineru.net
    model_version: vlm
```

```bash
DLIGHTRAG_PARSER_SIDECARS__MINERU__API_TOKEN=...
```

For an independent MinerU API/router service:

```yaml
parser_sidecars:
  mineru:
    api_mode: local
    local_endpoint: https://your-mineru-service.company.internal
```

The setting name `local_endpoint` follows LightRAG/MinerU's local-protocol
environment contract. The endpoint can still be remote from the DlightRAG
container as long as it exposes the compatible MinerU HTTP API.

3. Enable auth before exposing REST or MCP:

```bash
DLIGHTRAG_AUTH_MODE=simple
DLIGHTRAG_API_AUTH_TOKEN=<openssl-rand-base64-32>
```

or:

```bash
DLIGHTRAG_AUTH_MODE=jwt
DLIGHTRAG_JWT_SECRET=<openssl-rand-base64-64>
DLIGHTRAG_JWT_ALGORITHM=HS256
```

When auth is enabled, set explicit CORS origins in `config.yaml` rather than
using `["*"]`.

4. Configure model credentials and PostgreSQL secrets through the deployment
secret store. Keep non-secret behavior in `config.yaml` so it can be reviewed
and versioned.

### Native API

Use this when DlightRAG runs outside Docker while PostgreSQL stays in Docker:

```bash
docker compose up -d postgres
uv sync
uv run dlightrag-api
```

Native runs can ingest host paths directly because the API process sees the
same filesystem as your shell.

### First API Calls

Local-source ingestion paths must be visible to the API process. In Docker,
prefer Web upload or mount a host directory into the API container.

```bash
curl -X POST http://localhost:8100/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type": "local", "path": "/absolute/path/visible/to/api"}'

curl -X POST http://localhost:8100/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?"}'

curl -X POST http://localhost:8100/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?", "stream": false}'
```

Full request and response details are in
[docs/response-schema.md](docs/response-schema.md).

### Python SDK

```bash
uv add dlightrag
cp .env.example .env
```

```python
import asyncio
from dotenv import load_dotenv
from dlightrag import DlightragConfig, RAGServiceManager

load_dotenv()

async def main():
    config = DlightragConfig()
    manager = await RAGServiceManager.create(config)
    try:
        workspace = "research_notes"
        await manager.acreate_workspace(workspace, display_name="Research Notes")
        await manager.aingest(workspace, source_type="local", path="./docs")

        contexts = await manager.aretrieve("What are the key findings?", workspace=workspace)
        print(contexts.contexts)

        answer = await manager.aanswer("What are the key findings?", workspace=workspace)
        print(answer.answer)
    finally:
        await manager.close()

asyncio.run(main())
```

### MCP Server

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

MCP tools: `retrieve`, `answer`, `ingest`, `list_files`, `delete_files`,
`list_workspaces`, `create_workspace`, and `delete_workspace`.

## Architecture

<p align="center">
  <img src="docs/architecture.svg" alt="DlightRAG Architecture" width="1080" />
</p>

<sub>Source: <a href="docs/architecture.drawio">docs/architecture.drawio</a>
and <a href="docs/module-layers.md">docs/module-layers.md</a>.</sub>

### Runtime Responsibilities

```text
Clients
  -> REST / Web / MCP / SDK adapters
  -> RAGServiceManager
       workspace routing, user scope, federation, read-after-write barriers
  -> RAGService
       one workspace runtime, ingest, retrieve, answer, reset
  -> LightRAG main
       parser routing, staged ingest, chunks, doc status, KG, vectors
  -> DlightRAG PostgreSQL stores
       metadata index, BM25 indexes, workspace and role metadata
```

LightRAG remains the core RAG engine. DlightRAG does not reimplement parser
sidecars, document status, KG extraction, or LightRAG `mix` retrieval.

### Ingestion Flow

```text
source file or upload
  -> DlightRAG metadata normalization
  -> LightRAG parser routing
       DOCX native route by default
       MinerU route for PDFs, Office files, images, tables, and equations
  -> LightRAG staged ingest
       chunks, multimodal semantic text, KG entities/relations, vector rows
  -> DlightRAG post-ingest alignment
       raw image embedding overwrites the canonical LightRAG drawing chunk vector
       chunk language labels update BM25 partial indexes
       declared metadata updates filterable columns
```

Source images and parser-extracted images both go through LightRAG's
multimodal path. DlightRAG does not create a second VLM description chunk.
Instead, it aligns the existing canonical LightRAG visual chunk with a raw
image embedding so visual retrieval, citations, and asset serving use the same
chunk id.

### Retrieval And Answer Flow

```text
query
  -> query planning and optional metadata filter inference
  -> strict metadata in-filtering when filters are explicit
  -> LightRAG mix retrieval
  -> direct query-image retrieval when the user attaches images
  -> pg_textsearch BM25 over the same candidate scope
  -> RRF fusion
  -> rerank
  -> answer packing with citations and bounded images
```

DlightRAG always uses LightRAG `mix` as the base retrieval mode. The DlightRAG
hybrid layer is separate from LightRAG's `hybrid` query mode.

### PostgreSQL Topology

DlightRAG has one application-level PostgreSQL endpoint. REST, Web, MCP, and
SDK surfaces all use the same configured write-capable endpoint, and LightRAG's
normal staged pipeline supports ingest and query in the same service process.

If production infrastructure uses managed read replicas, keep that routing in
the PostgreSQL/proxy/platform layer and expose the endpoint that DlightRAG
should use. DlightRAG does not carry a separate query runtime role, replica
credentials, or read-after-write policy. That keeps application behavior
identical across local, Docker, and cloud deployments.

## Configuration

Configuration uses typed settings in [config.yaml](config.yaml) and secrets or
deployment-only overrides in `.env`. The checked-in config is intentionally
curated: it keeps product and deployment choices visible while leaving rare
tuning to code defaults and [docs/config-reference.md](docs/config-reference.md).

Priority:

```text
constructor args > environment variables > .env > config.yaml > defaults
```

DlightRAG-owned environment variables use the `DLIGHTRAG_` prefix. Double
underscores address nested objects:

```bash
DLIGHTRAG_LLM__DEFAULT__API_KEY=...
DLIGHTRAG_EMBEDDING__API_KEY=...
DLIGHTRAG_RERANK__API_KEY=...
```

### Parser And MinerU

DlightRAG defaults to LightRAG native parsing for DOCX and MinerU for other
supported document formats. Parser routing is a product default, not a normal
user-facing setting. DlightRAG exposes the sidecar endpoint and visual context
controls needed for local/cloud deployment.

Important parser settings:

| Setting | Default | Meaning |
|---|---|---|
| `parser_sidecars.vlm.surrounding_leading_max_tokens` | `128` | Leading page context sent to VLM analysis |
| `parser_sidecars.vlm.surrounding_trailing_max_tokens` | `128` | Trailing page context sent to VLM analysis |
| `parser_sidecars.mineru.api_mode` | `local` | Uses a MinerU-compatible API endpoint instead of the official API |
| `parser_sidecars.mineru.local_endpoint` | `http://127.0.0.1:8210` | Native local sidecar endpoint for local development |
| `parser_sidecars.mineru.auxiliary_block_policy` | `conservative` | Drop discarded blocks, headers, footers, and printed page numbers |

`conservative` keeps ambiguous notes such as `aside_text`, `margin_note`, and
`page_footnote`. Set `extended` only when those should also be removed before
LightRAG chunking.

### Models And Roles

LLM role names follow LightRAG:

| Role | What it drives |
|---|---|
| `extract` | Entity and relationship extraction during ingest |
| `keyword` | LightRAG retrieval keyword extraction |
| `query` | Query planning and answer generation |
| `vlm` | Visual analysis for images, tables, equations, and drawings |

Unset roles fall back to `llm.default`. The checked-in defaults use
role-specific `extract` and `keyword` models and let `query` / `vlm` fall back
to the default multimodal LLM.

Concurrency defaults:

| Setting | Default | Scope |
|---|---:|---|
| `max_async` | `8` | Shared LLM role queues and DlightRAG planner/answer calls |
| `embedding_func_max_async` | `16` | Embedding queue concurrency |
| `max_parallel_insert` | `3` | LightRAG staged insert workers |
| `max_parallel_parse_native` | `5` | Native parser workers |
| `max_parallel_parse_mineru` | `2` | MinerU parser workers |
| `max_parallel_analyze` | `5` | VLM analyze workers |

### Embeddings

The embedding model must be multimodal because text chunks, query text, query
images, source images, and parser-extracted images share one vector space.

Default:

```yaml
embedding:
  provider: voyage
  model: voyage-multimodal-3.5
  dim: 1024
  asymmetric: auto
```

Supported providers include `voyage`, `dashscope_qwen`,
`qwen_openai_compatible`, `gemini`, and `jina`. Changing `embedding.dim` or the
embedding model after indexing requires clearing the workspace and rebuilding
vector indexes.

### Metadata

Metadata is explicit-schema first:

- Declare filterable custom fields once in `metadata.fields`.
- REST, MCP, and SDK ingest calls pass metadata values, not schema
  declarations.
- Declared fields are normalized and promoted to filterable columns.
- Undeclared metadata can be stored as JSONB enrichment when
  `allow_ad_hoc_json: true`, but it is not filterable.
- Explicit API/user filters are strict and never fall back to global retrieval.
- LLM-inferred filters can fall back to unfiltered retrieval when they
  over-infer and match no candidates.

Example:

```yaml
metadata:
  allow_ad_hoc_json: true
  default_ingest_policy: validate
  fields:
    department:
      type: string
      filter_ops: ["exact"]
```

See [docs/response-schema.md](docs/response-schema.md#metadata-schema-and-policy).

### BM25

BM25 uses DlightRAG-managed pg_textsearch indexes over LightRAG's document
chunks. Ingest labels each chunk with `dlightrag_bm25_language`. Query-time
language detection selects one matching partial index; unsupported or
ambiguous queries use the `simple` fallback.

Checked-in profiles:

| Language | Text config |
|---|---|
| Chinese | `public.jiebacfg` from pg_jieba |
| English | `english` |
| German | `german` |
| Swedish | `swedish` |
| Spanish | `spanish` |
| French | `french` |
| Fallback | `simple` |

Each non-fallback BM25 profile maps to exactly one language. The fallback
profile must not declare languages.

### Storage

DlightRAG's supported core storage stack is PostgreSQL 18:

| Component | Backend |
|---|---|
| Vector store | `PGVectorStorage` with pgvector |
| Graph store | `PGGraphStorage` with Apache AGE |
| KV store | `PGKVStorage` |
| Document status | `PGDocStatusStorage` |
| BM25 | pg_textsearch |

Default vector indexing uses HNSW over `VECTOR(dim)`. `HNSW_HALFVEC`,
`pg_hnsw_*`, pool sizing, statement-cache, retry, and session-GUC overrides
are advanced deployment settings, documented in [docs/PG.md](docs/PG.md) and
[docs/config-reference.md](docs/config-reference.md#postgresql). Server-level
memory, WAL, preload library, and shared-memory settings belong to the
PostgreSQL deployment or `docker-compose.yml`.

### Auth

Modes:

| Mode | Use case |
|---|---|
| `none` | Local loopback development only |
| `simple` | Shared bearer token |
| `jwt` | User-scoped deployments with signed tokens |

Simple token:

```bash
openssl rand -base64 32
DLIGHTRAG_AUTH_MODE=simple
DLIGHTRAG_API_AUTH_TOKEN=<generated>
```

JWT:

```bash
openssl rand -base64 64
DLIGHTRAG_AUTH_MODE=jwt
DLIGHTRAG_JWT_SECRET=<generated>
DLIGHTRAG_JWT_ALGORITHM=HS256
```

Clients send `Authorization: Bearer <token>`. JWT tokens must include `sub`,
which becomes the authenticated `user_id`.

### Rerank And Answer Breadth

Reranking happens before answer-context packing. Root config exposes the
high-signal breadth controls:

| Setting | Default |
|---|---:|
| `answer.candidate_top_k` | `60` |
| `answer.context_top_k` | `30` |
| `answer.max_images` | `6` |

Image compression budgets are advanced transport limits; see
[docs/config-reference.md](docs/config-reference.md#image-budgets). Use
`/retrieve` for the broader pre-answer retrieval set. `/answer` returns
contexts and sources aligned with what the answer model saw.

### Citations

Citation validation is always part of answer finalization. Web source-panel
semantic highlights are enabled by default and run after answer generation with
the keyword LLM role, bounded by timeout/concurrency settings. REST and MCP
answer payloads are not affected by Web highlight enrichment.

### Langfuse

Langfuse tracing is optional. If both keys are absent, tracing is a no-op.

```bash
DLIGHTRAG_LANGFUSE_PUBLIC_KEY=pk-...
DLIGHTRAG_LANGFUSE_SECRET_KEY=sk-...
```

Set non-secret behavior in `config.yaml`: `langfuse_host`,
`langfuse_environment`, `langfuse_release`, `langfuse_sample_rate`,
`langfuse_timeout`, `langfuse_flush_at`, `langfuse_flush_interval`, and
`langfuse_export_external_spans`.

Local self-hosted helper:

```bash
make langfuse-up
make langfuse-health
make langfuse-logs
make langfuse-down
```

## API Surface

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest` | Ingest local, Azure Blob, or AWS S3 content |
| `POST` | `/ingest/blob` | Upload one file via multipart form and ingest it |
| `POST` | `/retrieve` | Return contexts and sources without answer generation |
| `POST` | `/answer` | Return or stream an LLM answer with contexts and sources |
| `GET` | `/files` | List ingested documents |
| `DELETE` | `/files` | Delete documents |
| `GET` | `/files/failed` | List documents stuck in `DocStatus.FAILED` |
| `POST` | `/files/retry` | Retry failed documents |
| `GET` | `/api/files/{file_path}` | Serve local source files or redirect Azure Blob sources |
| `GET` | `/metadata/{doc_id}` | Read document metadata |
| `POST` | `/metadata/{doc_id}` | Merge or replace document metadata |
| `POST` | `/metadata/search` | Find document IDs matching metadata filters |
| `GET` | `/images/{workspace}/{chunk_id}` | Serve full or thumbnail visual assets |
| `POST` | `/reset` | Reset workspace storage |
| `GET` | `/workspaces` | List registered workspaces |
| `POST` | `/workspaces` | Create an empty workspace |
| `DELETE` | `/workspaces/{workspace}` | Delete/reset one workspace |
| `GET` | `/health` | Health and storage status |

Workspace-scoped read/write endpoints accept optional `workspace`.
Workspace lifecycle endpoints name the workspace explicitly. Query endpoints
accept `workspaces` for federated search. `/answer` streams by default; pass
`stream: false` for a single JSON response.

## Development

```bash
git clone https://github.com/hanlianlu/dlightrag.git
cd dlightrag
cp .env.example .env
uv sync
```

Local CI:

```bash
make ci
make ci-full
make ci-e2e
```

Opt-in PG18 E2E smoke:

```bash
docker compose -f docker-compose.yml -f docker-compose.e2e.yml up -d --build postgres
DLIGHTRAG_RUN_E2E_PG18=1 \
DLIGHTRAG_E2E_POSTGRES_PORT=55432 \
uv run pytest tests/e2e -m e2e_pg18 -q
```

The E2E smoke expects PostgreSQL 18 with pgvector, Apache AGE,
pg_textsearch, and pg_jieba installed. It uses deterministic fake model
functions by default.

## References

- [docs/response-schema.md](docs/response-schema.md) - REST, MCP, and SDK payloads.
- [docs/retrieval_answer_mechanism.md](docs/retrieval_answer_mechanism.md) - retrieval, filters, fusion, and answer generation.
- [docs/module-layers.md](docs/module-layers.md) - code organization and import boundaries.
- [docs/PG.md](docs/PG.md) - PostgreSQL requirements and tuning notes.
- [docs/config-reference.md](docs/config-reference.md) - advanced config overrides not shown in root config.
- [LightRAG API Server docs](https://github.com/HKUDS/LightRAG/blob/main/docs/LightRAG-API-Server.md) - upstream parser routing and MinerU official API contract.
- [MinerU Docker deployment docs](https://opendatalab.github.io/MinerU/quick_start/docker_deployment/) - Linux/WSL2 Docker support and macOS warning.

## License

Apache License 2.0. See [LICENSE](LICENSE).

Built by HanlianLyu. Contributions welcome.
