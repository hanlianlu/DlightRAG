# DlightRAG

[![PyPI](https://img.shields.io/pypi/v/dlightrag)](https://pypi.org/project/dlightrag/)
[![CI](https://github.com/hanlianlu/dlightrag/actions/workflows/ci.yml/badge.svg)](https://github.com/hanlianlu/dlightrag/actions/workflows/ci.yml)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/hanlianlu/DlightRAG)

DlightRAG is a multimodal RAG service built on LightRAG main. It ingests
documents and images into isolated workspaces, lets LightRAG own parser
sidecars, document status, chunks, knowledge graph, and vectors, and adds
PostgreSQL metadata filtering, BM25, direct image retrieval, and cited answers.

Use it as a Web UI, REST API, Python SDK, or MCP server for AI agents.

Status: Python 3.12+. Storage: PostgreSQL 18 with pgvector, Apache AGE, and
pg_textsearch. License: Apache-2.0.

## Start Here

Most local work uses this shape:

```text
Browser / curl / Python / MCP client
  -> DlightRAG API + Web UI  (Docker or native, port 8100)
  -> PostgreSQL 18           (Docker, port 5432)
  -> LightRAG main           (parser sidecars, KG, vectors, doc status)
  -> DlightRAG stores        (metadata index, BM25, workspace/role metadata)
```

### Prerequisites

- Docker Desktop for the recommended local setup.
- Python 3.12+ and `uv` for native development or SDK use.
- Model credentials for the providers configured in [`config.yaml`](config.yaml).
- A multimodal embedding model. The checked-in default is
  `voyage-multimodal-3.5`.
- A local MinerU HTTP sidecar for PDF, PPTX, XLSX, and other non-DOCX
  document parsing.

If you do not use Docker, provide PostgreSQL 18 with `pgvector`, Apache AGE,
and `pg_textsearch` installed and preloaded as required by [`docs/PG.md`](docs/PG.md).

### Recommended Local Setup

1. Create `.env` and fill credentials only:

```bash
git clone https://github.com/hanlianlu/dlightrag.git
cd dlightrag
cp .env.example .env
```

At minimum, set the LLM and embedding API keys needed by `config.yaml`:

```bash
DLIGHTRAG_LLM__DEFAULT__API_KEY=...
DLIGHTRAG_EMBEDDING__API_KEY=...
DLIGHTRAG_LLM__ROLES__EXTRACT__API_KEY=...
DLIGHTRAG_LLM__ROLES__KEYWORD__API_KEY=...
```

Normal settings such as model names, parser routing, ports, logging,
PostgreSQL endpoints, and retrieval knobs live in `config.yaml`.

The default document route uses LightRAG native parsing for DOCX and MinerU
for other document types. `config.yaml` is local-first and points native
DlightRAG at `http://127.0.0.1:8210`. If ArtRAG already has a MinerU service
running on port `8210`, DlightRAG can reuse it.

2. Start a local MinerU sidecar for document parsing if one is not already running:

```bash
cp .env.mineru.example .env.mineru
make mineru-install
make mineru-api
```

The helper creates a dedicated `.venv-mineru` so MinerU's ML dependencies do
not enter DlightRAG's runtime environment. With no explicit
`MINERU_INSTALL_EXTRAS`, the installer uses `mineru[core,mlx]` on Apple
Silicon and `mineru[core]` elsewhere. `.env.mineru.example` defaults to
`core,mlx` for this local development setup. Keep DlightRAG runtime parser
settings in `config.yaml`. The Makefile targets are the stable public entry
points; the implementation lives under `scripts/mineru/`. On GPU servers,
prefer MinerU's official Docker Compose `api` or `router` profile; set
`parser_sidecars.mineru.local_endpoint` to that service, usually
`http://127.0.0.1:8000` or `http://127.0.0.1:8002`.

On macOS, you can keep the sidecar running through launchd:

```bash
make mineru-service-install
make mineru-service-status
make mineru-service-logs
```

3. Start the service stack:

```bash
docker compose up -d
docker compose ps
curl http://localhost:8100/health
```

`docker compose up -d` starts:

| Service | Purpose | Host Port |
|---|---|---|
| `dlightrag-api` | REST API + Web UI | `8100` |
| `dlightrag-mcp` | MCP streamable HTTP server | `127.0.0.1:8101` |
| `postgres` | PostgreSQL 18 + pgvector + AGE + pg_textsearch | `5432` |

When DlightRAG itself runs in Docker, compose maps
`MINERU_LOCAL_ENDPOINT` inside the app containers to
`http://host.docker.internal:8210` by default. Override
`MINERU_DOCKER_LOCAL_ENDPOINT` when the MinerU API is attached to a Docker
network, for example `http://mineru-api:8000`.

4. Open the Web UI:

```text
http://localhost:8100/web/
```

Use the Files panel to upload documents or images, then ask a question in the
chat composer. This is the easiest first ingest path for Docker because the
uploaded file is staged inside DlightRAG's managed runtime volume at
`working_dir/inputs/<workspace>/`, which is also the `INPUT_DIR` root used by
LightRAG parser workers.

### First API Calls

Local-source ingestion paths must be visible to the API process. For native
API runs, pass a normal host path. For Docker, either use the Web UI upload or
mount a host directory into the API container.

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

Full request and response details live in
[`docs/response-schema.md`](docs/response-schema.md).

## Architecture

<p align="center">
  <img src="docs/architecture.svg" alt="DlightRAG Architecture" width="1080" />
</p>

<sub>Source: <a href="docs/architecture.drawio">docs/architecture.drawio</a> (runtime flow) and <a href="docs/module-layers.md">docs/module-layers.md</a> (module layers).</sub>

### Mental Model

DlightRAG has one runtime RAG path:

```text
source file
  -> LightRAG parser sidecars
       text, tables, equations, document-derived images
  -> LightRAG ingest
       chunks, entities, relationships, graph, text vectors, doc status
  -> direct image embedding
       native images and parser-extracted image sidecars into LightRAG chunks/vectors
  -> DlightRAG metadata/BM25 layer
       declared metadata index, in-filter scope, pg_textsearch BM25
  -> retrieval
       LightRAG mix + direct image vectors + BM25 + RRF + rerank + citations
```

Core responsibilities:

| Layer | Responsibility |
|---|---|
| API, MCP, Web UI | Interface adapters, auth, uploads, streaming responses. |
| `RAGServiceManager` | Multi-workspace routing, federation, health, read-after-write barriers. |
| `RAGService` | Per-workspace lifecycle, ingest, retrieve, answer, reset. |
| LightRAG | Parser routing, sidecars, chunks, KG entities/relationships, vectors, doc status, `mix` retrieval. |
| DlightRAG stores | Declared metadata index, BM25 state, workspace/role metadata. |
| Model adapters | LLM, VLM, embedding, asymmetric task routing, reranking. |

Long-form pipeline notes are in
[`docs/retrieval_answer_mechanism.md`](docs/retrieval_answer_mechanism.md).

### Retrieval Shape

DlightRAG always uses LightRAG `mix` as the base retrieval mode. The DlightRAG
hybrid layer is separate from LightRAG's `hybrid` mode:

```text
query
  -> intent-aware query planning and metadata filtering
  -> LightRAG mix retrieval
  -> direct image retrieval when query images are present
  -> pg_textsearch BM25 over the same candidate scope
  -> RRF fusion
  -> rerank
  -> answer generation with citations
```

Metadata filters are in-filtered. Explicit API/user filters are strict; if
they match no candidates, retrieval returns no matches. LLM-inferred filters
can fall back to unfiltered retrieval when they over-infer and resolve to an
empty candidate set.

## Operating Modes

### Docker API + Web UI

Recommended for local service use:

```bash
docker compose up -d
docker compose logs -f dlightrag-api
```

The API and Web UI share port `8100`; the Web UI is at `/web/`. The MCP server
is mapped to `127.0.0.1:8101` by default.

For local model servers running on the host machine, use
`host.docker.internal` in `base_url` settings rather than `localhost`.

### Native API

Use this when DlightRAG runs outside Docker while PostgreSQL stays in Docker:

```bash
docker compose up -d postgres
uv sync
uv run dlightrag-api
```

Native runs can ingest host paths directly because the API process sees the
same filesystem as your shell.

### Python SDK

Install and run against your configured PostgreSQL and model providers:

```bash
uv add dlightrag        # or: pip install dlightrag
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

Use stdio when the agent starts DlightRAG as a subprocess:

```bash
uv tool install dlightrag
cp .env.example .env
```

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

### Primary / Replica Process Roles

For production concurrency, separate write and read processes:

```text
Ingest/admin workers -> primary PostgreSQL
Query workers        -> hot standby / read replica PostgreSQL
Primary              -> physical streaming replication -> replica
```

Set `runtime_role: query` on read-only query workers and configure
`postgres_replica_*`. Query role attaches to existing LightRAG and DlightRAG
schema without DDL and rejects ingest, delete, metadata update, retry, and
reset operations.

Set `read_after_write_mode: wait_for_replay` on ingest/admin workers when a
write acknowledgement must wait until the replica has replayed the current
primary WAL LSN. Default `eventual` favors throughput and allows short replica
lag.

Details: [`docs/database-role-architecture.md`](docs/database-role-architecture.md).

## Configuration

Configuration uses structured app settings in [`config.yaml`](config.yaml) and
secrets or deployment-only overrides in `.env`.

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

Parser sidecar defaults live in `config.yaml` under `parser_sidecars`.
DlightRAG bridges those typed settings into LightRAG's upstream env names at
startup. `.env.example` keeps only secrets such as provider keys,
PostgreSQL passwords, and `DLIGHTRAG_PARSER_SIDECARS__MINERU__API_TOKEN`.
Docker-only sidecar endpoint overrides use `MINERU_DOCKER_LOCAL_ENDPOINT`.

### MinerU Local Sidecar

DlightRAG delegates MinerU parsing to LightRAG main. It does not ship a
separate MinerU client or parser path. The local contract is the MinerU HTTP
API used by LightRAG:

```text
POST /tasks
GET  /tasks/{task_id}
GET  /tasks/{task_id}/result
GET  /health
```

Recommended defaults in `config.yaml`:

| Setting | Default | Why |
|---|---|---|
| `parser_sidecars.mineru.api_mode` | `local` | Keeps document parsing local by default. |
| `parser_sidecars.mineru.local_endpoint` | `http://127.0.0.1:8210` | Reuses ArtRAG or DlightRAG's local `make mineru-api` sidecar. |
| `parser_sidecars.mineru.local_backend` | `hybrid-auto-engine` | Uses MinerU's local hybrid parser when the service supports it. |
| `parser_sidecars.mineru.local_parse_method` | `auto` | Lets MinerU choose text-layer extraction vs OCR per page. |
| `parser_sidecars.mineru.enable_table` / `enable_formula` | `true` | Preserves table and equation extraction for LightRAG ingest. |

DlightRAG applies a narrow LightRAG parser hygiene patch at startup when the
current upstream MinerU IR builder still indexes page furniture. The patch
drops page furniture families such as headers, footers, page numbers, margin
notes, page footnotes, and discarded blocks before LightRAG builds chunks,
while preserving semantic and multimodal blocks such as text, lists, code,
tables, equations, images, and charts.

The official MinerU API remains available by explicitly changing
`parser_sidecars.mineru.api_mode: official` in `config.yaml` and setting
`DLIGHTRAG_PARSER_SIDECARS__MINERU__API_TOKEN` in `.env`; it is not the
checked-in default.

Local MinerU service installation uses `.env.mineru`, not `.env`:

```bash
MINERU_INSTALL_EXTRAS=core,mlx
MINERU_API_HOST=127.0.0.1
MINERU_API_PORT=8210
```

Those keys affect only `make mineru-install`, `make mineru-api`, and the macOS
LaunchAgent helper. They do not change DlightRAG parser behavior; align
`parser_sidecars.mineru.local_endpoint` in `config.yaml` if you change the
host or port. If you update from an older local checkout that already installed
the LaunchAgent, run `make mineru-service-install` once so launchd points at
the current `scripts/mineru/api.sh` helper.

### Model Providers

LLM provider blocks support:

| Provider | SDK | Typical use |
|---|---|---|
| `openai` | AsyncOpenAI | OpenAI, Azure OpenAI, OpenRouter, Ollama, Xinference, local OpenAI-compatible endpoints. |
| `anthropic` | Anthropic SDK | Claude models. |
| `gemini` | Google GenAI SDK | Gemini models. |

LightRAG-aligned role overrides live under `llm.roles`:

| Role | What it drives |
|---|---|
| `extract` | KG entity and relationship extraction during ingest. |
| `keyword` | LightRAG retrieval keyword extraction. |
| `query` | Query planning and answer generation. |
| `vlm` | Visual analysis for image, table, equation, and drawing sidecars. |

Unset roles fall back to `llm.default`.

The checked-in defaults route `extract` and `keyword` through DeepSeek's
OpenAI-compatible Chat Completions endpoint using `deepseek-v4-flash` with
thinking disabled through `model_kwargs.thinking.type: disabled`. `query` and
`vlm` continue to fall back to the multimodal default LLM unless explicitly
overridden.

### Embeddings

The embedding model must be multimodal. Text chunks, native images, and
parser-extracted image sidecars must share one vector space.

| Provider | Example model | Notes |
|---|---|---|
| `voyage` | `voyage-multimodal-3.5` | Default provider; recommended dimension `1024`. |
| `dashscope_qwen` | `qwen3-vl-embedding-2b` | DashScope multimodal payloads. |
| `qwen_openai_compatible` | `qwen3-vl-embedding-2b` via LM Studio or vLLM | OpenAI-compatible local endpoint. |
| `gemini` | `gemini-embedding-2` | Uses symmetric fallback when task routing is unavailable. |
| `jina` | `jina-embeddings-v4` | Uses task routing when supported. |

`embedding.asymmetric: auto` enables provider task routing when available.
Otherwise DlightRAG relies on LightRAG's symmetric fallback. Changing
`embedding.dim` after indexing requires clearing the workspace and rebuilding
vector indexes.

### Rerank Image Budget

Reranking happens before answer-context packing. Multimodal rerankers therefore
use their own image payload budget instead of relying on `answer.*` limits:

| Setting | Default | Meaning |
|---|---:|---|
| `rerank.image_max_bytes` | `1500000` | Maximum compressed binary bytes per rerank image before base64 expansion. |
| `rerank.image_max_total_bytes` | `8000000` | Maximum compressed binary image bytes per rerank model request. |
| `rerank.image_max_px` | `1280` | Maximum long edge after rerank-stage bounding. |
| `rerank.image_min_px` | `768` | Long-edge floor before oversized rerank images are skipped. |
| `rerank.image_quality` | `86` | Initial JPEG quality for recompressed rerank images. |
| `rerank.image_min_quality` | `76` | JPEG quality floor before oversized rerank images are skipped. |

### Answer Image Budget

Answer generation uses one shared image budget for user `query_images` and
retrieved visual chunks. The defaults favor VLM understanding over aggressive
payload minimization:

| Setting | Default | Meaning |
|---|---:|---|
| `answer.candidate_top_k` | `60` | Retrieval candidates fetched for `/answer` before final prompt packing. |
| `answer.context_top_k` | `30` | Maximum chunks kept in the final answer prompt. |
| `answer.max_images` | `6` | Maximum images sent to the answer model. |
| `answer.image_max_bytes` | `3000000` | Maximum compressed binary bytes per image before base64 expansion. |
| `answer.image_max_total_bytes` | `24000000` | Maximum compressed binary bytes across all answer images. |
| `answer.image_max_px` | `1536` | Maximum long edge after bounding. |
| `answer.image_min_px` | `1024` | Long-edge floor before oversized images are skipped. |
| `answer.image_quality` | `88` | Initial JPEG quality for recompressed images. |
| `answer.image_min_quality` | `72` | JPEG quality floor before oversized images are skipped. |

Already-budgeted JPEG, PNG, and WebP payloads pass through unchanged. Images
that cannot fit without crossing the quality or size floor are skipped instead
of being sent as low-fidelity previews.

For `/answer`, DlightRAG retrieves `answer.candidate_top_k` candidates, reranks
them, then packs up to `answer.context_top_k` chunks into the prompt. In
LightRAG terms, this over-fetch maps to `QueryParam.chunk_top_k`; `top_k`
remains the independent KG entity/relationship breadth. User `query_images`
occupy the shared image budget first. Pure visual chunks whose image cannot be
sent are skipped and the packer backfills from later candidates; mixed
text+image chunks keep their text. `/answer` therefore returns contexts and
sources aligned with what the answer model saw. Use `/retrieve` when you need
the broader pre-answer retrieval set.

### Metadata Filtering

Metadata is explicit-schema first:

- Declared fields are normalized and filterable.
- Undeclared metadata can be stored as JSONB enrichment, but is not filterable
  by default.
- Explicit API/user filters are strict and never fall back to global retrieval.
- LLM-inferred filters include confidence/evidence for observability; empty
  inferred candidates fall back to unfiltered retrieval.
- Non-empty inferred candidates constrain both semantic and BM25 retrieval.

### Storage

DlightRAG's supported core storage stack is PostgreSQL 18 only:

| Component | Backend |
|---|---|
| Vector store | `PGVectorStorage` with pgvector |
| Graph store | `PGGraphStorage` with Apache AGE |
| KV store | `PGKVStorage` |
| Document status | `PGDocStatusStorage` |
| BM25 | pg_textsearch |

Default vector indexing uses `pg_vector_index_type: HNSW` with `VECTOR(dim)`.
`HNSW_HALFVEC` is explicit opt-in and should only be used after deciding the
workspace embedding dimension and rebuilding indexes.

`pg_hnsw_ef_search` and `postgres_session_settings` are applied to both the
LightRAG PostgreSQL pool and DlightRAG's domain-store pool. Server-level
PostgreSQL memory and WAL settings remain deployment configuration in
`docker-compose.yml` or your managed Postgres service.

The checked-in Docker stack also sets `shm_size: 8gb` for both PostgreSQL
primary and replica. That controls container `/dev/shm` for HNSW builds and is
separate from `shared_buffers`, `work_mem`, and `maintenance_work_mem`.

More PostgreSQL notes: [`docs/PG.md`](docs/PG.md).

### Workspaces

Each workspace is isolated by PostgreSQL workspace columns and LightRAG graph
namespace. Federated search can query multiple workspaces, but those workspaces
must use compatible embedding models and dimensions.

## Auth And Observability

### Auth

If anyone other than you can reach REST port `8100` or MCP port `8101`, enable
bearer auth:

```bash
openssl rand -base64 32
echo "DLIGHTRAG_API_AUTH_TOKEN=<generated>" >> .env
```

Set `auth_mode: simple` in `config.yaml`.

Clients send:

```text
Authorization: Bearer <generated>
```

The same token guards REST and MCP. JWT auth is also available through
`auth_mode: jwt`.

### Langfuse

Langfuse tracing is optional. If both keys are absent, tracing is a no-op.

```bash
DLIGHTRAG_LANGFUSE_PUBLIC_KEY=pk-...
DLIGHTRAG_LANGFUSE_SECRET_KEY=sk-...
```

Set `langfuse_host` and `langfuse_export_external_spans` in `config.yaml`
when the defaults are not right for the deployment.

For local self-hosted Langfuse:

```bash
make langfuse-up
make langfuse-health
make langfuse-logs
make langfuse-down
```

The helper keeps the Langfuse stack outside this repo at `../langfuse-local`
and writes matching local project keys into both env files. Its local-only
implementation is grouped under `scripts/langfuse/`; use the Makefile targets
instead of calling those helpers directly.

## API Surface

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest` | Ingest local, Azure Blob, or AWS S3 content. |
| `POST` | `/retrieve` | Return contexts and sources without answer generation. |
| `POST` | `/answer` | Return or stream an LLM answer with contexts and sources. |
| `GET` | `/files` | List ingested documents. |
| `DELETE` | `/files` | Delete documents. |
| `GET` | `/files/failed` | List documents stuck in `DocStatus.FAILED`. |
| `POST` | `/files/retry` | Retry failed documents. |
| `GET` | `/metadata/{doc_id}` | Read document metadata. |
| `POST` | `/metadata/{doc_id}` | Merge or replace document metadata. |
| `POST` | `/metadata/search` | Find document IDs matching metadata filters. |
| `GET` | `/images/{workspace}/{chunk_id}` | Serve full or thumbnail visual chunk assets for source panels. |
| `POST` | `/reset` | Reset workspace storage. |
| `GET` | `/workspaces` | List registered workspaces and registry records. |
| `POST` | `/workspaces` | Create an empty workspace. |
| `DELETE` | `/workspaces/{workspace}` | Delete/reset one workspace. |
| `GET` | `/health` | Health and storage status. |

All write endpoints accept optional `workspace`. Read endpoints accept
`workspaces` for federated search. `/answer` streams by default; pass
`stream: false` only when a single JSON response is required.

## Development

```bash
git clone https://github.com/hanlianlu/dlightrag.git
cd dlightrag
cp .env.example .env
uv sync
```

Common checks:

```bash
uv run pytest tests/unit
uv run pytest tests/integration
uv run ruff check src tests
uv run pyright
```

Full local test command:

```bash
uv run pytest -q
```

Opt-in PG18 + LightRAG smoke:

```bash
docker compose -f docker-compose.yml -f docker-compose.e2e.yml up -d --build postgres
DLIGHTRAG_RUN_E2E_PG18=1 \
DLIGHTRAG_E2E_POSTGRES_PORT=55432 \
uv run pytest tests/e2e -m e2e_pg18 -q
```

The E2E smoke expects PostgreSQL 18 with `pgvector`, Apache AGE, and
`pg_textsearch` installed, and `age,pg_textsearch` in
`shared_preload_libraries`. It uses deterministic fake model functions by
default, so it validates the storage/runtime path without external model API
calls.

## Useful References

- [docs/response-schema.md](docs/response-schema.md) - REST and SDK payloads.
- [docs/retrieval_answer_mechanism.md](docs/retrieval_answer_mechanism.md) - retrieval, filters, fusion, and answer generation.
- [docs/database-role-architecture.md](docs/database-role-architecture.md) - primary/replica process roles.
- [docs/module-layers.md](docs/module-layers.md) - code organization and import boundaries.
- [docs/PG.md](docs/PG.md) - PostgreSQL requirements and tuning notes.

## License

Apache License 2.0. See [LICENSE](LICENSE).

Built by HanlianLyu. Contributions welcome.
