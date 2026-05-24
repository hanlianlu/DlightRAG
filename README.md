# DlightRAG

[![PyPI](https://img.shields.io/pypi/v/dlightrag)](https://pypi.org/project/dlightrag/)
[![CI](https://github.com/hanlianlu/dlightrag/actions/workflows/ci.yml/badge.svg)](https://github.com/hanlianlu/dlightrag/actions/workflows/ci.yml)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/hanlianlu/DlightRAG)

Multimodal RAG with knowledge graph and contextual intelligence. Understands what your documents say, how concepts connect, and what the pages look like. Production-ready.

Most RAG systems treat documents as hierarchical text and search by similarity agentically — visual context is lost, entity relationships are missed, context filtering is limited. DlightRAG combines knowledge graph understanding with dynamic multimodal retrieval to close these gaps.

From text-heavy reports to chart-filled presentations — it adapts to your documents without information compromise. Inquiry answers come with inline citations grounded in actual document content. Flexibly ship it as a ready-to-run service, integrate into your backend, or expose as a tool for AI agents.

## Features

- **Single LightRAG-main multimodal path** — documents go through LightRAG parser sidecars; native and extracted images get direct multimodal embeddings
- **Knowledge graph + vector + BM25 retrieval** — LightRAG `mix` mode, PostgreSQL vector in-filtering, pg_textsearch BM25, and RRF fusion
- **Multimodal ingestion** — PDF, images, Office documents, Azure Blob Storage, and AWS S3 via one ingestion engine
- **Broad LLM support** — Native SDKs for OpenAI, Anthropic, Gemini + any OpenAI-compatible endpoint
- **Cross-workspace federation** — Query across embedding-compatible workspaces with well managed merging
- **Citation and highlighting** — Inline citations with source, page, and highlighting attribution
- **Observability** — Hierarchical Langfuse traces for pipelines, retrieval, reranking, generations, and embeddings; no-op when disabled
- **Four interfaces** — Web UI, REST API, Python SDK, and MCP server


## Architecture

<p align="center">
  <img src="docs/architecture.png" alt="DlightRAG Architecture" width="1080" />
</p>

<sub>Source: <a href="docs/architecture.drawio">docs/architecture.drawio</a> (runtime data flow) &middot; <a href="docs/module-layers.md">docs/module-layers.md</a> (code-organisation layers)</sub>


## Quick Start

> **Defaults shipped in `config.yaml`:** `unified` RAG mode + `google/gemini-2.5-flash-lite` chat (via an OpenAI-compatible gateway) + `voyage-multimodal-3.5` embedding (Voyage). Swap providers or models by editing `config.yaml` — see [Configuration](#configuration).

### Web UI
##### Click the image to watch demo (YouTube)
<a href="https://www.youtube.com/watch?v=F1RXUW4Xfuc">
  <img src="docs/DlightRAG_GUI_img.png" alt="Watch Demo on YouTube" width="1440" />
</a>

If you already have the REST API running (via Docker or `dlightrag-api`), the Web UI is available at:

```
http://localhost:8100/web/
```

Without Docker:

```bash
uv add dlightrag        # or: pip install dlightrag
cp .env.example .env    # set API keys in .env
dlightrag-api
```

### Docker (Self-Hosted)

```bash
git clone https://github.com/hanlianlu/dlightrag.git && cd dlightrag
cp .env.example .env    # set API keys in .env; edit config.yaml for models/providers
docker compose up
```

Includes PostgreSQL 18 (pgvector + Apache AGE + pg_textsearch), REST API (`:8100`), and MCP server (`:8101`, host-mapped to loopback by default — see [Deployment & auth](#deployment--auth) before exposing externally).

> **Local models (Ollama, Xinference, etc.):** use `host.docker.internal` instead of `localhost` in `base_url` settings.

```bash
curl http://localhost:8100/health

curl -X POST http://localhost:8100/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type": "local", "path": "/app/dlightrag_storage/docs"}'

curl -X POST http://localhost:8100/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?"}'

curl -X POST http://localhost:8100/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?", "stream": true}'
```

### Python SDK

```bash
uv add dlightrag        # or: pip install dlightrag
cp .env.example .env    # set API keys in .env
```

```python
import asyncio
from dotenv import load_dotenv
from dlightrag import RAGServiceManager, DlightragConfig

load_dotenv()  # load .env

async def main():
    config = DlightragConfig()
    # Async factory: parallel-warms every workspace and initializes Langfuse tracing.
    # Bare `RAGServiceManager(config)` also works but defers warmup until first call.
    manager = await RAGServiceManager.create(config)
    try:
        await manager.aingest(workspace="default", source_type="local", path="./docs")

        result = await manager.aretrieve("What are the key findings?")
        print(result.contexts)

        result = await manager.aanswer("What are the key findings?")
        print(result.answer)
    finally:
        await manager.close()

asyncio.run(main())
```

> Requires PostgreSQL 18 with pgvector, Apache AGE, and pg_textsearch. DlightRAG's core path is PostgreSQL-only.

### MCP Server (for AI Agents)

Two transports — pick by how the agent runs:

**stdio** — agent spawns dlightrag-mcp as a subprocess (Claude Desktop, Cursor):

```bash
uv tool install dlightrag
cp .env.example .env        # set API keys in .env
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

**streamable-http** — agent connects over HTTP (remote / multi-client):

```bash
DLIGHTRAG_MCP_TRANSPORT=streamable-http \
DLIGHTRAG_MCP_HOST=127.0.0.1 \
dlightrag-mcp
# agent posts to http://127.0.0.1:8101/mcp
```

Tools: `retrieve`, `answer`, `ingest`, `list_files`, `delete_files`, `list_workspaces` — all workspace-isolated.


## Deployment & auth

Pick the row matching your use case:

| Scenario | Transport | Bind | Bearer token |
|---|---|---|---|
| Local agent (Claude Desktop / Cursor) | `stdio` | n/a | not needed |
| Self-hosted, single-machine | `streamable-http` | `127.0.0.1` (default) | not needed |
| `docker compose up` (default) | `streamable-http` | container `0.0.0.0`, host port `127.0.0.1:8101` | not needed |
| LAN / team access | `streamable-http` | `0.0.0.0` | **required** |
| Production / public network | `streamable-http` behind reverse proxy + TLS | proxy → `127.0.0.1` | **required** |

**Rule of thumb:** if anyone other than you can reach port `8100` (REST) or `8101` (MCP), set a token.

```bash
openssl rand -base64 32                                     # generate
echo "DLIGHTRAG_AUTH_MODE=simple" >> .env                   # enable bearer auth
echo "DLIGHTRAG_API_AUTH_TOKEN=<generated>" >> .env         # set
# clients send: Authorization: Bearer <generated>
```

The same token guards both REST and MCP. The MCP server logs a multi-line warning at startup if it binds non-loopback without a token configured.


## API & Internals

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest` | Ingest from local, Azure Blob, or AWS S3 |
| `POST` | `/retrieve` | Contexts + sources, no LLM call (response still ships `answer: null` for shape parity with `/answer`) |
| `POST` | `/answer` | LLM answer + contexts + sources (`stream` is explicit; `true` for SSE, `false` for JSON) |
| `GET` | `/files` | List ingested documents |
| `DELETE` | `/files` | Delete documents |
| `GET` | `/files/failed` | List documents stuck in `DocStatus.FAILED` |
| `POST` | `/files/retry` | Re-ingest all FAILED documents (replace=True, source-aware) |
| `GET` | `/api/files/{path}` | Serve/download a file (local: stream, Azure: 302 SAS redirect) |
| `GET` | `/metadata/{doc_id}` | Read a document's metadata JSONB |
| `POST` | `/metadata/{doc_id}` | Merge custom keys into a document's metadata JSONB |
| `POST` | `/metadata/search` | Find document IDs matching a key/value filter dict |
| `POST` | `/reset` | Reset workspace(s) — drop storage, clear indexes |
| `GET` | `/workspaces` | List available workspaces |
| `GET` | `/health` | Health check with storage status |

All write endpoints accept optional `workspace`; read endpoints accept `workspaces` list for cross-workspace federated search. See [Deployment & auth](#deployment--auth) for token setup.

- **Request/response schema** — [`docs/response-schema.md`](docs/response-schema.md) for ingestion parameters, retrieval contexts, sources, media, SSE streaming, citations, and multimodal queries.
- **Retrieval & answer pipeline** — [`docs/retrieval_answer_mechanism.md`](docs/retrieval_answer_mechanism.md) for LightRAG mix retrieval, direct image search, metadata in-filtering, BM25 fusion, reranking, and answer generation.
- **Database roles** — [`docs/database-role-architecture.md`](docs/database-role-architecture.md) for primary/replica process roles and read-only query workers.


## Configuration

Configuration uses a hybrid system — structured app settings in [`config.yaml`](config.yaml), secrets and deployment in `.env`.

**Priority:** constructor args > env vars > `.env` > `config.yaml` > defaults

See [`config.yaml`](config.yaml) for all application settings and [`.env.example`](.env.example) for secrets/deployment reference.

> **Env var naming:** all variables use the `DLIGHTRAG_` prefix. Single underscore (`_`) is part of the field name (e.g. `DLIGHTRAG_POSTGRES_HOST` → `postgres_host`). Double underscore (`__`) means nested object (e.g. `DLIGHTRAG_LLM__DEFAULT__MODEL` → `llm.default.model`). See `.env.example` for details.

### Unified LightRAG-Main Path

DlightRAG now has one ingestion and retrieval path. Documents are enqueued
through LightRAG main's parser routing; native images and parser-extracted
image sidecars are additionally embedded directly with the configured
multimodal embedding model.

The path is:

1. LightRAG parser sidecars handle document text, tables, equations, and
   document-derived images.
2. LightRAG owns chunking, entity extraction, relationship extraction, graph
   construction, and `mix` retrieval.
3. DlightRAG records metadata, document artifacts, and chunk provenance in
   PostgreSQL side tables.
4. DlightRAG adds direct image vector search, metadata in-filtering, BM25, RRF
   fusion, reranking, citations, and answer generation.

`default_mode` remains `mix`. It is an invariant for LightRAG retrieval, not a
public mode switch.

### Embeddings

The embedding model must be multimodal. Text chunks, native images, and
parser-extracted images must share one vector space.

| Provider | Example model | Notes |
|---|---|---|
| `voyage` | `voyage-multimodal-3.5` | Default provider; recommended dimension `1024` |
| `dashscope_qwen` | `qwen3-vl-embedding-2b` | DashScope multimodal payloads |
| `qwen_openai_compatible` | `qwen3-vl-embedding-2b` via LM Studio/vLLM | OpenAI-compatible local endpoint |
| `gemini` | `gemini-embedding-2` | Uses symmetric fallback when task routing is unavailable |
| `jina` | `jina-embeddings-v4` | Uses task routing when supported |

`embedding.asymmetric: auto` enables provider task routing when the adapter
supports it. Otherwise DlightRAG relies on LightRAG's symmetric fallback.
Changing `embedding.dim` after data is indexed requires clearing the workspace
and rebuilding vector indexes.

### Providers

Three native SDKs — choose per model block in `config.yaml`:

| Provider | SDK | Use for |
|----------|-----|---------|
| `openai` (default) | AsyncOpenAI | OpenAI, Azure OpenAI, Qwen/DashScope, MiniMax, Ollama, Xinference, any OpenAI-compatible endpoint |
| `anthropic` | Anthropic SDK | Anthropic Claude models |
| `gemini` | Google GenAI SDK | Google Gemini models |

All three SDKs ship in the base install; no extras to install.

```yaml
# config.yaml — OpenAI-compatible (Ollama example)
llm:
  default:
    provider: openai
    model: gemma4:26b-a4b-it-q8_0
    base_url: http://localhost:11434/v1

# config.yaml — Anthropic (native SDK)
llm:
  default:
    provider: anthropic
    model: claude-sonnet-4-20250514

# config.yaml — Google Gemini (native SDK)
llm:
  default:
    provider: gemini
    model: gemini-2.5-pro
```

API keys go in `.env`:
```bash
DLIGHTRAG_LLM__DEFAULT__API_KEY=sk-...
DLIGHTRAG_EMBEDDING__API_KEY=sk-...
```

#### Per-role LLM Overrides

LightRAG role overrides use LightRAG's role names: `extract`, `keyword`, and
`query`. DlightRAG also wires `vlm` for image/table/equation analysis. Each
unset role falls back to `llm.default`.

| Role | What it drives | Recommended model class |
|---|---|---|
| `extract` | KG entity & relationship extraction during ingest | Reliable extraction model |
| `keyword` | LightRAG retrieval keyword extraction | Cheap and low-latency model |
| `query` | Answer generation and query planning | Balanced to heavy model |
| `vlm` | Visual analysis for image/table/equation sidecars | Vision-capable model |

```yaml
# config.yaml
llm:
  default:
    provider: openai
    model: google/gemini-2.5-flash-lite
    base_url: https://openrouter.ai/api/v1
  roles:
    extract:
      provider: anthropic
      model: claude-sonnet-4-20250514
    keyword:
      provider: openai
      model: gpt-4.1-mini
    query:
      provider: openai
      model: gpt-4.1
    vlm:
      provider: gemini
      model: gemini-2.5-flash
```

### Metadata Filtering

Metadata is explicit-schema first. Declared fields are normalized and can be
used for exact filtering; undeclared enrichment is stored in JSONB but is not
filterable by default.

LLM intent-aware filtering is source-aware:

- Explicit user/API filters are strict. If the filter resolves to an empty
  candidate set, retrieval returns no matches instead of falling back to global
  search.
- LLM-inferred filters must carry evidence spans that appear in the query. If
  an inferred filter resolves to an empty candidate set, retrieval falls back
  to unfiltered search because the planner may have over-inferred intent.
- Non-empty inferred candidate sets still constrain semantic and BM25 retrieval.

### Storage Backends

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
workspace's embedding dimension and rebuilding indexes.

For production concurrency, run separate process roles:

```text
Ingest/admin workers -> primary PostgreSQL
Query workers        -> hot standby / read replica PostgreSQL
```

Set `runtime_role: query` on read-only API workers and configure
`postgres_replica_*`. Query role attaches to existing LightRAG/DlightRAG schema
without DDL and rejects ingest, delete, metadata update, retry, and reset
operations. Ingest/admin roles remain primary-only for migrations and writes.

### Workspaces

Each workspace is isolated by PostgreSQL workspace columns and LightRAG graph
namespace. Workspaces must use the same embedding model and dimension if they
are queried together with federation.

### Reranking

Set in `config.yaml` under the `rerank:` block:

| Setting | Default | Description |
|---------|---------|-------------|
| `rerank.strategy` | `chat_llm_reranker` | `chat_llm_reranker`, `jina_reranker`, `aliyun_reranker`, `azure_cohere`, `local_reranker` |
| `rerank.model` | (strategy default) | Model name sent to the endpoint |
| `rerank.base_url` | (provider default) | Custom endpoint URL for any compatible service |
| `rerank.api_key` | — | Set in `.env` as `DLIGHTRAG_RERANK__API_KEY` |

| Strategy | Default model | API key |
|---------|---------------|---------|
| `chat_llm_reranker` | uses `rerank.provider/model` if set, otherwise the default LLM; selected model must support images when reranking visual chunks | reuses default LLM key unless `DLIGHTRAG_RERANK__API_KEY` is set |
| `jina_reranker` | `jina-reranker-m0` | `DLIGHTRAG_RERANK__API_KEY` |
| `aliyun_reranker` | `gte-rerank` | `DLIGHTRAG_RERANK__API_KEY` |
| `azure_cohere` | `cohere-rerank-v3.5` | `DLIGHTRAG_RERANK__API_KEY` |
| `local_reranker` | (set `rerank.model` + `rerank.base_url`) | (none — local endpoint) |

For self-hosted rerankers (Xinference, vLLM, TEI etc.), use `local_reranker` with `rerank.base_url` + `rerank.model`. For any other OpenAI-compatible `/rerank` endpoint, point `rerank.base_url` at it.

### Observability (Langfuse)

DlightRAG includes native tracing using [Langfuse](https://langfuse.com/). When configured, it records hierarchical observations for service pipelines, retrieval, reranking, LLM generations, and embedding calls.

Langfuse is optional. If `DLIGHTRAG_LANGFUSE_PUBLIC_KEY` and
`DLIGHTRAG_LANGFUSE_SECRET_KEY` are both omitted, tracing is disabled and the
observability layer is a pure no-op. If you want tracing, set both keys from the
target Langfuse project. This is true for both Langfuse Cloud and local
self-hosted Langfuse; only the key source and host URL differ.

`DLIGHTRAG_LANGFUSE_HOST` is the Langfuse API/UI base URL that the DlightRAG
process can reach.

**Langfuse Cloud**

To enable tracing with Langfuse Cloud, create a project in your chosen Cloud
region, copy that project's keys, then set:

```bash
DLIGHTRAG_LANGFUSE_PUBLIC_KEY=pk-...
DLIGHTRAG_LANGFUSE_SECRET_KEY=sk-...
DLIGHTRAG_LANGFUSE_HOST=https://cloud.langfuse.com        # EU Cloud, also DlightRAG's default
# DLIGHTRAG_LANGFUSE_HOST=https://us.cloud.langfuse.com   # US Cloud
# DLIGHTRAG_LANGFUSE_HOST=https://jp.cloud.langfuse.com   # Japan Cloud
# DLIGHTRAG_LANGFUSE_HOST=https://hipaa.cloud.langfuse.com # HIPAA Cloud
DLIGHTRAG_LANGFUSE_EXPORT_EXTERNAL_SPANS=false
```

**Local self-host**

DlightRAG does not embed the Langfuse web/worker/database services, but it does
ship the local setup helper around the
[official Langfuse v3 Docker Compose stack](https://langfuse.com/self-hosting/local).
Requirements: Docker with Docker Compose, and internet access the first time the
stack is downloaded.

From the DlightRAG repo:

```bash
cp .env.example .env
# Edit .env for DlightRAG's normal model/storage settings.
# Leave local Langfuse keys blank unless you intentionally want fixed keys.
make langfuse-up
make langfuse-health
```

For a first local setup, run only `make langfuse-up` yourself. It includes the
download/preparation and key bootstrap steps. `make langfuse-health` is the
verification step after the containers start.

`make langfuse-up` is the full local Langfuse setup path:

1. `make langfuse-stack` downloads the official Langfuse `docker-compose.yml`
   into `../langfuse-local` if it is missing, then patches it for local ports.
2. `make langfuse-bootstrap` writes matching headless project keys into both env
   files.
3. Docker Compose starts the Langfuse web, worker, Postgres, ClickHouse, Redis,
   and MinIO containers.

| File | Written keys |
|---|---|
| `../langfuse-local/.env` | `LANGFUSE_INIT_PROJECT_PUBLIC_KEY`, `LANGFUSE_INIT_PROJECT_SECRET_KEY` |
| `.env` | `DLIGHTRAG_LANGFUSE_PUBLIC_KEY`, `DLIGHTRAG_LANGFUSE_SECRET_KEY`, `DLIGHTRAG_LANGFUSE_HOST` |

Default local endpoints:

| Service | URL |
|---|---|
| Langfuse UI/API | `http://localhost:3300` |
| DlightRAG Langfuse host | `DLIGHTRAG_LANGFUSE_HOST=http://localhost:3300` |

The local compose stack is kept outside this repo at `../langfuse-local` so it
can hold Langfuse data and secrets without committing them to DlightRAG. The
helper binds host ports to loopback and avoids common development ports such as
`3000`, `5432`, `6379`, `8123`, and `9000`.

| Command | When to run it | What it does |
|---|---|---|
| `make langfuse-up` | Normal setup/start command | Runs `langfuse-stack`, then `langfuse-bootstrap`, then starts Langfuse with Docker Compose |
| `make langfuse-health` | After `make langfuse-up` | Checks `http://localhost:3300/api/public/health` |
| `make langfuse-logs` | When startup/debugging needs logs | Tails the local Langfuse compose logs |
| `make langfuse-down` | When you want to stop Langfuse | Stops the local Langfuse compose stack |
| `make langfuse-stack` | Optional advanced/debug step | Downloads/patches `../langfuse-local/docker-compose.yml` without syncing keys or starting containers |
| `make langfuse-bootstrap` | Optional advanced/debug step | Syncs project keys into `../langfuse-local/.env` and DlightRAG `.env` without starting containers |

To log into the local Langfuse UI, read the bootstrap user from the local stack
env file:

```bash
grep '^LANGFUSE_INIT_USER_' ../langfuse-local/.env
```

Do not set `LANGFUSE_INIT_PROJECT_PUBLIC_KEY` or
`LANGFUSE_INIT_PROJECT_SECRET_KEY` for normal local use. `make langfuse-up`
creates or reuses a local pair and writes the matching DlightRAG values for you.

These two values are the API credentials that
[headless Langfuse initialization](https://langfuse.com/self-hosting/administration/headless-initialization)
seeds into the local project on startup. They are not copied from the UI before
first startup. If you override them, they are not arbitrary throwaway labels:
the public key and secret key must be the exact pair used by both Langfuse and
DlightRAG, and the secret key should be a strong random value. A mismatch causes
DlightRAG tracing requests to be rejected by Langfuse.

Only preselect fixed project keys when you need deterministic local credentials,
for example in repeatable local automation. Set them before first startup:

```bash
LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-my-local-project \
LANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-use-a-long-random-secret \
make langfuse-up
```

If you create or rotate Langfuse project keys after DlightRAG has already
started, restart the DlightRAG process so it reloads `.env`. With Docker Compose,
recreate the affected containers rather than only restarting existing ones:

```bash
docker compose up -d --force-recreate dlightrag-api dlightrag-mcp
```

If DlightRAG runs inside Docker while Langfuse is bound on the host, use a host
URL reachable from that container, for example `http://host.docker.internal:3300`
on Docker Desktop.

By default DlightRAG exports only observations created by its own wrappers. Set
`DLIGHTRAG_LANGFUSE_EXPORT_EXTERNAL_SPANS=true` only if you intentionally want
Langfuse to also ingest third-party OpenTelemetry GenAI/LLM spans.


## Development

```bash
git clone https://github.com/hanlianlu/dlightrag.git && cd dlightrag
cp .env.example .env && uv sync
docker compose up -d                # PostgreSQL + API + MCP
docker compose up postgres -d       # PostgreSQL only
```

```bash
uv run pytest tests/unit            # unit tests (no external services)
uv run pytest tests/integration     # integration tests (requires PostgreSQL)
uv run ruff check src/ tests/ scripts/ --fix && uv run ruff format src/ tests/ scripts/
```


## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

Built by HanlianLyu. Contributions welcome!
