# DlightRAG

[![PyPI](https://img.shields.io/pypi/v/dlightrag)](https://pypi.org/project/dlightrag/)
[![CI](https://github.com/hanlianlu/dlightrag/actions/workflows/ci.yml/badge.svg)](https://github.com/hanlianlu/dlightrag/actions/workflows/ci.yml)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/hanlianlu/DlightRAG)

Multimodal RAG with knowledge graph and contextual intelligence. Understands what your documents say, how concepts connect, and what the pages look like. Production-ready.

Most RAG systems treat documents as hierarchical text and search by similarity agentically — visual context is lost, entity relationships are missed, context filtering is limited. DlightRAG combines knowledge graph understanding with dynamic multimodal retrieval to close these gaps.

From text-heavy reports to chart-filled presentations — it adapts to your documents without information compromise. Inquiry answers come with inline citations grounded in actual document content. Flexibly ship it as a ready-to-run service, integrate into your backend, or expose as a tool for AI agents.

## Features

- **Dual multimodal RAG modes** — Caption mode (parse → caption → embed) for pipeline-based multimodal paradigm; Unified mode (render → multimodal embed) for modern multimodal paradigm
- **Knowledge graph + vector + visual retrieval** — Multi-strategy retrieval across knowledge graph and vector similarity [LightRAG](https://github.com/HKUDS/LightRAG), visual content, and dynamic metadata filters
- **Multimodal ingestion** — PDF, Images, Office Documents from local filesystem, Azure Blob Storage etc.
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

Includes PostgreSQL (pgvector + AGE), REST API (`:8100`), and MCP server (`:8101`, host-mapped to loopback by default — see [Deployment & auth](#deployment--auth) before exposing externally).

> **Local models (Ollama, Xinference, etc.):** use `host.docker.internal` instead of `localhost` in `base_url` settings.

```bash
curl http://localhost:8100/health

curl -X POST http://localhost:8100/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type": "local", "path": "/app/dlightrag_storage/sources"}'

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

> Requires PostgreSQL with pgvector + AGE, or JSON fallback for development (see [Configuration](#configuration)).

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
echo "DLIGHTRAG_API_AUTH_TOKEN=<generated>" >> .env         # set
# clients send: Authorization: Bearer <generated>
```

The same token guards both REST and MCP. The MCP server logs a multi-line warning at startup if it binds non-loopback without a token configured.


## API & Internals

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest` | Ingest from local, Azure Blob, or AWS S3 |
| `POST` | `/retrieve` | Contexts + sources, no LLM call (response still ships `answer: null` for shape parity with `/answer`) |
| `POST` | `/answer` | LLM answer + contexts + sources (`stream: true` for SSE) |
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
- **Retrieval & answer pipeline** — [`docs/retrieval_answer_mechanism.md`](docs/retrieval_answer_mechanism.md) for unified vs caption mode, visual resolution, reranking, Step 1+2 merge.


## Configuration

Configuration uses a hybrid system — structured app settings in [`config.yaml`](config.yaml), secrets and deployment in `.env`.

**Priority:** constructor args > env vars > `.env` > `config.yaml` > defaults

See [`config.yaml`](config.yaml) for all application settings and [`.env.example`](.env.example) for secrets/deployment reference.

> **Env var naming:** all variables use the `DLIGHTRAG_` prefix. Single underscore (`_`) is part of the field name (e.g. `DLIGHTRAG_POSTGRES_HOST` → `postgres_host`). Double underscore (`__`) means nested object (e.g. `DLIGHTRAG_CHAT__MODEL` → `chat.model`). See `.env.example` for details.

### RAG Mode

The first decision — determines your ingestion pipeline, model requirements, and retrieval behavior.

| Mode | Pipeline | Best for |
|------|----------|----------|
| `caption` | Document parsing → VLM captioning → text embedding → KG | Text-heavy documents, structured elements |
| `unified` (default)| Page rendering → multimodal embedding → VLM entity extraction → KG | Visually rich documents (charts, diagrams, complex layouts) |

**Caption mode parsers** (`parser` in config.yaml):

| Parser | Description |
|--------|-------------|
| `mineru` (default) | MinerU PDF parser — fast, good for text-heavy documents |
| `docling` | Docling parser — structure-aware parser with Docling JSON post-processing for headings and page metadata |
| `vlm` | VLM-based OCR — renders pages and uses the visual model to extract structured content; no external parser dependency |

Docling and VLM caption paths use Docling's HybridChunker for structure-aware
chunking. MinerU uses LightRAG's default text chunking after DlightRAG's
content policy filter removes parser noise.

**Caption parser options:**

| Setting | Applies to | Description |
|---------|------------|-------------|
| `mineru_backend` | MinerU | Parser backend (`pipeline`, `vlm-auto-engine`, `vlm-http-client`, `hybrid-auto-engine`, `hybrid-http-client`) |
| `mineru_timeout` | MinerU | Optional parser timeout in seconds; unset leaves MinerU unbounded |
| `mineru_vlm_url` | MinerU | Remote VLM server URL for `*-http-client` backends |
| `docling_table_mode` | Docling | `fast` or `accurate` TableFormer mode |
| `docling_tables` | Docling | Enable table structure recognition |
| `docling_allow_ocr` | Docling | Allow OCR for scanned content |
| `docling_artifacts_path` | Docling | Optional local Docling model artifacts path |

**Model usage by stage:**

Each stage resolves its model via the per-role overrides below; if a role is unset, it falls back to `chat`.

| Stage | Caption | Unified | Role override |
|-------|---------|---------|---------------|
| Image captioning | chat (RAGAnything vision function) | — | `chat` |
| Table / equation captioning | chat (RAGAnything vision function) | — | `chat` |
| VLM OCR / visual page description | `vlm` override → chat | `vlm` override → chat | `vlm` |
| Entity extraction | chat | chat | `extract` |
| Embedding | embedding model | embedding model (multimodal) | (separate `embedding` block) |
| Rerank (chat_llm_reranker) | rerank override → chat | rerank override → chat (vision-capable if page images are present) | `rerank.*` |
| Rerank (API strategy) | jina_reranker / aliyun_reranker / azure_cohere / local_reranker | jina_reranker / aliyun_reranker / azure_cohere / local_reranker | (separate `rerank` block) |
| Keyword extraction (per-query) | chat | chat | `keywords` |
| Answer generation | chat | chat (VLM, sees text excerpts + page images) | `query` |

> **Important:** Any role that receives `image_url` content must use a vision-capable model. With the default fallbacks, `chat` is used for RAGAnything captioning, VLM parser, unified visual extraction, multimodal query enhancement, answer generation, and `chat_llm_reranker` when no explicit `vlm`, `query`, or `rerank.*` override is set. The reranker does not consume the `vlm` role implicitly; reranker-specific model choices belong under `rerank.*`.

For unified mode, set `rag_mode: unified` in `config.yaml` and use multimodal models:

```yaml
# config.yaml
rag_mode: unified

chat:
  model: qwen3-vl-32b          # must support vision

embedding:
  model: Qwen3-VL-Embedding    # must be multimodal
  dim: 4096
```

> **Limitations:** A workspace is locked to one mode after first ingestion. Page images ~3-7 MB/page at 250 DPI.

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
chat:
  provider: openai
  model: qwen3:8b
  base_url: http://localhost:11434/v1

# config.yaml — Anthropic (native SDK)
chat:
  provider: anthropic
  model: claude-sonnet-4-20250514

# config.yaml — Google Gemini (native SDK)
chat:
  provider: gemini
  model: gemini-2.5-pro
```

API keys go in `.env`:
```bash
DLIGHTRAG_CHAT__API_KEY=sk-...
DLIGHTRAG_EMBEDDING__API_KEY=sk-...
```

#### Per-role LLM Overrides

LightRAG role overrides (`extract`, `keywords`, `query`) are built on
LightRAG 1.5.0's role registry. DlightRAG also has a local `vlm` model block
for its own visual paths. Each unset role falls back to `chat` — start with
`chat` only, split out a role later when cost or quality needs it.

| Role | What it drives | Recommended model class |
|---|---|---|
| `extract` | KG entity & relation extraction during ingest | Heavy reasoning (Claude Sonnet / GPT-5) |
| `keywords` | Per-query keyword extraction | Cheap & fast (Haiku / Gemini Flash Lite) |
| `query` | Answer generation + retrieval planning | Balanced–heavy (Claude Opus / GPT-5) |
| `vlm` | DlightRAG-local visual paths: VLM OCR, multimodal query enhancement, unified extractor | Vision-strong (GPT-5-vision / Gemini 2.5 Flash) |

```yaml
# config.yaml
extract:
  provider: anthropic
  model: claude-sonnet-4-20250514

# Cheap local fallback for high-volume keyword extraction:
keywords:
  provider: openai
  model: gemma4:9b-it-q4_K_M
  base_url: http://host.docker.internal:11434/v1
  api_key: ollama
```

### Storage Backends

Set in `config.yaml`:

| Setting | Default | Options |
|---------|---------|---------|
| `vector_storage` | `PGVectorStorage` | PGVectorStorage, MilvusVectorDBStorage, NanoVectorDBStorage, ... |
| `graph_storage` | `PGGraphStorage` | PGGraphStorage, Neo4JStorage, NetworkXStorage, ... |
| `kv_storage` | `PGKVStorage` | PGKVStorage, JsonKVStorage, RedisKVStorage, ... |
| `doc_status_storage` | `PGDocStatusStorage` | PGDocStatusStorage, JsonDocStatusStorage, ... |

> **Note:** When using PostgreSQL backends, LightRAG maps its internal namespace names to different table names (e.g. `text_chunks` → `LIGHTRAG_DOC_CHUNKS`, `full_docs` → `LIGHTRAG_DOC_FULL`). DlightRAG's unified mode adds a `visual_chunks` table via its own KV storage.

### Workspaces

Each workspace has its own knowledge graph, vector store, and document index. `workspace` in config.yaml (default: `default`) is automatically bridged to backend-specific env vars — no manual setup needed.

| Backend type | Isolation mechanism |
|---|---|
| PostgreSQL (PG*) | `workspace` column / graph name in same database |
| Neo4j / Memgraph | Label prefix |
| Milvus / Qdrant | Collection prefix |
| MongoDB / Redis | Collection scope |
| JSON / Nano / NetworkX / Faiss | Subdirectory under `working_dir/<workspace>/` |

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
| `chat_llm_reranker` | uses `rerank.provider/model` if set, otherwise `chat`; selected model must support images when reranking visual chunks | reuses chat key unless `DLIGHTRAG_RERANK__API_KEY` is set |
| `jina_reranker` | `jina-reranker-m0` | `DLIGHTRAG_RERANK__API_KEY` |
| `aliyun_reranker` | `gte-rerank` | `DLIGHTRAG_RERANK__API_KEY` |
| `azure_cohere` | `cohere-rerank-v3.5` | `DLIGHTRAG_RERANK__API_KEY` |
| `local_reranker` | (set `rerank.model` + `rerank.base_url`) | (none — local endpoint) |

For self-hosted rerankers (Xinference, vLLM, TEI etc.), use `local_reranker` with `rerank.base_url` + `rerank.model`. For any other OpenAI-compatible `/rerank` endpoint, point `rerank.base_url` at it.

### Observability (Langfuse)

DlightRAG includes native tracing using [Langfuse](https://langfuse.com/). When configured, it records hierarchical observations for service pipelines, retrieval, reranking, LLM generations, and embedding calls. If keys are omitted, the tracing module operates as a pure no-op.

To enable observability, set the following in your `.env`:

```bash
DLIGHTRAG_LANGFUSE_PUBLIC_KEY=pk-...
DLIGHTRAG_LANGFUSE_SECRET_KEY=sk-...
# DLIGHTRAG_LANGFUSE_HOST=https://cloud.langfuse.com  # Optional: defaults to cloud
# DLIGHTRAG_LANGFUSE_EXPORT_EXTERNAL_SPANS=false       # Optional: keep false to avoid double-counting auto GenAI spans
```

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
