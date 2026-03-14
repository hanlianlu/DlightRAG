# API Reference

Request and response structures for `ingest`, `retrieve`, and `answer` — shared across Python SDK, REST API, MCP server, and Web UI.


## Ingestion

### Python SDK

```python
from dlightrag import RAGService, DlightragConfig

service = await RAGService.create(config=DlightragConfig())

# Local files or directory
result = await service.aingest(source_type="local", path="./docs")

# Azure Blob Storage
result = await service.aingest(
    source_type="azure_blob",
    container_name="documents",
    prefix="reports/",       # or blob_path="reports/q1.pdf"
)

# Snowflake
result = await service.aingest(source_type="snowflake", query="SELECT * FROM docs")
```

### REST API

```bash
curl -X POST http://localhost:8100/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type": "local", "path": "/data/docs"}'
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `source_type` | `string` | yes | `local`, `azure_blob`, `snowflake` |
| `path` | `string` | local | File or directory path |
| `container_name` | `string` | azure_blob | Blob container name |
| `blob_path` | `string` | — | Specific blob (mutually exclusive with `prefix`) |
| `prefix` | `string` | — | Blob prefix filter |
| `query` | `string` | snowflake | SQL query |
| `table` | `string` | — | Snowflake table name |
| `replace` | `boolean` | — | Replace existing documents with same name |
| `workspace` | `string` | — | Target workspace (default: `default`) |

### MCP Server

Same parameters as REST API, passed as tool arguments.

### Ingestion Response

```json
{
  "status": "success",
  "source_type": "local",
  "processed": 5,
  "skipped": 2,
  "total_files": 7,
  "source_path": "/data/docs",
  "skipped_files": ["duplicate.pdf", "already_indexed.pdf"],
  "stats": { "total": 7, "indexed": 5, "dropped_by_type": 0 }
}
```

| Field | Type | Description |
|---|---|---|
| `status` | `string` | `success` or `error` |
| `processed` | `int` | Files successfully ingested |
| `skipped` | `int` | Files skipped (dedup or unsupported) |
| `total_files` | `int` | Total files found |
| `skipped_files` | `list` | Names of skipped files |
| `error` | `string` | Error message (only when `status: "error"`) |


## Retrieval & Answer

### Quick Reference

| Interface | `retrieve` | `answer` | Streaming |
|---|---|---|---|
| Python SDK | `RetrievalResult` | `RetrievalResult` | `(contexts, token_iter)` |
| REST API | JSON object | JSON object | SSE (`stream: true`) |
| MCP Server | JSON text | JSON text | N/A |
| Web UI | — | SSE (HTML) | Built-in |

### Python SDK

```python
# Retrieve: contexts only, no LLM answer
result = await service.aretrieve(query="What are the key findings?")
result.answer     # None
result.contexts   # RetrievalContexts: {"chunks": [...], "entities": [...], "relationships": [...]}

# Answer: contexts + LLM-generated answer
result = await service.aanswer(query="What are the key findings?")
result.answer      # "The key findings are... [1-1] [2-3]"
result.contexts    # same structure as retrieve
result.references  # [Reference(id=1, title="report.pdf"), ...] (unified mode with structured output)

# Streaming answer
contexts, token_iter = await service.aanswer_stream(query="What are the key findings?")
# contexts (RetrievalContexts) available immediately
async for token in token_iter:
    print(token, end="")
```

**Parameters** (shared by all three methods):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | required | Search query |
| `mode` | `str` | `"mix"` | `local`, `global`, `hybrid`, `naive`, `mix` |
| `top_k` | `int \| None` | config default | Total results to retrieve |
| `chunk_top_k` | `int \| None` | config default | Chunk-level results |
| `multimodal_content` | `list[dict]` | `None` | Up to 3 images (unified mode) |

### REST API

```bash
# Retrieve
curl -X POST http://localhost:8100/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "key findings", "mode": "mix"}'

# Answer
curl -X POST http://localhost:8100/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "key findings"}'

# Streaming answer
curl -X POST http://localhost:8100/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "key findings", "stream": true}'
```

**Non-streaming response:**

```json
{
  "answer": "The key findings are... [1-1] [2-3]",
  "contexts": { "chunks": [...], "entities": [...], "relationships": [...] },
  "references": [{"id": 1, "title": "report.pdf"}, {"id": 2, "title": "spec.pdf"}],
  "sources": [...]
}
```

**SSE streaming** (`stream: true`): events are newline-delimited JSON.

| Event | Payload | Description |
|---|---|---|
| `context` | `{type, data, sources}` | Full contexts + sources (sent first) |
| `token` | `{type, content}` | LLM answer token (repeats) |
| `references` | `{type, data}` | Structured references (after all tokens, before done) |
| `done` | `{type}` | Stream complete |
| `error` | `{type, message}` | Error mid-stream |

```
data: {"type":"context","data":{"chunks":[...],"entities":[...],"relationships":[...]},"sources":[...]}

data: {"type":"token","content":"The key findings"}

data: {"type":"token","content":" are..."}

data: {"type":"references","data":[{"id":1,"title":"report.pdf"}]}

data: {"type":"done"}
```

**Additional REST parameters:** `workspaces` (list, for federated search).

### MCP Server

MCP tools return JSON text with `sources` at top level:

```json
{
  "answer": "The key findings are... [1-1]",
  "contexts": { "chunks": [...], "entities": [...], "relationships": [...] },
  "references": [{"id": 1, "title": "report.pdf"}],
  "sources": [...]
}
```


## Contexts Object

All modes return `contexts` as a `RetrievalContexts` TypedDict with three arrays. Chunks are the primary retrieval unit; entities and relationships come from the knowledge graph.

```python
from dlightrag.core.retrieval.protocols import RetrievalContexts, ChunkContext, EntityContext, RelationshipContext
```

### chunks

```json
{
  "chunk_id": "abc123",
  "reference_id": "1",
  "file_path": "/data/report.pdf",
  "content": "Page text content...",
  "page_idx": 2,
  "image_data": "iVBORw0KGgo...",
  "relevance_score": 0.87
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `chunk_id` | string | yes | Unique chunk identifier |
| `reference_id` | string | yes | Document-level ID (groups chunks from the same file) |
| `file_path` | string | yes | Source file path |
| `content` | string | yes | Chunk text content |
| `page_idx` | int \| null | no | **1-based** page number |
| `image_data` | string \| null | no | Base64-encoded page image (unified mode only) |
| `relevance_score` | float \| null | no | 0–1 relevance score (when reranking is enabled) |
| `metadata` | object | no | Extra metadata (`file_name`, `file_type`, etc.) |
| `_workspace` | string | no | Source workspace (federated queries only) |

### entities

```json
{
  "entity_name": "PostgreSQL",
  "entity_type": "TECHNOLOGY",
  "description": "An open-source relational database",
  "source_id": "abc123"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `entity_name` | string | yes | Entity name/label |
| `entity_type` | string | yes | Category (Person, Organization, Technology, etc.) |
| `description` | string | yes | Summary description |
| `source_id` | string | yes | Comma-separated `chunk_id` values linking to source chunks |
| `reference_id` | string | no | Document reference (inferred from source_id) |

### relationships

```json
{
  "src_id": "PostgreSQL",
  "tgt_id": "pgvector",
  "description": "extension for vector similarity search",
  "source_id": "abc123"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `src_id` | string | yes | Source entity name |
| `tgt_id` | string | yes | Target entity name |
| `description` | string | yes | Relationship description |
| `source_id` | string | yes | Comma-separated `chunk_id` values linking to source chunks |
| `reference_id` | string | no | Document reference (inferred from source_id) |


## Sources

Sources are document-level groupings derived from chunks via `build_sources()`. They appear in REST/MCP responses and drive the Web UI's source panel.

```json
{
  "id": "1",
  "title": "report.pdf",
  "path": "/data/report.pdf",
  "type": "file",
  "url": "file://sources/local/report.pdf",
  "chunks": [
    {
      "chunk_id": "abc123",
      "chunk_idx": 0,
      "page_idx": 2,
      "content": "First 200 characters of content...",
      "image_data": null
    },
    {
      "chunk_id": "def456",
      "chunk_idx": 1,
      "page_idx": 5,
      "content": "Another chunk...",
      "image_data": "iVBORw0KGgo..."
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `id` | string | Reference ID (matches `reference_id` in chunks) |
| `title` | string \| null | Document title (filename or metadata) |
| `path` | string | Source file path |
| `type` | string \| null | File type |
| `url` | string \| null | Resolved download URL (via PathResolver) |
| `chunks` | list | Chunk snippets sorted by `page_idx` |

Each **chunk snippet** within a source:

| Field | Type | Description |
|---|---|---|
| `chunk_id` | string | Unique chunk identifier |
| `chunk_idx` | int | 0-based position within this source |
| `page_idx` | int \| null | 1-based page number |
| `content` | string | Filtered display content |
| `image_data` | string \| null | Base64 page image (unified mode) |
| `highlight_phrases` | list \| null | Semantic highlight phrases (when available) |


## References

When using unified mode with a provider that supports structured output, the `answer` response includes a `references` array containing document-level references cited in the answer. This is a validated subset of `sources` — only documents actually cited by the LLM appear here.

```json
{
  "id": 1,
  "title": "report.pdf"
}
```

| Field | Type | Description |
|---|---|---|
| `id` | int | Reference number matching `[n]` in inline citations |
| `title` | string | Document title/filename |

**Relationship to `sources`:** `sources` contains all documents from retrieval; `references` contains only those the LLM cited. For providers that don't support structured output (Ollama, Xinference) or caption mode, `references` is an empty array.

**Supported providers:** OpenAI, Azure OpenAI, Anthropic, Google Gemini, Qwen, Minimax, OpenRouter.


## Citations

When using `answer`, the LLM response may contain inline citations in two formats:

| Format | Example | Meaning |
|---|---|---|
| `[ref_id-chunk_idx]` | `[1-2]` | Chunk-level: document 1, chunk 2 |
| `[n]` | `[3]` | Document-level: all chunks from document 3 |

- `ref_id` maps to `reference_id` in chunks and `id` in sources
- `chunk_idx` is **1-based**, matching the chunk's position within its document

### Resolving a citation

To trace `[1-2]` back to source material:

1. Find chunks where `reference_id == "1"` — these are all chunks from that document
2. The 2nd chunk (1-based) in that group is the cited chunk
3. Use `chunk_id` to look up the source in `sources` (by matching `id`)
4. Use `page_idx` on the chunk for the page number
5. Use `file_path` or source `url` to access the original file via `GET /api/files/{path}`


## Multimodal Queries (Unified Mode)

Upload images alongside a text query for visual similarity search:

```python
# Python SDK
with open("photo.png", "rb") as f:
    img_bytes = f.read()

result = await service.aanswer(
    query="What does this diagram show?",
    multimodal_content=[{"type": "image", "data": img_bytes}],
)
```

```bash
# REST API — base64-encoded image
curl -X POST http://localhost:8100/answer \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does this diagram show?",
    "multimodal_content": [{"type": "image", "data": "<base64>"}]
  }'

# REST API — file path (server-accessible)
curl -X POST http://localhost:8100/answer \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does this diagram show?",
    "multimodal_content": [{"type": "image", "img_path": "/data/diagram.png"}]
  }'
```

Maximum 3 images per query. Only available in unified mode.
