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
| Python SDK | `RetrievalResult` | `RetrievalResult` | `(contexts, raw, token_iter)` |
| REST API | JSON object | JSON object | SSE (`stream: true`) |
| MCP Server | JSON text | JSON text | N/A |
| Web UI | — | SSE (HTML) | Built-in |

### Python SDK

```python
# Retrieve: contexts only, no LLM answer
result = await service.aretrieve(query="What are the key findings?")
result.answer     # None
result.contexts   # {"chunks": [...], "entities": [...], "relationships": [...]}
result.raw        # {"sources": [...], "media": [...]}

# Answer: contexts + LLM-generated answer
result = await service.aanswer(query="What are the key findings?")
result.answer     # "The key findings are... [1-1] [2-3]"
result.contexts   # same structure as retrieve
result.raw        # same structure as retrieve

# Streaming answer
contexts, raw, token_iter = await service.aanswer_stream(query="What are the key findings?")
# contexts and raw are available immediately
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
  "raw": { "sources": [...], "media": [...] }
}
```

**SSE streaming** (`stream: true`): events are newline-delimited JSON.

| Event | Payload | Description |
|---|---|---|
| `context` | `{type, data, raw}` | Full contexts + sources (sent first) |
| `token` | `{type, content}` | LLM answer token (repeats) |
| `done` | `{type}` | Stream complete |
| `error` | `{type, message}` | Error mid-stream |

```
data: {"type":"context","data":{"chunks":[...],"entities":[...],"relationships":[...]},"raw":{"sources":[...],"media":[...]}}

data: {"type":"token","content":"The key findings"}

data: {"type":"token","content":" are..."}

data: {"type":"done"}
```

**Additional REST parameters:** `workspaces` (list, for federated search), `conversation_history` (list, for answer only).

### MCP Server

MCP tools return JSON text. The shape is slightly flattened — `sources` is promoted to top level (no `raw` wrapper):

```json
{
  "answer": "The key findings are... [1-1]",
  "contexts": { "chunks": [...], "entities": [...], "relationships": [...] },
  "sources": [...]
}
```


## Contexts Object

All modes return `contexts` with three arrays. Chunks are the primary retrieval unit; entities and relationships come from the knowledge graph.

### chunks

```json
{
  "chunk_id": "abc123",
  "reference_id": "1",
  "file_path": "/data/report.pdf",
  "content": "Page text content...",
  "page_idx": 2,
  "relevance_score": 0.87
}
```

| Field | Type | Description |
|---|---|---|
| `chunk_id` | string | Unique chunk identifier |
| `reference_id` | string | Document-level ID (groups chunks from the same file) |
| `file_path` | string | Source file path |
| `content` | string | Chunk text content |
| `page_idx` | int | **1-based** page number |
| `relevance_score` | float | 0–1 relevance score (present when reranking is enabled) |

### entities

```json
{
  "entity_name": "PostgreSQL",
  "entity_type": "TECHNOLOGY",
  "description": "An open-source relational database",
  "source_id": "abc123"
}
```

### relationships

```json
{
  "src_id": "PostgreSQL",
  "tgt_id": "pgvector",
  "description": "extension for vector similarity search",
  "source_id": "abc123"
}
```

> `source_id` in entities/relationships is a comma-separated list of `chunk_id` values linking back to source chunks.


## Sources and Media

The `raw` object (REST/SDK) or `sources` array (MCP) contains document-level metadata and associated media.

### Caption Mode

**sources:**
```json
{
  "id": "1",
  "type": "file",
  "title": "report.pdf",
  "path": "/data/report.pdf",
  "url": "file://sources/local/report.pdf",
  "snippet": "First 100 characters of content...",
  "chunk_ids": ["abc123", "def456"]
}
```

**media** (extracted images):
```json
{
  "id": "a1b2c3",
  "type": "image",
  "reference_id": "1",
  "source_chunk_id": "abc123",
  "title": "report.pdf",
  "path": "/data/images/figure_001.png",
  "url": "file://sources/local/images/figure_001.png",
  "caption": "Figure 1: System architecture"
}
```

### Unified Mode

**sources:**
```json
{
  "doc_id": "doc-abc123",
  "title": "Quarterly Report",
  "author": "Author Name",
  "path": "/data/report.pdf"
}
```

**media** (rendered page images, base64-encoded):
```json
{
  "chunk_id": "abc123",
  "page_index": 0,
  "image_data": "iVBORw0KGgo...",
  "relevance_score": 0.92
}
```

> **Note:** In media, `page_index` is **0-based** (raw storage value). In chunks, `page_idx` is **1-based** (display value).


## Citations

When using `answer`, the LLM response may contain inline citations in two formats:

| Format | Example | Meaning |
|---|---|---|
| `[ref_id-chunk_idx]` | `[1-2]` | Chunk-level: document 1, chunk 2 |
| `[n]` | `[3]` | Document-level: all chunks from document 3 |

- `ref_id` maps to `reference_id` in chunks and `id`/`doc_id` in sources
- `chunk_idx` is **1-based**, matching the chunk's position within its document

### Resolving a citation

To trace `[1-2]` back to source material:

1. Find chunks where `reference_id == "1"` — these are all chunks from that document
2. The 2nd chunk (1-based) in that group is the cited chunk
3. Use `chunk_id` to look up the source in `sources` (by matching `id` or `doc_id`)
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
