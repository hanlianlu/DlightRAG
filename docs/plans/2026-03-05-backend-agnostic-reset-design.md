# Backend-Agnostic Reset Design

## Problem

`scripts/reset.py` hardcodes PostgreSQL + Apache AGE cleanup logic. If the user
configures a different storage backend (e.g. NanoVectorDB, Neo4J, JsonKV), the
reset script silently does nothing for those backends — or worse, errors out
trying to connect to a non-existent PostgreSQL.

## Goal

Make `reset.py` work correctly for **any** storage backend combination configured
in `DlightragConfig`, without hardcoding backend-specific cleanup logic.

## Approach: Delegate to LightRAG `drop()`

LightRAG's `StorageNameSpace` base class defines `async def drop()` on every
storage implementation. Each backend knows how to clean itself up:

- **PG storages**: `DELETE FROM table WHERE workspace = ?`
- **Json/NanoVector**: delete local files, reset in-memory state
- **Neo4J/Memgraph**: `MATCH (n) DETACH DELETE n`
- **Milvus**: drop collection
- **Redis/Mongo**: scan & delete by namespace prefix

We leverage this by initializing a `RAGService` (which creates LightRAG with the
correct backends from config), then calling `drop()` on each of LightRAG's 12
storage instances.

## Design

### Flow

```
reset.py
  1. RAGService.create()  →  initializes LightRAG with config-driven backends
  2. lightrag = service.ingestion.rag.lightrag
  3. For each of lightrag's 12 storages:
       - dry_run: print storage name + class type
       - real run: await storage.drop()
  4. Clean DlightRAG hash_index (if present)
  5. Clean local files (unless --keep-files)
  6. await service.close()
  7. Print summary
```

### LightRAG Storage Instances (12 total)

| Attribute                     | Type          | Purpose                    |
|-------------------------------|---------------|----------------------------|
| `full_docs`                   | KV storage    | Full document text         |
| `text_chunks`                 | KV storage    | Chunked text               |
| `full_entities`               | KV storage    | Extracted entities         |
| `full_relations`              | KV storage    | Extracted relations        |
| `entity_chunks`               | KV storage    | Entity-chunk mapping       |
| `relation_chunks`             | KV storage    | Relation-chunk mapping     |
| `entities_vdb`                | Vector DB     | Entity embeddings          |
| `relationships_vdb`           | Vector DB     | Relationship embeddings    |
| `chunks_vdb`                  | Vector DB     | Chunk embeddings           |
| `chunk_entity_relation_graph` | Graph storage | Knowledge graph            |
| `llm_response_cache`         | KV storage    | LLM response cache         |
| `doc_status`                  | Doc status    | Document processing status |

### CLI Interface (unchanged)

```
uv run scripts/reset.py                  # reset all (with confirmation)
uv run scripts/reset.py --dry-run        # preview what would be dropped
uv run scripts/reset.py --keep-files     # drop storages, keep local files
uv run scripts/reset.py -y               # skip confirmation prompt
```

### What Gets Deleted

- `_reset_tables()` — hardcoded PostgreSQL DELETE FROM logic
- `_reset_graphs()` — hardcoded AGE drop_graph logic
- `_load_env()` — manual env parsing (replaced by DlightragConfig)
- `_TABLE_PREFIXES` constant

### What Gets Kept

- `_reset_local()` — local file cleanup (backend-independent)
- `_format_size()` — size formatting helper
- `--dry-run`, `--keep-files`, `-y` flags
- Confirmation prompt and summary output

### Dry-Run Output

```
Storage backends (from config):
  KV:         PGKVStorage
  Vector:     PGVectorStorage
  Graph:      PGGraphStorage
  DocStatus:  PGDocStatusStorage
  Workspace:  default

[DRY RUN] Would drop 12 storages:
  full_docs (PGKVStorage)
  text_chunks (PGKVStorage)
  full_entities (PGKVStorage)
  full_relations (PGKVStorage)
  entity_chunks (PGKVStorage)
  relation_chunks (PGKVStorage)
  entities_vdb (PGVectorStorage)
  relationships_vdb (PGVectorStorage)
  chunks_vdb (PGVectorStorage)
  chunk_entity_relation_graph (PGGraphStorage)
  llm_response_cache (PGKVStorage)
  doc_status (PGDocStatusStorage)

Local files:
  [DRY RUN] ./dlightrag_storage: 42 files (12.3 MB) (sources/ preserved)
```

## Scope

- **In scope**: `scripts/reset.py` rewrite
- **Out of scope**: `scripts/cli.py` (already config-driven via RAGService)

## Testing

- Unit tests not practical (requires live storage backends)
- Manual verification with `--dry-run` on current PG setup
- Verify `drop()` return values are logged correctly
