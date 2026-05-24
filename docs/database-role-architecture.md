# Database Role Architecture

DlightRAG supports PostgreSQL process roles instead of in-process dynamic
connection switching.

```text
ingest/admin workers -> primary PostgreSQL
query workers        -> hot standby / read replica PostgreSQL
primary              -> physical streaming replication -> replica
```

## Roles

| Role | PostgreSQL target | Allowed operations |
|---|---|---|
| `ingest` | primary | startup DDL, ingest, delete, metadata update, retry, reset, query |
| `admin` | primary | same write/admin surface as ingest |
| `query` | replica | retrieve, answer, list files, list/search metadata, health |

`query` role rejects mutating APIs and performs read-only attach:

- LightRAG storages are bound to an existing PostgreSQL pool without
  `initialize_storages()` DDL.
- DlightRAG metadata, artifact, provenance, visual chunk, and BM25 stores verify
  schema/index existence without creating tables or indexes.
- Missing schema is a startup error; initialize or migrate on primary first.

## Configuration

Use `runtime_role` to select the process role. Existing `postgres_*` fields are
the primary endpoint for compatibility. `postgres_replica_*` fields are used by
query workers; unset replica fields fall back to primary for local development.

```yaml
runtime_role: query
postgres_host: primary.internal
postgres_user: dlightrag
postgres_replica_host: replica.internal
postgres_replica_user: dlightrag_reader
read_after_write_mode: eventual
```

## Tradeoffs

- Read-after-write is eventually consistent unless the caller waits for replica
  replay or explicitly uses an ingest/admin surface.
- Hot standby queries can be canceled by WAL replay conflicts. PostgreSQL
  `max_standby_streaming_delay` and `hot_standby_feedback` are operational
  tradeoffs, not application toggles.
- Extension versions must match across primary and replica: PostgreSQL 18,
  `pgvector`, Apache AGE, and `pg_textsearch`.
