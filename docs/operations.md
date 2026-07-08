# Operations

This page is for operators running maintenance commands against existing
DlightRAG and LightRAG storage. It owns rebuilds, maintenance safety, and
recovery workflows. PostgreSQL deployment tuning lives in
[postgresql.md](postgresql.md); configuration fields live in
[configuration.md](configuration.md).

The commands here are not part of normal ingestion or query traffic.

## Offline Vector Storage Rebuild

`dlightrag-rebuild-vdb` rebuilds LightRAG vector storage from existing
LightRAG graph and chunk records while using DlightRAG's configured storage,
workspace, embedding provider, BM25 language labeling, and sidecar alignment
rules.

It does not ingest files, parse documents, create document status records, or
change source documents. It only checks or rewrites vector rows derived from
data that already exists in LightRAG storage.

Use it when:

- a LightRAG upgrade recommends rebuilding vector storage
- vector rows are missing, stale, or inconsistent with the graph/chunk stores
- `--target check` reports graph/vector mismatches
- the embedding configuration was intentionally changed and the vector schema
  already supports the resulting embedding dimensions

Do not use it as a normal retry path for failed ingestion. Use the failed-file
retry endpoints for that case.

### Targets

| Target | Writes storage | Behavior |
|---|---:|---|
| `check` | no | Compare graph records with entity and relationship vector rows. |
| `graph` | yes | Rebuild entity and relationship vectors from the graph store. |
| `chunks` | yes | Rebuild chunk vectors from LightRAG text chunks, then refresh BM25 language labels and restore DlightRAG sidecar image-vector alignment. |
| `all` | yes | Run graph and chunk vector rebuilds, then run the chunk post-rebuild maintenance. |

`graph`, `chunks`, and `all` require `--yes`. Treat them as offline
maintenance: stop DlightRAG API, MCP, ingest workers, and any other process
that can write to the same LightRAG storage before running them.

### Native Or Repo Run

Start with a read-only check:

```bash
uv run dlightrag-rebuild-vdb --target check
```

For a rebuild, stop service writers first, then run:

```bash
uv run dlightrag-rebuild-vdb --target all --yes
```

For an installed package outside the repo:

```bash
dlightrag-rebuild-vdb --env-file /absolute/path/to/.env --target check
dlightrag-rebuild-vdb --env-file /absolute/path/to/.env --target all --yes
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--env-file PATH` | Load a specific `.env` file instead of relying on the current working directory. |
| `--batch-size N` | Source records per rebuild batch. The default follows upstream LightRAG. |
| `--no-restore-sidecar-alignment` | Skip DlightRAG's direct image-vector restoration after a chunk rebuild. |

### Docker Compose Run

When DlightRAG runs through this repository's Compose stack, run the command in
the same app image so it sees the same `config.yaml`, environment, network, and
storage volume:

```bash
# Read-only check — safe while the stack is running:
docker compose run --rm dlightrag-api dlightrag-rebuild-vdb --target check

# Rebuild — stop writers first, then restart them afterward:
docker compose stop dlightrag-api dlightrag-mcp lightrag-gui
docker compose run --rm dlightrag-api dlightrag-rebuild-vdb --target all --yes
docker compose up -d dlightrag-api dlightrag-mcp
```

If you do not run `lightrag-gui`, omit it from the stop command. If you do run
it, restart it with the other services after the rebuild.

### DlightRAG Chunk Post-Rebuild Maintenance

Upstream LightRAG's rebuild can rebuild chunk vectors from chunk text. DlightRAG
adds two post-rebuild steps after successful `chunks` and `all` targets.

First, when BM25 is enabled, it scans LightRAG text chunks and refreshes each
row's `dlightrag_bm25_language` label with the configured BM25 language
profiles. This keeps query-language-routed BM25 partial indexes aligned after
offline chunk rebuilds.

Second, DlightRAG restores direct image-vector alignment:

1. It reads processed documents and their sidecar locations.
2. It reuses DlightRAG's multimodal embedder and ingestion alignment code.
3. It overwrites canonical LightRAG drawing chunk vectors with direct image
   vectors when the configured image embedding probe succeeds.

This preserves visual retrieval behavior after `--target chunks` or
`--target all`. Use `--no-restore-sidecar-alignment` only for debugging or for
a text-only embedding deployment where direct image-vector retrieval is
intentionally disabled.

### Operational Notes

- Take a PostgreSQL backup before destructive rebuilds on production data.
- Run the command with the same `.env`, `config.yaml`, workspace, and embedding
  provider settings used by the service.
- Do not change embedding dimensions unless the underlying vector schema has
  already been recreated or migrated for the new dimension.
- A nonzero exit code means the rebuild reported errors or storage setup failed;
  inspect the command output before restarting writers.
