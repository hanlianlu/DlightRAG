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

## Local Langfuse Observability

DlightRAG can send LLM, embedding, and reranking traces to Langfuse. The repo
bundles a self-hosted Langfuse stack for local development so you do not need a
Langfuse Cloud account. It runs as its own Docker Compose project on isolated
ports and is wired to DlightRAG by two helper scripts under `scripts/langfuse/`.

Non-secret SDK behavior (`langfuse_host`, `langfuse_export_external_spans`,
`langfuse_environment`, sample rate, timeout) lives in
[configuration.md](configuration.md). This section covers running the stack.

Prerequisites: Docker and Compose. The Langfuse stack is separate from the
DlightRAG Compose stack and stores its files in a sibling directory
(`../langfuse-local` by default, `LANGFUSE_LOCAL_DIR`).

### Start And View Traces

```bash
make langfuse-up
```

That single target runs the whole chain: it downloads and port-patches the
official Langfuse compose file, syncs project keys into both env files, then
starts the stack in the background. Then open the UI:

- URL: `http://localhost:3300`
- Email: `admin@localhost.local`
- Password: the generated `LANGFUSE_INIT_USER_PASSWORD` in
  `../langfuse-local/.env`
- Project: `DlightRAG_Local` → **Tracing → Traces**

Traces appear only after DlightRAG runs a model call (a query or ingest), so
trigger one request if the list is empty.

### Targets

| Target | Behavior |
|---|---|
| `make langfuse-stack` | Download the official Langfuse compose file and patch it for isolated local ports. Does not start anything. |
| `make langfuse-bootstrap` | Run `langfuse-stack`, then sync project keys so the local stack and DlightRAG's `.env` share one credential pair. |
| `make langfuse-up` | Run `langfuse-bootstrap`, then start the stack (`docker compose up -d`). |
| `make langfuse-down` | Stop the stack. |
| `make langfuse-reset` | Delete the stack's data volumes (all traces). Destructive; needs `CONFIRM=1`. |
| `make langfuse-restart` | Re-sync keys and recreate the web/worker services. |
| `make langfuse-status` | Show stack container status. |
| `make langfuse-logs` | Follow web/worker logs (foreground; Ctrl+C to exit). |
| `make langfuse-health` | Curl the host-facing health endpoint. |

### How Keys Are Synced

`scripts/langfuse/headless.py` keeps one public/secret key pair on both sides:
it writes `LANGFUSE_INIT_PROJECT_*` into `../langfuse-local/.env` (the stack's
headless initialization) and the matching `DLIGHTRAG_LANGFUSE_PUBLIC_KEY` /
`DLIGHTRAG_LANGFUSE_SECRET_KEY` into DlightRAG's `.env`. Both keys are required;
if either is missing, DlightRAG starts with tracing disabled. The keys are
generated once and reused, so re-running bootstrap does not rotate them.

Langfuse only reads `LANGFUSE_INIT_*` on first database initialization. After
the project exists, changing a key in `.env` does not change the key stored in
Langfuse. To rotate keys, change them in the Langfuse UI or reset the stack's
data volume.

### Trace Endpoint Address

`langfuse_host` in [config.yaml](../config.yaml) is where the app sends traces.
It is non-secret, so it stays in `config.yaml`; `make langfuse-up` only writes
the secret keys to `.env`. Set it for how DlightRAG itself runs:

| DlightRAG run mode | `langfuse_host` |
|---|---|
| Docker Compose (default) | `http://host.docker.internal:3300` |
| Native (non-Docker) | `http://localhost:3300` |

Inside a container `localhost` is the container itself, so a Dockerized app
reaches the host-published Langfuse port through the Docker host alias, the same
way it reaches the MinerU sidecar. Your browser and `make langfuse-health` use
`http://localhost:3300` because they run on the host. On native Linux Docker, a
host alias reaching a loopback-bound port can need extra host networking. If
`langfuse_host` is unset, DlightRAG falls back to Langfuse Cloud.

### Cost Tracking

By default a generation shows token usage but no cost: DlightRAG reports token
counts on every LLM call and reports a cost only when the provider returns one.
Add cost in either of two ways.

Model prices in Langfuse (works for any provider). In the Langfuse UI, open
Settings and add a model definition whose match pattern matches the model name
DlightRAG sends (the generation's `model`, e.g. `deepseek-v4-flash`), with input
and output prices. Langfuse then computes cost from the reported
`input`/`output`/`total` token usage.

Provider-returned cost (OpenRouter). OpenRouter can return the actual charged
cost when usage accounting is enabled. This is provider-specific, so enable it
per model rather than in the shared default, and keep it off non-OpenRouter
roles (they may reject an unknown `usage` body field):

```yaml
llm:
  default:
    provider: openai
    base_url: https://openrouter.ai/api/v1
    model: google/gemini-3.1-flash-lite
    model_kwargs:
      usage:
        include: true
```

DlightRAG forwards the response's `usage.cost` to Langfuse as the generation's
cost. Providers that do not return a cost field (for example DeepSeek) need the
model-price approach above. Recreate the app containers after editing
`config.yaml`.

### Apply Changes To A Running Stack

The DlightRAG app services load `.env` with Compose `env_file`, which is read
when a container is created. `docker compose restart` does not reload it. After
`make langfuse-up` first writes the Langfuse keys (or after you change
`langfuse_host` in `config.yaml`), recreate the app containers so they pick up
the values:

```bash
docker compose up -d --force-recreate dlightrag-api dlightrag-mcp
```

Confirm the target from the logs:

```bash
docker compose logs dlightrag-api | grep -i "langfuse tracing"
# Langfuse tracing enabled → http://host.docker.internal:3300
```

If a log line shows `→ https://cloud.langfuse.com`, the app started without
`langfuse_host` set and is using the code default; set it in `config.yaml` and
recreate the containers.

### Stop Or Disable

- Stop the stack: `make langfuse-down`.
- Disable tracing without removing the stack: clear
  `DLIGHTRAG_LANGFUSE_PUBLIC_KEY` and `DLIGHTRAG_LANGFUSE_SECRET_KEY` in `.env`,
  then recreate the app containers. With no keys, DlightRAG runs with tracing
  off and never contacts Langfuse.

### Reset And Password Recovery

The UI login password and the project API keys are seeded once, on the stack's
first database initialization, from `LANGFUSE_INIT_*` in
`../langfuse-local/.env`. After that the database is authoritative: editing
those values in `.env` has no effect, and the stored password is a one-way hash.

Try the non-destructive options first:

- Forgot the UI password, `.env` still present: read it back.

  ```bash
  grep LANGFUSE_INIT_USER_PASSWORD ../langfuse-local/.env
  ```

- Locked out of the UI but the keys are intact: traces are still readable
  through the public API with the key pair (no UI login required).

  ```bash
  curl -u "$DLIGHTRAG_LANGFUSE_PUBLIC_KEY:$DLIGHTRAG_LANGFUSE_SECRET_KEY" \
    http://localhost:3300/api/public/traces
  ```

- Deleted `../langfuse-local/.env`: the API keys are recoverable because
  DlightRAG's `.env` still holds the same pair. `make langfuse-bootstrap`
  rewrites the stack env from those keys (and generates a new password). The
  running database keeps the old password until you reset it.

Full reset (last resort). This deletes all Langfuse traces. DlightRAG's own
data is a separate Compose project and is not affected:

```bash
make langfuse-bootstrap                 # regenerate ../langfuse-local/.env (fresh password)
make langfuse-reset CONFIRM=1           # remove the stack's data volumes
make langfuse-up                        # re-initialize from the new seed
grep LANGFUSE_INIT_USER_PASSWORD ../langfuse-local/.env   # read the new password
```

Because bootstrap recovers the same API keys from DlightRAG's `.env`, tracing
keeps working after the reset with no change on the DlightRAG side; only the
trace history and the login password change.
