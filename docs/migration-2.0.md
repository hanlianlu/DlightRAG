# Migrating to DlightRAG 2.0

DlightRAG 2.0 deliberately has no compatibility layer for the 1.x package or
configuration layout. Upgrade the dependency declarations, Python imports,
YAML, environment variables, and programmatic configuration together. A mixed
1.x/2.0 deployment should fail early rather than silently read a stale value.

This guide targets the 1.9 package and configuration layout immediately
preceding 2.0.

## What Does Not Migrate

The consolidation does not rename or rewrite persisted business data:

- PostgreSQL schemas, tables, and records stay in place;
- SQL object names beginning with `dlightrag_agent_` intentionally stay in
  place;
- Docker volume names stay in place;
- the workspace input and managed-storage layout stays in place when
  `deployment.working_dir` keeps the former `working_dir` value;
- Owner Profile Memory remains the separate `dlightrag-memory` distribution and
  keeps the `dlightrag_memory` import root.

No data-conversion command is required for this consolidation. Normal release
backup and rollback procedures still apply.

## 1. Preserve a Rollback Point

Before changing the application environment:

1. Record the exact 1.x root and companion package versions.
2. Save the current `config.yaml`, `.env`, Compose overrides, deployment
   manifests, and secret-manager key names.
3. Take the database and volume backups required by your normal release policy.
4. Keep the 1.x configuration files with the 1.x rollback artifact. A 2.0
   configuration is not a rollback configuration for 1.x.

Do not rename existing `dlightrag_agent_*` database objects or volumes because
the names look historical. Version 2.0 continues to use them.

## 2. Replace Distribution Dependencies

Version 1.x installed four internal companion distributions with the root
product. Version 2.0 publishes only two distributions:

| 1.x distribution | 2.0 action |
| --- | --- |
| `dlightrag` | Upgrade to `dlightrag==2.0.0`. |
| `dlightrag-ai` | Remove; its code is in the root wheel. |
| `dlightrag-agent-core` | Remove; its code is in the root wheel. |
| `dlightrag-rag-core` | Remove; its code is in the root wheel. |
| `dlightrag-memory` | Upgrade to `dlightrag-memory==2.0.0`; it remains standalone. |

The root wheel requires the matching Memory wheel. A private index or release
pipeline must publish `dlightrag-memory` before `dlightrag`.

Provider extras from `dlightrag-ai` no longer exist. OpenAI-compatible,
Anthropic, Gemini, LightRAG, AWS S3, and Azure Blob support are direct root
dependencies.

For a non-editable environment, remove stale 1.x companion wheels before the
2.0 install so obsolete import roots cannot hide missed source changes:

```bash
python -m pip uninstall -y dlightrag-ai dlightrag-agent-core dlightrag-rag-core
python -m pip install --upgrade "dlightrag-memory==2.0.0" "dlightrag==2.0.0"
```

UV workspace users should remove the three old package members and old
`tool.uv.sources` entries, retain only `packages/memory`, then refresh the lock
file.

## 3. Update Python Imports

The former top-level import packages do not remain as aliases:

| 1.x prefix | 2.0 prefix |
| --- | --- |
| `dlightrag_ai` | `dlightrag.ai` |
| `dlightrag_agent` | `dlightrag.agent` |
| `dlightrag_rag` | `dlightrag.rag` |
| `dlightrag_memory` | Unchanged. |

Update the full module path, not only the package dependency. For example:

```python
# 1.x
from dlightrag_ai.messages import AssistantTurn
from dlightrag_agent.tools import AgentTool
from dlightrag_rag.settings import RagSettings

# 2.0
from dlightrag.ai.messages import AssistantTurn
from dlightrag.agent.tools import AgentTool
from dlightrag.rag.settings import RagSettings
```

Some undocumented composition helpers and broad package-level re-exports were
removed during consolidation. Import an owned type from its defining module and
compose through the root application or documented service interfaces. Do not
recreate deleted one-adapter factories or protocols as local compatibility
wrappers.

A repository-wide search for the three old prefixes should return no executable
source, tests, scripts, or deployment-generated Python before rollout.

## 4. Rebuild Configuration Around Eight Sections

The only accepted top-level server sections are:

1. `deployment`
2. `storage`
3. `models`
4. `corpus`
5. `answer`
6. `access`
7. `interfaces`
8. `observability`

All sections and nested models reject unknown fields and are frozen after
construction. Start with the checked-in 2.0 `config.yaml` and `.env.example`,
then copy intentional values from the 1.x deployment. Do not paste the old file
under a wrapper or keep both old and new keys.

Configuration precedence is:

```text
constructor arguments > environment variables > .env > config.yaml > defaults
```

### YAML and constructor path mapping

The following table covers every 1.x root field group. A wildcard means the
field name is retained below the new parent unless a more specific row says
otherwise.

| 1.x path | 2.0 path |
| --- | --- |
| `service_role` | `deployment.service_role` |
| `workspace` | `deployment.workspace` |
| `working_dir` | `deployment.working_dir` |
| `postgres_*` | `storage.postgres.*` after removing the `postgres_` prefix |
| `pg_vector_index_type` | `storage.lightrag.vector_index_type` |
| `pg_hnsw_m` | `storage.lightrag.hnsw_m` |
| `pg_hnsw_ef_construction` | `storage.lightrag.hnsw_ef_construction` |
| `pg_hnsw_ef_search` | `storage.lightrag.hnsw_ef_search` |
| `vector_storage`, `graph_storage`, `kv_storage`, `doc_status_storage`, `vector_db_kwargs` | `storage.lightrag.*` |
| `llm` | `models.chat` |
| `model_capacity_overrides` | `models.capacity_overrides` |
| `embedding` | `models.embedding` |
| `rerank` | `models.rerank` |
| `max_async` | `models.max_concurrency` |
| `embedding_func_max_async` | `models.embedding.max_concurrency` |
| `embedding_batch_num` | `models.embedding.batch_size` |
| `embedding_request_timeout` | `models.embedding.timeout` |
| `parser` | `corpus.parser` |
| `parser_sidecars` | `corpus.sidecars` |
| `extraction` | `corpus.extraction` |
| `rag_pipeline_max_async` | `corpus.ingestion.pipeline.max_concurrency` |
| `max_parallel_*`, `queue_size_*` | `corpus.ingestion.pipeline.*` |
| `chunk_p_token_size` | `corpus.ingestion.chunk_token_size` |
| `ingestion_replace_default` | `corpus.ingestion.replace_default` |
| `retain_remote_source_files` | `corpus.ingestion.retain_remote_source_files` |
| `url_ingest_max_bytes` | `corpus.ingestion.url_max_bytes` |
| `url_ingest_private_host_allowlist` | `corpus.ingestion.url_private_host_allowlist` |
| `max_upload_bytes` | `corpus.ingestion.max_upload_bytes` |
| `ingest_timeout` | `corpus.ingestion.timeout` |
| `top_k`, `chunk_top_k`, `bm25_*`, `rrf_k`, `direct_visual_top_k` | `corpus.retrieval.*` |
| `metadata_filter_exact_vector_threshold`, `max_entity_tokens`, `max_relation_tokens`, `max_total_tokens` | `corpus.retrieval.*` |
| `kg_chunk_pick_method`, `kg_entity_types` | `corpus.retrieval.*` |
| `retrieval_timeout` | `corpus.retrieval.timeout` |
| `blob_connection_string`, `azure_sas_expiry`, `s3_presign_expiry`, `s3_region` | `corpus.sources.*` |
| `visual_assets` | `corpus.visual_assets` |
| former fields inside `answer` | `answer.generation.*` |
| `runtime` | `answer.runtime` |
| `agent` | `answer.agent` |
| `citations` | `answer.citations` |
| `web_conversations` | `answer.conversations` |
| `web_search` | `answer.web_search` |
| `auth_mode`, `api_auth_token`, `allow_insecure_no_auth` | `access.auth_mode`, `access.api_token`, `access.allow_insecure_no_auth` |
| `jwt_*`, `cors_allow_origins`, `web_identity` | `access.*` |
| `access_control` | `access.control` |
| `api_host`, `api_port` | `interfaces.api.host`, `interfaces.api.port` |
| `mcp_transport`, `mcp_host`, `mcp_port`, `mcp_allowed_*`, `mcp_resource_server_url` | `interfaces.mcp.*` after removing the `mcp_` prefix |
| `max_upload_size_mb` | `interfaces.max_upload_size_mb` |
| `log_level`, `langfuse_*` | `observability.*` |

The `answer` name is the main YAML ambiguity: the old `answer.max_images` shape
is now `answer.generation.max_images`, while durable run fields move from
`runtime` to `answer.runtime`.

### Representative YAML conversion

```yaml
# 1.x
llm:
  default:
    provider: openai
    model: example-model
embedding:
  provider: voyage
  model: voyage-multimodal-3.5
workspace: finance
runtime:
  answer_worker_concurrency: 16
answer:
  max_images: 12

# 2.0
models:
  chat:
    default:
      provider: openai
      model: example-model
  embedding:
    provider: voyage
    model: voyage-multimodal-3.5
deployment:
  workspace: finance
answer:
  runtime:
    answer_worker_concurrency: 16
  generation:
    max_images: 12
```

### Environment-variable conversion

A server environment variable is `DLIGHTRAG_` followed by every YAML path
segment in uppercase, joined with double underscores.

| 1.x variable | 2.0 variable |
| --- | --- |
| `DLIGHTRAG_LLM__DEFAULT__API_KEY` | `DLIGHTRAG_MODELS__CHAT__DEFAULT__API_KEY` |
| `DLIGHTRAG_LLM__ROLES__QUERY__API_KEY` | `DLIGHTRAG_MODELS__CHAT__ROLES__QUERY__API_KEY` |
| `DLIGHTRAG_EMBEDDING__API_KEY` | `DLIGHTRAG_MODELS__EMBEDDING__API_KEY` |
| `DLIGHTRAG_RERANK__API_KEY` | `DLIGHTRAG_MODELS__RERANK__API_KEY` |
| `DLIGHTRAG_POSTGRES_HOST` | `DLIGHTRAG_STORAGE__POSTGRES__HOST` |
| `DLIGHTRAG_POSTGRES_PASSWORD` | `DLIGHTRAG_STORAGE__POSTGRES__PASSWORD` |
| `DLIGHTRAG_WORKSPACE` | `DLIGHTRAG_DEPLOYMENT__WORKSPACE` |
| `DLIGHTRAG_WORKING_DIR` | `DLIGHTRAG_DEPLOYMENT__WORKING_DIR` |
| `DLIGHTRAG_PARSER_SIDECARS__MINERU__API_TOKEN` | `DLIGHTRAG_CORPUS__SIDECARS__MINERU__API_TOKEN` |
| `DLIGHTRAG_WEB_SEARCH__API_KEY` | `DLIGHTRAG_ANSWER__WEB_SEARCH__API_KEY` |
| `DLIGHTRAG_AUTH_MODE` | `DLIGHTRAG_ACCESS__AUTH_MODE` |
| `DLIGHTRAG_API_AUTH_TOKEN` | `DLIGHTRAG_ACCESS__API_TOKEN` |
| `DLIGHTRAG_JWT_VERIFICATION_KEY` | `DLIGHTRAG_ACCESS__JWT_VERIFICATION_KEY` |
| `DLIGHTRAG_API_HOST` | `DLIGHTRAG_INTERFACES__API__HOST` |
| `DLIGHTRAG_MCP_TRANSPORT` | `DLIGHTRAG_INTERFACES__MCP__TRANSPORT` |
| `DLIGHTRAG_LANGFUSE_SECRET_KEY` | `DLIGHTRAG_OBSERVABILITY__LANGFUSE_SECRET_KEY` |
| `DLIGHTRAG_BLOB_CONNECTION_STRING` | `DLIGHTRAG_CORPUS__SOURCES__BLOB_CONNECTION_STRING` |

The SDK client variables `DLIGHTRAG_API_URL` and `DLIGHTRAG_API_TOKEN` remain
valid for remote client configuration. They are auxiliary client inputs, not
flat aliases for server listener or authentication settings. Use
`DLIGHTRAG_ACCESS__API_TOKEN` to configure the server.

Standard provider-owned variables such as `AWS_*` retain their provider-defined
meaning. Use typed `DLIGHTRAG_CORPUS__SIDECARS__...` values for parser sidecar
configuration; DlightRAG projects them into the raw LightRAG process variables.

## 5. Update Programmatic Configuration

`DlightragConfig` accepts the same eight sections as YAML. Construct canonical
settings from their owning modules and pass them as sections. Do not mutate a
loaded configuration or pass removed flat keyword arguments.

```python
from dlightrag.ai.settings import EmbeddingSettings, ModelsSettings
from dlightrag.config import DeploymentSettings, DlightragConfig

config = DlightragConfig(
    deployment=DeploymentSettings(workspace="finance"),
    models=ModelsSettings(
        embedding=EmbeddingSettings(
            provider="voyage",
            model="voyage-multimodal-3.5",
        )
    ),
)
```

The canonical object graph is frozen, including collection-valued settings.
Create a replacement section or reload configuration when a value must change.
Runtime code should accept the narrow owning section rather than the whole root
configuration whenever possible.

## 6. Validate Before Starting Services

Run validation in the same environment and working directory as the service so
it sees the real `.env` and `config.yaml`:

```bash
python -c "from dlightrag.config import load_config; load_config(); print('configuration valid')"
```

Then verify that stale imports and distributions are absent:

```bash
python -c "import dlightrag.ai, dlightrag.agent, dlightrag.rag, dlightrag_memory"
python -m pip show dlightrag dlightrag-memory
python -m pip show dlightrag-ai dlightrag-agent-core dlightrag-rag-core
```

The final command should report that the three old distributions are not found.
In the source repository, also use the checked two-wheel verifier:

```bash
rm -rf dist
uv build --all-packages --out-dir dist
uv run python scripts/verify_workspace_wheels.py --dist dist --smoke-installed
```

Treat any unknown-field or unknown-`DLIGHTRAG_` error as a migration defect.
Remove or relocate the old input; do not suppress the validation.

## 7. Roll Out

1. Deploy the matching Memory and root 2.0 artifacts together.
2. Start one writer process against the existing PostgreSQL endpoint and the
   existing `deployment.working_dir`.
3. Verify liveness, readiness, workspace listing, retrieval, one durable answer,
   and one ingestion operation appropriate for the environment.
4. Start additional writers, readers, API, and MCP processes only after the
   first process is healthy.
5. Confirm that every process received the same canonical model, storage,
   access, and interface sections.

If rollback is required, stop all 2.0 processes and restore the complete 1.x
application plus its preserved 1.x configuration. The consolidation itself does
not require reversing a database or volume-name migration.

## Related Documentation

- [Changelog](../CHANGELOG.md)
- [Configuration](configuration.md)
- [Architecture](architecture.md)
- [Security](security.md)
- [PostgreSQL](postgresql.md)
- [Operations](operations.md)
