# Changelog

This file records user-visible changes to DlightRAG. Migration guides preserve
the package/configuration contract for each breaking major.

## 3.0.0 — Unreleased

Agent 3.0 replaces the old Research orchestration and journal contract with a
product-neutral event loop, run-local ToolRegistry, typed Session journal with a
parent-linked derived view, Context Contributions, model-aware reserves, replay-stable compaction and
controls, and deterministic terminal-answer finalization. Fast remains a
lightweight durable invocation but now shares context, model, Evidence,
citation, Profile Memory, and usage infrastructure without an Agent Session.

Research gains foreground parallel Child Sessions with roster/status/wait/cancel,
context/model/tool selection, inclusive usage, and transactional parent Evidence
adoption. Execution modes are exactly `disabled`, `trust`, and `sandbox`; sandbox
requires a trusted adapter. Progressive global/workspace Skills, three trusted
Python extension seams, and thin allowlisted outbound MCP tools are available.
Embedding transport is now protocol-specific: official OpenAI/Azure OpenAI v1,
conservative OpenAI-compatible text servers, Voyage, Gemini Embedding 2, Jina
v4 fused multimodal, Cohere Embed v4, and Azure Cohere v4 have dedicated
adapters. Provider-owned URLs, strict indexed OpenAI responses, local token and
batch budgets, ordered auto-splitting, float/vector validation, usage telemetry,
and bounded Retry-After-aware retries replace the former universal payload.
Native Ollama embedding and generic OpenAI-compatible image extensions are
removed. Query/document task semantics are mandatory for known retrieval
models, and multimodal startup probes verify both image-query and native fused
document paths.

Profile Memory adds stable proposal/commit, replay-stable recall, local
single-user identity, timeout fallback, and idempotent tombstone forget. REST,
inbound MCP, Python, and Web expose their applicable start/status/steer/follow-up/
cancel/resume/fork, transcript, child, usage, Evidence, and lineage projections.
Both distributions and the frontend are version `3.0.0`. Pre-3.0 Agent runs and
schemas are intentionally incompatible; see
[Migrating to DlightRAG 3.0](docs/migration-3.0.md).

## 2.0.0 — Historical

DlightRAG 2.0 is a hard-breaking package and configuration consolidation. It
replaces four tightly coupled internal distributions with one root product
distribution, keeps Owner Profile Memory independently installable, and makes
one immutable eight-section configuration the only server configuration
contract.

### Highlights

- The `dlightrag` wheel now contains the AI, Agent, and RAG implementation under
  `dlightrag.ai`, `dlightrag.agent`, and `dlightrag.rag`.
- `dlightrag-memory` remains a separate, host-neutral distribution and retains
  the `dlightrag_memory` import package and MCP entry point.
- Chat, embedding, rerank, parser, source, and provider integrations are
  batteries-included direct dependencies of `dlightrag`; there are no provider
  extras to select.
- Configuration now has exactly eight top-level sections: `deployment`,
  `storage`, `models`, `corpus`, `answer`, `access`, `interfaces`, and
  `observability`.
- All canonical configuration models are strict and frozen. Runtime components
  consume their owning section instead of copied snapshots of root settings.
- Configuration precedence is constructor arguments, environment variables,
  `.env`, `config.yaml`, then code defaults.
- Docling is the default durable parser. The checked-in Docker-first config
  consumes an independently operated host service on port 5001 rather than
  enabling the optional Compose Docling profile; MinerU remains opt-in.
- Root and Memory wheels are built and smoke-tested as isolated installations,
  including their packaged data and type markers.

### Breaking changes

#### Distribution and import layout

The following distributions no longer exist:

- `dlightrag-ai`
- `dlightrag-agent-core`
- `dlightrag-rag-core`

Their import roots have no compatibility shims. Replace `dlightrag_ai`,
`dlightrag_agent`, and `dlightrag_rag` imports with the corresponding
`dlightrag.ai`, `dlightrag.agent`, and `dlightrag.rag` paths. Applications that
used one of the former internal wheels directly must depend on `dlightrag`
instead.

#### Configuration

The 1.x flat and partially nested YAML keys, environment variables, and
constructor fields are not aliases in 2.0. Unknown `DLIGHTRAG_` server variables
and unknown YAML or constructor fields fail configuration loading. In
particular:

- `llm`, `embedding`, and `rerank` move under `models`;
- PostgreSQL and LightRAG storage fields move under `storage`;
- parser, ingestion, retrieval, source, and visual-asset fields move under
  `corpus`;
- generation, durable-runtime, agent, citation, conversation, and Web-search
  fields move under `answer`;
- authentication and workspace authorization move under `access`;
- API and MCP listeners move under `interfaces`;
- logging and Langfuse fields move under `observability`.

Nested environment variables mirror the YAML path with double-underscore
separators. For example, the server PostgreSQL password is now
`DLIGHTRAG_STORAGE__POSTGRES__PASSWORD`.

`DLIGHTRAG_API_URL` and `DLIGHTRAG_API_TOKEN` remain client-only SDK variables;
they do not restore the removed flat server configuration contract. The server
bearer token is `DLIGHTRAG_ACCESS__API_TOKEN`.

Canonical settings instances cannot be mutated after construction. Code that
constructed `DlightragConfig` with 1.x fields or modified settings in place must
construct the matching 2.0 section instead.

See [Migrating to DlightRAG 2.0](docs/migration-2.0.md) for package commands,
path mappings, environment examples, validation, rollout, and rollback.

### Internal simplification

- Removed the one-caller Agent Loop wrapper and unreachable provider/all-tool
  termination states; Answer orchestration now owns its research loop.
- Removed unused federation options and made federation explicitly
  multi-workspace.
- Replaced one-adapter protocol/factory wrappers for schema lookup, projection,
  corpus construction, execution environment, and workspace pooling with their
  real functions or concrete owner.
- Consolidated LightRAG contract verification into the PostgreSQL adapter,
  centralized model endpoint fingerprints, and removed duplicate lifecycle
  helpers.
- Unified corpus error and download-target identities across RAG, services, and
  transports instead of translating or reconstructing them.
- Removed dead retrieval state, production-only test adapters, unused package
  barrels, and obsolete exports.

These are internal API removals. Consumers should use the documented REST, Web,
MCP, SDK, application-service, and canonical settings surfaces rather than the
removed composition helpers.

### Persistence and operations

- No PostgreSQL business-data migration is introduced by the consolidation.
- Existing SQL object names, including names beginning with
  `dlightrag_agent_`, remain unchanged.
- Existing volume names and on-disk workspace layout remain unchanged.
- Operators must preserve the same `deployment.working_dir` and PostgreSQL
  endpoint when upgrading an existing deployment.
- Root depends on `dlightrag-memory==2.0.0`, so publish or mirror the Memory
  wheel before the root wheel.

### Verification performed

- Ruff formatting and linting
- strict Pyright checking
- Ruff security linting with the Bandit ruleset
- all 22 import-linter architecture contracts
- 3,034 unit and PostgreSQL integration tests
- root and Memory source-distribution and wheel builds
- isolated installed-wheel smoke tests for both distributions
