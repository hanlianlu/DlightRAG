# Architecture Reorganization Plan

This document records the accepted application-engine-adapters reorganization plan. Canonical architecture documents and diagrams still describe the current tree. Update those documents in the same milestone that makes their claims true.

Do not start source movement from this file until the user authorizes a milestone. This file records target owners, file mapping, order, and gates.

## Target call path

```text
HTTP / MCP
  -> Application (use cases, access, config, lifecycle, caller contracts)
     -> Engine (AI, Agent, Runtime, RAG, Answer)

Offline rebuild commands (installed, writers stopped)
  -> Engine RAG + PostgreSQL rebuild functions

PostgreSQL, Observability, outbound MCP
  implement Application and/or Engine ports; they are not inbound callers.

create_application() wires the graph once, then leaves the request path.
CLI and evaluation reuse the HTTP adapter's internal async client.
External callers use REST or in-process Application. There is no public SDK.
```

## Target tree

```text
src/dlightrag/
  __init__.py                      # Application, DlightragConfig, create_application, __version__
  _compose.py                      # private composition; not a public zone

  application/
    __init__.py                    # small Application facade
    application.py                 # Application object, start/close, capability accessors
    health.py
    config/                        # loading + existing configuration sections; keys unchanged
    access/
    settings/                      # config -> runtime projections (today's model_settings.py)
    answer_runs/                   # today's services.answers
    corpus_admin/                  # today's services.corpora
    retrieval/                     # today's services.retrieval + adapters.retrieval projection
    memory/                        # product gate over dlightrag-memory
    web_conversations/             # durable conversation contracts and lifecycle

  engine/
    ai/                            # move of src/dlightrag/ai
    agent/                         # move of src/dlightrag/agent
    runtime/                       # move of src/dlightrag/runtime
    rag/                           # move of src/dlightrag/rag; internal workspace/lightrag/corpus/retrieval
    answer/                        # move of src/dlightrag/answer; execution/orchestration/fast/research

  adapters/
    http/                          # REST + browser + shared streaming + internal client + static assets
    mcp/                           # inbound server + outbound tool client
    postgres/                      # core, corpus, answer, web
    observability/                 # Langfuse adapter

packages/memory/                   # unchanged independent distribution
frontend/                          # unchanged TypeScript source
```

Root facade exports only `Application`, `DlightragConfig`, `create_application`, and `__version__`. Engine does not re-export a barrel of types at `engine/__init__.py`. Adapters do not re-export storage implementations at the adapters root.

## Discipline

- Depth is leverage at a small interface, not line count.
- Callers and tests cross the same facade. Internal protocols stay private.
- One concrete adapter class may implement several narrow owner interfaces when transactions require it.
- Underscore names mark mechanism that must not be imported across zones. Domain modules keep ordinary names.
- Same-milestone deletion of old paths. No shims, empty tombstone packages, or dual trees.
- REST paths, config keys, environment variables, PostgreSQL names, and product behavior stay unchanged.
- Tests stay in unit / integration / e2e. Rearrange a test file only when ownership or discovery actually improves.
- AI, Access-as-policy, Runtime, and Observability have no ownership inversion to fix beyond the zone move.

## File mapping

Owners: **move**, **split**, **merge**, **delete**, **keep**, **internalize**.

### Root

| Current | Target | Action |
|---|---|---|
| `__init__.py` | `__init__.py` | Keep as tiny facade; keep `__version__`; export `create_application` instead of implying Application constructs adapters |
| `application.py` | `application/application.py` plus `_compose.py` | Split lifecycle/facade from private composition |
| `config.py` | `application/config/` | Split loading from configuration sections; preserve `DlightragConfig` and env keys |
| `health.py` | `application/health.py` | Move |
| `model_settings.py` | `application/settings/` | Move and rename by capability projection, not "model" |

### Access and services

| Current | Target | Action |
|---|---|---|
| `access/*` | `application/access/` | Move |
| `services/answers.py` | `application/answer_runs/` | Split facade from private protocols. Own request, result, artifact, steer/follow-up/fork, and error contracts that HTTP/MCP call |
| `services/corpora.py` | `application/corpus_admin/` | Split facade from ingest contracts. Do not add rebuild use cases; rebuild is offline and exclusive |
| `services/retrieval.py` | `application/retrieval/` | Move the caller-awaited use case and its request/result/error contracts |
| `services/retrieval.py` `RetrievalPlannerRuntime` | `rag/retrieval/` in M1, then `engine/rag/retrieval/` in M3 | Park beside today's RAG retrieval before the Engine RAG move. Application retrieval must not keep this AI/RAG lifecycle object |
| `services/memory.py` | `application/memory/` | Move product gate and Memory errors transports handle |
| `services/errors.py` | distributed | Delete barrel. `StorageSchemaError` -> Application errors. `UnsafeUploadNameError` -> corpus_admin. Transports import only Application errors; Application may wrap Engine identities |
| `services/__init__.py` | — | Delete |
| `adapters/retrieval.py` | `application/retrieval/` private | Internalize Answer projection reuse; not an external adapter |
| `web/conversation_models.py` durable records | `application/web_conversations/` in M1 | Move `LinkedTurn`, `ConversationSnapshot`, store conflicts/unavailable/schema errors, and the store protocol. Browser Pydantic types stay in `web/` until M6 |
| `web/conversations.py` lifecycle | `application/web_conversations/` in M1 | Move persistence lifecycle. Browser URL/presentation helpers stay in `web/` until M6 |

### Agent, AI, Runtime

| Current | Target | Action |
|---|---|---|
| `ai/*` | `engine/ai/` | Move; keep provider subtree |
| `agent/*` except broad facades | `engine/agent/` | Move |
| `agent/tools/__init__.py` | contracts-only facade | Narrow; filesystem tools remain internal modules |
| `agent/environment/__init__.py` | ports-only facade | Narrow; `LocalExecutionEnvironment` stays internal |
| `agent/session/memory.py` | keep name inside Agent | Keep; it is the in-memory repository adapter, not Profile Memory |
| `runtime/*` | `engine/runtime/` | Move |

Do not move Agent local filesystem tools, local execution, or in-memory session repository into product adapters.

### RAG

| Current | Target | Action |
|---|---|---|
| `rag/workspace_rag.py`, `pool.py`, `lifecycle.py`, `workspaces.py` | `engine/rag/workspace/` | Split facade from collaborators; keep one runtime/store bundle |
| `rag/lightrag_*`, `_lightrag_patches.py` | `engine/rag/lightrag/` | Internalize framework bridge; no public port until a second runtime exists |
| `rag/ingestion/*`, `sourcing/*`, `source_download.py`, `reset.py`, `visual_assets.py` | `engine/rag/corpus/` | Move write/source lifecycle |
| `rag/retrieval/*`, `federation.py`, `rerank.py` | `engine/rag/retrieval/` | Move query planning and result formation |
| `rag/ports/*` | owned by workspace/corpus/retrieval | Split; delete the technical Ports bucket |
| `rag/contracts.py`, `settings.py` | owning RAG submodule or workspace | Move to the owner that defines the values |

### Answer — Application-owned caller contracts

HTTP and MCP import these only through Application. Do not send them to Engine with the remaining-answer catch-all.

| Current | Target | Action |
|---|---|---|
| `answer/client_contracts.py` | `application/answer_runs/` | Move caller JSON-safe dump and shared payload helpers |
| `answer/errors.py` | `application/answer_runs/` or `application/` errors | Move caller-facing Answer and Memory-unavailable errors |
| `answer/runs/execution.py` request/input types | `application/answer_runs/` | Move acceptance input contracts |
| `answer/runs/results.py` client projections | `application/answer_runs/` | Move transport result projection. Engine may keep durable snapshot codecs privately |
| `answer/runs/snapshots.py` | `engine/answer/` | Keep Engine execution snapshots unless a type is already part of the Application result facade |
| `answer/sources.py` | `application/answer_runs/` | Move source/download link projection used by REST and browser |
| `answer/prepared_input.py`, `mode.py`, `routing.py` | `application/answer_runs/` | Move acceptance/routing policy types |
| `answer/capabilities.py`, `capability.py` | `application/answer_runs/` | Move capability summaries transports already return |
| `answer/citations/schemas.py` and payload builders used by transports | `application/answer_runs/` | Move caller citation/result schemas. Engine keeps citation parsing/finalization |

### Answer — Engine execution

| Current | Target | Action |
|---|---|---|
| `answer/agent/orchestrator.py` | `engine/answer/orchestration/` | Move mode-neutral routing, resource bind, shared finalization |
| `answer/agent/context.py`, `compaction.py` | `engine/answer/research/` | Move Research-only assembly |
| `answer/agent/__init__.py` | — | Delete |
| `answer/executor.py` | `engine/answer/execution/` plus Fast/Research internals | Split. Facade keeps plan-for-acceptance and execute-accepted-run. Fast boundaries/session host -> Fast. Research effects/child drive/controls -> Research |
| `answer/session_host.py` | `engine/answer/fast/` | Move |
| `answer/context.py` | `engine/answer/` with a duty-specific name | Keep distinct from Research control context |
| remaining Engine-only `answer/*` | `engine/answer/` | Move prompts, evidence, resources, tools, publication, synthesizer, media, history, workspace, model runtime |

`research_history_input_measure` must not be imported by Application from an Engine implementation module. Expose the acceptance measurement through the Answer execution facade. Application answer_runs owns the types HTTP and MCP import.

### HTTP, Web, API, SDK

| Current | Target | Action |
|---|---|---|
| `api/server.py` lifespan/factory | `adapters/http/` server composition | Move; create Application via `create_application` |
| `api/routes/*` | `adapters/http/rest/` | Move |
| `api/models.py`, `payloads.py`, `auth.py`, `middleware.py` | `adapters/http/rest/` | Move transport-only types |
| `api/answer_stream.py` | `adapters/http/streaming/` | Move shared SSE/cursor/keepalive |
| `web/routes/*` | `adapters/http/browser/` | Move |
| `web/presentation.py`, `markdown.py`, `safe_html.py`, `app_shell.py`, `static_files.py`, `sse.py`, `events.py`, `answer_events.py` | `adapters/http/browser/` | Move browser projection |
| `web/auth.py`, `edge_identity.py`, `deps.py` | `adapters/http/browser/` | Move |
| `web/attachment_*`, `file_models.py`, `requests.py` | `adapters/http/browser/` | Move transport parsing |
| `web/conversation_models.py` browser Pydantic types | `adapters/http/browser/` in M6 | Durable records already moved in M1 |
| `web/conversations.py` browser coordination | `adapters/http/browser/` in M6 | Persistence lifecycle already moved in M1 |
| `web/static/` | `adapters/http/browser/static/` | Move runtime assets; update Vite `outDir` and Hatch artifacts |
| `frontend/` | `frontend/` | Keep TypeScript source at repo root |
| `sdk/client.py` async client | `adapters/http/client/` | Internalize |
| `sdk/http.py`, `attachments.py`, `requests.py` | `adapters/http/client/` | Internalize helpers used by CLI/eval |
| `sdk/client.py` `SyncAnswerRunClient` | — | Delete |
| `sdk/__init__.py`, `api/__init__.py`, `web/__init__.py` | — | Delete after moves |

### MCP, observability, rebuild commands, PostgreSQL

| Current | Target | Action |
|---|---|---|
| `mcp/server.py` | `adapters/mcp/` plus `tools/{answer_runs,retrieval,memory,corpus_admin}.py` | Split handlers; one server/lifespan. Workspace and capability tools live with corpus_admin |
| `mcp/auth.py`, `cli.py`, `contracts.py` | `adapters/mcp/` | Move |
| `adapters/mcp_tools.py` | `adapters/mcp/outbound.py` | Move outbound client beside inbound server |
| `observability/*` | `adapters/observability/` | Move |
| `maintenance/rebuild_bm25.py` | `engine/rag/corpus/` rebuild CLI plus existing PostgreSQL BM25 rebuild | Delete the Maintenance package. Keep the `dlightrag-rebuild-bm25` console script name. Implementation calls PostgreSQL rebuild, not Application |
| `maintenance/rebuild_vdb.py` | `engine/rag/corpus/` rebuild CLI | Keep the `dlightrag-rebuild-vdb` console script name. Implementation wraps LightRAG vector rebuild plus DlightRAG BM25/sidecar post-steps. Writers must be stopped. Not a Compose service or Makefile target |
| `maintenance/__init__.py` | — | Delete |
| `adapters/postgres/_pool.py`, `_operations.py`, `_migrations.py`, `_locks.py`, `_errors.py`, `_version.py`, `identifiers.py` | `adapters/postgres/core/` | Move shared SQL mechanism |
| `adapters/postgres/corpus.py`, `corpus_*.py`, `_corpus_schema.py`, `ingest_jobs.py`, `file_panel.py`, `pg_metadata_index.py`, `lightrag_*.py`, `workspaces.py` | `adapters/postgres/corpus/` | Group |
| `adapters/postgres/answer_runs.py` | `adapters/postgres/answer/` | Keep one public store; split private schema/codec/run/event/artifact/control/lease/retention modules |
| `adapters/postgres/session_repository.py` | `adapters/postgres/answer/` | Keep one deep session adapter; extract private SQL/codecs only if needed |
| `adapters/postgres/workspace.py`, `memory_settings.py` | `adapters/postgres/answer/` | Answer-owned activation and workspace inventory |
| `adapters/postgres/web_conversations.py` | `adapters/postgres/web/` | May import postgres.answer one way for atomic create; reverse forbidden |
| `adapters/__init__.py` | empty or delete | No barrel of implementations |

`PGAnswerRunStore` remains one transaction owner. `PGAgentSessionRepository` remains `{load, transact}`. Do not split those public classes per table.

### Independent package

| Current | Target | Action |
|---|---|---|
| `packages/memory/**` | `packages/memory/**` | Keep independent distribution |

## Milestones

Each milestone is one reviewable, independently shippable source change. Old paths die in the same milestone. Docs and SVGs that the milestone falsifies are updated in the same milestone.

### M1 — Application zone

Move Config, Access, Application, health, settings projections, and services use cases under Application. Introduce `_compose.py` and `create_application`.

Also in this milestone, because `dlightrag.services` dies here:

- Move caller-facing Answer request/result/error/client contracts into `application/answer_runs/` even though Engine Answer files stay put until M4.
- Extract Web Conversation durable records and persistence lifecycle into `application/web_conversations/`. Leave browser presentation and URL helpers in `web/` until M6.
- Move `RetrievalPlannerRuntime` into today's `rag/retrieval/` (not Application). M3 relocates it with RAG.

Existing API, Web, and MCP may keep some deep Answer/RAG imports until M6 only where this milestone did not already re-home the contract. Do not add new transport-to-Engine imports.

Exit: `dlightrag.services` gone. Application owns answer_runs, corpus_admin, retrieval, memory, and web_conversations caller contracts. Root can create a started Application. Browser leftovers remain under `web/` by name only.

### M2 — Engine foundations

Move AI, Agent, Runtime under `engine/`. Narrow Agent tool/environment facades. Update import-linter, wheel smoke, and Agent facade assertions to new modules.

Exit: no top-level `ai`, `agent`, or `runtime` packages.

### M3 — Engine RAG

Move RAG under `engine/rag/` with workspace, lightrag, corpus, retrieval internals, including the planner runtime parked in M1. Split ports to owners. Keep one Workspace runtime. Relocate rebuild CLIs beside corpus rebuild functions and retarget the two existing console scripts.

Exit: no top-level `rag` package. Corpus and Retrieval views are narrow; LightRAG remains internal. `dlightrag-rebuild-bm25` and `dlightrag-rebuild-vdb` still work and no longer live under `maintenance/`.

### M4 — Engine Answer

Move remaining Engine Answer implementation under `engine/answer/`. Create execution, orchestration, fast, and research internals. Delete `answer.agent`. Split `answer/executor.py`. Application acceptance already owns caller contracts from M1 and talks to the execution facade, not Research files.

Exit: no top-level `answer` package and no second Agent folder. Transports still must not import `engine.answer`.

### M5 — Backend adapters

Rehome PostgreSQL, Observability, and outbound MCP client under `adapters/`. Group PostgreSQL by core/corpus/answer/web. Split `answer_runs.py` privately. Keep public store and session repository depths. Rebuild CLIs already moved in M3; this milestone only fails if `maintenance/` still exists.

Exit: no top-level `observability` or `maintenance`. PostgreSQL internals are grouped. Adapter barrels do not export implementations. Rebuild remains installed commands, not an adapter zone.

### M6 — Inbound adapters and internal HTTP client

Merge API and Web into `adapters/http/`. Move MCP server and split tools into answer_runs, retrieval, memory, and corpus_admin handlers. Move Vite output and vendored static files. Internalize async HTTP client. Delete public SDK and sync client. Update remaining console scripts to new module paths without renaming the commands.

Exit: no top-level `api`, `web`, `mcp`, or `sdk`. HTTP and MCP import Application, not Engine.

### M7 — Final architecture closure

No leftover source moves. If an old path still exists, the owning milestone failed and must be reopened. M7 tightens import-linter and architecture tests to the three-zone rules, audits README, architecture, interfaces, configuration, operations, durable-run, retrieval, PostgreSQL, security docs, and the four SVGs, and runs Architecture/Oracle review.

Exit: completion criteria below are all true, and the milestone diff contains no package moves.

## Documentation policy

This plan describes the accepted future. `docs/architecture.md`, `docs/interfaces.md`, `docs/configuration.md`, `docs/domain-language.md` only as needed for renamed product terms, and the SVG figures describe code that already exists.

Per milestone, update every canonical sentence or diagram the milestone makes false. Do not draw the unfinished tree as current. M7 is the consistency audit, not the first documentation pass.

Domain language already has Application, Answer Service, Retrieval, Corpus Administration, Web Conversation, Agent Session, and Full Development Reset. Module directories use those terms (`answer_runs`, `corpus_admin`, `web_conversations`) without inventing Bootstrap, Operator, SDK, or Core as product zones. `application/config/` holds configuration sections, not an Operator product.

## Gates

During a milestone: targeted unit tests for moved owners, import-linter on the new paths, and Ruff/Pyright as soon as imports compile.

Before a milestone commit:

- `make ci`
- relevant PostgreSQL integration tests
- no staged leftover files from other work

At milestone end:

- `make ci-full`
- Bandit `-lll`
- ShellCheck
- frontend CI when HTTP/static/frontend packaging changed; otherwise still run it at M6 and M7
- wheel smoke, including installed import identity for remaining public facades
- isolated PG18 e2e

Review:

- Standards + Spec axes before each milestone commit
- Architecture/Oracle at M7
- Grok 4.6 for architecture/Standards/Oracle; DeepSeek v4-pro for Spec

## Data

No schema migration and no compatibility tables. Python layout is not a storage change.

If local development data blocks validation, preview a full development reset, optionally force-disconnect local sessions, and reset only the local DlightRAG development database and working directory. Do not reset production or foreign environments. Prefer isolated temporary databases for PG18 e2e.

Source rollback is `git revert` of the milestone.

## Unchanged on purpose

- REST paths and payloads
- `DLIGHTRAG_*` configuration keys, including `answer.agent`
- PostgreSQL table and migration scope names, including `dlightrag_agent_*`
- `dlightrag-api`, `dlightrag-mcp`, `dlightrag-rebuild-bm25`, `dlightrag-rebuild-vdb` command names
- package version `2.0.0` until a separate release decision
- `packages/memory` independence
- `frontend/` as the TypeScript source tree
- Fast and Research product behavior

## Completion criteria

The reorganization is done only when all of the following hold:

- Application, Engine, and Adapters are the only visible product code zones besides the independent Memory distribution.
- The root facade exports only Application, DlightragConfig, `create_application`, and `__version__`.
- HTTP and MCP call Application and do not deep-import Engine.
- Engine does not import Application or Adapters.
- PostgreSQL, HTTP, MCP, and Observability are not first-class root packages.
- There is no Maintenance zone. Rebuild stays two installed commands.
- `services`, `api`, `web`, and `sdk` packages are gone.
- No compatibility shims, old-path docs, or wheel assertions for deleted modules.
- Product behavior, REST paths, config keys, and persistence semantics are unchanged.
- Canonical docs and SVGs describe the landed tree only.
