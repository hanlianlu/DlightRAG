# DlightRAG Answer Layer Architecture Upgrade

**Date**: 2026-03-14
**Status**: Approved (pending implementation)

## Problem Statement

The current architecture has three issues:

1. **Answer generation at the wrong layer**: `RAGService` and its backends (`VisualRetriever`, `RetrievalEngine`, `UnifiedRepresentEngine`) each implement answer generation. `RAGServiceManager` federates by calling each workspace's `aanswer()` independently, then naively concatenates answers (`"\n\n---\n\n"`) and references (no re-numbering). This makes round-robin meaningless for answers and produces broken citations.

2. **FREETEXT_REMINDER in user prompt**: When `structured=False`, the freetext references reminder is appended to the user prompt. The `structured=True` path puts its equivalent (`_ANSWER_STRUCTURED_SUFFIX`) in the system prompt. This inconsistency should be unified — both modes should keep format instructions in the system prompt.

3. **Logger naming**: All answer and reference logs go to a single `dlightrag.references` logger regardless of functional domain. After the architecture change, log point locations also need to be reorganized.

## Solution Overview

1. **New `AnswerEngine` module** (`core/answer.py`): Centralized answer generation that receives merged retrieval contexts and produces a single answer with proper citations.
2. **RAGServiceManager orchestration change**: `aanswer()` becomes `aretrieve()` → `AnswerEngine.generate()`. Federation only happens at the retrieval level.
3. **RAGService and backends become retrieval-only**: Remove all `aanswer()`/`aanswer_stream()` methods.
4. **Remove conversation_history from backend**: Web UI handles query rewriting; backend becomes stateless.
5. **Workspace compatibility filter**: New `dlightrag_workspace_meta` table to track embedding model per workspace.
6. **Prompt and logging cleanup**: FREETEXT_REMINDER moves to system prompt; loggers split by function.

## Architecture: Before and After

### Before

```
API/MCP/Web
    ↓
RAGServiceManager
    ├── single ws: RAGService.aanswer() → Backend.aanswer()
    └── multi ws:  federated_answer() → parallel Backend.aanswer() → concat answers
```

### After

```
API/MCP/Web
    ↓
RAGServiceManager
    ├── aretrieve() → federated_retrieve() → merge contexts
    ├── aanswer()   → aretrieve() → AnswerEngine.generate()
    └── aanswer_stream() → aretrieve() → AnswerEngine.generate_stream()
```

## Detailed Design

### 1. AnswerEngine (`src/dlightrag/core/answer.py`) — NEW

```python
class AnswerEngine:
    """Unified answer generation from retrieval contexts.

    Receives merged RetrievalResult (from any number of workspaces),
    builds prompts, calls LLM/VLM, parses citations, returns answer.
    Mode-agnostic: detects image_data in chunks to choose VLM vs LLM path.
    """

    def __init__(
        self,
        config: DlightragConfig,
        vision_model_func: Callable | None = None,
        llm_model_func: Callable | None = None,
    ) -> None: ...

    async def generate(
        self,
        query: str,
        contexts: RetrievalContexts,
    ) -> RetrievalResult:
        """Non-streaming: contexts → answer + references."""

    async def generate_stream(
        self,
        query: str,
        contexts: RetrievalContexts,
    ) -> tuple[RetrievalContexts, AsyncIterator[str]]:
        """Streaming: contexts → (contexts, token_iterator)."""
```

#### Mode-agnostic routing

AnswerEngine does not know about caption/unified mode. It inspects chunks:

| Chunks contain `image_data`? | Path |
|-----|------|
| Yes | VLM multimodal call (`_build_vlm_messages` with inline base64 images) |
| No | LLM text-only call |
| Mixed | VLM path (any image triggers multimodal) |

#### Internal flow (generate)

1. Determine `structured` via `provider_supports_structured_vision(provider)`
2. Build system prompt via `get_answer_system_prompt(structured)`
3. Build user prompt: KG context + reference list + question (no conversation history, no FREETEXT_REMINDER)
4. Build citation indexer from contexts
5. Call VLM or LLM based on image_data presence
6. Parse response (structured JSON or freetext)
7. Log via `dlightrag.answer` logger
8. Return `RetrievalResult(answer, contexts, references)`

#### Internal flow (generate_stream)

Same as generate steps 1-4, then:

5. Call VLM/LLM with `stream=True`
6. If structured: wrap with `AnswerStream(token_iterator, StreamingAnswerParser())`
7. Return `(contexts, token_iterator)`

#### Methods migrated from VisualRetriever

- `_build_vlm_messages(system_prompt, user_prompt, chunks)` → `@staticmethod`, unchanged
- `_build_citation_indexer(contexts)` → `@staticmethod`, unchanged
- `_format_kg_context(contexts)` → convert to `@staticmethod` (method body does not use `self`)
- `_build_user_prompt(query, contexts)` → new, composes KG context + ref list + question

#### multimodal_content handling

`multimodal_content` (image uploads for visual similarity search) continues to flow through `**kwargs` from `RAGServiceManager.aanswer()` → `self.aretrieve()` → backend. AnswerEngine does not receive `multimodal_content` — it only sees the already-retrieved contexts (which include image_data from visual chunks). The retrieval layer handles multimodal query enhancement as before.

### 2. RAGServiceManager Changes (`src/dlightrag/core/servicemanager.py`)

#### New aanswer flow

```python
class RAGServiceManager:
    def __init__(self, config):
        # ...existing...
        self._answer_engine: AnswerEngine | None = None

    def _get_answer_engine(self) -> AnswerEngine:
        """Lazy-create AnswerEngine from global config."""
        if self._answer_engine is None:
            self._answer_engine = AnswerEngine(
                config=self._config,
                vision_model_func=get_vision_model_func(self._config),
                llm_model_func=get_llm_model_func(self._config),
            )
        return self._answer_engine

    async def aanswer(self, query, *, workspace=None, workspaces=None, **kwargs):
        ws_list = workspaces or [workspace or self._config.workspace]
        retrieval = await self.aretrieve(query, workspaces=ws_list, **kwargs)
        engine = self._get_answer_engine()
        return await engine.generate(query, retrieval.contexts)

    async def aanswer_stream(self, query, *, workspace=None, workspaces=None, **kwargs):
        ws_list = workspaces or [workspace or self._config.workspace]
        retrieval = await self.aretrieve(query, workspaces=ws_list, **kwargs)
        engine = self._get_answer_engine()
        return await engine.generate_stream(query, retrieval.contexts)
```

New public method for web UI layer:

```python
def get_llm_func(self) -> Callable:
    """Return the global LLM model function (for web UI query rewriting etc.)."""
    return get_llm_model_func(self._config)
```

Key changes:
- No separate single-ws vs multi-ws answer path — always `aretrieve()` → `AnswerEngine`
- `aanswer_stream` now supports multi-workspace federation (previously single-ws only)
- `conversation_history` removed from all signatures
- `federated_answer` import removed
- New `get_llm_func()` public method exposes LLM function without leaking `_config`

#### Workspace compatibility filter

```python
async def list_workspaces(self, *, compatible_only: bool = False) -> list[str]:
    all_ws = await self._list_all_workspaces()
    if not compatible_only:
        return all_ws
    return await self._filter_compatible(all_ws)

async def _filter_compatible(self, workspaces: list[str]) -> list[str]:
    compatible = []
    for ws in workspaces:
        meta = await self._get_workspace_metadata(ws)
        if meta is None or meta.get("embedding_model") == self._config.embedding_model:
            compatible.append(ws)
    return compatible
```

New PostgreSQL table:

```sql
CREATE TABLE IF NOT EXISTS dlightrag_workspace_meta (
    workspace TEXT PRIMARY KEY,
    embedding_model TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

Written on first successful ingestion per workspace (upsert in `RAGService.aingest()`).

Default behavior unchanged: `aanswer(query)` uses only the default workspace. Frontend calls `list_workspaces(compatible_only=True)` to get valid options for multi-workspace queries.

### 3. RAGService Simplification (`src/dlightrag/core/service.py`)

**Delete**:
- `aanswer()` method
- `aanswer_stream()` method
- `_truncate_history()` method
- `truncate_conversation_history` import

**Retain unchanged**:
- `aretrieve()`, `aingest()`, `alist_ingested_files()`, `adelete_files()`
- All lifecycle methods (`initialize()`, `close()`, etc.)

**Add**: Workspace meta upsert in `aingest()` on success.

### 4. RetrievalBackend Protocol (`src/dlightrag/core/retrieval/protocols.py`)

Remove `aanswer` and `aanswer_stream` from the protocol:

```python
@runtime_checkable
class RetrievalBackend(Protocol):
    async def aretrieve(
        self, query: str, *, mode: str = "mix",
        top_k: int | None = None, chunk_top_k: int | None = None,
        **kwargs
    ) -> RetrievalResult: ...
```

### 5. Backend Simplification

#### UnifiedRepresentEngine (`src/dlightrag/unifiedrepresent/engine.py`)

- Delete `aanswer()`, `aanswer_stream()`, `_build_conversation_context()`
- `aretrieve()`: remove `conversation_history` handling
- Retain `_extract_image_bytes()` (multimodal retrieval still needs it)

#### VisualRetriever (`src/dlightrag/unifiedrepresent/retriever.py`)

- Delete `answer()`, `answer_stream()`
- `retrieve()`: remove `conversation_context` parameter
- Delete `_build_vlm_messages()`, `_build_citation_indexer()`, `_format_kg_context()` (moved to AnswerEngine)
- Retain `retrieve()`, `query_by_visual_embedding()`, reranking methods

#### RetrievalEngine (`src/dlightrag/captionrag/retrieval.py`)

- Delete `aanswer()`, `aanswer_stream()`
- Retain `aretrieve()`, `_attach_page_idx()`

### 6. Federation Changes (`src/dlightrag/core/federation.py`)

- Delete `federated_answer()` function entirely
- `merge_results()`: remove answer concatenation (lines 68-69) and references concatenation (lines 71-74). Only merge contexts (chunks, entities, relationships via round-robin).
- `federated_retrieve()`: unchanged

### 7. Prompt Changes (`src/dlightrag/unifiedrepresent/prompts.py`)

```python
def get_answer_system_prompt(structured: bool = False) -> str:
    if structured:
        return _ANSWER_CORE + _ANSWER_STRUCTURED_SUFFIX
    return _ANSWER_CORE + FREETEXT_REMINDER  # CHANGED: was just _ANSWER_CORE
```

- Delete `UNIFIED_ANSWER_SYSTEM_PROMPT` alias
- `FREETEXT_REMINDER` stays defined in this file but is no longer imported by VisualRetriever (AnswerEngine uses it indirectly via `get_answer_system_prompt`)

### 8. Logging Reorganization (`src/dlightrag/utils/logging.py`)

Logger rename:

```python
def log_answer_llm_output(location, *, structured, provider, query, ...):
    logger = logging.getLogger("dlightrag.answer")  # was "dlightrag.references"
    ...

def log_references(location, refs, **context):
    logger = logging.getLogger("dlightrag.answer")  # was "dlightrag.references"
    ...
```

Location parameter updates:

| Old | New |
|-----|-----|
| `ragservice.answer` | `answer_engine.generate` |
| `ragservice.answer_stream` | `answer_engine.generate_stream` |
| `manager.federation` | deleted |

### 9. Web UI Query Rewriting (`src/dlightrag/web/routes.py`)

New helper function:

```python
async def _rewrite_query(
    user_message: str,
    conversation_history: list[dict[str, str]],
    llm_func: Callable,
) -> str:
    """Rewrite user message into standalone query using conversation context.

    Skips LLM call if no conversation history (first turn).
    Uses last 10 turns (20 messages) for context.
    Raises on LLM failure (no fallback — if LLM is down, answer will also fail).
    """
    if not conversation_history:
        return user_message

    system = (
        "You are a query rewriter. Given a conversation history and a follow-up "
        "message, rewrite the follow-up into a standalone search query that "
        "captures the full intent. Output ONLY the rewritten query, nothing else. "
        "If the message is already self-contained, return it unchanged."
    )

    history_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in conversation_history[-20:]
    )

    user_prompt = (
        f"Conversation history:\n{history_text}\n\n"
        f"Follow-up message: {user_message}\n\n"
        f"Standalone query:"
    )

    return await llm_func(user_prompt, system_prompt=system)
```

Route change:

```python
# Before
contexts, stream = await manager.aanswer_stream(query, conversation_history=history, ...)

# After
standalone_query = await _rewrite_query(query, history, llm_func)
contexts, stream = await manager.aanswer_stream(standalone_query, ...)
```

LLM function: Obtained via a new public method `manager.get_llm_func()` (avoids accessing private `_config`). Created once at route initialization and cached.

### 10. API/MCP Interface Changes

#### REST API (`src/dlightrag/api/server.py`)

- Remove `conversation_history` from answer endpoint request schema
- Remove `conversation_history` from kwargs passed to `manager.aanswer()` / `manager.aanswer_stream()`

#### MCP Server (`src/dlightrag/mcp/server.py`)

- Remove `conversation_history` from tool parameter schema
- Remove `conversation_history` from kwargs passed to `manager.aanswer()`

## Files Changed Summary

| # | File | Action |
|---|------|--------|
| 1 | `core/answer.py` | **NEW** — AnswerEngine |
| 2 | `core/servicemanager.py` | Refactor aanswer/aanswer_stream, add workspace filter, lazy AnswerEngine, `get_llm_func()` |
| 3 | `core/federation.py` | Delete `federated_answer()`, simplify `merge_results()` |
| 4 | `core/service.py` | Delete answer methods, `_truncate_history`, add workspace meta upsert |
| 5 | `core/retrieval/protocols.py` | Remove aanswer/aanswer_stream from protocol |
| 6 | `unifiedrepresent/engine.py` | Delete answer methods, conversation_history handling |
| 7 | `unifiedrepresent/retriever.py` | Delete answer methods, migrate helpers to AnswerEngine |
| 8 | `captionrag/retrieval.py` | Delete answer methods |
| 9 | `unifiedrepresent/prompts.py` | FREETEXT_REMINDER into system prompt, delete alias |
| 10 | `utils/logging.py` | Rename logger to `dlightrag.answer` |
| 11 | `web/routes.py` | Add `_rewrite_query()`, remove conversation_history passthrough |
| 12 | `api/server.py` | Remove conversation_history from endpoints |
| 13 | `mcp/server.py` | Remove conversation_history from tool schema |

## Migration Notes

- **No data migration needed**: The new `dlightrag_workspace_meta` table is created in `_ensure_pg_schema()`. Existing workspaces get their metadata written on next ingestion. For non-PG backends (JSON, Redis, MongoDB), `_filter_compatible()` returns `meta is None` for all workspaces, treating them as compatible. This is the intended behavior — compatibility filtering is a PG-only feature until metadata storage is extended to other backends.
- **API breaking change**: `conversation_history` removed from REST API answer endpoints and MCP tool schema. API/MCP consumers must implement their own query rewriting. This is intentional — programmatic consumers (agents, API clients) are expected to send well-formed standalone queries.
- **Caption mode answer quality**: Previously used LightRAG's built-in `aquery_llm()` prompt (which has its own KG context formatting and no citation support). Now uses the unified `_ANSWER_CORE` prompt with `[n-m]` citation instructions. This is a deliberate behavioral change that unifies citation handling across modes. Requires integration testing to verify caption mode answer quality does not regress.
- **Caption mode streaming performance improvement**: Previously, `RetrievalEngine.aanswer_stream()` made two LightRAG calls (`aquery_data()` + `aquery()` with `stream=True`), effectively running the query twice. The new flow makes one retrieval call + one LLM streaming call, reducing latency.
- **Timeout consideration**: The current `asyncio.timeout(request_timeout)` wraps the entire `aanswer()` call. With the new flow (retrieve + answer as two steps), the same timeout covers both. Web UI adds a query rewriting LLM call before `aanswer()`, which is outside the manager's timeout. If rewriting is slow, users will see a longer wait. This is acceptable — if the LLM is slow for rewriting, it will also be slow for answering.
- **Frontend JS unchanged**: `citation.js` continues to send `conversation_history` in the request body. The web UI route uses it for query rewriting, then discards it. No frontend changes needed.

## Test Changes

The following test files exercise `aanswer()`, `aanswer_stream()`, or `federated_answer()` and will need updates:

| Test File | Changes Needed |
|-----------|---------------|
| `tests/unit/test_federation.py` | Remove `federated_answer` tests, keep `federated_retrieve` and `merge_results` tests (update `merge_results` assertions to exclude answer/references) |
| `tests/unit/test_service.py` | Remove `aanswer`/`aanswer_stream` delegation tests |
| `tests/unit/test_servicemanager.py` | Rewrite `aanswer`/`aanswer_stream` tests to verify retrieve → AnswerEngine flow |
| `tests/unit/test_unified_engine.py` | Remove `aanswer`/`aanswer_stream` tests |
| `tests/unit/test_retrieval_engine.py` | Remove `aanswer`/`aanswer_stream` tests |
| `tests/unit/test_retrieval_protocol.py` | Update protocol conformance to only require `aretrieve` |
| `tests/unit/test_api_server.py` | Remove `conversation_history` from answer endpoint tests |
| **NEW** `tests/unit/test_answer_engine.py` | Test AnswerEngine: VLM path, LLM path, structured/freetext, citation parsing, streaming |
| **NEW** `tests/unit/test_query_rewrite.py` | Test `_rewrite_query`: with/without history, LLM failure propagation |
