# Answer Layer Architecture Upgrade — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move answer generation from RAGService layer to RAGServiceManager via a new AnswerEngine, unify prompt handling, add workspace compatibility filter, web UI query rewriting, and reorganize logging.

**Architecture:** New `AnswerEngine` module receives merged retrieval contexts and produces answers with proper citations. RAGServiceManager orchestrates: `aretrieve()` → round-robin merge → `AnswerEngine.generate()`. Backends become retrieval-only. Web UI handles conversational query rewriting; backend becomes stateless.

**Tech Stack:** Python 3.12, pytest + pytest-asyncio, FastAPI, asyncpg, LightRAG, Pydantic

**Spec:** `docs/superpowers/specs/2026-03-14-answer-layer-architecture-design.md`

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `src/dlightrag/core/answer.py` | Centralized answer generation (prompt, LLM/VLM call, citation, streaming) | **CREATE** |
| `src/dlightrag/unifiedrepresent/prompts.py` | VLM prompt templates | MODIFY (FREETEXT_REMINDER into system prompt) |
| `src/dlightrag/utils/logging.py` | Answer/reference logging helpers | MODIFY (rename logger) |
| `src/dlightrag/core/servicemanager.py` | Multi-workspace coordinator | MODIFY (rewire aanswer, add workspace filter, get_llm_func) |
| `src/dlightrag/core/federation.py` | Federated retrieval | MODIFY (delete federated_answer, simplify merge_results) |
| `src/dlightrag/core/retrieval/protocols.py` | RetrievalBackend protocol | MODIFY (remove aanswer/aanswer_stream) |
| `src/dlightrag/core/service.py` | RAGService facade | MODIFY (delete answer methods, add workspace meta upsert) |
| `src/dlightrag/unifiedrepresent/engine.py` | Unified mode orchestrator | MODIFY (delete answer methods) |
| `src/dlightrag/unifiedrepresent/retriever.py` | Visual retrieval pipeline | MODIFY (delete answer methods, migrate helpers) |
| `src/dlightrag/captionrag/retrieval.py` | Caption mode retrieval | MODIFY (delete answer methods) |
| `src/dlightrag/api/server.py` | REST API | MODIFY (remove conversation_history) |
| `src/dlightrag/mcp/server.py` | MCP server | MODIFY (remove conversation_history) |
| `src/dlightrag/web/routes.py` | Web UI routes | MODIFY (add query rewriting) |
| `tests/unit/test_answer_engine.py` | AnswerEngine tests | **CREATE** |
| `tests/unit/test_query_rewrite.py` | Query rewrite tests | **CREATE** |

---

## Chunk 1: Foundation — Prompts, Logging, AnswerEngine

### Task 1: FREETEXT_REMINDER prompt fix

**Files:**
- Modify: `src/dlightrag/unifiedrepresent/prompts.py:68-82`
- Test: `tests/unit/test_answer_prompt.py` (existing)

- [ ] **Step 1: Read existing prompt test to understand patterns**

Run: `cat tests/unit/test_answer_prompt.py`

- [ ] **Step 2: Update `get_answer_system_prompt` to include FREETEXT_REMINDER for freetext mode**

In `src/dlightrag/unifiedrepresent/prompts.py`, change:

```python
# BEFORE (line 76-78)
def get_answer_system_prompt(structured: bool = False) -> str:
    if structured:
        return _ANSWER_CORE + _ANSWER_STRUCTURED_SUFFIX
    return _ANSWER_CORE

# AFTER
def get_answer_system_prompt(structured: bool = False) -> str:
    if structured:
        return _ANSWER_CORE + _ANSWER_STRUCTURED_SUFFIX
    return _ANSWER_CORE + FREETEXT_REMINDER
```

- [ ] **Step 3: Delete the `UNIFIED_ANSWER_SYSTEM_PROMPT` alias (line 82)**

Delete:
```python
UNIFIED_ANSWER_SYSTEM_PROMPT = get_answer_system_prompt(structured=False)
```

Remove `"UNIFIED_ANSWER_SYSTEM_PROMPT"` from `__all__` list.

- [ ] **Step 4: Run existing prompt tests**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/test_answer_prompt.py -v`
Expected: PASS (update assertions if any test checks the old behavior)

- [ ] **Step 5: Commit**

```bash
git add src/dlightrag/unifiedrepresent/prompts.py tests/unit/test_answer_prompt.py
git commit -m "refactor: move FREETEXT_REMINDER from user prompt to system prompt"
```

---

### Task 2: Logging reorganization

**Files:**
- Modify: `src/dlightrag/utils/logging.py:1-45`

- [ ] **Step 1: Rename logger in `log_answer_llm_output`**

In `src/dlightrag/utils/logging.py`, change line 19:
```python
# BEFORE
logger = logging.getLogger("dlightrag.references")
# AFTER
logger = logging.getLogger("dlightrag.answer")
```

- [ ] **Step 2: Rename logger in `log_references`**

Change line 40:
```python
# BEFORE
logger = logging.getLogger("dlightrag.references")
# AFTER
logger = logging.getLogger("dlightrag.answer")
```

- [ ] **Step 3: Run all tests to verify no breakage**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/ -x -q`
Expected: All PASS (logger name is internal, no tests assert on it)

- [ ] **Step 4: Commit**

```bash
git add src/dlightrag/utils/logging.py
git commit -m "refactor: rename logger from dlightrag.references to dlightrag.answer"
```

---

### Task 3: AnswerEngine — non-streaming `generate()`

**Files:**
- Create: `src/dlightrag/core/answer.py`
- Create: `tests/unit/test_answer_engine.py`

- [ ] **Step 1: Write failing tests for AnswerEngine.generate()**

Create `tests/unit/test_answer_engine.py`:

```python
"""Tests for AnswerEngine — centralized answer generation."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dlightrag.core.retrieval.protocols import RetrievalContexts, RetrievalResult


def _make_contexts(*, with_images: bool = False) -> RetrievalContexts:
    """Build minimal RetrievalContexts for testing."""
    chunk: dict = {
        "chunk_id": "c1",
        "reference_id": "1",
        "file_path": "report.pdf",
        "content": "Revenue grew 20%",
        "page_idx": 1,
    }
    if with_images:
        chunk["image_data"] = "iVBORw0KGgoAAAANSUhEUg=="  # tiny base64
    return RetrievalContexts(
        chunks=[chunk],
        entities=[{"entity_name": "Revenue", "entity_type": "METRIC", "description": "Revenue metric"}],
        relationships=[],
    )


class TestAnswerEngineGenerate:
    """Test non-streaming answer generation."""

    async def test_llm_text_path_when_no_images(self):
        """When chunks have no image_data, uses llm_model_func (text path)."""
        from dlightrag.core.answer import AnswerEngine

        llm_func = AsyncMock(return_value="Revenue grew 20% [1-1]")
        engine = AnswerEngine(
            config=MagicMock(effective_vision_provider="openai"),
            llm_model_func=llm_func,
        )
        contexts = _make_contexts(with_images=False)

        with patch("dlightrag.core.answer.provider_supports_structured_vision", return_value=False):
            result = await engine.generate("What is revenue?", contexts)

        assert result.answer is not None
        assert "Revenue" in result.answer
        llm_func.assert_awaited_once()

    async def test_vlm_path_when_images_present(self):
        """When chunks have image_data, uses vision_model_func (VLM path)."""
        from dlightrag.core.answer import AnswerEngine

        vlm_func = AsyncMock(return_value="Revenue grew 20% [1-1]")
        engine = AnswerEngine(
            config=MagicMock(effective_vision_provider="openai"),
            vision_model_func=vlm_func,
            llm_model_func=AsyncMock(),
        )
        contexts = _make_contexts(with_images=True)

        with patch("dlightrag.core.answer.provider_supports_structured_vision", return_value=False):
            result = await engine.generate("What is revenue?", contexts)

        assert result.answer is not None
        vlm_func.assert_awaited_once()

    async def test_structured_json_parsing(self):
        """Structured mode parses JSON response into answer + references."""
        from dlightrag.core.answer import AnswerEngine

        structured_response = json.dumps({
            "answer": "Revenue grew 20% [1-1]",
            "references": [{"id": 1, "title": "report.pdf"}],
        })
        vlm_func = AsyncMock(return_value=structured_response)
        engine = AnswerEngine(
            config=MagicMock(effective_vision_provider="openai"),
            vision_model_func=vlm_func,
        )
        contexts = _make_contexts(with_images=True)

        with patch("dlightrag.core.answer.provider_supports_structured_vision", return_value=True):
            result = await engine.generate("What is revenue?", contexts)

        assert result.answer == "Revenue grew 20% [1-1]"
        assert len(result.references) == 1
        assert result.references[0].title == "report.pdf"

    async def test_no_model_funcs_returns_none_answer(self):
        """When no model functions provided, returns None answer."""
        from dlightrag.core.answer import AnswerEngine

        engine = AnswerEngine(config=MagicMock(effective_vision_provider="openai"))
        contexts = _make_contexts()
        result = await engine.generate("query", contexts)
        assert result.answer is None

    async def test_structured_parse_failure_degrades_to_raw(self):
        """When structured JSON parsing fails, falls back to raw text."""
        from dlightrag.core.answer import AnswerEngine

        vlm_func = AsyncMock(return_value="not valid json")
        engine = AnswerEngine(
            config=MagicMock(effective_vision_provider="openai"),
            vision_model_func=vlm_func,
        )
        contexts = _make_contexts(with_images=True)

        with patch("dlightrag.core.answer.provider_supports_structured_vision", return_value=True):
            result = await engine.generate("query", contexts)

        assert result.answer == "not valid json"
        assert result.references == []

    async def test_returns_contexts_unchanged(self):
        """generate() passes through the original contexts."""
        from dlightrag.core.answer import AnswerEngine

        llm_func = AsyncMock(return_value="answer")
        engine = AnswerEngine(
            config=MagicMock(effective_vision_provider="openai"),
            llm_model_func=llm_func,
        )
        contexts = _make_contexts()

        with patch("dlightrag.core.answer.provider_supports_structured_vision", return_value=False):
            result = await engine.generate("query", contexts)

        assert result.contexts is contexts
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/test_answer_engine.py -v`
Expected: FAIL (ImportError — `dlightrag.core.answer` does not exist)

- [ ] **Step 3: Implement AnswerEngine**

Create `src/dlightrag/core/answer.py`:

```python
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""AnswerEngine — centralized answer generation from retrieval contexts.

Receives merged RetrievalContexts (from any number of workspaces), builds
prompts, calls LLM/VLM, parses citations, and returns structured answers.
Mode-agnostic: detects image_data in chunks to choose VLM vs LLM path.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from dlightrag.citations.indexer import CitationIndexer
from dlightrag.core.retrieval.protocols import RetrievalContexts, RetrievalResult
from dlightrag.models.llm import provider_supports_structured_vision
from dlightrag.models.schemas import StructuredAnswer
from dlightrag.unifiedrepresent.prompts import get_answer_system_prompt
from dlightrag.utils.logging import log_answer_llm_output, log_references

logger = logging.getLogger("dlightrag.answer")


class AnswerEngine:
    """Unified answer generation from retrieval contexts.

    Receives merged RetrievalResult (from any number of workspaces),
    builds prompts, calls LLM/VLM, parses citations, returns answer.
    Mode-agnostic: detects image_data in chunks to choose VLM vs LLM path.
    """

    def __init__(
        self,
        config: Any,
        vision_model_func: Callable | None = None,
        llm_model_func: Callable | None = None,
    ) -> None:
        self._config = config
        self._vision_model_func = vision_model_func
        self._llm_model_func = llm_model_func
        self._provider: str = getattr(config, "effective_vision_provider", "openai")

    async def generate(
        self,
        query: str,
        contexts: RetrievalContexts,
    ) -> RetrievalResult:
        """Non-streaming: retrieval contexts → answer + references."""
        structured = provider_supports_structured_vision(self._provider)
        system_prompt = get_answer_system_prompt(structured=structured)
        user_prompt = self._build_user_prompt(query, contexts)

        chunks = contexts.get("chunks", [])
        has_images = any(c.get("image_data") for c in chunks)

        log_answer_llm_output(
            "answer_engine.generate",
            structured=structured,
            provider=self._provider,
            query=query,
        )

        if has_images and self._vision_model_func:
            result = await self._generate_vlm(
                system_prompt, user_prompt, chunks, structured, query,
            )
        elif self._llm_model_func:
            result = await self._generate_llm(
                system_prompt, user_prompt, structured, query,
            )
        else:
            return RetrievalResult(answer=None, contexts=contexts)

        try:
            log_references(
                "answer_engine.generate",
                result.references,
                query=query,
                provider=self._provider,
                structured=structured,
            )
        except Exception:
            logger.debug("log_references failed", exc_info=True)

        return RetrievalResult(
            answer=result.answer,
            contexts=contexts,
            references=result.references,
        )

    async def generate_stream(
        self,
        query: str,
        contexts: RetrievalContexts,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        """Streaming: retrieval contexts → (contexts, token_iterator)."""
        structured = provider_supports_structured_vision(self._provider)
        system_prompt = get_answer_system_prompt(structured=structured)
        user_prompt = self._build_user_prompt(query, contexts)

        chunks = contexts.get("chunks", [])
        has_images = any(c.get("image_data") for c in chunks)

        if has_images and self._vision_model_func:
            messages = self._build_vlm_messages(system_prompt, user_prompt, chunks)
            token_iter = await self._vision_model_func(
                user_prompt,
                messages=messages,
                stream=True,
                response_schema=StructuredAnswer if structured else None,
            )
        elif self._llm_model_func:
            token_iter = await self._llm_model_func(
                user_prompt,
                system_prompt=system_prompt,
                stream=True,
            )
        else:
            return contexts, None

        if structured and hasattr(token_iter, "__aiter__"):
            from dlightrag.models.streaming import AnswerStream, StreamingAnswerParser

            parser = StreamingAnswerParser()
            token_iter = AnswerStream(token_iter, parser)

        return contexts, token_iter

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _generate_vlm(
        self,
        system_prompt: str,
        user_prompt: str,
        chunks: list[dict[str, Any]],
        structured: bool,
        query: str,
    ) -> StructuredAnswer:
        """VLM multimodal answer generation (chunks have image_data)."""
        messages = self._build_vlm_messages(system_prompt, user_prompt, chunks)

        if structured:
            raw = await self._vision_model_func(
                user_prompt,
                messages=messages,
                response_schema=StructuredAnswer,
            )
            log_answer_llm_output(
                "answer_engine.generate",
                structured=True,
                provider=self._provider,
                query=query,
                raw=raw,
            )
            try:
                return StructuredAnswer.model_validate_json(raw)
            except Exception as e:
                log_answer_llm_output(
                    "answer_engine.generate",
                    structured=True,
                    provider=self._provider,
                    query=query,
                    raw=raw,
                    parse_error=e,
                )
                logger.warning("Structured answer parse failed, degrading to raw text")
                return StructuredAnswer(answer=raw, references=[])
        else:
            answer_text = await self._vision_model_func(user_prompt, messages=messages)
            log_answer_llm_output(
                "answer_engine.generate",
                structured=False,
                provider=self._provider,
                query=query,
                answer_text=answer_text,
            )
            return StructuredAnswer(answer=answer_text, references=[])

    async def _generate_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        structured: bool,
        query: str,
    ) -> StructuredAnswer:
        """LLM text-only answer generation (no images in chunks)."""
        answer_text = await self._llm_model_func(
            user_prompt,
            system_prompt=system_prompt,
        )
        log_answer_llm_output(
            "answer_engine.generate",
            structured=structured,
            provider=self._provider,
            query=query,
            answer_text=answer_text,
        )
        return StructuredAnswer(answer=answer_text, references=[])

    def _build_user_prompt(self, query: str, contexts: RetrievalContexts) -> str:
        """Build user prompt from contexts: KG context + reference list + question."""
        kg_context = self._format_kg_context(contexts)
        indexer = self._build_citation_indexer(contexts)
        ref_list = indexer.format_reference_list()
        return "\n\n".join([
            f"Knowledge Graph Context:\n{kg_context}",
            f"Reference Document List:\n{ref_list}",
            f"Question: {query}",
        ])

    @staticmethod
    def _build_vlm_messages(
        system_prompt: str, user_prompt: str, chunks: list[dict],
    ) -> list[dict]:
        """Build OpenAI-format multimodal messages with inline base64 images."""
        content: list[dict] = []
        for item in chunks:
            img_data = item.get("image_data")
            if img_data:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_data}"},
                    }
                )
        content.append({"type": "text", "text": user_prompt})
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    @staticmethod
    def _build_citation_indexer(contexts: RetrievalContexts) -> CitationIndexer:
        """Build a CitationIndexer from retrieval contexts."""
        flat: list[dict[str, Any]] = []
        for items in contexts.values():
            if isinstance(items, list):
                flat.extend(items)
        indexer = CitationIndexer()
        indexer.build_index(flat)
        return indexer

    @staticmethod
    def _format_kg_context(contexts: RetrievalContexts) -> str:
        """Format KG context (entities + relationships) as text for prompt."""
        parts: list[str] = []
        entities = contexts.get("entities", [])
        if entities:
            parts.append("## Entities")
            for e in entities[:20]:
                name = e.get("entity_name", "")
                etype = e.get("entity_type", "")
                desc = e.get("description", "")
                parts.append(f"- **{name}** ({etype}): {desc}")
        rels = contexts.get("relationships", [])
        if rels:
            parts.append("\n## Relationships")
            for r in rels[:20]:
                src = r.get("src_id", "")
                tgt = r.get("tgt_id", "")
                desc = r.get("description", "")
                parts.append(f"- {src} -> {tgt}: {desc}")
        return "\n".join(parts) if parts else "No knowledge graph context available."


__all__ = ["AnswerEngine"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/test_answer_engine.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/dlightrag/core/answer.py tests/unit/test_answer_engine.py
git commit -m "feat: add AnswerEngine for centralized answer generation"
```

---

### Task 4: AnswerEngine — streaming `generate_stream()` tests

**Files:**
- Modify: `tests/unit/test_answer_engine.py`

- [ ] **Step 1: Add streaming tests**

Append to `tests/unit/test_answer_engine.py`:

```python
class TestAnswerEngineGenerateStream:
    """Test streaming answer generation."""

    async def test_stream_returns_contexts_and_iterator(self):
        """generate_stream returns (contexts, async_iterator)."""
        from dlightrag.core.answer import AnswerEngine

        async def fake_stream(*args, **kwargs):
            async def gen():
                yield "Hello "
                yield "world"
            return gen()

        engine = AnswerEngine(
            config=MagicMock(effective_vision_provider="openai"),
            llm_model_func=fake_stream,
        )
        contexts = _make_contexts(with_images=False)

        with patch("dlightrag.core.answer.provider_supports_structured_vision", return_value=False):
            result_contexts, token_iter = await engine.generate_stream("query", contexts)

        assert result_contexts is contexts
        assert token_iter is not None

        tokens = []
        async for t in token_iter:
            tokens.append(t)
        assert tokens == ["Hello ", "world"]

    async def test_stream_no_model_returns_none_iterator(self):
        """When no model functions, returns (contexts, None)."""
        from dlightrag.core.answer import AnswerEngine

        engine = AnswerEngine(config=MagicMock(effective_vision_provider="openai"))
        contexts = _make_contexts()
        result_contexts, token_iter = await engine.generate_stream("query", contexts)
        assert result_contexts is contexts
        assert token_iter is None

    async def test_stream_vlm_path_with_images(self):
        """When chunks have images, uses vision_model_func for streaming."""
        from dlightrag.core.answer import AnswerEngine

        async def fake_vlm_stream(*args, **kwargs):
            assert kwargs.get("stream") is True
            async def gen():
                yield "answer"
            return gen()

        engine = AnswerEngine(
            config=MagicMock(effective_vision_provider="openai"),
            vision_model_func=fake_vlm_stream,
            llm_model_func=AsyncMock(),
        )
        contexts = _make_contexts(with_images=True)

        with patch("dlightrag.core.answer.provider_supports_structured_vision", return_value=False):
            _, token_iter = await engine.generate_stream("query", contexts)

        assert token_iter is not None
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/test_answer_engine.py -v`
Expected: All 9 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_answer_engine.py
git commit -m "test: add AnswerEngine streaming tests"
```

---

## Chunk 2: Manager Rewire + Federation

### Task 5: RAGServiceManager — rewire aanswer to use AnswerEngine

**Files:**
- Modify: `src/dlightrag/core/servicemanager.py`
- Modify: `tests/unit/test_servicemanager.py`

- [ ] **Step 1: Update test_servicemanager.py to test new flow**

Find the `TestRouting` class in `tests/unit/test_servicemanager.py`. The existing tests for `aanswer` call `svc.aanswer()` — these need to verify the new flow: `aretrieve()` → `AnswerEngine.generate()`.

Add/replace tests:

```python
class TestAnswerViaEngine:
    """aanswer and aanswer_stream route through AnswerEngine."""

    async def test_aanswer_calls_retrieve_then_engine(self):
        """aanswer() calls aretrieve() then AnswerEngine.generate()."""
        mgr = RAGServiceManager.__new__(RAGServiceManager)
        mgr._config = MagicMock(workspace="default", request_timeout=30)
        mgr._services = {}
        mgr._lock = None
        mgr._last_error_ts = None
        mgr._answer_engine = None

        mock_result = RetrievalResult(answer=None, contexts={"chunks": [], "entities": [], "relationships": []})
        mgr.aretrieve = AsyncMock(return_value=mock_result)

        engine_result = RetrievalResult(answer="test answer", contexts=mock_result.contexts)
        mock_engine = AsyncMock()
        mock_engine.generate = AsyncMock(return_value=engine_result)
        mgr._get_answer_engine = MagicMock(return_value=mock_engine)

        result = await mgr.aanswer("query")
        mgr.aretrieve.assert_awaited_once()
        mock_engine.generate.assert_awaited_once_with("query", mock_result.contexts)
        assert result.answer == "test answer"

    async def test_aanswer_stream_calls_retrieve_then_engine(self):
        """aanswer_stream() calls aretrieve() then AnswerEngine.generate_stream()."""
        mgr = RAGServiceManager.__new__(RAGServiceManager)
        mgr._config = MagicMock(workspace="default", request_timeout=30)
        mgr._services = {}
        mgr._lock = None
        mgr._last_error_ts = None
        mgr._answer_engine = None

        mock_result = RetrievalResult(answer=None, contexts={"chunks": [], "entities": [], "relationships": []})
        mgr.aretrieve = AsyncMock(return_value=mock_result)

        mock_engine = AsyncMock()
        mock_engine.generate_stream = AsyncMock(return_value=(mock_result.contexts, None))
        mgr._get_answer_engine = MagicMock(return_value=mock_engine)

        contexts, token_iter = await mgr.aanswer_stream("query")
        mock_engine.generate_stream.assert_awaited_once()
        assert token_iter is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/test_servicemanager.py::TestAnswerViaEngine -v`
Expected: FAIL (old aanswer still delegates to RAGService)

- [ ] **Step 3: Implement RAGServiceManager changes**

In `src/dlightrag/core/servicemanager.py`:

Add imports at top (after existing imports):
```python
from dlightrag.core.answer import AnswerEngine
from dlightrag.models.llm import get_llm_model_func, get_vision_model_func
```

Add to `__init__`:
```python
self._answer_engine: AnswerEngine | None = None
```

Add new methods:
```python
def _get_answer_engine(self) -> AnswerEngine:
    """Lazy-create AnswerEngine from global config."""
    if self._answer_engine is None:
        self._answer_engine = AnswerEngine(
            config=self._config,
            vision_model_func=get_vision_model_func(self._config),
            llm_model_func=get_llm_model_func(self._config),
        )
    return self._answer_engine

def get_llm_func(self):
    """Return the global LLM model function (for web UI query rewriting etc.)."""
    return get_llm_model_func(self._config)
```

Replace `aanswer()` method (lines 186-205):
```python
async def aanswer(
    self,
    query: str,
    *,
    workspace: str | None = None,
    workspaces: list[str] | None = None,
    **kwargs: Any,
) -> RetrievalResult:
    """Answer from one or more workspaces: retrieve → AnswerEngine."""
    ws_list = workspaces or [workspace or self._config.workspace]
    retrieval = await self.aretrieve(query, workspaces=ws_list, **kwargs)
    engine = self._get_answer_engine()
    return await engine.generate(query, retrieval.contexts)
```

Replace `aanswer_stream()` method (lines 207-230):
```python
async def aanswer_stream(
    self,
    query: str,
    *,
    workspace: str | None = None,
    workspaces: list[str] | None = None,
    **kwargs: Any,
) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
    """Streaming answer from one or more workspaces: retrieve → AnswerEngine."""
    ws_list = workspaces or [workspace or self._config.workspace]
    retrieval = await self.aretrieve(query, workspaces=ws_list, **kwargs)
    engine = self._get_answer_engine()
    return await engine.generate_stream(query, retrieval.contexts)
```

Remove `federated_answer` from imports (line 19):
```python
# BEFORE
from dlightrag.core.federation import federated_answer, federated_retrieve
# AFTER
from dlightrag.core.federation import federated_retrieve
```

- [ ] **Step 4: Update existing tests that test old aanswer routing**

In `tests/unit/test_servicemanager.py`, find tests in `TestRouting` that test `aanswer` or `aanswer_stream` delegating to `svc.aanswer()` / `svc.aanswer_stream()`. Remove or update them to match the new flow.

Also remove any tests that reference `federated_answer`.

- [ ] **Step 5: Run tests**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/test_servicemanager.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/dlightrag/core/servicemanager.py tests/unit/test_servicemanager.py
git commit -m "refactor: rewire RAGServiceManager.aanswer to use AnswerEngine"
```

---

### Task 6: Federation cleanup — delete `federated_answer`, simplify `merge_results`

**Files:**
- Modify: `src/dlightrag/core/federation.py`
- Modify: `tests/unit/test_federation.py`

- [ ] **Step 1: Update federation tests**

In `tests/unit/test_federation.py`:
- Remove all tests for `federated_answer` (the `TestFederatedAnswer` class)
- In `TestMergeResults`, update assertions: `merge_results` no longer produces `answer` or `references` fields. Remove assertions on `merged.answer` and `merged.references`.

- [ ] **Step 2: Simplify `merge_results` in federation.py**

In `src/dlightrag/core/federation.py`, in `merge_results()`:

Delete lines 67-74 (answer concatenation and references merge):
```python
    # DELETE these lines:
    # Build merged answer (concatenate non-None answers)
    answers = [r.answer for r in results if r.answer]
    merged_answer = "\n\n---\n\n".join(answers) if answers else None

    # Merge references (naive concatenation, no re-numbering)
    merged_refs = []
    for r in results:
        merged_refs.extend(r.references)
```

Update the return statement:
```python
    return RetrievalResult(
        answer=None,
        contexts=RetrievalContexts(
            chunks=merged_chunks,
            entities=merged_entities,
            relationships=merged_relations,
        ),
    )
```

- [ ] **Step 3: Delete `federated_answer` function entirely (lines 188-247)**

Remove the entire `federated_answer` function from `federation.py`.

- [ ] **Step 4: Run tests**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/test_federation.py -v`
Expected: All remaining tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/dlightrag/core/federation.py tests/unit/test_federation.py
git commit -m "refactor: delete federated_answer, simplify merge_results to contexts-only"
```

---

## Chunk 3: Backend Strip + Protocol

### Task 7: Strip RetrievalBackend protocol + RetrievalEngine

**Files:**
- Modify: `src/dlightrag/core/retrieval/protocols.py`
- Modify: `src/dlightrag/captionrag/retrieval.py`
- Modify: `tests/unit/test_retrieval_protocol.py`
- Modify: `tests/unit/test_retrieval_engine.py`

- [ ] **Step 1: Update protocol — remove aanswer and aanswer_stream**

In `src/dlightrag/core/retrieval/protocols.py`, remove the `aanswer` and `aanswer_stream` method signatures from the `RetrievalBackend` protocol. Keep only `aretrieve`.

- [ ] **Step 2: Update protocol tests**

In `tests/unit/test_retrieval_protocol.py`, remove tests that check for `aanswer` and `aanswer_stream` in the protocol. Update protocol conformance tests to only require `aretrieve`.

- [ ] **Step 3: Delete `aanswer` and `aanswer_stream` from RetrievalEngine**

In `src/dlightrag/captionrag/retrieval.py`:
- Delete `aanswer()` method (lines 88-135)
- Delete `aanswer_stream()` method (lines 137-193)
- Keep `aretrieve()` and `_attach_page_idx()` unchanged

- [ ] **Step 4: Update RetrievalEngine tests**

In `tests/unit/test_retrieval_engine.py`:
- Remove `TestRetrievalEngineAanswer` class
- Remove `TestRetrievalEngineAanswerStream` class
- Keep `TestRetrievalEngineAretrieve` unchanged

- [ ] **Step 5: Run tests**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/test_retrieval_protocol.py tests/unit/test_retrieval_engine.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/dlightrag/core/retrieval/protocols.py src/dlightrag/captionrag/retrieval.py tests/unit/test_retrieval_protocol.py tests/unit/test_retrieval_engine.py
git commit -m "refactor: strip aanswer from RetrievalBackend protocol and RetrievalEngine"
```

---

### Task 8: Strip VisualRetriever + UnifiedRepresentEngine

**Files:**
- Modify: `src/dlightrag/unifiedrepresent/retriever.py`
- Modify: `src/dlightrag/unifiedrepresent/engine.py`
- Modify: `tests/unit/test_unified_engine.py`

- [ ] **Step 1: Delete answer methods from VisualRetriever**

In `src/dlightrag/unifiedrepresent/retriever.py`:
- Delete `answer()` method (lines 294-410)
- Delete `answer_stream()` method (lines 412-495)
- Delete `_build_vlm_messages()` (lines 755-772) — migrated to AnswerEngine
- Delete `_build_citation_indexer()` (lines 798-812) — migrated to AnswerEngine
- Delete `_format_kg_context()` (lines 774-796) — migrated to AnswerEngine
- Remove `conversation_context` parameter from `retrieve()` signature (line 71) and from `_text_retrieve` calls
- Remove unused imports: `CitationIndexer`
- Keep `retrieve()`, `query_by_visual_embedding()`, reranking methods, `_parse_rerank_score()`

- [ ] **Step 2: Delete answer methods from UnifiedRepresentEngine**

In `src/dlightrag/unifiedrepresent/engine.py`:
- Delete `aanswer()` method (lines 262-287)
- Delete `aanswer_stream()` method (lines 289-309)
- Delete `_build_conversation_context()` method (lines 222-234)
- In `aretrieve()` (line 248): remove `conversation_history` extraction from kwargs
- Keep `aingest()`, `aretrieve()`, `adelete_doc()`, `aclose()`, `_extract_image_bytes()`

- [ ] **Step 3: Update unified engine tests**

In `tests/unit/test_unified_engine.py`:
- Remove `TestAanswer` class
- Remove `TestProtocolCompliance` tests for `aanswer`/`aanswer_stream` if present
- Update remaining tests that reference conversation_context
- Keep `TestAretrieve`, `TestAingest`, `TestAdeleteDoc`, etc.

- [ ] **Step 4: Run tests**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/test_unified_engine.py tests/unit/test_visual_retriever.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/dlightrag/unifiedrepresent/retriever.py src/dlightrag/unifiedrepresent/engine.py tests/unit/test_unified_engine.py
git commit -m "refactor: strip answer methods from VisualRetriever and UnifiedRepresentEngine"
```

---

### Task 9: Strip RAGService answer methods

**Files:**
- Modify: `src/dlightrag/core/service.py`
- Modify: `tests/unit/test_service.py`

- [ ] **Step 1: Delete answer methods from RAGService**

In `src/dlightrag/core/service.py`:
- Delete `_truncate_history()` method (lines 789-798)
- Delete `aanswer()` method (lines 800-822)
- Delete `aanswer_stream()` method (lines 824-846)
- Delete the `truncate_conversation_history` import (line 28)

- [ ] **Step 2: Update RAGService tests**

In `tests/unit/test_service.py`:
- Remove `TestConversationHistoryTruncation` class
- Remove any tests that call `svc.aanswer()` or `svc.aanswer_stream()`
- Keep ingestion, retrieval, file management, and initialization tests

- [ ] **Step 3: Run full test suite**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/ -x -q`
Expected: All PASS — the answer chain is fully rewired

- [ ] **Step 4: Commit**

```bash
git add src/dlightrag/core/service.py tests/unit/test_service.py
git commit -m "refactor: strip answer methods from RAGService, remove conversation_history"
```

---

## Chunk 4: Workspace Meta + Interface Cleanup

### Task 10: Workspace compatibility filter

**Files:**
- Modify: `src/dlightrag/core/service.py` (add meta upsert in aingest)
- Modify: `src/dlightrag/core/servicemanager.py` (add filter + table creation)

- [ ] **Step 1: Add `dlightrag_workspace_meta` table creation in `_ensure_pg_schema`**

In `src/dlightrag/core/service.py`, in `_ensure_pg_schema()` method (after existing table creation around line 286):

```python
await conn.execute("""CREATE TABLE IF NOT EXISTS dlightrag_workspace_meta (
    workspace TEXT PRIMARY KEY,
    embedding_model TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
)""")
```

- [ ] **Step 2: Add workspace meta upsert in `aingest()`**

In `src/dlightrag/core/service.py`, at the end of `aingest()` (before the return statements), add:

```python
# Record workspace metadata for compatibility filtering
await self._upsert_workspace_meta()
```

Add the helper method to RAGService:

```python
async def _upsert_workspace_meta(self) -> None:
    """Record embedding model for this workspace (PG-only, best-effort)."""
    if not self.config.kv_storage.startswith("PG"):
        return
    try:
        import asyncpg
        conn = await asyncpg.connect(
            host=self.config.postgres_host,
            port=self.config.postgres_port,
            user=self.config.postgres_user,
            password=self.config.postgres_password,
            database=self.config.postgres_database,
        )
        try:
            await conn.execute(
                """INSERT INTO dlightrag_workspace_meta (workspace, embedding_model)
                   VALUES ($1, $2)
                   ON CONFLICT (workspace)
                   DO UPDATE SET embedding_model = $2, updated_at = NOW()""",
                self.config.workspace,
                self.config.embedding_model,
            )
        finally:
            await conn.close()
    except Exception:
        logger.debug("workspace meta upsert failed", exc_info=True)
```

- [ ] **Step 3: Add compatibility filter to RAGServiceManager**

In `src/dlightrag/core/servicemanager.py`, refactor `list_workspaces()` and add filter:

```python
async def list_workspaces(self, *, compatible_only: bool = False) -> list[str]:
    """Discover available workspaces based on storage backend.

    Args:
        compatible_only: If True, filter to workspaces whose embedding
            model matches the current global config (PG-only).
    """
    all_ws = await self._list_all_workspaces()
    if not compatible_only:
        return all_ws
    return await self._filter_compatible(all_ws)

async def _list_all_workspaces(self) -> list[str]:
    """Internal: discover all workspaces regardless of compatibility."""
    # (move existing list_workspaces body here)
    ...

async def _filter_compatible(self, workspaces: list[str]) -> list[str]:
    """Filter workspaces to those using the same embedding model."""
    if not self._config.kv_storage.startswith("PG"):
        return workspaces  # non-PG: no metadata, all compatible
    compatible = []
    try:
        import asyncpg
        conn = await asyncpg.connect(
            host=self._config.postgres_host,
            port=self._config.postgres_port,
            user=self._config.postgres_user,
            password=self._config.postgres_password,
            database=self._config.postgres_database,
        )
        try:
            for ws in workspaces:
                row = await conn.fetchrow(
                    "SELECT embedding_model FROM dlightrag_workspace_meta WHERE workspace = $1",
                    ws,
                )
                if row is None or row["embedding_model"] == self._config.embedding_model:
                    compatible.append(ws)
        finally:
            await conn.close()
    except Exception:
        logger.debug("workspace compatibility filter failed, returning all", exc_info=True)
        return workspaces
    return compatible
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/test_servicemanager.py tests/unit/test_service.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/dlightrag/core/service.py src/dlightrag/core/servicemanager.py
git commit -m "feat: add workspace compatibility filter with dlightrag_workspace_meta table"
```

---

### Task 11: API + MCP — remove conversation_history

**Files:**
- Modify: `src/dlightrag/api/server.py`
- Modify: `src/dlightrag/mcp/server.py`
- Modify: `tests/unit/test_api_server.py`

- [ ] **Step 1: Remove conversation_history from API server**

In `src/dlightrag/api/server.py`:
- In `AnswerRequest` model (around line 104): delete `conversation_history` field
- Lines 199-200: remove `if body.conversation_history: kwargs["conversation_history"] = ...`
- Also update the logger name in the log_references catch block (line 228): change `dlightrag.references` to `dlightrag.answer`

- [ ] **Step 2: Remove conversation_history from MCP server**

In `src/dlightrag/mcp/server.py`:
- Lines 144-155: remove `conversation_history` from answer tool `inputSchema.properties`
- Lines 239-240: remove `if arguments.get("conversation_history"): kwargs[...] = ...`

- [ ] **Step 3: Update API server tests**

In `tests/unit/test_api_server.py`:
- In `TestAnswerEndpoint` and `TestAnswerStreamMode`: remove any tests or assertions about `conversation_history`
- Keep all other assertions about answer behavior

- [ ] **Step 4: Run tests**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/test_api_server.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/dlightrag/api/server.py src/dlightrag/mcp/server.py tests/unit/test_api_server.py
git commit -m "refactor: remove conversation_history from API and MCP interfaces"
```

---

## Chunk 5: Web UI Query Rewriting

### Task 12: Web UI query rewriting

**Files:**
- Modify: `src/dlightrag/web/routes.py`
- Create: `tests/unit/test_query_rewrite.py`

- [ ] **Step 1: Write failing tests for `_rewrite_query`**

Create `tests/unit/test_query_rewrite.py`:

```python
"""Tests for web UI conversational query rewriting."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


class TestRewriteQuery:
    """Test the _rewrite_query helper."""

    async def test_no_history_returns_original(self):
        """First turn (no history) returns message unchanged, no LLM call."""
        from dlightrag.web.routes import _rewrite_query

        llm = AsyncMock()
        result = await _rewrite_query("What is revenue?", None, llm)
        assert result == "What is revenue?"
        llm.assert_not_awaited()

    async def test_empty_history_returns_original(self):
        """Empty history list returns message unchanged."""
        from dlightrag.web.routes import _rewrite_query

        llm = AsyncMock()
        result = await _rewrite_query("What is revenue?", [], llm)
        assert result == "What is revenue?"
        llm.assert_not_awaited()

    async def test_with_history_calls_llm(self):
        """With conversation history, calls LLM and returns rewritten query."""
        from dlightrag.web.routes import _rewrite_query

        llm = AsyncMock(return_value="What were the Q3 revenue numbers?")
        history = [
            {"role": "user", "content": "Tell me about revenue"},
            {"role": "assistant", "content": "Revenue grew 20% in Q3."},
        ]
        result = await _rewrite_query("more details", history, llm)
        assert result == "What were the Q3 revenue numbers?"
        llm.assert_awaited_once()

    async def test_uses_last_20_messages(self):
        """Only last 20 messages (10 turns) are included."""
        from dlightrag.web.routes import _rewrite_query

        llm = AsyncMock(return_value="rewritten")
        history = [{"role": "user", "content": f"msg {i}"} for i in range(30)]
        await _rewrite_query("follow up", history, llm)
        call_args = llm.call_args
        prompt = call_args[0][0]
        assert "msg 10" in prompt
        assert "msg 9" not in prompt  # trimmed

    async def test_llm_failure_propagates(self):
        """LLM failure raises (no fallback)."""
        from dlightrag.web.routes import _rewrite_query

        llm = AsyncMock(side_effect=RuntimeError("LLM down"))
        history = [{"role": "user", "content": "hello"}]
        with pytest.raises(RuntimeError, match="LLM down"):
            await _rewrite_query("follow up", history, llm)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/test_query_rewrite.py -v`
Expected: FAIL (ImportError — `_rewrite_query` does not exist in routes)

- [ ] **Step 3: Implement `_rewrite_query` in routes.py**

In `src/dlightrag/web/routes.py`, add the helper function (near top of file, after imports):

```python
async def _rewrite_query(
    user_message: str,
    conversation_history: list[dict[str, str]] | None,
    llm_func,
) -> str:
    """Rewrite user message into standalone query using conversation context.

    Skips LLM call if no conversation history (first turn).
    Uses last 10 turns (20 messages) for context.
    Raises on LLM failure (no fallback).
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

- [ ] **Step 4: Run query rewrite tests**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/test_query_rewrite.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Update route to use `_rewrite_query`**

In `src/dlightrag/web/routes.py`, in the `answer_stream` function:

Replace conversation_history handling (around lines 99-111):

```python
# BEFORE
kwargs: dict[str, Any] = {}
if conversation_history:
    kwargs["conversation_history"] = conversation_history
try:
    contexts, token_iter = await manager.aanswer_stream(
        query=query,
        workspace=workspace,
        workspaces=workspaces,
        multimodal_content=multimodal_content,
        **kwargs,
    )

# AFTER
try:
    # Rewrite query using conversation context
    llm_func = manager.get_llm_func()
    standalone_query = await _rewrite_query(query, conversation_history, llm_func)

    contexts, token_iter = await manager.aanswer_stream(
        query=standalone_query,
        workspace=workspace,
        workspaces=workspaces,
        multimodal_content=multimodal_content,
    )
```

- [ ] **Step 6: Remove `history_char_budget` meta event**

In the same file, remove the budget calculation (around lines 180-191) since conversation history management is now client-side:

```python
# DELETE the meta event lines:
# max_tokens = cfg.max_conversation_tokens
# max_msgs = cfg.max_conversation_turns * 2
# ...
# yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
```

- [ ] **Step 7: Run full test suite**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/ -x -q`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/dlightrag/web/routes.py tests/unit/test_query_rewrite.py
git commit -m "feat: add web UI conversational query rewriting, remove conversation_history passthrough"
```

---

### Task 13: Final verification

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 2: Verify no remaining references to deleted code**

Run these grep checks:
```bash
cd /Users/hanlianlyu/Github/DlightRAG/src
# Should find NO matches (except imports in __init__.py or __all__ re-exports):
grep -rn "federated_answer" dlightrag/ --include="*.py"
grep -rn "UNIFIED_ANSWER_SYSTEM_PROMPT" dlightrag/ --include="*.py"
grep -rn "dlightrag.references" dlightrag/ --include="*.py"
grep -rn "_truncate_history" dlightrag/ --include="*.py"
```

Expected: No matches (or only in migration/changelog comments)

- [ ] **Step 3: Verify AnswerEngine is the only answer path**

```bash
# Should only match AnswerEngine and its tests:
grep -rn "def aanswer\|def aanswer_stream\|def answer_stream\|def answer(" dlightrag/ --include="*.py"
```

Expected: No matches in service/engine/retriever files

- [ ] **Step 4: Commit any final fixes**

```bash
git add -A
git commit -m "refactor: answer layer architecture upgrade complete"
```
