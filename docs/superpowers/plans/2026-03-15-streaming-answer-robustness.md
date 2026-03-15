# Streaming Answer Robustness Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make streaming answers robust by always using freetext prompts for streaming and unifying reference extraction into a single 3-level fallback function.

**Architecture:** Streaming always uses freetext prompt (no JSON state machine). Non-streaming keeps structured output when supported. A new `extract_references()` function provides 3-level fallback (JSON block → `### References` section → empty) used by both paths. Consumer-side manual fallback logic is removed.

**Tech Stack:** Python, pytest, asyncio, Pydantic

**Spec:** `docs/superpowers/specs/2026-03-15-streaming-answer-robustness-design.md`

---

## Chunk 1: Core extraction + simplified streaming

### Task 1: `extract_references()` — tests + implementation

**Files:**
- Create: `tests/unit/test_extract_references.py`
- Modify: `src/dlightrag/citations/parser.py:119-148`

- [ ] **Step 1: Write failing tests for `extract_references`**

Create `tests/unit/test_extract_references.py`:

```python
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for extract_references() 3-level fallback."""

from __future__ import annotations

from dlightrag.citations.parser import extract_references


class TestLevel1JsonBlock:
    def test_full_structured_json(self) -> None:
        """Full structured JSON — extract answer and refs."""
        raw = '{"answer": "Growth is 15% [1-1].", "references": [{"id": 1, "title": "report.pdf"}]}'
        answer, refs = extract_references(raw)
        assert answer == "Growth is 15% [1-1]."
        assert len(refs) == 1
        assert refs[0].title == "report.pdf"

    def test_freetext_then_json_block(self) -> None:
        """Freetext followed by JSON block — strip JSON, extract refs."""
        raw = (
            "Based on the analysis, growth is 15% [1-1].\n\n"
            '{"answer": "Growth is 15% [1-1].", '
            '"references": [{"id": 1, "title": "report.pdf"}]}'
        )
        answer, refs = extract_references(raw)
        assert "Based on the analysis" in answer
        assert '{"answer"' not in answer
        assert len(refs) == 1

    def test_json_with_code_fence(self) -> None:
        """JSON wrapped in ```json fence."""
        raw = '```json\n{"answer": "Hello.", "references": [{"id": 1, "title": "doc.pdf"}]}\n```'
        answer, refs = extract_references(raw)
        assert answer == "Hello."
        assert len(refs) == 1

    def test_json_no_answer_key(self) -> None:
        """JSON block has only references, no answer key — strip JSON from raw."""
        raw = (
            "The analysis shows growth.\n\n"
            '{"references": [{"id": 1, "title": "report.pdf"}]}'
        )
        answer, refs = extract_references(raw)
        assert "The analysis shows growth." in answer
        assert '{"references"' not in answer
        assert len(refs) == 1

    def test_code_fenced_json_no_answer_key(self) -> None:
        """Code-fenced JSON without answer key — strip fence and JSON."""
        raw = (
            "The analysis shows growth.\n\n"
            '```json\n{"references": [{"id": 1, "title": "report.pdf"}]}\n```'
        )
        answer, refs = extract_references(raw)
        assert "```" not in answer
        assert "The analysis shows growth." in answer
        assert len(refs) == 1

    def test_malformed_references_array(self) -> None:
        """JSON exists but references is not a list — use answer from JSON."""
        raw = '{"answer": "Text.", "references": "not-a-list"}'
        answer, refs = extract_references(raw)
        assert answer == "Text."
        assert refs == []

    def test_json_with_empty_references(self) -> None:
        """JSON with empty references array — use answer from JSON."""
        raw = '{"answer": "No refs.", "references": []}'
        answer, refs = extract_references(raw)
        assert answer == "No refs."
        assert refs == []

    def test_json_with_invalid_ref_entries(self) -> None:
        """Some ref entries are invalid — skip bad ones, keep good ones."""
        raw = '{"answer": "Text.", "references": [{"id": 1, "title": "a.pdf"}, {"bad": true}]}'
        answer, refs = extract_references(raw)
        assert len(refs) == 1
        assert refs[0].title == "a.pdf"


class TestLevel2ReferencesSection:
    def test_standard_references_section(self) -> None:
        """Standard ### References section at end of text."""
        raw = (
            "Growth is 15% [1].\n\n"
            "### References\n"
            "- [1] report.pdf\n"
            "- [2] summary.docx"
        )
        answer, refs = extract_references(raw)
        assert answer == "Growth is 15% [1]."
        assert len(refs) == 2
        assert refs[0].title == "report.pdf"
        assert refs[1].title == "summary.docx"

    def test_references_heading_case_insensitive(self) -> None:
        """Heading case variants work."""
        raw = "Answer text.\n\n## references\n[1] doc.pdf"
        answer, refs = extract_references(raw)
        assert len(refs) == 1


class TestLevel3Empty:
    def test_plain_text_no_refs(self) -> None:
        """Pure text without any refs structure — return as-is."""
        raw = "This is a plain answer with no references."
        answer, refs = extract_references(raw)
        assert answer == raw
        assert refs == []

    def test_empty_string(self) -> None:
        raw = ""
        answer, refs = extract_references(raw)
        assert answer == ""
        assert refs == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_extract_references.py -v`
Expected: FAIL with `ImportError: cannot import name 'extract_references'`

- [ ] **Step 3: Implement `extract_references`**

In `src/dlightrag/citations/parser.py`, add after the existing `parse_freetext_references` function (around line 148):

```python
def extract_references(raw: str) -> tuple[str, list[Reference]]:
    """Extract references from LLM output with 3-level fallback.

    Returns (cleaned_answer_text, references).

    Level 1 — JSON block: find and parse JSON containing "references" array,
              strip JSON block from answer text.
    Level 2 — ### References section: parse [n] Title lines,
              strip references section from answer text.
    Level 3 — Empty: return original text + empty list.
    """
    # Level 1: JSON block extraction
    answer, refs, found = _try_json_extraction(raw)
    if found:
        return answer, refs

    # Level 2: ### References section
    answer, refs = parse_freetext_references(raw)
    if refs:
        return answer, refs

    # Level 3: empty
    return raw, []


def _try_json_extraction(raw: str) -> tuple[str, list[Reference], bool]:
    """Attempt to extract references from a JSON block in the text.

    Returns (answer, refs, json_found). The ``json_found`` flag is True
    when a JSON block was successfully parsed (even if refs is empty),
    so callers can distinguish "no JSON" from "JSON with no refs".
    """
    import json

    from dlightrag.utils.text import extract_json

    json_str = extract_json(raw)
    # If extract_json returned the original text unchanged, no JSON found
    if json_str == raw and not raw.lstrip().startswith("{"):
        return raw, [], False

    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return raw, [], False

    # JSON parsed — determine answer text
    if "answer" in data:
        answer = data["answer"]
    else:
        # Strip the JSON block from the original text
        answer = _strip_json_block(raw, json_str)

    raw_refs = data.get("references")
    if not isinstance(raw_refs, list):
        return answer, [], True

    refs: list[Reference] = []
    for r in raw_refs:
        try:
            refs.append(Reference.model_validate(r))
        except Exception:
            pass

    return answer, refs, True


def _strip_json_block(raw: str, json_str: str) -> str:
    """Strip a JSON block (and surrounding code fences) from raw text."""
    import re

    # Try to strip a fenced block first: ```json\n...\n```
    fenced = re.compile(r"```(?:json)?\s*" + re.escape(json_str) + r"\s*```")
    cleaned = fenced.sub("", raw).strip()
    if cleaned != raw.strip():
        return cleaned

    # Try to find the exact JSON substring in raw
    idx = raw.find(json_str)
    if idx >= 0:
        before = raw[:idx].rstrip()
        after = raw[idx + len(json_str) :].lstrip()
        parts = [p for p in (before, after) if p]
        return "\n\n".join(parts) if parts else ""
    # Fallback: strip from first { onward
    start = raw.find("{")
    if start > 0:
        return raw[:start].rstrip()
    return ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_extract_references.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_extract_references.py src/dlightrag/citations/parser.py
git commit -m "feat: add extract_references() with 3-level fallback"
```

---

### Task 2: Simplify `AnswerStream`, delete `StreamingAnswerParser`

**Files:**
- Modify: `src/dlightrag/models/streaming.py` (full rewrite)
- Modify: `tests/unit/test_streaming_parser.py` (full rewrite)

- [ ] **Step 1: Rewrite tests for simplified `AnswerStream`**

Replace the contents of `tests/unit/test_streaming_parser.py`:

```python
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for simplified AnswerStream (passthrough + post-stream ref extraction)."""

from __future__ import annotations

import pytest

from dlightrag.models.streaming import AnswerStream


@pytest.mark.asyncio
class TestAnswerStream:
    async def test_passthrough_tokens(self) -> None:
        """All tokens are yielded as-is."""

        async def fake_stream():
            yield "Hello "
            yield "world."

        stream = AnswerStream(fake_stream())
        parts = []
        async for token in stream:
            parts.append(token)
        assert "".join(parts) == "Hello world."

    async def test_extracts_references_from_freetext(self) -> None:
        """Post-stream extraction finds ### References section."""

        async def fake_stream():
            yield "Growth is 15% [1].\n\n"
            yield "### References\n"
            yield "- [1] report.pdf"

        stream = AnswerStream(fake_stream())
        parts = []
        async for token in stream:
            parts.append(token)
        assert len(stream.references) == 1
        assert stream.references[0].title == "report.pdf"
        assert "### References" not in stream.answer

    async def test_extracts_references_from_json_block(self) -> None:
        """Post-stream extraction finds trailing JSON block."""

        async def fake_stream():
            yield "Based on the data [1-1].\n\n"
            yield '{"answer": "Based on the data [1-1].", '
            yield '"references": [{"id": 1, "title": "report.pdf"}]}'

        stream = AnswerStream(fake_stream())
        parts = []
        async for token in stream:
            parts.append(token)
        # Raw stream includes JSON (inherent streaming limitation)
        assert '{"answer"' in "".join(parts)
        # But .answer is cleaned
        assert '{"answer"' not in stream.answer
        assert len(stream.references) == 1

    async def test_no_references(self) -> None:
        """Plain text — empty references, answer equals full text."""

        async def fake_stream():
            yield "Just a plain answer."

        stream = AnswerStream(fake_stream())
        async for _ in stream:
            pass
        assert stream.references == []
        assert stream.answer == "Just a plain answer."

    async def test_empty_stream(self) -> None:
        """Empty stream — no crash."""

        async def fake_stream():
            return
            yield  # make it an async generator

        stream = AnswerStream(fake_stream())
        parts = []
        async for token in stream:
            parts.append(token)
        assert parts == []
        assert stream.references == []
        assert stream.answer == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_streaming_parser.py -v`
Expected: FAIL (imports `AnswerStream` with old constructor signature)

- [ ] **Step 3: Rewrite `streaming.py`**

Replace the contents of `src/dlightrag/models/streaming.py`:

```python
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Streaming answer wrapper with post-stream reference extraction."""

from __future__ import annotations

from collections.abc import AsyncIterator

from dlightrag.citations.parser import extract_references
from dlightrag.models.schemas import Reference


class AnswerStream(AsyncIterator[str]):
    """Async iterator that passes through tokens and extracts references post-stream.

    Yields all tokens as-is for real-time display. After the stream ends,
    ``self.references`` contains extracted references and ``self.answer``
    contains the cleaned answer text (JSON blocks / reference sections stripped).

    Consumers should use ``self.answer`` (not locally accumulated text) for
    post-stream processing like CitationProcessor.
    """

    def __init__(self, raw_iterator: AsyncIterator[str]) -> None:
        self._raw = raw_iterator
        self._parts: list[str] = []
        self._gen = self._iterate()
        self.references: list[Reference] = []
        self.answer: str = ""

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        return await self._gen.__anext__()

    async def _iterate(self) -> AsyncIterator[str]:  # type: ignore[override]
        async for chunk in self._raw:
            self._parts.append(chunk)
            yield chunk
        full = "".join(self._parts)
        self.answer, self.references = extract_references(full)


__all__ = ["AnswerStream"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_streaming_parser.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/dlightrag/models/streaming.py tests/unit/test_streaming_parser.py
git commit -m "refactor: replace StreamingAnswerParser with passthrough AnswerStream"
```

---

## Chunk 2: Answer engine + consumer integration

### Task 3: Update `generate_stream` and `_parse_response` in `answer.py`

**Files:**
- Modify: `src/dlightrag/core/answer.py:135-211` (generate_stream), `300-360` (_parse_response)
- Modify: `tests/unit/test_answer_engine.py` (3 streaming tests + _parse_response tests)

- [ ] **Step 1: Update `test_answer_engine.py` streaming tests**

In `tests/unit/test_answer_engine.py`, find and update these 3 tests:

**Test 1** — `test_stream_vlm_structured_wraps_with_answer_stream` (around line 293):
Change: remove the `vlm_func.supports_structured = True` line. The assertion `isinstance(token_iter, AnswerStreamCls)` should still pass because streaming now always wraps.

**Test 2** — `test_stream_text_structured_wraps_with_answer_stream` (around line 313):
Same change — remove `llm_func.supports_structured = True`. Still assert `isinstance`.

**Test 3** — `test_stream_text_freetext_not_wrapped` (around line 332):
Rename to `test_stream_text_freetext_also_wrapped`. Change `llm_func.supports_structured = False`. Change assertion from `not isinstance(token_iter, AnswerStreamCls)` to `isinstance(token_iter, AnswerStreamCls)` — freetext now also wraps.

Also update `test_stream_text_structured_no_response_format` (around line 559):
This test is still valid — streaming should never pass `response_format`. Keep as-is.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_answer_engine.py -k "stream" -v`
Expected: FAIL (AnswerStream constructor changed, wrapping logic changed)

- [ ] **Step 3: Update `generate_stream` in `answer.py`**

Replace `generate_stream` method (lines ~135-211) with:

```python
    async def generate_stream(
        self,
        query: str,
        contexts: RetrievalContexts,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        """Streaming answer generation.

        Always uses freetext prompt regardless of supports_structured,
        since streaming cannot enforce response_format. Wraps the token
        stream with AnswerStream for post-stream reference extraction.
        """
        has_images = self._has_images(contexts)
        model_func = self._select_model_func(has_images)

        if model_func is None:
            logger.info("[AE] generate_stream: no model_func, returning None")
            return contexts, None

        # Always freetext for streaming — structured prompt is wrong here
        system_prompt = get_answer_system_prompt(structured=False)
        user_prompt = self._build_user_prompt(query, contexts)

        logger.info(
            "[AE] generate_stream: has_images=%s chunks=%d query=%s",
            has_images,
            len(contexts.get("chunks", [])),
            query[:60],
        )

        log_answer_llm_output(
            "answer_engine.generate_stream",
            structured=False,
            query=query,
        )

        if has_images:
            messages = self._build_vlm_messages(system_prompt, user_prompt, contexts["chunks"])
            token_iterator = await model_func(
                user_prompt,
                messages=messages,
                stream=True,
            )
        else:
            token_iterator = await model_func(
                user_prompt,
                system_prompt=system_prompt,
                stream=True,
            )

        logger.info(
            "[AE] generate_stream: model_func returned type=%s",
            type(token_iterator).__name__,
        )

        # Always wrap with AnswerStream (passthrough + post-stream ref extraction)
        if hasattr(token_iterator, "__aiter__"):
            logger.info("[AE] generate_stream: wrapping with AnswerStream")
            token_iterator = AnswerStream(token_iterator)

        return contexts, token_iterator
```

- [ ] **Step 4: Update `_parse_response` to use `extract_references`**

Replace the fallback calls in `_parse_response` (lines ~300-360).

Replace the `parse_freetext_references` import:
```python
from dlightrag.citations.parser import parse_freetext_references
```
with:
```python
from dlightrag.citations.parser import extract_references
```

(Drop `parse_freetext_references` entirely — it's no longer used in this file after the replacements below.)

In `_parse_response`, replace both occurrences of:
```python
answer_text, refs = parse_freetext_references(raw)
```
with:
```python
answer_text, refs = extract_references(raw)
```

Also remove the unused `StreamingAnswerParser` import from `answer.py`. The import line:
```python
from dlightrag.models.streaming import AnswerStream, StreamingAnswerParser
```
becomes:
```python
from dlightrag.models.streaming import AnswerStream
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_answer_engine.py -v`
Expected: all PASS

- [ ] **Step 6: Run all unit tests**

Run: `uv run pytest tests/unit/ -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add src/dlightrag/core/answer.py tests/unit/test_answer_engine.py
git commit -m "refactor: generate_stream always freetext, _parse_response uses extract_references"
```

---

### Task 4: Consumer cleanup — `server.py` and `routes.py`

**Files:**
- Modify: `src/dlightrag/api/server.py:276-319`
- Modify: `src/dlightrag/web/routes.py:145-194`

- [ ] **Step 1: Update `server.py` SSE consumer**

In `src/dlightrag/api/server.py`, find the `event_generator` function (around line 276).

**1a.** Replace the reference extraction block (lines ~291-303):

```python
            # Extract references: structured (AnswerStream) or freetext (parse)
            refs = getattr(token_iter, "references", None)
            if not refs and full_answer:
                full_answer, parsed_refs = parse_freetext_references(full_answer)
                refs = parsed_refs or None

            logger.info(
                "[API] SSE: token_iter type=%s refs_count=%s",
                type(token_iter).__name__,
                len(refs) if refs else 0,
            )

            refs_data = [r.model_dump() for r in refs] if refs else []
```

With:

```python
            # References extracted by AnswerStream post-stream
            refs = getattr(token_iter, "references", None) or []
            # Use cleaned answer for downstream processing (JSON/ref sections stripped)
            clean_answer = getattr(token_iter, "answer", None) or full_answer

            logger.info("[API] SSE: refs_count=%d", len(refs))

            refs_data = [r.model_dump() for r in refs]
```

**1b.** Update the `CitationProcessor.process` call (line ~314). Change:
```python
            if full_answer and flat_contexts:
                processor = CitationProcessor(contexts=flat_contexts, available_sources=all_sources)
                cited = processor.process(full_answer)
```
to:
```python
            if clean_answer and flat_contexts:
                processor = CitationProcessor(contexts=flat_contexts, available_sources=all_sources)
                cited = processor.process(clean_answer)
```

**1c.** Remove the `parse_freetext_references` import from this file (it's no longer used).

- [ ] **Step 2: Update `routes.py` SSE consumer**

In `src/dlightrag/web/routes.py`, find the same pattern (around line 161).

**2a.** Replace the reference extraction + ref reconstruction block (lines ~161-179):

```python
            # Extract references: structured (AnswerStream) or freetext (parse)
            refs = getattr(token_iter, "references", None)
            if not refs and full_answer:
                full_answer, parsed_refs = parse_freetext_references(full_answer)
                refs = parsed_refs or None

            logger.info(
                "[WebUI] SSE: token_iter type=%s refs_count=%s",
                type(token_iter).__name__,
                len(refs) if refs else 0,
            )
            if refs:
                refs_data = [r.model_dump() for r in refs]
                yield f"event: references\ndata: {json.dumps(refs_data)}\n\n"
                # Reconstruct references section for display rendering.
                # The answer text itself is kept clean (no ### References),
                # but the rendered HTML includes a references block.
                ref_lines = [f"[{r.id}] {r.title}" for r in refs]
                full_answer += "\n\n### References\n" + "\n".join(ref_lines)
```

With:

```python
            # References extracted by AnswerStream post-stream
            refs = getattr(token_iter, "references", None) or []
            clean_answer = getattr(token_iter, "answer", None) or full_answer

            logger.info("[WebUI] SSE: refs_count=%d", len(refs))

            if refs:
                refs_data = [r.model_dump() for r in refs]
                yield f"event: references\ndata: {json.dumps(refs_data)}\n\n"
                # Reconstruct references section for display rendering.
                ref_lines = [f"[{r.id}] {r.title}" for r in refs]
                clean_answer += "\n\n### References\n" + "\n".join(ref_lines)
```

**2b.** Update the `CitationProcessor.process` call (line ~193). Change:
```python
            result = processor.process(full_answer)
```
to:
```python
            result = processor.process(clean_answer)
```

**2c.** Remove the `parse_freetext_references` import from this file.

- [ ] **Step 3: Run all unit tests**

Run: `uv run pytest tests/unit/ -v`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add src/dlightrag/api/server.py src/dlightrag/web/routes.py
git commit -m "refactor: remove consumer-side ref fallback, use AnswerStream.answer"
```

---

### Task 5: Deprecate `parse_freetext_references` + final verification

**Files:**
- Modify: `src/dlightrag/citations/parser.py`

- [ ] **Step 1: Add deprecation note to `parse_freetext_references`**

In `src/dlightrag/citations/parser.py`, add a deprecation docstring to `parse_freetext_references`:

```python
def parse_freetext_references(raw: str) -> tuple[str, list[Reference]]:
    """Extract references from freetext LLM output.

    .. deprecated::
        Use :func:`extract_references` instead, which provides 3-level
        fallback (JSON block → ### References → empty).

    ...existing docstring...
    """
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/unit/ -v`
Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add src/dlightrag/citations/parser.py
git commit -m "docs: deprecate parse_freetext_references in favor of extract_references"
```
