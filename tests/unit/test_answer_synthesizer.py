# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the AnswerSynthesizer messages-first interface."""

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from dlightrag_agent.session.fold import PriorTurns
from dlightrag_ai.capacity import ModelProfile
from dlightrag_ai.scheduler import ModelScheduler
from dlightrag_rag.retrieval import RetrievalContexts

from dlightrag.answer.citations.streaming import AnswerStream
from dlightrag.answer.errors import AnswerInputOverflowError
from dlightrag.answer.synthesizer import NO_CONTEXT_DISCLAIMER, AnswerSynthesizer
from tests.unit.conftest import answer_image_policy, answer_model_profile

_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def _image_block(payload: str = _PNG_B64) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{payload}"}}


def _source_metadata(file_path: str, **extra: object) -> dict[str, object]:
    file_name = file_path.rsplit("/", 1)[-1]
    return {
        "source_uri": f"local://default/{file_name}",
        "source_download_locator": file_path,
        **extra,
    }


def _text_contexts() -> RetrievalContexts:
    return {
        "chunks": [
            {
                "chunk_id": "c1",
                "reference_id": "1",
                "file_path": "/docs/report.pdf",
                "content": "Revenue grew 15%.",
                "page_number": 3,
                "_workspace": "default",
                "metadata": _source_metadata("/docs/report.pdf"),
            },
        ],
        "entities": [
            {
                "entity_name": "Revenue",
                "entity_type": "Metric",
                "description": "Total revenue",
                "source_id": "c1",
            },
        ],
        "relationships": [],
    }


def _image_contexts() -> RetrievalContexts:
    return {
        "chunks": [
            {
                "chunk_id": "c1",
                "reference_id": "1",
                "file_path": "/docs/chart.pdf",
                "content": "Chart showing growth",
                "page_number": 1,
                "image_data": _PNG_B64,
                "_workspace": "default",
                "metadata": _source_metadata("/docs/chart.pdf"),
            },
        ],
        "entities": [],
        "relationships": [],
    }


def _multi_doc_contexts() -> RetrievalContexts:
    return {
        "chunks": [
            {
                "chunk_id": "c1",
                "reference_id": "1",
                "file_path": "/docs/report.pdf",
                "content": "Revenue data.",
                "page_number": 3,
                "image_data": _PNG_B64,
                "_workspace": "default",
                "metadata": _source_metadata("/docs/report.pdf", title="2025 Annual Report"),
            },
            {
                "chunk_id": "c2",
                "reference_id": "1",
                "file_path": "/docs/report.pdf",
                "content": "Expenses data.",
                "page_number": 7,
                "image_data": _PNG_B64,
                "_workspace": "default",
                "metadata": _source_metadata("/docs/report.pdf"),
            },
            {
                "chunk_id": "c3",
                "reference_id": "2",
                "file_path": "/docs/other.pdf",
                "content": "Other info.",
                "page_number": 1,
                "_workspace": "default",
                "metadata": _source_metadata("/docs/other.pdf"),
            },
        ],
        "entities": [],
        "relationships": [],
    }


# ---------------------------------------------------------------------------
# TestAnswerSynthesizerPolicy
# ---------------------------------------------------------------------------


def _stream_func(*tokens: str) -> AsyncMock:
    """A model callable that streams *tokens*, like every real answer provider."""

    async def _tokens() -> AsyncIterator[str]:
        for token in tokens:
            yield token

    return AsyncMock(side_effect=lambda **_kwargs: _tokens())


async def _drain(token_iter: object) -> str:
    return "".join([token async for token in cast(AsyncIterator[str], token_iter)])


class TestAnswerSynthesizerPolicy:
    """Evidence packing, image budgets, and citation cleanup on the one path."""

    @pytest.mark.asyncio
    async def test_evidence_is_prepared_off_the_event_loop(self, monkeypatch) -> None:
        import dlightrag.answer.synthesizer as answer_module

        calls = []

        async def fake_to_thread(func, *args, **kwargs):
            calls.append(func.__name__)
            return func(*args, **kwargs)

        monkeypatch.setattr(
            answer_module,
            "asyncio",
            SimpleNamespace(to_thread=fake_to_thread),
            raising=False,
        )
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=answer_model_profile(),
            model_func=_stream_func("a"),
        )

        await synth.generate_stream("query", _image_contexts())

        assert "_prepare_model_call" in calls

    @pytest.mark.asyncio
    async def test_images_are_sent_when_the_policy_allows_them(self) -> None:
        model_func = _stream_func("ok")
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(max_images=6),
            model_profile=answer_model_profile(),
            model_func=model_func,
        )

        await synth.generate_stream("describe", _image_contexts())

        user_content = model_func.call_args.kwargs["messages"][1]["content"]
        assert any(item.get("type") == "image_url" for item in user_content)
        assert any(item.get("type") == "text" for item in user_content)

    @pytest.mark.asyncio
    async def test_each_call_gets_a_fresh_budget_from_one_policy(self) -> None:
        policy = answer_image_policy(max_images=1)
        synth = AnswerSynthesizer(
            image_policy=policy,
            model_profile=answer_model_profile(),
            model_func=_stream_func("ok [1-1]."),
        )

        _, first = await synth.generate_stream("describe", _image_contexts())
        _, second = await synth.generate_stream("describe", _image_contexts())

        assert cast(Any, first).trace["answer_images_total"] == 1
        assert cast(Any, second).trace["answer_images_total"] == 1
        assert policy.max_images == 1

    @pytest.mark.asyncio
    async def test_returns_answer_packed_contexts(self) -> None:
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(max_images=0),
            model_profile=answer_model_profile(),
            model_func=_stream_func("answer"),
        )
        contexts: RetrievalContexts = {
            "chunks": [
                {
                    "chunk_id": "visual-only",
                    "reference_id": "1",
                    "file_path": "/docs/figures.pdf",
                    "content": "",
                    "image_data": _PNG_B64,
                    "_workspace": "default",
                    "metadata": _source_metadata("/docs/figures.pdf"),
                },
                *_text_contexts()["chunks"],
            ],
            "entities": [],
            "relationships": [],
        }

        packed, token_iter = await synth.generate_stream("q", contexts)

        assert packed is not contexts
        assert [c["chunk_id"] for c in packed["chunks"]] == ["c1"]
        assert cast(Any, token_iter).trace["answer_context_images_skipped"] == 1

    @pytest.mark.asyncio
    async def test_the_settled_answer_drops_a_model_generated_references_tail(self) -> None:
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=answer_model_profile(),
            model_func=_stream_func("Growth is 15% [1-1].", "\n\n### References\n- [1] x.pdf"),
        )

        _, token_iter = await synth.generate_stream("query", _text_contexts())
        await _drain(token_iter)

        settled = cast(Any, token_iter).answer
        assert "Growth is 15%" in settled
        assert "### References" not in settled


# ---------------------------------------------------------------------------
# TestAnswerSynthesizerStream
# ---------------------------------------------------------------------------


class TestAnswerSynthesizerStream:
    @pytest.mark.asyncio
    async def test_generate_stream_wraps_with_answer_stream(self) -> None:
        async def mock_tokens():
            for token in ["Hello", " world"]:
                yield token

        model_func = AsyncMock(return_value=mock_tokens())
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=answer_model_profile(),
            model_func=model_func,
        )

        _ctx, token_iter = await synth.generate_stream("test", _text_contexts())

        from dlightrag.answer.citations.streaming import AnswerStream

        assert isinstance(token_iter, AnswerStream)

    @pytest.mark.asyncio
    async def test_generate_stream_no_model_func(self) -> None:
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=answer_model_profile(),
            model_func=None,
        )
        contexts: RetrievalContexts = {"chunks": []}
        ctx, token_iter = await synth.generate_stream("test", contexts)
        assert token_iter is None
        assert ctx is contexts

    @pytest.mark.asyncio
    async def test_generate_stream_empty_context_disclaims_general_knowledge(self) -> None:
        async def mock_stream():
            yield "I am DlightRAG."

        model_func = AsyncMock(return_value=mock_stream())
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=answer_model_profile(),
            model_func=model_func,
        )
        contexts: RetrievalContexts = {"chunks": [], "entities": [], "relationships": []}

        ctx, token_iter = await synth.generate_stream("who are u", contexts)

        assert ctx == contexts
        assert token_iter is not None
        assert [token async for token in token_iter] == [
            f"{NO_CONTEXT_DISCLAIMER}\n\n",
            "I am DlightRAG.",
        ]

    @pytest.mark.asyncio
    async def test_closing_no_context_answer_releases_scheduled_model_stream(self) -> None:
        scheduler = ModelScheduler(max_concurrency=1)
        source_closed = asyncio.Event()
        competing_started = asyncio.Event()

        async def source():
            try:
                yield "first"
                await asyncio.Event().wait()
            finally:
                source_closed.set()

        async def model_func(**_kwargs: Any):
            return scheduler.stream(source)

        async def competing() -> None:
            competing_started.set()

        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=answer_model_profile(),
            model_func=model_func,
        )
        contexts: RetrievalContexts = {"chunks": [], "entities": [], "relationships": []}
        _ctx, token_iter = await synth.generate_stream("who are u", contexts)
        assert isinstance(token_iter, AnswerStream)
        assert await anext(token_iter) == f"{NO_CONTEXT_DISCLAIMER}\n\n"
        assert await anext(token_iter) == "first"
        waiting = asyncio.create_task(scheduler.run(competing))
        await asyncio.sleep(0)
        assert not competing_started.is_set()

        await token_iter.aclose()

        assert source_closed.is_set()
        await waiting
        assert competing_started.is_set()

    @pytest.mark.asyncio
    async def test_generate_stream_returns_packed_contexts_and_tokens(self) -> None:
        async def mock_stream():
            for token in ["Hello", " ", "world"]:
                yield token

        model_func = AsyncMock(return_value=mock_stream())
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=answer_model_profile(),
            model_func=model_func,
        )

        result_contexts, token_iter = await synth.generate_stream("query", _text_contexts())

        assert result_contexts is not _text_contexts()
        assert token_iter is not None
        assert [t async for t in token_iter] == ["Hello", " ", "world"]

    @pytest.mark.asyncio
    async def test_generate_stream_no_response_format_and_streams(self) -> None:
        async def mock_stream():
            yield "text"

        model_func = AsyncMock(return_value=mock_stream())
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=answer_model_profile(),
            model_func=model_func,
        )

        await synth.generate_stream("query", _text_contexts())

        call_kwargs = model_func.call_args.kwargs
        assert "response_format" not in call_kwargs
        assert call_kwargs.get("stream") is True

    @pytest.mark.asyncio
    async def test_generate_stream_passes_messages_and_indexer(self) -> None:
        async def mock_stream():
            yield "token"

        model_func = AsyncMock(return_value=mock_stream())
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=answer_model_profile(),
            model_func=model_func,
        )

        _, token_iter = await synth.generate_stream("query", _text_contexts())

        from dlightrag.answer.citations.streaming import AnswerStream

        messages = model_func.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert isinstance(token_iter, AnswerStream)
        assert token_iter._indexer is not None


# ---------------------------------------------------------------------------
# TestAnswerSynthesizerCapacity
# ---------------------------------------------------------------------------


class TestAnswerSynthesizerCapacity:
    def test_evidence_capacity_uses_the_full_residual_model_input(self) -> None:
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=ModelProfile(
                context_window_tokens=10_000,
                max_input_tokens=9_000,
                max_output_tokens=1_000,
            ),
        )

        prepared = synth._prepare_model_call(
            "question", _text_contexts(), conversation_history=PriorTurns()
        )

        assert prepared.trace["answer_input_limit_tokens"] == 8_500
        assert prepared.trace["context_policy_revision"] == "m1-v1"
        assert prepared.trace["answer_evidence_capacity_tokens"] == 8_500 - (
            prepared.trace["answer_input_tokens"] - prepared.trace["answer_evidence_tokens"]
        )
        assert prepared.trace["answer_evidence_capacity_tokens"] > 6_000

    def test_fixed_evidence_overflow_raises_without_trimming_evidence(self) -> None:
        contexts: RetrievalContexts = {
            "chunks": [
                {
                    "chunk_id": "rag-large",
                    "reference_id": "rag-doc",
                    "file_path": "rag.pdf",
                    "content": "immutable workspace evidence " * 50,
                }
            ],
            "entities": [],
            "relationships": [],
        }
        original_content = contexts["chunks"][0]["content"]
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=answer_model_profile(context_window_tokens=200),
        )

        with pytest.raises(AnswerInputOverflowError):
            synth._prepare_model_call("question", contexts, conversation_history=PriorTurns())

        assert contexts["chunks"][0]["content"] == original_content

    def test_pinned_history_is_not_locally_trimmed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import dlightrag.answer.synthesizer as answer_module

        monkeypatch.setattr(answer_module, "answer_core", lambda: "SYS")

        contexts: RetrievalContexts = {
            "chunks": [
                {
                    "chunk_id": "rag-1",
                    "reference_id": "rag-doc",
                    "file_path": "rag.pdf",
                    "content": "workspace evidence",
                    "metadata": {
                        "source_uri": "file:///rag.pdf",
                        "source_download_locator": "rag.pdf",
                    },
                    "_workspace": "default",
                }
            ],
            "entities": [],
            "relationships": [],
        }
        original_chunks = [dict(contexts["chunks"][0])]
        # A window whose input budget only fits recent history plus fixed input.
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=answer_model_profile(context_window_tokens=500),
        )

        history = PriorTurns(
            [
                {"role": "user", "content": "OLD-HISTORY " + ("old " * 200)},
                {"role": "assistant", "content": "old answer " * 60},
                {"role": "user", "content": "RECENT-HISTORY follow-up"},
                {"role": "assistant", "content": "recent answer"},
            ]
        )

        with pytest.raises(AnswerInputOverflowError):
            synth._prepare_model_call(
                "current question",
                contexts,
                conversation_history=history,
            )

        assert len(history) == 4
        assert contexts["chunks"] == original_chunks


# ---------------------------------------------------------------------------
# TestAnswerSynthesizerHelpers
# ---------------------------------------------------------------------------


class TestAnswerSynthesizerHelpers:
    def test_format_kg_context_with_entities_and_rels(self) -> None:
        contexts: RetrievalContexts = {
            "chunks": [],
            "entities": [
                {
                    "entity_name": "Acme",
                    "entity_type": "Company",
                    "description": "A company",
                    "source_id": "s1",
                },
            ],
            "relationships": [
                {
                    "src_id": "Acme",
                    "tgt_id": "Revenue",
                    "description": "generates",
                    "source_id": "s1",
                },
            ],
        }
        result = AnswerSynthesizer._format_kg_context(contexts)
        assert "## Entities" in result
        assert "**Acme**" in result
        assert "## Relationships" in result
        assert "Acme -> Revenue" in result

    def test_format_kg_context_empty(self) -> None:
        contexts: RetrievalContexts = {"chunks": [], "entities": [], "relationships": []}
        assert (
            AnswerSynthesizer._format_kg_context(contexts)
            == "No knowledge graph context available."
        )

    def test_format_kg_context_includes_doc_level_tags(self) -> None:
        from dlightrag.answer.citations.indexer import CitationIndexer

        contexts: RetrievalContexts = {
            "chunks": [
                {
                    "chunk_id": "c1",
                    "reference_id": "1",
                    "file_path": "/docs/report.pdf",
                    "content": "Revenue data.",
                    "page_number": 1,
                },
            ],
            "entities": [
                {
                    "entity_name": "Revenue",
                    "entity_type": "Metric",
                    "description": "Total revenue grew 15%",
                    "source_id": "c1",
                },
            ],
            "relationships": [
                {
                    "src_id": "Acme",
                    "tgt_id": "Revenue",
                    "description": "reports",
                    "source_id": "c1",
                },
            ],
        }
        flat: list = []
        for items in contexts.values():
            if isinstance(items, list):
                flat.extend(items)
        indexer = CitationIndexer()
        indexer.build_index(flat)

        result = AnswerSynthesizer._format_kg_context(contexts, indexer=indexer)
        assert "(from [1])" in result

    def test_build_citation_indexer(self) -> None:
        indexer = AnswerSynthesizer._build_citation_indexer(_text_contexts())
        assert indexer.get_max_chunk_idx("1") > 0


# ---------------------------------------------------------------------------
# TestBuildExcerptBlocks
# ---------------------------------------------------------------------------


class TestBuildExcerptBlocks:
    def test_groups_chunks_by_document(self) -> None:
        from dlightrag.answer.citations.indexer import CitationIndexer

        contexts = _multi_doc_contexts()
        indexer = CitationIndexer()
        indexer.build_index(list(contexts["chunks"]))

        blocks = AnswerSynthesizer._build_excerpt_blocks(contexts, indexer=indexer)

        all_text = "\n".join(b["text"] for b in blocks if b.get("type") == "text")
        assert "Document [1]" in all_text
        assert "Document [2]" in all_text
        assert "report.pdf" in all_text
        assert "other.pdf" in all_text

    def test_images_interleaved_with_document(self) -> None:
        contexts = _multi_doc_contexts()
        blocks = AnswerSynthesizer._build_excerpt_blocks(contexts)

        image_blocks = [b for b in blocks if b.get("type") == "image_url"]
        assert len(image_blocks) == 2

    def test_empty_chunks_returns_empty_blocks(self) -> None:
        assert AnswerSynthesizer._build_excerpt_blocks({"chunks": []}) == []
