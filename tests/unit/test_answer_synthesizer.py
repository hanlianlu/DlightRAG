# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the AnswerSynthesizer messages-first interface."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from dlightrag.core.answer.errors import AnswerInputOverflowError, CurrentImagePayloadError
from dlightrag.core.answer.synthesizer import NO_CONTEXT_DISCLAIMER, AnswerSynthesizer
from dlightrag.core.retrieval.protocols import RetrievalContexts
from dlightrag.utils.images import MODEL_IMAGE_MAX_PIXELS

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
# TestAnswerSynthesizerGenerate
# ---------------------------------------------------------------------------


class TestAnswerSynthesizerGenerate:
    @pytest.mark.asyncio
    async def test_generate_with_freetext_response(self) -> None:
        raw = "AI is artificial intelligence [1-1].\n\n### References\n- [1] AI Overview"
        model_func = AsyncMock(return_value=raw)
        synth = AnswerSynthesizer(image_max_pixels=MODEL_IMAGE_MAX_PIXELS, model_func=model_func)

        result = await synth.generate("What is AI?", _text_contexts())

        assert result.answer is not None
        assert "AI is artificial intelligence" in result.answer
        assert "### References" not in result.answer
        assert len(result.references) == 1
        assert result.warnings == []
        call_kwargs = model_func.call_args.kwargs
        assert "messages" in call_kwargs
        assert "response_format" not in call_kwargs

    @pytest.mark.asyncio
    async def test_generate_preserves_citation_reference_ids(self) -> None:
        model_func = AsyncMock(return_value="The other document applies here [2-1].")
        synth = AnswerSynthesizer(image_max_pixels=MODEL_IMAGE_MAX_PIXELS, model_func=model_func)

        result = await synth.generate("Which document applies?", _multi_doc_contexts())

        assert len(result.references) == 1
        assert result.references[0].id == "2"
        assert result.references[0].title == "other.pdf"

    @pytest.mark.asyncio
    async def test_generate_passes_prepared_indexer_to_finalize_answer(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dlightrag.core.answer.synthesizer as answer_module
        from dlightrag.citations.indexer import CitationIndexer

        sentinel_indexer = CitationIndexer()
        prepared_contexts = _text_contexts()
        prepared = answer_module._PreparedModelCall(
            contexts=prepared_contexts,
            messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}],
            indexer=sentinel_indexer,
            trace={
                "answer_context_chunks": 1,
                "answer_context_images_sent": 0,
                "answer_context_images_skipped": 0,
            },
            no_context=False,
        )

        async def fake_to_thread(func, *args, **kwargs):  # noqa: ANN001, ANN202
            return prepared

        finalize_answer = Mock(return_value=SimpleNamespace(answer="done", sources=[]))
        monkeypatch.setattr(
            answer_module,
            "asyncio",
            SimpleNamespace(to_thread=fake_to_thread),
            raising=False,
        )
        monkeypatch.setattr("dlightrag.citations.finalize_answer", finalize_answer)

        model_func = AsyncMock(return_value="raw answer")
        synth = AnswerSynthesizer(image_max_pixels=MODEL_IMAGE_MAX_PIXELS, model_func=model_func)

        await synth.generate("query", _text_contexts())

        finalize_answer.assert_called_once_with(
            "raw answer",
            prepared_contexts,
            indexer=sentinel_indexer,
        )

    @pytest.mark.asyncio
    async def test_generate_no_model_func(self) -> None:
        synth = AnswerSynthesizer(image_max_pixels=MODEL_IMAGE_MAX_PIXELS, model_func=None)
        contexts: RetrievalContexts = {"chunks": []}
        result = await synth.generate("test", contexts, warnings=["kept"])
        assert result.answer is None
        assert result.contexts is contexts
        assert result.warnings == ["kept"]

    @pytest.mark.asyncio
    async def test_generate_empty_context_disclaims_general_knowledge(self) -> None:
        model_func = AsyncMock(return_value="The capital of France is Paris.")
        synth = AnswerSynthesizer(image_max_pixels=MODEL_IMAGE_MAX_PIXELS, model_func=model_func)
        contexts: RetrievalContexts = {"chunks": [], "entities": [], "relationships": []}

        result = await synth.generate("what is the capital of France?", contexts)

        assert result.answer == f"{NO_CONTEXT_DISCLAIMER}\n\nThe capital of France is Paris."
        assert result.trace["answer_no_context"] is True
        model_func.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_forwards_incoming_warnings_to_result(self) -> None:
        model_func = AsyncMock(return_value="answer [1-1].")
        synth = AnswerSynthesizer(image_max_pixels=MODEL_IMAGE_MAX_PIXELS, model_func=model_func)

        result = await synth.generate("q", _text_contexts(), warnings=["one attachment truncated"])

        assert result.warnings == ["one attachment truncated"]

    @pytest.mark.asyncio
    async def test_generate_empty_chunks_with_query_image_still_calls_model(self) -> None:
        model_func = AsyncMock(return_value="The image shows a chart.")
        synth = AnswerSynthesizer(
            image_max_pixels=MODEL_IMAGE_MAX_PIXELS,
            model_func=model_func,
            effective_max_images=3,
        )
        contexts: RetrievalContexts = {"chunks": [], "entities": [], "relationships": []}

        await synth.generate("describe this image", contexts, query_images=[_image_block()])

        model_func.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_images(self) -> None:
        raw = "ok\n\n### References\n- [1] chart.pdf"
        model_func = AsyncMock(return_value=raw)
        synth = AnswerSynthesizer(
            image_max_pixels=MODEL_IMAGE_MAX_PIXELS,
            model_func=model_func,
            effective_max_images=6,
        )

        await synth.generate("describe", _image_contexts())

        messages = model_func.call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        assert any(item.get("type") == "image_url" for item in user_content)
        assert any(item.get("type") == "text" for item in user_content)

    @pytest.mark.asyncio
    async def test_generate_returns_structured_answer_images_for_sdk(self) -> None:
        model_func = AsyncMock(return_value="The chart shows growth [1-1].")
        synth = AnswerSynthesizer(
            image_max_pixels=MODEL_IMAGE_MAX_PIXELS,
            model_func=model_func,
            effective_max_images=6,
        )

        result = await synth.generate("describe", _image_contexts())

        assert result.answer_images == [
            {
                "id": "c1",
                "chunk_id": "c1",
                "source_ref": "1-1",
                "url": "/images/default/c1?size=full",
                "thumbnail_url": "/images/default/c1?size=thumb",
                "label": "chart.pdf · Page 1",
                "answer_image_sent": True,
            }
        ]
        assert result.answer_blocks == [
            {"type": "markdown", "text": "The chart shows growth [1-1]."},
            {"type": "image_ref", "image_id": "c1"},
        ]

    @pytest.mark.asyncio
    async def test_current_image_and_evidence_share_one_budget(self) -> None:
        model_func = AsyncMock(return_value="ok")
        synth = AnswerSynthesizer(
            image_max_pixels=MODEL_IMAGE_MAX_PIXELS,
            model_func=model_func,
            effective_max_images=2,
        )
        contexts: RetrievalContexts = {
            "chunks": [
                {
                    "chunk_id": "visual-only",
                    "reference_id": "1",
                    "file_path": "/docs/chart.pdf",
                    "content": "",
                    "image_data": _PNG_B64,
                    "_workspace": "default",
                    "metadata": _source_metadata("/docs/chart.pdf"),
                }
            ],
            "entities": [],
            "relationships": [],
        }

        result = await synth.generate("describe this", contexts, query_images=[_image_block()])

        assert [chunk["chunk_id"] for chunk in result.contexts["chunks"]] == ["visual-only"]
        assert result.trace["answer_images_current"] == 1
        assert result.trace["answer_images_rag"] == 1
        assert result.trace["answer_context_images_sent"] == 1
        assert result.trace["answer_context_images_skipped"] == 0
        assert "answer_composer_image_budget_used_bytes" not in result.trace
        messages = model_func.call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        assert sum(1 for item in user_content if item.get("type") == "image_url") == 2
        assert any(
            item.get("type") == "text" and "User-attached images" in item.get("text", "")
            for item in user_content
        )

    @pytest.mark.asyncio
    async def test_generate_no_response_format(self) -> None:
        raw = "Revenue grew 15% [1-1].\n\n### References\n- [1] report.pdf"
        model_func = AsyncMock(return_value=raw)
        synth = AnswerSynthesizer(image_max_pixels=MODEL_IMAGE_MAX_PIXELS, model_func=model_func)

        await synth.generate("query", _text_contexts())

        assert "response_format" not in model_func.call_args.kwargs

    @pytest.mark.asyncio
    async def test_generate_returns_answer_packed_contexts(self) -> None:
        raw = "answer\n\n### References\n- [1] report.pdf"
        model_func = AsyncMock(return_value=raw)
        synth = AnswerSynthesizer(
            image_max_pixels=MODEL_IMAGE_MAX_PIXELS,
            model_func=model_func,
            effective_max_images=0,
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
        result = await synth.generate("q", contexts)

        assert result.contexts is not contexts
        assert [c["chunk_id"] for c in result.contexts["chunks"]] == ["c1"]
        assert result.trace["answer_context_images_skipped"] == 1

    @pytest.mark.asyncio
    async def test_generate_prepares_model_payload_off_event_loop(self, monkeypatch) -> None:
        import dlightrag.core.answer.synthesizer as answer_module

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
        model_func = AsyncMock(return_value="answer")
        synth = AnswerSynthesizer(image_max_pixels=MODEL_IMAGE_MAX_PIXELS, model_func=model_func)

        await synth.generate("query", _image_contexts())

        assert "_prepare_model_call" in calls

    @pytest.mark.asyncio
    async def test_generate_strips_model_generated_references_tail(self) -> None:
        raw = "Growth is 15% [1-1].\n\n### References\n- [1] report.pdf"
        model_func = AsyncMock(return_value=raw)
        synth = AnswerSynthesizer(image_max_pixels=MODEL_IMAGE_MAX_PIXELS, model_func=model_func)

        result = await synth.generate("query", _text_contexts())

        assert result.answer is not None
        assert "Growth is 15%" in result.answer
        assert "### References" not in result.answer
        assert len(result.references) == 1
        assert result.references[0].title == "report.pdf"


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
        synth = AnswerSynthesizer(image_max_pixels=MODEL_IMAGE_MAX_PIXELS, model_func=model_func)

        _ctx, token_iter = await synth.generate_stream("test", _text_contexts())

        from dlightrag.citations.streaming import AnswerStream

        assert isinstance(token_iter, AnswerStream)

    @pytest.mark.asyncio
    async def test_generate_stream_no_model_func(self) -> None:
        synth = AnswerSynthesizer(image_max_pixels=MODEL_IMAGE_MAX_PIXELS, model_func=None)
        contexts: RetrievalContexts = {"chunks": []}
        ctx, token_iter = await synth.generate_stream("test", contexts)
        assert token_iter is None
        assert ctx is contexts

    @pytest.mark.asyncio
    async def test_generate_stream_exposes_warnings_on_stream(self) -> None:
        async def mock_stream():
            yield "text"

        model_func = AsyncMock(return_value=mock_stream())
        synth = AnswerSynthesizer(image_max_pixels=MODEL_IMAGE_MAX_PIXELS, model_func=model_func)

        _ctx, token_iter = await synth.generate_stream(
            "q", _text_contexts(), warnings=["stream warning"]
        )

        assert token_iter is not None
        assert cast(Any, token_iter).warnings == ["stream warning"]

    @pytest.mark.asyncio
    async def test_generate_stream_empty_context_disclaims_general_knowledge(self) -> None:
        async def mock_stream():
            yield "I am DlightRAG."

        model_func = AsyncMock(return_value=mock_stream())
        synth = AnswerSynthesizer(image_max_pixels=MODEL_IMAGE_MAX_PIXELS, model_func=model_func)
        contexts: RetrievalContexts = {"chunks": [], "entities": [], "relationships": []}

        ctx, token_iter = await synth.generate_stream("who are u", contexts)

        assert ctx == contexts
        assert token_iter is not None
        assert [token async for token in token_iter] == [
            f"{NO_CONTEXT_DISCLAIMER}\n\n",
            "I am DlightRAG.",
        ]

    @pytest.mark.asyncio
    async def test_generate_stream_returns_packed_contexts_and_tokens(self) -> None:
        async def mock_stream():
            for token in ["Hello", " ", "world"]:
                yield token

        model_func = AsyncMock(return_value=mock_stream())
        synth = AnswerSynthesizer(image_max_pixels=MODEL_IMAGE_MAX_PIXELS, model_func=model_func)

        result_contexts, token_iter = await synth.generate_stream("query", _text_contexts())

        assert result_contexts is not _text_contexts()
        assert token_iter is not None
        assert [t async for t in token_iter] == ["Hello", " ", "world"]

    @pytest.mark.asyncio
    async def test_generate_stream_no_response_format_and_streams(self) -> None:
        async def mock_stream():
            yield "text"

        model_func = AsyncMock(return_value=mock_stream())
        synth = AnswerSynthesizer(image_max_pixels=MODEL_IMAGE_MAX_PIXELS, model_func=model_func)

        await synth.generate_stream("query", _text_contexts())

        call_kwargs = model_func.call_args.kwargs
        assert "response_format" not in call_kwargs
        assert call_kwargs.get("stream") is True

    @pytest.mark.asyncio
    async def test_generate_stream_passes_messages_and_indexer(self) -> None:
        async def mock_stream():
            yield "token"

        model_func = AsyncMock(return_value=mock_stream())
        synth = AnswerSynthesizer(image_max_pixels=MODEL_IMAGE_MAX_PIXELS, model_func=model_func)

        _, token_iter = await synth.generate_stream("query", _text_contexts())

        from dlightrag.citations.streaming import AnswerStream

        messages = model_func.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert isinstance(token_iter, AnswerStream)
        assert token_iter._indexer is not None

    @pytest.mark.asyncio
    async def test_stream_and_nonstream_prepare_byte_identical_evidence(self) -> None:
        captured: dict[str, object] = {}

        async def capture_nonstream(*, messages):
            captured["nonstream"] = messages
            return "answer [1-1]."

        async def stream_tokens():
            yield "answer [1-1]."

        async def capture_stream(*, messages, stream):
            captured["stream"] = messages
            return stream_tokens()

        contexts = _multi_doc_contexts()

        synth_a = AnswerSynthesizer(
            image_max_pixels=MODEL_IMAGE_MAX_PIXELS,
            model_func=capture_nonstream,
            effective_max_images=6,
        )
        await synth_a.generate("compare", contexts)

        synth_b = AnswerSynthesizer(
            image_max_pixels=MODEL_IMAGE_MAX_PIXELS,
            model_func=capture_stream,
            effective_max_images=6,
        )
        await synth_b.generate_stream("compare", contexts)

        assert captured["stream"] == captured["nonstream"]


# ---------------------------------------------------------------------------
# TestAnswerSynthesizerImageBudget
# ---------------------------------------------------------------------------


class TestAnswerSynthesizerImageBudget:
    def test_history_image_blocks_are_budgeted(self) -> None:
        synth = AnswerSynthesizer(
            image_max_pixels=MODEL_IMAGE_MAX_PIXELS,
            effective_max_images=3,
        )

        prepared = synth._prepare_model_call(
            "prompt",
            {"chunks": []},
            conversation_history=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "previous"}, _image_block()],
                }
            ],
        )

        history_content = prepared.messages[1]["content"]
        assert any(block.get("type") == "image_url" for block in history_content)

    def test_current_query_images_reserve_before_history(self) -> None:
        synth = AnswerSynthesizer(
            image_max_pixels=MODEL_IMAGE_MAX_PIXELS,
            effective_max_images=1,
        )

        prepared = synth._prepare_model_call(
            "prompt",
            {"chunks": []},
            query_images=[_image_block()],
            conversation_history=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "previous"}, _image_block()],
                }
            ],
        )

        history_content = prepared.messages[1]["content"]
        final_user_content = prepared.messages[2]["content"]
        assert not any(block.get("type") == "image_url" for block in history_content)
        assert sum(1 for block in final_user_content if block.get("type") == "image_url") == 1
        assert prepared.trace["answer_images_current"] == 1

    def test_current_image_count_overflow_is_strict(self) -> None:
        synth = AnswerSynthesizer(
            image_max_pixels=MODEL_IMAGE_MAX_PIXELS,
            effective_max_images=1,
        )

        with pytest.raises(CurrentImagePayloadError, match="2 current-turn images"):
            synth._prepare_model_call(
                "prompt",
                {"chunks": []},
                query_images=[_image_block(), _image_block()],
            )

    def test_current_image_byte_overflow_is_strict(self) -> None:
        synth = AnswerSynthesizer(
            image_max_pixels=MODEL_IMAGE_MAX_PIXELS,
            effective_max_images=1,
            image_max_total_bytes=1,
        )

        with pytest.raises(CurrentImagePayloadError, match="current image query_image_1"):
            synth._prepare_model_call(
                "prompt",
                {"chunks": []},
                query_images=[_image_block()],
            )

    def test_single_budget_allocates_current_then_history_then_evidence(self) -> None:
        contexts: RetrievalContexts = {
            "chunks": [
                {
                    "chunk_id": "c1",
                    "reference_id": "1",
                    "content": "chart",
                    "image_data": _PNG_B64,
                    "file_path": "/docs/report.pdf",
                }
            ]
        }
        synth = AnswerSynthesizer(
            image_max_pixels=MODEL_IMAGE_MAX_PIXELS,
            effective_max_images=2,
        )

        prepared = synth._prepare_model_call(
            "prompt",
            contexts,
            query_images=[_image_block()],
            conversation_history=[
                {"role": "user", "content": [{"type": "text", "text": "prev"}, _image_block()]}
            ],
        )

        total_images = sum(
            1
            for message in prepared.messages
            if isinstance(message["content"], list)
            for block in message["content"]
            if isinstance(block, dict) and block.get("type") == "image_url"
        )
        assert total_images == 2
        assert prepared.trace["answer_images_current"] == 1
        assert prepared.trace["answer_images_history"] == 1
        assert prepared.trace["answer_images_rag"] == 0
        assert prepared.trace["answer_images_total"] == 2
        assert "answer_composer_image_budget_used_bytes" not in prepared.trace

    def test_selected_history_image_has_history_trace(self) -> None:
        prepared = AnswerSynthesizer(
            image_max_pixels=MODEL_IMAGE_MAX_PIXELS,
            effective_max_images=2,
        )._prepare_model_call(
            "prompt",
            {"chunks": []},
            query_images=[_image_block()],
            history_images=[_image_block()],
        )

        assert prepared.trace["answer_images_current"] == 1
        assert prepared.trace["answer_images_history"] == 1
        assert prepared.trace["answer_images_total"] == 2
        final_content = prepared.messages[-1]["content"]
        image_blocks = [block for block in final_content if block.get("type") == "image_url"]
        assert len(image_blocks) == 2
        assert any(
            block.get("text") == "## Referenced conversation images\n" for block in final_content
        )

    def test_selected_history_image_is_answer_evidence_without_current_image(self) -> None:
        prepared = AnswerSynthesizer(
            image_max_pixels=MODEL_IMAGE_MAX_PIXELS,
            effective_max_images=1,
        )._prepare_model_call(
            "prompt",
            {"chunks": []},
            history_images=[_image_block()],
        )

        assert prepared.no_context is False
        assert "answer_no_context" not in prepared.trace
        assert prepared.trace["answer_images_current"] == 0
        assert prepared.trace["answer_images_history"] == 1


# ---------------------------------------------------------------------------
# TestAnswerSynthesizerCapacity
# ---------------------------------------------------------------------------


class TestAnswerSynthesizerCapacity:
    def test_default_evidence_ceiling_is_156000(self) -> None:
        synth = AnswerSynthesizer(image_max_pixels=MODEL_IMAGE_MAX_PIXELS)

        prepared = synth._prepare_model_call("question", _text_contexts(), conversation_history=[])

        assert prepared.trace["answer_context_window_tokens"] == 260_000
        assert prepared.trace["answer_evidence_ceiling"] == 156_000

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
            image_max_pixels=MODEL_IMAGE_MAX_PIXELS,
            context_window_tokens=200,
        )

        with pytest.raises(AnswerInputOverflowError):
            synth._prepare_model_call("question", contexts, conversation_history=[])

        assert contexts["chunks"][0]["content"] == original_content

    def test_oldest_history_dropped_first_without_mutating_evidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dlightrag.core.answer.synthesizer as answer_module

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
            image_max_pixels=MODEL_IMAGE_MAX_PIXELS,
            context_window_tokens=33_000,
        )

        prepared = synth._prepare_model_call(
            "current question",
            contexts,
            conversation_history=[
                {"role": "user", "content": "OLD-HISTORY " + ("old " * 200)},
                {"role": "assistant", "content": "old answer " * 60},
                {"role": "user", "content": "RECENT-HISTORY follow-up"},
                {"role": "assistant", "content": "recent answer"},
            ],
        )

        history_text = "\n".join(str(message["content"]) for message in prepared.messages[1:-1])
        assert "OLD-HISTORY" not in history_text
        assert "RECENT-HISTORY" in history_text
        assert contexts["chunks"] == original_chunks
        assert prepared.trace["answer_history_messages_dropped"] == 2


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
        from dlightrag.citations.indexer import CitationIndexer

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
        from dlightrag.citations.indexer import CitationIndexer

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
