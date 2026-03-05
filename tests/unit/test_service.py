# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for RAGService facade."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dlightrag.config import DlightragConfig
from dlightrag.service import RAGService

# ---------------------------------------------------------------------------
# TestRAGServiceAingest
# ---------------------------------------------------------------------------


class TestRAGServiceAingest:
    """Test ingestion logic — replace defaults, azure lifecycle."""

    def _make_initialized_service(self, config: DlightragConfig) -> RAGService:
        service = RAGService(config=config)
        service._initialized = True
        service.ingestion = MagicMock()
        service.ingestion.aingest_from_local = AsyncMock(
            return_value=MagicMock(model_dump=MagicMock(return_value={"status": "success"}))
        )
        service.ingestion.aingest_from_azure_blob = AsyncMock(
            return_value=MagicMock(model_dump=MagicMock(return_value={"status": "success"}))
        )
        service.rag_text = MagicMock()
        service.rag_vision = MagicMock()
        return service

    async def test_aingest_not_initialized_raises(self, test_config: DlightragConfig) -> None:
        service = RAGService(config=test_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            await service.aingest(source_type="local", path="/tmp/f.pdf")

    async def test_aingest_replace_default_from_config(self, test_config: DlightragConfig) -> None:
        test_config.ingestion_replace_default = True
        service = self._make_initialized_service(test_config)
        await service.aingest(source_type="local", path="/tmp/file.pdf")
        call_kwargs = service.ingestion.aingest_from_local.call_args
        assert call_kwargs.kwargs["replace"] is True

    async def test_aingest_replace_explicit_overrides_config(
        self, test_config: DlightragConfig
    ) -> None:
        test_config.ingestion_replace_default = True
        service = self._make_initialized_service(test_config)
        await service.aingest(source_type="local", path="/tmp/file.pdf", replace=False)
        call_kwargs = service.ingestion.aingest_from_local.call_args
        assert call_kwargs.kwargs["replace"] is False

    # -- Azure blob lifecycle --

    async def test_aingest_azure_defaults_prefix_when_neither_set(
        self, test_config: DlightragConfig
    ) -> None:
        """When neither blob_path nor prefix provided, prefix defaults to '' (entire container)."""
        service = self._make_initialized_service(test_config)
        mock_source = AsyncMock()
        await service.aingest(source_type="azure_blob", source=mock_source, container_name="c")
        call_kwargs = service.ingestion.aingest_from_azure_blob.call_args.kwargs
        assert call_kwargs["prefix"] == ""
        assert call_kwargs.get("blob_path") is None

    async def test_aingest_azure_calls_aclose(self, test_config: DlightragConfig) -> None:
        """source.aclose() is called after successful ingestion."""
        service = self._make_initialized_service(test_config)
        mock_source = AsyncMock()
        await service.aingest(source_type="azure_blob", source=mock_source, container_name="c")
        mock_source.aclose.assert_awaited_once()

    async def test_aingest_azure_calls_aclose_on_error(self, test_config: DlightragConfig) -> None:
        """source.aclose() is called even when ingestion raises."""
        service = self._make_initialized_service(test_config)
        service.ingestion.aingest_from_azure_blob = AsyncMock(
            side_effect=RuntimeError("ingestion failed")
        )
        mock_source = AsyncMock()
        with pytest.raises(RuntimeError, match="ingestion failed"):
            await service.aingest(source_type="azure_blob", source=mock_source, container_name="c")
        mock_source.aclose.assert_awaited_once()


# ---------------------------------------------------------------------------
# TestRAGServiceRerank
# ---------------------------------------------------------------------------


class TestRAGServiceRerank:
    """Test _rerank_chunks with mocked LLM."""

    async def test_rerank_sorts_by_relevance(self, test_config: DlightragConfig) -> None:
        service = RAGService(config=test_config)
        service._initialized = True

        chunks = [
            {"content": "low relevance"},
            {"content": "high relevance"},
        ]

        async def mock_rerank(**kwargs):
            return [{"index": 1}, {"index": 0}]

        with patch("dlightrag.service.get_rerank_func", return_value=mock_rerank):
            result = await service._rerank_chunks(chunks, "query")

        assert result[0]["content"] == "high relevance"
        assert result[1]["content"] == "low relevance"

    async def test_rerank_failure_returns_original(self, test_config: DlightragConfig) -> None:
        service = RAGService(config=test_config)
        service._initialized = True

        chunks = [{"content": "a"}, {"content": "b"}]

        async def mock_rerank(**kwargs):
            raise RuntimeError("LLM error")

        with patch("dlightrag.service.get_rerank_func", return_value=mock_rerank):
            result = await service._rerank_chunks(chunks, "query")

        assert result == chunks


# ---------------------------------------------------------------------------
# TestRAGServiceClose
# ---------------------------------------------------------------------------


class TestRAGServiceClose:
    """Test cleanup logic."""

    async def test_close_handles_errors(self, test_config: DlightragConfig) -> None:
        service = RAGService(config=test_config)
        service._initialized = True
        mock_ingestion = MagicMock()
        mock_ingestion.rag = MagicMock()
        mock_ingestion.rag.finalize_storages = AsyncMock(side_effect=RuntimeError("cleanup failed"))
        service.ingestion = mock_ingestion
        service.rag_text = None
        service.rag_vision = None

        # Should not raise
        await service.close()


# ---------------------------------------------------------------------------
# TestRAGServiceRetrieve
# ---------------------------------------------------------------------------


class TestRAGServiceRetrieve:
    """Test aretrieve and aanswer dispatch logic."""

    def _make_retrieval_service(self, config: DlightragConfig) -> RAGService:
        service = RAGService(config=config)
        service._initialized = True
        service.enable_rerank = False

        rag_text = MagicMock()
        rag_text.aquery_data_with_multimodal = AsyncMock(return_value=MagicMock())
        rag_text.lightrag = MagicMock()

        rag_vision = MagicMock()
        rag_vision.aquery_data_with_multimodal = AsyncMock(return_value=MagicMock())
        rag_vision.lightrag = MagicMock()

        service.rag_text = rag_text
        service.rag_vision = rag_vision
        service.ingestion = MagicMock()
        return service

    @patch("dlightrag.service.augment_retrieval_result", new_callable=AsyncMock,
           return_value=MagicMock())
    async def test_aretrieve_uses_text_rag_by_default(self, mock_augment, test_config):
        service = self._make_retrieval_service(test_config)
        await service.aretrieve("test query")
        service.rag_text.aquery_data_with_multimodal.assert_awaited_once()
        service.rag_vision.aquery_data_with_multimodal.assert_not_awaited()

    @patch("dlightrag.service.augment_retrieval_result", new_callable=AsyncMock,
           return_value=MagicMock())
    async def test_aretrieve_uses_vision_rag_with_multimodal(self, mock_augment, test_config):
        service = self._make_retrieval_service(test_config)
        await service.aretrieve("test query", multimodal_content=[{"type": "image"}])
        service.rag_vision.aquery_data_with_multimodal.assert_awaited_once()
        service.rag_text.aquery_data_with_multimodal.assert_not_awaited()

    async def test_aretrieve_not_initialized_raises(self, test_config):
        service = RAGService(config=test_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            await service.aretrieve("query")


# ---------------------------------------------------------------------------
# TestConversationHistoryTruncation
# ---------------------------------------------------------------------------


class TestConversationHistoryTruncation:
    """Test aanswer conversation history truncation logic."""

    def _make_answer_service(self, config: DlightragConfig) -> RAGService:
        service = RAGService(config=config)
        service._initialized = True
        service.enable_rerank = False

        rag = MagicMock()
        rag.aquery_llm_with_multimodal = AsyncMock(return_value=MagicMock())
        rag.lightrag = MagicMock()

        service.rag_text = rag
        service.rag_vision = rag
        service.ingestion = MagicMock()
        return service

    @patch("dlightrag.service.augment_retrieval_result", new_callable=AsyncMock,
           return_value=MagicMock())
    async def test_history_truncated_by_turns(self, mock_augment, test_config):
        """History exceeding max_conversation_turns*2 is truncated from front."""
        test_config.max_conversation_turns = 2  # max 4 messages
        service = self._make_answer_service(test_config)

        history = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
        await service.aanswer("query", conversation_history=history)

        call_kwargs = service.rag_text.aquery_llm_with_multimodal.call_args.kwargs
        passed_history = call_kwargs.get("conversation_history", [])
        assert len(passed_history) <= 4

    @patch("dlightrag.service.augment_retrieval_result", new_callable=AsyncMock,
           return_value=MagicMock())
    async def test_none_history_passes_through(self, mock_augment, test_config):
        """None history does not add conversation_history kwarg."""
        service = self._make_answer_service(test_config)
        await service.aanswer("query", conversation_history=None)

        call_kwargs = service.rag_text.aquery_llm_with_multimodal.call_args.kwargs
        assert "conversation_history" not in call_kwargs


# ---------------------------------------------------------------------------
# TestRAGServiceFileManagement
# ---------------------------------------------------------------------------


class TestRAGServiceFileManagement:
    """Test alist_ingested_files and adelete_files delegation."""

    async def test_alist_not_initialized_raises(self, test_config):
        service = RAGService(config=test_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            await service.alist_ingested_files()

    async def test_adelete_not_initialized_raises(self, test_config):
        service = RAGService(config=test_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            await service.adelete_files(filenames=["a.pdf"])

    async def test_alist_delegates_to_ingestion(self, test_config):
        service = RAGService(config=test_config)
        service._initialized = True
        service.ingestion = MagicMock()
        service.ingestion.alist_ingested_files = AsyncMock(return_value=[{"doc_id": "d1"}])
        result = await service.alist_ingested_files()
        assert result == [{"doc_id": "d1"}]
        service.ingestion.alist_ingested_files.assert_awaited_once()

    async def test_adelete_delegates_to_ingestion(self, test_config):
        service = RAGService(config=test_config)
        service._initialized = True
        service.ingestion = MagicMock()
        service.ingestion.adelete_files = AsyncMock(return_value=[{"status": "deleted"}])
        result = await service.adelete_files(filenames=["a.pdf"])
        assert result == [{"status": "deleted"}]
        call_kwargs = service.ingestion.adelete_files.call_args.kwargs
        assert call_kwargs["filenames"] == ["a.pdf"]
