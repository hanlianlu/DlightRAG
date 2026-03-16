"""Tests for embedding callables."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestOpenAIEmbedding:
    @pytest.mark.asyncio
    async def test_embed_texts(self):
        from dlightrag.models.embedding import _openai_embedding

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]
        mock_client.embeddings.create.return_value = mock_response

        result = await _openai_embedding(
            ["hello", "world"],
            model="text-embedding-3-large",
            api_key="sk-test",
            _client=mock_client,
        )
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


class TestLiteLLMEmbedding:
    @pytest.mark.asyncio
    async def test_embed_texts(self):
        from dlightrag.models.embedding import _litellm_embedding

        mock_response = MagicMock()
        mock_response.data = [
            {"embedding": [0.1, 0.2]},
            {"embedding": [0.3, 0.4]},
        ]

        with patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response):
            result = await _litellm_embedding(
                ["hello", "world"],
                model="text-embedding-3-large",
                api_key="sk-test",
            )
        assert result == [[0.1, 0.2], [0.3, 0.4]]
