"""Tests for enhance_query_with_images()."""

from __future__ import annotations

import pytest

from dlightrag.unifiedrepresent.multimodal_query import enhance_query_with_images


@pytest.mark.asyncio
async def test_single_image():
    """Single image query produces enhanced text with description."""
    calls = []

    async def mock_vlm(prompt, *, image_data=None):
        calls.append({"prompt": prompt, "image_data": image_data})
        return "A bar chart showing Q3 revenue"

    result = await enhance_query_with_images(
        query="Analyze this chart",
        images=[b"fake_png_bytes"],
        vision_model_func=mock_vlm,
    )
    assert "Analyze this chart" in result
    assert "A bar chart showing Q3 revenue" in result
    assert len(calls) == 1
    assert calls[0]["image_data"] == b"fake_png_bytes"


@pytest.mark.asyncio
async def test_multiple_images():
    """Multiple images each get VLM descriptions, all combined."""
    call_count = 0

    async def mock_vlm(prompt, *, image_data=None):
        nonlocal call_count
        call_count += 1
        return f"Description for image {call_count}"

    result = await enhance_query_with_images(
        query="Compare these",
        images=[b"img1", b"img2", b"img3"],
        vision_model_func=mock_vlm,
    )
    assert call_count == 3
    assert "Image 1 content:" in result
    assert "Image 2 content:" in result
    assert "Image 3 content:" in result


@pytest.mark.asyncio
async def test_with_conversation_context():
    """Conversation context is included in enhanced query."""

    async def mock_vlm(prompt, *, image_data=None):
        return "Chart description"

    result = await enhance_query_with_images(
        query="Follow up question",
        images=[b"img"],
        vision_model_func=mock_vlm,
        conversation_context="Previous discussion about Q3 earnings",
    )
    assert "Follow up question" in result
    assert "Previous discussion about Q3 earnings" in result
    assert "Chart description" in result


@pytest.mark.asyncio
async def test_empty_images_returns_query():
    """Empty images list returns original query unchanged."""

    async def mock_vlm(prompt, *, image_data=None):
        raise AssertionError("Should not be called")

    result = await enhance_query_with_images(
        query="Just text query",
        images=[],
        vision_model_func=mock_vlm,
    )
    assert result == "Just text query"
