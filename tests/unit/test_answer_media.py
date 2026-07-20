# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unit tests for transport-neutral cited-image projection."""

from typing import Any

from dlightrag.core.answer.media import answer_images_from_sources


def _source(chunk_attrs: dict[str, Any]) -> Any:
    chunk = type("C", (), chunk_attrs)()
    return type("S", (), {"id": "s1", "title": "Doc", "chunks": [chunk]})()


def test_cited_chunk_renders_even_when_raw_image_not_sent() -> None:
    sources = [
        _source(
            {
                "chunk_id": "c1",
                "chunk_idx": 1,
                "image_url": "/img/c1",
                "thumbnail_url": "/img/c1/thumb",
            }
        )
    ]
    contexts: Any = {"chunks": [{"chunk_id": "c1", "_answer_image_sent": False}]}

    images = answer_images_from_sources(sources, contexts=contexts)

    assert [i["chunk_id"] for i in images] == ["c1"]
    assert images[0]["answer_image_sent"] is False


def test_sent_chunk_is_annotated_true() -> None:
    sources = [
        _source(
            {
                "chunk_id": "c1",
                "chunk_idx": 1,
                "image_url": "/img/c1",
                "thumbnail_url": "/img/c1/thumb",
            }
        )
    ]
    contexts: Any = {"chunks": [{"chunk_id": "c1", "_answer_image_sent": True}]}

    images = answer_images_from_sources(sources, contexts=contexts)

    assert images[0]["answer_image_sent"] is True


def test_cited_chunk_without_any_url_is_excluded() -> None:
    sources = [
        _source(
            {
                "chunk_id": "c2",
                "chunk_idx": 1,
                "image_url": None,
                "thumbnail_url": None,
            }
        )
    ]

    images = answer_images_from_sources(sources, contexts={"chunks": []})

    assert images == []
