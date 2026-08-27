# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unit tests for transport-neutral cited-image projection."""

from typing import Any

from dlightrag.engine.answer.media import evidence_images_from_sources


def _source(chunk_attrs: dict[str, Any]) -> Any:
    chunk = type("C", (), chunk_attrs)()
    return type(
        "S",
        (),
        {"id": "s1", "title": "Doc", "workspace": "default", "chunks": [chunk]},
    )()


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

    images = evidence_images_from_sources(sources, contexts=contexts)

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

    images = evidence_images_from_sources(sources, contexts=contexts)

    assert images[0]["answer_image_sent"] is True


def test_gallery_label_includes_cited_page() -> None:
    sources = [
        _source(
            {
                "chunk_id": "c1",
                "chunk_idx": 1,
                "page_number": 7,
                "image_url": "/img/c1",
                "thumbnail_url": "/img/c1/thumb",
            }
        )
    ]

    images = evidence_images_from_sources(sources, contexts={"chunks": []})

    assert images[0]["label"] == "Doc · Page 7"


def test_gallery_label_falls_back_to_title_without_page() -> None:
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

    images = evidence_images_from_sources(sources, contexts={"chunks": []})

    assert images[0]["label"] == "Doc"


def test_same_chunk_id_in_two_workspaces_keeps_both_images() -> None:
    legal_chunk = type(
        "C",
        (),
        {
            "chunk_id": "shared-hash",
            "chunk_idx": 1,
            "image_url": "/images/legal/shared-hash?size=full",
            "thumbnail_url": "/images/legal/shared-hash?size=thumb",
        },
    )()
    finance_chunk = type(
        "C",
        (),
        {
            "chunk_id": "shared-hash",
            "chunk_idx": 1,
            "image_url": "/images/finance/shared-hash?size=full",
            "thumbnail_url": "/images/finance/shared-hash?size=thumb",
        },
    )()
    sources = [
        type(
            "S",
            (),
            {"id": "1", "title": "Legal", "workspace": "legal", "chunks": [legal_chunk]},
        )(),
        type(
            "S",
            (),
            {
                "id": "2",
                "title": "Finance",
                "workspace": "finance",
                "chunks": [finance_chunk],
            },
        )(),
    ]
    contexts: Any = {
        "chunks": [
            {"chunk_id": "shared-hash", "_workspace": "legal", "_answer_image_sent": True},
            {
                "chunk_id": "shared-hash",
                "_workspace": "finance",
                "_answer_image_sent": False,
            },
        ]
    }

    images = evidence_images_from_sources(sources, contexts=contexts)

    assert [image["id"] for image in images] == [
        "legal:shared-hash",
        "finance:shared-hash",
    ]
    assert [image["chunk_id"] for image in images] == ["shared-hash", "shared-hash"]
    assert [image["answer_image_sent"] for image in images] == [True, False]


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

    images = evidence_images_from_sources(sources, contexts={"chunks": []})

    assert images == []
