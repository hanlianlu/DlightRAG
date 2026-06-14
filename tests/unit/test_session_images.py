# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for session-scoped query image memory."""

from __future__ import annotations

from types import SimpleNamespace

from dlightrag.core.session_images import SessionImageStore


def test_session_image_store_assigns_ids_and_resolves_images() -> None:
    store = SessionImageStore(max_images_per_session=3, max_sessions=2, ttl_seconds=60)

    ids = store.store("s1", ["img-a", "img-b"])

    assert ids == ["img_0", "img_1"]
    assert store.get("s1", ["img_1", "missing", "img_0"]) == ["img-b", "img-a"]


def test_session_image_store_evicts_old_images_per_session() -> None:
    store = SessionImageStore(max_images_per_session=2, max_sessions=2, ttl_seconds=60)

    ids = store.store("s1", ["img-a", "img-b", "img-c"])

    assert ids == ["img_0", "img_1", "img_2"]
    assert store.get("s1", ["img_0", "img_1", "img_2"]) == ["img-b", "img-c"]


def test_session_image_store_evicts_old_sessions() -> None:
    store = SessionImageStore(max_images_per_session=2, max_sessions=1, ttl_seconds=60)

    store.store("s1", ["img-a"])
    store.store("s2", ["img-b"])

    assert store.get("s1", ["img_0"]) == []
    assert store.get("s2", ["img_0"]) == ["img-b"]


def test_session_image_store_clear_removes_session() -> None:
    store = SessionImageStore(max_images_per_session=2, max_sessions=2, ttl_seconds=60)
    store.store("s1", ["img-a"])

    store.clear("s1")

    assert store.get("s1", ["img_0"]) == []


def test_manager_session_image_ttl_tracks_checkpoint_ttl() -> None:
    from dlightrag.core.servicemanager import RAGServiceManager

    manager = RAGServiceManager.__new__(RAGServiceManager)
    manager._session_images = None
    manager._config = SimpleNamespace(
        checkpoint_session_ttl_days=30,
        query_images=SimpleNamespace(
            session_max_images=50,
            session_max_sessions=100,
        ),
    )

    store = manager._get_session_images()

    assert store._ttl == 30 * 24 * 60 * 60
