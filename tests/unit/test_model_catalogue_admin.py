# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Application-owned runtime model catalogue publication semantics."""

import hashlib
import json
from collections.abc import Awaitable, Callable

import pytest

from dlightrag.application.model_catalogue import (
    ModelCatalogueAdmin,
    ModelCatalogueEntryNotFoundError,
    ModelCatalogueReadOnlyError,
    ModelCatalogueRevisionConflict,
    ModelCatalogueValidationError,
    StoredModelCatalogue,
)
from dlightrag.engine.ai.catalog import (
    CatalogueEntry,
    ModelCatalogue,
    catalogue_overlay_data,
    catalogue_overlay_revision,
    parse_catalogue_entry,
)
from dlightrag.engine.ai.settings import ModelSettings


def _entry(
    *,
    context: int = 100_000,
    off: str | None = "disabled",
) -> dict[str, object]:
    return {
        "provider": "openai",
        "model": "test-model",
        "base_url": "https://api.example.test/v1",
        "profile": {
            "context_window_tokens": context,
            "max_input_tokens": None,
            "max_output_tokens": 10_000,
            "supports_images": True,
            "reasoning": {
                "format": "openrouter",
                "levels": {
                    "off": off,
                    "minimal": None,
                    "low": "low",
                    "medium": None,
                    "high": "high",
                    "xhigh": None,
                    "max": None,
                },
            },
        },
    }


def _revision(entries: list[dict[str, object]]) -> str:
    encoded = json.dumps(
        entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _catalogue(*, context: int = 100_000) -> ModelCatalogue:
    builtin = parse_catalogue_entry(_entry(context=context))
    return ModelCatalogue(
        builtin_revision=_revision([builtin.as_dict()]),
        builtin_entries=(builtin,),
    )


class FakeStore:
    def __init__(self, catalogue: ModelCatalogue) -> None:
        self.revision = catalogue_overlay_revision(())
        self.overlay: object = []
        self.listener: Callable[[], Awaitable[None]] | None = None
        self.publishes: list[dict[str, object]] = []
        self.publish_result = True
        self.initialized: bool | None = None
        self.closed = False

    async def initialize(self, *, validate_only: bool) -> None:
        self.initialized = validate_only

    async def load(self) -> StoredModelCatalogue:
        return StoredModelCatalogue(revision=self.revision, overlay=self.overlay)

    async def publish(
        self,
        *,
        expected_revision: str,
        revision: str,
        overlay: object,
        actor: str,
    ) -> bool:
        self.publishes.append(
            {
                "expected_revision": expected_revision,
                "revision": revision,
                "overlay": overlay,
                "actor": actor,
            }
        )
        if not self.publish_result or expected_revision != self.revision:
            return False
        self.revision = revision
        self.overlay = overlay
        return True

    async def start_listener(self, on_change: Callable[[], Awaitable[None]]) -> None:
        self.listener = on_change

    async def aclose(self) -> None:
        self.closed = True


async def _admin(
    *,
    configured: tuple[ModelSettings, ...] = (),
    read_only: bool = False,
) -> tuple[ModelCatalogueAdmin, FakeStore, ModelCatalogue, list[str]]:
    catalogue = _catalogue()
    store = FakeStore(catalogue)
    invalidations: list[str] = []
    admin = ModelCatalogueAdmin(
        store=store,
        configured_models=lambda: configured,
        catalogue=catalogue,
        on_publish=lambda snapshot: invalidations.append(snapshot.revision),
        read_only=read_only,
    )
    await admin.start()
    return admin, store, catalogue, invalidations


@pytest.mark.asyncio
async def test_upsert_publishes_one_complete_overlay_and_projects_effective_source() -> None:
    admin, store, _catalogue_instance, invalidations = await _admin()
    initial = admin.read()

    view = await admin.upsert(
        _entry(context=200_000), expected_revision=initial.revision, actor="a"
    )

    assert store.publishes[0]["expected_revision"] == catalogue_overlay_revision(())
    assert store.publishes[0]["actor"] == "a"
    assert view.revision != initial.revision
    assert view.models[0].source == "overlay"
    assert view.models[0].profile["context_window_tokens"] == 200_000
    assert invalidations == [view.revision]


@pytest.mark.asyncio
async def test_identical_builtin_override_still_advances_concurrency_revision() -> None:
    admin, _store, _catalogue_instance, _invalidations = await _admin()
    initial = admin.revision

    view = await admin.upsert(_entry(), expected_revision=initial, actor="a")

    assert view.revision != initial
    assert view.models[0].source == "overlay"


@pytest.mark.asyncio
async def test_read_only_deployment_rejects_publication() -> None:
    admin, store, _catalogue_instance, _invalidations = await _admin(read_only=True)

    with pytest.raises(ModelCatalogueReadOnlyError, match="read-only"):
        await admin.upsert(_entry(), expected_revision=admin.revision, actor="a")

    assert store.publishes == []


@pytest.mark.asyncio
async def test_stale_revision_is_rejected_before_store_mutation() -> None:
    admin, store, _catalogue_instance, _invalidations = await _admin()

    with pytest.raises(ModelCatalogueRevisionConflict) as exc_info:
        await admin.upsert(
            _entry(context=200_000), expected_revision="sha256:" + "0" * 64, actor="a"
        )

    assert exc_info.value.current_revision == admin.revision
    assert store.publishes == []


@pytest.mark.asyncio
async def test_remove_restores_builtin_and_requires_an_existing_overlay() -> None:
    admin, _store, _catalogue_instance, _invalidations = await _admin()
    changed = await admin.upsert(
        _entry(context=200_000),
        expected_revision=admin.revision,
        actor="a",
    )

    restored = await admin.remove(
        provider=" OpenAI ",
        model=" test-model ",
        base_url="https://api.example.test/v1",
        expected_revision=changed.revision,
        actor="a",
    )

    assert restored.models[0].source == "builtin"
    assert restored.models[0].profile["context_window_tokens"] == 100_000
    with pytest.raises(ModelCatalogueEntryNotFoundError):
        await admin.remove(
            provider="openai",
            model="test-model",
            base_url="https://api.example.test/v1",
            expected_revision=restored.revision,
            actor="a",
        )


@pytest.mark.asyncio
async def test_update_that_breaks_a_configured_reasoning_role_is_not_published() -> None:
    settings = ModelSettings(
        provider="openai",
        model="test-model",
        base_url="https://api.example.test/v1",
        reasoning="off",
    )
    admin, store, _catalogue_instance, _invalidations = await _admin(configured=(settings,))

    with pytest.raises(ModelCatalogueValidationError, match="would break configured model"):
        await admin.upsert(_entry(off=None), expected_revision=admin.revision, actor="a")

    assert store.publishes == []


@pytest.mark.asyncio
async def test_failed_store_cas_reloads_authoritative_revision_before_conflict() -> None:
    admin, store, catalogue, _invalidations = await _admin()
    concurrent: CatalogueEntry = parse_catalogue_entry(_entry(context=150_000))
    candidate = catalogue.preview((concurrent,))
    store.overlay = catalogue_overlay_data((concurrent,))
    store.revision = catalogue_overlay_revision((concurrent,))
    store.publish_result = False

    with pytest.raises(ModelCatalogueRevisionConflict) as exc_info:
        await admin.upsert(_entry(context=200_000), expected_revision=admin.revision, actor="a")

    assert exc_info.value.current_revision == candidate.revision
    assert admin.read().models[0].profile["context_window_tokens"] == 150_000


@pytest.mark.asyncio
async def test_listener_reload_applies_committed_snapshot_and_invalidates_caches() -> None:
    admin, store, catalogue, invalidations = await _admin()
    committed = parse_catalogue_entry(_entry(context=175_000))
    candidate = catalogue.preview((committed,))
    store.overlay = catalogue_overlay_data((committed,))
    store.revision = catalogue_overlay_revision((committed,))

    assert store.listener is not None
    await store.listener()

    assert admin.revision == candidate.revision
    assert admin.read().models[0].profile["context_window_tokens"] == 175_000
    assert invalidations == [candidate.revision]


@pytest.mark.asyncio
async def test_empty_overlay_revision_survives_a_builtin_catalogue_upgrade() -> None:
    old_catalogue = _catalogue(context=100_000)
    upgraded_catalogue = _catalogue(context=200_000)
    store = FakeStore(old_catalogue)
    admin = ModelCatalogueAdmin(
        store=store,
        configured_models=tuple,
        catalogue=upgraded_catalogue,
    )

    await admin.start()

    assert admin.read().models[0].profile["context_window_tokens"] == 200_000
    assert admin.revision == upgraded_catalogue.revision


@pytest.mark.asyncio
async def test_start_rejects_stored_revision_that_does_not_match_content() -> None:
    catalogue = _catalogue()
    store = FakeStore(catalogue)
    store.revision = "sha256:" + "0" * 64
    admin = ModelCatalogueAdmin(
        store=store,
        configured_models=tuple,
        catalogue=catalogue,
    )

    with pytest.raises(ModelCatalogueValidationError, match="does not match"):
        await admin.start(validate_only=True)

    assert store.initialized is True
    assert admin.is_ready is False
