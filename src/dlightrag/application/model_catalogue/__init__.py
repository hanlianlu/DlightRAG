# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Single behavior owner for runtime model catalogue reads and publication."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from dlightrag.engine.ai.catalog import (
    MODEL_CATALOGUE,
    CatalogueEntry,
    CatalogueSnapshot,
    ModelCatalogue,
    catalogue_overlay_data,
    catalogue_overlay_revision,
    fallback_model_profile,
    parse_catalogue_entry,
    parse_catalogue_overlay,
)
from dlightrag.engine.ai.fingerprints import model_endpoint_fingerprint, model_fingerprint
from dlightrag.engine.ai.reasoning import ReasoningConfigurationError, resolve_reasoning
from dlightrag.engine.ai.settings import ModelSettings


class ModelCatalogueSchemaError(RuntimeError):
    """The durable runtime catalogue schema is incompatible."""


class ModelCatalogueUnavailableError(RuntimeError):
    """The runtime catalogue has not completed PostgreSQL synchronization."""


class ModelCatalogueReadOnlyError(RuntimeError):
    """This deployment may consume but not publish catalogue changes."""


class ModelCatalogueRevisionConflict(RuntimeError):
    """A writer used a stale effective catalogue revision."""

    def __init__(self, current_revision: str) -> None:
        self.current_revision = current_revision
        super().__init__(f"model catalogue revision changed; current={current_revision}")


class ModelCatalogueValidationError(ValueError):
    """A proposed overlay or configured role is invalid."""


class ModelCatalogueEntryNotFoundError(KeyError):
    """A custom endpoint or built-in override does not exist."""


@dataclass(frozen=True, slots=True)
class StoredModelCatalogue:
    revision: str
    overlay: object


class ModelCatalogueStore(Protocol):
    async def initialize(self, *, validate_only: bool) -> None: ...

    async def load(self) -> StoredModelCatalogue: ...

    async def publish(
        self,
        *,
        expected_revision: str,
        revision: str,
        overlay: object,
        actor: str,
    ) -> bool: ...

    async def start_listener(self, on_change: Callable[[], Awaitable[None]]) -> None: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ModelCatalogueEntryView:
    provider: str
    model: str
    base_url: str | None
    profile: Mapping[str, object]
    source: str


@dataclass(frozen=True, slots=True)
class ModelCatalogueView:
    revision: str
    models: tuple[ModelCatalogueEntryView, ...]


class ModelCatalogueAdmin:
    """Validate, publish, reload, and project one effective model catalogue.

    REST, Web, and MCP adapters all cross this interface; no transport owns
    profile validation, optimistic concurrency, or overlay semantics.
    """

    def __init__(
        self,
        *,
        store: ModelCatalogueStore,
        configured_models: Callable[[], Sequence[ModelSettings]],
        catalogue: ModelCatalogue = MODEL_CATALOGUE,
        on_publish: Callable[[CatalogueSnapshot], None] | None = None,
        read_only: bool = False,
    ) -> None:
        self._store = store
        self._configured_models = configured_models
        self._catalogue = catalogue
        self._on_publish = on_publish
        self._read_only = read_only
        self._ready = False
        self._stored_overlay_revision: str | None = None
        self._lock: asyncio.Lock | None = None

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def revision(self) -> str:
        self._require_ready()
        return self._catalogue.revision

    async def start(self, *, validate_only: bool = False) -> None:
        await self._store.initialize(validate_only=validate_only)
        await self._reload(require_match=True)
        await self._store.start_listener(self.reload)
        self._ready = True

    async def reload(self) -> None:
        """Re-read authoritative PostgreSQL state after a NOTIFY wake."""
        async with self._get_lock():
            await self._reload(require_match=True)

    def read(self) -> ModelCatalogueView:
        self._require_ready()
        return self._view(self._catalogue.snapshot)

    async def upsert(
        self,
        value: Mapping[str, Any],
        *,
        expected_revision: str,
        actor: str,
    ) -> ModelCatalogueView:
        try:
            entry = parse_catalogue_entry(dict(value))
        except (RuntimeError, ValueError) as exc:
            raise ModelCatalogueValidationError(str(exc)) from None
        async with self._get_lock():
            self._require_writable()
            self._require_expected_revision(expected_revision)
            overlay = {item.fingerprint: item for item in self._catalogue.overlay}
            overlay[entry.fingerprint] = entry
            snapshot = await self._publish(
                tuple(overlay.values()),
                expected_revision=expected_revision,
                actor=actor,
            )
            return self._view(snapshot)

    async def remove(
        self,
        *,
        provider: str,
        model: str,
        base_url: str | None,
        expected_revision: str,
        actor: str,
    ) -> ModelCatalogueView:
        """Remove a runtime entry and reveal its startup or built-in baseline."""
        fingerprint = model_endpoint_fingerprint(
            provider.strip().lower(),
            model.strip(),
            base_url,
        )
        async with self._get_lock():
            self._require_writable()
            self._require_expected_revision(expected_revision)
            overlay = {item.fingerprint: item for item in self._catalogue.overlay}
            if fingerprint not in overlay:
                raise ModelCatalogueEntryNotFoundError(
                    f"runtime model catalogue entry does not exist: {fingerprint.model}"
                )
            del overlay[fingerprint]
            snapshot = await self._publish(
                tuple(overlay.values()),
                expected_revision=expected_revision,
                actor=actor,
            )
            return self._view(snapshot)

    async def _publish(
        self,
        overlay: tuple[CatalogueEntry, ...],
        *,
        expected_revision: str,
        actor: str,
    ) -> CatalogueSnapshot:
        if not actor.strip():
            raise ModelCatalogueValidationError("catalogue actor cannot be empty")
        candidate = self._catalogue.preview(overlay)
        self._validate_configured_models(candidate)
        stored_revision = self._stored_overlay_revision
        if stored_revision is None:
            raise ModelCatalogueUnavailableError(
                "runtime model catalogue has no synchronized overlay revision"
            )
        next_stored_revision = catalogue_overlay_revision(overlay)
        published = await self._store.publish(
            expected_revision=stored_revision,
            revision=next_stored_revision,
            overlay=catalogue_overlay_data(overlay),
            actor=actor,
        )
        if not published:
            await self._reload(require_match=True)
            raise ModelCatalogueRevisionConflict(self._catalogue.revision)
        self._stored_overlay_revision = next_stored_revision
        snapshot = self._catalogue.replace_overlay(overlay)
        self._notify_publish(snapshot)
        return snapshot

    async def _reload(self, *, require_match: bool) -> None:
        stored = await self._store.load()
        try:
            overlay = parse_catalogue_overlay(stored.overlay)
            candidate = self._catalogue.preview(overlay)
            self._validate_configured_models(candidate)
        except (RuntimeError, ValueError, ReasoningConfigurationError) as exc:
            raise ModelCatalogueValidationError(str(exc)) from None
        overlay_revision = catalogue_overlay_revision(overlay)
        if overlay_revision != stored.revision:
            if require_match:
                raise ModelCatalogueValidationError(
                    "stored model catalogue revision does not match its overlay content"
                )
            return
        self._stored_overlay_revision = stored.revision
        if candidate.revision != self._catalogue.revision or overlay != self._catalogue.overlay:
            snapshot = self._catalogue.replace_overlay(overlay)
            self._notify_publish(snapshot)

    def _validate_configured_models(self, snapshot: CatalogueSnapshot) -> None:
        for settings in self._configured_models():
            fingerprint = model_fingerprint(settings)
            profile = snapshot.resolve(fingerprint) or fallback_model_profile(fingerprint)
            try:
                resolve_reasoning(profile.reasoning, settings.reasoning)
                resolve_reasoning(
                    profile.reasoning,
                    settings.effective_agentic_reasoning,
                )
            except ReasoningConfigurationError as exc:
                raise ModelCatalogueValidationError(
                    f"catalogue would break configured model {settings.model!r}: {exc}"
                ) from None

    def _require_writable(self) -> None:
        if self._read_only:
            raise ModelCatalogueReadOnlyError(
                "runtime model catalogue is read-only on this deployment"
            )

    def _require_expected_revision(self, expected_revision: str) -> None:
        self._require_ready()
        if expected_revision != self._catalogue.revision:
            raise ModelCatalogueRevisionConflict(self._catalogue.revision)

    def _require_ready(self) -> None:
        if not self._ready:
            raise ModelCatalogueUnavailableError(
                "runtime model catalogue has not synchronized with PostgreSQL"
            )

    def _notify_publish(self, snapshot: CatalogueSnapshot) -> None:
        if self._on_publish is not None:
            self._on_publish(snapshot)

    def _view(self, snapshot: CatalogueSnapshot) -> ModelCatalogueView:
        startup = snapshot.startup_fingerprints
        overlay = snapshot.overlay_fingerprints
        return ModelCatalogueView(
            revision=snapshot.revision,
            models=tuple(
                ModelCatalogueEntryView(
                    provider=entry.provider,
                    model=entry.model,
                    base_url=entry.base_url,
                    profile=entry.as_dict()["profile"],  # type: ignore[arg-type]
                    source=(
                        "overlay"
                        if entry.fingerprint in overlay
                        else "config"
                        if entry.fingerprint in startup
                        else "builtin"
                    ),
                )
                for entry in snapshot.entries
            ),
        )

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def aclose(self) -> None:
        self._ready = False
        await self._store.aclose()


__all__ = [
    "ModelCatalogueAdmin",
    "ModelCatalogueEntryNotFoundError",
    "ModelCatalogueEntryView",
    "ModelCatalogueReadOnlyError",
    "ModelCatalogueRevisionConflict",
    "ModelCatalogueSchemaError",
    "ModelCatalogueStore",
    "ModelCatalogueUnavailableError",
    "ModelCatalogueValidationError",
    "ModelCatalogueView",
    "StoredModelCatalogue",
]
