# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Resolve one Answer request's resources and current-image policy.

Acceptance pins current image links and execution rebuilds the admitted
resources through the same resolver, so both sides see one admission policy.
"""

import asyncio
import hashlib
import hmac
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from dlightrag.engine.answer.capabilities import (
    AnswerCapabilityCoordinator,
    RequestModelContext,
)
from dlightrag.engine.answer.errors import (
    AnswerResourceAdmissionError,
    CurrentImagePayloadError,
)
from dlightrag.engine.answer.execution.input import (
    AnswerRunRequest,
    AttachmentReference,
    LinkReference,
)
from dlightrag.engine.answer.image_capability import (
    AnswerImageCapability,
    check_answer_image_capability,
    check_answer_image_count,
)
from dlightrag.engine.answer.images import AnswerImageBudget
from dlightrag.engine.answer.mode import ResolvedMode, resource_role
from dlightrag.engine.answer.model_runtime import AnswerModelRuntime
from dlightrag.engine.answer.resources import ResourceInput, ResourceRegistry
from dlightrag.engine.answer.resources.models import (
    ResourceManifestEntry,
    ResourceRegistryError,
)
from dlightrag.engine.answer.resources.registry import FetchedBytesSink
from dlightrag.engine.answer.web_sources import WebSourceService
from dlightrag.engine.public_http import fetch_public_http
from dlightrag.engine.rag.corpus.sources.source_contract import safe_source_filename
from dlightrag.engine.runtime.records import artifact_digest

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AnswerResourceSettings:
    max_attachments: int
    max_attachment_bytes: int
    max_total_attachment_bytes: int
    image_max_bytes: int
    image_max_pixels: int


@dataclass(frozen=True, slots=True)
class ResolvedAnswerResources:
    models: RequestModelContext
    web_sources: WebSourceService | None
    registry: ResourceRegistry | None
    resource_manifest: tuple[ResourceManifestEntry, ...]
    current_images: list[dict[str, Any]]
    current_image_count: int
    image_budget: AnswerImageBudget | None
    query_images: list[dict[str, Any]] | None


class AnswerResourceResolver:
    """Resolve request resources and visual policy exactly once."""

    def __init__(
        self,
        *,
        settings: AnswerResourceSettings,
        models: AnswerModelRuntime,
        capabilities: AnswerCapabilityCoordinator,
    ) -> None:
        self._settings = settings
        self._models = models
        self._capabilities = capabilities

    async def pin_current_image_links(
        self,
        request: AnswerRunRequest,
        attachment_bytes: Sequence[bytes],
    ) -> tuple[AnswerRunRequest, list[bytes]]:
        """Materialize declared image links once for acceptance and durable replay."""
        if len(request.attachments) != len(attachment_bytes):
            raise ValueError("current attachment references and bytes must have equal length")
        image_count = sum(
            resource_role(filename=attachment.filename, mime_type=attachment.mime_type) == "image"
            for attachment in request.attachments
        ) + sum(
            resource_role(filename=link.filename or link.url, mime_type=link.mime_type) == "image"
            for link in request.links
        )
        if image_count:
            capabilities = await self._capabilities.refresh_answer()
            check_answer_image_count(
                image_count=image_count,
                configured_ceiling=(
                    capabilities.answer.configured_ceiling if capabilities.answer is not None else 0
                ),
            )
        links: list[LinkReference] = []
        pinned_link_attachments: list[AttachmentReference] = []
        pinned_link_bytes: list[bytes] = []
        for link in request.links:
            if (
                resource_role(
                    filename=link.filename or link.url,
                    mime_type=link.mime_type,
                )
                != "image"
            ):
                links.append(link)
                continue
            data = await self.materialize_link_image(link.url)
            if data is None:
                raise CurrentImagePayloadError(
                    f"current image {link.filename or link.url} could not be fetched and verified"
                )
            try:
                mime_type, _data_uri = await asyncio.to_thread(
                    _verified_current_image_data_uri,
                    data,
                    max_pixels=self._settings.image_max_pixels,
                )
            except ValueError as exc:
                raise CurrentImagePayloadError(
                    f"current image {link.filename or link.url} {exc}"
                ) from exc
            pinned_link_attachments.append(
                AttachmentReference(
                    digest=artifact_digest(data),
                    filename=safe_source_filename(link.filename),
                    mime_type=mime_type,
                    ordinal=len(pinned_link_attachments),
                    byte_size=len(data),
                )
            )
            pinned_link_bytes.append(data)
        offset = len(pinned_link_attachments)
        attachments = [
            *pinned_link_attachments,
            *(
                replace(attachment, ordinal=offset + index)
                for index, attachment in enumerate(request.attachments)
            ),
        ]
        return (
            replace(
                request,
                links=tuple(links),
                attachments=tuple(attachments),
            ),
            [*pinned_link_bytes, *attachment_bytes],
        )

    async def resolve(
        self,
        resources: list[ResourceInput] | None,
        *,
        models: RequestModelContext,
        confirm_image_context: Callable[
            [RequestModelContext],
            Awaitable[tuple[RequestModelContext, AnswerImageCapability | None]],
        ],
        fetched_bytes_sink: FetchedBytesSink | None = None,
        resolved_mode: ResolvedMode,
        resource_identity: str | None = None,
    ) -> ResolvedAnswerResources:
        """Resolve resource capabilities and image transport.

        ``resource_identity`` is the Run's recorded salt; every handle and cursor
        the registry mints comes from it.
        """
        declared_image_count = sum(
            1
            for resource in resources or ()
            if resource.loader is None
            and resource_role(
                filename=resource.filename or resource.url,
                mime_type=resource.declared_mime,
            )
            == "image"
        )
        image_capability: AnswerImageCapability | None = None
        if declared_image_count:
            models, image_capability = await confirm_image_context(models)
            self._check_current_image_admission(
                image_count=declared_image_count,
                capability=image_capability,
            )
        (
            current_images,
            remaining_resources,
            current_image_resources,
        ) = await self.prepare_current_images(resources)
        if current_images and not declared_image_count:
            models, image_capability = await confirm_image_context(models)
        self._check_current_image_admission(
            image_count=len(current_images),
            capability=image_capability,
        )

        web_sources = self._models.web_sources()
        registry = self.build_resource_context(
            remaining_resources,
            web_sources=web_sources,
            fetched_bytes_sink=fetched_bytes_sink,
            resource_identity=resource_identity,
        )
        try:
            current_image_resource_ids = (
                tuple(registry.register(resource) for resource in current_image_resources)
                if registry is not None
                else ()
            )
            resource_manifest = registry.manifest() if registry is not None else ()
            image_budget = self._capabilities.answer_image_policy(models.query).new_budget()
            query_images = (
                await self.budget_agent_images(
                    current_images,
                    image_budget,
                    current_image_resource_ids if resolved_mode == "research" else (),
                )
                or None
            )

            return ResolvedAnswerResources(
                models=models,
                web_sources=web_sources,
                registry=registry,
                resource_manifest=resource_manifest,
                current_images=current_images,
                current_image_count=len(current_images),
                image_budget=image_budget,
                query_images=query_images,
            )
        except BaseException:
            if registry is not None:
                await registry.aclose()
            raise

    async def prepare_current_images(
        self,
        resources: list[ResourceInput] | None,
    ) -> tuple[list[dict[str, Any]], list[ResourceInput], list[ResourceInput]]:
        """Build verified current-image blocks while retaining attachments as resources."""
        if not resources:
            return [], [], []
        images: list[dict[str, Any]] = []
        remaining: list[ResourceInput] = []
        image_resources: list[ResourceInput] = []
        for resource in resources:
            data: bytes | None = None
            if resource.loader is not None:
                remaining.append(resource)
                continue
            if resource.content is not None:
                data = resource.content
            elif (
                resource.url is not None
                and resource_role(
                    filename=resource.filename or resource.url,
                    mime_type=resource.declared_mime,
                )
                == "image"
            ):
                data = await self.materialize_link_image(resource.url)
            if data is None:
                if (
                    resource.url is not None
                    and resource_role(
                        filename=resource.filename or resource.url,
                        mime_type=resource.declared_mime,
                    )
                    == "image"
                ):
                    raise CurrentImagePayloadError(
                        f"current image {resource.filename or resource.url} "
                        "could not be fetched and verified"
                    )
                remaining.append(resource)
                continue
            try:
                mime, data_uri = await asyncio.to_thread(
                    _verified_current_image_data_uri,
                    data,
                    max_pixels=self._settings.image_max_pixels,
                )
            except ValueError as exc:
                if (
                    resource_role(
                        filename=resource.filename or resource.url,
                        mime_type=resource.declared_mime,
                    )
                    == "image"
                ):
                    raise CurrentImagePayloadError(
                        f"current image {resource.filename or len(images) + 1} {exc}"
                    ) from exc
                remaining.append(resource)
                continue
            images.append({"type": "image_url", "image_url": {"url": data_uri}})
            image_resource = ResourceInput(
                filename=resource.filename,
                content=data,
                declared_mime=mime,
            )
            remaining.append(image_resource)
            image_resources.append(image_resource)
        return images, remaining, image_resources

    @staticmethod
    def _check_current_image_admission(
        *,
        image_count: int,
        capability: AnswerImageCapability | None,
    ) -> None:
        if image_count <= 0:
            return
        if capability is None:
            check_answer_image_capability(image_count=image_count, capability=None)
            return
        check_answer_image_count(
            image_count=image_count,
            configured_ceiling=capability.configured_ceiling,
        )
        check_answer_image_capability(image_count=image_count, capability=capability)

    async def materialize_link_image(self, url: str) -> bytes | None:
        """Fetch one current-image link under SSRF revalidation."""
        try:
            result = await fetch_public_http(
                url,
                max_bytes=self._settings.image_max_bytes,
                timeout=120.0,
            )
            return result.content
        except Exception:
            logger.warning("Failed to materialize current image link", exc_info=True)
            return None

    def build_resource_context(
        self,
        resources: list[ResourceInput] | None,
        *,
        web_sources: WebSourceService | None = None,
        fetched_bytes_sink: FetchedBytesSink | None = None,
        resource_identity: str | None = None,
    ) -> ResourceRegistry:
        """Register the admitted resources for read and view.

        The registry always exists in Research-capable composition so ``read(url=...)``
        does not depend on an Execution Environment or a configured provider. Its
        handles and cursors are minted from the Run's ``resource_identity`` and from
        nothing the deployment holds, so a resume mints the handles the Run already
        printed; without one they are random to this registry.
        """
        registry = ResourceRegistry(
            max_attachments=self._settings.max_attachments,
            max_attachment_bytes=self._settings.max_attachment_bytes,
            max_total_attachment_bytes=self._settings.max_total_attachment_bytes,
            url_text_fallback=(web_sources.extract if web_sources is not None else None),
            fetched_bytes_sink=fetched_bytes_sink,
            resource_secret=_run_secret(resource_identity, b"answer-resource-identity"),
            cursor_secret=_run_secret(resource_identity, b"answer-resource-cursor"),
        )
        try:
            for resource in resources or []:
                registry.register(resource)
        except (ValueError, ResourceRegistryError) as exc:
            raise AnswerResourceAdmissionError() from exc

        return registry

    @staticmethod
    async def budget_agent_images(
        current_images: list[dict[str, Any]],
        budget: AnswerImageBudget,
        resource_ids: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        def build() -> list[dict[str, Any]]:
            blocks: list[dict[str, Any]] = []
            for index, image in enumerate(current_images, start=1):
                block = budget.add_user_image(image, label=f"query_image_{index}")
                if block is None:
                    raise CurrentImagePayloadError(
                        f"current image query_image_{index} could not fit the answer image budget"
                    )
                if index <= len(resource_ids):
                    blocks.append(
                        {
                            "type": "text",
                            "text": (
                                f"[current image {index} | resource: {resource_ids[index - 1]}]"
                            ),
                        }
                    )
                blocks.append(block)
            return blocks

        return await asyncio.to_thread(build)


def _run_secret(resource_identity: str | None, purpose: bytes) -> bytes | None:
    """Derive one purpose's key from a Run's identity, keeping handles apart from cursors."""
    if resource_identity is None:
        return None
    return hmac.new(bytes.fromhex(resource_identity), purpose, hashlib.sha256).digest()


def _verified_current_image_data_uri(data: bytes, *, max_pixels: int) -> tuple[str, str]:
    from dlightrag.engine.ai.media import image_bytes_to_data_uri, verify_web_image_bytes

    mime = verify_web_image_bytes(data, max_pixels=max_pixels)
    return mime, image_bytes_to_data_uri(data, fallback_mime=mime)


__all__ = [
    "AnswerResourceResolver",
    "AnswerResourceSettings",
    "ResolvedAnswerResources",
]
