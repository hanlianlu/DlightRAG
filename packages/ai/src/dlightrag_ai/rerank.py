# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""HTTP rerank provider execution over immutable AI settings."""

from typing import Any

import httpx

from dlightrag_ai.providers.rerank_base import PreparedDocument, RerankProvider
from dlightrag_ai.providers.rerank_providers import RERANK_PROVIDERS
from dlightrag_ai.scheduler import ModelScheduler
from dlightrag_ai.settings import RerankSettings
from dlightrag_ai.telemetry import NOOP_TELEMETRY, Telemetry, telemetry_error_message


def _provider_for(settings: RerankSettings) -> RerankProvider:
    provider = RERANK_PROVIDERS.get(settings.strategy)
    if provider is None:
        raise ValueError(f"Unknown HTTP rerank strategy: {settings.strategy}")
    return provider


def rerank_accepts_images(settings: RerankSettings) -> bool:
    """Return the configured HTTP rerank provider's image capability."""
    return _provider_for(settings).accepts_images


class RerankModel:
    """Own one HTTP rerank provider client and its request lifecycle."""

    def __init__(
        self,
        settings: RerankSettings,
        provider: RerankProvider,
        *,
        scheduler: ModelScheduler,
        telemetry: Telemetry = NOOP_TELEMETRY,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.model = settings.model or provider.default_model
        self._scheduler = scheduler
        self._telemetry = telemetry
        self._client = httpx.AsyncClient(timeout=60.0)

    async def score(
        self,
        query: str,
        documents: list[PreparedDocument],
        *,
        top_n: int,
    ) -> list[dict[str, Any]]:
        """Return provider-indexed relevance scores for prepared documents."""
        return await self._scheduler.run(lambda: self._score(query, documents, top_n=top_n))

    async def _score(
        self,
        query: str,
        documents: list[PreparedDocument],
        *,
        top_n: int,
    ) -> list[dict[str, Any]]:
        payload = self.provider.build_payload(
            model=self.model,
            query=query,
            documents=documents,
            top_n=top_n,
        )
        async with self._telemetry.observe(
            f"rerank_request/{self.settings.strategy}",
            as_type="span",
            metadata={"document_count": len(documents), "top_n": top_n},
            model=self.model,
        ) as observation:
            try:
                response = await self._client.post(
                    self.provider.request_url(self.settings.base_url, self.model),
                    json=payload,
                    headers=self.provider.request_headers(self.settings.api_key),
                )
                response.raise_for_status()
                scores = self.provider.parse_results(response.json())
            except Exception as exc:
                observation.update(
                    level="ERROR",
                    status_message=telemetry_error_message(self._telemetry, exc),
                )
                raise
            observation.update(output={"score_count": len(scores)})
            return scores

    async def aclose(self) -> None:
        """Release the HTTP connection pool."""
        await self._client.aclose()


def create_rerank_model(
    settings: RerankSettings,
    *,
    scheduler: ModelScheduler,
    telemetry: Telemetry = NOOP_TELEMETRY,
) -> RerankModel:
    """Validate and build one configured HTTP rerank model."""
    provider = _provider_for(settings)
    if provider.requires_api_key and not settings.api_key:
        raise ValueError(f"{settings.strategy} requires api_key")
    if provider.requires_base_url and not settings.base_url:
        raise ValueError(f"{settings.strategy} requires base_url")
    return RerankModel(
        settings,
        provider,
        scheduler=scheduler,
        telemetry=telemetry,
    )


__all__ = ["RerankModel", "create_rerank_model", "rerank_accepts_images"]
