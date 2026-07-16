# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Concrete HTTP rerank providers and the strategy registry.

Each class owns only its wire deviations; the shared driver in
:mod:`dlightrag.models.rerank` handles transport, image bounding, scoring, and
top-k. ``chat_llm_reranker`` is deliberately absent -- it is an LLM listwise
prompt, not an HTTP rerank protocol, so it lives in the driver module.
"""

from typing import Any
from urllib.parse import urlparse

from dlightrag.models.providers.rerank_base import PreparedDocument, RerankProvider


def _text_documents(documents: list[PreparedDocument]) -> list[str]:
    """Text-only document list (drops any bounded image)."""
    return [text for text, _ in documents]


class JinaRerankProvider(RerankProvider):
    """Jina rerank endpoint; v3 is text-only, m0 accepts image documents."""

    default_base_url = "https://api.jina.ai/v1/rerank"
    default_model = "jina-reranker-v3"
    accepts_images = True

    def build_payload(
        self, *, model: str, query: str, documents: list[PreparedDocument], top_n: int
    ) -> dict[str, Any]:
        docs: list[Any] = [{"image": uri} if uri else text for text, uri in documents]
        return {
            "model": model,
            "query": query,
            "documents": docs,
            "top_n": top_n,
            "return_documents": False,
        }


class VoyageRerankProvider(RerankProvider):
    """Voyage reranker: text documents, ``top_k`` with truncation."""

    default_base_url = "https://api.voyageai.com/v1/rerank"
    default_model = "rerank-2.5"

    def build_payload(
        self, *, model: str, query: str, documents: list[PreparedDocument], top_n: int
    ) -> dict[str, Any]:
        return {
            "model": model,
            "query": query,
            "documents": _text_documents(documents),
            "top_k": top_n,
            "truncation": True,
        }

    def parse_results(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        return data["data"]


class CohereRerankProvider(RerankProvider):
    """Cohere public reranker: text documents, ``top_n``."""

    default_base_url = "https://api.cohere.com/v2/rerank"
    default_model = "rerank-v4.0-pro"

    def build_payload(
        self, *, model: str, query: str, documents: list[PreparedDocument], top_n: int
    ) -> dict[str, Any]:
        return {
            "model": model,
            "query": query,
            "documents": _text_documents(documents),
            "top_n": top_n,
        }


class AliyunRerankProvider(RerankProvider):
    """Alibaba Model Studio reranker.

    ``qwen3-rerank`` uses the flat OpenAI-compatible body; the multimodal
    ``qwen3-vl-rerank`` uses DashScope's native ``input``/``parameters`` body.
    """

    default_model = "qwen3-rerank"
    accepts_images = True
    requires_base_url = True

    def build_payload(
        self, *, model: str, query: str, documents: list[PreparedDocument], top_n: int
    ) -> dict[str, Any]:
        if model == "qwen3-rerank":
            return {
                "model": model,
                "query": query,
                "documents": _text_documents(documents),
                "top_n": top_n,
            }
        docs: list[Any] = [{"image": uri} if uri else text for text, uri in documents]
        return {
            "model": model,
            "input": {"query": {"text": query}, "documents": docs},
            "parameters": {"top_n": top_n},
        }

    def parse_results(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        if "output" in data:
            return data["output"].get("results", [])
        return data.get("results", [])


class HttpRerankProvider(RerankProvider):
    """Generic ``/rerank`` endpoint (self-hosted or hosted) in the common shape.

    The universal entry point: any standard rerank server that speaks
    ``{model, query, documents:[{text|image}], top_n} -> {results:[...]}`` works
    with only ``base_url`` (and optional ``model``) -- no code change.
    """

    default_model = "default"
    accepts_images = True
    requires_base_url = True
    requires_api_key = False

    def request_url(self, base_url: str | None, model: str) -> str:
        if base_url is None:
            raise ValueError("local_reranker requires base_url")
        return base_url.rstrip("/") + "/rerank"

    def build_payload(
        self, *, model: str, query: str, documents: list[PreparedDocument], top_n: int
    ) -> dict[str, Any]:
        docs: list[Any] = [{"image": uri} if uri else {"text": text} for text, uri in documents]
        return {"model": model, "query": query, "documents": docs, "top_n": top_n}


def _is_azure_ai_services_host(endpoint: str) -> bool:
    host = (urlparse(endpoint).hostname or "").rstrip(".").lower()
    return host == "services.ai.azure.com" or host.endswith(".services.ai.azure.com")


def _azure_cohere_rerank_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/rerank"):
        return base
    if base.endswith("/providers/cohere"):
        return f"{base}/v2/rerank"
    if _is_azure_ai_services_host(base):
        return f"{base}/providers/cohere/v2/rerank"
    return f"{base}/v1/rerank"


class AzureCohereRerankProvider(RerankProvider):
    """Azure AI Services Cohere reranker (text-only, raw ``Authorization``)."""

    default_model = "cohere-rerank-v4.0-pro"
    requires_base_url = True

    def request_url(self, base_url: str | None, model: str) -> str:
        if base_url is None:
            raise ValueError("azure_cohere requires base_url")
        return _azure_cohere_rerank_url(base_url)

    def request_headers(self, api_key: str | None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = api_key
        return headers

    def build_payload(
        self, *, model: str, query: str, documents: list[PreparedDocument], top_n: int
    ) -> dict[str, Any]:
        return {
            "model": model,
            "query": query,
            "documents": _text_documents(documents),
            "top_n": top_n,
        }


#: Strategy name -> stateless provider singleton. ``chat_llm_reranker`` is not an
#: HTTP protocol and is handled separately by ``build_rerank_func``.
RERANK_PROVIDERS: dict[str, RerankProvider] = {
    "jina_reranker": JinaRerankProvider(),
    "voyage_reranker": VoyageRerankProvider(),
    "cohere_reranker": CohereRerankProvider(),
    "aliyun_reranker": AliyunRerankProvider(),
    "azure_cohere": AzureCohereRerankProvider(),
    "local_reranker": HttpRerankProvider(),
}
