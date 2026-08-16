# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Provider-owned chat completion lifecycle and telemetry."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import aclosing
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

from dlightrag_ai.fingerprints import model_fingerprint
from dlightrag_ai.providers import get_provider
from dlightrag_ai.providers.base import CompletionProvider
from dlightrag_ai.scheduler import ModelScheduler
from dlightrag_ai.settings import ModelSettings
from dlightrag_ai.structured import StructuredOutput
from dlightrag_ai.telemetry import (
    NOOP_TELEMETRY,
    Telemetry,
    telemetry_error_message,
    telemetry_messages,
)

logger = logging.getLogger(__name__)


def _is_default_openai_endpoint(base_url: str | None) -> bool:
    if not base_url:
        return True
    parsed = urlparse(base_url)
    return parsed.scheme in {"http", "https"} and parsed.netloc == "api.openai.com"


def structured_response_format(
    structured_output: StructuredOutput,
    settings: ModelSettings,
    *,
    provider: CompletionProvider | None = None,
) -> dict[str, Any]:
    """Resolve the configured structured-output transport."""
    mode = settings.structured_output
    if mode == "json_object":
        return {"type": "json_object"}
    if mode == "json_schema":
        return structured_output.response_format_for_provider(settings.provider)
    if settings.provider == "openai" and _is_default_openai_endpoint(settings.base_url):
        return structured_output.response_format_for_provider("openai")
    if provider is not None and provider.supports_native_json_schema:
        return structured_output.response_format_for_provider(settings.provider)
    return {"type": "json_object"}


class CompletionModel:
    """One closeable messages-first model bound to immutable settings."""

    def __init__(
        self,
        settings: ModelSettings,
        *,
        scheduler: ModelScheduler,
        telemetry: Telemetry = NOOP_TELEMETRY,
    ) -> None:
        self.settings = settings
        self.fingerprint = model_fingerprint(settings)
        self._scheduler = scheduler
        self._telemetry = telemetry
        self._provider = get_provider(
            settings.provider,
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout,
            max_retries=settings.max_retries,
        )

    async def __call__(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        """Complete one chat request or return a telemetry-owned token stream."""
        stream = bool(kwargs.pop("stream", False))
        if stream:
            usage_holder = kwargs.pop("usage_holder", None)
            return self._scheduler.stream(
                lambda: self._stream(messages, kwargs, usage_holder=usage_holder)
            )
        kwargs.pop("usage_holder", None)
        return await self._scheduler.run(lambda: self._complete(messages, kwargs))

    def _observation_kwargs(
        self,
        messages: list[dict[str, Any]],
        request: dict[str, Any],
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "provider": self.fingerprint.provider,
            "endpoint_fingerprint": self.fingerprint.endpoint_fingerprint,
        }
        metadata.update(
            {
                key: value
                for key, value in request.items()
                if key not in {"structured_output", "response_format"}
            }
        )
        model_parameters = {
            **(
                {"temperature": self.settings.temperature}
                if self.settings.temperature is not None
                else {}
            ),
            **{
                key: value
                for key, value in metadata.items()
                if isinstance(value, str | int | float | bool)
            },
        }
        return {
            "as_type": "generation",
            "input": (
                telemetry_messages(messages) if self._telemetry.capture_sensitive_data else None
            ),
            "metadata": metadata or None,
            "model": self.fingerprint.model,
            "model_parameters": model_parameters or None,
        }

    def _request_options(self, request: dict[str, Any]) -> tuple[dict[str, Any], Any, Any]:
        response_format = request.pop("response_format", None)
        max_tokens = request.pop("max_tokens", None)
        structured_output = request.pop("structured_output", None)
        if structured_output is not None:
            if not isinstance(structured_output, StructuredOutput):
                raise TypeError("structured_output must be a StructuredOutput")
            response_format = response_format or structured_response_format(
                structured_output,
                self.settings,
                provider=self._provider,
            )
        model_kwargs = {**self.settings.model_kwargs_copy(), **request}
        return model_kwargs, response_format, max_tokens

    async def _complete(
        self,
        messages: list[dict[str, Any]],
        request: dict[str, Any],
    ) -> Any:
        observation_kwargs = self._observation_kwargs(messages, request)
        structured_output = request.get("structured_output")
        model_kwargs, response_format, max_tokens = self._request_options(dict(request))
        async with self._telemetry.observe(
            f"llm_{self.settings.model}",
            **observation_kwargs,
        ) as observation:
            try:
                result = await self._provider.complete(
                    messages=messages,
                    model=self.settings.model,
                    temperature=self.settings.temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    model_kwargs=model_kwargs,
                )
            except Exception as exc:
                if (
                    structured_output is not None
                    and self.settings.provider == "openai"
                    and isinstance(response_format, dict)
                    and response_format.get("type") == "json_schema"
                ):
                    logger.warning(
                        "Strict structured output failed for %s; retrying with json_object",
                        self.settings.model,
                        exc_info=True,
                    )
                    try:
                        result = await self._provider.complete(
                            messages=messages,
                            model=self.settings.model,
                            temperature=self.settings.temperature,
                            max_tokens=max_tokens,
                            response_format={"type": "json_object"},
                            model_kwargs=model_kwargs,
                        )
                    except Exception as fallback_exc:
                        observation.update(
                            level="ERROR",
                            status_message=telemetry_error_message(
                                self._telemetry,
                                fallback_exc,
                            ),
                        )
                        raise
                else:
                    observation.update(
                        level="ERROR",
                        status_message=telemetry_error_message(self._telemetry, exc),
                    )
                    raise
            output: dict[str, Any] = {"text_length": len(result)}
            if self._telemetry.capture_sensitive_data:
                output["text"] = str(result)
            observation.update(
                output=output,
                usage_details=getattr(result, "usage_details", None),
                cost_details=getattr(result, "cost_details", None),
            )
            return result

    async def _stream(
        self,
        messages: list[dict[str, Any]],
        request: dict[str, Any],
        *,
        usage_holder: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str]:
        observation_kwargs = self._observation_kwargs(messages, request)
        model_kwargs, response_format, max_tokens = self._request_options(dict(request))
        active_usage_holder = usage_holder if usage_holder is not None else {}
        chunks: list[str] = []
        text_length = 0
        first_chunk = True
        async with self._telemetry.observe(
            f"llm_{self.settings.model}",
            **observation_kwargs,
        ) as observation:
            try:
                stream = self._provider.stream(
                    messages=messages,
                    model=self.settings.model,
                    temperature=self.settings.temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    model_kwargs=model_kwargs,
                    usage_holder=active_usage_holder,
                )
                async with aclosing(stream):
                    async for chunk in stream:
                        if first_chunk:
                            observation.update(completion_start_time=datetime.now(UTC))
                            first_chunk = False
                        text_length += len(chunk)
                        if self._telemetry.capture_sensitive_data:
                            chunks.append(chunk)
                        yield chunk
            except asyncio.CancelledError, GeneratorExit:
                raise
            except BaseException as exc:
                observation.update(
                    level="ERROR",
                    status_message=telemetry_error_message(self._telemetry, exc),
                )
                raise
            finally:
                output: dict[str, Any] = {"text_length": text_length}
                if self._telemetry.capture_sensitive_data:
                    output["text"] = "".join(chunks)
                observation.update(
                    output=output,
                    usage_details=active_usage_holder.get("usage_details"),
                    cost_details=active_usage_holder.get("cost_details"),
                )

    async def aclose(self) -> None:
        """Release the provider SDK client and its connection pools."""
        await self._provider.aclose()


__all__ = ["CompletionModel", "structured_response_format"]
