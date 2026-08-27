# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Provider-owned chat completion lifecycle and telemetry."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import aclosing
from datetime import UTC, datetime
from typing import Any

from dlightrag.engine.ai.fingerprints import model_fingerprint
from dlightrag.engine.ai.providers import get_provider
from dlightrag.engine.ai.providers.base import CompletionProvider
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.settings import ModelSettings
from dlightrag.engine.ai.structured import StructuredOutput
from dlightrag.engine.ai.telemetry import (
    NOOP_TELEMETRY,
    Telemetry,
    telemetry_error_message,
    telemetry_messages,
)

logger = logging.getLogger(__name__)


_JSON_OBJECT_HINT = "Respond with JSON."


def _content_mentions_json(content: Any) -> bool:
    if isinstance(content, str):
        return "json" in content.casefold()
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str) and "json" in part.casefold():
                return True
            if isinstance(part, dict) and "json" in str(part.get("text") or "").casefold():
                return True
    return False


def _append_json_hint(content: Any) -> Any:
    if isinstance(content, str):
        return f"{content.rstrip()}\n{_JSON_OBJECT_HINT}"
    if isinstance(content, list):
        return [*content, {"type": "text", "text": _JSON_OBJECT_HINT}]
    return _JSON_OBJECT_HINT


def _messages_for_json_object(
    messages: list[dict[str, Any]],
    response_format: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """json_object requires the word json in some message; fold it into system."""
    if response_format is None or response_format.get("type") != "json_object":
        return messages
    if any(_content_mentions_json(message.get("content")) for message in messages):
        return messages
    if messages and messages[0].get("role") == "system":
        first = dict(messages[0])
        first["content"] = _append_json_hint(first.get("content"))
        return [first, *messages[1:]]
    return [{"role": "system", "content": _JSON_OBJECT_HINT}, *messages]


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
    if settings.provider == "openai":
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
            outbound = _messages_for_json_object(messages, response_format)
            try:
                result = await self._provider.complete(
                    messages=outbound,
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
                    json_object = {"type": "json_object"}
                    try:
                        result = await self._provider.complete(
                            messages=_messages_for_json_object(messages, json_object),
                            model=self.settings.model,
                            temperature=self.settings.temperature,
                            max_tokens=max_tokens,
                            response_format=json_object,
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
                    messages=_messages_for_json_object(messages, response_format),
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
