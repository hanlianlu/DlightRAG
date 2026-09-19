# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Provider-owned chat completion lifecycle and telemetry."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import aclosing
from datetime import UTC, datetime
from typing import Any

from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.ai.catalog import resolve_model_profile
from dlightrag.engine.ai.fingerprints import model_invocation_fingerprint
from dlightrag.engine.ai.providers import get_provider
from dlightrag.engine.ai.reasoning import (
    REASONING_LEVELS,
    ResolvedReasoning,
    merge_reasoning_kwargs,
    resolve_reasoning,
)
from dlightrag.engine.ai.replay import messages_for_model
from dlightrag.engine.ai.response_policy import validate_response_extensions
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.settings import ModelSettings
from dlightrag.engine.ai.structured import StructuredOutput
from dlightrag.engine.ai.structured_transport import (
    JSON_SCHEMA_TRANSPORT_CACHE,
    confirms_json_schema_unsupported,
    rejects_json_schema,
)
from dlightrag.engine.ai.telemetry import (
    NOOP_TELEMETRY,
    Telemetry,
    telemetry_error_message,
    telemetry_messages,
)

logger = logging.getLogger(__name__)


_JSON_OBJECT_HINT = "Respond with JSON."
_JSON_OBJECT_FORMAT = {"type": "json_object"}


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
) -> dict[str, Any]:
    """Resolve the configured structured-output transport.

    ``StructuredOutput.response_format_for_provider`` owns the protocol-to-transport
    decision; this only applies the operator's explicit opt-out. Every protocol in
    ``ChatProvider`` serves a strict schema, so ``auto`` and ``json_schema`` agree.
    """
    if settings.structured_output == "json_object":
        return {"type": "json_object"}
    return structured_output.response_format_for_provider(settings.provider)


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
        self.fingerprint = model_invocation_fingerprint(settings)
        self._scheduler = scheduler
        self._telemetry = telemetry
        self._provider = get_provider(
            settings.provider,
            api_key=settings.api_key,
            base_url=settings.base_url,
            api_family=settings.api_family,
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
            "api_family": self.fingerprint.api_family,
        }
        metadata.update(
            {
                key: value
                for key, value in request.items()
                if key not in {"structured_output", "response_format", "model_profile", "reasoning"}
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
            "input": (
                telemetry_messages(messages) if self._telemetry.capture_sensitive_data else None
            ),
            "metadata": metadata or None,
            "model": self.fingerprint.model,
            "model_parameters": model_parameters or None,
        }

    def _request_options(
        self,
        request: dict[str, Any],
    ) -> tuple[dict[str, Any], Any, Any, ResolvedReasoning | None]:
        response_format = request.pop("response_format", None)
        max_tokens = request.pop("max_tokens", None)
        structured_output = request.pop("structured_output", None)
        model_profile = request.pop("model_profile", None)
        if model_profile is not None and not isinstance(model_profile, ModelProfile):
            raise TypeError("model_profile must be a ModelProfile")
        requested = request.pop("reasoning", self.settings.reasoning)
        if requested is not None and requested not in REASONING_LEVELS:
            raise ValueError(f"unsupported reasoning level: {requested!r}")
        resolved = resolve_reasoning(
            (model_profile or resolve_model_profile(self.fingerprint.endpoint)).reasoning,
            requested,
        )
        if resolved is not None and resolved.requested != resolved.effective:
            logger.info(
                "Clamped reasoning level for %s from %s to %s",
                self.settings.model,
                resolved.requested,
                resolved.effective,
            )
        if structured_output is not None:
            if not isinstance(structured_output, StructuredOutput):
                raise TypeError("structured_output must be a StructuredOutput")
            response_format = response_format or structured_response_format(
                structured_output,
                self.settings,
            )
            if (
                isinstance(response_format, dict)
                and response_format.get("type") == "json_schema"
                and JSON_SCHEMA_TRANSPORT_CACHE.rejected(self.fingerprint)
            ):
                # This endpoint already rejected the type; skip the known 400.
                # The runtime downgrade below would have produced the same
                # json_object request anyway.
                response_format = _JSON_OBJECT_FORMAT
        raw = {**self.settings.model_kwargs_copy(), **request}
        if self.settings.api_family == "response":
            validate_response_extensions(raw)
        return (
            merge_reasoning_kwargs(
                raw,
                resolved,
                api_family=self.settings.api_family,
            ),
            response_format,
            max_tokens,
            resolved,
        )

    @staticmethod
    def _reasoning_metadata(resolved: ResolvedReasoning | None) -> dict[str, str]:
        if resolved is None:
            return {}
        return {
            "reasoning_requested": resolved.requested,
            "reasoning_effective": resolved.effective,
        }

    def _approve_json_object_retry(
        self,
        exc: BaseException,
        *,
        structured_output: object,
        response_format: object,
    ) -> bool:
        """Decide, remember, and announce one json_object retry after a rejection."""
        eligible = (
            structured_output is not None
            and self.settings.provider == "openai"
            and isinstance(response_format, dict)
            and response_format.get("type") == "json_schema"
            and rejects_json_schema(exc)
        )
        if not eligible:
            return False
        if confirms_json_schema_unsupported(exc):
            JSON_SCHEMA_TRANSPORT_CACHE.remember_rejected(self.fingerprint)
        logger.warning(
            "Strict structured output failed for %s; retrying with json_object: %s",
            self.settings.model,
            exc,
        )
        return True

    def _provider_stream(
        self,
        *,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None,
        max_tokens: Any,
        model_kwargs: dict[str, Any],
        usage_holder: dict[str, Any],
    ) -> AsyncGenerator[str]:
        return self._provider.stream(
            messages=_messages_for_json_object(messages, response_format),
            model=self.settings.model,
            temperature=self.settings.temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            model_kwargs=model_kwargs,
            usage_holder=usage_holder,
        )

    async def _complete(
        self,
        messages: list[dict[str, Any]],
        request: dict[str, Any],
    ) -> Any:
        structured_output = request.get("structured_output")
        model_kwargs, response_format, max_tokens, resolved = self._request_options(dict(request))
        observation_kwargs = self._observation_kwargs(
            messages,
            {**request, **self._reasoning_metadata(resolved)},
        )
        async with self._telemetry.observe(
            "generate-completion",
            **observation_kwargs,
        ) as observation:
            prepared = messages_for_model(messages, self.fingerprint)
            outbound = _messages_for_json_object(
                prepared,
                response_format,
            )
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
                if self._approve_json_object_retry(
                    exc,
                    structured_output=structured_output,
                    response_format=response_format,
                ):
                    try:
                        result = await self._provider.complete(
                            messages=_messages_for_json_object(
                                prepared,
                                _JSON_OBJECT_FORMAT,
                            ),
                            model=self.settings.model,
                            temperature=self.settings.temperature,
                            max_tokens=max_tokens,
                            response_format=_JSON_OBJECT_FORMAT,
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
        model_kwargs, response_format, max_tokens, resolved = self._request_options(dict(request))
        observation_kwargs = self._observation_kwargs(
            messages,
            {**request, **self._reasoning_metadata(resolved)},
        )
        structured_output = request.get("structured_output")
        active_usage_holder = usage_holder if usage_holder is not None else {}
        chunks: list[str] = []
        text_length = 0
        first_chunk = True
        yielded = False
        async with self._telemetry.observe(
            "generate-completion",
            **observation_kwargs,
        ) as observation:
            prepared = messages_for_model(messages, self.fingerprint)

            def _record_chunk(chunk: str) -> None:
                nonlocal first_chunk, text_length
                if first_chunk:
                    observation.update(completion_start_time=datetime.now(UTC))
                    first_chunk = False
                text_length += len(chunk)
                if self._telemetry.capture_sensitive_data:
                    chunks.append(chunk)

            try:
                stream = self._provider_stream(
                    messages=prepared,
                    response_format=response_format,
                    max_tokens=max_tokens,
                    model_kwargs=model_kwargs,
                    usage_holder=active_usage_holder,
                )
                async with aclosing(stream):
                    async for chunk in stream:
                        yielded = True
                        _record_chunk(chunk)
                        yield chunk
            except asyncio.CancelledError, GeneratorExit:
                raise
            except BaseException as exc:
                if yielded or not self._approve_json_object_retry(
                    exc,
                    structured_output=structured_output,
                    response_format=response_format,
                ):
                    observation.update(
                        level="ERROR",
                        status_message=telemetry_error_message(self._telemetry, exc),
                    )
                    raise
                try:
                    stream = self._provider_stream(
                        messages=prepared,
                        response_format=_JSON_OBJECT_FORMAT,
                        max_tokens=max_tokens,
                        model_kwargs=model_kwargs,
                        usage_holder=active_usage_holder,
                    )
                    async with aclosing(stream):
                        async for chunk in stream:
                            yielded = True
                            _record_chunk(chunk)
                            yield chunk
                except asyncio.CancelledError, GeneratorExit:
                    raise
                except BaseException as fallback_exc:
                    observation.update(
                        level="ERROR",
                        status_message=telemetry_error_message(
                            self._telemetry,
                            fallback_exc,
                        ),
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
