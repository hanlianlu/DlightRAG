# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Stateless OpenAI Responses transport under the OpenAI-compatible provider."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dlightrag.engine.ai.providers.base import CompletionOutput, usage_mapping, usage_to_dict
from dlightrag.engine.ai.response_policy import (
    ResponseRequestError,
    validate_response_extensions,
)


class ResponseStatusError(RuntimeError):
    """A Responses request ended without one valid completed model output."""


def _response_content(content: object) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ResponseRequestError("Response message content must be text or content parts")
    projected: list[dict[str, Any]] = []
    for raw in content:
        if isinstance(raw, str):
            projected.append({"type": "input_text", "text": raw})
            continue
        if not isinstance(raw, Mapping):
            raise ResponseRequestError("Response content parts must be objects")
        part_type = raw.get("type")
        if part_type == "text":
            projected.append({"type": "input_text", "text": str(raw.get("text") or "")})
            continue
        raise ResponseRequestError(f"unsupported Response content part: {part_type!r}")
    return projected


def response_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project complete canonical local context into stateless Response messages."""
    projected: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role not in {"system", "developer", "user", "assistant"}:
            raise ResponseRequestError(f"unsupported Response message role: {role!r}")
        if message.get("tool_calls"):
            raise ResponseRequestError("Response tool history requires the tool transport")
        if message.get("provider_state") is not None:
            raise ResponseRequestError("Response provider state requires native replay support")
        projected.append(
            {
                "role": role,
                "content": _response_content(message.get("content", "")),
            }
        )
    return projected


def response_text_config(response_format: dict[str, Any] | None) -> dict[str, Any] | None:
    """Translate DlightRAG's structured-output contract to Responses text.format."""
    if response_format is None:
        return None
    format_type = response_format.get("type")
    if format_type == "json_object":
        return {"format": {"type": "json_object"}}
    if format_type != "json_schema":
        raise ResponseRequestError(f"unsupported Response text format: {format_type!r}")
    schema = response_format.get("json_schema")
    if not isinstance(schema, Mapping):
        raise ResponseRequestError("json_schema response format requires an object")
    name = schema.get("name")
    definition = schema.get("schema")
    if not isinstance(name, str) or not name or not isinstance(definition, Mapping):
        raise ResponseRequestError("json_schema response format requires name and schema")
    return {
        "format": {
            "type": "json_schema",
            "name": name,
            "schema": dict(definition),
            "strict": bool(schema.get("strict", True)),
        }
    }


def _output_text(response: Any) -> tuple[str, tuple[str, ...]]:
    text: list[str] = []
    refusals: list[str] = []
    for item in getattr(response, "output", None) or ():
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", None) or ():
            content_type = getattr(content, "type", None)
            if content_type == "output_text":
                value = getattr(content, "text", None)
                if isinstance(value, str):
                    text.append(value)
            elif content_type == "refusal":
                refusal = getattr(content, "refusal", None)
                refusals.append(str(refusal or "provider refusal"))
    return "".join(text), tuple(refusals)


def _cost_details(usage: Any) -> dict[str, float] | None:
    cost = usage_mapping(usage).get("cost")
    if isinstance(cost, int | float) and not isinstance(cost, bool):
        return {"total": float(cost)}
    return None


def _completion_output(response: Any) -> CompletionOutput:
    status = getattr(response, "status", None)
    text, refusals = _output_text(response)
    if refusals:
        raise ResponseStatusError("Responses request was refused")
    if status == "completed":
        if not text:
            raise ResponseStatusError("Responses request completed without text")
        stop_reason = "stop"
    elif status == "incomplete":
        details = getattr(response, "incomplete_details", None)
        if getattr(details, "reason", None) != "max_output_tokens":
            raise ResponseStatusError("Responses request ended incomplete")
        stop_reason = "length"
    elif status == "failed":
        error = getattr(response, "error", None)
        code = getattr(error, "code", None)
        suffix = f" ({code})" if code else ""
        raise ResponseStatusError(f"Responses request failed{suffix}")
    else:
        raise ResponseStatusError(f"Responses request ended in unsupported status {status!r}")
    usage = getattr(response, "usage", None)
    return CompletionOutput(
        text,
        usage_details=usage_to_dict(usage),
        cost_details=_cost_details(usage),
        stop_reason=stop_reason,
    )


async def complete_response(
    client: Any,
    messages: list[dict[str, Any]],
    model: str,
    *,
    temperature: float | None,
    max_tokens: int | None,
    response_format: dict[str, Any] | None,
    model_kwargs: dict[str, Any] | None,
) -> CompletionOutput:
    """Issue one foreground, stateless non-streaming Responses request."""
    call_kwargs: dict[str, Any] = {
        "model": model,
        "input": response_input(messages),
        "background": False,
        "store": False,
        "truncation": "disabled",
    }
    if temperature is not None:
        call_kwargs["temperature"] = temperature
    if max_tokens is not None:
        call_kwargs["max_output_tokens"] = max_tokens
    text = response_text_config(response_format)
    if text is not None:
        call_kwargs["text"] = text
    if model_kwargs:
        options = dict(model_kwargs)
        reasoning = options.pop("reasoning", None)
        validate_response_extensions(options)
        if reasoning is not None:
            call_kwargs["reasoning"] = reasoning
        if options:
            call_kwargs["extra_body"] = options
    response = await client.responses.create(**call_kwargs)
    return _completion_output(response)


__all__ = [
    "ResponseStatusError",
    "complete_response",
    "response_input",
    "response_text_config",
]
