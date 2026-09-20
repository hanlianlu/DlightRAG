# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Stateless OpenAI Responses transport under the OpenAI-compatible provider."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from dlightrag.engine.ai.messages import (
    AssistantTurn,
    ToolCall,
    ToolChoice,
    ToolDefinition,
)
from dlightrag.engine.ai.providers.base import CompletionOutput, usage_mapping, usage_to_dict
from dlightrag.engine.ai.response_policy import (
    ResponseRequestError,
    validate_response_extensions,
)

_RESPONSE_REPLAY_KEY = "response_replay"
_RESPONSE_REPLAY_VERSION = 1


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


def _canonical_tool_call(raw: object) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise ResponseRequestError("Response Tool calls must be objects")
    call_id = raw.get("id")
    function = raw.get("function")
    if not isinstance(call_id, str) or not call_id or not isinstance(function, Mapping):
        raise ResponseRequestError("Response Tool call requires id and function")
    name = function.get("name")
    arguments = function.get("arguments")
    if not isinstance(name, str) or not name or not isinstance(arguments, str):
        raise ResponseRequestError("Response Tool call requires name and encoded arguments")
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def _plain_json(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _plain_json(dump(mode="json", exclude_none=True))
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    fields = getattr(value, "__dict__", None)
    if isinstance(fields, dict):
        return {
            str(key): _plain_json(item)
            for key, item in fields.items()
            if not str(key).startswith("_")
        }
    raise ResponseStatusError(f"unsupported native Response value: {type(value).__name__}")


def _item_type(item: object) -> object:
    if isinstance(item, Mapping):
        return item.get("type")
    return getattr(item, "type", None)


def _item_value(item: object, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _message_text(item: object) -> tuple[str, tuple[str, ...]]:
    text: list[str] = []
    refusals: list[str] = []
    content_items = _item_value(item, "content") or ()
    if not isinstance(content_items, list | tuple):
        raise ResponseStatusError("Response message content is malformed")
    for content in content_items:
        content_type = _item_type(content)
        if content_type == "output_text":
            value = _item_value(content, "text")
            if not isinstance(value, str):
                raise ResponseStatusError("Response output_text is malformed")
            text.append(value)
        elif content_type == "refusal":
            refusals.append(str(_item_value(content, "refusal") or "provider refusal"))
        else:
            raise ResponseStatusError(f"unsupported Response message content type {content_type!r}")
    return "".join(text), tuple(refusals)


def _reasoning_text(item: object) -> str:
    summaries: list[str] = []
    for summary in _item_value(item, "summary") or ():
        if _item_type(summary) == "summary_text":
            text = _item_value(summary, "text")
            if isinstance(text, str) and text:
                summaries.append(text)
    if summaries:
        return "".join(summaries)
    content: list[str] = []
    for part in _item_value(item, "content") or ():
        if _item_type(part) == "reasoning_text":
            text = _item_value(part, "text")
            if isinstance(text, str) and text:
                content.append(text)
    return "".join(content)


def _normalized_tool_call(item: object) -> ToolCall:
    call_id = _item_value(item, "call_id")
    name = _item_value(item, "name")
    encoded = _item_value(item, "arguments")
    if not isinstance(call_id, str) or not call_id:
        raise ResponseStatusError("Response function_call requires call_id")
    if not isinstance(name, str) or not name or not isinstance(encoded, str):
        raise ResponseStatusError("Response function_call is malformed")
    try:
        arguments = json.loads(encoded)
        if not isinstance(arguments, dict):
            raise TypeError("tool arguments must be a JSON object")
    except (json.JSONDecodeError, TypeError) as exc:
        return ToolCall(
            id=call_id,
            name=name,
            arguments={},
            argument_error=str(exc),
        )
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _response_parts(
    response: Any,
) -> tuple[str, str, tuple[ToolCall, ...], tuple[str, ...]]:
    text: list[str] = []
    reasoning: list[str] = []
    calls: list[ToolCall] = []
    refusals: list[str] = []
    for item in getattr(response, "output", None) or ():
        item_type = _item_type(item)
        if item_type == "message":
            item_text, item_refusals = _message_text(item)
            text.append(item_text)
            refusals.extend(item_refusals)
        elif item_type == "reasoning":
            value = _reasoning_text(item)
            if value:
                reasoning.append(value)
        elif item_type == "function_call":
            calls.append(_normalized_tool_call(item))
        else:
            raise ResponseStatusError(f"unsupported Response output item type {item_type!r}")
    return "".join(text), "".join(reasoning), tuple(calls), tuple(refusals)


def _validate_finalized_item(item: Mapping[str, Any]) -> None:
    item_type = item.get("type")
    if item_type not in {"reasoning", "message", "function_call"}:
        raise ResponseStatusError(f"unsupported Response replay item type {item_type!r}")
    if item.get("status") not in {None, "completed"}:
        raise ResponseStatusError("Response replay contains a partial output item")
    if item_type == "reasoning":
        if not isinstance(item.get("id"), str):
            raise ResponseStatusError("Response reasoning replay requires an item id")
        return
    if item_type == "message":
        if item.get("role") != "assistant" or not isinstance(item.get("content"), list):
            raise ResponseStatusError("Response message replay is malformed")
        _message_text(item)
        return
    _normalized_tool_call(item)


def _finalized_output_items(response: Any) -> list[dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    for raw in getattr(response, "output", None) or ():
        item = _plain_json(raw)
        if not isinstance(item, dict):
            raise ResponseStatusError("Response output item is malformed")
        _validate_finalized_item(item)
        finalized.append(item)
    return finalized


def _native_replay_items(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    state = message.get("provider_state")
    if not isinstance(state, Mapping):
        raise ResponseRequestError("Response provider replay state is malformed")
    replay = state.get(_RESPONSE_REPLAY_KEY)
    if not isinstance(replay, Mapping):
        raise ResponseRequestError("Response provider replay state is malformed")
    if replay.get("v") != _RESPONSE_REPLAY_VERSION:
        raise ResponseRequestError("Response provider replay version is unsupported")
    raw_items = replay.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ResponseRequestError("Response provider replay items are malformed")
    items: list[dict[str, Any]] = []
    try:
        for raw in raw_items:
            if not isinstance(raw, Mapping):
                raise ResponseRequestError("Response provider replay item is malformed")
            item = _plain_json(raw)
            if not isinstance(item, dict):
                raise ResponseRequestError("Response provider replay item is malformed")
            _validate_finalized_item(item)
            items.append(item)
    except ResponseStatusError as exc:
        raise ResponseRequestError(str(exc)) from exc

    native_text: list[str] = []
    native_calls: list[tuple[str, str]] = []
    for item in items:
        if item.get("type") == "message":
            value, refusals = _message_text(item)
            if refusals:
                raise ResponseRequestError("Response replay cannot contain a refusal")
            native_text.append(value)
        elif item.get("type") == "function_call":
            native_calls.append((str(item.get("call_id") or ""), str(item.get("name") or "")))
    content = message.get("content", "")
    if not isinstance(content, str) or "".join(native_text) != content:
        raise ResponseRequestError("Response provider replay text does not match canonical text")
    canonical_calls = [_canonical_tool_call(raw) for raw in message.get("tool_calls") or ()]
    if native_calls != [(call["call_id"], call["name"]) for call in canonical_calls]:
        raise ResponseRequestError("Response provider replay calls do not match canonical calls")
    return items


def response_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project complete canonical local context into stateless Response items."""
    projected: list[dict[str, Any]] = []
    pending_calls: set[str] = set()
    for message in messages:
        role = message.get("role")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending_calls:
                raise ResponseRequestError("Response Tool output requires a matching function call")
            if message.get("attachments"):
                raise ResponseRequestError("Response Tool-result images are not implemented")
            content = message.get("content", "")
            if not isinstance(content, str):
                raise ResponseRequestError("Response Tool output must be text")
            projected.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": content,
                }
            )
            pending_calls.remove(call_id)
            continue

        if pending_calls:
            raise ResponseRequestError("Response function calls are missing Tool outputs")
        if role not in {"system", "developer", "user", "assistant"}:
            raise ResponseRequestError(f"unsupported Response message role: {role!r}")
        if role != "assistant":
            if message.get("tool_calls") or message.get("provider_state") is not None:
                raise ResponseRequestError("Response replay state belongs to an assistant message")
            projected.append(
                {
                    "role": role,
                    "content": _response_content(message.get("content", "")),
                }
            )
            continue

        raw_calls = message.get("tool_calls") or ()
        if not isinstance(raw_calls, list | tuple):
            raise ResponseRequestError("Response Tool calls must be a list")
        calls = [_canonical_tool_call(raw) for raw in raw_calls]
        call_ids = [call["call_id"] for call in calls]
        if len(call_ids) != len(set(call_ids)):
            raise ResponseRequestError("Response Tool call ids must be unique")
        if message.get("provider_state") is not None:
            projected.extend(_native_replay_items(message))
        else:
            content = _response_content(message.get("content", ""))
            if content or not calls:
                projected.append({"role": "assistant", "content": content})
            projected.extend(calls)
        pending_calls.update(call_ids)
    if pending_calls:
        raise ResponseRequestError("Response function calls are missing Tool outputs")
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


def _cost_details(usage: Any) -> dict[str, float] | None:
    cost = usage_mapping(usage).get("cost")
    if isinstance(cost, int | float) and not isinstance(cost, bool):
        return {"total": float(cost)}
    return None


def _terminal_status(response: Any) -> str:
    status = getattr(response, "status", None)
    if status == "failed":
        error = getattr(response, "error", None)
        code = getattr(error, "code", None)
        suffix = f" ({code})" if code else ""
        raise ResponseStatusError(f"Responses request failed{suffix}")
    if status not in {"completed", "incomplete"}:
        raise ResponseStatusError(f"Responses request ended in unsupported status {status!r}")
    if status == "incomplete":
        details = getattr(response, "incomplete_details", None)
        if getattr(details, "reason", None) != "max_output_tokens":
            raise ResponseStatusError("Responses request ended incomplete")
    return status


def _completion_output(response: Any) -> CompletionOutput:
    status = _terminal_status(response)
    text, _reasoning, calls, refusals = _response_parts(response)
    if refusals:
        raise ResponseStatusError("Responses request was refused")
    if calls:
        raise ResponseStatusError("Responses text request returned unexpected function calls")
    if status == "completed" and not text:
        raise ResponseStatusError("Responses request completed without text")
    usage = getattr(response, "usage", None)
    return CompletionOutput(
        text,
        usage_details=usage_to_dict(usage),
        cost_details=_cost_details(usage),
        stop_reason="stop" if status == "completed" else "length",
    )


def _assistant_turn(response: Any) -> AssistantTurn:
    status = _terminal_status(response)
    text, reasoning, calls, refusals = _response_parts(response)
    if refusals:
        raise ResponseStatusError("Responses request was refused")
    usage = getattr(response, "usage", None)
    if status == "incomplete":
        return AssistantTurn(
            text=text,
            reasoning=reasoning,
            tool_calls=(),
            stop_reason="length",
            usage_details=usage_to_dict(usage),
            cost_details=_cost_details(usage),
        )
    if not text and not calls:
        raise ResponseStatusError("Responses request completed without text or function calls")
    items = _finalized_output_items(response)
    return AssistantTurn(
        text=text,
        reasoning=reasoning,
        tool_calls=calls,
        stop_reason="tool_use" if calls else "stop",
        usage_details=usage_to_dict(usage),
        cost_details=_cost_details(usage),
        provider_state={
            _RESPONSE_REPLAY_KEY: {
                "v": _RESPONSE_REPLAY_VERSION,
                "items": items,
            }
        },
    )


def _response_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "strict": False,
        }
        for tool in tools
    ]


def _call_kwargs(
    messages: list[dict[str, Any]],
    model: str,
    *,
    temperature: float | None,
    max_tokens: int | None,
    response_format: dict[str, Any] | None,
    model_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
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
    return call_kwargs


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
    call_kwargs = _call_kwargs(
        messages,
        model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        model_kwargs=model_kwargs,
    )
    response = await client.responses.create(**call_kwargs)
    return _completion_output(response)


async def complete_response_tool_turn(
    client: Any,
    messages: list[dict[str, Any]],
    model: str,
    *,
    tools: list[ToolDefinition],
    tool_choice: ToolChoice,
    temperature: float | None,
    max_tokens: int | None,
    model_kwargs: dict[str, Any] | None,
) -> AssistantTurn:
    """Issue one local-function Response turn and retain finalized replay items."""
    call_kwargs = _call_kwargs(
        messages,
        model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=None,
        model_kwargs=model_kwargs,
    )
    if tools:
        call_kwargs["tools"] = _response_tools(tools)
        call_kwargs["tool_choice"] = tool_choice
        call_kwargs["parallel_tool_calls"] = True
    response = await client.responses.create(**call_kwargs)
    return _assistant_turn(response)


__all__ = [
    "ResponseStatusError",
    "complete_response",
    "complete_response_tool_turn",
    "response_input",
    "response_text_config",
]
