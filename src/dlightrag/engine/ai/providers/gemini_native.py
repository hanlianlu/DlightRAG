# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Google Gemini through the stateless Interactions API."""

import json
import re
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai._gaos.lib.compat_errors import APIStatusError

from dlightrag.engine.ai.messages import (
    AssistantTurn,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    content_with_attachments,
    message_text,
)
from dlightrag.engine.ai.providers.base import (
    CompletionOutput,
    CompletionProvider,
    capture_stream_usage,
    closing_stream,
    usage_to_dict,
)
from dlightrag.engine.ai.structured import json_schema_from_response_format

#: ``provider_state`` key holding one model turn's output steps as Gemini returned them,
#: so its thoughts and their signatures go back verbatim, in place.
_STEPS_KEY = "interaction_steps"
_OUTPUT_STEPS = frozenset({"thought", "model_output", "function_call"})
_MODEL_KWARGS = frozenset(
    {"safety_settings", "service_tier", "thinking_level", "thinking_summaries"}
)
_GENERATION_KWARGS = ("thinking_level", "thinking_summaries")
_TOOL_CHOICE = {"auto": "auto", "required": "any", "none": "none"}
_DATA_URL = re.compile(r"^data:([^;,]+);base64,(.+)$", re.DOTALL)
#: Interactions usage under the counter names DlightRAG reads for Gemini.
_USAGE_KEYS = {
    "total_input_tokens": "prompt_tokens",
    "total_cached_tokens": "cached_content_tokens",
    "total_output_tokens": "candidates_tokens",
    "total_thought_tokens": "thoughts_tokens",
    "total_tool_use_tokens": "tool_use_prompt_tokens",
}


class InteractionRequestError(ValueError):
    """A canonical request cannot be written as a Gemini interaction."""


class InteractionStatusError(RuntimeError):
    """A Gemini interaction ended without one usable model output."""


def _plain(value: Any) -> Any:
    """A JSON-safe copy of one SDK value, as a Session can store it."""
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json", by_alias=True, exclude_none=True)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


def _image(url: str) -> dict[str, str]:
    match = _DATA_URL.match(url)
    if match is None:
        return {"type": "image", "uri": url}
    return {"type": "image", "mime_type": match.group(1), "data": match.group(2)}


def _content(content: object) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks: list[dict[str, Any]] = []
    for raw in content if isinstance(content, list) else ():
        block = {"type": "text", "text": raw} if isinstance(raw, str) else raw
        kind = block.get("type") if isinstance(block, Mapping) else None
        if kind == "text":
            if block.get("text"):
                blocks.append({"type": "text", "text": str(block["text"])})
        elif kind == "image_url":
            image = block["image_url"]
            blocks.append(_image(str(image["url"] if isinstance(image, Mapping) else image)))
        else:
            raise InteractionRequestError(f"unsupported content part for Gemini: {kind!r}")
    return blocks


def _function_result(message: Mapping[str, Any]) -> dict[str, Any]:
    """One tool result, with the images it took as subcontent of the result itself."""
    text = str(message.get("content") or "")
    images = [
        _image(str(attachment["data_url"]))
        for attachment in message.get("attachments") or ()
        if isinstance(attachment, Mapping) and attachment.get("data_url")
    ]
    return {
        "type": "function_result",
        "call_id": str(message.get("tool_call_id") or ""),
        "name": str(message.get("name") or ""),
        "result": [*_content(text), *images] if images else text,
        "is_error": bool(message.get("is_error", False)),
    }


def _function_call(call: Mapping[str, Any]) -> dict[str, Any]:
    function = call.get("function") or {}
    try:
        arguments = json.loads(str(function.get("arguments") or "{}"))
    except json.JSONDecodeError:
        arguments = {}
    return {
        "type": "function_call",
        "id": str(call.get("id") or ""),
        "name": str(function.get("name") or ""),
        "arguments": arguments if isinstance(arguments, dict) else {},
    }


def _tool_call(step: Mapping[str, Any]) -> ToolCall:
    arguments = step.get("arguments")
    return ToolCall(
        id=str(step.get("id") or ""),
        name=str(step.get("name") or ""),
        arguments=arguments if isinstance(arguments, dict) else {},
        argument_error=None if isinstance(arguments, dict) else "arguments are not an object",
    )


def _turn_parts(steps: list[dict[str, Any]]) -> tuple[str, str, list[ToolCall]]:
    """The text, thought summaries and function calls of one model turn's steps."""
    text: list[str] = []
    reasoning: list[str] = []
    calls: list[ToolCall] = []
    for step in steps:
        kind = step.get("type")
        if kind == "thought":
            reasoning.extend(
                str(item.get("text") or "")
                for item in step.get("summary") or ()
                if item.get("type") == "text"
            )
        elif kind == "model_output":
            for item in step.get("content") or ():
                if item.get("type") != "text":
                    raise InteractionStatusError(f"unsupported Gemini output {item.get('type')!r}")
                text.append(str(item.get("text") or ""))
        elif kind == "function_call":
            calls.append(_tool_call(step))
        else:
            raw = step.get("raw")  # where the SDK keeps a step type it does not know
            kind = raw.get("type", kind) if isinstance(raw, Mapping) else kind
            raise InteractionStatusError(f"unsupported Gemini output step {kind!r}")
    return "".join(text), "".join(reasoning), calls


def _model_turn(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One assistant message as model steps, natively when it carries same-model state."""
    text = message_text(message.get("content"))
    calls = [_function_call(call) for call in message.get("tool_calls") or ()]
    state = message.get("provider_state")
    if state is None:
        output = [{"type": "model_output", "content": [{"type": "text", "text": text}]}]
        return [*(output if text else []), *calls]
    steps = state.get(_STEPS_KEY) if isinstance(state, Mapping) else None
    if not (
        isinstance(steps, list)
        and steps
        and all(isinstance(step, dict) and step.get("type") in _OUTPUT_STEPS for step in steps)
    ):
        raise InteractionRequestError("Gemini replay state is malformed")
    native_text, _reasoning, native_calls = _turn_parts(steps)
    if native_text != text or [(call.id, call.name) for call in native_calls] != [
        (call["id"], call["name"]) for call in calls
    ]:
        raise InteractionRequestError("Gemini replay state does not match the canonical turn")
    # Stateless mode needs every thought back exactly as received, where it was received.
    return [dict(step) for step in steps]


def interaction_input(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """Project complete local context into a system instruction and stateless input steps."""
    system: list[str] = []
    steps: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            system.append(message_text(message.get("content")))
        elif role == "user":
            content = _content(content_with_attachments(message))
            steps.append({"type": "user_input", "content": content})
        elif role == "assistant":
            steps.extend(_model_turn(message))
        elif role == "tool":
            steps.append(_function_result(message))
        else:
            raise InteractionRequestError(f"unsupported message role for Gemini: {role!r}")
    return "\n\n".join(part for part in system if part) or None, steps


def _response_format(response_format: dict[str, Any]) -> dict[str, Any]:
    schema = json_schema_from_response_format(response_format)
    if schema is not None:
        return {"type": "text", "mime_type": "application/json", "schema": schema}
    if response_format.get("type") == "json_object":
        return {"type": "text", "mime_type": "application/json"}
    raise InteractionRequestError(
        f"unsupported response format for Gemini: {response_format.get('type')!r}"
    )


def interaction_request(
    messages: list[dict[str, Any]],
    model: str,
    *,
    max_tokens: int | None,
    model_kwargs: dict[str, Any] | None,
    response_format: dict[str, Any] | None = None,
    tools: list[ToolDefinition] | None = None,
    tool_choice: ToolChoice = "auto",
) -> dict[str, Any]:
    """Build one stateless interaction from canonical messages and provider options."""
    options = dict(model_kwargs or {})
    unknown = sorted(set(options) - _MODEL_KWARGS)
    if unknown:
        raise InteractionRequestError(
            f"Gemini model_kwargs accept only {', '.join(sorted(_MODEL_KWARGS))}; "
            f"got {', '.join(unknown)}"
        )
    generation = {key: options.pop(key) for key in _GENERATION_KWARGS if key in options}
    if max_tokens is not None:
        generation["max_output_tokens"] = max_tokens
    system, steps = interaction_input(messages)
    # Stateless by design: DlightRAG's Sessions own the conversation (durable history,
    # forks, compaction, crash replay, switching providers), so Google holds no second
    # copy, and store=true would retain users' evidence (55 days by default on paid tiers).
    request: dict[str, Any] = {"model": model, "input": steps, "store": False, **options}
    if system is not None:
        request["system_instruction"] = system
    if tools:
        request["tools"] = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in tools
        ]
        generation["tool_choice"] = _TOOL_CHOICE[tool_choice]
    if generation:
        request["generation_config"] = generation
    if response_format is not None:
        request["response_format"] = _response_format(response_format)
    return request


def _usage(usage: object) -> dict[str, int] | None:
    counters = usage_to_dict(usage)
    if not counters:
        return None
    return {_USAGE_KEYS.get(key, key): value for key, value in counters.items()}


@dataclass(frozen=True, slots=True)
class _Outcome:
    """How one interaction ended: its status, output steps, usage and error codes."""

    status: str
    steps: list[dict[str, Any]]
    usage: dict[str, int] | None
    errors: tuple[str, ...] = ()

    @classmethod
    def of(cls, interaction: Any) -> _Outcome:
        return cls(
            status=str(getattr(interaction, "status", None)),
            steps=[_plain(step) for step in getattr(interaction, "steps", None) or ()],
            usage=_usage(getattr(interaction, "usage", None)),
            errors=tuple(
                str(getattr(error, "code", None) or "")
                for error in getattr(interaction, "errors", None) or ()
            ),
        )


def _assistant_turn(outcome: _Outcome) -> AssistantTurn:
    if outcome.status not in {"completed", "requires_action", "incomplete"}:
        codes = ", ".join(code for code in outcome.errors if code)
        suffix = f" ({codes})" if codes else ""
        raise InteractionStatusError(f"Gemini interaction ended {outcome.status!r}{suffix}")
    text, reasoning, calls = _turn_parts(outcome.steps)
    if outcome.status == "incomplete":
        # Out of output tokens: the text so far stands, and an unfinished call never runs.
        return AssistantTurn(
            text=text,
            reasoning=reasoning,
            tool_calls=(),
            stop_reason="length",
            usage_details=outcome.usage,
        )
    if outcome.status == "requires_action" and not calls:
        raise InteractionStatusError("Gemini interaction requires action but made no call")
    if not text and not calls:
        raise InteractionStatusError("Gemini interaction completed without text or calls")
    if any(call.argument_error for call in calls):
        raise InteractionStatusError("Gemini returned function arguments that are not an object")
    thought = any(step.get("type") == "thought" for step in outcome.steps)
    return AssistantTurn(
        text=text,
        reasoning=reasoning,
        tool_calls=tuple(calls),
        stop_reason="tool_use" if calls else "stop",
        usage_details=outcome.usage,
        provider_state={_STEPS_KEY: outcome.steps} if thought else None,
    )


def _completion(turn: AssistantTurn) -> CompletionOutput:
    if turn.tool_calls:
        raise InteractionStatusError("Gemini returned function calls to a request without tools")
    return CompletionOutput(
        turn.text,
        usage_details=turn.usage_details,
        stop_reason=turn.stop_reason,
    )


def _append_text(items: list[dict[str, Any]], text: str) -> None:
    if items and items[-1].get("type") == "text":
        items[-1]["text"] = str(items[-1].get("text") or "") + text
    else:
        items.append({"type": "text", "text": text})


class _InteractionStream:
    """One streamed interaction: text as it arrives, then the steps it completed with.

    The completed event's own steps win; a completed event that carries none is
    answered from the steps the deltas built, so call arguments and signatures are whole.
    """

    def __init__(self) -> None:
        self._steps: dict[int, dict[str, Any]] = {}
        self._arguments: dict[int, list[str]] = {}
        self._usage: Any = None
        self._terminal: Any = None

    def accept(self, event: Any) -> str | None:
        """Take one SSE event and return the text it adds, if any."""
        kind = getattr(event, "event_type", None)
        if kind == "error":
            code = getattr(getattr(event, "error", None), "code", None)
            suffix = f" ({code})" if code else ""
            raise InteractionStatusError(f"Gemini interaction stream failed{suffix}")
        if kind == "step.start":
            step = _plain(event.step)
            self._steps[event.index] = step
            if step.get("type") == "model_output":
                return "".join(
                    str(item.get("text") or "")
                    for item in step.get("content") or ()
                    if item.get("type") == "text"
                )
        elif kind == "step.delta":
            metadata = getattr(event, "metadata", None)
            self._usage = getattr(metadata, "total_usage", None) or self._usage
            return self._delta(event.index, _plain(event.delta))
        elif kind == "step.stop":
            self._usage = getattr(event, "usage", None) or self._usage
        elif kind == "interaction.completed":
            self._terminal = event.interaction
        return None

    def _delta(self, index: int, delta: dict[str, Any]) -> str | None:
        kind = delta.get("type")
        if kind == "text":
            text = str(delta.get("text") or "")
            step = self._steps.setdefault(index, {"type": "model_output"})
            _append_text(step.setdefault("content", []), text)
            return text
        if kind == "thought_summary":
            summary = self._steps.setdefault(index, {"type": "thought"}).setdefault("summary", [])
            content = delta.get("content") or {}
            if content.get("type") == "text":
                _append_text(summary, str(content.get("text") or ""))
            else:
                summary.append(content)
        elif kind == "thought_signature":
            step = self._steps.setdefault(index, {"type": "thought"})
            step["signature"] = str(step.get("signature") or "") + str(delta.get("signature") or "")
        elif kind == "arguments_delta":
            self._arguments.setdefault(index, []).append(str(delta.get("arguments") or ""))
        return None

    def outcome(self) -> _Outcome:
        if self._terminal is None:
            raise InteractionStatusError("Gemini interaction stream ended before completing")
        outcome = _Outcome.of(self._terminal)
        steps = outcome.steps or [self._assembled(index) for index in sorted(self._steps)]
        usage = outcome.usage or _usage(self._usage)
        return _Outcome(outcome.status, steps, usage, outcome.errors)

    def _assembled(self, index: int) -> dict[str, Any]:
        step = self._steps[index]
        if index in self._arguments:
            encoded = "".join(self._arguments[index])
            try:
                step["arguments"] = json.loads(encoded or "{}")
            except json.JSONDecodeError:
                step["arguments"] = encoded  # cut off mid-call: an unfinished call never runs
        return step


class GeminiProvider(CompletionProvider):
    """Google Gemini models through the Interactions API, statelessly.

    Every request carries the complete local context with ``store=false``: no
    ``previous_interaction_id``, background run, webhook, environment or agent.
    ``model_kwargs`` accept only ``safety_settings``, ``service_tier`` and the
    thinking controls typed reasoning writes. Interactions take no sampling
    parameters: ModelSettings refuses a configured temperature for Gemini, so the
    shared ``temperature`` argument (an internal default such as the image probe's)
    is never sent.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client: Any = None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aio.aclose()
            self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            options: dict[str, Any] = {}
            if self._base_url is not None:
                options["base_url"] = self._base_url
            if self._timeout:
                options["timeout"] = int(self._timeout * 1000)
            self._client = genai.Client(
                api_key=self._api_key,
                http_options=genai.types.HttpOptions(**options),
            )
        return self._client

    def _interactions(self) -> Any:
        interactions = self._get_client().aio.interactions
        # The SDK retries a transient failure max_retries times. google-genai's Interactions
        # bridge reads HttpRetryOptions.attempts as that retry count, not as attempts, and
        # cannot express zero, so the budget is set on the resource itself.
        interactions.sdk_configuration.retry_config.max_retries = self._max_retries
        return interactions

    async def _create(self, request: dict[str, Any]) -> Any:
        try:
            return await self._interactions().create(**request)
        except APIStatusError as exc:
            # The HTTP status is the verdict. An error body the SDK's schema does not expect
            # is chained as a parse failure whose "expected schema" text would read as a
            # non-retryable request fault, so nothing is chained behind it.
            exc.__cause__ = exc.__context__ = None
            raise

    async def _events(
        self,
        request: dict[str, Any],
        stream: _InteractionStream,
    ) -> AsyncGenerator[str]:
        events = await self._create({**request, "stream": True})
        async with closing_stream(events):
            async for event in events:
                text = stream.accept(event)
                if text:
                    yield text

    def _turn(self, outcome: _Outcome) -> AssistantTurn:
        turn = _assistant_turn(outcome)
        self.last_reasoning = turn.reasoning
        return turn

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> CompletionOutput:
        del temperature
        request = interaction_request(
            messages,
            model,
            max_tokens=max_tokens,
            model_kwargs=model_kwargs,
            response_format=response_format,
        )
        return _completion(self._turn(_Outcome.of(await self._create(request))))

    async def complete_tool_turn(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        tools: list[ToolDefinition],
        tool_choice: ToolChoice = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> AssistantTurn:
        del temperature
        request = interaction_request(
            messages,
            model,
            max_tokens=max_tokens,
            model_kwargs=model_kwargs,
            tools=tools,
            tool_choice=tool_choice,
        )
        return self._turn(_Outcome.of(await self._create(request)))

    async def complete_tool_turn_streaming(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        tools: list[ToolDefinition],
        emit_text: Callable[[str], Awaitable[None]],
        tool_choice: ToolChoice = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> AssistantTurn:
        del temperature
        request = interaction_request(
            messages,
            model,
            max_tokens=max_tokens,
            model_kwargs=model_kwargs,
            tools=tools,
            tool_choice=tool_choice,
        )
        stream = _InteractionStream()
        events = self._events(request, stream)
        async with aclosing(events):
            async for text in events:
                await emit_text(text)
        return self._turn(stream.outcome())

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        usage_holder: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str]:  # type: ignore[override]
        del temperature
        request = interaction_request(
            messages,
            model,
            max_tokens=max_tokens,
            model_kwargs=model_kwargs,
            response_format=response_format,
        )
        stream = _InteractionStream()
        events = self._events(request, stream)
        async with aclosing(events):
            async for text in events:
                yield text
        output = _completion(self._turn(stream.outcome()))
        capture_stream_usage(usage_holder, output.usage_details)

    async def stream_tool_text(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
        usage_holder: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str]:  # type: ignore[override]
        stream = self.stream(
            messages,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            model_kwargs=model_kwargs,
            usage_holder=usage_holder,
        )
        async with aclosing(stream):
            async for token in stream:
                yield token


__all__ = [
    "GeminiProvider",
    "InteractionRequestError",
    "InteractionStatusError",
    "interaction_input",
    "interaction_request",
]
