# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What one model call ran with, as the wrapper that sent it knows it.

A step of a Session records the model, the parameters and the prompt it ran on (ADR 0043).
The wrapper that calls the provider knows the first two for the attempt that produced the
result, so it states them here and hands them up with the result. It knows nothing of the Run,
the role or the entry, which belong to whoever commits the result.

Only an allowlist leaves the wrapper: the identity of the endpoint (a fingerprint, never its
URL), the sampling and output limits, the type of the response format, the reasoning level
requested and the one used, a digest of the remaining provider options, and two timings: how long
the request waited for a slot in the model scheduler and how long the attempt that produced the
result took. The options themselves, request headers, credentials and response ids never leave it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from dlightrag.engine.ai.telemetry import hides_secret_value, is_secret_key

if TYPE_CHECKING:
    from dlightrag.engine.ai.fingerprints import ModelInvocationFingerprint
    from dlightrag.engine.ai.reasoning import ResolvedReasoning

#: What a credential is replaced by before the options are digested. No length and no digest of
#: its own: a digest of a low-entropy secret is a brute-force oracle, so the secret is not hashed
#: and rotating one does not move the options digest.
REDACTED: Final = "[redacted]"

#: Parts of an option's name that make its value route or authenticate a request rather than
#: shape what the model does: a header container (`extra_headers`, `default_headers`), an
#: authentication object, cookies, credentials, a query string, a URL or an endpoint. Such an
#: option is left out whole, by what it holds, because naming the entries inside a header
#: container would miss the one called `X-Custom`.
_REQUEST_CONTAINER_MARKERS: Final = (
    "header",
    "auth",
    "cookie",
    "credential",
    "query",
    "url",
    "endpoint",
)

#: The longest model name a server may state. It arrives from outside the process and is stored
#: with the step, so an answer that is not a plain name states nothing.
_SERVED_MODEL_MAX_CHARS: Final = 200


def served_model_name(value: object) -> str | None:
    """The model name a server states for a response, when it states a plain one.

    A name that is not text, is empty, is long, or holds a control character states nothing: the
    step is stored whatever the server said, and a value PostgreSQL cannot hold would otherwise
    fail the commit of an answer for the sake of a note about it.
    """
    if not isinstance(value, str):
        return None
    name = value.strip()
    if not name or len(name) > _SERVED_MODEL_MAX_CHARS or not name.isprintable():
        return None
    return name


def _milliseconds(seconds: float) -> int:
    """Whole milliseconds in an interval read off the monotonic clock, never negative."""
    return max(0, round(seconds * 1000))


def _is_request_container(key: str) -> bool:
    normalized = key.lower()
    return any(marker in normalized for marker in _REQUEST_CONTAINER_MARKERS)


def _without_credentials(value: Any) -> Any:
    """The same options with every request container dropped and every secret masked."""
    if isinstance(value, Mapping):
        kept: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if _is_request_container(key):
                continue
            if is_secret_key(key) and hides_secret_value(key, item):
                kept[key] = REDACTED
            else:
                kept[key] = _without_credentials(item)
        return kept
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_without_credentials(item) for item in value]
    return value


def options_digest(model_kwargs: Mapping[str, Any]) -> str:
    """SHA-256 over the provider options, with request containers dropped and secrets masked.

    It tells a reader that two calls ran with the same options, and nothing of what they were.
    Equal options give equal digests whatever order they were written in, and whichever
    credential they carried.
    """
    canonical = json.dumps(
        _without_credentials(model_kwargs),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelCall:
    """The model and parameters one call really ran with.

    It describes the attempt that produced the result, not the call as first planned: an agentic
    request that fell back to ordinary reasoning, or a strict schema that fell back to a JSON
    object, states the parameters it was finally sent with.

    ``queued_ms`` is how long the request waited for a slot in the model scheduler before it was
    sent, and 0 when it did not wait. ``duration_ms`` is how long the attempt that produced the
    result took, from sending its request to the end of its response (for a stream, to the last
    chunk read, which counts a consumer that is slow to take the chunks). Both are whole
    milliseconds read off the monotonic clock, and neither takes part in any digest.
    """

    fingerprint: ModelInvocationFingerprint
    served_model: str | None
    temperature: float | None
    max_tokens: int | None
    tool_choice: str | None
    response_format: str | None
    reasoning: tuple[str, str] | None
    options_digest: str
    queued_ms: int
    duration_ms: int

    @classmethod
    def of(
        cls,
        fingerprint: ModelInvocationFingerprint,
        *,
        served_model: object,
        temperature: float | None,
        max_tokens: int | None,
        tool_choice: str | None,
        response_format: Mapping[str, Any] | None,
        reasoning: ResolvedReasoning | None,
        model_kwargs: Mapping[str, Any],
        queued_seconds: float,
        duration_seconds: float,
    ) -> ModelCall:
        """State a call from what the wrapper sent: its settings, limits, merged options and
        the two intervals it timed, in seconds of the monotonic clock."""
        format_type = response_format.get("type") if response_format else None
        return cls(
            fingerprint=fingerprint,
            served_model=served_model_name(served_model),
            temperature=temperature,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            response_format=format_type if isinstance(format_type, str) else None,
            reasoning=None if reasoning is None else (reasoning.requested, reasoning.effective),
            options_digest=options_digest(model_kwargs),
            queued_ms=_milliseconds(queued_seconds),
            duration_ms=_milliseconds(duration_seconds),
        )

    def as_json(self) -> dict[str, Any]:
        """The model, parameters and timings as a step stores them."""
        return {
            "model": {**self.fingerprint.as_json(), "served_model": self.served_model},
            "params": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "tool_choice": self.tool_choice,
                "response_format": self.response_format,
                "reasoning": (
                    None
                    if self.reasoning is None
                    else {"requested": self.reasoning[0], "effective": self.reasoning[1]}
                ),
                "options_digest": self.options_digest,
            },
            "queued_ms": self.queued_ms,
            "duration_ms": self.duration_ms,
        }


__all__ = ["REDACTED", "ModelCall", "options_digest", "served_model_name"]
