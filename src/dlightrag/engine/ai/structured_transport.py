# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Process-local json_schema transport facts for OpenAI-compatible endpoints.

The schema-capable transport is the default, but a minority of compatible
endpoints reject the ``json_schema`` ``response_format`` type outright. That is
a fact about one model endpoint, not about a conversation, so an explicit
rejection is remembered per model fingerprint and later requests go straight to
``json_object`` instead of paying the same 400 again.

Two gates stay separate on purpose. Downgrading one request is cheap, so
:func:`rejects_json_schema` also accepts a schema-validation complaint. Memory
is permanent for the process, so only :func:`confirms_json_schema_unsupported`
wording — the endpoint declaring the type unavailable — is remembered; a
malformed-schema complaint must not silently become a permanent downgrade.
"""

from dlightrag.engine.ai.fingerprints import ModelInvocationFingerprint

#: Any structured-output transport token; one must appear for either gate.
_SCHEMA_TOKENS = ("response_format", "json_schema", "schema")
#: The endpoint declaring the structured-output transport unavailable.
_UNAVAILABLE_MARKERS = (
    "unavailable",
    "unsupported",
    "not supported",
    "not available",
    "does not support",
    "unknown",
    "unrecognized",
    "unrecognised",
)
#: A rejected request is worth one json_object retry but proves no capability.
_REJECTION_MARKERS = (*_UNAVAILABLE_MARKERS, "invalid")


def _mentions_structured_transport(text: str) -> bool:
    return any(token in text for token in _SCHEMA_TOKENS)


def rejects_json_schema(exc: BaseException) -> bool:
    """True when one json_object retry is a sensible reaction to this error."""
    text = str(exc).casefold()
    return _mentions_structured_transport(text) and any(
        marker in text for marker in _REJECTION_MARKERS
    )


def confirms_json_schema_unsupported(exc: BaseException) -> bool:
    """True only when the endpoint declared the json_schema transport unavailable."""
    text = str(exc).casefold()
    return _mentions_structured_transport(text) and any(
        marker in text for marker in _UNAVAILABLE_MARKERS
    )


class JsonSchemaTransportCache:
    """Negative cache of model fingerprints that rejected the json_schema type."""

    def __init__(self) -> None:
        self._rejected: set[ModelInvocationFingerprint] = set()

    def rejected(self, fingerprint: ModelInvocationFingerprint) -> bool:
        return fingerprint in self._rejected

    def remember_rejected(self, fingerprint: ModelInvocationFingerprint) -> None:
        self._rejected.add(fingerprint)

    def clear(self) -> None:
        self._rejected.clear()


#: One process shares the transport lesson across every CompletionModel.
JSON_SCHEMA_TRANSPORT_CACHE = JsonSchemaTransportCache()


__all__ = [
    "JSON_SCHEMA_TRANSPORT_CACHE",
    "JsonSchemaTransportCache",
    "confirms_json_schema_unsupported",
    "rejects_json_schema",
]
