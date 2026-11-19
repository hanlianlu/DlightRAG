# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One classifier for every value that would leave the process.

A value is **private** or **exportable** ([ADR 0039][adr]). A private value is
masked to its presence, its length, and — for content, never for a credential —
a digest; an exportable value is exported, bounded by the caller's own limits.
Nothing here asks whether monitoring is on: it always is, and this module is what
makes that safe.

Two rules the code states rather than leaves to a caller:

- **A credential is never digested.** A digest of a low-entropy secret is a
  brute-force oracle, so a secret is replaced by ``[redacted]`` and nothing else.
- **A count under a token name is not a secret.** ``max_tokens`` and
  ``chunk_token_size`` are numbers a reviewer needs; ``otp_token`` is not, and a
  number under a name that ends in ``token`` still hides.

[adr]: ../../../../docs/adr/0039-observability-is-always-on-and-the-seam-masks-private-data.md
"""

from __future__ import annotations

import hashlib
from typing import Any, Final, Literal

type PrivacyClass = Literal["private", "exportable"]

#: Name fragments whose values are secrets, in settings fields, provider options
#: such as request headers, and telemetry payloads. One list, so a secret hidden
#: from a settings dump is hidden from an exported trace too.
SECRET_KEY_PATTERNS: Final[tuple[str, ...]] = (
    "api_key",
    "api-key",
    "api_secret",
    "api_token",
    "authorization",
    "secret",
    "verification_key",
    "password",
    "connection_string",
    "milvus_uri",
    "account_key",
    "sas_token",
    "token",
)

#: Name fragments that make a value a *location on a filesystem* rather than a
#: credential. Kept apart from :data:`SECRET_KEY_PATTERNS` on purpose: a path
#: under such a name is private only while it names a home directory, so it is
#: masked as content (with a basename a reader can follow) rather than redacted
#: as a secret. An endpoint path such as ``/v1/chat/completions`` matches none of
#: this and stays exportable.
PATH_KEY_PATTERNS: Final[tuple[str, ...]] = (
    "path",
    "filepath",
    "file_path",
    "filename",
    "file_name",
    "directory",
    "dirname",
    "folder",
    "cwd",
    "home",
)

#: What a home directory is called, on the platforms this product runs on. A
#: value that merely names a filesystem location is not private; one that names
#: the operator's own account is, because it carries a username and the shape of
#: a personal machine into a shared store.
_HOME_PREFIXES: Final[tuple[str, ...]] = (
    "/Users/",
    "/home/",
    "/root/",
    "\\Users\\",
    "C:\\Users\\",
)

#: What a masked credential is replaced by. No length, no digest: see the module
#: docstring.
REDACTED: Final[str] = "[redacted]"

#: What inline media is replaced by, everywhere, regardless of its key.
IMAGE_OMITTED: Final[str] = "[image omitted]"

#: The bound one exported string carries when the caller states none.
MAX_TEXT: Final[int] = 4000

#: How much of a digest a placeholder carries. Enough to tell two values apart in
#: a trace, too little to reconstruct one.
DIGEST_CHARS: Final[int] = 12


def is_secret_key(key: object) -> bool:
    """Whether a field or header name holds a secret."""
    normalized = str(key).lower()
    return any(pattern in normalized for pattern in SECRET_KEY_PATTERNS)


def names_a_token_count(key: object) -> bool:
    """Whether a number under this name is a count rather than a credential."""
    normalized = str(key).lower()
    matched = {pattern for pattern in SECRET_KEY_PATTERNS if pattern in normalized}
    return matched == {"token"} and not normalized.endswith("token")


def names_a_filesystem_location(key: object) -> bool:
    """Whether a value under this name is a path, a directory, or a filename."""
    normalized = str(key).lower()
    return any(pattern in normalized for pattern in PATH_KEY_PATTERNS)


def looks_like_a_home_path(value: object) -> bool:
    """Whether a value names a home directory, and so the operator who owns it.

    Only text is inspected: a path is a string, and a value of any other type
    under a path name is left to the branches that already handle it. Absence is
    a fact about a value, never a reason to export it.
    """
    if not isinstance(value, str):
        return False
    return any(prefix in value for prefix in _HOME_PREFIXES)


def hides_secret_value(key: object, value: object) -> bool:
    """Whether a value under the secret name ``key`` must be hidden.

    Anything that can carry the secret hides unless it is empty: text, bytes,
    containers, models and other objects alike, so a value type nobody listed
    cannot slip out. A flag stays readable, since one bit carries no credential.
    A number stays readable only as a count under a name that merely contains
    "token" (``max_tokens``, ``chunk_token_size``); under any other secret name,
    or a name that ends in "token" (``otp_token``), a number can be the
    credential itself.
    """
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return not names_a_token_count(key)
    try:
        return bool(value)
    except Exception:  # a truth value that cannot be read still hides
        return True


#: Which rule made a value private. One statement of the rule, so the class a caller
#: acts on (:func:`classify`) and the rendering a masker writes (:func:`_masked_value`)
#: are read off the same answer rather than re-tested apart.
type PrivateShape = Literal["credential", "path", "bytes", "media", "content"]


def _private_shape(key: object, value: object) -> PrivateShape | None:
    """Which private rule a named value falls under, or None when it is exportable.

    The whole rule lives here, and both answers come off it: what leaves (``private``
    or ``exportable``) and how a private value is rendered. A second caller that
    recomposed the halves of a rule — "looks like a home path" without "the key names
    a filesystem location", say — would be a second statement of it, free to disagree
    with this one about the same value.
    """
    if is_secret_key(key) and hides_secret_value(key, value):
        return "credential"
    if names_a_filesystem_location(key) and looks_like_a_home_path(value):
        # A path that names a home directory names the operator: the username and
        # the layout of a personal machine. It is private as *content* — the
        # basename stays readable so a reader can still follow the line — and
        # never as a credential, which is why it is not a SECRET_KEY_PATTERN.
        return "path"
    if isinstance(value, bytes | bytearray | memoryview):
        return "bytes"
    if isinstance(value, dict) and value.get("type") == "image_url":
        return "media"
    return None


def classify(key: object, value: object) -> PrivacyClass:
    """Classify one named value. Private means masked before it leaves."""
    return "exportable" if _private_shape(key, value) is None else "private"


def digest(value: object) -> str:
    """Return the short content digest a private placeholder carries.

    A value whose text cannot be read still has a digest: it is taken over the
    type's name, because failing to describe a value must not fail the export.
    """
    try:
        encoded = value if isinstance(value, bytes) else str(value).encode("utf-8", "surrogatepass")
    except Exception:
        encoded = type(value).__name__.encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:DIGEST_CHARS]


def private_placeholder(value: object, *, label: str | None = None) -> str:
    """Return the masked form of a private *content* value.

    Never call this for a credential: :data:`REDACTED` is what a secret becomes.
    A placeholder states that a value existed and how large it was, which is what
    makes a redacted trace diagnosable without carrying the value. A value that
    cannot describe itself is still masked, not returned.
    """
    name = "private" if label is None else f"private {label}"
    if isinstance(value, bytes | bytearray | memoryview):
        raw = bytes(value)
        return f"[{name}: {len(raw)} bytes, sha256:{digest(raw)}]"
    try:
        text = str(value)
    except Exception:
        return f"[{name}: unprintable value of type {type(value).__name__}]"
    return f"[{name}: {len(text)} chars, sha256:{digest(text)}]"


def private_path_placeholder(value: object) -> str:
    """Return the masked form of a path that names a home directory.

    The directory that carries the username is dropped; the *basename* survives,
    because it is what makes a masked log line followable — a reader sees that
    ``toolchain.py`` logged the line without learning where the operator keeps it.
    The ADR 0039 shape is kept around it, so the digest is the correlation handle
    for a file whose name is not distinctive. A value that cannot describe itself
    is still masked, never returned.
    """
    try:
        text = str(value)
    except Exception:
        return f"[private path: unprintable value of type {type(value).__name__}]"
    basename = text.replace("\\", "/").rstrip("/").rpartition("/")[2]
    label = f"private path: {basename}" if basename else "private path"
    return f"[{label}: {len(text)} chars, sha256:{digest(text)}]"


def bounded_text(value: object, *, max_length: int = MAX_TEXT) -> str:
    """Bound one exported string while preserving its original line structure."""
    try:
        text = str(value)
    except Exception:
        return f"[unprintable value of type {type(value).__name__}]"
    if len(text) <= max_length:
        return text
    return f"{text[:max_length]}... [truncated {len(text) - max_length} chars]"


def mask_payload(data: Any) -> Any:
    """Mask a payload before export: secrets, media, bytes, and long text."""
    if data is None or isinstance(data, bool | int | float):
        return data
    if isinstance(data, dict):
        if data.get("type") == "image_url":
            return {"type": "image_url", "image_url": IMAGE_OMITTED}
        masked: dict[Any, Any] = {}
        for key, value in data.items():
            readable = _readable_key(key)
            masked[readable] = (
                _masked_value(readable, value)
                if classify(readable, value) == "private"
                else mask_payload(value)
            )
        return masked
    if isinstance(data, list):
        return [mask_payload(item) for item in data]
    if isinstance(data, tuple):
        return [mask_payload(item) for item in data]
    if isinstance(data, bytes | bytearray | memoryview):
        return f"[bytes omitted: {len(bytes(data))}]"
    if isinstance(data, str):
        return bounded_text(data)
    # Fail closed: a type nobody listed becomes a placeholder, never the value.
    return private_placeholder(data)


def _masked_value(key: object, value: object) -> Any:
    """Render a value :func:`classify` called private, by the shape that made it so.

    A credential keeps no length and no digest — a digest of a low-entropy secret
    is a brute-force oracle — while content keeps both, because that is what makes
    a masked payload diagnosable. Media keeps its own marker, and a path keeps its
    basename so the line stays followable. Which of those applies is the
    classifier's own answer: this reads the shape the rule returned above instead
    of testing one half of the rule again.
    """
    shape = _private_shape(key, value)
    if shape == "credential":
        return REDACTED
    if shape == "media":
        return {"type": "image_url", "image_url": IMAGE_OMITTED}
    if shape == "bytes" and isinstance(value, bytes | bytearray | memoryview):
        # The isinstance narrows the value the classifier already called bytes; the
        # rule that decided it is the classifier's, and only the type is read here.
        return f"[bytes omitted: {len(bytes(value))}]"
    if shape == "path":
        return private_path_placeholder(value)
    return private_placeholder(value)


def _readable_key(key: object) -> object:
    """Return a key the classifier can read; an unreadable one is replaced."""
    try:
        str(key)
    except Exception:
        return "[unprintable key]"
    return key


__all__ = [
    "DIGEST_CHARS",
    "IMAGE_OMITTED",
    "MAX_TEXT",
    "PATH_KEY_PATTERNS",
    "REDACTED",
    "SECRET_KEY_PATTERNS",
    "PrivacyClass",
    "bounded_text",
    "classify",
    "digest",
    "hides_secret_value",
    "is_secret_key",
    "looks_like_a_home_path",
    "mask_payload",
    "names_a_filesystem_location",
    "names_a_token_count",
    "private_path_placeholder",
    "private_placeholder",
]
