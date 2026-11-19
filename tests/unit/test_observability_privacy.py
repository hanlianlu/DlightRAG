# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The one privacy classifier: private is masked, exportable is exported.

[ADR 0039](../../docs/adr/0039-observability-is-always-on-and-the-seam-masks-private-data.md)
states the rule; these are the facts a deployment depends on. There is no switch
to turn monitoring off, so nothing here may depend on one.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from uuid import UUID, uuid4

import dlightrag_observability
import pytest
from dlightrag_observability import otlp as otlp_module
from dlightrag_observability.privacy import (
    REDACTED,
    SECRET_KEY_PATTERNS,
    classify,
    digest,
    hides_secret_value,
    is_secret_key,
    mask_payload,
    names_a_token_count,
    private_path_placeholder,
    private_placeholder,
)

from dlightrag.engine.ai.telemetry import SECRET_KEY_PATTERNS as APP_SECRET_KEY_PATTERNS
from dlightrag.engine.ai.telemetry import is_secret_key as _app_is_secret_key


@pytest.mark.parametrize(
    "key",
    ["api_key", "API-KEY", "authorization", "account_key", "sas_token", "connection_string"],
)
def test_a_secret_name_is_recognised_whatever_its_case(key: str) -> None:
    assert is_secret_key(key)


def test_a_count_under_a_token_name_is_not_a_secret() -> None:
    assert names_a_token_count("max_tokens")
    assert names_a_token_count("chunk_token_size")
    assert not names_a_token_count("otp_token")
    assert not hides_secret_value("max_tokens", 512)
    assert hides_secret_value("otp_token", 512)


def test_an_empty_or_boolean_value_under_a_secret_name_stays_readable() -> None:
    assert not hides_secret_value("api_key", None)
    assert not hides_secret_value("api_key", True)
    assert not hides_secret_value("api_key", "")
    assert hides_secret_value("api_key", "sk-1")


def test_a_credential_is_redacted_without_a_digest() -> None:
    """A digest of a low-entropy secret is a brute-force oracle."""
    masked = mask_payload({"api_key": "sk-live-123", "authorization": "Bearer x"})
    assert masked == {"api_key": REDACTED, "authorization": REDACTED}
    assert digest("sk-live-123") not in str(masked)


def test_inline_media_and_bytes_never_travel() -> None:
    masked = mask_payload(
        {
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": "data:..."}]},
            ],
            "blob": b"\x00\x01\x02",
        }
    )
    assert masked["messages"][0]["content"][0]["image_url"] == "[image omitted]"
    assert masked["blob"] == "[bytes omitted: 3]"


def test_a_long_string_is_bounded_and_says_so() -> None:
    masked = mask_payload({"text": "x" * 5000})
    assert str(masked["text"]).endswith("chars]")
    assert len(str(masked["text"])) < 4100


@pytest.mark.parametrize(
    "key",
    [
        "code.file.path",
        "code.filepath",
        "search_toolchain.fd.path",
        "cwd",
        "home",
        "workspace_directory",
        "file_name",
    ],
)
def test_a_path_naming_a_home_directory_is_private_whatever_key_carries_it(key: str) -> None:
    """The operator's account name is not structural metadata.

    The key only has to name a filesystem location; the *value* is what decides.
    """
    assert classify(key, "/Users/operator/Github/dlightrag/src/toolchain.py") == "private"
    assert classify(key, "/home/deploy/app/answer.py") == "private"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        # A container path is a location but names nobody.
        ("code.file.path", "/app/.venv/lib/python3.14/site-packages/dlightrag/x.py"),
        # An endpoint path is a location and is not a filesystem path at all.
        ("endpoint_path", "/v1/chat/completions"),
        ("base_url", "https://api.deepseek.com/v1"),
        # The rule is about home directories, not about the string "/Users/".
        ("note", "the /Users/ directory exists on macOS"),
        ("count", 3),
    ],
)
def test_a_location_that_names_nobody_stays_exportable(key: str, value: object) -> None:
    assert classify(key, value) == "exportable"


def test_a_home_path_keeps_its_basename_and_loses_the_directory() -> None:
    """A reader must still be able to follow the line.

    The basename is what makes a masked log line followable; the directory is what
    carried the operator's name, so it is the directory that goes.
    """
    path = "/Users/operator/.dlightrag/tools/fd/10.5.0/fd"
    masked = mask_payload({"search_toolchain.fd.path": path})
    placeholder = str(masked["search_toolchain.fd.path"])
    assert placeholder == private_path_placeholder(path)
    assert placeholder == f"[private path: fd: {len(path)} chars, sha256:{digest(path)}]"
    assert "/Users/" not in placeholder
    assert "operator" not in placeholder
    # Two records from the same file still correlate through the digest.
    assert digest(path) in placeholder


def test_a_path_that_cannot_describe_itself_is_still_masked() -> None:
    """Fail closed: a path the masker cannot read must not travel as itself."""

    class _Boom:
        def __str__(self) -> str:
            raise RuntimeError("cannot describe itself")

        __hash__ = object.__hash__

    assert "unprintable" in private_path_placeholder(_Boom())


def test_the_path_rule_is_asked_once_and_only_by_the_classifier() -> None:
    """One home for the rule: the classifier, and every caller asks it.

    The exporter used to recompose the rule from its two halves, and the masker used
    to re-test one of them after ``classify`` had already required both, so the same
    rule was stated three times and any copy of it could move alone. No value tells
    the copies apart — they agree about every value they are asked about — so this
    reads the package's own source instead: the two predicates are named in the
    classifier's module and nowhere else, and inside it each is called exactly once,
    by the one function that states the rule.
    """
    package = Path(dlightrag_observability.__file__).parent
    for module in sorted(package.glob("*.py")):
        text = module.read_text(encoding="utf-8")
        if module.name == "privacy.py":
            # One definition of each predicate, plus the one call in the rule table.
            assert text.count("names_a_filesystem_location(") == 2
            assert text.count("looks_like_a_home_path(") == 2
            continue
        assert "names_a_filesystem_location" not in text, module.name
        assert "looks_like_a_home_path" not in text, module.name


def test_a_private_placeholder_names_presence_and_size_not_content() -> None:
    value = "a question the reader typed"
    placeholder = private_placeholder(value)
    assert placeholder == f"[private: {len(value)} chars, sha256:{digest(value)}]"
    assert value not in placeholder


def test_the_two_secret_lists_are_one_rule() -> None:
    """The seam and the settings dump must hide the same names, or one leaks what the other keeps."""
    assert tuple(SECRET_KEY_PATTERNS) == tuple(APP_SECRET_KEY_PATTERNS)
    for name in ("api_key", "authorization", "sas_token", "max_tokens", "query", "model"):
        assert is_secret_key(name) is _app_is_secret_key(name)


class _Boom:
    """A value whose text cannot be read, and whose presence must not break an export."""

    def __str__(self) -> str:
        raise RuntimeError("this value cannot describe itself")

    def __repr__(self) -> str:
        raise RuntimeError("nor this way")


def test_masking_fails_closed_and_never_raises() -> None:
    """A private value the masker cannot describe is masked anyway, and the walk continues."""
    masked = mask_payload({"note": _Boom(), "query": "kept", "blob": b"ab", "count": 3})
    assert "Boom" in str(masked["note"])
    assert masked["query"] == "kept"
    assert masked["count"] == 3


def test_an_unreadable_key_does_not_break_the_walk() -> None:
    class Key:
        def __str__(self) -> str:
            raise RuntimeError("unreadable")

        __hash__ = object.__hash__

    masked = mask_payload({Key(): "value"})
    assert "[unprintable key]" in masked


def test_a_value_no_branch_lists_is_masked_not_returned() -> None:
    """The tail of the walk is a placeholder: an unlisted type must not travel as itself."""
    value = object()
    masked = mask_payload({"payload": value})
    assert masked["payload"] is not value
    assert "private" in str(masked["payload"])


def test_the_classifier_is_the_decision_the_masker_takes() -> None:
    """Every value the classifier calls private is replaced, and every other one is not."""
    payload = {
        "api_key": "sk-1",
        "image": {"type": "image_url", "image_url": "data:..."},
        "blob": b"ab",
        "query": "kept",
    }
    masked = mask_payload(payload)
    for key, value in payload.items():
        expected = "private" if masked[key] != value else "exportable"
        assert classify(key, value) == expected


class _Stage(Enum):
    DRAFT = "draft"


@pytest.mark.parametrize(
    "value",
    [
        UUID("12345678-1234-5678-1234-567812345678"),
        datetime(2026, 11, 19, 8, 30, tzinfo=UTC),
        date(2026, 11, 19),
        _Stage.DRAFT,
        PurePosixPath("/app/answer.py"),
    ],
    ids=["uuid", "datetime", "date", "enum", "path"],
)
def test_an_identifier_or_a_structural_value_is_exported_as_the_text_it_reads_as(
    value: object,
) -> None:
    """ADR 0039 exports identifiers and structural metadata; none of them is a value to hide.

    These types used to fall to the tail of the walk and leave as ``[private: ...]``, so a
    run's id and a record's timestamp were unreadable in the store they were exported to.
    """
    assert mask_payload({"run_id": value}) == {"run_id": str(value)}
    assert mask_payload([value]) == [str(value)]


def test_a_secret_name_still_hides_an_identifier() -> None:
    """The key is read beside the converted value, so a token that is a UUID is a token."""
    assert mask_payload({"api_key": uuid4(), "token": _Stage.DRAFT}) == {
        "api_key": REDACTED,
        "token": REDACTED,
    }


def test_a_path_object_follows_the_path_rule() -> None:
    """A `Path` is text to the classifier: a home directory is private, a container path is not."""
    home = Path("/Users/operator/app/toolchain.py")
    masked = mask_payload(
        {"search_toolchain.fd.path": home, "code.file.path": PurePosixPath("/app/x.py")}
    )
    assert masked["search_toolchain.fd.path"] == private_path_placeholder(str(home))
    assert masked["code.file.path"] == "/app/x.py"


def test_a_set_a_frozenset_and_a_tuple_are_exported_as_lists() -> None:
    masked = mask_payload(
        {"tags": {"alpha"}, "ids": frozenset({"beta"}), "pair": ("gamma", "delta")}
    )
    assert masked == {"tags": ["alpha"], "ids": ["beta"], "pair": ["gamma", "delta"]}
    # What is inside is still classified: a set of secrets under a secret name is hidden whole.
    assert mask_payload({"api_key": {"sk-1"}}) == {"api_key": REDACTED}


def test_the_exporter_serializes_these_types_without_a_fallback_encoder() -> None:
    """`json.dumps` needs no ``default=str`` once the masker has returned only JSON types."""
    attribute = otlp_module._json_attribute(  # noqa: SLF001 - the shaping step under test
        {
            "run_id": uuid4(),
            "at": datetime(2026, 11, 19, tzinfo=UTC),
            "stage": _Stage.DRAFT,
            "tags": {"x"},
        }
    )
    decoded = json.loads(attribute)
    assert decoded["stage"] == "_Stage.DRAFT"
    assert decoded["tags"] == ["x"]
    assert decoded["at"] == "2026-11-19 00:00:00+00:00"
