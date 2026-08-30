# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Strict parsing and content-addressing tests for the packaged model catalog."""

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from dlightrag.engine.ai import catalog
from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.ai.catalog import CatalogueEntry
from dlightrag.engine.ai.fingerprints import ModelFingerprint, normalized_endpoint_fingerprint
from dlightrag.engine.ai.reasoning import ReasoningLevels, ReasoningProfile


def _canonical_revision(models: Sequence[object]) -> str:
    canonical = json.dumps(
        models,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _valid_model(
    *,
    provider: object = "openai",
    model: object = "test-model",
    base_url: object = "https://api.example.test/v1",
) -> dict[str, object]:
    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "profile": {
            "context_window_tokens": 1024,
            "max_input_tokens": None,
            "max_output_tokens": 128,
            "supports_images": False,
            "reasoning": {
                "format": "openai",
                "levels": {
                    "off": "none",
                    "minimal": "minimal",
                    "low": "low",
                    "medium": "medium",
                    "high": "high",
                    "xhigh": None,
                    "max": None,
                },
            },
        },
    }


def _payload(models: Sequence[object] | None = None) -> dict[str, object]:
    resolved_models = list(models) if models is not None else [_valid_model()]
    return {
        "revision": _canonical_revision(resolved_models),
        "models": resolved_models,
    }


def _load_text(
    text: str,
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[str, tuple[CatalogueEntry, ...]]:
    (tmp_path / "model_catalog.json").write_text(text, encoding="utf-8")
    monkeypatch.setattr(catalog, "files", lambda _package: tmp_path)
    return catalog._load_catalog()


def _load_payload(
    payload: object,
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[str, tuple[CatalogueEntry, ...]]:
    return _load_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False),
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
    )


def test_packaged_catalog_revision_is_bound_to_canonical_models() -> None:
    path = Path(catalog.__file__).with_name("model_catalog.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert re.fullmatch(r"sha256:[0-9a-f]{64}", payload["revision"])
    assert payload["revision"] == _canonical_revision(payload["models"])


def test_catalog_rejects_malformed_json_without_leaking_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="model catalog is not valid JSON") as exc_info:
        _load_text(
            '{"revision":"should-not-leak",',
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
        )

    assert "should-not-leak" not in str(exc_info.value)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_catalog_rejects_nonstandard_json_constants(
    constant: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    text = '{"revision":"sha256:' + "0" * 64 + f'","models":[{constant}]}}'

    with pytest.raises(RuntimeError, match="model catalog is not valid JSON"):
        _load_text(text, monkeypatch=monkeypatch, tmp_path=tmp_path)


def test_catalog_rejects_semantic_change_with_retained_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _payload()
    profile = payload["models"][0]["profile"]  # type: ignore[index]
    profile["max_output_tokens"] = 64  # type: ignore[index]

    with pytest.raises(RuntimeError, match="revision"):
        _load_payload(payload, monkeypatch=monkeypatch, tmp_path=tmp_path)


@pytest.mark.parametrize(
    "endpoint",
    [
        "not-a-url",
        "ftp://should-not-leak.example/v1",
        "https://should-not-leak.example:not-a-port/v1?token=secret",
    ],
)
def test_catalog_rejects_malformed_nonempty_base_url_without_leaking_it(
    endpoint: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _payload([_valid_model(base_url=endpoint)])

    with pytest.raises(RuntimeError, match=r"models\[0\]\.base_url") as exc_info:
        _load_payload(payload, monkeypatch=monkeypatch, tmp_path=tmp_path)

    message = str(exc_info.value)
    assert "should-not-leak" not in message
    assert "secret" not in message


def test_catalog_rejects_string_false_boolean(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _valid_model()
    model["profile"]["supports_images"] = "false"  # type: ignore[index]

    with pytest.raises(RuntimeError, match=r"models\[0\]\.profile\.supports_images"):
        _load_payload(_payload([model]), monkeypatch=monkeypatch, tmp_path=tmp_path)


def test_catalog_rejects_string_integer_coercion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _valid_model()
    model["profile"]["max_output_tokens"] = "128"  # type: ignore[index]

    with pytest.raises(RuntimeError, match=r"models\[0\]\.profile\.max_output_tokens"):
        _load_payload(_payload([model]), monkeypatch=monkeypatch, tmp_path=tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("context_window_tokens", True),
        ("context_window_tokens", 1024.0),
        ("max_input_tokens", False),
        ("max_output_tokens", 128.0),
        ("supports_images", 0),
        ("reasoning", True),
    ],
)
def test_catalog_rejects_non_exact_profile_json_types(
    field: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _valid_model()
    model["profile"][field] = value  # type: ignore[index]

    with pytest.raises(RuntimeError, match=rf"models\[0\]\.profile\.{field}"):
        _load_payload(_payload([model]), monkeypatch=monkeypatch, tmp_path=tmp_path)


@pytest.mark.parametrize(
    ("payload", "path"),
    [
        ([], "root"),
        ({"revision": f"sha256:{'0' * 64}", "models": {}}, "root.models"),
        (_payload([None]), "models[0]"),
        (_payload([{"provider": "openai"}]), "models[0]"),
        (_payload([{**_valid_model(), "profile": []}]), "models[0].profile"),
    ],
)
def test_catalog_rejects_invalid_root_models_item_and_profile_shapes(
    payload: object,
    path: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match=re.escape(path)):
        _load_payload(payload, monkeypatch=monkeypatch, tmp_path=tmp_path)


@pytest.mark.parametrize(
    ("level", "key", "change"),
    [
        ("root", "revision", "missing"),
        ("root", "models", "missing"),
        ("root", "unexpected", "extra"),
        ("model", "base_url", "missing"),
        ("model", "api_key", "extra"),
        ("profile", "max_input_tokens", "missing"),
        ("profile", "supports_images", "missing"),
        ("profile", "support_images", "extra"),
    ],
)
def test_catalog_rejects_missing_and_extra_keys(
    level: str,
    key: str,
    change: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _payload()
    model = payload["models"][0]  # type: ignore[index]
    containers: dict[str, dict[str, Any]] = {
        "root": payload,
        "model": model,
        "profile": model["profile"],
    }
    container = containers[level]
    if change == "missing":
        container.pop(key)
    else:
        container[key] = "unexpected"
    if "revision" in payload:
        payload["revision"] = _canonical_revision(payload.get("models", []))  # type: ignore[arg-type]
    path = {"root": "root", "model": "models[0]", "profile": "models[0].profile"}[level]

    with pytest.raises(RuntimeError, match=re.escape(f"{path}.{key}")):
        _load_payload(payload, monkeypatch=monkeypatch, tmp_path=tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider", 1),
        ("provider", ""),
        ("provider", " OpenAI "),
        ("provider", "OpenAI"),
        ("model", 1),
        ("model", ""),
        ("model", " test-model "),
        ("base_url", 443),
        ("base_url", ""),
    ],
)
def test_catalog_rejects_noncanonical_identity_fields(
    field: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _valid_model()
    model[field] = value

    with pytest.raises(RuntimeError, match=rf"models\[0\]\.{field}"):
        _load_payload(_payload([model]), monkeypatch=monkeypatch, tmp_path=tmp_path)


@pytest.mark.parametrize(
    ("first_url", "second_url"),
    [
        ("HTTPS://API.EXAMPLE.TEST:443/v1/", "https://api.example.test/v1"),
        (None, None),
    ],
)
def test_catalog_rejects_duplicate_normalized_fingerprints_with_both_indices(
    first_url: str | None,
    second_url: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models = [_valid_model(base_url=first_url), _valid_model(base_url=second_url)]

    with pytest.raises(RuntimeError) as exc_info:
        _load_payload(_payload(models), monkeypatch=monkeypatch, tmp_path=tmp_path)

    message = str(exc_info.value)
    assert "models[0]" in message
    assert "models[1]" in message


@pytest.mark.parametrize(
    ("level", "needle", "replacement", "path"),
    [
        ("root", '"models":', '"models":[],"models":', "root.models"),
        (
            "model",
            '"provider":"openai"',
            '"provider":"openai","provider":"openai"',
            "models[0].provider",
        ),
        (
            "profile",
            '"supports_images":false',
            '"supports_images":false,"supports_images":false',
            "models[0].profile.supports_images",
        ),
    ],
)
def test_catalog_rejects_duplicate_json_members(
    level: str,
    needle: str,
    replacement: str,
    path: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del level
    text = json.dumps(_payload(), separators=(",", ":"))
    duplicated = text.replace(needle, replacement, 1)
    assert duplicated != text

    with pytest.raises(RuntimeError, match=re.escape(path)):
        _load_text(duplicated, monkeypatch=monkeypatch, tmp_path=tmp_path)


@pytest.mark.parametrize(
    ("revision", "message"),
    [
        ("2026-08-23", "form"),
        (f"sha256:{'A' * 64}", "form"),
        (f"sha256:{'0' * 64}", "does not match"),
    ],
)
def test_catalog_rejects_invalid_revision_form_and_digest(
    revision: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["revision"] = revision

    with pytest.raises(RuntimeError, match=rf"root\.revision.*{message}"):
        _load_payload(payload, monkeypatch=monkeypatch, tmp_path=tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("context_window_tokens", 0),
        ("max_input_tokens", 2048),
        ("max_output_tokens", 0),
    ],
)
def test_catalog_reports_model_profile_numeric_invariants_at_the_field(
    field: str,
    value: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _valid_model()
    model["profile"][field] = value  # type: ignore[index]

    with pytest.raises(RuntimeError, match=rf"models\[0\]\.profile\.{field}"):
        _load_payload(_payload([model]), monkeypatch=monkeypatch, tmp_path=tmp_path)


def test_catalog_accepts_complete_profiles_with_null_and_valid_http_endpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models = [
        _valid_model(model="default-endpoint", base_url=None),
        _valid_model(model="routed-endpoint", base_url="http://api.example.test:8080/v1"),
    ]
    payload = _payload(models)

    revision, parsed = _load_payload(payload, monkeypatch=monkeypatch, tmp_path=tmp_path)

    assert revision == _canonical_revision(models)
    assert isinstance(parsed, tuple)
    assert parsed[0].fingerprint == ModelFingerprint(
        provider="openai", model="default-endpoint", endpoint_fingerprint=None
    )
    assert parsed[0].profile == ModelProfile(
        context_window_tokens=1024,
        max_output_tokens=128,
        supports_images=False,
        reasoning=ReasoningProfile(
            format="openai",
            levels=ReasoningLevels(
                off="none",
                minimal="minimal",
                low="low",
                medium="medium",
                high="high",
                xhigh=None,
                max=None,
            ),
        ),
    )
    assert parsed[1].fingerprint == ModelFingerprint(
        provider="openai",
        model="routed-endpoint",
        endpoint_fingerprint=normalized_endpoint_fingerprint("http://api.example.test:8080/v1"),
    )
