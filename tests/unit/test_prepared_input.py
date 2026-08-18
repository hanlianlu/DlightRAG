# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the bounded prepared input and its canonical serialization."""

import pytest

from dlightrag.answer.prepared_input import (
    MAX_PREPARED_INPUT_BYTES,
    PreparedAnswerInput,
    PreparedInputTooLargeError,
    encode_prepared_input,
    minimal_prepared_input,
    prepared_input_bytes,
    prepared_input_with_payload,
)
from dlightrag.runtime.records import answer_run_request_fingerprint


def test_exactly_8mib_passes_and_one_byte_over_fails() -> None:
    base = minimal_prepared_input(query="what is the state?")
    base_size = len(prepared_input_bytes(base))
    # Measure the fixed overhead of appending one padding entry.
    trial = prepared_input_with_payload(base, extra_bytes=0)
    overhead = len(prepared_input_bytes(trial)) - base_size

    exact = prepared_input_with_payload(
        base, extra_bytes=MAX_PREPARED_INPUT_BYTES - base_size - overhead
    )
    assert len(encode_prepared_input(exact)) == MAX_PREPARED_INPUT_BYTES

    over = prepared_input_with_payload(
        base, extra_bytes=MAX_PREPARED_INPUT_BYTES - base_size - overhead + 1
    )
    with pytest.raises(PreparedInputTooLargeError):
        encode_prepared_input(over)


def test_fingerprint_is_unchanged_by_prepared_profile_changes() -> None:
    public_request = {"query": "q", "workspaces": ["default"]}
    fingerprint = answer_run_request_fingerprint(public_request)

    plain = PreparedAnswerInput(
        session_id=minimal_prepared_input(query="q").session_id,
        fingerprint=fingerprint,
        query="q",
        workspaces=("default",),
    )
    enriched = PreparedAnswerInput(
        session_id=plain.session_id,
        fingerprint=fingerprint,
        query="q",
        workspaces=("default",),
        profile_facts=(
            {"model": "gpt-4.1-mini", "window": 120_000},
            {"adapter": "openai"},
        ),
        history=({"role": "user", "content": "earlier"},),
    )

    # Enrichment changes the prepared payload but never the stored fingerprint.
    assert enriched.fingerprint == plain.fingerprint == fingerprint
    assert plain.canonical_json() != enriched.canonical_json()


def test_canonical_json_is_stable_across_dict_ordering() -> None:
    first = PreparedAnswerInput(
        session_id=minimal_prepared_input(query="q").session_id,
        fingerprint="a" * 64,
        query="q",
        workspaces=("default",),
        profile_facts=({"a": 1, "b": 2},),
    )
    second = PreparedAnswerInput(
        session_id=first.session_id,
        fingerprint="a" * 64,
        query="q",
        workspaces=("default",),
        profile_facts=({"b": 2, "a": 1},),
    )
    assert first.canonical_json() == second.canonical_json()


def test_session_id_must_be_canonical_uuid() -> None:
    with pytest.raises(ValueError):
        PreparedAnswerInput(
            session_id="not-a-uuid",
            fingerprint="a" * 64,
            query="q",
            workspaces=("default",),
        )


def test_query_workspaces_and_fingerprint_are_required_and_validated() -> None:
    session_id = minimal_prepared_input(query="q").session_id
    with pytest.raises(ValueError):
        PreparedAnswerInput(
            session_id=session_id, fingerprint="a" * 64, query="  ", workspaces=("default",)
        )
    with pytest.raises(ValueError):
        PreparedAnswerInput(session_id=session_id, fingerprint="a" * 64, query="q", workspaces=())
    with pytest.raises(ValueError):
        PreparedAnswerInput(
            session_id=session_id, fingerprint="short", query="q", workspaces=("default",)
        )
