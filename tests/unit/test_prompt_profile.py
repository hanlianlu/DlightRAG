# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for centralized prompt profile assembly."""

from __future__ import annotations

import json

import dlightrag.prompts as prompts
import dlightrag.prompts.identity as identity
from dlightrag.prompts import (
    ANSWER_CORE,
    HIGHLIGHT_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    VISUAL_RERANK_PROMPT,
    get_answer_system_prompt,
)
from dlightrag.prompts.guidance import (
    ANSWER_CITATION_EXAMPLE,
    ANSWER_CONTEXT_GUIDANCE,
    CITATION_GUIDANCE,
    HIGHLIGHT_GUIDANCE,
    HIGHLIGHT_RESPONSE_FORMAT,
    LISTWISE_RERANK_PROMPT,
    PLANNER_GUIDANCE,
    RERANK_GUIDANCE,
)
from dlightrag.prompts.identity import CORE_IDENTITY


def test_prompt_profile_exposes_single_core_identity() -> None:
    identity_names = [name for name in dir(identity) if name.endswith("IDENTITY")]
    assert identity_names == ["CORE_IDENTITY"]
    assert CORE_IDENTITY.startswith("You are ")


def test_answer_prompt_is_assembled_from_core_identity_and_guidance() -> None:
    prompt = get_answer_system_prompt()

    assert prompt == ANSWER_CORE
    assert CORE_IDENTITY in prompt
    assert ANSWER_CONTEXT_GUIDANCE in prompt
    assert CITATION_GUIDANCE in prompt
    assert ANSWER_CITATION_EXAMPLE in prompt


def test_planner_prompt_uses_core_identity_and_planner_guidance() -> None:
    assert CORE_IDENTITY in PLANNER_SYSTEM_PROMPT
    assert PLANNER_GUIDANCE in PLANNER_SYSTEM_PROMPT
    assert "{schema_section}" in PLANNER_SYSTEM_PROMPT
    assert "{history_section}" in PLANNER_SYSTEM_PROMPT
    assert "filter_evidence" in PLANNER_SYSTEM_PROMPT


def test_rag_side_prompts_are_assembled_from_guidance() -> None:
    assert RERANK_GUIDANCE in VISUAL_RERANK_PROMPT
    assert CORE_IDENTITY in HIGHLIGHT_SYSTEM_PROMPT
    assert HIGHLIGHT_GUIDANCE in HIGHLIGHT_SYSTEM_PROMPT


def test_highlight_system_prompt_uses_literal_json_braces() -> None:
    assert "{{" not in HIGHLIGHT_SYSTEM_PROMPT
    assert "}}" not in HIGHLIGHT_SYSTEM_PROMPT


def test_highlight_response_format_contains_parseable_json_example() -> None:
    prefix = "Return JSON only, for example: "
    example = HIGHLIGHT_RESPONSE_FORMAT.removeprefix(prefix).split(
        ". Confidence must",
        1,
    )[0]

    parsed = json.loads(example)

    assert parsed == {"phrases": ["phrase1", "phrase2"], "confidence": 0.8}
    assert "0.0-1.0" not in HIGHLIGHT_SYSTEM_PROMPT


def test_exported_guidance_constants_do_not_declare_identity() -> None:
    guidance_names = [
        "ANSWER_CONTEXT_GUIDANCE",
        "CITATION_GUIDANCE",
        "ANSWER_CITATION_EXAMPLE",
        "PLANNER_GUIDANCE",
        "RERANK_GUIDANCE",
        "VISUAL_RERANK_PROMPT_TEMPLATE",
        "LISTWISE_RERANK_PROMPT",
        "HIGHLIGHT_GUIDANCE",
        "HIGHLIGHT_RESPONSE_FORMAT",
        "HIGHLIGHT_USER_PROMPT",
    ]

    for name in guidance_names:
        assert name in prompts.__all__
        assert "You are " not in getattr(prompts, name)
    assert "You are " not in LISTWISE_RERANK_PROMPT
