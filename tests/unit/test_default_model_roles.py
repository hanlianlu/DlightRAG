# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Regression coverage for checked-in role model defaults."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_extract_and_keyword_default_to_deepseek_non_thinking() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

    for role_name in ("extract", "keyword"):
        role = config["llm"]["roles"][role_name]
        assert role["provider"] == "openai"
        assert role["model"] == "deepseek-v4-flash"
        assert role["base_url"] == "https://api.deepseek.com"
        assert role["model_kwargs"] == {"thinking": {"type": "disabled"}}


def test_query_and_vlm_still_fall_back_to_multimodal_default() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

    assert "query" not in config["llm"]["roles"]
    assert "vlm" not in config["llm"]["roles"]
    assert config["llm"]["default"]["model"] == "google/gemini-3.5-flash"
