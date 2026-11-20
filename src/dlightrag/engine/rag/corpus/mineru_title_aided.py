# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Resolve the title model's reasoning dialect and write MinerU's title-aided config.

`dlightrag-mineru` owns `~/mineru.json`: it merges the section, preserves every
other top-level key, and writes the file atomically at mode 0600. What only DlightRAG
knows is which request fields *this deployment's* endpoint reads to turn reasoning off
-- that is a property of the model catalogue -- so the knowledge stays here and is
injected into the package's command as a resolver (ADR 0054).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dlightrag_mineru import title_aided

from dlightrag.engine.ai.catalog import resolve_model_profile
from dlightrag.engine.ai.fingerprints import model_endpoint_fingerprint
from dlightrag.engine.ai.reasoning import (
    cheapest_supported_reasoning,
    reasoning_request_kwargs,
    resolve_reasoning,
)

SUMMARY = "Write MinerU's title-aided LLM config from DlightRAG's model catalogue"


def reasoning_fields(base_url: str, model: str) -> dict[str, Any]:
    """The fields that keep the title model from reasoning, as its endpoint reads them.

    Resolved the way DlightRAG resolves its own models: ``off`` when the model can
    turn reasoning off, else its cheapest verified level. An uncatalogued endpoint gets
    the protocol its address implies, as DlightRAG's own ``off`` would.
    """
    fingerprint = model_endpoint_fingerprint("openai", model, base_url)
    profile = resolve_model_profile(fingerprint).reasoning
    if profile is None:
        return {}
    level = "off" if profile.levels.off is not None else cheapest_supported_reasoning(profile)
    return reasoning_request_kwargs(resolve_reasoning(profile, level))


def parser() -> argparse.ArgumentParser:
    """The command line for `dlightrag-mineru-title-aided`."""
    command = argparse.ArgumentParser(
        prog="dlightrag-mineru-title-aided", description=SUMMARY, suggest_on_error=True
    )
    command.add_argument("--disable", action="store_true", help="Turn title correction off")
    command.add_argument(
        "--target", type=Path, default=None, help="The config to merge into (default ~/mineru.json)"
    )
    return command


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for `dlightrag-mineru-title-aided`."""
    arguments = parser().parse_args(argv)
    return title_aided.from_environment(
        target=arguments.target,
        disable=arguments.disable,
        resolver=reasoning_fields,
    )


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    sys.exit(main())
