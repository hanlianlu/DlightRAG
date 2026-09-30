#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Merge MinerU title-aided LLM config into ~/mineru.json.

Called by :file:`title_aided.sh`.  Reads any existing ``~/mineru.json``,
overwrites only ``llm-aided-config.title_aided``, and preserves all other
top-level keys (e.g. ``models-dir``, ``model-source``).

Enabling it stores the request fields that set the title model's reasoning as
``extra_body``, which ``sitecustomize`` sends with each title request. DlightRAG's
model catalogue owns every endpoint's reasoning dialect, so the enable path runs
under DlightRAG's environment; disabling needs only the standard library.
"""

import json
import os
import sys
from typing import Any


def _reasoning_fields(base_url: str, model: str) -> dict[str, Any]:
    """The fields that keep the title model from reasoning, as its endpoint reads them.

    Resolved the way DlightRAG resolves its own models: ``off`` when the model can
    turn reasoning off, else its cheapest verified level. An uncatalogued endpoint
    gets the protocol its address implies, as DlightRAG's own ``off`` would.
    """
    from dlightrag.engine.ai.catalog import resolve_model_profile
    from dlightrag.engine.ai.fingerprints import model_endpoint_fingerprint
    from dlightrag.engine.ai.reasoning import (
        cheapest_supported_reasoning,
        reasoning_request_kwargs,
        resolve_reasoning,
    )

    fingerprint = model_endpoint_fingerprint("openai", model, base_url)
    profile = resolve_model_profile(fingerprint).reasoning
    if profile is None:
        return {}
    level = "off" if profile.levels.off is not None else cheapest_supported_reasoning(profile)
    return reasoning_request_kwargs(resolve_reasoning(profile, level))


def main() -> None:
    disable = len(sys.argv) == 3 and sys.argv[2] == "--disable"
    if not disable and len(sys.argv) != 6:
        print(
            f"Usage: {sys.argv[0]} TARGET --disable | "
            "TARGET API_KEY BASE_URL MODEL ENABLE_THINKING",
            file=sys.stderr,
        )
        sys.exit(2)

    target = sys.argv[1]

    existing: dict = {}
    if os.path.isfile(target):
        try:
            with open(target) as fh:
                raw = fh.read().strip()
            if raw:
                existing = json.loads(raw)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"WARNING: could not parse {target} — starting fresh ({exc})")

    api_key = ""
    base_url = ""
    model = ""
    extra_body: dict[str, Any] = {}
    if disable:
        title_aided = {"enable": False}
    else:
        api_key = sys.argv[2]
        base_url = sys.argv[3]
        model = sys.argv[4]
        # Thinking on leaves reasoning to the provider's default.
        if sys.argv[5].lower() != "true":
            extra_body = _reasoning_fields(base_url, model)
        title_aided = {
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
            "extra_body": extra_body,
            "enable": True,
        }

    existing.setdefault("llm-aided-config", {})
    existing["llm-aided-config"]["title_aided"] = title_aided

    with open(target, "w") as fh:
        json.dump(existing, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    if disable:
        print(f"==> Disabled title-aided config in {target}")
        return

    # Never echo key material (even partially): report only presence + length.
    key_status = f"set ({len(api_key)} chars)" if api_key else "MISSING"
    print(f"==> Wrote {target}")
    print(f"    model    : {model}")
    print(f"    url      : {base_url}")
    print(f"    key      : {key_status}")
    print(f"    reasoning: {json.dumps(extra_body) if extra_body else 'provider default'}")


if __name__ == "__main__":
    main()
