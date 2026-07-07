#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Merge MinerU title-aided LLM config into ~/mineru.json.

Called by :file:`title_aided.sh`.  Reads any existing ``~/mineru.json``,
overwrites only ``llm-aided-config.title_aided``, and preserves all other
top-level keys (e.g. ``models-dir``, ``model-source``).
"""

import json
import os
import sys


def main() -> None:
    if len(sys.argv) != 6:
        print(
            f"Usage: {sys.argv[0]} TARGET API_KEY BASE_URL MODEL ENABLE_THINKING", file=sys.stderr
        )
        sys.exit(2)

    target = sys.argv[1]
    api_key = sys.argv[2]
    base_url = sys.argv[3]
    model = sys.argv[4]
    enable_thinking = sys.argv[5].lower() == "true"

    existing: dict = {}
    if os.path.isfile(target):
        try:
            with open(target) as fh:
                raw = fh.read().strip()
            if raw:
                existing = json.loads(raw)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"WARNING: could not parse {target} — starting fresh ({exc})")

    title_aided = {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "enable_thinking": enable_thinking,
        "enable": True,
    }

    existing.setdefault("llm-aided-config", {})
    existing["llm-aided-config"]["title_aided"] = title_aided

    with open(target, "w") as fh:
        json.dump(existing, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    # Never echo key material (even partially): report only presence + length.
    key_status = f"set ({len(api_key)} chars)" if api_key else "MISSING"
    print(f"==> Wrote {target}")
    print(f"    model : {model}")
    print(f"    url   : {base_url}")
    print(f"    key   : {key_status}")


if __name__ == "__main__":
    main()
