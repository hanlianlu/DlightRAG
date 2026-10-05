# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The golden examples of the built-in interactive-html Skill.

A model reads them through ``load_skill`` and the report tests build them, from the same files, so
what the Skill shows is what is tested.
"""

from pathlib import Path

DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "src/dlightrag/engine/agent/builtin_skills/interactive-html/references"
)


def examples() -> dict[str, Path]:
    """Every example by short name: ``example-brief.html`` is ``brief``."""
    return {
        path.name.removeprefix("example-").removesuffix(".html"): path
        for path in sorted(DIRECTORY.glob("example-*.html"))
    }
