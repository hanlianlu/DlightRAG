#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Generate deterministic Pygments CSS for the web frontend.

Prints the full contents of src/dlightrag/web/static/pygments.css to stdout.
"""

from __future__ import annotations

import re

from pygments.formatters import HtmlFormatter

TARGET_PATH = "src/dlightrag/web/static/pygments.css"
REGEN_COMMAND = f"uv run scripts/generate_pygments_css.py > {TARGET_PATH}"

STYLE_VARIANTS: tuple[tuple[str, str, str], ...] = (
    ("light", "Friendly", "friendly"),
    ("dark", "GitHub Dark", "github-dark"),
)

# Only strip background paint so theme container remains authoritative.
BACKGROUND_DECL_RE = re.compile(r"\s*background(?:-color)?\s*:\s*[^;{}]+;?")
EMPTY_RULE_RE = re.compile(r"\{\s*\}(?:\s*/\*.*\*/)?\s*$")


def _strip_background_declarations(rule: str) -> str:
    return BACKGROUND_DECL_RE.sub("", rule)


def _iter_style_rules(mode: str, pygments_style: str) -> list[str]:
    root = f'[data-color-mode="{mode}"] .highlight'
    formatter = HtmlFormatter(style=pygments_style)
    rules: list[str] = []
    for line in formatter.get_style_defs(root).splitlines():
        if not line.startswith(root):
            continue
        cleaned = _strip_background_declarations(line)
        if EMPTY_RULE_RE.search(cleaned):
            continue
        rules.append(cleaned.rstrip())
    return rules


def generate_css() -> str:
    lines: list[str] = [
        "/* Generated file. Do not edit by hand. */",
        f"/* Regenerate with: {REGEN_COMMAND} */",
        "/* Source styles: Pygments HtmlFormatter Friendly (light), GitHub Dark (dark). */",
        "/* Background and background-color declarations are intentionally removed. */",
    ]

    for mode, section_name, pygments_style in STYLE_VARIANTS:
        lines.append("")
        lines.append(f"/* {section_name} */")
        lines.extend(_iter_style_rules(mode=mode, pygments_style=pygments_style))

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    print(generate_css(), end="")


if __name__ == "__main__":
    main()
