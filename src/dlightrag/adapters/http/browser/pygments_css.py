# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The code-highlight stylesheet, built from the Pygments that renders the markup.

One light and one dark palette, each scoped to its colour mode. Generating it here keeps the
class names in the answers and the rules that colour them on one Pygments version.
"""

import hashlib
import re
from functools import cache

from pygments.formatters import HtmlFormatter

# (colour mode, comment label, Pygments style)
_STYLES = (
    ("light", "Xcode", "xcode"),
    ("dark", "GitHub Dark", "github-dark"),
)
# Two upstream foregrounds fall below contrast on the surfaces they sit on.
_FOREGROUND_FIXES = {
    "light": {"#836C28": "#7D6622"},
    "dark": {"#6E7681": "#858D98"},
}
# Only background paint is stripped: the theme's code container owns the background.
_BACKGROUND = re.compile(r"\s*background(?:-color)?\s*:\s*[^;{}]+;?")
_EMPTY_RULE = re.compile(r"\{\s*\}(?:\s*/\*.*\*/)?\s*$")


def _rules(mode: str, style: str) -> list[str]:
    root = f'[data-color-mode="{mode}"] .highlight'
    rules: list[str] = []
    for line in HtmlFormatter(style=style).get_style_defs(root).splitlines():
        if not line.startswith(root):
            continue
        rule = _BACKGROUND.sub("", line)
        if _EMPTY_RULE.search(rule):
            continue
        for original, replacement in _FOREGROUND_FIXES[mode].items():
            rule = rule.replace(f"color: {original}", f"color: {replacement}")
        rules.append(rule.rstrip())
    return rules


@cache
def pygments_css() -> tuple[bytes, str]:
    """The stylesheet and its ETag; built once per process."""
    lines: list[str] = []
    for mode, label, style in _STYLES:
        lines += [f"/* {label} */", *_rules(mode, style), ""]
    body = "\n".join(lines).encode()
    return body, f'"{hashlib.sha256(body).hexdigest()[:16]}"'
