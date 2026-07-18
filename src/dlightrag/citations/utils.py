"""Utility functions for citation processing."""

import re
from typing import Any

# LightRAG's multimodal chunk builder (lightrag.pipeline
# ._build_mm_chunks_from_sidecars._render) prefixes each multimodal chunk with
# bracketed labels — a documented, regression-pinned contract also consumed by
# lightrag.operate._MM_DISPLAY_NAME_PATTERN. Normalize them for human display:
# render equation LaTeX as math and drop the internal name/type slugs so only
# the natural-language description remains. If the contract ever changes the
# markers no longer match and the raw text passes through unchanged.
_EQUATION_NAME_RE = re.compile(r"^\[Equation Name\].*$", re.MULTILINE)
_MM_LABEL_LINE_RE = re.compile(r"^\[(?:Image Name|Image Type|Table Name)\].*$", re.MULTILINE)
_MATH_DELIMITER_RE = re.compile(r"\$|\\\(|\\\[")
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")


def split_source_ids(source_id: Any) -> list[str]:
    """Split comma-separated source_id string into list of chunk IDs."""
    if source_id is None:
        return []
    return [part.strip() for part in str(source_id).split(",") if part.strip()]


def _wrap_equation_math(content: str) -> str:
    """Render an equation chunk's LaTeX body as MathJax display math.

    LightRAG builds equation chunks with the raw LaTeX body first, then an
    ``[Equation Name]`` label line, then the description. The body carries no
    math delimiters, so MathJax cannot render it. Wrap the body in ``$$``
    (unless it already has delimiters) and drop the internal name label.
    """
    match = _EQUATION_NAME_RE.search(content)
    if match is None:
        return content
    body = content[: match.start()].strip()
    description = content[match.end() :].lstrip("\n")
    if body and not _MATH_DELIMITER_RE.search(body):
        body = f"$$\n{body}\n$$"
    return f"{body}\n\n{description}" if description else body


def filter_content_for_display(content: str, max_chars: int | None = None) -> str:
    """Normalize a chunk's LightRAG multimodal markers for human display.

    Renders equation LaTeX as math and strips the internal image/table
    name/type label lines, leaving the natural-language description. Non-
    multimodal text passes through unchanged.
    """
    content = _wrap_equation_math(content)
    content = _MM_LABEL_LINE_RE.sub("", content)
    result = _EXCESS_BLANK_LINES_RE.sub("\n\n", content).strip()

    if max_chars and len(result) > max_chars:
        result = result[:max_chars] + "..."

    return result
