# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One read and one parse of each source file, shared by every suite that scans the tree.

The scanning tests read the same few hundred modules; caching here makes the whole suite pay for
each file once. Trees are shared, so a test only reads them.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@functools.cache
def python_files(*roots: Path) -> tuple[Path, ...]:
    """Every Python file under the given directories, in path order."""
    return tuple(
        sorted(
            path for root in roots for path in root.rglob("*.py") if "__pycache__" not in path.parts
        )
    )


@functools.cache
def source_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@functools.cache
def syntax_tree(path: Path) -> ast.Module:
    return ast.parse(source_text(path), filename=str(path))
