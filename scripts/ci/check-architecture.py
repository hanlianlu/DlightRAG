#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Architecture boundary check — core/internal packages must not import from transport packages."""

import ast
import sys
from pathlib import Path

# Packages that must NOT import from TRANSPORT_PACKAGES
INTERNAL_PACKAGES = [
    "dlightrag.config",
    "dlightrag.core",
    "dlightrag.storage",
    "dlightrag.models",
    "dlightrag.utils",
    "dlightrag.sourcing",
    "dlightrag.observability",
    "dlightrag.prompts",
    "dlightrag.citations",
]

# Transport/interface packages — internal code must not depend on these
TRANSPORT_PACKAGES = [
    "dlightrag.web",
    "dlightrag.api",
    "dlightrag.mcp",
]

# Check: dlightrag.core,storage,models,utils,sourcing,etc → must NOT import web/api/mcp
# web, api, mcp can import from anything (they're at the top of the dependency chain)


def _extract_imports(filepath: Path) -> list[tuple[int, str]]:
    """Extract (lineno, module_name) for all imports from dlightrag.*."""
    tree = ast.parse(filepath.read_text(), filename=str(filepath))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.lineno, node.module))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name))
    return imports


def _is_dlightrag_import(module: str) -> bool:
    return module == "dlightrag" or module.startswith("dlightrag.")


def check(src_root: Path) -> int:
    violations: list[tuple[str, int, str]] = []  # (file, lineno, import)

    for pkg in INTERNAL_PACKAGES:
        pkg_path = src_root / pkg.replace(".", "/")
        if pkg_path.is_dir():
            py_files = list(pkg_path.rglob("*.py"))
        elif pkg_path.with_suffix(".py").is_file():
            py_files = [pkg_path.with_suffix(".py")]
        else:
            continue

        for py_file in py_files:
            if py_file.name.startswith("__"):
                continue  # __init__.py imports are allowed (deferred/conditional)
            rel = py_file.relative_to(src_root)
            for lineno, module in _extract_imports(py_file):
                if not _is_dlightrag_import(module):
                    continue
                for transport in TRANSPORT_PACKAGES:
                    if module == transport or module.startswith(transport + "."):
                        violations.append((str(rel), lineno, module))

    if violations:
        print(f"Architecture violations ({len(violations)}):")
        for file, lineno, imp in sorted(violations):
            print(f"  {file}:{lineno} → imports {imp}")
        print("\nInternal/core packages must not import from transport packages (web/api/mcp).")
        return 1

    print("Architecture check passed — no layer violations.")
    return 0


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    sys.exit(check(src_root))
