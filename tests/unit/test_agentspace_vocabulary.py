# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The corpus and the agent's files are two spaces, and each module names only its own.

ADR 0075 separated the reader's words — Workspace is the corpus scope, Agentspace is the
agent's files — and ADR 0076 renamed the agentspace language layer so a declaration alone tells
a reader which space it belongs to. That layer includes the Execution Environment's own
vocabulary (access, confinement, quota, text codec, inventory facts and rooted-file tools), whose
classes and functions name the Agentspace while its `WORKSPACE_*` constants keep the
configuration key's spelling. That split is easy to undo by accident: the next module that lands
beside the conversation snapshot code can spell its classes either way. This test is the rule the
ADR names: an agentspace-owned module declares no `Workspace*` name, a corpus-owned module
declares no `Agentspace*` name, and the three modules that used to say workspace are still named
agentspace.

It parses the sources rather than importing them, so it also sees a module whose imports no
longer resolve. Path classification is explicit and asserted below, so a module that moves out
of a classified package cannot silently drop out of the rule.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests.support.source_tree import python_files, syntax_tree

_SRC = Path(__file__).resolve().parents[2] / "src" / "dlightrag"

#: Modules whose declarations name the agent's durable file space (ADR 0076 Decision 1). The
#: Agent package is agent-side end to end — it may not import corpus code at all — so a
#: `Workspace*` declaration there would be the agentspace spelled either way, whichever
#: subpackage it landed in. Its configuration constants are `WORKSPACE_*`, not `Workspace*`,
#: and the prefix check leaves them to the configuration key they mirror.
_AGENTSPACE_PREFIXES = (
    "application/web_conversations/",
    "adapters/postgres/answer/",
    "adapters/postgres/web/web_conversations.py",
    "adapters/http/browser/conversation_models.py",
    "adapters/http/browser/routes/conversations.py",
    "engine/agent/",
    "engine/answer/agentspace.py",
    "engine/answer/execution_settings.py",
    "engine/runtime/agentspace.py",
)

#: Modules whose declarations name the corpus scope (ADR 0076 Decision 2).
_CORPUS_PREFIXES = (
    "application/access/",
    "application/corpus_admin/",
    "adapters/postgres/corpus/",
    "adapters/http/browser/workspace_models.py",
    "adapters/http/browser/routes/workspaces.py",
    "adapters/http/rest/",
    "adapters/mcp/",
    "engine/rag/",
)


def _modules() -> tuple[tuple[str, Path], ...]:
    """Every Python source under the package, as its package-relative path and file."""
    return tuple((path.relative_to(_SRC).as_posix(), path) for path in python_files(_SRC))


def _owns(relative_path: str, prefixes: tuple[str, ...]) -> bool:
    """Whether a module path is the one a configured prefix names."""
    return any(relative_path == prefix or relative_path.startswith(prefix) for prefix in prefixes)


def _declared_names(tree: ast.Module) -> set[str]:
    """Every declared class, function, method, type alias and module-level constant name."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.TypeAlias):
            names.add(node.name.id)
    for node in tree.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else ([node.target] if isinstance(node, ast.AnnAssign) else [])
        )
        for target in targets:
            # A lower-cased module constant is a value, not a declaration; the type aliases
            # ADR 0076 renames (``WorkspaceChange = Literal[...]``) start upper-cased.
            if isinstance(target, ast.Name) and target.id[:1].isupper():
                names.add(target.id)
    return names


def _offenders(prefixes: tuple[str, ...], forbidden_prefix: str) -> list[str]:
    """The ``module:name`` declarations in the classified modules that say the other word."""
    offenders: list[str] = []
    for relative_path, path in _modules():
        if not _owns(relative_path, prefixes):
            continue
        for name in sorted(_declared_names(syntax_tree(path))):
            if name.startswith(forbidden_prefix):
                offenders.append(f"{relative_path}:{name}")
    return offenders


def test_agentspace_modules_declare_no_workspace_name() -> None:
    assert _offenders(_AGENTSPACE_PREFIXES, "Workspace") == []


def test_corpus_modules_declare_no_agentspace_name() -> None:
    assert _offenders(_CORPUS_PREFIXES, "Agentspace") == []


def test_every_classified_prefix_names_at_least_one_module() -> None:
    relative_paths = tuple(relative_path for relative_path, _path in _modules())
    for prefix in (*_AGENTSPACE_PREFIXES, *_CORPUS_PREFIXES):
        assert any(_owns(relative_path, (prefix,)) for relative_path in relative_paths), (
            f"{prefix!r} names no module; a moved file would escape the vocabulary rule"
        )


def test_no_module_is_classified_as_both_spaces() -> None:
    assert not [
        relative_path
        for relative_path, _path in _modules()
        if _owns(relative_path, _AGENTSPACE_PREFIXES) and _owns(relative_path, _CORPUS_PREFIXES)
    ]
