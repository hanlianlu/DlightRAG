# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The drawings a reader sees the deployment and the layering by.

Both are checked-in SVG with no build step, and nothing else shows a reader those
shapes — the README leads with the deployment and Architecture opens Runtime Ownership
with the layering. A node nobody drew is invisible to that reader, and no test can say
whether a drawing is *good*. What these tests hold is reachability: the names the code and
the deployment actually have must reach their diagram, and the scaffolding a drawing needs
to announce itself to a reader who cannot see it must be whole. Wording, position, arrows
and emphasis stay an author's judgment.

There is deliberately no exemption list. A new Compose service, distribution, engine owner
or adapter package fails here, and the fix is a node in the diagram or a decision that the
thing is not part of the picture — never a line added to this file.
"""

from __future__ import annotations

import re
import tomllib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
_DIAGRAMS = _REPO / "docs" / "diagrams"
_SVG = "{http://www.w3.org/2000/svg}"

#: A pool member is one of several: `agent-browser-2` is drawn as the pool it belongs to.
_INSTANCE = re.compile(r"[-_]\d+$")

#: Below this length a squashed name is not evidence of anything: `ai` and `rag` sit inside
#: longer words, so those are matched against a drawn word instead of against the whole text.
_WHOLE_TEXT_MINIMUM = 4


def _squashed(text: str) -> str:
    """One comparable form: lowercase, without the punctuation a drawing is free to choose."""
    return "".join(character for character in text.lower() if character.isalnum())


@dataclass(frozen=True, slots=True)
class _Drawing:
    """What one diagram names, as the text it draws and the words that text is made of."""

    squashed: str
    words: frozenset[str]

    def names(self, name: str) -> bool:
        """Whether the diagram names this thing.

        A long name is found in the drawing's whole text, so the box that says
        `PostgreSQL 18` names the service `postgres`. A short one is found as a word,
        so renaming the `AI` box cannot pass on the `ai` inside another word.
        """
        squashed = _squashed(name)
        if len(squashed) >= _WHOLE_TEXT_MINIMUM:
            return squashed in self.squashed
        return any(word == squashed or word.startswith(squashed) for word in self.words)


def _drawing(file_name: str) -> _Drawing:
    """Read one diagram's drawn text, leaving its stylesheet out of the comparison."""
    root = ET.parse(_DIAGRAMS / file_name).getroot()
    text = " ".join(
        " ".join(element.itertext()) for element in root.iter() if element.tag != f"{_SVG}style"
    )
    words = frozenset(word for word in (_squashed(part) for part in re.split(r"\W+", text)) if word)
    return _Drawing(squashed=_squashed(text), words=words)


def _compose_services() -> tuple[str, ...]:
    """Every service the shipped Compose stack can start, profile-gated ones included."""
    stack = yaml.safe_load((_REPO / "docker-compose.yml").read_text(encoding="utf-8"))
    return tuple(sorted({_INSTANCE.sub("", name) for name in stack["services"]}))


def _distributions() -> tuple[str, ...]:
    """The root wheel and every independently installable member of the UV workspace."""
    root = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    members = (root["project"]["name"],)
    return tuple(
        sorted(
            {
                *members,
                *(
                    tomllib.loads((_REPO / member / "pyproject.toml").read_text(encoding="utf-8"))[
                        "project"
                    ]["name"]
                    for member in root["tool"]["uv"]["workspace"]["members"]
                ),
            }
        )
    )


def _packages(directory: Path) -> tuple[str, ...]:
    """Every Python package directly under one source directory, caches aside."""
    return tuple(
        sorted(
            entry.name
            for entry in directory.iterdir()
            if entry.is_dir() and not entry.name.startswith("_")
        )
    )


def test_every_compose_service_reaches_the_system_diagram() -> None:
    """The stack a deployment starts is what the system diagram shows it starting."""
    drawing = _drawing("system.svg")
    assert [name for name in _compose_services() if not drawing.names(name)] == []


def test_every_distribution_reaches_the_layering_diagram() -> None:
    """Every distribution is drawn, including the host tool no wheel installs."""
    drawing = _drawing("architecture.svg")
    assert [name for name in _distributions() if not drawing.names(name)] == []


def test_every_engine_owner_reaches_the_layering_diagram() -> None:
    """The sibling owners inside Engine are the diagram's subject."""
    drawing = _drawing("architecture.svg")
    owners = _packages(_REPO / "src" / "dlightrag" / "engine")
    assert [name for name in owners if not drawing.names(name)] == []


def test_every_adapter_package_reaches_the_layering_diagram() -> None:
    """An adapter package nobody draws is a boundary a reader cannot see."""
    drawing = _drawing("architecture.svg")
    packages = _packages(_REPO / "src" / "dlightrag" / "adapters")
    assert [name for name in packages if not drawing.names(name)] == []
