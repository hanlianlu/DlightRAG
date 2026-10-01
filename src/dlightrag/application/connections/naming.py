# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The names a model calls Connection tools by.

A published tool is named ``mcp__<connection>__<tool>``: ``<connection>`` comes from the owner's
label for the Connection and ``<tool>`` from the server's own name for the tool, so the name says
whose tool it is and what its server calls it. A name holds only letters, digits, and ``_``,
starts with a letter, and fits 64 characters: the subset every supported provider accepts.

A short hash appears only where readable text alone would collide or not fit: in ``<tool>`` when
sanitizing merges two server names or the name is too long, and in ``<connection>`` when the label
keeps no letter or digit, or another of the owner's Connections already holds it.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence

from dlightrag.engine.answer.execution.connection_binding import CONNECTION_TOOL_PREFIX

from .models import CatalogueTool, ConnectionsError, RemoteTool

#: The longest tool name every supported provider accepts.
MAX_TOOL_NAME = 64
_SEPARATOR = "__"
#: The most of a label a name keeps; the server's tool name has the rest.
_LABEL_CHARS = 24
_HASH_CHARS = 6
_NOT_LABEL = re.compile(r"[^A-Za-z0-9]+")
_NOT_NAME = re.compile(r"[^A-Za-z0-9_]")


def name_catalogue(
    tools: Sequence[RemoteTool],
    *,
    label: str,
    connection_id: str,
    others: Iterable[str] = (),
) -> tuple[CatalogueTool, ...]:
    """Name one catalogue for publication.

    ``others`` are names from the latest catalogue each of the owner's other live Connections
    published, at least one from each. Their Connection parts are taken, so however the
    owner's Connections are bound together no two tools share a name, and no built-in tool's
    name starts with the Connection prefix. The same label, catalogue, and taken parts always
    give the same names.
    """
    taken = {part for name in others if (part := _connection_part_of(name)) is not None}
    prefix = CONNECTION_TOOL_PREFIX + _connection_part(label, connection_id, taken) + _SEPARATOR
    room = MAX_TOOL_NAME - len(prefix)
    sanitized = {tool.remote_name: _NOT_NAME.sub("_", tool.remote_name) for tool in tools}
    uses = Counter(sanitized.values())
    named: list[CatalogueTool] = []
    for tool in tools:
        text = sanitized[tool.remote_name]
        # A server name sanitizing leaves alone keeps its text. One that sanitizing merged into
        # another's, or that does not fit, keeps what fits and gains a hash of the server name.
        if len(text) > room or (uses[text] > 1 and text != tool.remote_name):
            text = f"{text[: room - _HASH_CHARS - 1]}_{_digest(tool.remote_name)}"
        named.append(
            CatalogueTool(
                tool.remote_name, tool.description, tool.input_schema, local_name=prefix + text
            )
        )
    if len({tool.local_name for tool in named}) != len(named):
        # Only names built to collide get here, and such a catalogue is rejected whole.
        raise ConnectionsError("catalogue")
    return tuple(named)


def _connection_part(label: str, connection_id: str, taken: set[str]) -> str:
    """The label's letters and digits, unless another Connection holds them."""
    ascii_label = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode()
    readable = _NOT_LABEL.sub("_", ascii_label).strip("_")[:_LABEL_CHARS].rstrip("_")
    head = readable[: _LABEL_CHARS - _HASH_CHARS - 1].rstrip("_")
    hashed = f"{head}_{_digest(connection_id)}" if head else _digest(connection_id)
    # A Connection id is longer than any label part and unique to its owner, so it is never taken.
    return next(part for part in (readable, hashed, connection_id) if part and part not in taken)


def _connection_part_of(name: str) -> str | None:
    """The Connection part of a published name, or None for a name of another shape."""
    if not name.startswith(CONNECTION_TOOL_PREFIX):
        return None
    part, separator, _ = name.removeprefix(CONNECTION_TOOL_PREFIX).partition(_SEPARATOR)
    return part if separator else None


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:_HASH_CHARS]
