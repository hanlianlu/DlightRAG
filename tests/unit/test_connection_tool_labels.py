# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Run tool labels: the display-only read a browser edge uses for Connection tools.

The interface under test is one call returning one mapping. Everything it hides
-- pin resolution, owner scoping, whitespace, bounding, an ineligible reader --
is exercised through that call, so the test never reaches past the seam.
"""

from typing import Any, cast

import pytest

from dlightrag.application.connections import Connections
from dlightrag.application.connections.models import PinnedToolFact


class _Store:
    def __init__(self, facts: tuple[PinnedToolFact, ...] = ()) -> None:
        self.facts = facts
        self.reads: list[tuple[str, str]] = []

    async def pinned_tool_facts(self, *, owner_id: str, run_id: str) -> tuple[PinnedToolFact, ...]:
        self.reads.append((owner_id, run_id))
        return self.facts


def _connections(facts: tuple[PinnedToolFact, ...] = ()) -> tuple[Connections, _Store]:
    store = _Store(facts)
    return Connections(store=cast(Any, store), mcp=cast(Any, None)), store


@pytest.mark.asyncio
async def test_label_names_the_owner_connection_and_the_remote_tool() -> None:
    connections, _ = _connections(
        (
            PinnedToolFact("mcp_c1_hash", "Personal tools", "search_issues"),
            PinnedToolFact("mcp_c2_hash", "Work notes", "create_page"),
        )
    )

    labels = await connections.pinned_tool_labels(
        owner_id="owner-1", auth_mode="jwt", run_id="run-1"
    )

    assert labels == {
        "mcp_c1_hash": "Personal tools · search_issues",
        "mcp_c2_hash": "Work notes · create_page",
    }


@pytest.mark.asyncio
async def test_label_survives_one_missing_half_and_collapses_whitespace() -> None:
    """A deleted head or a nameless remote tool still yields a readable line."""

    connections, _ = _connections(
        (
            PinnedToolFact("mcp_a_hash", "  Multi\nline   label ", "list"),
            PinnedToolFact("mcp_b_hash", "", "ping"),
            PinnedToolFact("mcp_c_hash", "Notes", ""),
            PinnedToolFact("mcp_d_hash", "", ""),
            PinnedToolFact("mcp_e_hash", "x" * 200, "y" * 200),
        )
    )

    labels = await connections.pinned_tool_labels(
        owner_id="owner-1", auth_mode="none", run_id="run-1"
    )

    assert labels["mcp_a_hash"] == "Multi line label · list"
    assert labels["mcp_b_hash"] == "ping"
    assert labels["mcp_c_hash"] == "Notes"
    assert "mcp_d_hash" not in labels
    assert len(labels["mcp_e_hash"]) == 96


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("owner_id", "auth_mode"),
    [("owner-1", "simple"), ("", "jwt")],
)
async def test_an_ineligible_reader_labels_nothing_and_reads_nothing(
    owner_id: str, auth_mode: str
) -> None:
    connections, store = _connections((PinnedToolFact("mcp_c1_hash", "Label", "tool"),))

    labels = await connections.pinned_tool_labels(
        owner_id=owner_id, auth_mode=auth_mode, run_id="run-1"
    )

    assert labels == {}
    assert store.reads == []
