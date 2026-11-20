# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Pinned Connection surface, Settings declaration cost, and gated resource reads."""

from dataclasses import replace
from typing import Any, cast

import pytest

from dlightrag.application.connections.models import (
    BoundConnectionSurface,
    CatalogueTool,
    ConnectionsError,
    DispatchCredentials,
    PinnedCatalogue,
    StoredConnection,
)
from dlightrag.application.connections.policy import ConnectionPolicy
from dlightrag.application.connections.service import Connections
from dlightrag.engine.agent.tools.contracts import ToolResult, declaration_tokens
from dlightrag.engine.ai.messages import ToolDefinition
from dlightrag.engine.answer.execution.connection_binding import (
    ResearchToolClaim,
    RunConnectionBinding,
)
from dlightrag.engine.credential_cipher import CredentialCipher
from tests.support.tool_helpers import tool_runtime

_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}
_CATALOGUE = (
    CatalogueTool("search", "Search pages.", _SCHEMA, local_name="mcp__Notion__search"),
    CatalogueTool("create", "Create a page.", _SCHEMA, local_name="mcp__Notion__create"),
)
_BINDING = RunConnectionBinding("owner-1", "a" * 32, 1, 1, "b" * 64)


class _CatalogueStore:
    def __init__(self, rows: tuple[Any, ...] = ()) -> None:
        self.rows = rows

    async def research_catalogues(self, owner_id: str) -> tuple[Any, ...]:
        assert owner_id == "owner-1"
        return self.rows


class _Resources:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_resources(self, **kwargs: Any) -> ToolResult:
        self.calls.append(("list_resources", kwargs))
        return ToolResult.text("notion://docs/a — Page (text/markdown)")

    async def list_resource_templates(self, **kwargs: Any) -> ToolResult:
        self.calls.append(("list_resource_templates", kwargs))
        return ToolResult.text("No MCP resource templates.")

    async def read_resource(self, **kwargs: Any) -> ToolResult:
        self.calls.append(("read_resource", kwargs))
        return ToolResult.text("page body")


def _connections(store: Any, *, resources: _Resources | None = None) -> Connections:
    return Connections(
        store=cast(Any, store),
        mcp=cast(Any, None),
        resources=cast(Any, resources),
        cipher=CredentialCipher(None),
        policy=ConnectionPolicy(),
    )


async def test_binding_pins_a_secret_free_surface_with_server_instructions() -> None:
    store = _CatalogueStore((PinnedCatalogue(_BINDING, _CATALOGUE, "Notion", "Use page bodies."),))
    bindings = await _connections(store, resources=_Resources()).bind_research(owner_id="owner-1")

    assert bindings.bindings == (_BINDING,)
    assert bindings.surface == (
        BoundConnectionSurface(
            connection_id="a" * 32,
            token="Notion",
            label="Notion",
            instructions="Use page bodies.",
        ),
    )

    # The pinned surface carries the instructions whether or not the deployment also
    # serves the resource transport; the shipped composition always serves both.
    no_resources = await _connections(store).bind_research(owner_id="owner-1")
    assert no_resources.surface == bindings.surface


async def test_duplicate_labels_stay_unambiguous_and_bounded() -> None:
    long_label = "A" * 120
    rows = (
        PinnedCatalogue(_BINDING, _CATALOGUE, long_label),
        PinnedCatalogue(
            RunConnectionBinding("owner-1", "c" * 32, 2, 1, "d" * 64), _CATALOGUE, long_label
        ),
    )
    bindings = await _connections(_CatalogueStore(rows), resources=_Resources()).bind_research(
        owner_id="owner-1"
    )

    tokens = [entry.token for entry in bindings.surface]
    assert len(tokens) == 2
    assert tokens[0] != tokens[1]
    assert all(len(token) <= 80 for token in tokens)
    assert tokens[0].endswith("(" + "a" * 8 + ")")
    assert tokens[1].endswith("(" + "c" * 8 + ")")


def _stored(*, published: bool) -> StoredConnection:
    return StoredConnection(
        owner_id="owner-1",
        connection_id="a" * 32,
        revision=1,
        label="Notion",
        endpoint="https://mcp.example.com/rpc",
        enabled=True,
        activation_epoch=1,
        generation=1,
        grant_id=None,
        status="ready",
        catalogue=_CATALOGUE,
        catalogue_created_at="2026-10-22T09:00:00+00:00" if published else None,
        last_error_kind=None,
        authentication="none",
    )


class _ReadStore:
    def __init__(self, item: StoredConnection) -> None:
        self.item = item

    async def read(self, owner_id: str) -> tuple[str, tuple[StoredConnection, ...]]:
        assert owner_id == "owner-1"
        return "rev-1", (self.item,)


async def test_settings_projection_names_the_declaration_cost_of_a_published_catalogue() -> None:
    view = await _connections(_ReadStore(_stored(published=True))).read(owner_id="owner-1")
    connection = view.connections[0]

    assert connection.tool_count == 2
    expected = declaration_tokens(
        ToolDefinition(tool.local_name, tool.description, tool.input_schema) for tool in _CATALOGUE
    )
    assert connection.declaration_tokens == expected > 0

    # The displayed cost is the same array a pinned Run accounts for, not a second estimate.
    from dlightrag.engine.agent.session.plan import AgentRunPlan

    catalogues = _CatalogueStore((PinnedCatalogue(_BINDING, _CATALOGUE, "Notion"),))
    bound = await _connections(catalogues, resources=_Resources()).bind_research(owner_id="owner-1")
    plan = AgentRunPlan.from_tools(
        list(bound.tools), model_role="query", context_policy_revision="test-policy"
    )
    assert connection.declaration_tokens == plan.tool_schema_tokens

    unpublished = (
        await _connections(_ReadStore(_stored(published=False))).read(owner_id="owner-1")
    ).connections[0]
    assert (unpublished.tool_count, unpublished.declaration_tokens) == (None, None)

    empty = _stored(published=True)
    empty = replace(empty, catalogue=())
    empty_view = (await _connections(_ReadStore(empty)).read(owner_id="owner-1")).connections[0]
    assert (empty_view.tool_count, empty_view.declaration_tokens) == (0, 0)


class _GateStore:
    async def resource_gate(
        self, *, claim: ResearchToolClaim, runtime: Any, connection_id: str
    ) -> DispatchCredentials:
        assert connection_id == "a" * 32
        return DispatchCredentials(
            connection_id=connection_id,
            generation=1,
            activation_epoch=1,
            endpoint="https://mcp.example.com/rpc",
            grant_id=None,
            secret_version=0,
            envelope=None,
            authentication="none",
            endpoint_headers=(("X-Flavor", "pages"),),
        )


async def test_resource_reads_forward_to_the_transport_only_after_the_gate() -> None:
    resources = _Resources()
    connections = _connections(_GateStore(), resources=resources)
    claim = ResearchToolClaim("owner-1", "run-1", "worker-1", 1, _cancelled_nothing)
    runtime = tool_runtime(tool_name="mcp_list_resources")

    listed = await connections.list_resources(
        claim=claim, runtime=runtime, connection_id="a" * 32, cursor="page-2"
    )
    assert listed.text_content.startswith("notion://docs/a")
    read = await connections.read_resource(
        claim=claim, runtime=runtime, connection_id="a" * 32, uri="notion://docs/a"
    )
    assert read.text_content == "page body"
    assert [name for name, _ in resources.calls] == ["list_resources", "read_resource"]
    assert resources.calls[0][1]["cursor"] == "page-2"
    assert resources.calls[1][1]["uri"] == "notion://docs/a"
    assert resources.calls[0][1]["endpoint"] == "https://mcp.example.com/rpc"
    assert resources.calls[0][1]["endpoint_headers"] == (("X-Flavor", "pages"),)


async def test_resource_reads_refuse_a_deployment_without_the_transport() -> None:
    connections = _connections(_GateStore())
    claim = ResearchToolClaim("owner-1", "run-1", "worker-1", 1, _cancelled_nothing)

    with pytest.raises(ConnectionsError, match="resources unavailable"):
        await connections.list_resources(
            claim=claim,
            runtime=tool_runtime(tool_name="mcp_list_resources"),
            connection_id="a" * 32,
        )


async def _cancelled_nothing() -> None:
    return None


async def test_whitespace_and_long_prefix_labels_never_share_a_token() -> None:
    """The model must be able to name every pinned Connection unambiguously."""
    shared_prefix = "N" * 90
    bindings_by_id = (
        ("a" * 32, "Acme  Docs"),
        ("c" * 32, "Acme Docs"),
        ("d" * 32, shared_prefix + "one"),
        ("e" * 32, shared_prefix + "two"),
    )
    rows = tuple(
        PinnedCatalogue(
            RunConnectionBinding("owner-1", connection_id, generation, 1, "b" * 64),
            _CATALOGUE,
            label,
        )
        for generation, (connection_id, label) in enumerate(bindings_by_id, start=1)
    )
    bindings = await _connections(_CatalogueStore(rows), resources=_Resources()).bind_research(
        owner_id="owner-1"
    )

    tokens = [entry.token for entry in bindings.surface]
    assert len(set(tokens)) == len(tokens)
    assert all(len(token) <= 80 for token in tokens)
    # Each ambiguous pair is told apart by the identity, on both sides. Sorting is by
    # connection id, which here matches the row order.
    assert tokens[0].endswith("(" + "a" * 8 + ")")
    assert tokens[1].endswith("(" + "c" * 8 + ")")
    assert tokens[2].endswith("(" + "d" * 8 + ")")
    assert tokens[3].endswith("(" + "e" * 8 + ")")


def test_the_wire_refuses_two_connections_that_would_share_a_token() -> None:
    from dlightrag.engine.answer.execution.connection_binding import (
        ConnectionSurface,
        decode_connection_surface,
    )

    duplicate = [
        ConnectionSurface("a" * 32, "Notion", "Notion", "").as_json(),
        ConnectionSurface("c" * 32, "Notion", "Notion", "").as_json(),
    ]
    with pytest.raises(ValueError, match="Duplicate pinned Connection token"):
        decode_connection_surface(duplicate)


def test_a_connection_label_must_be_stated_text() -> None:
    from dlightrag.application.connections.models import ConnectionCommand

    for label in (" ", "   ", "no\nnewlines"):
        with pytest.raises(ValueError):
            ConnectionCommand(kind="create", label=label, endpoint="https://example.com/mcp")
