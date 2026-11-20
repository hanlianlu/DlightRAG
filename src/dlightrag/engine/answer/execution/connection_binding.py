# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Secret-free host contracts for durable Research definition binding.

Bindings pin local definitions, not remote code, account isolation or effects.
The host obtains claims from trusted Run execution, never model arguments.
"""

import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from dlightrag.engine.agent.tools.contracts import (
    AgentTool,
    ToolResult,
    ToolRuntime,
)

#: Every Connection tool's name starts with this and no other tool's does, so the name itself
#: tells the model that an external server the user connected provides the tool.
CONNECTION_TOOL_PREFIX = "mcp__"


def is_connection_tool(name: str) -> bool:
    """Whether a Research tool name is a Connection tool's."""
    return name.startswith(CONNECTION_TOOL_PREFIX)


@dataclass(frozen=True, slots=True)
class RunConnectionBinding:
    owner_id: str
    connection_id: str
    generation: int
    activation_epoch: int
    catalogue_digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.owner_id, str)
            or not 1 <= len(self.owner_id) <= 512
            or not isinstance(self.connection_id, str)
            or not re.fullmatch(r"[a-f0-9]{32}", self.connection_id)
            or type(self.generation) is not int
            or not 1 <= self.generation < 2**63
            or type(self.activation_epoch) is not int
            or not 1 <= self.activation_epoch < 2**63
            or not isinstance(self.catalogue_digest, str)
            or not re.fullmatch(r"[a-f0-9]{64}", self.catalogue_digest)
        ):
            raise ValueError("Invalid Run Connection binding")

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


def decode_connection_bindings(value: Any) -> tuple[RunConnectionBinding, ...]:
    """Bounded, secret-free single wire shape; unknown fields are rejected."""
    if not isinstance(value, list) or len(value) > MAX_RUN_CONNECTIONS:
        raise ValueError("Invalid Run Connection bindings")
    fields = {"owner_id", "connection_id", "generation", "activation_epoch", "catalogue_digest"}
    if any(not isinstance(item, Mapping) or set(item) != fields for item in value):
        raise ValueError("Invalid Run Connection binding fields")
    bindings = tuple(RunConnectionBinding(**item) for item in value)
    if len({(item.owner_id, item.connection_id) for item in bindings}) != len(bindings):
        raise ValueError("Duplicate Run Connection binding")
    return bindings


class StaleConnectionBindingError(RuntimeError):
    """Acceptance snapshot changed; rebuild once before reporting a conflict."""


@dataclass(frozen=True, slots=True)
class ResearchToolClaim:
    owner_id: str
    run_id: str
    worker_id: str
    fencing_epoch: int
    check_cancelled: Callable[[], Awaitable[None]]


class ResearchConnectionToolResolver(Protocol):
    async def __call__(
        self, *, bindings: tuple[RunConnectionBinding, ...], claim: ResearchToolClaim
    ) -> tuple[AgentTool, ...]: ...


#: The most one pinned Connection may contribute of its server's own instructions to a Run
#: prompt. The Connection policy bounds what is stored; this bounds what a decoded Run input
#: may carry, so a corrupt or hand-built payload cannot inflate one prompt.
MAX_CONNECTION_INSTRUCTIONS_BYTES = 131072
#: The most Connections one Run may pin: the Connection policy's own ceiling.
MAX_RUN_CONNECTIONS = 100
#: The most instructions one decoded Run input may carry in total. The Connection policy
#: bounds each server; this bounds a hand-built or corrupt payload's whole contribution.
MAX_RUN_CONNECTION_INSTRUCTIONS_BYTES = 1048576


@dataclass(frozen=True, slots=True)
class ConnectionSurface:
    """One enabled Connection pinned into a Run for resources and attributed guidance.

    Secret-free: the pin's identity, the token the model names the Connection by, its
    owner-visible label, and the bounded initialize instructions published with the pinned
    generation. It is written at acceptance and read at execution; it authorizes nothing.
    """

    connection_id: str
    token: str
    label: str
    instructions: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.connection_id, str) or not re.fullmatch(
            r"[a-f0-9]{32}", self.connection_id
        ):
            raise ValueError("Invalid pinned Connection identity")
        if (
            not isinstance(self.token, str)
            or not 1 <= len(self.token) <= 80
            or any(ord(char) < 32 for char in self.token)
        ):
            raise ValueError("Invalid pinned Connection token")
        if not isinstance(self.label, str) or not 1 <= len(self.label) <= 200:
            raise ValueError("Invalid pinned Connection label")
        if not isinstance(self.instructions, str):
            raise ValueError("Invalid pinned Connection instructions")
        if len(self.instructions.encode()) > MAX_CONNECTION_INSTRUCTIONS_BYTES:
            raise ValueError("Pinned Connection instructions are too large")

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


def decode_connection_surface(value: Any) -> tuple[ConnectionSurface, ...]:
    """Bounded, secret-free single wire shape; unknown fields are rejected."""
    if not isinstance(value, list) or len(value) > MAX_RUN_CONNECTIONS:
        raise ValueError("Invalid Run Connection surface")
    fields = {"connection_id", "token", "label", "instructions"}
    if any(not isinstance(item, Mapping) or set(item) != fields for item in value):
        raise ValueError("Invalid Run Connection surface fields")
    surface = tuple(ConnectionSurface(**item) for item in value)
    if len({item.connection_id for item in surface}) != len(surface):
        raise ValueError("Duplicate pinned Connection")
    if len({item.token for item in surface}) != len(surface):
        raise ValueError("Duplicate pinned Connection token")
    if (
        sum(len(item.instructions.encode()) for item in surface)
        > MAX_RUN_CONNECTION_INSTRUCTIONS_BYTES
    ):
        raise ValueError("Run Connection instructions are too large")
    return surface


class ConnectionResourceOperations(Protocol):
    """The Connection owner's live resource reads, gated by the pinned Run.

    A Run host holds only this port: it never sees a credential, an endpoint, or the store.
    """

    async def list_resources(
        self,
        *,
        claim: ResearchToolClaim,
        runtime: ToolRuntime,
        connection_id: str,
        cursor: str | None = None,
    ) -> ToolResult: ...

    async def list_resource_templates(
        self,
        *,
        claim: ResearchToolClaim,
        runtime: ToolRuntime,
        connection_id: str,
        cursor: str | None = None,
    ) -> ToolResult: ...

    async def read_resource(
        self,
        *,
        claim: ResearchToolClaim,
        runtime: ToolRuntime,
        connection_id: str,
        uri: str,
    ) -> ToolResult: ...
