# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Secret-free host contracts for durable Research definition binding.

Bindings pin local definitions, not remote code, account isolation or effects.
The host obtains claims from trusted Run execution, never model arguments.
"""

import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from dlightrag.engine.agent.tools import AgentTool


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
    if not isinstance(value, list) or len(value) > 100:
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
