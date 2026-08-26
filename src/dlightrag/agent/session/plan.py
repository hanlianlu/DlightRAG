# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The canonical immutable execution contract accepted for one Agent run."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

from dlightrag.agent.session.effects import ReplayPolicy, canonical_json
from dlightrag.agent.tools.contracts import AgentTool
from dlightrag.ai.tokens import estimate_tokens

AGENT_RUN_PLAN_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class AgentToolPlan:
    """One pinned provider definition and execution/replay contract."""

    definition_json: str
    guidance: str
    replay_policy: ReplayPolicy
    contract_version: int
    input_schema_digest: str

    def __post_init__(self) -> None:
        definition = self.definition
        name = definition.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Agent Tool Plan requires a named definition")
        if self.replay_policy not in {"replayable", "never"}:
            raise ValueError("Agent Tool Plan replay policy must be replayable or never")
        if self.contract_version < 1:
            raise ValueError("Agent Tool Plan contract version must be positive")
        if len(self.input_schema_digest) != 64:
            raise ValueError("Agent Tool Plan schema digest must be SHA-256")

    @classmethod
    def from_tool(cls, tool: AgentTool) -> AgentToolPlan:
        return cls(
            definition_json=canonical_json(asdict(tool.definition)),
            guidance=tool.guidance,
            replay_policy=tool.replay_policy,
            contract_version=tool.contract_version,
            input_schema_digest=tool.input_schema_digest,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AgentToolPlan:
        definition = payload.get("definition")
        if not isinstance(definition, Mapping):
            raise ValueError("Agent Tool Plan definition must be an object")
        return cls(
            definition_json=canonical_json(dict(definition)),
            guidance=str(payload.get("guidance") or ""),
            replay_policy=payload["replay_policy"],
            contract_version=int(payload["contract_version"]),
            input_schema_digest=str(payload["input_schema_digest"]),
        )

    @property
    def definition(self) -> dict[str, Any]:
        value = json.loads(self.definition_json)
        if not isinstance(value, dict):
            raise ValueError("Agent Tool Plan definition is not an object")
        return value

    @property
    def name(self) -> str:
        return str(self.definition["name"])

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "definition": self.definition,
            "guidance": self.guidance,
            "replay_policy": self.replay_policy,
            "contract_version": self.contract_version,
            "input_schema_digest": self.input_schema_digest,
        }


@dataclass(frozen=True, slots=True)
class AgentRunPlan:
    """One versioned Plan serialized identically at acceptance and execution."""

    model_role: str
    context_policy_revision: str
    tools: tuple[AgentToolPlan, ...]
    model_identity_json: str = "{}"
    model_profile_json: str = "{}"
    prompt_revision: str = "answer-prompts-v1"
    provider_attempt_limit: int = 2
    compaction_attempt_limit: int = 3
    max_pending_steers: int = 16
    max_pending_follow_ups: int = 16
    host_contract_version: int = 1
    schema_version: int = AGENT_RUN_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.model_role.strip():
            raise ValueError("Agent Run Plan model role cannot be empty")
        if not self.context_policy_revision.strip():
            raise ValueError("Agent Run Plan context policy revision cannot be empty")
        if self.schema_version != AGENT_RUN_PLAN_SCHEMA_VERSION:
            raise ValueError("Agent Run Plan schema version is not current")
        if not self.prompt_revision:
            raise ValueError("Agent Run Plan prompt revision cannot be empty")
        if (
            self.provider_attempt_limit < 1
            or self.compaction_attempt_limit < 1
            or self.max_pending_steers < 1
            or self.max_pending_follow_ups < 1
            or self.host_contract_version < 1
        ):
            raise ValueError("Agent Run Plan bounded policies must be positive")
        for label, value in (
            ("model identity", self.model_identity_json),
            ("model profile", self.model_profile_json),
        ):
            if not isinstance(json.loads(value), dict):
                raise ValueError(f"Agent Run Plan {label} must be an object")
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("Agent Run Plan tool names must be unique")

    @classmethod
    def from_tools(
        cls,
        tools: Sequence[AgentTool],
        *,
        model_role: str,
        context_policy_revision: str,
        model_identity: Mapping[str, Any] | None = None,
        model_profile: Mapping[str, Any] | None = None,
        prompt_revision: str = "answer-prompts-v1",
    ) -> AgentRunPlan:
        return cls(
            model_role=model_role,
            context_policy_revision=context_policy_revision,
            tools=tuple(
                AgentToolPlan.from_tool(tool) for tool in sorted(tools, key=lambda item: item.name)
            ),
            model_identity_json=canonical_json(dict(model_identity or {})),
            model_profile_json=canonical_json(dict(model_profile or {})),
            prompt_revision=prompt_revision,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AgentRunPlan:
        raw_tools = payload.get("tools")
        if not isinstance(raw_tools, list | tuple):
            raise ValueError("Agent Run Plan tools must be an array")
        if any(not isinstance(tool, Mapping) for tool in raw_tools):
            raise ValueError("Agent Run Plan tool entries must be objects")
        return cls(
            model_role=str(payload["model_role"]),
            context_policy_revision=str(payload["context_policy_revision"]),
            tools=tuple(AgentToolPlan.from_payload(tool) for tool in raw_tools),  # type: ignore[arg-type]
            model_identity_json=canonical_json(dict(payload.get("model_identity") or {})),
            model_profile_json=canonical_json(dict(payload.get("model_profile") or {})),
            prompt_revision=str(payload["prompt_revision"]),
            provider_attempt_limit=int(payload["provider_attempt_limit"]),
            compaction_attempt_limit=int(payload["compaction_attempt_limit"]),
            max_pending_steers=int(payload["max_pending_steers"]),
            max_pending_follow_ups=int(payload["max_pending_follow_ups"]),
            host_contract_version=int(payload["host_contract_version"]),
            schema_version=int(payload["schema_version"]),
        )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_role": self.model_role,
            "context_policy_revision": self.context_policy_revision,
            "model_identity": json.loads(self.model_identity_json),
            "model_profile": json.loads(self.model_profile_json),
            "prompt_revision": self.prompt_revision,
            "provider_attempt_limit": self.provider_attempt_limit,
            "compaction_attempt_limit": self.compaction_attempt_limit,
            "max_pending_steers": self.max_pending_steers,
            "max_pending_follow_ups": self.max_pending_follow_ups,
            "host_contract_version": self.host_contract_version,
            "tools": [tool.canonical_payload() for tool in self.tools],
        }

    def canonical_json(self) -> str:
        return canonical_json(self.canonical_payload())

    @property
    def digest(self) -> str:
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def tool_schema_tokens(self) -> int:
        definitions = [tool.definition for tool in self.tools]
        return estimate_tokens(canonical_json(definitions))


__all__ = [
    "AGENT_RUN_PLAN_SCHEMA_VERSION",
    "AgentRunPlan",
    "AgentToolPlan",
]
