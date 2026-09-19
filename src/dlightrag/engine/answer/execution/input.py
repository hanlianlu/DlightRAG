# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The immutable input and durable turn boundaries of one Answer run.

This is the seam between the coordinator, which owns durability, and the answer
orchestrator, which owns retrieval and synthesis. It holds no lifecycle state of
its own: the run row remains authoritative for status, turns, and cancellation.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dlightrag.engine.agent.session.plan import AgentRunPlan
from dlightrag.engine.ai.capacity import CONTEXT_POLICY_REVISION, ModelProfile
from dlightrag.engine.ai.catalog import current_model_catalog_revision
from dlightrag.engine.ai.fingerprints import ModelInvocationFingerprint
from dlightrag.engine.ai.reasoning import (
    REASONING_LEVELS,
    ReasoningLevels,
    ReasoningProfile,
    resolve_reasoning,
)
from dlightrag.engine.ai.settings import CHAT_MODEL_SELECTORS, ChatModelSelector, ModelSettings
from dlightrag.engine.answer.client_contracts import AnswerEffort, normalize_answer_effort
from dlightrag.engine.answer.execution.connection_binding import (
    RunConnectionBinding,
    decode_connection_bindings,
)
from dlightrag.engine.answer.mode import canonical_answer_mode
from dlightrag.engine.answer.resources.models import ResourceInput
from dlightrag.engine.rag.retrieval import RetrievalOptions
from dlightrag.engine.runtime.errors import IncompatibleActiveRunError, RunExecutionError


@dataclass(frozen=True, slots=True)
class AttachmentReference:
    """One ordered current attachment, addressed by owner artifact digest."""

    digest: str
    filename: str
    mime_type: str
    ordinal: int
    byte_size: int = 0

    def as_json(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "ordinal": self.ordinal,
            "byte_size": self.byte_size,
        }

    @property
    def resource_id(self) -> str:
        return f"attachment-{self.ordinal}"

    @property
    def history_resource_id(self) -> str:
        return f"history-attachment-{self.ordinal}"


async def _current_attachment_resource(
    reference: AttachmentReference,
    load: Callable[[], Awaitable[bytes]],
) -> ResourceInput:
    """Rebuild one current attachment under its durable declared MIME."""
    if reference.mime_type.strip().casefold().startswith("image/"):
        return ResourceInput(
            filename=reference.filename,
            declared_mime=reference.mime_type,
            content=await load(),
        )
    return ResourceInput(
        filename=reference.filename,
        declared_mime=reference.mime_type,
        loader=load,
    )


@dataclass(frozen=True, slots=True)
class LinkReference:
    """One ordered HTTPS attachment link, kept inert until an explicit read."""

    url: str
    filename: str | None
    ordinal: int
    mime_type: str | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "filename": self.filename,
            "ordinal": self.ordinal,
            "mime_type": self.mime_type,
        }


def in_memory_attachment_loader(content: bytes) -> Callable[[], Awaitable[bytes]]:
    """Return a stable async loader over bytes already admitted in memory."""

    async def load() -> bytes:
        return content

    return load


async def build_current_answer_resources(
    *,
    links: Sequence[LinkReference],
    attachments: Sequence[AttachmentReference],
    attachment_loaders: Sequence[Callable[[], Awaitable[bytes]]],
) -> list[ResourceInput]:
    """Rebuild links and current attachments exactly as the durable run will."""
    if len(attachments) != len(attachment_loaders):
        raise ValueError("current attachment references and loaders must have equal length")
    resources = [
        ResourceInput(
            filename=link.filename,
            url=link.url,
            declared_mime=link.mime_type,
        )
        for link in links
    ]
    for reference, load in zip(attachments, attachment_loaders, strict=True):
        resources.append(await _current_attachment_resource(reference, load))
    return resources


@dataclass(frozen=True, slots=True)
class PinnedModelProfile:
    """One accepted run's immutable model identity and capacity facts."""

    role: str
    fingerprint: ModelInvocationFingerprint
    profile: ModelProfile
    reasoning_settings: Mapping[str, Any] | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "role": self.role,
            **(
                {"reasoning_settings": dict(self.reasoning_settings)}
                if self.reasoning_settings is not None
                else {}
            ),
            "fingerprint": self.fingerprint.as_json(),
            "profile": {
                "context_window_tokens": self.profile.context_window_tokens,
                "max_input_tokens": self.profile.max_input_tokens,
                "max_output_tokens": self.profile.max_output_tokens,
                "supports_images": self.profile.supports_images,
                "reasoning": (
                    {
                        **self.profile.reasoning.as_dict(),
                        "best_effort": self.profile.reasoning.best_effort,
                    }
                    if self.profile.reasoning is not None
                    else None
                ),
            },
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> PinnedModelProfile:
        fingerprint = value.get("fingerprint")
        profile = value.get("profile")
        if not isinstance(profile, Mapping):
            raise ValueError("pinned model profile requires fingerprint and profile objects")
        if "reasoning" not in profile:
            raise ValueError("pinned model profile requires explicit reasoning facts")
        return cls(
            role=str(value.get("role") or ""),
            reasoning_settings=_pinned_reasoning_settings(value.get("reasoning_settings")),
            fingerprint=ModelInvocationFingerprint.from_json(fingerprint),
            profile=ModelProfile(
                context_window_tokens=int(profile["context_window_tokens"]),
                max_input_tokens=_optional_int(profile.get("max_input_tokens")),
                max_output_tokens=_optional_int(profile.get("max_output_tokens")),
                supports_images=bool(profile.get("supports_images")),
                reasoning=_reasoning_profile_from_json(profile.get("reasoning")),
            ),
        )


@dataclass(frozen=True, slots=True)
class AnswerRunRequest:
    """Normalized public request before model resolution and history projection."""

    query: str
    workspaces: tuple[str, ...] = ()
    history: tuple[Mapping[str, Any], ...] = ()
    episodic_summary: str = ""
    retrieval: RetrievalOptions = RetrievalOptions()
    filters: Mapping[str, Any] | None = None
    semantic_highlights: bool = False
    links: tuple[LinkReference, ...] = ()
    attachments: tuple[AttachmentReference, ...] = ()
    history_attachments: tuple[AttachmentReference, ...] = ()
    mode: str = "auto"
    parent_run_id: str | None = None
    continuation_kind: str | None = None
    agent_session_id: str = ""
    agent_lane_id: str = "main"
    source_lane_id: str | None = None
    requested_skill: str | None = None
    #: The caller's own agent effort for this run, when one was chosen.
    effort: AnswerEffort | None = None

    def as_request(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "workspaces": list(self.workspaces),
            "history": [dict(message) for message in self.history],
            "episodic_summary": self.episodic_summary,
            "top_k": self.retrieval.top_k,
            "chunk_top_k": self.retrieval.chunk_top_k,
            "federated_rerank": self.retrieval.federated_rerank,
            "filters": dict(self.filters) if self.filters else None,
            "semantic_highlights": self.semantic_highlights,
            "links": [item.as_json() for item in self.links],
            "attachments": [item.as_json() for item in self.attachments],
            "history_attachments": [item.as_json() for item in self.history_attachments],
            "mode": canonical_answer_mode(self.mode),
            "parent_run_id": self.parent_run_id,
            "continuation_kind": self.continuation_kind,
            "agent_session_id": self.agent_session_id,
            "agent_lane_id": self.agent_lane_id,
            "source_lane_id": self.source_lane_id,
            "requested_skill": self.requested_skill,
            "effort": self.effort,
        }

    @classmethod
    def from_request(cls, request: Mapping[str, Any]) -> AnswerRunRequest:
        """Project the public request fields without parsing execution metadata."""
        filters = request.get("filters")
        return cls(
            query=str(request.get("query") or ""),
            workspaces=tuple(str(value) for value in request.get("workspaces") or ()),
            history=tuple(dict(message) for message in request.get("history") or ()),
            episodic_summary=str(request.get("episodic_summary") or ""),
            retrieval=RetrievalOptions(
                top_k=_optional_int(request.get("top_k")),
                chunk_top_k=_optional_int(request.get("chunk_top_k")),
                federated_rerank=bool(request.get("federated_rerank")),
            ),
            filters=dict(filters) if isinstance(filters, Mapping) else None,
            semantic_highlights=bool(request.get("semantic_highlights")),
            links=_link_references(request.get("links")),
            attachments=_attachment_references(request.get("attachments")),
            history_attachments=_attachment_references(request.get("history_attachments")),
            mode=str(request.get("mode") or "auto"),
            parent_run_id=(str(request["parent_run_id"]) if request.get("parent_run_id") else None),
            continuation_kind=(
                str(request["continuation_kind"]) if request.get("continuation_kind") else None
            ),
            agent_session_id=str(request.get("agent_session_id") or ""),
            agent_lane_id=str(request.get("agent_lane_id") or "main"),
            source_lane_id=(
                str(request["source_lane_id"]) if request.get("source_lane_id") else None
            ),
            requested_skill=(
                str(request["requested_skill"]).strip() if request.get("requested_skill") else None
            ),
            effort=normalize_answer_effort(request.get("effort")),
        )


@dataclass(frozen=True, slots=True)
class AnswerRunInput:
    """The normalized, immutable request one accepted run executes.

    Workspace authorization is evaluated once before the run is accepted, so the
    stored input carries the resulting workspace set and never a token, mutable
    claim, transport header, temporary path, or authorization-dependent URL.
    """

    query: str
    pinned_models: tuple[PinnedModelProfile, ...]
    context_policy_revision: str
    model_catalog_revision: str
    idempotency_fingerprint: str
    agent_run_plan: AgentRunPlan | None = None
    run_connection_bindings: tuple[RunConnectionBinding, ...] = ()
    workspaces: tuple[str, ...] = ()
    history: tuple[Mapping[str, Any], ...] = ()
    episodic_summary: str = ""
    retrieval: RetrievalOptions = RetrievalOptions()
    filters: Mapping[str, Any] | None = None
    semantic_highlights: bool = False
    links: tuple[LinkReference, ...] = ()
    attachments: tuple[AttachmentReference, ...] = ()
    #: Earlier conversation uploads this run may read but never sends as a
    #: current-turn image; they point at artifacts an earlier run already stored.
    history_attachments: tuple[AttachmentReference, ...] = ()
    image_descriptions: tuple[str, ...] = ()
    #: Canonical Agent Session and selected Lane pinned at acceptance for every mode.
    agent_session_id: str = ""
    agent_lane_id: str = "main"
    source_lane_id: str | None = None
    #: The accepted resource manifest, present for research runs.
    resource_manifest: tuple[Mapping[str, Any], ...] = ()
    parent_run_id: str | None = None
    continuation_kind: str | None = None
    requested_skill: str | None = None
    #: The caller's own agent effort for this run, when one was chosen.
    effort: AnswerEffort | None = None

    def as_request(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "workspaces": list(self.workspaces),
            "history": [dict(message) for message in self.history],
            "top_k": self.retrieval.top_k,
            "chunk_top_k": self.retrieval.chunk_top_k,
            "federated_rerank": self.retrieval.federated_rerank,
            "filters": dict(self.filters) if self.filters else None,
            "semantic_highlights": self.semantic_highlights,
            "links": [item.as_json() for item in self.links],
            "attachments": [item.as_json() for item in self.attachments],
            "history_attachments": [item.as_json() for item in self.history_attachments],
            "episodic_summary": self.episodic_summary,
            "pinned_models": [item.as_json() for item in self.pinned_models],
            "context_policy_revision": self.context_policy_revision,
            "model_catalog_revision": self.model_catalog_revision,
            "idempotency_fingerprint": self.idempotency_fingerprint,
            "run_connection_bindings": [
                binding.as_json() for binding in self.run_connection_bindings
            ],
            "agent_run_plan": (
                self.agent_run_plan.canonical_payload() if self.agent_run_plan is not None else None
            ),
            "image_descriptions": list(self.image_descriptions),
            "agent_session_id": self.agent_session_id,
            "agent_lane_id": self.agent_lane_id,
            "source_lane_id": self.source_lane_id,
            "resource_manifest": [dict(item) for item in self.resource_manifest],
            "parent_run_id": self.parent_run_id,
            "continuation_kind": self.continuation_kind,
            "requested_skill": self.requested_skill,
            "effort": self.effort,
        }

    @classmethod
    def from_request(cls, request: Mapping[str, Any]) -> AnswerRunInput:
        attachments = _attachment_references(request.get("attachments"))
        history_attachments = _attachment_references(request.get("history_attachments"))
        links = _link_references(request.get("links"))
        filters = request.get("filters")
        pinned_models = tuple(
            PinnedModelProfile.from_json(item) for item in request.get("pinned_models") or ()
        )
        context_policy_revision = str(request.get("context_policy_revision") or "")
        model_catalog_revision = str(request.get("model_catalog_revision") or "")
        idempotency_fingerprint = str(request.get("idempotency_fingerprint") or "")
        raw_agent_run_plan = request.get("agent_run_plan")
        agent_run_plan = (
            AgentRunPlan.from_payload(raw_agent_run_plan)
            if isinstance(raw_agent_run_plan, Mapping)
            else None
        )
        if (
            not pinned_models
            or not context_policy_revision
            or not model_catalog_revision
            or not idempotency_fingerprint
        ):
            raise ValueError("answer run input is missing pinned model capacity facts")
        return cls(
            query=str(request.get("query") or ""),
            pinned_models=pinned_models,
            context_policy_revision=context_policy_revision,
            model_catalog_revision=model_catalog_revision,
            idempotency_fingerprint=idempotency_fingerprint,
            agent_run_plan=agent_run_plan,
            run_connection_bindings=decode_connection_bindings(
                request.get("run_connection_bindings", [])
            ),
            workspaces=tuple(str(value) for value in request.get("workspaces") or ()),
            history=tuple(dict(message) for message in request.get("history") or ()),
            episodic_summary=str(request.get("episodic_summary") or ""),
            retrieval=RetrievalOptions(
                top_k=_optional_int(request.get("top_k")),
                chunk_top_k=_optional_int(request.get("chunk_top_k")),
                federated_rerank=bool(request.get("federated_rerank")),
            ),
            filters=dict(filters) if isinstance(filters, Mapping) else None,
            semantic_highlights=bool(request.get("semantic_highlights")),
            links=links,
            attachments=attachments,
            history_attachments=history_attachments,
            image_descriptions=tuple(str(item) for item in request.get("image_descriptions") or ()),
            agent_session_id=str(request.get("agent_session_id") or ""),
            agent_lane_id=str(request.get("agent_lane_id") or "main"),
            source_lane_id=(
                str(request["source_lane_id"]) if request.get("source_lane_id") else None
            ),
            resource_manifest=tuple(dict(item) for item in request.get("resource_manifest") or ()),
            parent_run_id=(str(request["parent_run_id"]) if request.get("parent_run_id") else None),
            continuation_kind=(
                str(request["continuation_kind"]) if request.get("continuation_kind") else None
            ),
            requested_skill=(
                str(request["requested_skill"]).strip() if request.get("requested_skill") else None
            ),
            effort=normalize_answer_effort(request.get("effort")),
        )

    @classmethod
    def from_prepared_input(cls, prepared: Mapping[str, Any] | None) -> AnswerRunInput:
        """Decode one durable prepared input into the immutable run input."""
        if prepared is None:
            raise RunExecutionError(
                "run_execution_failed", "Answer run has no prepared input to execute."
            )
        return cls.from_request(prepared)


def model_reasoning_settings(settings: ModelSettings) -> dict[str, Any]:
    """Secret-free request levels, including auth-aware complete-role inheritance."""
    return {"ordinary": settings.reasoning, "agentic": settings.effective_agentic_reasoning}


def _pinned_reasoning_settings(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"ordinary", "agentic"}:
        raise ValueError("invalid pinned reasoning settings")
    settings = ModelSettings(
        model="pinned", reasoning=value["ordinary"], agentic_reasoning=value["agentic"]
    )
    return model_reasoning_settings(settings)


def pinned_model_selectors(pins: tuple[PinnedModelProfile, ...]) -> tuple[ChatModelSelector, ...]:
    roles = {item.role for item in pins}
    selectors = CHAT_MODEL_SELECTORS
    if len(pins) != len(selectors) or roles != set(selectors):
        raise IncompatibleActiveRunError(
            "answer run does not contain the complete pinned model role set"
        )
    if any(item.reasoning_settings is None for item in pins):
        raise IncompatibleActiveRunError("answer run is missing pinned model reasoning settings")
    return selectors


def child_model_guidance(pins: tuple[PinnedModelProfile, ...]) -> str:
    """Describe accepted endpoint facts, never a live catalogue on recovery."""
    intents = {
        "query": "preferred strongest reasoning tier: hardest research, planning, evidence adjudication and final review",
        "default": "general-purpose tier below query: ordinary analysis, synthesis, drafting and routine review",
        "extract": "routine extraction, normalization and structured work",
        "keyword": "lightweight keywords, labels and query rewriting",
        "vlm": "visual evidence, images, charts and document pages; not inherently cheap or fast",
    }
    lines = [
        "Model selectors are recommendations, not task categories or permissions; objectives are unrestricted. query is the default selector. Effective configured capabilities follow (tiers are intent, not guarantees):"
    ]
    for pin in pins:
        profile = pin.profile
        reasoning = (pin.reasoning_settings or {}).get("agentic")
        resolved = resolve_reasoning(profile.reasoning, reasoning)
        lines.append(
            f"{pin.role}: {intents[pin.role]}; model={pin.fingerprint.model}; images={profile.supports_images}; context_tokens={profile.context_window_tokens}; agentic_reasoning_request={reasoning or 'provider default'}; agentic_reasoning_effective={resolved.effective if resolved else 'provider default'}; reasoning_profile={profile.reasoning.as_dict() if profile.reasoning else 'none'}."
        )
    lines.append(
        "Reasoning max is a configured request level, not a universal capability guarantee. Image support follows the effective profile, not the vlm label. All selectors use the supported tool-calling wrapper; provider rejection is explicit."
    )
    return "\n".join(lines)


def validate_active_answer_input(
    prepared: Mapping[str, Any],
    *,
    model_invocation_fingerprint_for_role: Callable[
        [ChatModelSelector], ModelInvocationFingerprint
    ],
    model_settings_for_role: Callable[[ChatModelSelector], ModelSettings] | None = None,
) -> None:
    """Require one active Answer input to remain executable by this deployment."""
    try:
        run_input = AnswerRunInput.from_prepared_input(prepared)
    except (AttributeError, KeyError, TypeError, ValueError, RunExecutionError) as exc:
        raise IncompatibleActiveRunError(
            "active answer runs use an incompatible durable input schema; "
            "start a new Run with the current configuration"
        ) from exc
    pinned = {item.role: item for item in run_input.pinned_models}
    selectors = pinned_model_selectors(run_input.pinned_models)
    if model_settings_for_role is None or any(
        pinned[role].reasoning_settings != model_reasoning_settings(model_settings_for_role(role))
        for role in selectors
    ):
        raise IncompatibleActiveRunError(
            "active answer runs target another model reasoning configuration"
        )
    if run_input.context_policy_revision != CONTEXT_POLICY_REVISION:
        raise IncompatibleActiveRunError(
            "active answer runs use another context policy revision; "
            "start a new Run with the current configuration"
        )
    if run_input.model_catalog_revision != current_model_catalog_revision():
        raise IncompatibleActiveRunError(
            "active answer runs use another model catalog revision; "
            "start a new Run with the current configuration"
        )
    if any(
        pinned[role].fingerprint != model_invocation_fingerprint_for_role(role)
        for role in selectors
    ):
        raise IncompatibleActiveRunError(
            "active answer runs target another model invocation configuration; "
            "start a new Run with the current configuration"
        )


def _attachment_references(value: Any) -> tuple[AttachmentReference, ...]:
    return tuple(
        AttachmentReference(
            digest=str(item["digest"]),
            filename=str(item["filename"]),
            mime_type=str(item["mime_type"]),
            ordinal=int(item["ordinal"]),
            byte_size=int(item.get("byte_size") or 0),
        )
        for item in value or ()
    )


def _link_references(value: Any) -> tuple[LinkReference, ...]:
    return tuple(
        LinkReference(
            url=str(item["url"]),
            filename=(str(item["filename"]) if item.get("filename") else None),
            ordinal=int(item["ordinal"]),
            mime_type=(str(item["mime_type"]) if item.get("mime_type") else None),
        )
        for item in value or ()
    )


def _reasoning_profile_from_json(value: Any) -> ReasoningProfile | None:
    if value is None:
        return None
    required = {"format", "levels"}
    if not isinstance(value, Mapping):
        raise ValueError("pinned reasoning profile has an invalid shape")
    keys = set(value)
    if not required <= keys or not keys <= required | {"best_effort"}:
        raise ValueError("pinned reasoning profile has an invalid shape")
    best_effort = value.get("best_effort", False)
    if type(best_effort) is not bool:
        raise ValueError("pinned reasoning profile best_effort must be a boolean")
    raw_levels = value.get("levels")
    if not isinstance(raw_levels, Mapping) or set(raw_levels) != set(REASONING_LEVELS):
        raise ValueError("pinned reasoning profile requires every explicit level")
    levels: dict[str, str | None] = {}
    for level in REASONING_LEVELS:
        raw = raw_levels[level]
        if raw is not None and not isinstance(raw, str):
            raise ValueError(f"pinned reasoning level {level} must be a string or null")
        levels[level] = raw
    parsed_levels = ReasoningLevels(**levels)  # type: ignore[arg-type]
    if best_effort:
        return ReasoningProfile.unverified(
            format=str(value["format"]),
            levels=parsed_levels,
        )
    return ReasoningProfile(format=str(value["format"]), levels=parsed_levels)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


__all__ = [
    "AnswerRunInput",
    "AnswerRunRequest",
    "AttachmentReference",
    "build_current_answer_resources",
    "in_memory_attachment_loader",
    "LinkReference",
    "PinnedModelProfile",
    "ResourceInput",
    "validate_active_answer_input",
]
