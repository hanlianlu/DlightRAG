# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""JSON codec for one control turn's restorable Answer run state.

Every owning abstraction exports and restores its own state; this module only
joins those exports, replaces image payloads with stable references, and
enforces the versioned envelope and its compact size bound. Raw, data-URI, and
base64 image bytes never reach checkpoint JSON: a caller attachment becomes an
owner artifact digest plus its run ordinal, and a knowledge-base visual becomes
workspace, chunk, and sidecar identity.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol

from dlightrag.core.answer_runs.models import (
    CHECKPOINT_SCHEMA_VERSION,
    MAX_CHECKPOINT_BYTES,
    AgentRunState,
    CheckpointError,
)

#: Resolve one knowledge-base visual by workspace and chunk, or ``None`` when it
#: no longer exists. A missing corpus visual never fails a run.
type CorpusImageLoader = Callable[[str, str], Awaitable[str | None]]

_DATA_URI_PREFIX = "data:"
_IMAGE_KEYS = ("image_data", "url")


class ArtifactReader(Protocol):
    """The owner-scoped artifact reads and reference listing the codec may use."""

    async def load_artifact(self, *, owner_id: str, digest: str) -> bytes | None: ...

    async def list_run_artifacts(self, *, owner_id: str, run_id: str) -> tuple[Any, ...]: ...


async def encode_checkpoint_state(
    state: AgentRunState,
    *,
    owner_id: str,
    run_id: str,
    store: ArtifactReader,
) -> dict[str, Any]:
    """Return the versioned envelope this control turn would commit.

    Serialization and the size bound are evaluated here, before any checkpoint
    transaction opens, so a rejected checkpoint never leaves a half-written row.
    """
    references = await store.list_run_artifacts(owner_id=owner_id, run_id=run_id)
    ordinals: dict[str, int] = {}
    for reference in references:
        ordinals.setdefault(str(reference.digest), int(reference.ordinal))

    payload: dict[str, Any] = {
        "evidence": state.evidence.export_state(),
        "episode": state.episode.export_state(),
        "tool_results": state.tool_cache.export_results(),
        "resources": state.registry.export_state() if state.registry is not None else None,
    }
    _substitute_images(payload, ordinals=ordinals)
    envelope = {
        "version": CHECKPOINT_SCHEMA_VERSION,
        "completed_turns": state.completed_turns,
        "state": payload,
    }
    encoded = json.dumps(
        envelope, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(encoded) > MAX_CHECKPOINT_BYTES:
        raise CheckpointError(
            "checkpoint_too_large",
            "Answer run state exceeds the durable checkpoint size bound.",
        )
    return envelope


def decode_checkpoint_state(
    envelope: Mapping[str, Any],
    *,
    expected_completed_turns: int | None = None,
) -> tuple[dict[str, Any], int]:
    """Validate a stored envelope against the authoritative run row."""
    version = envelope.get("version")
    if not isinstance(version, int) or version != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(
            "checkpoint_incompatible",
            "Answer run checkpoint was written by an incompatible revision.",
        )
    completed_turns = envelope.get("completed_turns")
    state = envelope.get("state")
    if not isinstance(completed_turns, int) or not isinstance(state, dict):
        raise CheckpointError(
            "checkpoint_corrupt", "Answer run checkpoint is not restorable state."
        )
    if expected_completed_turns is not None and completed_turns != expected_completed_turns:
        raise CheckpointError(
            "checkpoint_corrupt",
            "Answer run checkpoint does not match its authoritative turn count.",
        )
    return state, completed_turns


async def restore_agent_state(
    state: AgentRunState,
    envelope: Mapping[str, Any],
    *,
    owner_id: str,
    run_id: str,
    store: ArtifactReader,
    expected_completed_turns: int | None = None,
    load_corpus_image: CorpusImageLoader | None = None,
) -> None:
    """Restore one run's memory in place before its next model call."""
    payload, completed_turns = decode_checkpoint_state(
        envelope, expected_completed_turns=expected_completed_turns
    )
    await _rehydrate_images(
        payload,
        owner_id=owner_id,
        store=store,
        load_corpus_image=load_corpus_image,
    )
    try:
        state.evidence.restore_state(payload["evidence"])
        state.episode.restore_state(payload["episode"])
        state.tool_cache.restore_results(payload.get("tool_results") or {})
        resources = payload.get("resources")
        if state.registry is not None and resources is not None:
            state.registry.restore_state(resources)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise CheckpointError(
            "checkpoint_corrupt", "Answer run checkpoint is not restorable state."
        ) from exc
    state.completed_turns = completed_turns
    state.trace["agent_turns"] = completed_turns


def _substitute_images(node: Any, *, ordinals: Mapping[str, int]) -> None:
    """Replace every image payload in place with a durable reference."""
    if isinstance(node, dict):
        typed = dict[str, Any](node)
        for key in _IMAGE_KEYS:
            payload = typed.get(key)
            if isinstance(payload, str) and _looks_like_image(payload):
                node[key] = _reference_for(typed, payload, ordinals=ordinals)
        for value in typed.values():
            _substitute_images(value, ordinals=ordinals)
        return
    if isinstance(node, list):
        for value in list[Any](node):
            _substitute_images(value, ordinals=ordinals)


def _reference_for(
    owner: Mapping[str, Any], payload: str, *, ordinals: Mapping[str, int]
) -> dict[str, Any]:
    chunk_id = owner.get("chunk_id")
    workspace = owner.get("_workspace")
    if chunk_id and workspace:
        return {
            "kind": "corpus",
            "workspace": str(workspace),
            "chunk_id": str(chunk_id),
            "sidecar": str(owner.get("sidecar_location") or ""),
        }
    media_type, raw = _decode_image(payload)
    digest = hashlib.sha256(raw).hexdigest()
    return {
        "kind": "attachment",
        "digest": digest,
        "ordinal": ordinals.get(digest),
        "media_type": media_type,
    }


async def _rehydrate_images(
    node: Any,
    *,
    owner_id: str,
    store: ArtifactReader,
    load_corpus_image: CorpusImageLoader | None,
) -> None:
    """Restore image payloads in place, preserving message and block order."""
    if isinstance(node, dict):
        typed = dict[str, Any](node)
        for key in _IMAGE_KEYS:
            reference = typed.get(key)
            if not isinstance(reference, Mapping) or "kind" not in reference:
                continue
            restored = await _restore_image(
                reference,
                owner_id=owner_id,
                store=store,
                load_corpus_image=load_corpus_image,
            )
            if restored is None:
                # A knowledge-base visual that no longer resolves drops only its
                # image block; the row keeps its text and citation identity.
                del node[key]
            else:
                node[key] = restored
        for value in typed.values():
            await _rehydrate_images(
                value, owner_id=owner_id, store=store, load_corpus_image=load_corpus_image
            )
        return
    if isinstance(node, list):
        for value in list[Any](node):
            await _rehydrate_images(
                value, owner_id=owner_id, store=store, load_corpus_image=load_corpus_image
            )


async def _restore_image(
    reference: Mapping[str, Any],
    *,
    owner_id: str,
    store: ArtifactReader,
    load_corpus_image: CorpusImageLoader | None,
) -> str | None:
    if reference.get("kind") == "corpus":
        if load_corpus_image is None:
            return None
        return await load_corpus_image(
            str(reference.get("workspace") or ""), str(reference.get("chunk_id") or "")
        )
    digest = str(reference.get("digest") or "")
    content = await store.load_artifact(owner_id=owner_id, digest=digest)
    if content is None:
        raise CheckpointError(
            "checkpoint_corrupt",
            "Answer run checkpoint references attachment bytes that no longer exist.",
        )
    encoded = base64.b64encode(content).decode("ascii")
    media_type = reference.get("media_type")
    return f"data:{media_type};base64,{encoded}" if media_type else encoded


def _looks_like_image(payload: str) -> bool:
    if payload.startswith(_DATA_URI_PREFIX):
        return ";base64," in payload
    return len(payload) > 32 and _decodes_as_base64(payload)


def _decodes_as_base64(payload: str) -> bool:
    try:
        base64.b64decode(payload, validate=True)
    except binascii.Error, ValueError:
        return False
    return True


def _decode_image(payload: str) -> tuple[str | None, bytes]:
    if payload.startswith(_DATA_URI_PREFIX):
        header, _, encoded = payload.partition(",")
        media_type = header[len(_DATA_URI_PREFIX) :].removesuffix(";base64") or None
        return media_type, base64.b64decode(encoded, validate=False)
    return None, base64.b64decode(payload, validate=False)


__all__ = [
    "ArtifactReader",
    "CheckpointError",
    "CorpusImageLoader",
    "decode_checkpoint_state",
    "encode_checkpoint_state",
    "restore_agent_state",
]
