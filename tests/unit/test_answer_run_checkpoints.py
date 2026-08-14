# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Contract tests for durable Answer run checkpoints.

Covers the exported state of every owning abstraction, the versioned JSON codec,
image-reference substitution and rehydration, the compact 8 MiB bound, and the
public failure kinds a worker must use instead of guessing at state.
"""

import base64
import json
from typing import Any

import pytest
from dlightrag_agent.tools import ToolResult

from dlightrag.core.answer_runs.checkpoints import (
    CheckpointError,
    decode_checkpoint_state,
    encode_checkpoint_state,
    restore_agent_state,
)
from dlightrag.core.answer_runs.models import (
    CHECKPOINT_SCHEMA_VERSION,
    MAX_CHECKPOINT_BYTES,
    AgentRunState,
)
from dlightrag.core.memory.episode import RunEpisode
from dlightrag.core.memory.evidence import EvidenceLedger
from dlightrag.core.resources.models import ResourceInput
from dlightrag.core.resources.registry import ResourceRegistry, ResourceStateMismatchError
from dlightrag.core.tools import ExactCallCache

_PNG = base64.b64encode(b"\x89PNG\r\n\x1a\nfake-corpus-visual").decode("ascii")
_ATTACHMENT_BYTES = b"\x89PNG\r\n\x1a\nfake-attachment"
_ATTACHMENT_B64 = base64.b64encode(_ATTACHMENT_BYTES).decode("ascii")
_FETCHED_BYTES = b"<html>the page as it was when the run fetched it</html>"


class _FakeStore:
    """Only the artifact reads and writes the codec is allowed to use."""

    def __init__(self, *, artifacts: dict[str, bytes] | None = None) -> None:
        self.artifacts = dict(artifacts or {})
        self.references: list[Any] = []

    async def load_artifact(self, *, owner_id: str, digest: str) -> bytes | None:
        return self.artifacts.get(digest)

    async def list_run_artifacts(self, *, owner_id: str, run_id: str) -> tuple[Any, ...]:
        return tuple(self.references)


def _corpus_row(chunk_id: str = "chunk-1", *, image: bool = True) -> dict[str, Any]:
    row: dict[str, Any] = {
        "chunk_id": chunk_id,
        "content": "corpus evidence text",
        "reference_id": "ref-a",
        "file_path": "book.pdf",
        "_workspace": "ws-main",
        "sidecar_location": "file:///parsed/book",
        "metadata": {"source_type": "corpus"},
    }
    if image:
        row["image_data"] = _PNG
    return row


async def _state_with_evidence() -> AgentRunState:
    evidence = EvidenceLedger()
    evidence.add_contexts(
        {
            "chunks": [_corpus_row()],
            "entities": [{"entity_name": "Alpha"}],
            "relationships": [{"src_id": "Alpha", "tgt_id": "Beta"}],
        }
    )
    episode = RunEpisode()
    episode.record(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "search_knowledge_base", "arguments": "{}"},
                        "thought_signature": "sig-1",
                    }
                ],
                "provider_state": {"reasoning": "native"},
            },
            {"role": "tool", "tool_call_id": "call-1", "name": "kb", "content": "found"},
        ]
    )
    cache = ExactCallCache()
    await cache.run("kb\x00{}", lambda: _ok("found"))
    registry = ResourceRegistry()
    registry.register(ResourceInput(content=b"doc-bytes", filename="a.txt"))
    return AgentRunState(
        evidence=evidence,
        episode=episode,
        tool_cache=cache,
        registry=registry,
        trace={"agent_turns": 1, "tool_observations": []},
        completed_turns=1,
    )


async def _ok(text: str) -> ToolResult:
    return ToolResult(content=text, details={"resource_id": "res-1"})


class TestOwnerExports:
    async def test_evidence_export_preserves_order_and_citation_identity(self) -> None:
        state = await _state_with_evidence()
        state.evidence.add_contexts({"chunks": [_corpus_row("chunk-2", image=False)]})
        exported = state.evidence.export_state()

        restored = EvidenceLedger()
        restored.restore_state(exported)

        assert [row["chunk_id"] for row in restored.contexts["chunks"]] == [
            "chunk-1",
            "chunk-2",
        ]
        assert [row["reference_id"] for row in restored.contexts["chunks"]] == [
            row["reference_id"] for row in state.evidence.contexts["chunks"]
        ]
        assert restored.row_count == state.evidence.row_count

    async def test_evidence_restore_keeps_dedup_identity(self) -> None:
        state = await _state_with_evidence()
        restored = EvidenceLedger()
        restored.restore_state(state.evidence.export_state())

        delta = restored.add_contexts({"chunks": [_corpus_row()]})

        assert not delta.changed
        assert len(restored.contexts["chunks"]) == 1

    async def test_episode_export_round_trips_provider_native_state(self) -> None:
        state = await _state_with_evidence()
        restored = RunEpisode()
        restored.restore_state(state.episode.export_state())

        assert restored.messages() == state.episode.messages()
        assert restored.messages()[0]["provider_state"] == {"reasoning": "native"}
        assert restored.messages()[0]["tool_calls"][0]["thought_signature"] == "sig-1"

    async def test_tool_cache_export_replays_completed_results_only(self) -> None:
        cache = ExactCallCache()
        await cache.run("kb\x00{}", lambda: _ok("found"))
        exported = cache.export_results()

        restored = ExactCallCache()
        restored.restore_results(exported)
        replayed = await restored.run("kb\x00{}", lambda: _ok("should not run"))

        assert replayed.cached is True
        assert replayed.details == {"resource_id": "res-1"}
        await cache.aclose()
        await restored.aclose()

    async def test_registry_export_restores_ids_cursors_and_next_ordinal(self) -> None:
        registry = ResourceRegistry()
        first = registry.register(ResourceInput(content=b"doc-bytes", filename="a.txt"))
        discovered = registry.register_discovered_link("https://example.com/a")
        assert discovered is not None
        ordinal = registry.allocate_fetched_ordinal(discovered)
        exported = registry.export_state()

        resumed = ResourceRegistry()
        replayed = resumed.register(ResourceInput(content=b"doc-bytes", filename="a.txt"))
        assert replayed != first
        resumed.restore_state(exported)

        assert [entry.resource_id for entry in resumed.manifest()] == [first, discovered]
        assert resumed.allocate_fetched_ordinal(discovered) == ordinal
        assert resumed.register_discovered_link("https://example.com/a") == discovered
        next_link = resumed.register_discovered_link("https://example.com/b")
        assert next_link is not None
        assert resumed.allocate_fetched_ordinal(next_link) == ordinal + 1
        await registry.aclose()
        await resumed.aclose()

    async def test_registry_never_reuses_a_checkpointed_fetched_ordinal(self) -> None:
        registry = ResourceRegistry()
        first = registry.register_discovered_link("https://example.com/a")
        assert first is not None
        allocated = registry.allocate_fetched_ordinal(first)
        assert registry.allocate_fetched_ordinal(first) == allocated
        exported = registry.export_state()

        resumed = ResourceRegistry()
        resumed.restore_state(exported)
        later = resumed.register_discovered_link("https://example.com/later")
        assert later is not None

        assert resumed.allocate_fetched_ordinal(later) > allocated
        await registry.aclose()
        await resumed.aclose()

    async def test_registry_rejects_a_catalog_that_lost_a_request_input(self) -> None:
        registry = ResourceRegistry()
        registry.register(ResourceInput(content=b"doc-bytes", filename="a.txt"))
        exported = registry.export_state()

        resumed = ResourceRegistry()
        with pytest.raises(ResourceStateMismatchError):
            resumed.restore_state(exported)
        await registry.aclose()
        await resumed.aclose()


class TestCheckpointCodec:
    async def test_encode_is_versioned_and_json_safe(self) -> None:
        state = await _state_with_evidence()
        store = _FakeStore()

        encoded = await encode_checkpoint_state(state, owner_id="owner", run_id="run", store=store)

        assert encoded["version"] == CHECKPOINT_SCHEMA_VERSION
        assert encoded["completed_turns"] == 1
        json.dumps(encoded, allow_nan=False)

    async def test_round_trip_preserves_evidence_episode_cache_and_resources(self) -> None:
        state = await _state_with_evidence()
        store = _FakeStore()
        encoded = await encode_checkpoint_state(state, owner_id="owner", run_id="run", store=store)

        resumed = AgentRunState(
            evidence=EvidenceLedger(),
            episode=RunEpisode(),
            tool_cache=ExactCallCache(),
            registry=ResourceRegistry(),
            trace={"agent_turns": 0, "tool_observations": []},
        )
        _registry(resumed).register(ResourceInput(content=b"doc-bytes", filename="a.txt"))
        await restore_agent_state(
            resumed,
            encoded,
            owner_id="owner",
            run_id="run",
            store=store,
            expected_completed_turns=1,
        )

        assert resumed.completed_turns == 1
        assert resumed.episode.messages() == state.episode.messages()
        assert [row["chunk_id"] for row in resumed.evidence.contexts["chunks"]] == ["chunk-1"]
        assert resumed.evidence.contexts["entities"] == [{"entity_name": "Alpha"}]
        replayed = await resumed.tool_cache.run("kb\x00{}", lambda: _ok("should not run"))
        assert replayed.cached is True
        await resumed.tool_cache.aclose()
        await _registry(resumed).aclose()

    async def test_corpus_image_is_stored_as_a_reference_not_bytes(self) -> None:
        state = await _state_with_evidence()
        store = _FakeStore()

        encoded = await encode_checkpoint_state(state, owner_id="owner", run_id="run", store=store)

        payload = json.dumps(encoded, ensure_ascii=False)
        assert _PNG not in payload
        row = encoded["state"]["evidence"]["contexts"]["chunks"][0]
        assert row["image_data"] == {
            "kind": "corpus",
            "workspace": "ws-main",
            "chunk_id": "chunk-1",
            "sidecar": "file:///parsed/book",
        }

    async def test_a_base64_shaped_string_without_image_identity_is_left_alone(self) -> None:
        """Only a corpus row or a data URI is an image; nothing else is guessed at."""
        state = _empty_state()
        state.evidence.add_contexts(
            {
                "chunks": [
                    {
                        "chunk_id": "chunk-text",
                        "content": "text only",
                        "reference_id": "ref-b",
                        # A 64-character digest decodes as base64 but is not an image,
                        # and this row carries no workspace identity.
                        "image_data": "a" * 64,
                    }
                ]
            }
        )
        store = _FakeStore()

        encoded = await encode_checkpoint_state(state, owner_id="owner", run_id="run", store=store)

        assert encoded["state"]["evidence"]["contexts"]["chunks"][0]["image_data"] == "a" * 64
        await state.tool_cache.aclose()
        await _registry(state).aclose()

    async def test_a_corpus_rows_source_url_stays_a_link(self) -> None:
        """Corpus identity names the row's visual, not every string beside it."""
        state = _empty_state()
        row = _corpus_row()
        row["url"] = "https://example.com/book.pdf"
        state.evidence.add_contexts({"chunks": [row]})
        store = _FakeStore()

        encoded = await encode_checkpoint_state(state, owner_id="owner", run_id="run", store=store)

        stored = encoded["state"]["evidence"]["contexts"]["chunks"][0]
        assert stored["url"] == "https://example.com/book.pdf"
        assert stored["image_data"]["kind"] == "corpus"
        await state.tool_cache.aclose()
        await _registry(state).aclose()

    async def test_missing_corpus_visual_drops_only_the_image_block(self) -> None:

        state = await _state_with_evidence()
        store = _FakeStore()
        encoded = await encode_checkpoint_state(state, owner_id="owner", run_id="run", store=store)

        resumed = _empty_state()
        _registry(resumed).register(ResourceInput(content=b"doc-bytes", filename="a.txt"))
        await restore_agent_state(
            resumed,
            encoded,
            owner_id="owner",
            run_id="run",
            store=store,
            expected_completed_turns=1,
            load_corpus_image=None,
        )

        row = resumed.evidence.contexts["chunks"][0]
        assert "image_data" not in row
        assert row["content"] == "corpus evidence text"
        assert row["reference_id"]
        await resumed.tool_cache.aclose()
        await _registry(resumed).aclose()

    async def test_corpus_visual_is_rehydrated_when_it_still_resolves(self) -> None:
        state = await _state_with_evidence()
        store = _FakeStore()
        encoded = await encode_checkpoint_state(state, owner_id="owner", run_id="run", store=store)

        async def _load(workspace: str, chunk_id: str) -> str | None:
            assert (workspace, chunk_id) == ("ws-main", "chunk-1")
            return _PNG

        resumed = _empty_state()
        _registry(resumed).register(ResourceInput(content=b"doc-bytes", filename="a.txt"))
        await restore_agent_state(
            resumed,
            encoded,
            owner_id="owner",
            run_id="run",
            store=store,
            expected_completed_turns=1,
            load_corpus_image=_load,
        )

        assert resumed.evidence.contexts["chunks"][0]["image_data"] == _PNG
        await resumed.tool_cache.aclose()
        await _registry(resumed).aclose()

    async def test_attachment_image_becomes_a_digest_ordinal_reference(self) -> None:
        state = _empty_state()
        digest = _sha256(_ATTACHMENT_BYTES)
        state.episode.record(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_ATTACHMENT_B64}"},
                        },
                    ],
                }
            ]
        )
        store = _FakeStore(artifacts={digest: _ATTACHMENT_BYTES})
        store.references.append(
            _Reference(digest=digest, ordinal=2, reference_kind="current_attachment")
        )

        encoded = await encode_checkpoint_state(state, owner_id="owner", run_id="run", store=store)

        payload = json.dumps(encoded, ensure_ascii=False)
        assert _ATTACHMENT_B64 not in payload
        block = encoded["state"]["episode"]["exchanges"][0][0]["content"][1]
        assert block["image_url"]["url"] == {
            "kind": "attachment",
            "digest": digest,
            "ordinal": 2,
            "media_type": "image/png",
        }

        resumed = _empty_state()
        await restore_agent_state(resumed, encoded, owner_id="owner", run_id="run", store=store)
        restored_block = resumed.episode.messages()[0]["content"]
        assert restored_block[0] == {"type": "text", "text": "look"}
        assert restored_block[1]["image_url"]["url"] == (f"data:image/png;base64,{_ATTACHMENT_B64}")
        await resumed.tool_cache.aclose()
        await _registry(resumed).aclose()

    async def test_missing_attachment_bytes_fail_the_run_as_corrupt(self) -> None:
        state = _empty_state()
        digest = _sha256(_ATTACHMENT_BYTES)
        state.episode.record(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_ATTACHMENT_B64}"},
                        }
                    ],
                }
            ]
        )
        store = _FakeStore(artifacts={digest: _ATTACHMENT_BYTES})
        store.references.append(
            _Reference(digest=digest, ordinal=0, reference_kind="current_attachment")
        )
        encoded = await encode_checkpoint_state(state, owner_id="owner", run_id="run", store=store)
        store.artifacts.clear()

        resumed = _empty_state()
        with pytest.raises(CheckpointError) as raised:
            await restore_agent_state(resumed, encoded, owner_id="owner", run_id="run", store=store)

        assert raised.value.kind == "checkpoint_corrupt"
        await resumed.tool_cache.aclose()
        await _registry(resumed).aclose()


class TestCheckpointFailureKinds:
    def test_unreadable_version_is_incompatible(self) -> None:
        with pytest.raises(CheckpointError) as raised:
            decode_checkpoint_state(
                {"version": CHECKPOINT_SCHEMA_VERSION + 1, "completed_turns": 1, "state": {}},
                expected_completed_turns=1,
            )
        assert raised.value.kind == "checkpoint_incompatible"

        with pytest.raises(CheckpointError) as missing:
            decode_checkpoint_state({"completed_turns": 1, "state": {}}, expected_completed_turns=1)
        assert missing.value.kind == "checkpoint_incompatible"

    def test_turn_number_mismatch_is_corrupt(self) -> None:
        with pytest.raises(CheckpointError) as raised:
            decode_checkpoint_state(
                {"version": CHECKPOINT_SCHEMA_VERSION, "completed_turns": 2, "state": {}},
                expected_completed_turns=3,
            )
        assert raised.value.kind == "checkpoint_corrupt"

    def test_unusable_state_shape_is_corrupt(self) -> None:
        with pytest.raises(CheckpointError) as raised:
            decode_checkpoint_state(
                {"version": CHECKPOINT_SCHEMA_VERSION, "completed_turns": 1, "state": []},
                expected_completed_turns=1,
            )
        assert raised.value.kind == "checkpoint_corrupt"

    async def test_oversized_compact_json_is_rejected_before_the_transaction(self) -> None:
        state = _empty_state()
        state.episode.record(
            [{"role": "assistant", "content": "x" * (MAX_CHECKPOINT_BYTES + 1024)}]
        )
        store = _FakeStore()

        with pytest.raises(CheckpointError) as raised:
            await encode_checkpoint_state(state, owner_id="owner", run_id="run", store=store)

        assert raised.value.kind == "checkpoint_too_large"
        await state.tool_cache.aclose()
        await _registry(state).aclose()

    async def test_bound_is_exactly_eight_mebibytes(self) -> None:
        assert MAX_CHECKPOINT_BYTES == 8 * 1024 * 1024


class TestFetchedResourceRestore:
    """Durable web bytes are run state, so a resume reads them, never the network."""

    async def test_restored_fetched_bytes_are_read_without_touching_the_network(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dlightrag.core.resources import registry as registry_module

        state = _empty_state()
        registry = _registry(state)
        resource_id = registry.register_discovered_link("https://example.com/a.html")
        assert resource_id is not None
        ordinal = registry.allocate_fetched_ordinal(resource_id)
        store = _FakeStore(artifacts={_sha256(_FETCHED_BYTES): _FETCHED_BYTES})
        store.references.append(
            _Reference(
                digest=_sha256(_FETCHED_BYTES),
                ordinal=ordinal,
                reference_kind="fetched_resource",
                resource_id=resource_id,
            )
        )
        encoded = await encode_checkpoint_state(state, owner_id="owner", run_id="run", store=store)

        monkeypatch.setattr(registry_module, "avalidate_public_https_url", _no_network)
        monkeypatch.setattr(registry_module, "afetch_public_https_bytes", _no_network)
        resumed = _empty_state()
        await restore_agent_state(resumed, encoded, owner_id="owner", run_id="run", store=store)

        assert await _registry(resumed).materialize(resource_id) == _FETCHED_BYTES
        await state.tool_cache.aclose()
        await resumed.tool_cache.aclose()
        await registry.aclose()
        await _registry(resumed).aclose()

    async def test_a_missing_fetched_blob_is_checkpoint_corrupt(self) -> None:
        state = _empty_state()
        registry = _registry(state)
        resource_id = registry.register_discovered_link("https://example.com/a.html")
        assert resource_id is not None
        ordinal = registry.allocate_fetched_ordinal(resource_id)
        store = _FakeStore(artifacts={_sha256(_FETCHED_BYTES): _FETCHED_BYTES})
        store.references.append(
            _Reference(
                digest=_sha256(_FETCHED_BYTES),
                ordinal=ordinal,
                reference_kind="fetched_resource",
                resource_id=resource_id,
            )
        )
        encoded = await encode_checkpoint_state(state, owner_id="owner", run_id="run", store=store)
        store.artifacts.clear()

        resumed = _empty_state()
        with pytest.raises(CheckpointError) as raised:
            await restore_agent_state(resumed, encoded, owner_id="owner", run_id="run", store=store)

        assert raised.value.kind == "checkpoint_corrupt"
        await state.tool_cache.aclose()
        await resumed.tool_cache.aclose()
        await registry.aclose()
        await _registry(resumed).aclose()


async def _no_network(url: str, **kwargs: Any) -> bytes:
    raise AssertionError(f"a restored run must not reach the network for {url}")


def _empty_state() -> AgentRunState:
    return AgentRunState(
        evidence=EvidenceLedger(),
        episode=RunEpisode(),
        tool_cache=ExactCallCache(),
        registry=ResourceRegistry(),
        trace={"agent_turns": 0, "tool_observations": []},
    )


def _registry(state: AgentRunState) -> ResourceRegistry:
    registry = state.registry
    assert registry is not None
    return registry


def _sha256(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


class _Reference:
    def __init__(
        self,
        *,
        digest: str,
        ordinal: int,
        reference_kind: str,
        resource_id: str = "",
    ) -> None:
        self.digest = digest
        self.ordinal = ordinal
        self.reference_kind = reference_kind
        self.resource_id = resource_id
