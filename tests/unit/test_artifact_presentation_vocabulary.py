# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One Artifact presentation vocabulary across every consumer.

A capability is named in six places that cannot share a constant: publication
owns the engine vocabulary, the durable runtime settlement may not import
Answer (`runtime-independence`), the storage CHECK is SQL, and the frontend
union is TypeScript. A slice that added a capability to some of them and not
others settles an Attachment the storage rejects or renders an Artifact no
client accepts, so the set is locked here instead of being trusted.
"""

import re
from pathlib import Path
from typing import get_args

import pytest

from dlightrag.adapters.http.browser.presentation import PresentationArtifact
from dlightrag.adapters.http.rest.models import AnswerArtifactResponse
from dlightrag.adapters.postgres.runtime import run_store
from dlightrag.engine.answer.publication import PresentationCapability
from dlightrag.engine.runtime.settlements import ArtifactAttachmentUpdate

_REPO = Path(__file__).resolve().parents[2]
_CONVERSATIONS = _REPO / "frontend/api/conversations.ts"
_CAPABILITIES = set(get_args(PresentationCapability))
_CHECK_NAME = "dlightrag_answer_artifact_attachments_presentation_check"


def _admitted_check_values(sql: str) -> set[str]:
    """Return the values one presentation CHECK statement admits."""
    (clause,) = re.findall(rf"{_CHECK_NAME}\s+CHECK\s*\(presentation IN\s*\(([^)]*)\)\)", sql)
    return set(re.findall(r"'([a-z_]+)'", clause))


def test_publication_is_the_canonical_vocabulary() -> None:
    assert _CAPABILITIES == {
        "image",
        "video",
        "markdown",
        "html",
        "pdf",
        "text",
        "download",
    }


def test_durable_settlement_admits_every_capability() -> None:
    for capability in sorted(_CAPABILITIES):
        settlement = ArtifactAttachmentUpdate(
            relative_path="artifacts/report.mp4",
            label="Report",
            content_digest="a" * 64,
            presentation=capability,
            size_bytes=1,
            session_id="session",
            intent_id="intent",
        )

        assert settlement.presentation == capability


def test_durable_settlement_refuses_an_unknown_capability() -> None:
    with pytest.raises(ValueError, match="presentation is invalid"):
        ArtifactAttachmentUpdate(
            relative_path="artifacts/report.mp4",
            label="Report",
            content_digest="a" * 64,
            presentation="stream",
            size_bytes=1,
            session_id="session",
            intent_id="intent",
        )


def test_fresh_schema_admits_every_capability() -> None:
    assert _admitted_check_values(run_store._CREATE_ARTIFACT_ATTACHMENTS) == _CAPABILITIES


def test_latest_migration_admits_every_capability() -> None:
    """An existing database gets the vocabulary its writer already uses."""
    attempts = [
        statement
        for migration in run_store.RUN_MIGRATIONS
        for statement in migration.statements
        if _CHECK_NAME in statement
    ]

    assert attempts, "no migration constrains the Artifact presentation vocabulary"
    assert _admitted_check_values(attempts[-1]) == _CAPABILITIES


@pytest.mark.parametrize(
    "model",
    [AnswerArtifactResponse, PresentationArtifact],
    ids=["rest", "browser"],
)
def test_transport_contracts_admit_every_capability(model: type) -> None:
    annotation = model.model_fields["presentation"].annotation

    assert set(get_args(annotation)) == _CAPABILITIES


def test_frontend_union_admits_every_capability() -> None:
    source = _CONVERSATIONS.read_text(encoding="utf-8")
    (picklist,) = re.findall(
        r"presentation: v\.picklist\(\[([^\]]*)\]\)",
        source,
    )

    assert set(re.findall(r"'([a-z_]+)'", picklist)) == _CAPABILITIES
