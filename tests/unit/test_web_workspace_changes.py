# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The workspace Inspector's snapshot diff: one helper serves both reads.

The snapshot list reports each snapshot's counts and the snapshot detail reports each
file's marker, and the two must agree about the same snapshot because they project through
one helper. These tests pin its vocabulary to the wire values and pin the one case that is
a product decision rather than arithmetic: a parent snapshot the retained history no longer
holds is read as an empty parent, so a pruned ancestor cannot make a kept snapshot claim a
removal it cannot show.
"""

from __future__ import annotations

import json
from typing import Any

from dlightrag.adapters.postgres.web.web_conversations import PGWebConversationStore

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64
_E = "e" * 64


def _manifest(*files: tuple[str, str, int]) -> str:
    """One snapshot's stored manifest_json, exactly as the JSONB column hands it back."""
    return json.dumps(
        [{"path": path, "digest": digest, "size": size} for path, digest, size in files]
    )


def _diff(manifest: Any, parent_manifest: Any) -> tuple[Any, Any]:
    return PGWebConversationStore._manifest_diff(  # pyright: ignore[reportPrivateUsage]
        manifest, parent_manifest
    )


def test_a_snapshot_marks_and_counts_each_kind_of_change() -> None:
    files, changes = _diff(
        _manifest(
            ("kept.md", _A, 1),
            ("edited.md", _B, 2),
            ("new.md", _C, 3),
        ),
        _manifest(
            ("kept.md", _A, 1),
            ("edited.md", _D, 2),
            ("dropped.md", _E, 4),
        ),
    )

    assert [(file.relative_path, file.change) for file in files] == [
        ("kept.md", "unchanged"),
        ("edited.md", "changed"),
        ("new.md", "added"),
    ]
    assert (changes.added, changes.changed, changes.removed) == (1, 1, 1)


def test_a_snapshot_without_a_parent_reads_every_file_as_added() -> None:
    removed_parent: Any = None
    files, changes = _diff(_manifest(("new.md", _C, 3), ("also-new.md", _D, 4)), removed_parent)

    assert [file.change for file in files] == ["added", "added"]
    assert (changes.added, changes.changed, changes.removed) == (2, 0, 0)


def test_a_parent_pruned_from_history_reads_as_no_parent_rather_than_as_removals() -> None:
    # The child still names its parent's run, but the row is gone. Reading the absence as
    # an empty parent is what keeps a kept snapshot diffsable; reading it as "everything
    # removed" would report a loss the retained history does not contain.
    child = _manifest(("new.md", _C, 3))
    pruned_parent: Any = None

    files, changes = _diff(child, pruned_parent)

    assert [file.change for file in files] == ["added"]
    assert (changes.added, changes.changed, changes.removed) == (1, 0, 0)


def test_identical_manifests_report_no_change() -> None:
    manifest = _manifest(("kept.md", _A, 1), ("also-kept.md", _B, 2))

    files, changes = _diff(manifest, manifest)

    assert [file.change for file in files] == ["unchanged", "unchanged"]
    assert (changes.added, changes.changed, changes.removed) == (0, 0, 0)


def test_a_file_the_snapshot_dropped_is_counted_only_as_removed() -> None:
    files, changes = _diff(
        _manifest(("kept.md", _A, 1)),
        _manifest(("kept.md", _A, 1), ("dropped.md", _D, 4), ("also-dropped.md", _E, 5)),
    )

    assert [file.relative_path for file in files] == ["kept.md"]
    assert (changes.added, changes.changed, changes.removed) == (0, 0, 2)


def test_both_stored_jsonb_forms_are_read_the_same_way() -> None:
    # asyncpg hands JSONB back as text; a decoded manifest arrives as a list. A read that
    # mixes the two (a page row beside a follow-up parent row) must not split the diff.
    as_text = _manifest(("kept.md", _A, 1))
    as_decoded = json.loads(as_text)

    text_files, text_changes = _diff(as_text, as_text)
    decoded_files, decoded_changes = _diff(as_decoded, as_decoded)

    assert [(file.relative_path, file.change) for file in text_files] == [
        (file.relative_path, file.change) for file in decoded_files
    ]
    assert [(file.relative_path, file.change) for file in text_files] == [("kept.md", "unchanged")]
    assert text_changes == decoded_changes
