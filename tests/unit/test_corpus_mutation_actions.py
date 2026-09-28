# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One list of Corpus Mutation actions, and every consumer keyed by it."""

from typing import get_args

from dlightrag.application.access.control import CORPUS_MUTATION_ACCESS_ACTIONS
from dlightrag.application.corpus_admin import mutations


def test_the_action_spec_covers_exactly_the_declared_actions() -> None:
    declared = set(get_args(mutations.CorpusMutationAction.__value__))

    assert set(mutations._ACTIONS) == declared


def test_every_action_has_its_own_authorization_boundary() -> None:
    # Access may not import Corpus Administration, so the two lists meet here.
    assert set(CORPUS_MUTATION_ACCESS_ACTIONS) == set(mutations._ACTIONS)


def test_only_ingest_is_safe_to_repeat_after_a_recovered_handoff() -> None:
    assert frozenset(mutations._ACTIONS) - mutations._DESTRUCTIVE_ACTIONS == {"ingest"}
