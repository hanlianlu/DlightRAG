# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Release metadata and removed Agent names stay synchronized."""

from scripts.verify_release_contract import ROOT, verify_repository


def test_repository_release_contract() -> None:
    verify_repository(ROOT)
