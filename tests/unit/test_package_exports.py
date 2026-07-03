# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Package-level SDK export contract."""

from __future__ import annotations

import pytest


def test_package_root_exports_manager_not_runtime_service() -> None:
    import dlightrag

    def read_runtime_service() -> object:
        return dlightrag.RAGService

    assert dlightrag.RAGServiceManager.__name__ == "RAGServiceManager"
    assert dlightrag.RetrievalResult.__name__ == "RetrievalResult"
    assert dlightrag.IngestSpec.__name__ == "IngestSpec"
    assert "RAGServiceManager" in dlightrag.__all__
    assert "RetrievalResult" in dlightrag.__all__
    with pytest.raises(AttributeError):
        read_runtime_service()
