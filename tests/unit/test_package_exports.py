# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Package-level SDK export contract."""


def test_package_root_exports_public_sdk() -> None:
    import dlightrag

    assert dlightrag.RAGServiceManager.__name__ == "RAGServiceManager"
    assert dlightrag.RetrievalResult.__name__ == "RetrievalResult"
    assert dlightrag.IngestSpec.__name__ == "IngestSpec"
