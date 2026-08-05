from dlightrag.citations.indexer import CitationIndexer, build_citation_index


def test_build_index_basic():
    indexer = CitationIndexer()
    contexts = [
        {"chunk_id": "c1", "reference_id": "1", "content": "text1"},
        {"chunk_id": "c2", "reference_id": "1", "content": "text2"},
        {"chunk_id": "c3", "reference_id": "2", "content": "text3"},
    ]
    indexer.build_index(contexts)
    assert indexer.get_chunk_idx("1", "c1") == 1
    assert indexer.get_chunk_idx("1", "c2") == 2
    assert indexer.get_chunk_idx("2", "c3") == 1


def test_reverse_lookup():
    indexer = CitationIndexer()
    indexer.build_index(
        [
            {"chunk_id": "c1", "reference_id": "1", "content": "text"},
        ]
    )
    assert indexer.get_chunk_id("1", 1) == "c1"
    assert indexer.get_chunk_id("1", 99) is None
    assert indexer.get_chunk_id("999", 1) is None


def test_entity_with_source_id():
    indexer = CitationIndexer()
    contexts = [
        {"chunk_id": "c1", "reference_id": "1", "content": "text"},
        {"chunk_id": "c2", "reference_id": "1", "content": "text2"},
        {"source_id": "c1, c2", "reference_id": "1", "description": "entity"},
    ]
    indexer.build_index(contexts)
    assert indexer.get_chunk_idx("1", "c1") == 1
    assert indexer.get_chunk_idx("1", "c2") == 2
    assert indexer.get_max_chunk_idx("1") == 2


def test_inject_chunk_idx():
    indexer = CitationIndexer()
    contexts = [
        {"chunk_id": "c1", "reference_id": "1", "content": "text"},
    ]
    indexer.build_index(contexts)
    enriched = indexer.inject_chunk_idx(contexts)
    assert enriched[0]["chunk_idx"] == 1


def test_build_citation_index_convenience():
    contexts = [
        {"chunk_id": "c1", "reference_id": "1", "content": "text"},
    ]
    indexer, enriched = build_citation_index(contexts)
    assert indexer.get_chunk_id("1", 1) == "c1"
    assert enriched[0]["chunk_idx"] == 1


def test_get_max_chunk_idx():
    indexer = CitationIndexer()
    indexer.build_index(
        [
            {"chunk_id": "c1", "reference_id": "1", "content": "a"},
            {"chunk_id": "c2", "reference_id": "1", "content": "b"},
            {"chunk_id": "c3", "reference_id": "1", "content": "c"},
        ]
    )
    assert indexer.get_max_chunk_idx("1") == 3
    assert indexer.get_max_chunk_idx("999") == 0


def test_doc_workspace_provenance_is_rebuilt_without_leaking_prior_index_state():
    indexer = CitationIndexer()
    indexer.build_index(
        [
            {
                "chunk_id": "c1",
                "reference_id": "1",
                "content": "current",
                "file_path": "/docs/report.pdf",
                "_workspace": "personnel",
            },
            {
                "chunk_id": "c2",
                "reference_id": "2",
                "content": "legacy",
                "file_path": "/docs/legacy.pdf",
            },
        ]
    )

    assert indexer.get_doc_workspace("1") == "personnel"
    assert indexer.get_doc_workspace("2") is None

    indexer.build_index(
        [
            {
                "chunk_id": "c3",
                "reference_id": "1",
                "content": "rebuilt",
                "file_path": "/docs/rebuilt.pdf",
                "_workspace": "research",
            }
        ]
    )

    assert indexer.get_doc_workspace("1") == "research"
    assert indexer.get_doc_workspace("2") is None
    assert indexer.get_chunk_id("2", 1) is None

    indexer.build_index(
        [
            {
                "chunk_id": "c4",
                "reference_id": "1",
                "content": "rebuilt legacy",
                "file_path": "/docs/rebuilt-legacy.pdf",
            }
        ]
    )

    assert indexer.get_doc_workspace("1") is None


def test_same_chunk_id_in_two_workspaces_resolves_reference_locally() -> None:
    indexer = CitationIndexer()
    indexer.build_index(
        [
            {
                "chunk_id": "shared-hash",
                "reference_id": "1",
                "content": "Legal evidence",
                "file_path": "/legal/report.pdf",
                "page_number": 3,
                "_workspace": "legal",
            },
            {
                "chunk_id": "shared-hash",
                "reference_id": "2",
                "content": "Finance evidence",
                "file_path": "/finance/report.pdf",
                "page_number": 9,
                "_workspace": "finance",
            },
        ]
    )

    assert indexer.get_chunk_id("1", 1) == "shared-hash"
    assert indexer.get_chunk_id("2", 1) == "shared-hash"
    assert indexer.get_doc_workspace("1") == "legal"
    assert indexer.get_doc_workspace("2") == "finance"
