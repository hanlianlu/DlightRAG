from dlightrag.citations.utils import split_source_ids, filter_content_for_display


def test_split_source_ids_comma_separated():
    assert split_source_ids("chunk1, chunk2, chunk3") == ["chunk1", "chunk2", "chunk3"]


def test_split_source_ids_none():
    assert split_source_ids(None) == []


def test_split_source_ids_empty():
    assert split_source_ids("") == []


def test_split_source_ids_single():
    assert split_source_ids("chunk1") == ["chunk1"]


def test_split_source_ids_whitespace():
    assert split_source_ids("  chunk1 ,  , chunk2  ") == ["chunk1", "chunk2"]


def test_filter_content_basic():
    content = "This is normal text.\nImage Path: /foo/bar.png\nMore text."
    result = filter_content_for_display(content)
    assert "Image Path:" not in result
    assert "normal text" in result
    assert "More text" in result


def test_filter_content_preserves_tables():
    content = "| Col1 | Col2 |\n|------|------|\n| A    | B    |"
    result = filter_content_for_display(content)
    assert "| Col1 | Col2 |" in result


def test_filter_content_truncate():
    content = "A" * 200
    result = filter_content_for_display(content, max_chars=50)
    assert len(result) <= 53  # 50 + "..."


def test_filter_content_removes_caption_none():
    content = "Text here.\nCaption: None\nMore text."
    result = filter_content_for_display(content)
    assert "Caption: None" not in result
    assert "Text here" in result
