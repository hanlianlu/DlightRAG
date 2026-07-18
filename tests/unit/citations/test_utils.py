from dlightrag.citations.utils import filter_content_for_display, split_source_ids


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


def test_filter_content_wraps_equation_math():
    content = "\\frac{M}{P} = Y L(i)\n[Equation Name]money_demand\n\nThis is the money demand."
    result = filter_content_for_display(content)
    assert "$$\n\\frac{M}{P} = Y L(i)\n$$" in result
    assert "[Equation Name]" not in result
    assert "This is the money demand." in result


def test_filter_content_equation_not_double_wrapped():
    content = "$x = 1$\n[Equation Name]trivial\n\nDescription."
    result = filter_content_for_display(content)
    assert "$$" not in result
    assert "$x = 1$" in result
    assert "[Equation Name]" not in result


def test_filter_content_strips_image_labels():
    content = "[Image Name]money_shock\n[Image Type]Chart\n\nThis chart shows a shift."
    result = filter_content_for_display(content)
    assert "[Image Name]" not in result
    assert "[Image Type]" not in result
    assert result == "This chart shows a shift."


def test_filter_content_strips_table_label():
    content = "[Table Name]crisis_sim\n\nThis table simulates a crisis."
    result = filter_content_for_display(content)
    assert "[Table Name]" not in result
    assert result == "This table simulates a crisis."


def test_filter_content_plain_text_passthrough():
    content = "Just some ordinary prose with no markers."
    assert filter_content_for_display(content) == content


def test_filter_content_preserves_tables():
    content = "| Col1 | Col2 |\n|------|------|\n| A    | B    |"
    result = filter_content_for_display(content)
    assert "| Col1 | Col2 |" in result


def test_filter_content_truncate():
    content = "A" * 200
    result = filter_content_for_display(content, max_chars=50)
    assert len(result) <= 53  # 50 + "..."
