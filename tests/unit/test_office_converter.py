# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for LibreOffice converter."""

from __future__ import annotations

from pathlib import Path

from dlightrag.config import DlightragConfig
from dlightrag.converters.office import LibreOfficeConverter


class TestLibreOfficeConverter:
    """Test converter configuration and should_convert logic."""

    def _make_converter(self, test_config: DlightragConfig) -> LibreOfficeConverter:
        return LibreOfficeConverter(test_config)

    def test_should_convert_xlsx(self, test_config: DlightragConfig) -> None:
        """Test that .xlsx triggers conversion."""
        converter = self._make_converter(test_config)
        assert converter.should_convert(Path("test.xlsx"))
        assert converter.should_convert(Path("test.xls"))

    def test_should_not_convert_pdf(self, test_config: DlightragConfig) -> None:
        """Test that .pdf does not trigger conversion."""
        converter = self._make_converter(test_config)
        assert not converter.should_convert(Path("test.pdf"))
        assert not converter.should_convert(Path("test.docx"))

    def test_should_skip_csv(self, test_config: DlightragConfig) -> None:
        """Test that .csv is skipped."""
        converter = self._make_converter(test_config)
        assert not converter.should_convert(Path("test.csv"))

    def test_respects_config_flag(self, tmp_path: Path) -> None:
        """Test that excel_auto_convert_to_pdf=False disables conversion."""
        config = DlightragConfig(  # type: ignore[call-arg]
            working_dir=str(tmp_path),
            openai_api_key="test-key",
            excel_auto_convert_to_pdf=False,
            kv_storage="JsonKVStorage",
            doc_status_storage="JsonDocStatusStorage",
            vector_storage="NanoVectorDBStorage",
            graph_storage="NetworkXStorage",
        )
        converter = LibreOfficeConverter(config)
        assert not converter.should_convert(Path("test.xlsx"))

    def test_docling_parser_skips_conversion(self, tmp_path: Path) -> None:
        """Test that docling parser disables Excel conversion."""
        config = DlightragConfig(  # type: ignore[call-arg]
            working_dir=str(tmp_path),
            openai_api_key="test-key",
            parser="docling",
            kv_storage="JsonKVStorage",
            doc_status_storage="JsonDocStatusStorage",
            vector_storage="NanoVectorDBStorage",
            graph_storage="NetworkXStorage",
        )
        converter = LibreOfficeConverter(config)
        assert not converter.should_convert(Path("test.xlsx"))

    def test_is_safe_to_delete(self, test_config: DlightragConfig) -> None:
        """Test safety check for file deletion."""
        converter = self._make_converter(test_config)

        # File within storage is safe
        safe_path = test_config.working_dir_path / "sources" / "test.xlsx"
        assert converter._is_safe_to_delete(safe_path)

        # File outside storage is not safe
        assert not converter._is_safe_to_delete(Path("/tmp/outside.xlsx"))


import openpyxl

from dlightrag.converters.office import PageSetup


class TestEstimateExcelPageWidth:
    """Test adaptive page width estimation from Excel column widths."""

    def _make_converter(self, test_config):
        return LibreOfficeConverter(test_config)

    def _make_xlsx(self, tmp_path, columns, sheet_configs=None):
        """Helper: create xlsx with specified column widths."""
        wb = openpyxl.Workbook()
        if sheet_configs:
            for i, cfg in enumerate(sheet_configs):
                ws = wb.active if i == 0 else wb.create_sheet(cfg["name"])
                if i == 0:
                    ws.title = cfg["name"]
                for col_letter, width in cfg["columns"]:
                    ws.column_dimensions[col_letter].width = width
                    ws.cell(row=1, column=openpyxl.utils.column_index_from_string(col_letter), value="data")
        else:
            ws = wb.active
            for col_letter, width in columns:
                ws.column_dimensions[col_letter].width = width
                ws.cell(row=1, column=openpyxl.utils.column_index_from_string(col_letter), value="data")
        path = tmp_path / "test.xlsx"
        wb.save(str(path))
        wb.close()
        return path

    def test_narrow_excel_gives_portrait(self, test_config, tmp_path):
        """3 narrow columns → total ~6cm + margins → clamped to 21cm → portrait."""
        path = self._make_xlsx(tmp_path, [("A", 10), ("B", 10), ("C", 10)])
        converter = self._make_converter(test_config)
        setup = converter._estimate_excel_page_width(path)
        assert isinstance(setup, PageSetup)
        assert setup.orientation == "portrait"
        assert setup.width_cm == 21.0
        assert setup.height_cm == 29.7

    def test_wide_excel_gives_landscape(self, test_config, tmp_path):
        """20 columns at width 15 → ~60cm + margins → landscape."""
        cols = [(openpyxl.utils.get_column_letter(i + 1), 15) for i in range(20)]
        path = self._make_xlsx(tmp_path, cols)
        converter = self._make_converter(test_config)
        setup = converter._estimate_excel_page_width(path)
        assert setup.orientation == "landscape"
        assert setup.width_cm > 21.0
        assert setup.width_cm <= 63.0
        assert setup.height_cm == 21.0

    def test_very_wide_clamped_to_max(self, test_config, tmp_path):
        """50 columns at width 20 → way over 63cm → clamped to 63."""
        cols = [(openpyxl.utils.get_column_letter(i + 1), 20) for i in range(50)]
        path = self._make_xlsx(tmp_path, cols)
        converter = self._make_converter(test_config)
        setup = converter._estimate_excel_page_width(path)
        assert setup.width_cm == 63.0

    def test_hidden_columns_excluded(self, test_config, tmp_path):
        """Hidden columns should not contribute to width."""
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.column_dimensions["A"].width = 10
        ws.column_dimensions["B"].width = 100
        ws.column_dimensions["B"].hidden = True
        ws.column_dimensions["C"].width = 10
        ws.cell(row=1, column=1, value="a")
        ws.cell(row=1, column=2, value="b")
        ws.cell(row=1, column=3, value="c")
        path = tmp_path / "hidden.xlsx"
        wb.save(str(path))
        wb.close()

        converter = self._make_converter(test_config)
        setup = converter._estimate_excel_page_width(path)
        assert setup.width_cm == 21.0
        assert setup.orientation == "portrait"

    def test_multi_sheet_uses_widest(self, test_config, tmp_path):
        """Width estimation should use the widest sheet."""
        path = self._make_xlsx(tmp_path, [], sheet_configs=[
            {"name": "Narrow", "columns": [("A", 10), ("B", 10)]},
            {"name": "Wide", "columns": [(openpyxl.utils.get_column_letter(i+1), 15) for i in range(20)]},
        ])
        converter = self._make_converter(test_config)
        setup = converter._estimate_excel_page_width(path)
        assert setup.width_cm > 21.0
        assert setup.orientation == "landscape"

    def test_xls_fallback(self, test_config, tmp_path):
        """xls files fall back to A4 landscape."""
        path = tmp_path / "old.xls"
        path.write_bytes(b"fake xls content")
        converter = self._make_converter(test_config)
        setup = converter._estimate_excel_page_width(path)
        assert setup.width_cm == 29.7
        assert setup.height_cm == 21.0
        assert setup.orientation == "landscape"

    def test_corrupt_file_fallback(self, test_config, tmp_path):
        """Corrupt xlsx falls back to A4 landscape."""
        path = tmp_path / "corrupt.xlsx"
        path.write_bytes(b"not a real xlsx")
        converter = self._make_converter(test_config)
        setup = converter._estimate_excel_page_width(path)
        assert setup.width_cm == 29.7
        assert setup.height_cm == 21.0
        assert setup.orientation == "landscape"
