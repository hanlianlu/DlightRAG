# Adaptive Excel Page Width & OfficeConverter Unification — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fixed A4 page width with adaptive width based on Excel column data, and unify Office→PDF conversion so PageRenderer reuses LibreOfficeConverter.

**Architecture:** `LibreOfficeConverter` gains adaptive width estimation via openpyxl and a new `file_to_pdf_bytes()` method. `PageRenderer` accepts an optional converter and delegates `_render_office` to it. `UnifiedRepresentEngine` wires the two together.

**Tech Stack:** openpyxl, LibreOffice (headless), pypdfium2, Python dataclasses

**Spec:** `docs/superpowers/specs/2026-03-11-adaptive-excel-page-width-design.md`

---

## Chunk 1: Core — PageSetup and Adaptive Width Estimation

### Task 1: Add openpyxl dependency

**Files:**
- Modify: `pyproject.toml:11-30` (dependencies list)

- [ ] **Step 1: Add openpyxl to dependencies**

In `pyproject.toml`, add `"openpyxl>=3.1"` to the `dependencies` list after the existing entries:

```toml
    "openpyxl>=3.1",
```

- [ ] **Step 2: Install and verify**

Run: `uv sync`
Expected: clean install with openpyxl resolved.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add openpyxl dependency for Excel column width reading"
```

---

### Task 2: Add PageSetup dataclass and _estimate_excel_page_width

**Files:**
- Modify: `src/dlightrag/converters/office.py:1-30` (imports + new dataclass)
- Modify: `src/dlightrag/converters/office.py:35-47` (new method on LibreOfficeConverter)
- Test: `tests/unit/test_office_converter.py`

- [ ] **Step 1: Write failing tests for _estimate_excel_page_width**

Add a new test class `TestEstimateExcelPageWidth` to `tests/unit/test_office_converter.py`:

```python
import openpyxl
from dlightrag.converters.office import LibreOfficeConverter, PageSetup


class TestEstimateExcelPageWidth:
    """Test adaptive page width estimation from Excel column widths."""

    def _make_converter(self, test_config):
        return LibreOfficeConverter(test_config)

    def _make_xlsx(self, tmp_path, columns, sheet_configs=None):
        """Helper: create xlsx with specified column widths.

        Args:
            columns: list of (letter, width) for single-sheet mode.
            sheet_configs: list of dicts with 'name' and 'columns' keys
                for multi-sheet mode. Overrides `columns` if provided.
        """
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
        ws.column_dimensions["B"].width = 100  # very wide but hidden
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
        # Only A+C contribute: ~20 units * 0.201 + 1.0 margin ≈ 5cm → clamped to 21
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
        # Should be based on the Wide sheet
        assert setup.width_cm > 21.0
        assert setup.orientation == "landscape"

    def test_xls_fallback(self, test_config, tmp_path):
        """xls files fall back to A4 landscape (openpyxl can't read them)."""
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_office_converter.py::TestEstimateExcelPageWidth -v`
Expected: FAIL — `PageSetup` not defined, `_estimate_excel_page_width` not defined.

- [ ] **Step 3: Implement PageSetup and _estimate_excel_page_width**

In `src/dlightrag/converters/office.py`, add import and dataclass at the top (after existing imports):

```python
from dataclasses import dataclass

# ... existing constants ...

# Conversion factor: 1 Excel width unit ≈ 0.201cm
# Based on Excel's default Calibri 11pt character width (7.59px at 96 DPI).
_EXCEL_UNIT_TO_CM = 0.201
_DEFAULT_COLUMN_WIDTH = 8.43  # Excel default
_PAGE_MARGIN_CM = 0.5
_MIN_PAGE_WIDTH_CM = 21.0   # A4 portrait width
_MAX_PAGE_WIDTH_CM = 63.0   # ~3x A4 landscape
_A4_PORTRAIT_HEIGHT_CM = 29.7
_A4_LANDSCAPE_HEIGHT_CM = 21.0


@dataclass
class PageSetup:
    """Page dimensions for PDF conversion."""
    width_cm: float
    height_cm: float
    orientation: str  # "landscape" | "portrait"
```

Then add the method to `LibreOfficeConverter`:

```python
    def _estimate_excel_page_width(self, source_path: Path) -> PageSetup:
        """Estimate optimal page width from Excel column widths.

        Reads column dimensions from all sheets via openpyxl, sums visible
        column widths, and returns a PageSetup with adaptive dimensions.
        Falls back to A4 landscape on any failure.
        """
        fallback = PageSetup(
            width_cm=29.7, height_cm=_A4_LANDSCAPE_HEIGHT_CM, orientation="landscape"
        )
        try:
            import openpyxl  # noqa: PLC0415
            from openpyxl.utils import get_column_letter  # noqa: PLC0415
        except ImportError:
            logger.warning("openpyxl not installed, using A4 landscape fallback")
            return fallback

        try:
            wb = openpyxl.load_workbook(str(source_path), data_only=True)
        except Exception:
            logger.warning("Cannot read %s with openpyxl, using A4 landscape fallback", source_path.name)
            return fallback

        try:
            max_width_cm = 0.0

            for ws in wb.worksheets:
                max_col = ws.max_column or 0
                if max_col == 0:
                    continue

                sheet_width = 0.0
                for col_idx in range(1, max_col + 1):
                    col_letter = get_column_letter(col_idx)
                    dim = ws.column_dimensions.get(col_letter)
                    if dim and dim.hidden:
                        continue
                    width = (dim.width if dim and dim.width else _DEFAULT_COLUMN_WIDTH)
                    sheet_width += width

                sheet_width_cm = sheet_width * _EXCEL_UNIT_TO_CM
                max_width_cm = max(max_width_cm, sheet_width_cm)

            if max_width_cm == 0.0:
                return fallback

            # Add margins (left + right)
            total_width_cm = max_width_cm + 2 * _PAGE_MARGIN_CM

            # Clamp
            total_width_cm = max(_MIN_PAGE_WIDTH_CM, min(_MAX_PAGE_WIDTH_CM, total_width_cm))

            # Orientation
            if total_width_cm <= _MIN_PAGE_WIDTH_CM:
                setup = PageSetup(
                    width_cm=_MIN_PAGE_WIDTH_CM,
                    height_cm=_A4_PORTRAIT_HEIGHT_CM,
                    orientation="portrait",
                )
            else:
                setup = PageSetup(
                    width_cm=total_width_cm,
                    height_cm=_A4_LANDSCAPE_HEIGHT_CM,
                    orientation="landscape",
                )

            logger.info(
                "Excel page setup for %s: %.1fcm x %.1fcm (%s)",
                source_path.name,
                setup.width_cm,
                setup.height_cm,
                setup.orientation,
            )
            return setup
        finally:
            wb.close()
```

Also add `PageSetup` to `__all__`:

```python
__all__ = [
    "LibreOfficeConverter",
    "OfficeConverterError",
    "PageSetup",
    ...
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_office_converter.py::TestEstimateExcelPageWidth -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dlightrag/converters/office.py tests/unit/test_office_converter.py
git commit -m "feat: add PageSetup dataclass and adaptive Excel width estimation"
```

---

### Task 3: Replace _set_ods_landscape_fit with parameterized _set_ods_page_setup

**Files:**
- Modify: `src/dlightrag/converters/office.py:185-233` (rename + parameterize)
- Modify: `src/dlightrag/converters/office.py:152-153` (call site in _convert_excel_to_pdf)
- Test: `tests/unit/test_office_converter.py`

- [ ] **Step 1: Write failing test for _set_ods_page_setup**

Add to `tests/unit/test_office_converter.py`:

```python
import xml.etree.ElementTree as ET
import zipfile


class TestSetOdsPageSetup:
    """Test ODS XML page setup modification."""

    def _make_converter(self, test_config):
        return LibreOfficeConverter(test_config)

    def _make_minimal_ods(self, tmp_path):
        """Create a minimal ODS file with styles.xml for testing."""
        ods_path = tmp_path / "test.ods"
        extract_dir = tmp_path / "ods_build"
        extract_dir.mkdir()

        # Write mimetype
        (extract_dir / "mimetype").write_text("application/vnd.oasis.opendocument.spreadsheet")

        # Write minimal styles.xml with a page-layout
        styles_xml = extract_dir / "styles.xml"
        styles_xml.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<office:document-styles '
            'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
            'xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" '
            'xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0">'
            '<office:automatic-styles>'
            '<style:page-layout style:name="pm1">'
            '<style:page-layout-properties fo:page-width="21cm" fo:page-height="29.7cm" '
            'style:print-orientation="portrait"/>'
            '</style:page-layout>'
            '</office:automatic-styles>'
            '</office:document-styles>'
        )

        with zipfile.ZipFile(ods_path, "w") as zf:
            zf.write(extract_dir / "mimetype", "mimetype", compress_type=zipfile.ZIP_STORED)
            zf.write(styles_xml, "styles.xml", compress_type=zipfile.ZIP_DEFLATED)

        return ods_path

    def test_sets_custom_dimensions(self, test_config, tmp_path):
        """Verify page width, height, orientation, margins are written to ODS."""
        ods_path = self._make_minimal_ods(tmp_path)
        converter = self._make_converter(test_config)
        setup = PageSetup(width_cm=45.0, height_cm=21.0, orientation="landscape")

        converter._set_ods_page_setup(ods_path, setup)

        # Extract and verify
        extract_dir = tmp_path / "verify"
        with zipfile.ZipFile(ods_path, "r") as zf:
            zf.extractall(extract_dir)

        tree = ET.parse(extract_dir / "styles.xml")
        ns = {
            "style": "urn:oasis:names:tc:opendocument:xmlns:style:1.0",
            "fo": "urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0",
        }
        props = tree.getroot().find(".//style:page-layout-properties", ns)
        assert props is not None

        fo = "urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
        style = "urn:oasis:names:tc:opendocument:xmlns:style:1.0"

        assert props.get(f"{{{fo}}}page-width") == "45.0cm"
        assert props.get(f"{{{fo}}}page-height") == "21.0cm"
        assert props.get(f"{{{style}}}print-orientation") == "landscape"
        assert props.get(f"{{{fo}}}margin-left") == "0.5cm"
        assert props.get(f"{{{fo}}}margin-right") == "0.5cm"
        assert props.get(f"{{{style}}}scale-to-X") == "1"
        assert props.get(f"{{{style}}}scale-to-Y") == "0"

    def test_portrait_setup(self, test_config, tmp_path):
        """Verify portrait orientation is set correctly."""
        ods_path = self._make_minimal_ods(tmp_path)
        converter = self._make_converter(test_config)
        setup = PageSetup(width_cm=21.0, height_cm=29.7, orientation="portrait")

        converter._set_ods_page_setup(ods_path, setup)

        extract_dir = tmp_path / "verify"
        with zipfile.ZipFile(ods_path, "r") as zf:
            zf.extractall(extract_dir)

        tree = ET.parse(extract_dir / "styles.xml")
        ns = {"style": "urn:oasis:names:tc:opendocument:xmlns:style:1.0"}
        props = tree.getroot().find(".//style:page-layout-properties", ns)
        assert props is not None
        assert props.get(f"{{{ns['style']}}}print-orientation") == "portrait"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_office_converter.py::TestSetOdsPageSetup -v`
Expected: FAIL — `_set_ods_page_setup` not defined.

- [ ] **Step 3: Rename and parameterize the method**

In `src/dlightrag/converters/office.py`:

1. Rename `_set_ods_landscape_fit` to `_set_ods_page_setup` and add `setup: PageSetup` parameter.
2. Replace all hardcoded values with `setup` fields.
3. Update call site in `_convert_excel_to_pdf`.

Replace `_set_ods_landscape_fit`. Note: this also adds explicit margin settings
(0.5cm all sides), which the old method left to ODS defaults.

```python
    def _set_ods_page_setup(self, ods_path: Path, setup: PageSetup) -> None:
        """Modify ODS XML to set page dimensions, orientation, and fit-to-width scaling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            extract_dir = temp_path / "ods_content"

            with zipfile.ZipFile(ods_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            styles_xml = extract_dir / "styles.xml"
            if not styles_xml.exists():
                logger.warning("styles.xml not found in ODS, skipping page setup")
                return

            ET.register_namespace("", "urn:oasis:names:tc:opendocument:xmlns:office:1.0")
            ET.register_namespace("style", "urn:oasis:names:tc:opendocument:xmlns:style:1.0")
            ET.register_namespace(
                "fo", "urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
            )
            ET.register_namespace("table", "urn:oasis:names:tc:opendocument:xmlns:table:1.0")

            tree = ET.parse(styles_xml)
            root = tree.getroot()

            ns = {
                "style": "urn:oasis:names:tc:opendocument:xmlns:style:1.0",
                "fo": "urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0",
            }

            fo_ns = ns["fo"]
            style_ns = ns["style"]
            margin = f"{_PAGE_MARGIN_CM}cm"

            for page_layout in root.findall(".//style:page-layout", ns):
                props = page_layout.find("style:page-layout-properties", ns)
                if props is not None:
                    props.set(f"{{{style_ns}}}print-orientation", setup.orientation)
                    props.set(f"{{{fo_ns}}}page-width", f"{setup.width_cm}cm")
                    props.set(f"{{{fo_ns}}}page-height", f"{setup.height_cm}cm")
                    props.set(f"{{{fo_ns}}}margin-left", margin)
                    props.set(f"{{{fo_ns}}}margin-right", margin)
                    props.set(f"{{{fo_ns}}}margin-top", margin)
                    props.set(f"{{{fo_ns}}}margin-bottom", margin)
                    props.set(f"{{{style_ns}}}scale-to-X", "1")
                    props.set(f"{{{style_ns}}}scale-to-Y", "0")

            tree.write(styles_xml, encoding="utf-8", xml_declaration=True)
            self._repack_ods(ods_path, extract_dir)
```

Update call site in `_convert_excel_to_pdf` (line ~153):

```python
            # Step 2: Modify ODS page setup (adaptive width)
            setup = self._estimate_excel_page_width(source_path)
            self._set_ods_page_setup(ods_path, setup)
```

- [ ] **Step 4: Run all office converter tests**

Run: `uv run pytest tests/unit/test_office_converter.py -v`
Expected: All tests PASS (including old tests).

- [ ] **Step 5: Commit**

```bash
git add src/dlightrag/converters/office.py tests/unit/test_office_converter.py
git commit -m "refactor: replace fixed A4 page setup with adaptive _set_ods_page_setup"
```

---

## Chunk 2: New Public API and PageRenderer Integration

### Task 4: Add file_to_pdf_bytes method

**Files:**
- Modify: `src/dlightrag/converters/office.py` (new method + __all__)
- Test: `tests/unit/test_office_converter.py`

- [ ] **Step 1: Write failing tests for file_to_pdf_bytes**

Add to `tests/unit/test_office_converter.py`:

```python
from unittest.mock import patch, MagicMock


class TestFileToPdfBytes:
    """Test file_to_pdf_bytes method."""

    def _make_converter(self, test_config):
        return LibreOfficeConverter(test_config)

    def test_excel_delegates_to_convert_excel_to_pdf(self, test_config, tmp_path):
        """xlsx files should go through the Excel→ODS→PDF pipeline."""
        converter = self._make_converter(test_config)
        xlsx_path = tmp_path / "test.xlsx"
        xlsx_path.touch()

        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        with patch.object(converter, "_convert_excel_to_pdf", return_value=fake_pdf):
            result = converter.file_to_pdf_bytes(xlsx_path)

        assert result == b"%PDF-1.4 fake"

    def test_docx_uses_standard_libreoffice(self, test_config, tmp_path):
        """docx files should use standard libreoffice conversion."""
        converter = self._make_converter(test_config)
        docx_path = tmp_path / "doc.docx"
        docx_path.touch()

        mock_result = MagicMock()
        mock_result.returncode = 0

        def fake_run(cmd, **kwargs):
            outdir = cmd[cmd.index("--outdir") + 1]
            (Path(outdir) / "doc.pdf").write_bytes(b"%PDF-docx")
            return mock_result

        with patch("dlightrag.converters.office.subprocess.run", side_effect=fake_run):
            result = converter.file_to_pdf_bytes(docx_path)

        assert result == b"%PDF-docx"

    def test_nonexistent_file_raises(self, test_config, tmp_path):
        """Missing file should raise OfficeConverterError."""
        converter = self._make_converter(test_config)
        with pytest.raises(OfficeConverterError, match="Source file not found"):
            converter.file_to_pdf_bytes(tmp_path / "missing.xlsx")

    def test_failed_conversion_raises(self, test_config, tmp_path):
        """Failed LibreOffice conversion should raise OfficeConverterError."""
        converter = self._make_converter(test_config)
        pptx_path = tmp_path / "slides.pptx"
        pptx_path.touch()

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error"

        with patch("dlightrag.converters.office.subprocess.run", return_value=mock_result):
            with pytest.raises(OfficeConverterError, match="conversion failed"):
                converter.file_to_pdf_bytes(pptx_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_office_converter.py::TestFileToPdfBytes -v`
Expected: FAIL — `file_to_pdf_bytes` not defined.

- [ ] **Step 3: Implement file_to_pdf_bytes**

Add to `LibreOfficeConverter` in `src/dlightrag/converters/office.py`:

```python
    # Extensions that file_to_pdf_bytes supports
    _OFFICE_EXTENSIONS = {".xls", ".xlsx", ".doc", ".docx", ".ppt", ".pptx"}

    def file_to_pdf_bytes(self, source_path: Path) -> bytes:
        """Convert an Office file to PDF and return the raw PDF bytes.

        Excel files use the adaptive-width Excel→ODS→PDF pipeline.
        Other Office formats use standard LibreOffice conversion.
        Raises OfficeConverterError on failure.
        """
        if not source_path.exists():
            raise OfficeConverterError(f"Source file not found: {source_path}")

        suffix = source_path.suffix.lower()
        if suffix not in self._OFFICE_EXTENSIONS:
            raise OfficeConverterError(f"Unsupported Office format: {suffix}")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)

                if suffix in EXCEL_EXTENSIONS:
                    pdf_path = self._convert_excel_to_pdf(
                        source_path, tmp_path, tmp_path / "output.pdf"
                    )
                else:
                    result = subprocess.run(
                        [
                            "libreoffice",
                            "--headless",
                            "--convert-to",
                            "pdf",
                            "--outdir",
                            str(tmp_path),
                            str(source_path),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=self.timeout,
                        encoding="utf-8",
                        errors="ignore",
                    )
                    pdf_path = tmp_path / (source_path.stem + ".pdf")
                    if result.returncode != 0 or not pdf_path.exists():
                        raise OfficeConverterError(
                            f"LibreOffice conversion failed: {result.stderr}"
                        )

                return pdf_path.read_bytes()

        except OfficeConverterError:
            raise
        except subprocess.TimeoutExpired as e:
            raise OfficeConverterError(
                f"LibreOffice conversion timed out after {self.timeout}s"
            ) from e
        except FileNotFoundError as e:
            raise OfficeConverterError(
                "LibreOffice not found. Install: apt-get install libreoffice"
            ) from e
        except Exception as e:
            raise OfficeConverterError(f"Unexpected conversion error: {e}") from e
```

Update `__all__` to export `PageSetup` and `file_to_pdf_bytes` is a method so no export needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_office_converter.py::TestFileToPdfBytes -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dlightrag/converters/office.py tests/unit/test_office_converter.py
git commit -m "feat: add file_to_pdf_bytes method for PageRenderer consumption"
```

---

### Task 5: Update convert_bytes_to_pdf to use adaptive width

**Files:**
- Modify: `src/dlightrag/converters/office.py:251-311` (remove apply_page_setup, always adaptive for Excel)
- Test: `tests/unit/test_office_converter.py`

- [ ] **Step 1: Write failing test**

Add to `tests/unit/test_office_converter.py`:

```python
class TestConvertBytesToPdfAdaptive:
    """Test that convert_bytes_to_pdf uses adaptive width for Excel."""

    def _make_converter(self, test_config):
        return LibreOfficeConverter(test_config)

    def test_excel_bytes_uses_adaptive_pipeline(self, test_config, tmp_path):
        """Excel MIME type should trigger adaptive conversion (via _convert_excel_to_pdf)."""
        converter = self._make_converter(test_config)

        fake_pdf_path = tmp_path / "output.pdf"
        fake_pdf_path.write_bytes(b"%PDF-adaptive")

        with patch.object(converter, "_convert_excel_to_pdf", return_value=fake_pdf_path) as mock:
            result = converter.convert_bytes_to_pdf(
                b"fake excel bytes",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        assert result == b"%PDF-adaptive"
        mock.assert_called_once()

    def test_apply_page_setup_param_removed(self, test_config):
        """apply_page_setup parameter should no longer exist."""
        converter = self._make_converter(test_config)
        import inspect
        sig = inspect.signature(converter.convert_bytes_to_pdf)
        assert "apply_page_setup" not in sig.parameters
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_office_converter.py::TestConvertBytesToPdfAdaptive -v`
Expected: FAIL — `apply_page_setup` still exists, Excel doesn't use adaptive pipeline.

- [ ] **Step 3: Update convert_bytes_to_pdf**

Replace the method in `src/dlightrag/converters/office.py`:

```python
    def convert_bytes_to_pdf(
        self,
        file_data: bytes,
        mime_type: str,
    ) -> bytes | None:
        """Convert Office document bytes to PDF bytes (in-memory).

        Excel files automatically use the adaptive-width pipeline.
        """
        ext_map = {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "application/msword": ".doc",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
            "application/vnd.ms-powerpoint": ".ppt",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
            "application/vnd.ms-excel": ".xls",
        }
        ext = ext_map.get(mime_type, ".docx")
        is_excel = ext in {".xlsx", ".xls"}

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                input_file = tmp_path / f"input{ext}"
                input_file.write_bytes(file_data)

                if is_excel:
                    pdf_path = self._convert_excel_to_pdf(
                        input_file, tmp_path, tmp_path / "output.pdf"
                    )
                else:
                    result = subprocess.run(
                        [
                            "libreoffice",
                            "--headless",
                            "--convert-to",
                            "pdf",
                            "--outdir",
                            str(tmp_path),
                            str(input_file),
                        ],
                        capture_output=True,
                        timeout=self.timeout,
                        encoding="utf-8",
                        errors="ignore",
                    )
                    pdf_path = tmp_path / "input.pdf"
                    if result.returncode != 0 or not pdf_path.exists():
                        logger.error(f"LibreOffice conversion failed: {result.stderr}")
                        return None

                return pdf_path.read_bytes() if pdf_path.exists() else None

        except subprocess.TimeoutExpired:
            logger.error(f"LibreOffice conversion timed out after {self.timeout}s")
            return None
        except FileNotFoundError:
            logger.error("LibreOffice not found. Please install libreoffice.")
            return None
        except Exception as e:
            logger.error(f"Office to PDF conversion failed: {e}")
            return None
```

- [ ] **Step 4: Run all office converter tests**

Run: `uv run pytest tests/unit/test_office_converter.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dlightrag/converters/office.py tests/unit/test_office_converter.py
git commit -m "refactor: convert_bytes_to_pdf always uses adaptive width for Excel"
```

---

### Task 6: PageRenderer integration — accept converter and delegate

**Files:**
- Modify: `src/dlightrag/unifiedrepresent/renderer.py:22-24` (_OFFICE_EXTENSIONS)
- Modify: `src/dlightrag/unifiedrepresent/renderer.py:35-49` (__init__)
- Modify: `src/dlightrag/unifiedrepresent/renderer.py:143-191` (_render_office)
- Add method: `_render_pdf_from_bytes`
- Test: `tests/unit/test_renderer.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_renderer.py`:

```python
class TestRenderOfficeWithConverter:
    """Test _render_office delegation to LibreOfficeConverter."""

    async def test_delegates_to_converter(self, tmp_path):
        """When converter is provided, _render_office uses file_to_pdf_bytes."""
        from dlightrag.unifiedrepresent.renderer import PageRenderer, RenderResult

        mock_converter = MagicMock()
        # Create a minimal valid PDF via pypdfium2
        # We'll mock _render_pdf_from_bytes instead to avoid needing real PDF bytes
        renderer = PageRenderer(dpi=144, converter=mock_converter)

        doc_path = tmp_path / "doc.xlsx"
        doc_path.touch()

        mock_converter.file_to_pdf_bytes.return_value = b"%PDF-fake"

        mock_render_result = RenderResult(
            pages=[(0, Image.new("RGB", (10, 10)))],
            metadata={"original_format": "pdf", "page_count": 1},
        )

        with patch.object(renderer, "_render_pdf_from_bytes", return_value=mock_render_result):
            result = await renderer._render_office(doc_path)

        mock_converter.file_to_pdf_bytes.assert_called_once_with(doc_path)
        assert result.metadata["original_format"] == "xlsx"

    async def test_no_converter_uses_fallback(self, tmp_path):
        """Without converter, falls back to bare LibreOffice."""
        from dlightrag.unifiedrepresent.renderer import PageRenderer, RenderResult

        renderer = PageRenderer(dpi=144)  # no converter

        doc_path = tmp_path / "doc.docx"
        doc_path.touch()

        mock_run_result = MagicMock()
        mock_run_result.returncode = 0
        mock_render_result = RenderResult(
            pages=[(0, Image.new("RGB", (10, 10)))],
            metadata={"original_format": "pdf", "page_count": 1},
        )

        def fake_run(cmd, **kwargs):
            outdir_idx = cmd.index("--outdir")
            outdir = cmd[outdir_idx + 1]
            (Path(outdir) / "doc.pdf").touch()
            return mock_run_result

        with (
            patch("dlightrag.unifiedrepresent.renderer.shutil.which", return_value="/usr/bin/libreoffice"),
            patch("dlightrag.unifiedrepresent.renderer.subprocess.run", side_effect=fake_run),
            patch.object(renderer, "_render_pdf_sync", return_value=mock_render_result),
        ):
            result = await renderer._render_office(doc_path)

        assert result.metadata["original_format"] == "docx"


class TestRenderPdfFromBytes:
    """Test _render_pdf_from_bytes."""

    def test_renders_from_bytes(self):
        """Verify PDF bytes are rendered to page images."""
        from dlightrag.unifiedrepresent.renderer import PageRenderer

        renderer = PageRenderer(dpi=72)

        mock_pil_img = Image.new("RGB", (100, 100), "white")
        mock_bitmap = MagicMock()
        mock_bitmap.to_pil.return_value = mock_pil_img

        mock_page = MagicMock()
        mock_page.render.return_value = mock_bitmap

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_doc.get_metadata_dict.return_value = {}

        with patch("dlightrag.unifiedrepresent.renderer.pdfium.PdfDocument", return_value=mock_doc) as mock_cls:
            result = renderer._render_pdf_from_bytes(b"%PDF-fake")

        # Verify PdfDocument was called with bytes, not a path
        mock_cls.assert_called_once_with(b"%PDF-fake")
        assert len(result.pages) == 1
        assert result.pages[0][1] is mock_pil_img
        mock_doc.close.assert_called_once()


class TestOfficeExtensionsExpanded:
    """Test that legacy Office formats are accepted."""

    async def test_xls_routes_to_render_office(self, tmp_path):
        """xls files should be handled by _render_office, not raise ValueError."""
        from dlightrag.unifiedrepresent.renderer import PageRenderer, RenderResult

        renderer = PageRenderer(dpi=144)
        xls_path = tmp_path / "old.xls"
        xls_path.touch()

        mock_result = RenderResult(
            pages=[(0, Image.new("RGB", (10, 10)))],
            metadata={"original_format": "xls"},
        )

        with patch.object(renderer, "_render_office", return_value=mock_result) as mock:
            result = await renderer.render_file(xls_path)

        mock.assert_called_once_with(xls_path)
        assert result.metadata["original_format"] == "xls"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_renderer.py::TestRenderOfficeWithConverter tests/unit/test_renderer.py::TestRenderPdfFromBytes tests/unit/test_renderer.py::TestOfficeExtensionsExpanded -v`
Expected: FAIL — converter not accepted, _render_pdf_from_bytes not defined, .xls raises ValueError.

- [ ] **Step 3: Implement PageRenderer changes**

In `src/dlightrag/unifiedrepresent/renderer.py`:

1. Expand `_OFFICE_EXTENSIONS`:

```python
_OFFICE_EXTENSIONS = frozenset({".docx", ".pptx", ".xlsx", ".doc", ".ppt", ".xls"})
```

2. Add `TYPE_CHECKING` import and update `__init__`:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dlightrag.converters.office import LibreOfficeConverter
```

```python
class PageRenderer:
    def __init__(
        self,
        dpi: int = 250,
        converter: LibreOfficeConverter | None = None,
    ) -> None:
        self.dpi = dpi
        self._converter = converter
```

3. Add `_render_pdf_from_bytes`:

```python
    def _render_pdf_from_bytes(self, pdf_bytes: bytes) -> RenderResult:
        """Render PDF from raw bytes (called via to_thread by _render_office)."""
        doc = pdfium.PdfDocument(pdf_bytes)
        try:
            scale = self.dpi / 72
            pages: list[tuple[int, Image.Image]] = []
            for idx in range(len(doc)):
                page = doc[idx]
                bitmap = page.render(scale=scale)
                pil_image = bitmap.to_pil()
                pages.append((idx, pil_image))

            metadata: dict[str, str | int] = {"original_format": "pdf", "page_count": len(doc)}
            try:
                meta = doc.get_metadata_dict()
                if meta.get("Title"):
                    metadata["title"] = meta["Title"]
                if meta.get("Author"):
                    metadata["author"] = meta["Author"]
                if meta.get("CreationDate"):
                    metadata["creation_date"] = meta["CreationDate"]
            except Exception:
                logger.debug("Could not extract PDF metadata from bytes")

            return RenderResult(pages=pages, metadata=metadata)
        finally:
            doc.close()
```

4. Update `_render_office`:

```python
    async def _render_office(self, path: Path) -> RenderResult:
        """Convert an Office document to PDF, then render."""
        if self._converter:
            def _convert_and_render(p: Path) -> RenderResult:
                pdf_bytes = self._converter.file_to_pdf_bytes(p)
                return self._render_pdf_from_bytes(pdf_bytes)

            render_result = await asyncio.to_thread(_convert_and_render, path)
            render_result.metadata["original_format"] = path.suffix.lower().lstrip(".")
            return render_result

        # Fallback: bare LibreOffice (no converter provided)
        lo_bin = shutil.which("libreoffice") or shutil.which("soffice")
        if lo_bin is None:
            raise RuntimeError(
                "LibreOffice is required to render Office documents but was not found on PATH. "
                "Install it (e.g. `apt install libreoffice-core`) and try again."
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = await asyncio.to_thread(
                subprocess.run,
                [lo_bin, "--headless", "--convert-to", "pdf", "--outdir", tmp_dir, str(path)],
                capture_output=True,
                timeout=120,
            )

            if result.returncode != 0:
                stderr = result.stderr.decode(errors="replace").strip()
                raise RuntimeError(
                    f"LibreOffice conversion failed (exit {result.returncode}): {stderr}"
                )

            pdf_path = Path(tmp_dir) / (path.stem + ".pdf")
            if not pdf_path.exists():
                raise RuntimeError(
                    f"LibreOffice conversion produced no output PDF "
                    f"(expected {pdf_path.name} in temp dir)"
                )

            render_result = self._render_pdf_sync(pdf_path)

        render_result.metadata["original_format"] = path.suffix.lower().lstrip(".")
        return render_result
```

- [ ] **Step 4: Run all renderer tests**

Run: `uv run pytest tests/unit/test_renderer.py -v`
Expected: All tests PASS (old + new).

- [ ] **Step 5: Commit**

```bash
git add src/dlightrag/unifiedrepresent/renderer.py tests/unit/test_renderer.py
git commit -m "feat: PageRenderer delegates Office conversion to LibreOfficeConverter"
```

---

### Task 7: Wire converter into UnifiedRepresentEngine

**Files:**
- Modify: `src/dlightrag/unifiedrepresent/engine.py:54` (one line change)

- [ ] **Step 1: Update engine wiring**

In `src/dlightrag/unifiedrepresent/engine.py`, change line 54:

From:
```python
        self.renderer = PageRenderer(dpi=config.page_render_dpi)
```

To:
```python
        from dlightrag.converters.office import LibreOfficeConverter

        converter = LibreOfficeConverter(config)
        self.renderer = PageRenderer(dpi=config.page_render_dpi, converter=converter)
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/unit/ -v --tb=short`
Expected: All tests PASS. No regressions.

- [ ] **Step 3: Commit**

```bash
git add src/dlightrag/unifiedrepresent/engine.py
git commit -m "feat: wire LibreOfficeConverter into UnifiedRepresentEngine"
```

---

## Chunk 3: Cleanup and Final Verification

### Task 8: Final verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/unit/ -v --tb=short`
Expected: All tests PASS.

- [ ] **Step 2: Run linter**

Run: `uv run ruff check src/dlightrag/converters/office.py src/dlightrag/unifiedrepresent/renderer.py src/dlightrag/unifiedrepresent/engine.py`
Expected: No errors.

- [ ] **Step 3: Run ruff format**

Run: `uv run ruff format src/dlightrag/converters/office.py src/dlightrag/unifiedrepresent/renderer.py src/dlightrag/unifiedrepresent/engine.py`
Expected: Files formatted.

- [ ] **Step 4: Verify the adaptive width with a quick smoke check**

Run the following in Python to verify the full flow conceptually:

```bash
uv run python3 -c "
from dlightrag.converters.office import PageSetup, _EXCEL_UNIT_TO_CM, _MIN_PAGE_WIDTH_CM, _MAX_PAGE_WIDTH_CM
# Verify constants
assert _EXCEL_UNIT_TO_CM == 0.201
assert _MIN_PAGE_WIDTH_CM == 21.0
assert _MAX_PAGE_WIDTH_CM == 63.0
# Verify PageSetup
setup = PageSetup(width_cm=45.0, height_cm=21.0, orientation='landscape')
assert setup.width_cm == 45.0
print('Smoke check passed')
"
```

- [ ] **Step 5: Final commit if any formatting changes**

```bash
git add -u
git commit -m "style: apply ruff formatting"
```
