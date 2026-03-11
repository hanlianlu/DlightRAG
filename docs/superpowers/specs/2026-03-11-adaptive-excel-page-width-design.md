# Adaptive Excel Page Width & OfficeConverter Unification

**Date:** 2026-03-11
**Status:** Approved

## Problem

Two issues in the current Office-to-PDF conversion:

1. **Fixed A4 page width** — `LibreOfficeConverter._set_ods_landscape_fit()` hardcodes A4
   landscape (29.7cm x 21cm). Wide Excel files get all columns squeezed into 29.7cm,
   producing tiny unreadable text that degrades downstream OCR/parsing quality.

2. **Duplicated conversion logic** — `PageRenderer._render_office()` (unified represent mode)
   has its own bare `libreoffice --headless --convert-to pdf` call with zero page optimization.
   `LibreOfficeConverter` (caption mode) has optimized conversion but is not reused.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Column width source | openpyxl (`data_only=True`, normal mode) | `read_only=True` lacks `column_dimensions`; verified via testing |
| Excel unit → cm | 1 unit ≈ 0.201cm (7.59px at 96 DPI) | Standard Excel rendering metric |
| Page width bounds | min 21cm, max 63cm | 21cm = A4 portrait width; 63cm ≈ 3x A4 landscape; beyond 63cm fit-to-width compresses |
| Margins | 0.5cm all sides | User requirement |
| Orientation | Auto: width <= 21cm → portrait, else landscape | Narrow Excels benefit from portrait; wide ones need landscape |
| `.xls` fallback | A4 landscape + fit-to-width | openpyxl doesn't support `.xls`; graceful degradation |
| Architecture | PageRenderer holds optional `LibreOfficeConverter` ref | Composition pattern consistent with project; backward compatible |
| `convert_bytes_to_pdf` | Also upgraded to adaptive width for Excel | Consistency across all conversion paths |

## Architecture

```
IngestionPipeline (caption mode)
    └─ LibreOfficeConverter
           ├─ should_convert()          # gate: only Excel, not docling
           ├─ convert_to_pdf()          # file path → PDF file (existing, upgraded)
           └─ convert_bytes_to_pdf()    # bytes → PDF bytes (existing, upgraded)

UnifiedRepresentEngine
    └─ PageRenderer(converter=LibreOfficeConverter)
           └─ _render_office()
                  └─ converter.convert_to_pdf_bytes()  # NEW: file → PDF bytes
                  └─ _render_pdf_from_bytes()           # render PDF bytes to images
```

`PageRenderer._render_office()` delegates to `LibreOfficeConverter` when a converter is
provided. Without a converter, the existing bare LibreOffice fallback is preserved for
backward compatibility.

## New/Modified Components

### `PageSetup` dataclass (new)

```python
@dataclass
class PageSetup:
    width_cm: float     # page width in cm
    height_cm: float    # page height in cm
    orientation: str    # "landscape" | "portrait"
```

### `LibreOfficeConverter` changes

#### `_estimate_excel_page_width(source_path) -> PageSetup` (new)

1. Open with `openpyxl.load_workbook(path, data_only=True)`.
2. Read active sheet `column_dimensions`; default width = 8.43 for unset columns.
3. Sum all column widths up to `max_column`, multiply by 0.201 cm/unit.
4. Add 1.0cm (left + right margins).
5. Clamp to [21cm, 63cm].
6. Orientation: width <= 21cm → portrait (21 x 29.7), else landscape (width x 21).
7. On any failure (`.xls`, corrupt file, import error): return A4 landscape fallback
   (29.7 x 21, landscape).

#### `_set_ods_page_setup(ods_path, setup: PageSetup)` (renamed from `_set_ods_landscape_fit`)

Same XML manipulation logic but parameterized:
- `page-width` = `setup.width_cm`
- `page-height` = `setup.height_cm`
- `print-orientation` = `setup.orientation`
- All margins = 0.5cm
- `scale-to-X` = 1, `scale-to-Y` = 0

#### `_convert_excel_to_pdf(source_path, output_dir, expected_pdf_path)` (modified)

Step 2 changes from `self._set_ods_landscape_fit(ods_path)` to:
```python
setup = self._estimate_excel_page_width(source_path)
self._set_ods_page_setup(ods_path, setup)
```

#### `convert_to_pdf_bytes(source_path: Path) -> bytes` (new)

Public method for `PageRenderer` consumption:
- Excel (`.xls`, `.xlsx`): uses `_convert_excel_to_pdf` flow → returns PDF bytes.
- Other Office (`.docx`, `.pptx`, `.doc`, `.ppt`): standard `libreoffice --headless --convert-to pdf`.
- Returns raw PDF bytes (not a file path).

#### `convert_bytes_to_pdf(file_data, mime_type, ...)` (modified)

For Excel MIME types: writes temp file → calls `_estimate_excel_page_width` →
full adaptive conversion flow. Replaces the `apply_page_setup` parameter with
automatic detection.

### `PageRenderer` changes

#### `__init__` signature

```python
def __init__(
    self,
    dpi: int = 250,
    converter: LibreOfficeConverter | None = None,
) -> None:
```

#### `_render_office(path)` (modified)

```
if self._converter:
    pdf_bytes = converter.convert_to_pdf_bytes(path)  # via asyncio.to_thread
    return self._render_pdf_from_bytes(pdf_bytes)
else:
    ... existing bare LibreOffice fallback ...
```

#### `_render_pdf_from_bytes(pdf_bytes)` (new)

Like `_render_pdf_sync` but accepts bytes instead of a file path. Uses
`pdfium.PdfDocument(pdf_bytes)` (pypdfium2 supports bytes input).

### `UnifiedRepresentEngine` change

```python
# engine.py __init__
from dlightrag.converters.office import LibreOfficeConverter
converter = LibreOfficeConverter(config)
self.renderer = PageRenderer(dpi=config.page_render_dpi, converter=converter)
```

### `pyproject.toml`

Add `openpyxl` to dependencies.

## Files Changed

| File | Type | Summary |
|------|------|---------|
| `src/dlightrag/converters/office.py` | Modified | `PageSetup`, `_estimate_excel_page_width`, `convert_to_pdf_bytes`, parameterized page setup, adaptive `convert_bytes_to_pdf` |
| `src/dlightrag/unifiedrepresent/renderer.py` | Modified | Accept optional converter, delegate `_render_office`, add `_render_pdf_from_bytes` |
| `src/dlightrag/unifiedrepresent/engine.py` | Modified | Pass converter to PageRenderer |
| `pyproject.toml` | Modified | Add `openpyxl` dependency |
| `tests/unit/test_office_converter.py` | Modified | Tests for adaptive width calculation, page setup, edge cases |
| `tests/unit/test_renderer.py` | Modified | Tests for converter delegation path |

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| `.xls` file | openpyxl can't read → fallback to A4 landscape |
| Corrupt/password-protected Excel | openpyxl raises → fallback to A4 landscape |
| Excel with 0 columns | fallback to A4 landscape |
| Excel with hidden columns | openpyxl still reports their width; included in calculation |
| Multiple sheets with different widths | Uses active sheet only |
| Calculated width exactly 21cm | Portrait orientation |
| Calculated width 63.1cm | Clamped to 63cm, fit-to-width compresses slightly |
| LibreOffice not installed | Existing error handling preserved |
| PageRenderer without converter | Existing bare conversion preserved |
