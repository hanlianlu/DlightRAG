---
name: office-documents
description: Use when the user asks for a Word, Excel, PowerPoint or PDF file (.docx, .xlsx, .pptx, .pdf).
---

# Office documents

Build the file type the user asked for, not HTML or Markdown in its place. The workspace Python has python-docx, openpyxl, python-pptx and fpdf2, and nothing needs installing or downloading: write the generator and the data it reads into `artifacts/` beside the file it produces, run it, reopen the result to check it, and attach the file naming them — `attach_artifact(path="report.docx", sources=["report.docx.build.py", "data.csv"])`. A Word file cannot link to what made it and this Run's workspace does not outlive it, so a deliverable attached without its sources can be read next turn but never rebuilt; naming them publishes them with it, and reopening the published file brings them back. Nothing here renders Office files, so judge their layout from the file's structure.

- **Word**: use paragraph styles for headings and a table style such as `Table Grid` so tables have borders; set an explicit East Asian font for Chinese, Japanese or Korean text.
- **Excel**: keep numbers, dates and identifiers as their real cell types. A formula is stored without a result until Excel opens the file, so also state the key totals in the Answer. Bold the header row and set column widths.
- **PowerPoint**: set the slide size explicitly (13.333 x 7.5 in for 16:9) and font sizes; one idea per slide; use `python-pptx` charts rather than pasted images; keep text inside the slide, and after saving reopen the file and check that every shape's left + width and top + height stay within the slide size.
- **PDF**: fpdf2's built-in fonts are Latin-1 only, and a glyph the set font lacks is dropped without an error. Take the image's font set from `/usr/local/share/fonts/dlightrag/`, or from the directory `$FONTS_DIR` names when that is set: `SourceSans3-{Regular,Bold,It,BoldIt}.ttf` as `latin`, `NotoSansSC-{Regular,Bold}.otf` as `sc`, `NotoSansJP-…` as `jp`, `NotoSansKR-…` as `kr`. Register all four, set the family of the document's language and `set_fallback_fonts` the others; the CJK families have no italic, and `latin` — the same shapes as the CJK Latin, at 0.895 of their scale — needs 1.117 times the size in a line that mixes the two. Emoji, Arabic, Hebrew, Thai and Devanagari are in none of them. pypdfium2 renders pages to PNG for `view`.
- Say where a figure comes from in words; a citation marker is not resolved in these files.
