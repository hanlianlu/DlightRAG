---
name: office-documents
description: Use when the user asks for a Word, Excel or PowerPoint file (.docx, .xlsx, .pptx).
---

# Office documents

Build the file type the user asked for, not HTML or Markdown in its place. The workspace Python has python-docx, openpyxl and python-pptx: write the script under `tmp/`, save the result in `artifacts/`, reopen it to check it, and attach it with `attach_artifact`. Nothing here renders Office files, so judge layout from the file's structure.

- **Word**: use paragraph styles for headings and a table style such as `Table Grid` so tables have borders; set an explicit East Asian font for Chinese, Japanese or Korean text.
- **Excel**: keep numbers, dates and identifiers as their real cell types. A formula is stored without a result until Excel opens the file, so also state the key totals in the Answer. Bold the header row and set column widths.
- **PowerPoint**: set the slide size explicitly (13.333 x 7.5 in for 16:9) and font sizes; one idea per slide; use `python-pptx` charts rather than pasted images; keep text inside the slide, and after saving reopen the file and check that every shape's left + width and top + height stay within the slide size.
- Say where a figure comes from in words; a citation marker is not resolved in these files.
