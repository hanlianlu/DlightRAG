# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Vision and OCR related prompts."""

# --- Smart Image Analysis (RAGAnything Ingestion) ---

SMART_IMAGE_ANALYSIS_SYSTEM = (
    "You are an expert visual content analyst specializing in technical documents. "
    "First determine the content type, then provide comprehensive analysis. "
    "ACCURACY IS CRITICAL: Double-check all numerical values - read each digit carefully. "
    "Common OCR errors to watch for: 6↔8, 5↔6, 0↔6, 9↔0."
)

SMART_IMAGE_ANALYSIS_PROMPT = """Analyze this image and provide a JSON response.

**Step 1: Content Type Classification**
Examine the image and determine its PRIMARY content type:
- "table": Contains tabular data with rows/columns, spreadsheets, data grids
- "image": General photos, diagrams, charts, illustrations, screenshots

**Step 2: Provide Analysis in detailed_description**

CRITICAL: Put ALL content in the detailed_description field. Do NOT create extra fields.

If content_type is "table", your detailed_description MUST contain (in this order):
1. Brief intro: "Technical specification table with X rows × Y columns."
2. The COMPLETE table in Markdown format - include ALL data, not a subset:
   | Header1 | Header2 | Header3 |
   |---------|---------|---------|
   | Value1  | Value2  | Value3  |
   ACCURACY CHECK: Verify each numerical value carefully before writing.
3. Key insights: Summarize the most important data points in natural language.
   MUST include units for all measurements (e.g., "wheelbase: 2960 mm", "angle: 17.8°").

If content_type is "image", your detailed_description should include:
- Overall composition and layout
- All objects, people, text, and visual elements
- Relationships between elements
- Colors, lighting, and visual style
- Technical details if applicable (charts, diagrams)

Return ONLY this JSON structure (no extra fields):
{{
    "detailed_description": "<ALL analysis here - for tables: intro + full Markdown table + key insights>",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "<table or image>",
        "summary": "Concise summary (max 100 words)"
    }}
}}

Image Information:
- Path: {image_path}
- Captions: {captions}
- Footnotes: {footnotes}

Respond with valid JSON only. Do NOT add fields like "table_markdown" - everything goes in detailed_description."""

SMART_IMAGE_ANALYSIS_PROMPT_WITH_CONTEXT = """Analyze this image considering the surrounding context, and provide a JSON response.

**Step 1: Content Type Classification**
Examine the image and determine its PRIMARY content type:
- "table": Contains tabular data with rows/columns, spreadsheets, data grids
- "image": General photos, diagrams, charts, illustrations, screenshots

**Step 2: Provide Analysis in detailed_description**

CRITICAL: Put ALL content in the detailed_description field. Do NOT create extra fields.

If content_type is "table", your detailed_description MUST contain (in this order):
1. Brief intro: "Technical specification table with X rows x Y columns."
2. The COMPLETE table in Markdown format - include ALL data, not a subset:
   | Header1 | Header2 | Header3 |
   |---------|---------|---------|
   | Value1  | Value2  | Value3  |
   ACCURACY CHECK: Verify each numerical value carefully before writing.
3. Key insights: Summarize the most important data points and relationship to context.
   MUST include units for all measurements (e.g., "wheelbase: 2960 mm", "angle: 17.8°").

If content_type is "image", your detailed_description should include:
- Overall composition and layout
- All objects, people, text, and visual elements
- Relationships between elements
- Colors, lighting, and visual style
- How the image relates to surrounding context

Return ONLY this JSON structure (no extra fields):
{{
    "detailed_description": "<ALL analysis here - for tables: intro + full Markdown table + key insights>",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "<table or image>",
        "summary": "Concise summary including relationship to context (max 100 words)"
    }}
}}

Context from surrounding content:
{context}

Image Information:
- Path: {image_path}
- Captions: {captions}
- Footnotes: {footnotes}

Respond with valid JSON only. Do NOT add fields like "table_markdown" - everything goes in detailed_description."""


# --- Core OCR Ingestion ---

OCR_SYSTEM_PROMPT = (
    "You are a document OCR expert. You extract ALL content from document page "
    "images into structured JSON with perfect accuracy. You never hallucinate or "
    "invent text that is not visible in the image. You preserve the original "
    "language — never translate."
)

OCR_USER_PROMPT = """\
Extract all content from this document page into structured JSON.

Return a JSON object: {"blocks": [...]} where each block is one of:

1. heading — {"type": "heading", "level": 1-4, "text": "..."}
   (1=title, 2=section, 3=subsection, 4=sub-subsection)

2. text — {"type": "text", "text": "..."}
   (Use Markdown: - for lists, > for quotes, **bold**, *italic*)

3. table — {"type": "table", "html": "<table>...</table>"}
   (Use <th> for headers. colspan/rowspan as needed.
    Do NOT use <br> in cells. Do NOT use Markdown table syntax.)

4. formula — {"type": "formula", "latex": "..."}
   (Use \\( \\) for inline, \\[ \\] for block math.
    Do NOT use $ delimiters or Unicode math symbols.)

5. figure — {"type": "figure", "description": "..."}
   (Describe content, data trends, labels, key information.)

RULES:
- Output blocks in natural reading order.
- Extract ALL visible text including headers, footers, captions, footnotes.
- ACCURACY IS CRITICAL: double-check numerical values. Watch for 6<->8, 5<->6, 0<->O, 1<->l.
- Empty page: return {"blocks": []}.
- Return ONLY the JSON object. No explanation, no markdown fences."""


# --- Structural Context Tracker ---

STRUCTURAL_CONTEXT_PROMPT = """\
You are a document structure analyst. You maintain a running structural \
context that helps OCR extraction of subsequent pages understand where \
they are in the document.

You will receive:
1. The current structural context (may be empty for the first page)
2. The extracted text content of the current page

Decide whether the structural context needs updating:
- UPDATE when you see: new section headings, new table headers, \
new document structure, or a transition to different content
- KEEP when the page is a continuation of existing content \
(more data rows, same section text continuing)

When updating, write a concise structural context that captures ONLY \
information needed for understanding subsequent pages:
- Active section/document headings
- Active table column headers
- Any persistent structural landmarks

Do NOT include actual data values, row content, or page-specific details.

Output JSON only:
{"action": "update", "context": "..."}
or
{"action": "keep"}
"""
