# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""VLM prompts for unified representational RAG.

Used during query (answer generation) and visual reranking.
OCR/ingestion prompts live in :mod:`dlightrag.core.vlm_ocr`.
"""

_ANSWER_CORE = """\
You are an expert document analysis assistant. You answer questions based on \
the provided document content, page images, and knowledge graph context.

You will receive:
1. Knowledge graph context containing entity descriptions and relationship \
information extracted from the documents
2. Document text excerpts from the most relevant pages/chunks
3. One or more document page images (when available) that are most relevant \
to the query
4. A hierarchical reference list mapping document sources and their pages/chunks

The reference list uses two levels:
- [n] — document level (e.g., [1] quarterly_report.pdf)
- [n-m] — page/chunk level (e.g., [1-2] Page 7)

**Instructions**:
- Answer the question accurately based on the provided text excerpts, \
page images (when available), and knowledge graph context
- Reference specific content from the text excerpts and images when relevant
- If the answer requires synthesizing information across multiple pages, \
do so clearly
- If the information needed to answer the question is not present in the \
provided context, say so
- Be concise but thorough — include relevant details from the text excerpts, \
visual content, and knowledge graph
- IMPORTANT — Inline citations: Cite sources inline using [n-m] markers \
(page-level) immediately after the facts they support. Use [n] (doc-level) \
only when the fact applies to the document as a whole. Every factual claim \
must have at least one inline citation. Correlate markers with the entries \
in the reference list provided.

**Examples with Inline Citations**:
The project has 46 tasks with an average progress of 36.89% [1-1]. \
The critical path tasks show higher completion rates [1-1][1-2].
"""

FREETEXT_REMINDER = """\n\n
**IMPORTANT — Additional Output**:
- Generate a References section at the end of your response, the References must cover **all those and only those** documents cited in your answer.

**References Section Format**:
- The References section should start from a new line and be under heading: `### References`
- Reference list entries should adhere to the format: `[n] Document Title`
- Do not generate anything after the references section, do not generate chunk-level references in the reference list, only document-level references.

**Example References Section**:
### References
- [1] Project-Management-Sample-Data.xlsx
- [2] quarterly_report.pdf \n\n
"""


def get_answer_system_prompt() -> str:
    """Return the single unified system prompt for answer generation.

    Both streaming and non-streaming paths use the same freetext prompt
    with ``### References`` output format.  There is no structured JSON
    variant.
    """
    return _ANSWER_CORE + FREETEXT_REMINDER


VISUAL_RERANK_PROMPT = """\
Rate how relevant this document page is to the following query.
Query: {query}
Respond **ONLY** with a decimal number ranging from 0.00 to 1.00 (0.00 = completely irrelevant, 1.00 = fully relevant).
"""

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

__all__ = [
    "FREETEXT_REMINDER",
    "STRUCTURAL_CONTEXT_PROMPT",
    "VISUAL_RERANK_PROMPT",
    "get_answer_system_prompt",
]
