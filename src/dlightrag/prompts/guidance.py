# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Task-specific guidance fragments for DlightRAG prompts."""

ANSWER_CONTEXT_GUIDANCE = """\
You answer questions using the provided document content, page images, and knowledge graph context.

You will receive:
1. Knowledge graph context containing entity descriptions and relationship information extracted from the documents
2. Document text excerpts from the most relevant pages/chunks
3. One or more document page images, when available, that are most relevant to the query
4. A hierarchical reference list mapping document sources and their pages/chunks

**Instructions**:
- Answer the question accurately based on the provided text excerpts, page images when available, and knowledge graph context
- Reference specific content from the text excerpts and images when relevant
- If the answer requires synthesizing information across multiple pages, do so clearly
- If the information needed to answer the question is not present in the provided context, say so
- Be concise but thorough; include relevant details from the text excerpts, visual content, and knowledge graph
"""

CITATION_GUIDANCE = """\
The reference list uses two levels:
- [n] -- document level (e.g., [1] quarterly_report.pdf)
- [n-m] -- page/chunk level (e.g., [1-2] Page 7)

**Citation Contract**:
- Cite sources inline using [n-m] markers (page-level) immediately after the facts they support
- Use [n] (doc-level) only when the fact applies to the document as a whole
- Every factual claim must have at least one inline citation
- Correlate markers with the entries in the reference list provided
- Cite max the 1-2 most directly relevant chunks per claim
- Avoid long chains of citations from the same source; if the evidence spans many pages, prefer a single [n] document-level citation for that source
- Do not add a "References", "Sources", or bibliography section; the system validates inline citations and builds sources separately
"""

ANSWER_CITATION_EXAMPLE = """\
**Example with Inline Citations**:
The project has 46 tasks with an average progress of 36.89% [1-1]. The critical path tasks show higher completion rates [1-1][1-2].
"""

PLANNER_GUIDANCE = """\
Given a user query (and optionally conversation history), produce a JSON response with these keys:

- "standalone_query": If conversation history is provided, rewrite the follow-up into
  a self-contained query capturing full intent. If no history or the query is already
  standalone, return it unchanged. This is the primary search query -- keep it complete.
- "bm25_query": Optional short keyword query for lexical BM25 retrieval. Use important
  nouns, identifiers, quoted phrases, filenames, and visible terms. Keep it shorter than
  standalone_query. Use null when standalone_query is already short and keyword-oriented.
- "referenced_image_ids": A list of prior image ids such as ["img_0"] only when the
  current query explicitly refers to images listed in conversation history. Do not invent
  ids. Use [] when the query does not reference prior images.
- "filters": An object with applicable fields from the metadata schema below.
  Only include fields you are highly confident about. Leave out uncertain fields.
- "filter_confidence": "high" only when the query explicitly asks to constrain
  by metadata (filename, title, author, date, extension, declared custom field).
  Use "low" when metadata interpretation is plausible but ambiguous.
- "filter_evidence": A list of objects for every filter you include. Each object
  must contain: field, value, evidence_span, intent_basis. evidence_span must be
  an exact phrase from the user query or conversation that justifies treating
  the value as a metadata constraint. Do not include filters without evidence.

Filter fields (use null for unmentioned):
- filename: exact normalized filename when the user gives a complete name with extension
- filename_stem: exact normalized filename without extension only when explicitly requested
- filename_pattern: SQL ILIKE pattern (% wildcards) only when the user explicitly gives a partial file identifier, wildcard-style pattern, camera/code identifier, or asks for a filename/title pattern rather than a broad topical search.
- file_extension: e.g. "pdf", "png" (lowercase, no dot)
- doc_title: exact normalized document title only when highly confident
- doc_author: exact normalized author name only when highly confident
- date_from / date_to: ISO 8601 dates for time ranges
- custom: {{"key": "value"}} for custom metadata

{schema_section}{custom_keys_hint}\
{history_section}\
Examples:

Query: "summarize the key findings in annual-report.pdf"
{{"standalone_query": "summarize the key findings in annual-report.pdf", "bm25_query": "key findings annual-report.pdf", "referenced_image_ids": [], "filters": {{"filename": "annual-report.pdf"}}, "filter_confidence": "high", "filter_evidence": [{{"field": "filename", "value": "annual-report.pdf", "evidence_span": "annual-report.pdf", "intent_basis": "filename_literal"}}]}}

Query: "what are the main revenue trends"
{{"standalone_query": "what are the main revenue trends", "bm25_query": "revenue trends", "referenced_image_ids": [], "filters": {{}}, "filter_confidence": "low", "filter_evidence": []}}

Query: "what is in IMG 9551?"
{{"standalone_query": "what is in IMG 9551?", "bm25_query": "IMG 9551", "referenced_image_ids": [], "filters": {{"filename_pattern": "%IMG%9551%"}}, "filter_confidence": "high", "filter_evidence": [{{"field": "filename_pattern", "value": "%IMG%9551%", "evidence_span": "IMG 9551", "intent_basis": "filename_pattern_literal"}}]}}

Query: "show me slide deck 3"
{{"standalone_query": "show me slide deck 3", "bm25_query": "slide deck 3", "referenced_image_ids": [], "filters": {{"filename_pattern": "%slide%deck%3%"}}, "filter_confidence": "high", "filter_evidence": [{{"field": "filename_pattern", "value": "%slide%deck%3%", "evidence_span": "slide deck 3", "intent_basis": "filename_pattern_literal"}}]}}

Query: "张三写的2024年财报分析"
{{"standalone_query": "张三写的2024年财报分析", "bm25_query": "张三 2024 财报分析", "referenced_image_ids": [], "filters": {{"doc_author": "张三", "date_from": "2024-01-01", "date_to": "2024-12-31"}}, "filter_confidence": "high", "filter_evidence": [{{"field": "doc_author", "value": "张三", "evidence_span": "张三写的", "intent_basis": "explicit_author_constraint"}}, {{"field": "date", "value": "2024", "evidence_span": "2024年", "intent_basis": "date_literal"}}]}}

Conversation history contains: user: compare these [attached images: img_0, img_1]
Current follow-up: "what about the second image?"
{{"standalone_query": "what about the second image?", "bm25_query": null, "referenced_image_ids": ["img_1"], "filters": {{}}, "filter_confidence": "low", "filter_evidence": []}}

Return valid JSON only, no markdown fences."""

RERANK_GUIDANCE = (
    "Use 0.00 for completely irrelevant content and 1.00 for perfectly relevant content."
)

VISUAL_RERANK_PROMPT_TEMPLATE = """\
Rate how relevant this document page is to the following query.
Query: {query}
Respond **ONLY** with a decimal number ranging from 0.00 to 1.00. {rerank_guidance}
"""

LISTWISE_RERANK_PROMPT = """\
Score the relevance of {n} items to the query below. Each item may be an image, text, or both.

Respond with ONLY a JSON array of {n} scores in order: [<float>, <float>, ...]
{rerank_guidance}

Query: {query}""".format(rerank_guidance=RERANK_GUIDANCE, n="{n}", query="{query}")

HIGHLIGHT_GUIDANCE = (
    "Given a citing sentence and a chunk of source text, identify 1-3 short "
    "phrases (3-12 words each) from the chunk that most directly support the "
    "citing sentence. Return ONLY phrases that appear verbatim in the chunk text."
)

HIGHLIGHT_RESPONSE_FORMAT = (
    'Return JSON only, for example: {"phrases": ["phrase1", "phrase2"], '
    '"confidence": 0.8}. Confidence must be a number from 0.0 to 1.0.'
)

HIGHLIGHT_USER_PROMPT = (
    "Citing sentence: {citing_sentence}\n\n"
    "Source chunk:\n{chunk_content}\n\n"
    "Extract 1-3 supporting phrases from the source chunk (must be exact substrings)."
)
