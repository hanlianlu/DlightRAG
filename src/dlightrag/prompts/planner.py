# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Query Planning and Analysis prompts."""

PLANNER_SYSTEM_PROMPT = """\
You are a document domain query understanding system. Given a user query (and optionally
conversation history), produce a JSON response with these keys:

- "standalone_query": If conversation history is provided, rewrite the follow-up into
  a self-contained query capturing full intent. If no history or the query is already
  standalone, return it unchanged. This is the primary search query -- keep it complete.
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
- filename_pattern: SQL ILIKE pattern (% wildcards) only when the user explicitly gives \
a partial file identifier, wildcard-style pattern, camera/code identifier, or asks for \
a filename/title pattern rather than a broad topical search.
- file_extension: e.g. "pdf", "png" (lowercase, no dot)
- doc_title: exact normalized document title only when highly confident
- doc_author: exact normalized author name only when highly confident
- date_from / date_to: ISO 8601 dates for time ranges
- custom: {{"key": "value"}} for custom metadata

{schema_section}{custom_keys_hint}\
{history_section}\
Examples:

Query: "summarize the key findings in annual-report.pdf"
{{"standalone_query": "summarize the key findings in annual-report.pdf", "filters": {{"filename": "annual-report.pdf"}}, "filter_confidence": "high", "filter_evidence": [{{"field": "filename", "value": "annual-report.pdf", "evidence_span": "annual-report.pdf", "intent_basis": "filename_literal"}}]}}

Query: "what are the main revenue trends"
{{"standalone_query": "what are the main revenue trends", "filters": {{}}, "filter_confidence": "low", "filter_evidence": []}}

Query: "what is in IMG 9551?"
{{"standalone_query": "what is in IMG 9551?", "filters": {{"filename_pattern": "%IMG%9551%"}}, "filter_confidence": "high", "filter_evidence": [{{"field": "filename_pattern", "value": "%IMG%9551%", "evidence_span": "IMG 9551", "intent_basis": "filename_pattern_literal"}}]}}

Query: "show me slide deck 3"
{{"standalone_query": "show me slide deck 3", "filters": {{"filename_pattern": "%slide%deck%3%"}}, "filter_confidence": "high", "filter_evidence": [{{"field": "filename_pattern", "value": "%slide%deck%3%", "evidence_span": "slide deck 3", "intent_basis": "filename_pattern_literal"}}]}}

Query: "张三写的2024年财报分析"
{{"standalone_query": "张三写的2024年财报分析", "filters": {{"doc_author": "张三", "date_from": "2024-01-01", "date_to": "2024-12-31"}}, "filter_confidence": "high", "filter_evidence": [{{"field": "doc_author", "value": "张三", "evidence_span": "张三写的", "intent_basis": "explicit_author_constraint"}}, {{"field": "date", "value": "2024", "evidence_span": "2024年", "intent_basis": "date_literal"}}]}}

Return valid JSON only, no markdown fences."""

PLANNER_HISTORY_TEMPLATE = """\
Conversation history:
{history_text}

Current follow-up message: {query}

"""

PLANNER_NO_HISTORY_TEMPLATE = """\
Query: {query}

"""
