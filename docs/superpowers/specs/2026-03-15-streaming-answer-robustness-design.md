# Streaming Answer Robustness Design

## Problem

The current streaming answer path uses a JSON-parsing state machine (`StreamingAnswerParser`)
that assumes the LLM will output valid `{"answer": "...", "references": [...]}` JSON. When the
LLM doesn't comply (common with smaller models or providers that don't enforce `response_format`
in streaming), three failures occur:

1. **Raw JSON leaks to user** — LLM outputs freetext followed by a JSON block; the parser
   switches to PASSTHROUGH and dumps everything (including raw JSON) to the user.
2. **Prompt mismatch** — `structured=True` uses the JSON prompt (no `### References` instruction),
   so when the parser degrades, the freetext fallback can't extract references either.
3. **References lost** — PASSTHROUGH `finish()` returns empty references; the consumer-side
   fallback (`parse_freetext_references`) also fails because the prompt didn't ask for
   `### References`.

Additionally, the non-streaming path has a gap: when `structured=True` but JSON parsing fails,
the fallback uses `parse_freetext_references` on text that was prompted for JSON, not freetext.

## Design

### Principle: streaming = freetext, non-streaming = structured

This follows production RAG system patterns (Perplexity, etc.): stream answer text directly
for real-time display, extract references post-stream. JSON wrapping is the wrong format for
streaming long-form answers — the `{"answer": "` prefix and `", "references": [...]}` suffix
are implementation artifacts.

### Architecture

```
Non-streaming generate()
├─ supports_structured=True
│   └─ response_format=StructuredAnswer → JSON parse
│       └─ failure → extract_references(raw)  [3-level fallback]
└─ supports_structured=False
    └─ freetext prompt → extract_references(raw)

Streaming generate_stream()  [ALL providers]
└─ freetext prompt (always, ignore supports_structured)
    └─ AnswerStream: passthrough tokens → post-stream extract_references(full_text)
```

### Component 1: `extract_references()` — unified extraction function

**Location**: `src/dlightrag/citations/parser.py`

```python
def extract_references(raw: str) -> tuple[str, list[Reference]]:
    """Extract references from LLM output with 3-level fallback.

    Returns (cleaned_answer_text, references).

    Level 1 — JSON block: find and parse JSON containing "references" array,
              strip JSON block from answer text.
    Level 2 — ### References section: parse [n] Title lines,
              strip references section from answer text.
    Level 3 — Empty: return original text + empty list.
    """
```

**Level 1 — JSON block extraction**:
- Use existing `extract_json()` to find JSON in the raw text.
- Parse `{"answer": "...", "references": [...]}` or just `{"references": [...]}`.
- If JSON has an `"answer"` key, use that as the clean text.
- Otherwise, strip the JSON block from the original raw text.
- Validate each reference via `Reference.model_validate()`.

**Level 2 — `### References` section**:
- Current `parse_freetext_references()` logic: find `### References` heading,
  parse `[n] Title` lines.
- Return answer text with references section stripped.

**Level 3 — Empty**:
- Return original text unchanged + empty list.

### Component 2: Simplified `AnswerStream`

**Location**: `src/dlightrag/models/streaming.py`

Delete `StreamingAnswerParser` (the JSON state machine). Replace with a simplified
`AnswerStream` that:

1. Passes through all tokens as-is (no buffering, no JSON detection).
2. Accumulates the full answer text.
3. On stream completion, calls `extract_references(full_text)` to populate
   `self.references` and `self.answer` (cleaned text).

```python
class AnswerStream(AsyncIterator[str]):
    def __init__(self, raw_iterator: AsyncIterator[str]) -> None:
        self.references: list[Reference] = []
        self.answer: str = ""
        self._parts: list[str] = []
        self._raw = raw_iterator
        self._gen = self._iterate()

    async def _iterate(self) -> AsyncIterator[str]:  # type: ignore[override]
        async for chunk in self._raw:
            self._parts.append(chunk)
            yield chunk
        full = "".join(self._parts)
        self.answer, self.references = extract_references(full)
```

**Important — streaming limitation**: In streaming mode, tokens are yielded to the user
in real-time. If the LLM appends a JSON block or `### References` section, those tokens
will have already been displayed to the user before `extract_references` can strip them.
The `self.answer` field contains the cleaned text (for post-stream processing like
`CitationProcessor`), but the SSE-streamed tokens cannot be un-sent. This is an inherent
limitation of passthrough streaming and is acceptable — the freetext prompt minimizes
the chance of JSON blocks appearing.

**`StructuredAnswer` model scope**: After this change, `StructuredAnswer` is only used
in the non-streaming `_parse_response` path. The streaming path no longer produces
`StructuredAnswer` objects.

### Component 3: `generate_stream` changes

**Location**: `src/dlightrag/core/answer.py`

- Always use `get_answer_system_prompt(structured=False)` for streaming,
  regardless of `supports_structured`.
- Always wrap with `AnswerStream` (since it's now just a passthrough + post-processing,
  no cost to always wrapping).
- Remove the `if structured` branching for streaming.

### Component 4: `_parse_response` changes

**Location**: `src/dlightrag/core/answer.py`

- Structured path JSON parse failure: replace `parse_freetext_references(raw)`
  with `extract_references(raw)`.
- Freetext path: replace `parse_freetext_references(raw)` with `extract_references(raw)`.
- This makes both paths use the same 3-level fallback.

### Component 5: Consumer cleanup

**Locations**: `src/dlightrag/api/server.py`, `src/dlightrag/web/routes.py`

Remove the manual fallback logic:

```python
# DELETE these lines:
refs = getattr(token_iter, "references", None)
if not refs and full_answer:
    full_answer, parsed_refs = parse_freetext_references(full_answer)
    refs = parsed_refs or None
```

Replace with reading `token_iter.references` via `getattr(token_iter, "references", None)`
for safety (mock iterators in tests may not have the attribute). Use `token_iter.answer`
(cleaned text) instead of locally accumulated `full_answer` for post-stream processing
(citation processing, source building) so that stripped JSON blocks and reference sections
don't leak into downstream processing.

## Files Changed

| File | Change |
|------|--------|
| `src/dlightrag/citations/parser.py` | Add `extract_references()`; keep `parse_freetext_references` as deprecated alias |
| `src/dlightrag/models/streaming.py` | Delete `StreamingAnswerParser`, simplify `AnswerStream` |
| `src/dlightrag/core/answer.py` | `generate_stream` always freetext; `_parse_response` uses `extract_references` |
| `src/dlightrag/api/server.py` | Remove manual ref fallback; use `token_iter.answer` for post-stream processing |
| `src/dlightrag/web/routes.py` | Remove manual ref fallback; use `token_iter.answer` for post-stream processing |
| `tests/unit/test_streaming_parser.py` | Rewrite for simplified AnswerStream (existing file) |
| `tests/unit/test_answer_engine.py` | Update streaming wrapping tests (3 tests affected) |
| `tests/unit/test_extract_references.py` | New: test 3-level fallback |

## Non-goals

- Modifying LightRAG upstream's `openai_complete_if_cache` for streaming `response_format` support.
- Changing the non-streaming structured prompt or `response_format` logic (it works correctly).
- Real-time reference streaming (references are always post-stream; this is standard practice).
