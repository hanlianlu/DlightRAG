# Prompt Profile Design

## Goal

Create a code-only prompt profile boundary for DlightRAG so the product has one
clear core identity and compact task-specific guidance, without exposing prompt
controls through configuration, API routes, or user inputs.

## Context

DlightRAG keeps product-owned prompt text centralized in `dlightrag.prompts`.
The prompt profile boundary is intentionally code-only: answer, planner,
rerank, and citation-highlight prompts share one core identity and compose
their task-specific behavior from guidance fragments.

This is serviceable for a generic document RAG system, but it makes domain
porting harder because developers must search across runtime modules to find the
prompt assumptions. The new boundary makes the generic document identity explicit
while keeping all prompt authority inside source code.

## Design Principles

- One core identity only. DlightRAG should sound like one product, not several
  agents with separate personas.
- Task-specific differences belong in guidance, not identity.
- Prompt text is high-privilege behavior. Do not add config, API, workspace, or
  user-controlled prompt profile loading.
- Keep the prompt layer dense. Move only behavior-relevant text, and avoid adding
  generic instructions that the model already follows reliably.
- Preserve current behavior as much as possible. This is a boundary extraction,
  not a prompt rewrite.
- Keep citation and structured-output contracts explicit because they are
  product invariants, not style preferences.

## Proposed File Boundaries

### `src/dlightrag/prompts/identity.py`

Owns the single product-level identity:

```python
CORE_IDENTITY = (
    "You are a rigorous document-grounded analysis assistant. "
    "You answer from provided evidence, preserve uncertainty, and avoid "
    "unsupported claims."
)
```

The exact text can stay close to the current answer identity. The important
property is that there is only one identity constant and every prompt path that
needs a system role starts from it.

### `src/dlightrag/prompts/guidance.py`

Owns compact task guidance that is specific to a prompt path:

- `ANSWER_CONTEXT_GUIDANCE`: what context the model receives and how to use KG,
  text excerpts, page images, and the reference list.
- `CITATION_GUIDANCE`: inline `[n]` / `[n-m]` citation rules and the rule against
  generating a references or sources section.
- `ANSWER_CITATION_EXAMPLE`: the existing short inline citation example.
- `PLANNER_GUIDANCE`: the existing planner JSON contract, filter evidence rules,
  metadata filter rules, and examples.
- `RERANK_GUIDANCE`: the listwise relevance scoring contract.
- `HIGHLIGHT_GUIDANCE`: the support-phrase extraction contract.

The guidance module is not a dumping ground for all prose. It only stores prompt
fragments that define behavior and are currently embedded in prompt strings.

### `src/dlightrag/prompts/rag.py`

Continues to export the existing public prompt names:

- `ANSWER_CORE`
- `get_answer_system_prompt()`
- `VISUAL_RERANK_PROMPT`
- `HIGHLIGHT_SYSTEM_PROMPT`
- `HIGHLIGHT_USER_PROMPT`

It assembles final prompt strings from `CORE_IDENTITY` plus guidance fragments.
Call sites should not need signature changes.

### `src/dlightrag/prompts/planner.py`

Continues to export:

- `PLANNER_SYSTEM_PROMPT`
- `PLANNER_HISTORY_TEMPLATE`
- `PLANNER_NO_HISTORY_TEMPLATE`

The planner prompt should start with `CORE_IDENTITY`, then add planner-specific
guidance. The JSON schema placeholders remain exactly where `QueryPlanner`
expects them.

### Runtime Modules

`src/dlightrag/models/rerank.py` should import the centralized listwise rerank
template instead of owning `_LISTWISE_PROMPT`.

## Non-Goals

- No external prompt profile file.
- No `config.yaml` prompt profile setting.
- No API route or workspace-level prompt override.
- No user-provided system prompt injection.
- No built-in multi-profile registry.
- No broad copyediting of prompt wording beyond what is required for clean
  assembly.

## Expected Data Flow

Answer generation:

1. `AnswerEngine.generate()` and `generate_stream()` call
   `get_answer_system_prompt()`.
2. `get_answer_system_prompt()` returns `ANSWER_CORE`.
3. `ANSWER_CORE` is assembled from `CORE_IDENTITY`, answer context guidance,
   citation guidance, and the existing example.

Query planning:

1. `QueryPlanner.plan()` formats `PLANNER_SYSTEM_PROMPT` with schema, custom key,
   and history sections.
2. `PLANNER_SYSTEM_PROMPT` includes `CORE_IDENTITY` plus planner guidance.
3. Structured output parsing remains unchanged.

Reranking and highlighting:

1. Rerank scoring uses the centralized rerank prompt template.
2. Citation highlighting uses the centralized highlight system guidance and the
   existing user prompt template.

## Error Handling

This change introduces no runtime loading, parsing, or user-configurable prompt
surface, so it should not add new runtime error modes. Import errors and
formatting mistakes are caught by unit tests and normal Python import checks.

## Testing Plan

Use TDD for implementation:

1. Add tests that import `CORE_IDENTITY` and verify `get_answer_system_prompt()`
   includes it.
2. Add tests that verify the answer prompt still includes inline citation rules,
   the existing `[1-1]` citation example, and does not ask for JSON or a
   references section.
3. Add planner prompt tests that verify `PLANNER_SYSTEM_PROMPT` includes
   `CORE_IDENTITY` while preserving the JSON/filter evidence contract.
4. Add rerank prompt tests that verify the listwise prompt remains formatted with
   query and item count.

After implementation, run the focused prompt, planner, and rerank unit tests.
Run the full unit test suite if the focused tests reveal any shared prompt
assembly risk.

## Migration Notes

Existing imports from `dlightrag.prompts` should keep working. New constants from
`identity.py` and `guidance.py` may be re-exported from `prompts/__init__.py` only
when tests or internal modules need them. Public runtime APIs do not change.

## Acceptance Criteria

- DlightRAG has exactly one core identity constant.
- Answer, planner, rerank, and highlight prompts assemble from the centralized
  identity or guidance modules.
- Existing prompt exports and call signatures remain compatible.
- No prompt profile configuration or API surface is added.
- Focused tests prove the new boundary and preserve existing prompt contracts.
