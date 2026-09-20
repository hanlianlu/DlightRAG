# Response API Family qualification

The common transport is implemented. Bounded live checks cover the two
endpoint/model pairs below; official OpenAI remains experimental and mock-tested.
[Configuration](configuration.md#api-family) owns current role selection;
[ADR 0027](adr/0027-api-family-selects-the-provider-wire.md) owns the decision.

## Verification matrix

| Scope | Evidence and limits |
|---|---|
| Direct DeepSeek `deepseek-flash` at `https://api.deepseek.com/responses` | Live text, JSON Object, streaming, reasoning replay, two local function calls and results, user images, Tool-image transport and stream close exercised. Image-answer accuracy varied; see below. The `/v1/responses` alias was not tested. |
| OpenRouter `z-ai/glm-5.3-flash` at `https://openrouter.ai/api/v1/responses` | Independently exercised the same functional cases. A complete two-call loop passed; another request returned text without calls under `tool_choice=auto`. No other model or upstream route is qualified by this result. |
| Official OpenAI | Installed-SDK HTTP/SSE contracts cover all five entrypoints, structured output, image ownership, finalized encrypted reasoning, assistant phases and failure terminals. **No official API key was available; not live-qualified.** |
| Recovery and accounting | Regression tests cover settled effects across restart, selected-branch Fork, whole-exchange compaction, cross-family replay stripping, provider input anchoring and cache-hit aggregation. |

## Method and delivery evidence

The 2026-09-20 checks used OpenAI Python SDK **3.3.0**, synthetic arithmetic,
locally generated PNGs and read-only local functions. Chat comparisons used the
same endpoint/model. Each matrix invocation allowed at most 16 HTTP attempts
per endpoint/family, zero SDK retries, a 60-second request timeout and 2048 output
tokens per request. The reasoning replay case used typed `high`; other matrix
cases used `low`.

Request checks verified full local input, `store=false`, `background=false`,
`truncation=disabled`, no remote continuation fields, and exact predecessor
reasoning replay even when the earlier turn made no Tool call. Recorded evidence
contains safe counters and outcomes, not credentials, model text or reasoning
payloads. Invalid early harness verdicts are excluded.

- After the configured role switch at `6950cfa3`, both selected roles passed
  structured-output and agentic-stream smokes from the recreated API container
  using their configured temperature/reasoning. API and MCP resolved the intended
  families; `/health` and `/ready` passed.
- [CI run 35501879849](https://github.com/hanlianlu/DlightRAG/actions/runs/35501879849)
  at that commit concluded **success** for fast, integration and browser-e2e.
  Local `make ci`: 4758 passed, 5 skipped. This is delivery evidence, not a
  substitute for the separate live checks.
- Reproducible offline contracts live in
  [provider attachment tests](../tests/unit/test_provider_attachment_contract.py),
  [provider tests](../tests/unit/test_providers.py),
  [replay tests](../tests/unit/test_replay.py),
  [restart tests](../tests/unit/test_agent_session_runtime.py),
  [Fork tests](../tests/unit/test_agent_session_tree.py),
  [compaction tests](../tests/unit/test_compaction.py),
  [input accounting tests](../tests/unit/test_answer_orchestrator.py), and
  [Run trace tests](../tests/unit/test_research_runtime_migration.py).

## Quality observations and remaining limits

- DeepSeek's repeated red Tool-image task passed twice and failed twice. An
  unhinted three-color comparison scored user images 3/3 in both families and
  Tool images 1/3 on Chat versus 2/3 on Response. These are answer-quality
  observations, not a demonstrated Responses-specific defect; the exact cause
  was not isolated.
- OpenRouter's zero-call answer did not follow the task instruction, but
  `tool_choice=auto` permits a text-only response. No call was shown to have been
  lost by the adapter. These observations do not add protocol acceptance or
  rollout gates, and the successful samples do not guarantee model accuracy.
- The model selects requests for **local** functions; authorization, execution
  and settlement remain local. No hosted provider tools were enabled.
- Short synthetic checks establish neither performance/cost superiority nor
  long-session accounting accuracy or stable OpenRouter cross-route behavior.
  Stream close does not prove upstream computation or billing stopped; failure,
  cancellation and scheduler-release guarantees are tested offline.
- `store=false` minimizes remote response state; **it is not ZDR**. Official
  OpenAI live qualification awaits an official key; Codex login and OpenRouter
  models are not substitutes.
