# ADR 0055 — the per-tool-result reserve is a floor, and one result may claim it in full

Status: accepted. Author: the capacity work.

## Context

One number used to answer two questions. `ContextPolicy.dynamic_context_reserve_tokens` was a flat
`40_000`: it was both the reserve Fast admits against and the size at which the runtime cut every
Tool result. On a profile that could afford it the two meanings agreed by accident. On one that
could not, the charge was `min(40_000, hard_input_limit - minimum_input_tokens)`, so a small window
paid its whole budget except the last 1,024 tokens and the history it could keep *was*
`minimum_input_tokens`. The guaranteed room for one Tool result and the model's history budget were
the same budget, and the smaller the profile, the more of it the reserve took.

The second symptom was that a large document had no delivery route at all. The reserve was the only
per-result allowance, so the loader could return no more than that and could not say that what it had
returned was a prefix. A built-in Skill's engine script —
528,878 bytes, 10,623 lines, about 132,031 estimated tokens — arrived as roughly its first third,
and the result said nothing about the fact: the model could not tell a whole document from a prefix,
had no offset to ask for the rest, and had no way to copy one out.
[ADR 0048](0048-skill-document-size-follows-the-industrys-rules.md) had already decided that
scripts are executed, not loaded, and set no size limit on them; the runtime had no way to hand one
over.

## Decision

**1. The reserve is a floor that shrinks with the profile, and never below the flat charge it
replaces.** `dynamic_context_reserve_tokens` stays the configured floor, now 60,000. The effective
reserve (`ContextPolicy.dynamic_context_reserve`, `src/dlightrag/engine/ai/capacity.py:161-181`) is

```
hard     = hard_input_limit(profile)
floor    = min(minimum_input_tokens, hard)
legacy   = min(40_000, max(0, hard - floor))     # the flat charge this change replaces
reserve  = max(0, min(60_000, max(hard // 3, legacy)))
```

and `observation_capacity` returns what the trigger leaves, `hard - min(reserve, max(0, hard -
floor))`: the room a proactive compaction leaves below the hard input limit, and therefore what one
Tool result is guaranteed (`capacity.py:202-210`).

Why a floor rather than a flat number: a constant a small window cannot afford is not a reserve, it
is a rewrite of that model's history budget. Why the third is clamped by `_LEGACY_FLAT_RESERVE_TOKENS`
(`capacity.py:16-19`): shrinking a constant is the one thing a floor may not do. A third alone is
smaller than the flat 40,000 on every mid-size profile — at a 128K window that reserves 16,384 output
tokens, the third is 37,205 against the 40,000 paid before — so a profile's guarantee would *fall*
under a change whose whole purpose is to raise it. The `max` against `legacy` is therefore a
monotonicity guarantee rather than tuning: the reserve is at least what the profile was already
charged, and a third decides only where a third is the larger number. The reserve still never exceeds
60,000, and it reaches that full floor from 180,000 tokens of hard input up.

Why not simply 60,000 flat: on the same 128K profile that would claim 54% of the hard input limit, and
Fast admits against the whole reserve rather than a shrunken one (`require_full_dynamic_reserve=True`,
`src/dlightrag/engine/answer/execution/acceptance.py:189,201`), so a flat 60,000 would leave its
fixed envelope only the 51,616 tokens above it on that profile. The shrink is what lets the floor be
charged in full only where the profile can pay it.

Measured (`uv run python`, the fixtures the tests use): a 128K window reserving 16,384 output tokens →
hard 111,616, reserve 40,000 (the legacy clamp), trigger 71,616, guarantee 40,000, ceiling 40,000;
the same window with a 1,024-token output cap → hard 126,976, reserve 42,325, trigger 84,651,
guarantee 42,325, ceiling 42,325; a 64K window with a 16,000-token output cap → hard 48,000,
reserve 40,000 (legacy), trigger 8,000, guarantee 40,000; a 32K window → hard 15,616, reserve 14,592
(legacy), trigger 1,024; a 200,000-token window with a 180,000-token input cap and a 16,000-token output cap → hard
180,000, reserve 60,000, trigger 120,000, ceiling 60,000.

**2. A result may claim the reserve, and no more.** `ContextPolicy.tool_result_ceiling(profile)` is
`max(0, min(observation_capacity, hard_input_limit - min(minimum_input_tokens, hard_input_limit)))`
(`capacity.py:212-228`): one result may claim exactly what the proactive compaction trigger leaves
it. A ceiling *above* the reserve was tried and retired, because it reintroduced the failure the
number exists to prevent. At three eighths of a 1M window one result could claim 368,856 tokens, so
three full-size results in one exchange measured 1,106,568 against a 923,616 trigger and a 983,616
hard input limit — and `select_compaction_boundary` keeps whole exchanges, so that batch is a turn no
compaction can shed. Holding one result to the reserve makes three of them cost 180,000, comfortably
inside the trigger that then fires and bounds the turn. The reserve is the number a compaction can
bound; anything larger is a bet that the model will not do it three times, which is what
`tests/unit/test_model_capacity.py:107` pins as `3 * tool_result_ceiling < compaction_trigger` on a 1M
profile.

The outer clamp is an invariant rather than an adjustment: `observation_capacity` already returns the
hard input limit less the request floor, so the `min` cannot move a shipped number — it is written so
the definition carries the rule instead of deriving it from the trigger's arithmetic. A window too
small to hold a request therefore exposes a zero ceiling, and a non-empty result there fails loudly
instead of being cut to nothing.

**3. The runtime cuts at the ceiling, and a cut is now an observable fact.** The two
`fit_tool_result` call sites in `ResearchRuntimeEffects` fit to the ceiling
(`src/dlightrag/engine/answer/research/runtime.py:878,891`), which is the guarantee itself, so a
turn's batch of full-size results stays inside the trigger that bounds it. The trigger retains
`min(reserve, hard_input_limit - floor)` and leaves the rest, so a result the size of the guarantee
fits under it by construction — one result plus one compaction still fits on every profile shape.

`fit_tool_result` merges `{"observed_truncated": True}` into the fitted result's `details` when, and
only when, it actually shortened the result
(`src/dlightrag/engine/agent/tools/capacity.py:25-63`); the key and the
`ToolResult.observed_truncated` reader are defined at
`src/dlightrag/engine/agent/tools/contracts.py:42-45,232-240`. It is the runtime's fact and is
never inferred from the text, so a tool that pages its own output leaves it unset. The
`tool_observations` trace entry keeps `capacity_tokens` as the guarantee and gains `ceiling_tokens`
as the ceiling (`runtime.py:909-910`), so a run's trace states both numbers the result was measured
against.

The evidence render budget deliberately receives `observation_capacity`, which is the ceiling as well
(`runtime.py:162-181,865-866`): frozen evidence is framing the runtime adds around passages, and it
is bounded by the number one result is guaranteed rather than by a second, larger one.

**4. A tool that can page, pages; a script is copied, not read.** `load_skill` gains `offset` and
`write_to` (`src/dlightrag/engine/agent/skills.py:138,146`), returns one window sized from the
ceiling, and ends a longer document with a
`[more; load_skill(name='…', path='…', offset=N)]` continuation naming the next offset
(`skills.py:677-720`). The model profile is bound where the tools are declared rather than guessed at
call time (`SkillsBundle.tools`, `skills.py:534`; `load_skill_tool`, `skills.py:774`), and the window is
measured on the *rendered* result, boundary line and continuation included, so the runtime never has
to cut a page. `write_to` copies a document's exact bytes into the Agent Workspace and answers with a
receipt — path, byte count, sha256 — instead of the text (`skills.py:723-771`); it is refused
together with `offset`, because it copies the whole document. `validate_model.py
--expect-engine-sha256` compares digests, so exact bytes are the requirement rather than a courtesy,
and the loader's boundary line is not part of the file. This is how ADR 0048's "scripts are
executed, not loaded" becomes true for a 528,878-byte script: the model copies the engine into the
workspace and runs it there, and none of its 132,031 tokens enter context. At the unified ceiling a
1M profile fits about 60,000 tokens per window, so that engine is three read pages, and one
`write_to` copy. The resource reader pages the same way — its pages are sized by the same guarantee
the fit uses — and keeps its `[more; cursor=…]` continuation
([Resource Reading](../resource-reading.md)).

## Not building

- **A per-batch share.** The evidence render budget states the reasoning the per-result allowance
  keeps: "It is deliberately not shared out across the Tools of a batch and deliberately not derived
  from the compaction trigger: freezing is what keeps a passage reusable, so shrinking it to stay
  under a trigger would trade a permanent prefix for a turn that compaction can bound anyway"
  (`runtime.py:170-175`). The ceiling is that per-Tool allowance; it is not divided across a batch.
- **A silent cut.** Nothing may be shortened without either a continuation the model can follow or
  `observed_truncated` in the result's details. `load_skill` and the resource reader page; every
  other tool is cut at the ceiling, and the cut is recorded.
- **A ceiling above the reserve.** Tried and retired for the reason in Decision 2: three full-size
  results at three eighths of a 1M window measured 1,106,568 tokens against a 923,616 trigger, and
  compaction keeps whole exchanges, so the batch was unsheddable. One result claims the reserve, and
  a window too small to hold a request claims nothing.
- **A second write tool, or a non-replayable `load_skill`.** `write_to` mutates the workspace while
  `load_skill` stays `replay_policy="replayable"`. The copy is idempotent — the same arguments
  produce the same bytes, the same receipt and no accretion — and `load_skill` is a read tool whose
  dominant use is reading, so making it non-replayable would charge every read for the one mutating
  argument. A pending copy that recovery settles as `outcome_unknown` loses a re-read, not a
  capability. Pinned by `test_a_write_to_replays_to_the_same_bytes_and_the_same_receipt` in
  `tests/unit/test_agent_skills.py`.

## Consequences

`CONTEXT_POLICY_REVISION` becomes `agent-v6-absolute-reserves` (`capacity.py:9`). An active Run
pinned to another revision must drain before this build deploys — the existing guard refuses the
mismatch (`src/dlightrag/application/retrieval/execution.py:216-220`) — so there is no resume
across the two policies.

- One result is held to the reserve: 60,000 tokens on a 1M profile rather than the 368,856 a
  three-eighths share allowed, so three full-size results cost 180,000 against a 923,616 trigger. The
  528,878-byte engine is therefore three read pages on that profile, or one `write_to` copy. No
  profile loses capacity against the policy this replaces: the mid-size window whose third would have
  been smaller than its old flat charge keeps the flat charge — a 128K window reserving 16,384 output
  tokens keeps a 40,000-token guarantee rather than 37,205 — and a 64K window keeps its 40,000-token
  guarantee.
- Fast admits against the whole reserve and keeps no history at all
  (`require_full_dynamic_reserve=True`), so its acceptance limit is `hard_input_limit - 60,000`: a
  64K window leaves a 48,000-token hard input limit and a Fast limit of −12,000, and an explicit Fast
  request is refused there rather than served with a shrunken reserve (Research still serves that
  window, with 8,000 tokens of history). No shipped model is affected — the smallest profile in the
  built-in catalogue has a hard input limit of 483,616 — and both the refusal and the unchanged
  Research behaviour are pinned by `test_fast_refuses_a_profile_that_cannot_hold_the_dynamic_reserve`
  in `tests/unit/test_answer_acceptance_budget.py`.
- `docs/retrieval-answer.md` and `docs/resource-reading.md` state the floor, the shrink and that the
  fit is that same number; a built-in Skill's hub tells the model to copy a script with
  `write_to` and run it, and to read a large reference in offset pages.
- The old failure is a test rather than a story: a page the loader returns fits
  `tool_result_ceiling` and carries no `observed_truncated`, and following the continuations
  reconstructs the document byte for byte.
