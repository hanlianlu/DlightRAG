# Prompt-prefix stability and cache-anchored context accounting

One Research request is the previous request plus new material: the Session fold only appends, each Tool's admitted evidence is frozen into that Tool's own durable result, and the Run's clock and control instruction ride *after* the transcript. The compaction trigger is measured against what the provider billed rather than only against the character estimator, and each turn's billed prompt and cache hits are aggregated into the Run trace.

## Status

Accepted and implemented. No configuration key: this is a cost and correctness property of composition, and both values it changes (evidence reachable in the request, and evidence reachable after compaction) stay inside the retired design's own vocabulary — evidence is still citable, still one request's factual memory, and still never a summary.

## Context

A prefix cache is billed, not assumed. DeepSeek's context cache persists *cache prefix units* and reports `prompt_cache_hit_tokens`; a later request can only hit the cache when it fully matches a persisted unit. Measured on this deployment with one model and endpoint (`deepseek-flash`, `api.deepseek.com`):

- Agent Session `3c9f9f19` (9 turns): 505,850 prompt tokens, **0% cache hit on every turn**. `0eaa40ad` (15 turns): 1,166,820 prompt tokens, 15% hit.
- The same model in Pi: 4,695 turns, 1,506,964,818 prompt tokens, **99.71% cache hit**; average billed miss per turn 933 tokens against a 320,972-token prompt.

Two composition decisions produced that gap, and neither is about context *size*:

1. **A minute-resolution clock in the system prompt.** `core_identity()` stated `The current time is {now:%Y-%m-%d %H:%M} UTC` at offset ~60 tokens of the system message, and the Research assembler rebuilt that message on every turn. Every turn whose request crossed a minute boundary therefore began a new prompt prefix: the 9 turns above ran 1-3 minutes apart (`agentic_reasoning: max` emitted 20k-35k reasoning tokens per turn) and reported zero hits, while turns inside one minute reported 56,064 and 62,848 hits of a 63,645 and 82,466-token prompt.
2. **The evidence pack was re-rendered after the growing fold.** Every turn rebuilt the whole accumulated ledger into one trailing user message. A later request could not be a prefix of an earlier one whenever new material preceded that pack, so the pack was re-billed at the full input rate on every turn. In `0eaa40ad` the reusable prefix was only the fold — 7,680 of 76,057 tokens on one turn, 11,008 of 105,614 on another — while the 49k-90k pack was charged again. The pack also duplicated resource-backed passages that the Tool result had already shown.

The estimator is the third input to the same decision. `ContextPolicy` carries no safety margin against it and says why: the character heuristic "undercounts recorded session content by a median of 8% and up to 46%", and anchoring on the provider's reported usage was "deliberately not done here". Pi's equivalent anchors on the last assistant message's own usage and estimates only the trailing messages. Nothing in the product read `prompt_cache_hit_tokens` at all: the counters reached Langfuse as raw usage fields and no ratio, notice, or per-Run summary existed.

## Decision

- **The system prompt is byte-stable.** `core_identity()` and `answer_core()` carry no clock. The clock is `clock_line(as_of)`, frozen per Run and carried as the second-to-last message of every Research request — immediately before the control instruction, which stays last so the final thing the model reads is what to do next — or immediately before the knowledge-graph and question block of the Fast user message. A value that changes there costs one message; a value that changes earlier costs the prompt.
- **Evidence text is frozen into the Tool result that admitted it.** `EvidenceLedger.take_admitted_text` renders only the rows admitted since the previous call, bounded by the same observation capacity the Runtime already uses — divided across the Tool batch still to run — with the batch's older rows collapsing to the re-readable handles. `ResearchRuntimeEffects` splices the result into that Tool's model-visible text, ordered so the Citation Contract's "label directly above the excerpt" holds: labels for passages the Tool's own text already carries, that text, then the passages it does not. A read is never carried twice; a body shorter than the classification threshold is rendered rather than assumed shown. The freeze is keyed by the Effect Intent that admitted the rows, so re-executing that intent reproduces the same bytes, and `rollback_show` retracts one freeze a refused or over-long Tool result could not use.
- **The per-request evidence pack is retired.** `ContextAssembler` no longer contributes accumulated evidence; the request is the transcript plus its trailing control instruction and clock. `observation_residual` and the pack's residual loop go with it: bounding the request is the compaction trigger's job, not composition's.
- **Evidence images keep their own lane.** `_durable_row` drops `image_data`, so pixels cannot enter the transcript the way text does. `EvidenceLedger.visual_blocks` renders them per request, bounded by the resolved image budget. This is the one place a Research request still re-renders, and it is bounded by policy rather than by corpus size.
- **Compaction keeps the compacted sources re-readable.** The framework fields `paths`/`durable_handles` were emptied when their Tool-Argument authority was withdrawn. `durable_handles` is now populated from the run's Evidence ledger — the record of what the run actually admitted — deduplicated, capped, and rendered as a list.
- **The compaction trigger is anchored on the provider.** `ContextAssembler.observe_provider_input` carries forward the gap between what the provider billed for the previous request and what this assembler measured for that same request, capped at one raw estimate, and skips requests that carried pixels because the estimator charges nothing for them by design. `accounted_input_tokens` is the measurement composition and the orchestrator's trigger both use.
- **Cache hits are observable.** `ResearchRuntimeEffects.call_provider` aggregates each turn's billed prompt and reported hits through `_record_prompt_cache` into `trace["prompt_cache"]` (`turns`, `prompt_tokens`, `cache_hit_tokens`, `cold_turns`), and a turn after the first whose large prompt reports a zero hit warns once with the billed size. A provider that reports no cache counters at all is not a cold turn.

## Considered options

- **Keep the clock in the system prompt at day resolution.** Rejected: it only moves the invalidation frequency from per-minute to per-day, and the same turn that crosses midnight pays the whole prompt. The clock belongs where it is cheap.
- **Freeze the clock once per Session instead of per Run.** Rejected: it moves the invalidation to the message that precedes the transcript for the *next* Run's conversation tail, and the tail is exactly what the cache is supposed to reuse.
- **Render the frozen evidence as its own durable Session Entry next to the Tool result.** Rejected: it is the same transcript, one more entry type, and one more way for a recovery to disagree with the Tool result it belongs to. Freezing inside the result the Runtime already settles atomically keeps replay exact by construction.
- **Keep the pack for its "everything, always visible" property.** Rejected as unaffordable and partly false: the pack already collapsed to handles under budget pressure, and it was re-billed per turn while the fold stayed cached. Evidence now reaches the model where and when it arrived, and after a compaction its handles remain.
- **Make the evidence shape a configuration key.** Rejected: a switch here would keep two prompt shapes, two test matrices, and two sets of measured behaviour alive for a decision that is strictly cheaper and no less grounded.
- **Track the freeze with a plain cursor.** Rejected: a resume that re-executes an intent whose effect was still pending finds the cursor already past its rows, renders nothing, and commits the Tool's bare body without its citation labels. Keying the freeze by intent costs one small map and makes the replay reproduce itself.
- **Give every Tool in a batch the full observation capacity.** Rejected: the batch is interpreted Tool by Tool with no compaction between them, so three searches could add three reserves in one turn. Sharing one capacity across the remaining Tools keeps the batch's total inside the reserve the policy intended.
- **Anchoring the estimator on the provider without the pixel guard.** Rejected: the estimator deliberately charges no tokens for image blocks, so the gap would measure the provider's image accounting and inflate the trigger.

## Consequences

A turn's request is now the previous turn's request plus its own new material, so a prefix cache can reuse everything except the newly admitted evidence and the two trailing messages. The 9-turn run above would have reused 76-88% of each prompt instead of 0% in the same-minute case and 0% across minutes for the clock reason; the two changes are independent and both apply.

Evidence text that a compaction covers is replaced by its handles rather than re-materialized, so a resumed long Run can read a source again but does not automatically re-see its passage. Compaction remains the one deliberate invalidation.

The control instruction is now present on every Research turn, including turns before any evidence was admitted; the retired composition attached it to the evidence pack, so a run that had admitted nothing sent no trailing message at all and its newest Tool observation was last. Being unconditional also makes the request shape uniform across turns, which is what the append-only property wants. The integration fixtures that read a provider request's last message to act on a Tool's own observation had to find the newest Tool result by role instead. Evidence pixels remain a per-request lane and their bytes remain run-local, unchanged in kind from before — a recovered Session still cannot re-render them, which is now stated where it is decided instead of implied by the pack.

The estimator no longer has to be honest on its own: a systematic undercount is corrected against the provider's own number, bounded so a single bad anchor cannot more than double the accounted input. `measure_control_input` stays pure for the fixed-envelope floor check; only request assembly records what the provider is about to answer.

A Tool result the durable store refuses, or one that would have been truncated, gives its frozen text back: those rows stay admitted and render with a later Tool result rather than disappearing with the refused one. Every other Tool result keeps its own evidence bytes, which is what makes the transcript replay exact.

A recovered Session starts a fresh assembler, so its first turn is measured by the
estimator alone and the anchor resumes after that turn; a compaction inside one
turn re-records the pre-compaction measure it was decided from, so the next anchor
compares a billed count against a request that was never sent and the correction
falls back to zero for one turn. Both directions are conservative and self-healing:
neither can leave the trigger anchored on a stale, larger number.

Stop conditions, to revisit this decision rather than extend it silently: a provider whose cache prefix units do not match append-only requests; evidence batches regularly saturating the observation capacity so that most passages arrive as handles; the visual lane growing past its image budget; or a measured Run where the anchored trigger compacts before the provider's own limit would have required it.
