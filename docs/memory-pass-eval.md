# Session Memory Pass — the judgement eval

The Session Memory Pass ([ADR 0045](adr/0045-profile-memory-is-maintained-per-session-and-owned-across-them.md))
has an ordinary correct output of nothing, so a pass that never writes looks exactly like a pass that works. This
suite separates the two: it replays fixed conversations through the production request path — the same
`consolidation_request`, the same `CompletionModel` request shape, the same `_VERDICT.parse` and
`plan_consolidation` — against several candidate models, and grades what each would have written.

## Cases

Two case files share one schema (`scripts/memory_pass_eval/synthetic_cases.json` documents it by example):

- **Synthetic** (committed, 24 cases). Seven positives — standing preferences stated outright or in passing, owner
  facts, a tool experience, a preference buried in a long assistant-heavy task. Nine negatives — one-off and
  repeated in-task instructions, research findings, a secret, a document's rules, a prompt injection in pasted web
  text, a third party's preference, and a sub-task's output mislabelled as a user turn. Six that exercise the
  existing profile — update, contradiction, fuse, restatement, a memory this conversation already contributed, and
  more distinct preferences than one model call may add. Two that exercise the earlier turns given as context — a
  standing preference the newest part states only as "这样", which only the context can explain, and a preference
  the earlier turns state that this conversation already contributed, followed by a plain task (no write, no
  duplicate from the context).
- **Real** (private, never committed). Conversation slices the pass actually read, recovered from traces and kept
  under the gitignored `docs/research/memory-pass-eval/real_cases.json`. Real conversations are where the
  preferences are implicit — a complaint about how the work was done, a demand repeated across Sessions — and
  that is the half the synthetic cases cannot stand in for.

A case may also list `context_turns` (the same shape as `turns`, empty by default): the turns just before `turns`,
which the pass shows its model as read-only context, as production does for the turns before a batch — user turns
whole, assistant turns cut to their first 500 characters, whatever the case lists — and which an earlier pass is taken to have
judged already. The judge sees neither `turns` nor `context_turns`.

Each case lists **must** items (a durable, general, owner-own memory the pass should write), **may** items
(defensible either way, never penalised), **must not** entries, and for profile cases the expected action and
targets. Labels are written before any candidate is run. A memory a case already holds may carry an `updated_at`, which
the pass shows as when it was last written (to the hour, UTC); one without it is shown with no time. The pass lists
its memories by number and the model's verdict names them by number, so a saved run's raw `verdict` carries numbers
and its `planned` writes carry the memory ids that `expect_actions` is scored against. A memory a case already
holds may also carry a `quote`, the owner's own words behind it, which the pass lists under the memory as it does in
production; one without it shows none.

## Grading

An LLM judge (`anthropic/claude-opus-5.5`, never a candidate) maps each planned write to a must or may item, to
"acceptable" (unlisted but a sound memory), or to "violation" (a must-not, a task detail, a finding, a document's
claim, a secret, a third party's view, or an instruction instead of a statement). The judge sees the gold list and
the writes, not the conversation, so an "acceptable" write is checked by reading it, not trusted. Expected actions
are checked mechanically against the planned targets. A write's `quote` is not part of what the judge sees or of
its cache key: it is the owner's evidence for a memory, and the memory is what is graded. It is measured beside the
grade instead. The summary reports, per config, `quoted` (the share of planned writes that kept the owner's words
behind them, after the host's check that the quote is a stretch of one of the newest part's user turns) and `quotes
dropped` (the quotes the model gave that the host refused as not the owner's, whether or not the write itself
survived planning): a high count of the second is a model that paraphrases or quotes the Agent, which the host
absorbs at the cost of the quote.

## How to run

```
uv run python scripts/run_memory_pass_eval.py --label prod
uv run python scripts/run_memory_pass_eval.py --configs ds-flash-on --system-prompt-file p.txt --label p2
uv run python scripts/run_memory_pass_eval.py --rescore docs/research/memory-pass-eval/runs/<dir>
```

Every config × case × three repetitions; results, judgements and `summary.md` land under
`docs/research/memory-pass-eval/runs/`. A rerun into the same directory resumes. `--print-production-prompt`
prints the shipped prompt as a starting point for a variant.

## Findings, 2026-10-17

Production prompt, 40 cases × 3 repetitions:

| config | must recall | real | synthetic | violations |
|---|---|---|---|---|
| `ds-flash-off` (production: deepseek-flash, thinking off) | 69% | **0%** | 95% | 0 |
| `ds-flash-on` (same model, thinking on) | 93% | 73% | 100% | 1 |
| `ds-pro` (deepseek-v4-pro, thinking off) | 67% | 0% | 92% | 0 |
| `glm-flash` (reasoning cannot be disabled) | 93% | 73% | 100% | 8 |
| `gemini-flash` (reasoning cannot be disabled) | 96% | 93% | 97% | 3 |
| `sonnet` (quality reference) | 94% | 80% | 100% | 0 |

- Without reasoning, DeepSeek catches every outright statement ("以后都…", "from now on…") and none of the real
  ones. The real preferences arrive as reactions to the work inside long task conversations, and a model that does
  not reason before answering takes the prompt's "returning no writes is the normal answer" at its word.
- Thinking on the same model recovers them at no material cost; the pass runs detached, so its latency (about 5 s)
  is invisible.
- A prompt that points the model at the owner's reactions to how the work is done, and away from what the task is
  about, lifts real recall to 87–100% with thinking on. Without thinking it reaches only 40%, so the prompt does
  not replace reasoning.
- What remains wrong is over-specific memory: a skill described down to its files, and environment facts about
  the Agent's own Agentspace (`tmp/` is wiped, no resvg) recorded as experience. No candidate on DeepSeek wrote a
  secret, a finding, an injected instruction, a third party's preference, or anything from the mislabelled
  sub-task output.

**Adopted.** The pass now calls the `query` role at its own `memory_consolidation.reasoning` (low), with the
revised prompt, which also keeps a title fixed for one deliverable with that deliverable and leaves the Agent's own
Agentspace out of experience. That configuration (`query-pass`), 40 cases × 3 repetitions:

| config | must recall | real | synthetic | violations | expected actions | mean latency |
|---|---|---|---|---|---|---|
| `query-pass` (query role, reasoning low, revised prompt) | 98% | 93% | 100% | 0 | 9/9 | 1.8 s |

Low effort is enough: it recalls what high effort did on this set, with no violations and at half the latency.

**No trigger.** Every successful Run's pass now reads all new text, with the turns before it as context. The
same configuration over 42 cases (C1 and C2 exercise the context) × 6 repetitions: 97% must recall, real 27/30,
synthetic 84/84, expected actions 18/18, and 3 violations in 252 runs — a skill recorded down to its files, a
preference that kept the owner's irritation, and one misread of "看不懂" as a dislike of charts. Recall on the real
slices stays within the run-to-run spread of the earlier runs.
