# ADR 0072 — The edit tool addresses a region, not only a string

## Status

Accepted by the owner on 2026-10-25 for the v1 scope M0–M4, and implemented: the resolver and
hunk renderer in `tools/editing.py`, contract v4 on the `edit` Tool, the Skill's filling procedure,
and the model-free acceptance run (`scripts/run_edit_eval.py`, `make edit-eval`). M5
(multi-language symbols) is planned, with a named default and a spike gate, and changes no part of
the contract this decision fixes.

## Context


The `edit` tool could only replace an exact `old_text`. To fill a section through it, a model has
to reproduce the whole old block verbatim: thousands of characters of text that the workspace text
layer already holds, in exactly the layout it happens to use. In practice models avoid that cost by
writing a Python patch script and running it with `bash`:

- Across the ten Runs recorded between 2026-10-19 and 2026-10-24 (824 agent-session entries):
  `bash` 281 calls, `read` 21, `write` 13, **`edit` 0**.
- Three `write` calls created patch/build scripts (`scripts/patch1.py`, `patch2.py`, `patch3.py`,
  `tmp/build_volvo_part1.py`, …); a Volvo case wrote 56.9KB of them.
- Those scripts encoded their targets as **regular expressions over layout**
  (`HISTORY = me\.build_history\( \[.*?\n\)\n`, `CONSENSUS_GAP = \{.*?\n\}\n`,
  `EXIT_MULTIPLE = L\.add\( "exit_multiple".*?\n\)\n`) and three of them failed the count guard
  with `anchor not found` / `anchor not found or ambiguous`; the agent then grepped the real lines
  and retried. The Run spent 13 model calls in that patch phase, each at 170–190K input tokens, on
  a Run whose total was 22.8M input tokens and 127K output tokens.
- A replacement that misses its anchor writes nothing: the failure is safe but silent to the model
  unless it reads the script's own guard — exactly what a file tool should not outsource to the
  model's shell.

The failure class is **layout-as-identity**. A regex can only recognise a block by the way it was
formatted at the moment the pattern was written; the stable identity of the thing being edited is
either a **name** (the assignment/function/class the starter declares) or a pair of **boundary
lines** (the section markers the starter already carries). The tool had no way to say either.

## Decision

**`edit` remains one tool with one span pipeline, and every edit names exactly one target of three
kinds.**

1. **The three targets.** Each operation carries exactly one of:
   - `old_text`: the current exact-text target, unchanged;
   - `symbol`: a named definition in the file's language. In v1, a Python file's top-level
     assignment (including a statement that unpacks several names), `def`, `async def` or `class`,
     or a dotted path to a class member (`Case.validate`, `Case.scale`), resolved with the standard
     library `ast`. The whole statement is replaced, decorator lines included, and a trailing
     comment on the line is allowed; a definition that shares a line with other code is refused
     rather than half-replaced. A parenthesized multi-line decorator is located by scanning back to
     its `@`; when that line cannot be established (a comment inside the continuation, for example)
     the edit is refused, never guessed;
   - `start_marker` + `end_marker`: two literal whole-line markers. The lines strictly between
     them are replaced and the marker lines are kept, in any file type. This is the native shape
     of the starter's `# ===== SECTION n …` boundaries and of the comment-delimited regions other
     languages use.
2. **One pipeline.** Every target resolves to a character span against the original text before
   anything is written. The whole batch must validate — one target per edit, unique target, unique
   and ordered markers, non-overlapping spans, insertions not duplicated at one offset, no no-op
   replacement — and then applies atomically or not at all. Encoding signature (including a UTF-8 or UTF-16 byte-order mark) and newline
   style are restored by the workspace text layer (`decode_workspace_text` /
   `encode_workspace_text`), not by this tool; that layer now carries the BOM it stripped.
3. **A failed call teaches the next one.** Refusals carry line numbers and candidates (which lines
   matched, the nearest symbol names, the marker that was not a whole line, the indentation a
   nested replacement needs). Results carry one summary line per edit (kind, target, line range,
   resulting line count) and a bounded hunk diff, so a big structural replacement does not emit
   hundreds of KB of diff into context. The hunks are ordinary unified diffs — line ranges widened
   to every line a span touches, and a missing final newline marked — so the unit suite can replay
   them and `git apply` accepts them.
4. **No new dependency in v1, and no pattern language.** `ast` is the standard library; markers are
   literal text. There is no regex target, no line-number target, no fuzzy matching: those encode
   layout or drift with the file, which is the failure this decision removes.
5. **One tool, not a second one.** A separate `rewrite`/`patch` tool would duplicate the same
   invariants (uniqueness, atomicity, overlap, encoding, scheduler, quota, workspace integrity,
   effects, spill, preview, plan pinning), add permanent prompt tokens for a second schema, and
   force the model to classify between overlapping tools. The resolver is an internal module with
   its own tests; a second public tool would only be justified if a different adapter — a different
   caller contract — actually appears.
6. **Contract v4, deployed at a run boundary.** The input schema changes, so
   `input_schema_digest` changes. A running Plan pins the digest of the tools it accepted; calls
   against a changed contract are refused `contract_changed` (fail-closed). There is no migration
   inside a Run, and none is needed: the change is deployed between Runs.
7. **Multi-language symbols are planned and must not change this contract.** `symbol` is worded
   language-agnostically and its supported languages grow behind an internal dispatch, so adding a
   language changes supported files, never the schema. The default backend is **tree-sitter**
   (core + per-language abi3 wheels + our own definition query per language, behind an optional
   extra) because it is self-contained, offline, and the definition query is data we own; the
   alternative is **ast-grep-py** if the M5 spike shows it clearly wins on per-language effort or
   on agent-authored structural patterns. A grammar pack that downloads parsers at first use is
   excluded: a Run's tools must not reach the network to edit a file. Python keeps the standard
   library parser.

## Milestones

This record fixes the plan the owner approved; each milestone is a separate change set.

- **M0** — extract the exact-resolution logic into a pure resolver with property tests, no
  behaviour change.
- **M1** — contract v4: the three targets above, Python symbols (module-level and dotted members)
  and whole-line markers, line-numbered errors, per-edit result summary.
- **M2** — replace the whole-file `difflib` comparison with a span-hunk diff computed from the
  resolved spans, add the result budget/spill rule, add performance tests.
- **M3** — the Skill and tool descriptions name the new procedure (fill `FILL` with markers; no
  regex or heredoc patch scripts).
- **M4** — an edit eval harness (A/B: before, after, separate-tool variant, prompt-only variant),
  acceptance thresholds, and the SQL observability queries the rollout is read from.
- **M5** — multi-language symbols: a time-boxed spike decides tree-sitter vs `ast-grep-py`; the
  backend lands behind an optional extra; languages are added by demand (Python is not in the
  queue; it uses `ast`). M5 starts only after M4's gates pass.

## Acceptance gates (M4)

The model-free run (`scripts/run_edit_eval.py --check`; `make edit-eval`) exercises the contract on
the files it exists for and exits non-zero when a threshold fails. Its 2026-10-25 result, from the
working tree this decision lands with:

| Task | Targets | Calls | Recorded workflow before | Bytes echoed by exact-only | Replacement bytes | Result |
|---|---|---|---|---|---|---|
| fill the starter's `FILL` sections | 9 | 1 batch | 6 (three failed patch scripts, three retries) | 97,651 | 1,528 | pass; filled file parses; every section header survives |
| replace `HISTORY`, `EXIT_MULTIPLE`, `CONSENSUS_GAP` | 3 | 1 batch | 1 (the old Tool could batch, echoing each block) | 5,763 | 113 | pass; file parses |
| the three recorded regex anchors | 3 | 1 batch | 6 | 5,763 | 57 | pass; each recorded pattern still matches nothing in the starter |
| a markdown section between two headings | 1 | 1 | 1 | 8,971 | 10 | pass; both headings survive |
| refusals stay actionable | 5 | 5 | 5 | — | — | pass (lines/candidates/fallback in every message) |

The exact-only Tool could also batch, but every operation had to echo the old bytes; the starter
and recorded-anchor rows therefore report the recorded script workflow (six calls) as the call
count this contract removes (documented from the trace; the harness prints it and cannot re-run
the trace), and the echoed bytes as the argument cost it removes. The FILL bodies
the harness writes are stubs, so the ratio gate is a shape check; the substantive number is the
97,651 bytes the exact-only tool would have had to echo. The gates the rollout
holds to: every target resolves in one batch with no refusal; an edited `.py` still parses; marker
lines survive; the replacement bytes stay under half the bytes an exact-only `old_text` had to
echo; a refusal names its lines or its fallback target; and every rendered hunk applies to the
original text. Rendering has a fast path from the resolved spans that is replayed against the
original before it is returned; when a span shape cannot be replayed faithfully (edits that merge
lines across each other), the standard whole-file comparison renders the diff instead, so the
result is always a patch or an explicit omission. The unit suite parses the headers against the
body counts, replays hand-written line-boundary/no-final-newline/form-feed cases and a
deterministic sample, and runs `git apply --check` plus `git apply` over representative edits.

**Still owed before rollout:** the agent-level A/B (current tool vs v4, with the Skill text as the
fourth variant) over the ten recorded Run tasks; the model-free run decides the contract and the
argument cost, not a model's tool-choice behaviour. Its harness is a follow-up change because it
needs provider calls and a Run budget, and no threshold here may be read as its result.

## Rollout observability

`scripts/run_edit_eval.py --sql` prints the queries the rollout is read from; the two facts that
tell whether the decision worked are already stored in `dlightrag_agent_session_entries`:

- `edit` calls and their outcomes, and the target kind each call used — the result summary carries
  `symbol "…"`, `between "…"` or `old_text "…"` in the stored entry text, so the query splits them
  (the pattern is anchored on the summary line, not the result's first line);
  the ten Runs before this decision recorded **zero** `edit` calls against 281 `bash` calls;
- calls whose text matches `%anchor not found%` (one session, three calls before the decision) and
  `write` receipts for `patch*/build*/edit*.py` scripts (three before the decision).

Both classes must reach zero in the Runs that fill such a starter while `edit` calls become non-zero. The
queries are global (scoping them to one Skill's Runs needs the Run-to-Skill join the session
entries already support); scope them when reading the gate. The
target-kind split is how the rollout reads whether the new targets are actually chosen, and a
sustained `symbol`/marker share below half of the region edits is the signal to revisit the tool
description and the Skill text before considering a second tool.

## Considered options

| Option | Why not |
|---|---|
| A second `rewrite`/`patch` tool | Same invariants and outcome as `edit`; duplicates the pipeline, adds a permanent schema to the prompt, and splits one decision across two overlapping tools. Reconsider only if a genuinely different caller contract appears. |
| A `regex` target on `edit` | Makes layout the identity and legalises exactly the anchor-drift class that failed three times in the observed Run; unreadable refusals; unbounded match spans. |
| Line-number targets | Drift with every earlier edit and every other writer; a stale line number writes the wrong region silently. |
| `apply_patch`/unified-diff input | Parsing and fuzzy application add ambiguity; the model pays context for hunk headers, and a diff is a worse address than a symbol or two marker lines. |
| LibCST (format-preserving CST) | Solves the different problem of preserving the old layout while editing inside a node; here the replacement text is written new, so the old layout is discarded on purpose. Adds a dependency and a slower parse for no gain. |
| tree-sitter / ast-grep-py in v1 | Both are right for M5's languages, but v1's only language is Python, and `ast` is stdlib, faster to adopt, and already exact for the need. A grammar dependency in v1 would make the first delivery depend on a packaging decision with no v1 consumer. |
| A grammar pack that downloads parsers on first use | Editing a file must work offline and deterministically; a first-use download is a network dependency inside a Tool. |
| Bash helper script in the image | No declaration, no effects, no replay/pinning facts, no read-only/ordering semantics; it moves the same workaround into the image instead of removing it. |
| Prompt/skill guidance only ("grep first, use exact edit") | Leaves the cost (reproduce a 34KB block) and the incentive (write a script) in place; the observed Run shows the guidance already existed and was not followable. |

## Consequences

- The documented procedure becomes executable: a `FILL` section is replaced by naming its two
  marker lines, and a named block by naming its symbol, without echoing the old text. The tool
  call count and the token cost of the observed patch phase fall; patch scripts no longer belong in
  a Run's `scripts/` directory.
- New refusals are introduced (ambiguous top-level name, a marker that is not a whole line,
  a nested replacement whose indentation cannot be established, a span overlap that mixes target
  kinds); every one names its line numbers and the fallback target, and `old_text` plus markers
  remain available when a structural target cannot resolve.
- The schema digest changes once; in-flight Runs that call `edit` would see `contract_changed` and
  must be deployed around, at a Run boundary. Prompt-cache prefixes change once.
- Multi-language support is additive: no schema, contract or digest change when a language lands,
  so the M5 work is scheduled by demand rather than by contract risk.
- Tests delivered with the change: resolver tests (no file change on any refusal; CRLF and a
  byte-exact UTF-8/UTF-16 BOM round trip; decorators; dotted members; unpacking including a starred
  target; duplicate names; markers not whole lines; overlap; no-op; duplicate insertions at one
  offset; trailing comments), hunk replay with header-count checks, a difflib fallback path and a
  `git apply` check, result-budget and performance tests, the M4 eval harness, and a guard over
  the Skill/Tool text that names the procedure.
