# ADR 0048 — a Skill document's size follows the industry's rules, not a round number

Status: accepted. Author: the Skills work.

## Context

`skills.py` carried `_MAX_SKILL_FILE_CHARS = 50_000`, applied uniformly by `SkillCatalog.read`
to every Skill document — hub, reference and script alike — and by the owner-publish
validation. The constant had no comment, no test naming it, and no ADR. It was a round
number.

The industry sets three different rules and none of them is a uniform character cap:

- **The hub has the one published limit.** Anthropic's skill-authoring guidance caps the
  SKILL.md body at **500 lines** "for optimal performance", with detail moved into
  reference files, and elsewhere at 5,000 words (ideally 1,500–2,000). The 1,024-character
  description limit we already enforce is the same guidance's number.
- **References have no size limit.** They are loaded on demand. The guidance asks for a
  table of contents above 100 lines and, for very large files (above 10,000 words), that
  the hub carry grep patterns instead of asking for a whole-file read.
- **Scripts are executed, not loaded.** Only their output enters context, so the guidance
  sets no size limit on them at all.

A uniform 50,000-character cap was simultaneously **looser** than the industry rule where
one exists (50,000 characters is about 8,300 words, against 5,000 for a hub) and
**stricter** where none exists (references and scripts).

## Decision

**The per-kind policy.**

1. **The hub is capped at 500 lines**, the industry's published number, and this Skill's
   hub is far below it (67 lines at the time of writing). The rule is enforced where this
   product sets its own rules — at publication — and asserted by the builtin-Skills test
   suite, not at read time: a longer hub is worse, not unreadable, and a hard failure in
   production would punish a user for a performance guideline. `_MAX_DESCRIPTION_CHARS`
   stays 1,024 for the same reason.
2. **References are uncapped.** They load on demand, and the existing hub pattern — a
   Files table with one line per document — is the progressive disclosure the industry
   guidance asks for. The table-of-contents convention the reference files already follow
   is stated in this ADR as the rule for anything above 100 lines.
3. **Scripts are uncapped.** They are executed; their output, not their text, is the
   context cost. The engine can grow again.
4. **No runtime size guard on documents — tests instead.** A guard would be a fourth
   rule with no industry source, and it would punish a legitimate document for a
   pathology the surrounding system already covers: the built-in Skills are pinned by
   the test suite (hub line rule, every dispatched path exists, references resolve),
   owner payloads are bounded by the existing total-byte quota, and a document's cost
   to context is the runtime's own budgeting, not this layer's. What is tested is the
   product's own content; what is not tested is nobody's guarantee to give.

## Consequences

- `SkillCatalog.read` raises no size error at all; the 500-line hub rule is enforced at
  publication and asserted by the builtin test suite, because a longer hub is worse, not
  unreadable.
- The builtin-Skills hub test asserts the 500-line rule so a builtin cannot drift past it
  unnoticed; the hub's token count is still printed for observation.

## Amendment 2026-10-19: the hub budget is tokens, and a Skill's declared reading has one too

The 500-line rule measured the unit that is easy to count, not the unit a reader pays.

Decision:

1. **A hub is at most 8,000 estimated tokens** (`estimate_tokens` in
   `src/dlightrag/engine/ai/tokens.py`), in addition to the 500-line rule, which stays as the
   formatting rule. The budget is a ceiling with room under it, not a target: the amendment's own
   content sits at about 4,000 tokens, and a future sentence is not a publication failure.
2. **A Skill's declared required reading totals at most 20,000 tokens** across the entries ADR 0057
   defines, counting the largest branch of an `any_of` — about a third of the per-turn result
   reserve. A declaration above it is not a reading a Run can complete, and an uncompletable
   declaration stops being a check.
3. **A numbered step that runs to a section is a style problem, not a size problem.** The builtin
   tests flag a step over 3,000 characters, and publication does not refuse one: a hub's context
   cost is the budget above, and how it divides its prose is its author's.

The rules are enforced where the 500-line rule is enforced: `publish_skill` refuses an owner
payload that breaks the hub-token and declared-reading budgets, and the builtin-Skill test suite
asserts those two, the 500 lines and the 3,000-character step guideline for the product's own
Skills. Read time stays unguarded (decision 4 above, unchanged): a document's cost is the runtime's
budgeting, and a hard failure in production would punish a reader for a guideline.

Deep references stay uncapped and become lookup-only.
