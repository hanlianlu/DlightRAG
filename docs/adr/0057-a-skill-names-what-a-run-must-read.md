# ADR 0057 — a Skill names what a Run must read, and whether it did is a recorded fact

## Status

Accepted and implemented. Runs recorded before the read fact existed are read as unsettled, never as clean.

Amended 2026-10-19: the reader gets the Run's record once, under the placements it covers, with the
per-read evidence folded under one disclosure (decision 6 says what the lines say, not that each is
always on screen).

Amended 2026-10-19: the required reading is a gate card, and a copy is still not a read.

## Context

A Skill is progressive: its hub is loaded first and its references are loaded on demand, and the
hub tells the model when each is due. Whether the model then reads them was a question nobody could
answer from a record. Live runs of a built-in Skill read its `SKILL.md` and ran its
scripts and read none of its twelve references, while an imperative placed in the hub at the point
of use ("before writing, read `references/x.md`") was read. The first scorer of that behaviour had
to infer the answer from the Run's transcript, which the REST route returns as its last 100
messages, so its figure was a lower bound that said nothing about the part of the Run that fell
outside the window.

The durable record could not do better, because it did not hold the fact. A `load_skill` call
published one thing on its `tool_progress` event: the Skill's name, as the call's subject. Which
document the call had answered, and how much of it, was in the Session transcript, which no
transport-facing read exposes (the attestation derivation in `application/answer/provenance.py`
names the same limit for a command line). So a Skill could say "read this", a Run could skip it, and
nothing anywhere said which had happened.

Two properties of the existing design decide the shape of the answer:

- **The size of a `load_skill` window is the reading model's, not the document's.** One window is
  fitted to the model's own tool-result ceiling (ADR 0055), so the first window is a few thousand
  characters for one model and the whole file for another.
- **An attestation is evidence, not proof, and is never a gate.** `provenance.py` already derives,
  from the Run's own durable events, whether a Skill's verifier ran. Its `require` policy withholds
  an Artifact on that verdict. A second kind of finding must not change what that policy withholds.

## Decision

**1. A Skill declares what a Run must read.** `SKILL.md` frontmatter takes an optional
`required_reads` list:

```yaml
required_reads:
  - path: references/fragment.md        # one Skill-relative file that must be read, OR
    when: before writing the fragment   # the step or condition, in the author's words
  - any_of:                             # several files; reading ANY ONE satisfies the entry
      - references/example-brief.html
      - references/example-dashboard.html
    when: before writing, the example nearest the request
```

Each entry is a mapping of exactly `when` and one of `path` (one file) or `any_of` (a non-empty list
of at most 8 files, no duplicates). `when` is required text, collapsed to one line and at most 120
characters. Every path is Skill-relative, passes the same check as `publishable` and `attestation`,
is normalized (`./x` is `x`), and is not `SKILL.md`: the hub is what loading the Skill hands over,
so requiring it would be requiring the load. A Skill declares at most 16 entries. A key outside
`path`, `any_of` and `when` is refused, because a typo (`paths:`, `whn:`) would otherwise be a
declaration that silently declares nothing, and a Skill that believes it requires a reading and does
not is the failure this field exists to make visible. Any violation is a `SkillFrontmatterError`
that names the entry and the rule.

Like `attestation`, `required_reads` is a claim the Skill makes about itself, so a declaration naming
a file the Skill does not carry is refused where it is made: discovery does not serve the Skill (it
stays the tier's document and stays protected, like one whose frontmatter fails), and
`publish_skill` rejects the payload.

**2. `load_skill` records what it read, on the durable event.** After a successful call, `load_skill`
reports a second progress update carrying `skill_path` (the normalized path), `skill_window_start`
and `skill_window_end` (the characters it returned), and `skill_document_chars` (the document's
length). The Runtime forwards exactly the keys in `TOOL_PROGRESS_FACT_KEYS` from a result's
`details` onto the `tool_progress` event, and only scalar values; anything else in `details` is never
forwarded, so a Tool cannot widen what the event log holds. A `write_to` copy reports `skill_path`
alone: a copy hands bytes to the workspace and is not a read by the model, so it never satisfies a
required read, but its presence tells a new record from one written before this fact existed. A
failed call reports no fact. A path longer than 512 characters reports nothing.

**3. A document is read iff its windows cover it, first character to last.** The windows recorded
for one version of a document (one `document_chars` value) must cover `[0, document_chars)` with no
gap. The rule is the whole document, and not "the first window", because window size is the reading
model's: a first-window rule would credit a reader that stopped after the opening with a different
amount of text depending on which model it was, and could not say what "read" means across the
models the product runs. Whole-document coverage is the one rule that does not depend on the reader.
Section-level rules were considered and rejected: a section has no stable address in the windowing
(offsets are characters, not headings), and the references a Skill requires are meant to be short
enough to read whole. A window the Runtime later cut to the model's residual
earns no credit: the settlement records the cut as the metadata bool `observed_truncated`, and a
call so settled keeps its path and loses its window. The rule and the coverage arithmetic live in `engine/agent/skill_reads.py`,
which imports only the standard library, so the attestation and the behaviour eval load the same
file and cannot disagree.

**4. The derivation reports unread declarations as evidence beside the verdict.**
`derive_run_attestation` computes `required_reads_missing` for the Skills the Run loaded. It is
**`None`** when the records cannot settle it: the log is incomplete or pruned, a load cannot be
named, a loaded Skill is no longer in the catalog, any successful `load_skill` call carries no read
fact (the record predates it), or no loaded Skill declares a read. Otherwise it is a tuple of the
entries not satisfied, **possibly empty**, and `[]` says every declared read was read to the end.
`None` and `[]` are never the same answer.

It is evidence, not a gate. It changes neither the verdict's status nor its reason, and `require`
withholds exactly what it withheld before. A Run that skipped a reading its Skill asked for is not
thereby unattested, and an Artifact built without it is not thereby withheld. The record carries it
as `required_reads_missing: null | [{skill, paths, when}]`.

**5. A Skill with no verifier still publishes a record when reads are missing.** A verdict that says
only "the loaded Skill declares no verifier" publishes no record beside an Artifact, because that is
a fact about the Skill and not the file. A Skill such as `interactive-html` declares no verifier and
now declares a required read, so the stamping rule is: publish the record when the verdict makes a
verifier claim, or when `required_reads_missing` is non-empty. That is what makes a skipped reading
visible on the HTML report the Run built without it.

**6. The reader sees one neutral line per missing read.** The Artifact's attestation record draws
`not read: <path> (<when>)` for each entry, `<path>` being the entry's paths joined with " or ". It is
drawn in the same neutral tone as `unknown`, with no alert role. When the record exists only because
a read is missing (its reason is `skill_has_no_verifier_contract`), the lines are the whole record:
the Skill made no verifier claim, so the reader is not told "Not attested".

## Consequences

- A Skill's own statement of what must be read is a checkable claim, and the behaviour eval scores it
  per run (`required_reads`, `required_reads_missing`, and `skill_reads` per file) from the event
  stream, replacing the transcript heuristic and its lower-bound caveat.
- Runs recorded before the read fact are unsettled: they read as `None`, so an old Run is never
  accused of a skipped reading and never credited with a clean one. They cannot be re-scored into
  either.
- A `write_to` copy never counts. A Skill whose required document is a script copied into the
  workspace and run has not "read" it in this sense, and should not declare it; the rule is about
  text put in front of the model.
- A spawned child's tool events are published through the parent Run's sink under the parent's run
  id, with the child's own call ids, and carry nothing that tells them apart. So a document a child
  read counts as read by the Run: a Run that delegated the reading is credited with it, even when the
  model that wrote the Artifact never saw it. This is reported as a limit, and no plumbing separates
  the two.
- A recorded window is evidence that the text was put in front of the model, not proof that the
  model attended to it. Like every record in the attestation, it removes the silent case, a Skill
  that says "read this" and a Run that never opened it, and nothing more.
- The window recorded is the one `load_skill` returned, fitted to the reading model's own ceiling,
  and the Runtime may still cut a result to the model's residual. The fact cannot see that, so the
  Runtime's settlement carries it (`observed_truncated` on the durable `tool_end`), and the read rule
  gives a cut call no window: it stays a post-fact record, never a read.
- `interactive-html` is the first Skill to declare a read: any one of its four example fragments,
  before the fragment is written. Another built-in Skill's twelve references are not declared by
  this change; declaring them is that Skill's own decision.

## Amendment 2026-10-19: the reader gets the Run's record once, and reads it by asking

`stamp_projected_artifacts` already puts one record on every Artifact of a Run -- the same verdict,
the same reason, the same missing reads -- so drawing it inside each Artifact said one Run-level
sentence as many times as the Answer listed files, and each skipped read added a line to every card.
Decision 6 still says what the evidence says; the shape it is shown in changes:

- The Answer draws the record once, under the placements it covers, as one line and one
  disclosure: the verdict, then the folded evidence. The reason and the per-read
  `not read: <path> (<when>)` lines are the evidence, so they fold under the control -- labelled
  with the count ("N reads skipped") when reads are missing and "Details" when only the reason or
  the verifier is -- because the reason is evidence beside the verdict and not a second verdict,
  and the line under a reader's files has to stay a line. Opening the disclosure draws the reason
  and the lines exactly as the server published them.
- The per-Artifact badge remains where a surface draws one Artifact by itself (the Artifact Canvas),
  and it draws the status alone: the Canvas is the file a reader opened, so the reason and the
  per-read lines wait on the Answer's record rather than crowding the surface that shows the file.
  An Answer's cards name what they open and carry no copy of the record.
- A withheld Artifact is outside the record's line: its own issue already names why it was withheld,
  so a Run whose every file was withheld has nothing on screen the record could add.

The copy is unchanged: the status vocabulary, the reader-facing reasons and the per-read lines are
the ones the server publishes, and the record keeps no alert role and no failure tone.

## Amendment 2026-10-19: a gate card is what a Run must read, and a copy is still not a read

Required reads were added so a Skill's "read this" was checkable. Their first live use showed the
declarations were not readable: one built-in Skill declared four references totalling 298,599
characters (~74,765 tokens), and the 2026-10-19 Run `01a15404…` read none of them — nor
`interactive-html`'s example, at 642–2,767 tokens. A declaration no Run can finish stops being a
check and becomes noise, and decision 3's premise ("the references a Skill requires are meant to be
short enough to read whole") was not met by the content.

Decision:

1. **A required read is a gate card.** A short document — 3–6 KB, roughly 750–1,500 tokens — states
   the gates for one step, the passing shape for each, and where the deep reference's sections are.
   The deep reference is lookup-only and is never declared. A Skill declares at most four cards.
2. **A card names the rule ids it gates, and is held to them.** A test asserts every id a card
   names exists in the validator and `rule_ids.md`, and the card's size is bounded, so a card
   cannot drift into a second reference.
3. **`interactive-html` replaces its example `any_of` with `references/fragment.md`** (~1,613
   tokens). The example becomes a starter (ADR 0038 amendment), which is a copy, not a read.
4. **The read rule is unchanged.** Only a whole-document load satisfies a declared read; a
   `write_to` copy never does. The starter therefore cannot mask an unread declaration — and a
   Skill should not declare what the starter delivers.

Consequences: the record becomes achievable (about 5,000 tokens of declared reading instead of
about 75,000), and a skipped read becomes a decision the Agent could make rather than a tax it could
not pay. Section-scoped reads remain rejected (decision 3): the card is the section.
