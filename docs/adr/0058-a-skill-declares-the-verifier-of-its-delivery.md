# ADR 0058 — a Skill declares the verifier of its delivery, and the platform attests the artifacts it produced

## Status

Accepted and implemented by reuse. The platform's Skill Artifact Attestation is the delivery record;
the separate delivery contract, claim and finalization gate was rejected as redundant. Re-running the
declared verifier against the delivered bytes is deferred for ordinary artifacts and would need an
amendment here; for the one artifact a Skill declares as its sealed artifact, the amendment of
2026-10-25 below closes it at the attach path.

Amended 2026-10-19: the wire still stamps the record on every Artifact, but the Answer draws it once
for the files it covers ([ADR 0057](0057-a-skill-names-what-a-run-must-read.md#amendment-2026-10-19-the-reader-gets-the-runs-record-once-and-reads-it-by-asking)),
so a reader sees one Run-level record rather than a badge repeated per file.

Amended 2026-10-19 (in-loop preview): the record is also returned to the Agent at attach time, as
evidence it can correct against while the Run is still going. It withholds nothing.


## Context

A run of the built-in analysis Skill is supposed to hand the reader an HTML page and a sealed
`model-output.json` whose own validator ends at `blocking 0`, and a run that hands over prose instead
has not delivered. The platform already had the pieces that make that claim inspectable:

- `SKILL.md` frontmatter declares `attestation: scripts/validate_model.py`, which
  `SkillCatalog.discover` parses and validates into `SkillMetadata.attestation`.
- `application/answer/provenance.py` derives a per-Run `ArtifactAttestation` from the Run's own
  durable events — which Skill it loaded, whether the declared verifier ran, and which declared
  `required_reads` it skipped — and stamps every artifact in the REST and Web projections.
- `answer.generation.artifact_attestation: mark | require` decides what a reader is shown: `mark`
  labels the record, `require` withholds an artifact the Run cannot attest.

We did not build on that. Three successive activation schemes were attempted instead: a
caller-declared `delivery_contracts` field, then activation from the run's own `load_skill` trace
fact, then a ledger that claimed a contract for a separate finalization gate. Each re-derived a
verdict the attestation already carries, each added a second judge of delivered bytes, and each added
a request/transport/claim surface the platform did not need. They are removed, not extended.

## Decision

**1. The Skill declares its own verifier.** `SKILL.md` frontmatter takes
`attestation: <skill-relative script path>`; discovery validates the path and refuses one that leaves
the Skill or does not exist. The Skill is the single source of truth for what should verify its
delivery.

**2. The platform derives the record from the Run's own records.** `provenance.py` reports, beside
each artifact, a status (`attested`, `unattested`, or `unknown`), the Skill, the declared verifier, a
reason when not attested, and the `required_reads_missing` entries. `attested` means a command
recorded in this Run named the Skill's verifier; it is evidence the verifier ran, not proof it ran
over this file. `unknown` is used where the records cannot settle the question — pruned events, a
truncated command line, a Skill gone from the catalog — and an empty `required_reads_missing` means
the records settle it.

**3. A deployment may withhold, never fail.** Under `artifact_attestation: require`, an artifact whose
records make an unattested verifier claim is withheld from the reader: its status becomes unavailable
and its links are removed, with the reason named. The Run still succeeds and the file stays where the
Run put it. `mark` labels the same record without withholding. Which claims `require` enforces is
fixed by the amendment below: a Run with no Skill and a Skill that declares no verifier made no claim,
and an `unknown` record settles nothing.

**4. No separate delivery contract, claim, gate or report is added.** There is no `delivery.json`, no
claim field in a delivered document, no finalization gate, no stored `delivery` report, no
`delivery_contracts` request field, and no transport surface for any of them. Loading or browsing a
Skill has no gating effect; the record comes from the Run's own tool calls.

**5. Re-verification is deferred, and is a real gap.** Attestation is evidence, not proof: a
hand-forged document, or one edited after the validator saw it, is still attested because a command
in the Run named the verifier. The deferred re-verification option is to run the declared verifier
against the *delivered bytes* and record its outcome: it would close that gap, but it needs a durable
record of those bytes and a policy for what a failure does to an already-succeeded Run. This ADR
records the gap rather than promising the check.

## Consequences

- A reader can see on every artifact whether the Skill's verifier was recorded and which declared
  reads the Run skipped; the record says evidence, not proof.
- Under `require`, unattested artifacts are withheld and the Run still succeeds; a Skill that declares
  no verifier publishes no record beside its artifacts.
- One source of truth (the Skill's frontmatter) and one derivation (the Run's durable records) replace
  three activation schemes, a claim format, a gate process, a results field and a transport surface.
- The removed system's behaviour-evaluation rows are left as history and footnoted; the behaviour
  suite now scores the platform's existing `attestation` records instead.
- The evidence-not-proof gap remains visible in this record until an amendment adds re-verification.

## Amendment 2026-10-19: `require` withholds a claim, not every file

Turning `require` on in the product exposed the difference between "this deployment hands over only
attested deliverables" and "this deployment hands over only files a verifier-declaring Skill
produced". The original reading withheld every Artifact that was not attested, including the
Artifacts of a Run that loaded no Skill and of a Skill that declares no verifier — which in this
product is most Runs (only one built-in Skill declares a verifier today). A policy that bans
the ordinary case cannot be turned on, so no deployment could use the gate.

`require` is now claim-scoped: it withholds an Artifact exactly when the derived verdict names a
verifier the delivery did not attest. The claim is present when the verdict names a verifier, which
is what the derivation sets when the loaded Skill declares one today: `skill_loaded_without_verifier`
(a loaded Skill declares a verifier and no recorded command ran it), `verifier_reported_failures` (it
ran and reported failures), and `arguments_truncated` (the declaration was read and a recorded call
was cut at the runtime's bound, so whether the verifier ran cannot be ruled out). The first two are
positive findings; the third is a claim the deployment cannot confirm, and a deployment that hands
over only verified deliverables cannot hand that one over either. Everything else is served with the
record it carries: `no_skill_loaded`, `skill_has_no_verifier_contract`, `contract_unavailable`,
`records_pruned` and `skill_name_unavailable` name no verifier, so no claim a deployment could
enforce was made. The predicate lives once, in `ArtifactAttestation.withholds_under_require`, and
both gates (the Artifact projection and the byte-serving read) read it, so a label and a refusal
cannot drift apart.

The refused set is not allowed to grow by inference: it grows only when the derivation learns a new
reason that names a verifier, and every reason it names stays in `ATTESTATION_REASONS`. A record
that could not name the Skill (`records_pruned`, `skill_name_unavailable`) is still served — the
deployment cannot hold a file to a claim the records never recorded.

The consequence for the product is the intended one: a Run of that Skill that skipped its
declared verifier now has its page and model withheld with the reason named, while a chart, a page
from `interactive-html`, or an office document from a Skill that declares no verifier is delivered
as before. The label rules are unchanged.

## Amendment 2026-10-19 (later): the full recorded command is read

The `arguments_truncated` reading above rested on a gap the derivation did not need to leave open.
A Tool publishes its subject as one bounded line (`TOOL_SUBJECT_MAX_CHARS`), while the arguments it
was executed with are durable in the Session transcript's Assistant entries. Reading only the event
log made a verifier named after that bound indistinguishable from one that never ran, so `require`
withheld deliveries whose verifier had run. The derivation now reads both records and joins them by
`call_id`: the event log says which calls ran and how each ended, the transcript says what each of
those calls was. A transcript entry with no matching event, or one whose Tool name disagrees for the
call id, is not evidence and is never read, so a model cannot attest a file by writing a command it
never got executed. `arguments_truncated` therefore now means what it says — part of the Run's
record is missing (a transcript the store cannot return, an entry trimmed by retention) — and the
reader-facing refusal says so in place of the old display-bound sentence. The withheld set does not
grow: a Run that never ran its declared verifier is still withheld, now with
the accurate reason `skill_loaded_without_verifier`.

## Amendment 2026-10-19 (final): the `require` mode is withdrawn

`artifact_attestation: require` is removed. The record stays and is published always; the deployment
setting that could withhold an Artifact for it is gone. Three reasons decided it:

- A deployment setting must not reach back into Answers already served. The derivation runs when a
  reader asks, so turning `require` on hid files that earlier Runs had already handed over. Nothing
  the model or the Run did was wrong; a config change was.
- The failure was silent and on the wrong side. A Run finished, its files existed, and the reader --
  not the Agent -- met the refusal. The Agent had already done its work and could no longer correct
  anything; a gate placed after delivery is not a gate, it is a disappearance.
- The gate's job is done in-loop. A Skill that declares a verifier states its own delivery contract;
  its validator refuses the document while the Agent can still fix it, and the Agent sees the error.
  The platform's part is the record: what the Run did and did not do, told beside the files, always.

`mark` was the shipped default and remains the only behaviour: every Artifact is handed over with the
record its Run's durable records settle, and `withholds_under_require`, the refusal message and the
byte-serving gates are deleted rather than left dormant. Re-verification remains outside this
decision; if a deployment later needs a delivery contract, it belongs at publication time and to the
Run, not to a reader's read of an Answer that already exists.

## Amendment 2026-10-19 (in-loop preview): the record is also shown to the Agent

The final amendment above left the platform outside the loop entirely: the record was derived when
a reader asked, and the Agent learned nothing while it could still act. A Run that skipped every
declared read and never ran its verifier delivered its files and met the record only after the Run
had ended — the same asymmetry that withdrew `require`, on the other side.

**The attempt's evidence is returned to the Agent at attach time, and it withholds nothing.** A
successful `attach_artifact` result also carries what the current attempt recorded: the declared
reads it has not covered, whether a command named a Skill's declared verifier, and, for an HTML
page, whether the document carries the toolkit's `report-source` block (ADR 0038). It states
evidence, not the reader-facing status: that verdict and its reason stay the durable derivation's,
judged from the Run's records when it settles, because a preview that re-derived them would be a
second implementation of the verdict and the two would drift. It is worded as a preview —
"recorded so far in this attempt" — and it changes no Artifact's status, bytes or availability. It
adds no gate, no withholding and no new failure: the Artifact is attached exactly as before, and
the durable derivation remains the reader's authority.

The preview is fed by an in-memory ledger of the current attempt, written where the Run's calls are
already observed: `load_skill`'s read fact and the execution seam that names each Tool call and its
arguments. It reuses the one read rule in `src/dlightrag/engine/agent/skill_reads.py`, and the
verifier-mention rule and execution-Tool set moved into the engine so the preview and the durable
derivation cannot disagree about which calls ran. A Run that was reclaimed and continued can lose
ledger entries, which is why the preview says "so far"; a fresh derivation mid-Run is deferred
because `read_run_tool_calls` follows the Run's subscription and returns only at its terminal event,
so it cannot answer before the Run ends without a new bounded snapshot reader.

The reader's own words are reused where they fit — `not read: <path> (<when>)` — and the tone is
neutral: this is evidence, shown early, not a second verdict.

