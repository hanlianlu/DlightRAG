# Profile Memory is maintained per Session and owned across them

One owner's Profile Memory gains a second writer. A Session Memory Pass runs after a
Run of a Session succeeds, reads that Session's own user and assistant turns, and adds
at most five records a pass — adding, updating, fusing or dropping against what the
owner already has rather than appending. The records stay
owner-scoped rows in the memory package, so recall across conversations is unchanged;
what is new is that a Session may contribute memories without a tool call and without
its owner having asked. The Agent — the model answering in a Run — is not that writer,
and its memory tools keep their behaviour; the pass calls a model of its own, with its
own prompt and its own role.

## Status

Accepted and implemented. A development database is reset rather than migrated
([ADR 0022](0022-session-owned-memory-and-run-owned-products.md)), though the
memory package also widens its closed checks and adds its column in place on an
older database, since it owns its own schema. The
package's record table gains one column and one value in each of its two check
constraints (`kind`, `origin_kind`), its operation table gains one action, and
DlightRAG gains one progress table. The local record table is empty, so the reset
loses nothing.

Amended: the five-record bound is per pass, not per Session. It counts what one
pass adds; updating or fusing records the owner already holds — whichever
Session wrote them — is maintenance rather than contribution and never counts,
so a pass may reach into an older Session's records even at the limit. The
earlier wording below ("counts the active records ... whose session_id is that
Session", "moves that contribution to the Session doing the merging") described
a standing-attribution bound that is superseded, and the live documents state
the per-pass bound instead.

Amended: the pass calls the `query` role at the reasoning level
`memory_consolidation.reasoning` sets (low by default), not `extract`. The
2026-10-17 eval ([Memory pass eval](../memory-pass-eval.md)) found that a pass that
does not reason recalls none of the owner's real preferences, where the same model
thinking recalls most of them. The level is the pass's own, a key beside
`session_record_limit`, so it does not follow the `query` role's `reasoning` (off)
or `agentic_reasoning` (the Agent's tool loop); the model profile clamps a level it
does not support. The prompt changed with it: it directs
the model at the owner's own reactions to how work is done, keeps a third party's
wish theirs, and leaves the Agent's own workspace facts (what is installed in it,
what survives between runs) out of experience. The Decision text below keeps its
original wording and is read through this note.

Amended: the records a pass writes are attributed to the Run that triggered them, and the
conversation shows them. The pass records the Run whose success scheduled it in
`MemoryProvenance.run_id` (the latest of the Runs a Session's runner coalesces into one pass), and
the stored plan keeps that Run (`plan_run_id` on the progress row, set exactly when a plan is) so a
resumed plan applies under the Run it was made for and every write keeps its
idempotency key and fingerprint. Under that Run's answer, the action row lists what the Run's
writers left still active, the Agent's tool writes (`answer_run`) and the pass's together, each with
an Undo, so the owner sees what was kept from this conversation and can take it back there as well as
in Settings. This is still no approval: nothing waits on the owner, a record is written as before,
and the owner may undo it afterwards.

Amended: the pass's model never sees a memory id. The memories shown in one request are numbered
from 1 through both lists, in the order listed, and the verdict names the memories it updates or
fuses by those numbers (`targets`, replacing `target_ids`); the host maps each number back to its
memory id when it plans the writes, and drops a write that names a number it did not show. The
mapping lives inside that one model call. The stored plan (`PlannedWrite.target_ids`) still holds
real ids, so resume, idempotency keys and fingerprints are unchanged. A mistyped 36-character id
used to be dropped silently; a small integer is hard to get wrong.

Amended: memories reach the models with when they were last written. Every line a model reads,
the injected block, the Agent's `memory_lookup` result and the pass's numbered list, carries the
record's `updated_at` (else `created_at`) to the hour in UTC, `- (preference, 2026-10-17 14:00) ...`,
and the block that lists them says once that times are UTC and are when each was last written,
and the MCP `memory_lookup` result carries it as `written_at_utc`.
`RECALL_LINE_OVERHEAD` grew from 16 to 34 to cover the longest such framing, so the same 8,000
characters hold fewer, longer lines. The pass is told that when memories conflict the more
recently written one is likelier current, and never to put a date or time in a memory's body: the
store keeps the time.

Amended: every successful Run's pass reads all the new text, to the end, and shows the model what came just
before it. The trigger is gone: `trigger_tokens` and the below-trigger outcome are removed from the
configuration and the code, so whatever user and assistant text the Session holds past the watermark is read,
however short, and a pass with none ends as `nothing_new`. What is longer than one model call is no longer
left for a later Run's pass. Under the one claim the pass reads a batch of at most `max_batch_tokens` (which
stays, and bounds one call), asks the model, stores the batch's plan, applies it and advances the watermark,
then reads the next batch from there until nothing is left; a plan found stored at claim time is resumed first
and the pass reads on from it. Each batch is two-phase as before, and its idempotency keys and mutation scope
name its own range, so batches never share one. The five-record bound, and the loop guard of ten, are per
model call, that is per batch: a pass of several batches gives each its own allowance. The claim is renewed
before every batch after the first, by an update fenced by its holder that only a live claim passes, so the
lease bounds one batch and not the pass. A pass whose renewal fails, whose plan the store refuses or whose
watermark it will not move has lost the claim and stops. The owner's capability is taken before every batch is
read, and a pass that finds it off stops, reading and moving nothing; its epoch is what guards that batch's
writes, so if the owner turns memory off and on, or clears it, while the model thinks, every write is refused
and the watermark still moves past the batch (text read under a stale epoch is never read again), and a stored
plan resumed from an earlier pass, whose epoch was not stored, is guarded by the epoch at resume time. A bound
of 25 batches per pass is a safety valve and logs when reached; the next Run's pass continues from the
watermark. Each model call is also given, as read-only context, the last five rounds before its batch,
following Hy-Memory's history context: the five most recent text-bearing user turns of the main Session and every
text-bearing assistant turn that follows each of them (an assistant turn before the earliest of the five is left
out), under the same filters as the read (no tool result, no Child notification), in its own section ahead of the
new text. User turns are shown whole; each assistant turn is cut to its first 500 characters under a trailing
ellipsis. The prompt says that part is for reading the newer one by (what "这样" or "that" points at) and
that what it said alone was already judged by an earlier pass. The candidate search is run on the last 2,000 characters of the context and the new text
together, so a reply too short to name its subject still finds the memories about what it answers. When an
owner's Profile Memory switch goes from off to on, every one of that owner's Agent Sessions starts afresh from
that moment: before the switch flips, and only from off, the Run store moves each Session's watermark and its
new nullable `context_floor` to its latest entry
sequence in one statement that never moves either back and leaves a held claim and a stored plan alone, so
nothing written while memory was off, whether or not a pass ever saw it off, is read or shown to a model as
context (the context read returns only turns above the floor, null limiting nothing), and a pass in flight
reads on from the watermark the store answers with. The reasons: Hy-Memory 2.0 and memorax-code both process
every completed turn or guarantee a final flush, and neither leaves a sub-threshold tail permanently unread,
whereas the 1500-token gate left unread for good a conversation that never reached it and the last part of one
that did (the median read per Run measured below, 996 estimated tokens, is under it); a pass now costs about
two seconds on the query role at low reasoning, so the gate no longer earns its keep; and once each pass reads
only what one Run added, an elliptical reply is unreadable without what came before it, which is why Hy-Memory
2.0 also gives its extractor the last few earlier turns as read-only context. The paragraph "The trigger is an
accumulation, and the delta is a watermark" and the `trigger_tokens` (1500) in the configuration paragraph
below describe the earlier design and are read through this note.

Amended: a lost pass is recovered, because the database is the queue. A pass is a detached task of the
process that scheduled it, so a restart (every deploy) cancels the passes pending or in flight, a failed
model call leaves its batch unread, a failed or cancelled Run schedules none, and the 25-batch valve stops
a pass short. Each of those was recovered only if the same Session later had another successful Run, and a
Session that never ran again kept its tail unread; the note above that retired the tail left these
causes standing. `SessionMemoryPassRunner` now owns a sweeper, started and stopped with it (composed in
`_compose.py`, started last in `Application.astart`, stopped in the runner's `aclose`). Thirty seconds
after the process starts and every ten minutes after that (module constants, not configuration), the
runner asks the Run store for at most twenty Sessions and hands each to the path a successful Run takes,
so the per-process bound of two, the per-Session coalescing and the cross-process claim apply unchanged.
A Session whose pass is already running in this process is skipped without asking for another one after
it, since a running pass reads to the end by itself. What "pending" means is read off the tables, with no
queue of its own: a main Agent Session (one that has a routing row) with a succeeded Answer Run, none of
whose Runs is queued or running (a live Run schedules its own pass when it succeeds), whose latest entry
is past its watermark (no progress row is a watermark of 0) with at least one own turn past it under the
pass's own filters (a user or assistant turn with text that is not a Child's notification: a tail of tool
results alone is not pending, since a pass that read nothing would not move the watermark and the Session
would be pending for ever), whose owner's Profile Memory is not off (a pass with memory off advances
nothing, so such a Session would be offered at every sweep), that no pass holds, and that is not backing
off. Its pass is attributed to the Session's most recent succeeded Answer Run, as if that Run had just
succeeded. A pass that ends `model_failed`, or by an error nothing expected, counts the failure on the
progress row (`failure_count` and `retry_after`, written under the claim's holder), and the sweep leaves
that Session alone for ten minutes after the first failure, twice as long after each, to six hours; a
pass that moves the watermark clears both. The other outcomes (a claim lost or held by another process,
memory off, nothing to read, a cancellation) are no failure of the Session's and count nothing. A pass a
successful Run schedules ignores the wait, since a new Run is new evidence and the claim does not read
it; its own outcome is counted like any other. The sweep is staged so that it stays cheap as the tables
grow: one pass over the Sessions' counters against their progress rows keeps those that are behind, and
every other condition is one index probe for each of them (the new
`idx_dlightrag_answer_run_routing_session` on `(owner_id, agent_session_id)`, which the migration
`session_memory_sweep` adds with the two columns, and the Runs' and the entries' primary keys); an
integration test reads the plan. Several processes may sweep at once and be offered the same Sessions:
the claim lets one pass run and sends the others away, so what keeps them from stampeding is the small
batch, not a lease per row. Because the sweep reads the tables, its first looks after a deployment also
take up every main Session that is behind (a Session that predates the pass, and a tail the trigger left
unread before it was removed), twenty at a time, newest conversation first. Rejected: a "dirty" flag or a
queue row written from the Session's commit or the Run's settlement, which the pass has stayed off on
purpose; and a recency horizon, which would leave an old tail unread for good.

Amended: a memory carries the owner's own words behind it. A verdict's write may give a `quote`, a verbatim
excerpt of what the owner said in the part being judged, and the record keeps it (a nullable `quote` column
the package adds in place, as it did `last_change_id`), so that recall shows the Agent the words a memory was
drawn from beside the memory, as Hy-Memory's MemHub shows the original quote with each memory. The reason is
grounding: the pass words a memory in the third person, and a paraphrase loses what the owner stressed and
how far they meant it. The words are the evidence the memory rests on, and they are the owner's, which is
why they are checked and not trusted. `plan_consolidation` keeps a quote only if, after whitespace is
squeezed to single spaces, it is a contiguous substring of one of the batch's user turns
(`ConsolidationRequest.owner_words`: the `[user]` turns of the newest part, never the context turns and never
an assistant turn), within 120 characters, and free of what a body may not carry (a citation marker, a
credential; the store enforces the same rule, `is_storable_quote`, on every operation that has one). A quote
that fails loses itself and not the write. The prompt gains one paragraph, copy the owner's words exactly,
only from the `[user]` turns of the newest part, and leave the quote out when the owner did not say it (an
experience the Agent learned alone has none by design), and the shape line shows `"quote"`; nothing else
in it changes. The quote belongs to the write that made the record's current form: an update's or a fuse's is
its own and is not inherited from its sources, and an undo restores the quote with the record it restores,
since the journal's snapshots carry it. The stored plan holds the quote the host kept, so a resumed plan
applies the same one; the operation's idempotency fingerprint includes it when present, so every earlier
fingerprint still replays. Only the pass writes quotes: the Agent's tools, REST and MCP take none, and MCP
`memory_lookup` returns the field. Recall puts it on a second line under the memory,
`  owner said: "…"` (the framing is English like the rest of the block; the words keep the owner's language),
in the injected block, in `memory_lookup` and in the pass's own numbered lists, where it helps judge
whether a statement refines a memory. It is charged to the recall budget: `recall_cost` and the first-fit
`_within` add the quote and `RECALL_QUOTE_OVERHEAD`, 17 (two spaces, `owner said:`, a space, two quotation
marks, a line break), for each record that has one, so the block still never renders more than recall
charged, and a full quote makes a record cost up to 137 characters more, so fewer records fit the same
8,000. `reserved_auto_recall_text` stays the worst case without change: a quote's framing is a larger share
of its line than a body's, so a block with quotes is less dense in tokens than the reserve's quote-free fill,
which the policy tests pin for every shape of record. An owner export and an owner erase reach the column
with the row, since both discover the schema.

## Context

Profile Memory today is written only when something asks for it. The Agent may call
`remember`, `forget` or `recall_memory`
(`src/dlightrag/engine/answer/tools/memory.py`); an owner may write through REST or
the Web Settings list (`src/dlightrag/application/memory/service.py`); an external
MCP client may call the four tools of `dlightrag-memory-mcp`. Reading is the only
half that is already partly automatic — a Run's bind injects the owner's standing
preferences and query-relevant facts into its request
(`src/dlightrag/engine/answer/execution/executor.py`, `render_auto_recall` in
`src/dlightrag/engine/answer/memory.py`).

Two consequences follow, and both are visible in the product before they are visible
in the design. A preference stated mid-conversation is remembered only if the model
chose to call a tool at that moment; nothing revisits the conversation afterwards. And
a profile accumulates: the only bound is a *per-Run* mutation cap of ten, which counts
settled operations in the operation journal
(`packages/memory/src/dlightrag_memory/_storage/pg.py`), so it refuses an eleventh
change but never says how many memories an owner should have.

The measured shape of local Sessions bounds what a mechanism may spend. Across the
five routed Agent Sessions in the local database — 23 Answer Runs, 385 tool results —
tool results are 94.4% of all entry characters (3,598,139 against 213,301 for user and
assistant turns together). The extraction pass therefore reads a small slice: 64,638
estimated tokens across those five Sessions, against 89,677,172 provider input tokens
the same Runs measured for themselves. Per Run the read volume has a median of 996
estimated tokens and a maximum of 39,566.

Three properties of the existing model are load-bearing for this decision and are easy
to misread from the outside.

`MemoryProvenance` already carries `run_id` and `session_id`
(`packages/memory/src/dlightrag_memory/models.py:17-24`), and every record row
stores both, but nothing filters, ranks or budgets by them: every read is
owner-scoped. A Session is therefore *attributable* today without being a scope, and
this decision keeps it that way.

`mutation_scope` is an opaque, caller-supplied bucket string with an indexed count of
settled `changed` operations behind it (`_COUNT_SCOPE_MUTATIONS`, `pg.py`). The count
has no time window: a scope accumulates for as long as its journal rows are retained.
The Agent's tools fill it with a Run id and a limit of ten, which is safe only because
a Run ends.

Session entries are never pruned one by one. `dlightrag_agent_session_entries` rows
are removed only by the cascade that deletes their whole Session, so a position in a
live Session's entry sequence never points into a gap.

Finally, the package states its own boundary. It exposes no scheduler, no watcher, no
hook and no automatic injection, and its extension example says so in as many words —
lifecycle belongs to the host. It has no LLM dependency of any kind: `asyncpg`, `mcp`
and `pydantic` are its only requirements. This decision respects that boundary rather
than widening it.

## Decision

Two actors write Profile Memory, and this decision keeps them apart on purpose. The
**Agent** is the model answering in a Run: it holds the memory tools, it is what
`answer_run` provenance means, and its tools are renamed here but not re-scoped. The
**Session Memory Pass** is a background consolidation of one Session: it calls a model
of its own on the `extract` role with its own prompt, it holds no tools, and it is what
`session_pass` provenance means. Each is an "agent" in the loose sense and neither is
the other, so this document names them rather than saying "the agent" — and the same
split runs through the rest of it, where a *tool* is something only the Agent has and a
*verdict* is something only the Pass produces.

**One Session contributes at most five records; the profile is never evicted.** The
bound counts the active records whose `origin_kind` is `session_pass` and whose
`session_id` is that Session. The Agent's own tool writes in the same Session are not
counted: they answer to the per-Run bound and to the owner, not to this one. A sixth
distinct memory must be fused into an existing record, replace one, or be dropped. A
fused or updated record belongs to the Session whose pass produced it, so a merge that
reaches into an older Session's records moves that contribution to the Session doing
the merging. A Session that yields nothing writes nothing, and that is the expected
outcome for a Session that only ran tasks. An owner's total profile stays unbounded; a
hard owner-wide ceiling would make a background inference displace a memory the owner
stated themselves, which is the wrong authority ordering.

**The five-record bound is one bound, not one per kind.** A per-kind bound would be
fifteen in the worst case, which is not what "0 to 5 memories" means.

**A successful Run triggers a pass; the coordinator does not know why.** A Run reaches
success along two paths in `RunCoordinator`
(`src/dlightrag/engine/runtime/coordinator.py`): an executor that returns
`Succeeded` is committed by `_finish_success`, and a Fast Run commits its own terminal
and returns `AlreadyCommittedTerminal`, which never passes through `_finish_success`.
Hooking only the first would leave every Fast conversation without a pass. So the
coordinator gains one settlement observer, injected at composition and called after a
committed success on either path; the coordinator itself names no memory concept, and
the observer ignores every Run whose kind is not `answer`. A failed or cancelled Run
triggers nothing. What a pass reads is not filtered by the Run that wrote it, though: a
later successful Run's pass reads every main-Session entry past the watermark,
including those a failed Run left.

The observer only schedules; the pass is a detached task that never rides the
settlement write barrier, because a model call awaited there would hold Run settlement
open and push shutdown past its write grace. Passes go through the process-wide model
scheduler like any other model work and are bounded to two at a time per process, so a
burst of settling Runs cannot starve live answers of model capacity.

**The trigger is an accumulation, and the delta is a watermark.** DlightRAG keeps one
progress row per Agent Session, cascading with it like the Session's other rows. It
records the entry `sequence` the pass has read through; the accumulation is measured
from the text past it each time rather than kept as a running count, so it can never
drift from the entries it counts. `estimate_tokens` measures it, `trigger_tokens`
decides when it is enough, and `max_batch_tokens` bounds one feed. Entries without
text — an assistant turn that only called tools — are never fetched, and a fetch that
fills its window is read even below the trigger, so no run of short or empty entries
can hold the watermark in place. The
read is the main-Session entries after the watermark, so no pass re-reads another's
text; and because entries are only ever removed with their whole Session, the
watermark never points into a gap and needs no recovery rule for one.

**One pass per Session at a time, across processes.** Several processes serve one
deployment, so an in-process guard cannot serialize a Session's passes. A pass claims
the progress row by a conditional update that records a holder and an expiry, the same
shape as the repository's Run leases; a claim that finds a live holder leaves, and an
expired claim is taken over. Every write a pass makes to the row but the claim itself
is fenced by its holder, so a pass whose claim was taken over can no longer move it.
Within one process, a Run that settles while its Session's pass is running asks for
one more pass once that one ends, so its turns are not left waiting for a later Run.

**A pass is two-phase, so a model's nondeterminism never meets idempotency.** The
package's idempotency is exact: a key reused with a different payload is refused
(`pg.py`). A model asked twice about the same text does not answer twice the same, so a
pass that simply re-ran after a crash would collide with the writes of its first
attempt — and a pass can die halfway through applying its verdict. So the verdict is
stored before anything is applied. Under its claim, a pass reads its range, calls the
model, and commits the verdict and the range to the progress row in one transaction;
it then applies each operation with an idempotency key derived from the Session, the
range and the operation's position in the verdict; and it finally advances the
watermark and clears the verdict. A pass that finds a stored verdict resumes applying
it and never calls the model again for that range; an operation already applied
replays as its stored receipt. A pass that dies before storing its verdict has written
nothing, and the next trigger reads the same range afresh.

**The owner's switch governs the pass as it governs the tools.** Each operation the
pass settles rechecks the owner's Profile Memory capability and epoch inside the
store's transaction, through the same guard the Agent's tools use. A pass that finds
the capability off advances the watermark without reading: a conversation held while
memory was off is never mined later.

**The pass reads the Session's own conversation, and nothing else.** Main-Session
`user_message` and `assistant_message` entries only. Tool results are excluded — they
are 94.4% of the text and hold exactly what a memory must not become: a research claim,
an excerpt, a citation. Child Sessions are excluded: a Run owns several Sessions that
share its `run_id`, and a Child's entries are task output, not the owner's own words.
The same holds for what a Child hands back: when a Child settles or asks for guidance,
the Run gives its own agent that result or question as the next input, and it is stored
as a main-Session `user_message` marked `origin: child_notification`. The pass skips it,
for the same reason as a Child Session's entries. Session Notes are excluded — they are
a different plane by ADR 0022 and are never Memory Records. The injected recall block is
part of a request rather than an entry, so the pass cannot read its own output back;
that is a test, not a hope.

**A model decides what is remembered; a counter decides how often it looks.** The pass
calls the configured `extract` role through the existing structured-output path. The
cheap mechanical work — retrieval, candidate gathering, quota arithmetic — gates the
call; the judgement is the model's. Zero records is the ordinary result and costs one
call. No score is stored: a `confidence` column was removed from this schema and
`verify()` refuses a database that still has it (`pg.py`), and this decision does not
bring it back.

**There are three kinds, because a record's `kind` decides how it is recalled.**
`preference` is what the owner wants of an answer and `fact` is what the owner is like;
both exist today. `experience` joins them for what the answering model has learned to
be true about its own work, carried forward rather than rediscovered. The third kind is
not a taxonomy refinement: `kind` decides which section a record stands in. A
`preference` and an `experience` apply to every request, so both stand whenever they
fit; a `fact` applies only when a request bears on it, so it reaches a request only
through relevance. An experience filed as a fact would surface only when a question
happened to share its words — and "this tool needed the extension installed last time"
shares nothing with most of what the owner asks. `fact` narrows to an owner fact and
nothing else. The extraction contract asks for one of the three by name, and the
store's re-filing rule — the same words under another kind re-file the record rather
than duplicating it — already handles a third kind, because it compares kinds for
inequality rather than enumerating them.

**One operation, one name, on every surface that exposes it.** The package's method
names are the vocabulary: `remember`, `forget`, `undo`, `fuse`, `recall`, `lookup`,
`browse`, `clear`. A tool takes the name of the method it calls, prefixed `memory_`, on
both tool surfaces — the Agent's and MCP's — so an operation has one tool name across
the system. Today it has several: the Agent's `remember`, `forget` and `recall_memory`
sit beside MCP's `memory_remember`, `memory_forget` and `memory_recall`, and both read
tools call `lookup`, which is a different operation from `recall` — `lookup` is
`recall` plus the newest other records with their identities, and that extra section is
what lets a caller correct a record recall could not find. The prefix is not
decoration: the Agent's tool list also holds `write`, `edit` and `bash`, and a tool
named only `undo` there reads as undoing a file edit. REST names resources rather than
methods, and keeps the paths it has. What a surface exposes is a separate question from
what an operation is called, and a withheld operation is recorded as withheld rather
than renamed.

| package method | Agent tool | MCP tool | REST |
|---|---|---|---|
| `remember` | `memory_remember` | `memory_remember` | `POST /memory` |
| `forget` | `memory_forget` | `memory_forget` | `DELETE /memory/{memory_id}` |
| `undo` | `memory_undo` | `memory_undo` | `POST /memory/changes/{change_id}/undo` |
| `fuse` | withheld | `memory_fuse` | `POST /memory/fuse` |
| `lookup` | `memory_lookup` | `memory_lookup` | none |
| `browse` | none | none | `GET /memory` |
| `clear` | none | none | `POST /memory/clear` |
| `recall` | none — a Run's bind injects it | none | none |

`fuse` is withheld from the Agent: the move of collapsing several records at once is
not handed to the model that answers, and the Pass that may make that move holds no
tool at all — it applies a verdict.

**The Agent gets `memory_undo`, addressed by record.** The operation exists in the
package and in MCP but not among the Agent's tools, and the reason is addressability:
`undo` takes a `change_id`, and the Agent's tool results carry record identities, not
the changes that produced them. Authority is not the reason, because `memory_undo`
reaches nothing the Agent cannot already reach. Undoing a plain remember forgets the
record, which `memory_forget` does. Undoing a supersession puts the earlier sentence
back, which `memory_remember` with that sentence and `supersedes_id` does. Undoing a
forget restores a record, but addressed by record the Agent can only name an active
one, so a forget is out of its reach.

So `memory_undo` takes exactly one of `memory_id` or an exact `body`, the way
`memory_forget` does, and the store resolves the named active record to the change that
produced its current form. `change_id` stays accepted for callers that hold one — MCP
and REST keep naming a change. Undo compensates one change, not a history: the package
refuses to undo an undo, and a compensation writes the restored sentence as a new
record rather than reviving the old row, so a record whose current form came from an
undo is a conflict to undo again. A named record that is not active is a conflict
rather than a guess, because guessing which earlier change the caller meant is how a
memory ends up in a state nobody asked for. `Memory.undo` gains the `mutation_scope`
and `mutation_limit` that `remember` and `forget` already take, so the Agent's undo
counts against the same per-Run bound as its other writes.

The store needs one column for this, `last_change_id` on the record row, set by every
settlement that produces or re-files a record and indexed with the owner. The
alternative — resolving through the operation journal's JSONB `receipt.memory_ids` —
works only with a JSONB index and leaves the invariant implicit, and `fuse`'s own
compensating undo has to ask the same question of the records a merge produced: which
change last touched this one.

**Candidates are fused against the profile, not appended to it.** The pass reads the
owner's existing active records through the same search a recall uses — now over every
kind, or a preference or experience could never be found to fuse with — and gives the
model the candidates alongside the new material. One call decides add, update, fuse
or drop — not two mechanisms, and not a second read surface.

**Who judges a merge, and who words the merged sentence.** The pass's model call judges
every candidate against the owner's records. A candidate bearing on nothing existing is
added; one that refines a single existing record updates it, which is `remember` with
`supersedes_id`; one close to several records at once is fused, which `supersedes_id`
cannot express because it names one predecessor; and one that adds nothing is dropped.
The call sees the new material and the candidates in one request, so it judges
similarity and words the result together.

So the "automatic" in this mechanism means *unprompted*, never *unjudged*. The host
holds no similarity rule and fuses nothing on its own; it applies a verdict, and every
fuse in the system comes from a judgement — the pass's, or the owner's. Two
consequences belong in the verdict rather than being discovered later: a fused body can
collide with an active record the verdict did not name, which the store reports as a
conflict, and a verdict that names only part of a group leaves the rest standing, which
is why the verdict names the whole group it means to replace.

An owner merging records from REST or MCP supplies the sentence instead, because no
model stands on that path — putting one there would either give the package an LLM
dependency or put a synchronous model call inside an HTTP request, when this
repository settles model work in Runs. The endpoint takes the sentence the owner
ended up with. The Settings list is where a merge would naturally be proposed and
edited, but it has no merge control yet: it labels what the pass wrote and lets the
owner forget it, and a merge from Settings is left to a later change. The package's rule covers both paths: **the
store never invents the text** — it validates, settles and journals a string it is
given, and `origin_kind` records who gave it.

**A `fuse` action is added to the package.** Fusing several records into one is one
atomic operation, not a chain of supersessions and an insert: a detached pass can die
between them, and the failure mode is the owner's memories disappearing.
`MemoryOperationAction` gains the value, the operation table's check constraint gains
it, `Memory` gains `fuse`, and the policy layer validates it as it validates the other
three. Undo of a fuse restores its sources, and the existing `_restores_an_active_body`
guard refuses a restore whose body another active record has since taken.

A fuse names two to ten records, an optional `kind`, and the body that replaces them.
`kind` is required only when the sources disagree: sources of one kind fuse under that
kind, and a mixed set must say which the result is — the case a `supersedes_id` re-file
cannot express. Refusals split the way they already split for the other operations.
What is wrong with the request itself — fewer than two records, more than ten, a
repeated id, an empty body or one over five hundred characters — is refused before
storage. A mixed set without a `kind` is refused too, but during settlement, because
the sources' kinds are stored state; the refusal rolls the settlement back. What is wrong with the store's current state — a
record that is not the owner's, one that is not active, a body that collides with
another active record — returns a receipt whose outcome is `conflict`. A set that is
not entirely active is a conflict rather than a partial merge, because a merge that
silently drops a source is how a memory disappears without anyone asking.

The receipt is the one every memory operation returns. Its `memory_ids` lists the new
record first and the superseded sources after it, in the order given; its single
`supersedes_id` is null for a fuse, because a fuse has several predecessors and they
are already in `memory_ids`. The outcome is `changed` or `conflict`; a replayed key
returns the stored receipt.

A management fuse leaves `mutation_scope` unset, like every other management write: the
per-Run bound guards a model's loop, and this caller is not a model.

```
POST /memory/fuse
Idempotency-Key: <1..255 chars>            # required, as on every memory mutation

{ "memory_ids": ["<uuid>", "<uuid>"],      # 2..10, distinct
  "kind": "preference",                    # optional; required when the sources differ
  "body": "<the merged statement>" }       # 1..500 chars, the same limit as remember

200 { "action": "fuse", "outcome": "changed" | "conflict",
      "change_id": "<uuid>", "kind": "preference", "body": "<merged>",
      "memory_ids": ["<new>", "<source>", "<source>"],
      "supersedes_id": null, "target_change_id": null }
409 { "detail": "<public message>", "error_type": "<by status>", "error_kind": null }
```

```
memory_fuse(memory_ids: string[] /* 2..10 */,
            body: string /* 1..500 */,
            kind?: "preference" | "fact" | "experience",
            idempotency_key: string /* 1..255 */)
  -> the receipt dict `_receipt` already returns
```

**A pass bounds itself per pass, not per Session.** Every operation a pass settles uses
a `mutation_scope` naming the Session and the range it read, with a limit of ten. A
Session-wide scope would accumulate for the Session's whole life, so any fixed limit
would eventually shut a long Session out of memory for good; a per-pass scope bounds
exactly what the limit is for, a single pass looping. The Agent's per-Run scope is
unchanged, and the two never share a bucket.

**Background writes announce themselves.** A new `origin_kind`, `session_pass`, joins
the closed set, so a record the pass inferred is distinguishable from one the owner
asked for — in the Settings list, in the operation journal, and in any export.

**The owner can see and delete what the pass wrote; the owner does not approve it.**
The Settings memory list already pages an owner's active records; it gains a source
label, and `session_pass` records are deletable there like any other. Approval was
rejected: a queue nobody works through turns "zero memories" from an outcome into a
default, and a pending state would be a fourth lifecycle value orthogonal to the three
that exist.

**Injection is defended in the prompt and at the reading end.** The pass's system
prompt states the contract — record a third-person statement about the owner or about
the answering model's own work, never an instruction — and the block a Run injects
keeps saying that it is context and not instruction, with the current request taking
priority. No mechanical imperative-detector is added: its cost is certain and silent,
since a legitimately-phrased preference that reads as a command would be discarded,
while a motivated injection rephrases itself as a statement and passes.

**The mechanism is a host concern with a package-shaped hole.** The pure parts — the
prompt, the verdict types, applying a verdict to the store — belong to
`dlightrag-memory`. The trigger, the progress row, the entry read, the model call and
the claim belong to DlightRAG. The package gains no LLM dependency and no lifecycle.

**What one recall injects is bounded by characters, and being left out is not
deletion.** `RECALL_CHAR_BUDGET` is the total for one recall — the block a Run's bind
injects, and the same two sections of a `memory_lookup`; the newest other records a
lookup lists are bounded by count only. It rises from four thousand to eight thousand
characters and is shared by two sections that answer different questions.

The relevant section answers "what bears on this request". It is filled first, from a
search over every kind: the exact, BM25 and dense legs filter `kind = 'fact'` today,
and the filter goes, while the vectors and BM25 language they need are already computed
for every record whatever its kind. It holds at most `RECALL_TOP_K` records and at most
half the budget, four thousand characters. The standing section answers "what applies
whatever the request": preferences and experiences, newest first, from a fetch
separated from `RECALL_TOP_K` and raised to a hundred, the store's own clamp. It takes
whatever the relevant section left — never less than four thousand characters, and more
when the request matched little. A fixed even split would strand most of the relevant
half on most requests, since ten short records fill a few hundred characters; filling
relevance first and letting its unused share flow to the standing section gives each
section its floor without wasting either.

That ordering is what makes a crowded-out record reachable. A preference or experience
pushed out of the standing section by newer ones is still found by relevance when a
request bears on it, both in what a Run's bind injects and in what `memory_lookup`
returns; and a `fact` reaches a request only that way, as today. With relevance able to
bring any record back, the standing section need only hold what applies to every
request, which is why half the budget is enough for it.

**A record appears at most once in one recall, and duplication of meaning is settled
at write time.** Identity is the only deduplication a recall performs: a record the
relevant section shows is left out of the standing section, the legs of the search are
already merged by record in `_fused`, and a lookup's newest-other list already excludes
what the other two sections show. Deduplicating meaning when reading — comparing
records' wording on every recall — would cost a comparison per request and would hide,
rather than repair, a profile that says one thing twice. The repair belongs to the
writers: the store's unique normalized body refuses an exact repeat, and the Session
Memory Pass fuses records that say the same thing differently. The one repetition kept
on purpose is between the injected block and a `memory_lookup` result in the same Run:
the block carries no identities, and identities are what a lookup is for.

The block says what it left out. When the standing section could not show every
standing record, the block ends with one line giving how many it left out, so the
Agent knows a search can find more than it was handed. The line names no tool, because
the same block reaches Fast, which holds none; `memory_lookup`'s own description says
it finds the standing records an injected block left out.

A record that does not fit is not deleted, not flagged and not superseded — it stays in
the store, remains findable by relevance and listable and forgettable in Settings, and
simply stops standing until a later change puts it back in front. Refusal is not
eviction, and neither is this. Neither section ranks beyond its own order —
newest-first for standing, fused relevance for relevant: a stored score would weight
whatever was recalled most rather than what matters most, and it would make a new
preference hardest to admit.

Every line gains its kind label, and that forces the budget's accounting to become
honest. `recall_sections` labels kind only in the newest-other section; with
`experience` standing beside `preference`, and any kind able to appear among relevant
records, an unlabelled line cannot be told from another, though one is what the owner
asked for and another what the answering model learned. The budget counted the wrong
thing for this: `_within` sums `len(record.body)` while a rendered line also carries a
bullet and now a label, and `reserved_auto_recall_text` reserves
`min(RECALL_CHAR_BUDGET, 2 * RECALL_TOP_K * MEMORY_BODY_LIMIT)` plus a fixed
five-character slack. So `_within` counts rendered length, and the reservation becomes
budget-driven — the character budget, the framing of the most records the body limit
permits inside it, and the closing count line — which stays correct whatever the fetch
limits become.

`RecallResult` changes shape with this: its preference and fact sections become a
standing section and a relevant section, and it carries the number of standing records
it did not show. The store's `search_facts` becomes `search`.

**Configuration follows the existing shape.** `answer.agent.memory_consolidation`,
beside `answer.agent.session_notes`: `enabled`, `trigger_tokens` (1500),
`max_batch_tokens` (40000), and `session_record_limit` (5). `enabled` is the
deployment's switch; each owner's Profile Memory switch still governs that owner.

## Considered options

- **A hard owner-wide ceiling of five.** Rejected: it binds the whole profile, so a
  background inference would evict what the owner stated. ADR 0022's rule that refusal
  is not eviction cuts the other way here, and an owner-wide cap is eviction with
  extra steps.
- **Counting every record the Session's provenance names.** Rejected: the Agent's own
  tool writes would consume the pass's allowance, and an owner who told the Agent five
  things would get no consolidation at all.
- **Hooking only `_finish_success`.** Rejected: a Fast Run commits its own terminal and
  never passes through it, so every Fast conversation would go unread.
- **A new durable Run kind for the pass.** Rejected: ADR 0007 closes `run_kind`, a
  leased Run that makes one model call adds a lifecycle, a lane and observability for
  work that is explicitly allowed to be lost, and it would appear in the inspector as a
  Run the owner never submitted.
- **Riding the Session Note promotion at tool settlement.** Rejected for the judging
  half: promotion settles inside the Run's write barrier, and a model call there would
  hold the settlement open and break the guarantee that memory never fails a Run.
- **Re-calling the model when a pass is retried.** Rejected: the second answer differs
  from the first, the package refuses a reused key with a different payload, and the
  operations the first attempt already applied would be stranded.
- **A Session-wide `mutation_scope`.** Rejected: the count has no time window, so any
  fixed limit is eventually exhausted and a long Session stops being consolidated.
- **Reading every entry type, tool results included.** Rejected: 94.4% of the text,
  and it would import exactly the material `remember` forbids — research claims,
  excerpts, citations — into a store that is injected into every future Run.
- **Making the pass read Session Notes.** Rejected: a note is bytes at a path with no
  kind, no order and no attribution, and ADR 0022 says a Session Note is never a
  Memory Record.
- **A mechanical pipeline with no model.** Rejected: it cannot fuse two differently
  worded statements of one preference, which is the whole of "融合", and on this
  deployment it would rest on embeddings that are optional configuration.
- **Reusing `lookup` as the reconciliation read.** Rejected: recall returns a fact only
  on evidence that it bears on the query, which is the gap `lookup`'s own newest-other
  section exists to cover; a candidate memory is not a query, and that section is
  ordered by time, not similarity.
- **A `confidence` or `salience` score on the record.** Rejected: the column was
  removed and `verify()` fails a database that still carries it. A pass decides at
  write time and stores the outcome, not a number.
- **Filing `experience` records as `fact`s.** Rejected: a `fact` reaches a request only
  through relevance, so what the answering model learned about its own work would
  surface only when a question happened to share its words.
- **A fourth `MemoryKind` for the pass's own writes, or a `source` column beside
  `kind`.** Rejected: whose statement a record is and how it was learned are different
  questions, and `session_pass` already answers the second as provenance.
- **Leaving the Agent's tools unprefixed.** Rejected: one surface would carry two
  conventions, and a bare `undo` beside `write` and `edit` reads as undoing a file.
- **Putting a `change_id` into `memory_lookup`'s result so the Agent could name a
  change.** Rejected: a read-only tool would carry a field whose only use is to mutate,
  and it solves nothing the record selector does not.
- **Resolving a record's latest change out of the journal's JSONB receipt.** Rejected:
  it needs a JSONB index, and it leaves implicit an invariant `fuse`'s compensating
  undo also depends on.
- **Approval before an inferred memory takes effect.** Rejected: it makes zero the
  default, and it adds a lifecycle value orthogonal to `active`, `superseded` and
  `forgotten`.
- **Raising `RECALL_CHAR_BUDGET` alone.** Rejected: with real memories a few dozen
  characters long, the ten-record fetch binds first and the larger budget is never
  reached.
- **Leaving the search fact-only.** Rejected: a preference or experience crowded out of
  the standing section could then be recalled by no one but the owner paging Settings,
  and the pass could find no similar preference or experience to fuse with —
  consolidation would silently work for one kind in three.
- **Filling the standing section first, as today.** Rejected: it can fill the whole
  budget, and then relevance — the one way back for a crowded-out record — gets
  nothing.
- **A fixed even split.** Rejected: ten short relevant records fill a few hundred
  characters, so the relevant half would sit mostly empty on most requests while the
  standing section was cut short.
- **Deduplicating meaning at read time.** Rejected: a comparison per recall, and it
  hides a profile that says one thing twice instead of repairing it; the unique
  normalized body and the pass's fuse repair it at write time.

## Consequences

Landing order, one sequence:

1. The `experience` kind and the recall in the package: `MemoryKind`, the record
   table's check constraint, the policy's kind set; `search` over every kind in place
   of the fact-only `search_facts`; the relevant section filled first under
   `RECALL_TOP_K` and half the budget, the standing section of preferences and
   experiences taking the rest from a fetch of a hundred, identity deduplication
   between them, and the closing count line; `RecallResult` reshaped to those sections;
   `RECALL_CHAR_BUDGET` raised to eight thousand; kind labels on every line; `_within`
   counting rendered length; and `reserved_auto_recall_text` made budget-driven. These
   land together: each section's floor means nothing until the count stops binding
   first, and the label is affordable only once the accounting counts it.
2. The `fuse` operation in the package: the action value, the check constraint,
   `Memory.fuse`, the policy validation, and the compensating undo with its restore
   guard.
3. The tool rename: the Agent's `remember`, `forget` and `recall_memory` become
   `memory_remember`, `memory_forget` and `memory_lookup`, and MCP's `memory_recall`
   becomes `memory_lookup`. Everything that names them moves in the same step:
   `CHILD_FORBIDDEN_TOOLS` in `src/dlightrag/engine/answer/tools/composition.py`,
   the `tool.name == "remember"` checks in
   `src/dlightrag/engine/answer/execution/acceptance.py` and
   `src/dlightrag/engine/answer/orchestration/orchestrator.py`, the
   `_PROFILE_MEMORY_GUIDANCE` section of `src/dlightrag/engine/answer/prompts/agent.py`,
   the name-based test assertions, and `packages/memory/examples/pi-extension.md`. This
   changes each tool's schema digest recorded in a step's `call` (ADR 0043), the pinned
   Tool Plan and the prompt prefix, which is why it is one deliberate step.
4. The `session_pass` provenance kind and its constraint.
5. The `last_change_id` column and its index, and `Memory.undo` accepting a record
   selector and a mutation scope and limit.
6. The Agent's `memory_undo` tool, bound like `memory_forget` and added to
   `CHILD_FORBIDDEN_TOOLS` beside `memory_remember` and `memory_forget`, or a Child
   could compensate a write it is not allowed to make.
7. The package-side pure parts: the pass's prompt, the verdict types, and applying a
   verdict to the store.
8. The host side: the progress table, the coordinator's settlement observer on both
   success paths, the claim, the two-phase pass, the capability guard, the per-process
   bound under the model scheduler, and the configuration section.
9. The Settings source label, the REST and MCP `fuse` operations, and the test that the
   pass cannot read an injected recall block.

Live documents revised with the implementation: `docs/retrieval-answer.md` (the
Session Memory Pass and the renamed tools), `docs/interfaces.md` (the recall sections,
`POST /memory/fuse`, the tool names) and `docs/configuration.md`
(`answer.agent.memory_consolidation`). Earlier ADRs that name `recall_memory`,
`remember` or `forget` describe their own time and are left as written.

The package's public interface grows in several places at once — a `kind` value, an
`origin_kind` value, an action, a column, `Memory.fuse`, `Memory.undo`'s selector and
scope, `search` in place of `search_facts`, and `RecallResult`'s new sections — so the
next release, which moves every workspace package's version in lockstep, carries it;
its own tests, which assert the MCP tool set exactly, are extended deliberately rather
than incidentally.

The memory kind is written out in six places, and each moves with the new value:
`MemoryKind` in `packages/memory/src/dlightrag_memory/models.py`, the kind set in
`packages/memory/src/dlightrag_memory/policy.py`, `_MemoryKind` in
`packages/memory/src/dlightrag_memory/mcp_server.py`, `MemoryKindInput` in
`src/dlightrag/engine/answer/tools/memory.py`, and the request models of
`src/dlightrag/adapters/http/rest/routes/memory.py` and
`src/dlightrag/adapters/http/browser/routes/memory.py`. The descriptions of
`memory_remember` on both tool surfaces and `Memory Record`'s definition in
`docs/domain-language.md` move with them, since today they say "preference or fact"
and forbid task state.

Two follow-ups are recorded rather than solved. A consented Session's dataset export
already includes a Profile Memory the Agent recalls (ADR 0044), so a memory the pass
wrote from a conversation enters training data with no user turn visibly authoring it;
the switch that governs contribution is the only gate, and whether that is enough is
that decision's question. And deleting a conversation does not delete what the pass
derived from it: the records are owner-scoped and outlive the Session, as an Agent's
`remember` does today, except that nobody asked for these — the Settings source label
is how an owner finds and removes them.

Residual risks. A Session that stops before its watermark reaches its last entry leaves
that tail unread until the same Session's next successful Run, or for good — bounded
loss, chosen over an unbounded read. Doubling the reserved recall block raises the
fixed input every accept must leave room for, so a small-context model is likelier to
find Fast unviable; acceptance already logs when that happens, and that log is how it
would be seen. Passes compete with live answers for model capacity, which the
per-process bound limits but does not remove. And the measure of success is human
judgement over a fixed set of Sessions, not a metric: the mechanism's ordinary correct
output is nothing at all.

Amended: the first of those risks, an unread tail, is retired by the amendment above. A pass reads to
the end of its Session, so a conversation that never accumulated enough text, or the last part of one that
did, is read by the pass of the Run that wrote it. What can still wait is only what a failed or cancelled
Run wrote (it schedules no pass; the next successful Run's pass reads it), a batch whose model call failed
(the next Run's pass asks again), and whatever lies past the 25-batch safety valve. A pass of several
batches makes several model calls, and each call carries up to eight turns of context of up to 500
characters.

Amended: what that note still left waiting is recovered by the sweep (the amendment in Status). A pass lost
to a restart, a batch whose model call failed, the text a failed or cancelled Run wrote, and whatever lay
past the 25-batch valve are found again from the tables at the next look, at most one interval away. Two
kinds of unread text remain, by design: a Session backing off after failed passes, which waits at most six
hours, and any Session whose owner has Profile Memory off, which is never read. "Explicitly allowed to be
lost", said above of a pass as the reason against a durable Run kind, is read as: a pass may be lost, and is
found again.
