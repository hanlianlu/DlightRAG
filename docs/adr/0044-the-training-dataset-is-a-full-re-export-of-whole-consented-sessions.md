# The training dataset is a full re-export of whole consented Sessions

`scripts/export_traces.py --dataset DIR` packs the conversations their owners contributed
([ADR 0041](0041-a-conversation-improves-the-product-only-when-its-owner-asked.md)) into a
de-identified, split and reproducible dataset: `train.jsonl`, `eval.jsonl` and a `datasheet.json`
that says what is in them and what is not. Each dataset is read afresh from the database, whole
Sessions at a time. Nothing is merged into an older one, so an owner who turns the switch off is
simply absent from the next.

## Status

Accepted and implemented. There is no migration and no new table. It completes what ADR 0041
listed as still to do, the de-identification of what is exported and its packaging into a dataset;
the archive that would keep a conversation after its owner deletes it is still not built. The
operator's raw export, one record per Run on stdout, remains the tool for inspecting the database
and now writes the same record. It follows the accepted
[ADR 0042](0042-runtime-policies-are-read-at-the-door-and-pinned-into-the-run.md) and
[ADR 0043](0043-a-step-records-the-model-parameters-and-prompt-digests-it-ran-on.md), whose `call`
record a step passes through unchanged.

## Context

ADR 0041 decided who may be exported and left the export as a host tool that writes one record per
Run. That is not yet something a model can be trained on or evaluated against:

- Every id in it is the database's own. An owner id is an unsalted SHA-256 of the issuer and the user
  id, which anyone who knows the issuer and an email can confirm, or, in `none` and `simple`
  authentication, a constant anyone can compute from the source. A dataset that leaves the
  operator's machine must not carry them.
- It has no split, so nothing keeps what a model was evaluated on out of what it learned from, and
  nothing keeps that true when more is exported later.
- It repeats itself. Each evidence row is a snapshot of the whole ledger after one tool call, so a
  Run with forty tool calls holds forty growing copies of the same passages (the product itself
  slims all but the newest of them once a Run is old), and its event stream holds every streamed
  token again.
- It holds what no consumer can use or should have: the provider's opaque replay state, the Run's
  error message, its whole stored result, and, for a Run still going, the prepared input, which
  carries the salt of its resource handles and its connection bindings.
- It says nothing of how long anything took, and nothing of what the owner thought of an answer,
  both of which are in the database.

## Decision

**An episode is one Run, one JSONL line.** An action is an assistant message (content, reasoning,
tool calls, and the `call` that made it, [ADR 0043](0043-a-step-records-the-model-parameters-and-prompt-digests-it-ran-on.md)).
An observation is the tool result that answers a call. The end is the last assistant message with
the Run's `status` and `stop_reason`. A Child is a nested episode, tied to the step that spawned it
by `parent_call_id`. The reward is the owner's judgement of an answer, kept as they gave it in
`labels`, `{agent, sequence, value, reasons, note, given_at}` from `dlightrag_answer_feedback`,
with a Child's judgement named by the Child and the sequence of the Child's own step. There is one
reward convention: the raw labels, and never a number computed from them. `given_at` is when the
owner first judged the message, the column the table keeps for the training record; a changed mind
replaces the vote, reasons and note and keeps that time.

**The unit is the Session, and each dataset is a full re-export.** A dataset takes every finished
Run (succeeded, failed or cancelled) of every Session ADR 0041's rule admits, so a Session is in or
out as a whole. `--owner`, `--since` and `--until` apply to a Session as a unit, by when it began,
with the same `LEAST` of its row's creation and its first remaining Run's acceptance that the
consent rule uses: a Session that began before `--since` is out although a Run of it came later,
and one that began before `--until` is in with all its Runs. `--limit`, `--run-id`, `--dedupe` and
`--out` are refused with `--dataset`, because each would cut a Session or leave the directory. A
Run still queued or running is not read: it has no end to learn from, and its prepared input is
execution state with secrets in it. Every consent condition is the query's, so no option widens it.
Re-reading everything, rather than adding to an older dataset, is what makes the switch work: an
owner who turns it off is absent from the next dataset, and no dataset has to be edited to forget
them. A copy already written is not recalled, as ADR 0041 says.

**The record is the existing one, with these changes.** It is schema version 5.

- Dropped: `provider_state` from every step, because only the provider that wrote it can read it;
  the `tool_events` list, whose timing moves into the steps and whose rest is the streamed text the
  entries already hold; `error_message`, with `error_kind` kept; the stored `result`, but for
  `result.sources`; and the passage text of the evidence snapshots.
- What the answer cites is `result.sources`. The citation processor builds it from the answer's own
  markers: each source names its `cited_chunk_ids` and holds only the cited chunks, passage text
  included (`citations/source_builder.py`), and the product's own compaction keeps the text of a
  chunk in an evidence snapshot only when its source is among them (`run_store.py`,
  `_COMPACT_RUN_EVIDENCE`). So "cited" is known reliably, and the cited passages are held once, in
  the sources, instead of in every snapshot. `evidence` keeps each snapshot's digest and locator,
  which tell which ledger state a tool call left (a Research tool settlement writes one, with an
  empty locator). What the agent saw, cited or not, is in its tool results.
- `steps[].schema_version` was always null, because the entries query never selected it. It is
  selected now: it is the version of the payload format, the one fact a reader needs to parse a
  payload safely across product versions.
- `session` is `{id, head_entry_id, carried}`. `head_entry_id` is the Run's last entry. `carried`
  holds, whole and root first, the entries of the chain above the Run's first step that no exported
  Run wrote, because retention pruned their Run, its owner deleted it, or an option left it out. A
  Run really ran on that context, so a record without it would teach a model to answer without
  what it was shown. The rest of the chain is the steps of the Session's other records, and every
  step names its parent, so a consumer rebuilds the chain by joining steps and carried entries on
  entry ids, with nothing dangling. The export computes the chain to find what is carried and does
  not write it: a list of every entry above a Run, in every Run, grows with the square of a
  conversation.
- `call` is passed through untouched.

**Words of a deleted conversation have one way in.** A Run's own steps are the entries that state
it as their writer, so the words of a fork its owner deleted are in no record: they are not on the
chain of any surviving Run. But a Session is shared by a conversation and its forks, and deleting a
conversation deletes its Runs and leaves their entries (ADR 0041). If the surviving conversation
is a fork of the deleted one, its chain passes through the deleted one's entries, and they are
carried as context, as are the entries of a Run that retention pruned. So a conversation its owner
deleted can still appear in the dataset, as context in the records of a fork of it. A deleted
parent and a retention-pruned one are the same to the database, which keeps the entries and no
Run in both cases, so the export cannot carry the one and withhold the other. This decision
accepts that case, because the Run really ran on those words, and the datasheet states it among
its limits.

**Time is stated where it is measured.** A step has its entry's `timestamp`, and an assistant
message or a compaction its `call.queued_ms` and `call.duration_ms`, both from the monotonic clock
of the model call and passed through. A tool result has `tool.started_at`, `tool.ended_at` and
`tool.duration_ms`, the database's times of the Run's `tool_start` and `tool_end` events for its
call and the whole milliseconds between them (half a millisecond rounds up). A Child has
`started_at` (its row's making), `finished_at` (its last entry's timestamp, or its row's last
change if it wrote none) and `duration_seconds`, and has none while it has not reached
`succeeded`, `failed` or `cancelled`. The Run keeps `created_at`, `started_at`, `finished_at` and
`duration_seconds`. Three facts shape this and are stated in the datasheet:

- Entry timestamps are `datetime.now(UTC)` of the worker when it built the entry, just before the
  commit, and the other times are the database's `NOW()`, the start of the writing transaction.
  They are two clocks, and the timestamps are not commit times.
- The two events are keyed to their call by the id the model gave it, in their payload, and a
  Child's tool calls land in the same stream as its parent's, told apart by nothing else. So a call
  gets a time only when its Run's events hold at most one start and at most one end for it and no
  other result of the Run carries its id; an unmatched start or end gives its own time and no
  duration, and anything ambiguous gives none. A tool run again after a crash starts twice. A Run
  whose events retention trimmed has no tool times. None is guessed.
- The interval includes the commit of each event, so it is longer than the tool's own run, which
  the `tool_end` payload also states (`duration_ms`, monotonic). On twelve calls of the development
  database it was between 2 and 164 ms longer. The interval is what the dataset records, because it
  is exact from the two events and the same for every tool, and the payload's figure is not used.

**Every id becomes an HMAC-SHA256, and the text is not scrubbed.** Owner, `submitted_by`, Session,
Run, Entry and Child Session ids, and the ids the Runtime mints for a tool call's intent and
attempt, a compaction's projection, a steer and a fork's lane, become the hex HMAC-SHA256 of the
stored id under an operator key, `DLIGHTRAG_EXPORT_KEY`, at least 32 characters, read from the
repository `.env` (or the environment) by the tool's existing `read_env`. Dataset mode refuses to
run without it, and raw mode does not need it. The same key gives the same pseudonyms, so
successive datasets link up and parent links still resolve inside the files. An id is
replaced wherever it stands: in a field, as a key, in a tool call's arguments, and inside the
words of a result. A UUID the export did not meet, such as the id of a page the agent was handed,
is left as written, because it is not the product's. An owner id is a SHA-256 digest, which like a
UUID cannot be a word, so it is found inside text too: the path of an owner's skill directory
names it, and a tool that fails writes the path into its result. An id of some other shape, such
as an owner in a test fixture, could be a word inside a sentence and is replaced only where a
value is exactly it. Ids are learnt from the places the export itself writes them, including an
input (a parent Run that retention pruned) and a payload, and then replaced wherever they stand,
which catches ids no field list would: the Fast path writes the Run's own id as the
`acceptance_id` of its entries. Ids that are not replaced are named in the datasheet: tool call ids
the provider issued, resource handles, corpus document, chunk and source ids, and digests.

Text is not scrubbed. A pattern finds an email and misses a name, a Chinese one above all, and it
corrupts tool arguments and URLs, which a model must be able to learn to write. The datasheet says
plainly that the text is not de-identified and can hold third parties' content: web pages the
agent fetched, files the owner attached, personal connections such as Notion, and the Profile
Memory the agent recalled. The files are written with mode 0600 in a directory of 0700, as is the
raw export's `--out` file (its directory is left as it is), and the key appears in none of them.
The datasheet carries only a fingerprint of it, the first eight hex digits of the HMAC of a
constant, which two datasets share exactly when they share a key.

**The split is a hash of the stored Session id.** A Session goes to `eval` when u < f, where u is
the first eight bytes of `SHA-256("dlightrag-split-v1" || 0x00 || session_id)` as a big-endian
integer divided by 2**64, and f is `--eval-fraction`, 0.10 by default. A Session's Runs and
Children are on its side, and each record states it as `split`. There is no secret and no stored
manifest: exporting on more data never moves an old Session, and raising f moves Sessions from train
to eval and never back. The hash is of the stored id, not its pseudonym, so the side does not
change with the key and can be recomputed by anyone who knows the id.

**The output is `train.jsonl`, `eval.jsonl` and `datasheet.json`, and it is reproducible.** The
directory must not exist or must be empty. Records are sorted by (Session pseudonym, Run creation
time, Run pseudonym) with keys sorted, so the same database, key and options write byte-identical
`train.jsonl` and `eval.jsonl`, and the datasheet differs only in `generated_at`. This needed one
fix: evidence rows were ordered by `(run_id, result_ordinal)`, and every snapshot has ordinal 0,
so their order was the heap's; they are ordered to the end of the primary key. The datasheet is
written last, so a directory without one holds a dataset that did not finish. It states the
exporter's commit, the schema version, when it was made, the selection rule and consent basis
(the owners who contribute and the earliest moment a permission began), counts of owners,
Sessions, Runs, Children and steps by mode, status and split, the time window, the label
distribution (votes, reasons, notes, the unlabelled share, main against Child), what is left out
and why, the split version and f, the key's fingerprint, the SHA-256 and size of each file, how
each time is measured, and the honest limits. Its counts are computed from the records written,
not beside them.

**The key lives in `.env`, so the application must accept the name.** The application refuses to
start when its environment or `.env` holds a `DLIGHTRAG_*` name that is not a setting, and
`docker-compose.yml` hands `.env` to its services. `DLIGHTRAG_EXPORT_KEY` joins the names that
share the namespace without being settings (`_AUXILIARY_ENV_NAMES`, with the HTTP client's), and
configuration.md says so. An image that predates this still refuses a `.env` that holds the key,
so deploy first and add the key after. The key is a secret like the others in `.env`, and it is not
passed to the agent's own subprocesses, which strip `DLIGHTRAG_*`.

## Not building

- **Preference pairs, or pair synthesis.** The product does not regenerate an answer, so the honest
  sources of a pair were looked for in the development database and found at none: a fork asks a new
  question, a thumb down followed by a thumb up is a different state, a cancel and a resubmit is no
  judgement, and Children have different objectives. A dataset with pairs would be built from
  nothing the owner said.
- **Labels inferred from a cancellation, a re-ask or a fork.** The owner's thumb is a judgement.
  Those are behaviours, and reading them as one puts words in the owner's mouth.
- **A text scrubber,** for the reasons above.
- **An archive of deleted conversations.** Nothing consumes one yet, and ADR 0041 says that
  deleting deletes. Retention expiry is never archived.
- **Incremental datasets,** and a manifest to keep their sides. The split needs none, and a full
  re-export is what lets the switch work.
- **A server endpoint or a screen.** It is an operator's tool on the host, and the server gains no
  surface that could be asked for conversations.
- **A split by owner.** One owner's Sessions are all on one side, which isolates people and costs
  most of the data of a deployment with a few owners. It is a different decision, and the limits
  say that a split by Session does not isolate people.
- **Shards, and a `prompts.jsonl`.** The records are one file a side, and the call records hold
  digests of the prompt and no prompt text, so there is nothing to write.
- **Passage text in the evidence snapshots.** See above.

## Considered options

- **Evidence rows with the cited passages' text, and no `result`.** Rejected. Every snapshot would
  repeat the cited passages, and the sources already hold them once.
- **Whole-text scrubbing with a pattern, or a model.** Rejected, as above.
- **Pseudonyms from a random table kept beside the dataset.** Rejected. A table must be stored
  and protected as the key is, and a key makes the same pseudonyms again with nothing to keep.
- **A list of the chain above each Run, `ancestor_entry_ids`.** Rejected. Every step names its
  parent, so the chain is rebuilt from the Session's steps and the Run's carried entries, and the
  list adds nothing while it grows with the square of a conversation: on a synthetic conversation
  of 120 turns and 1,080 entries it was 76% of a 5.8 MB `train.jsonl`, against 1.4 MB without it.
- **A split by hash of the Run.** Rejected: a Run carries its Session's earlier text, so a Run on
  one side and its follow-up on the other would put the question in training and the answer in
  evaluation.
- **Including Runs still going.** Rejected, above.

## Consequences

- A dataset is as large as the conversations it holds; a synthetic conversation of 120 turns and
  1,080 entries is 1.4 MB, and the export of it takes under a second. `carried` can repeat, in
  every record whose chain passes through the same pruned entries.
- The datasheet's counts and file hashes are of what is written, so a dataset can be checked
  against its datasheet, and two datasets can be told apart by the hash.
- Tests hold the rules on a scratch database through the command: the words of a deleted fork in
  no file; labels only for exported Runs, a Child's on its step and a changed mind replacing the
  first; no stored id in any file, with every Run's chain rebuilding to the root from the steps and
  the carried entries alone and a pruned parent's entries carried; the sides of Sessions, their
  stability under more data and under a larger share; whole-Session selection by the Session's
  start, the switch off, a Session begun before the grant and a Run still going; reproducibility,
  the datasheet's counts and hashes, and the modes and contents of the files, a raw `--out` file
  included; the refusals; tool and Child timing with every unmatched, ambiguous or trimmed case;
  and the application's acceptance of the key's name in `.env`.
