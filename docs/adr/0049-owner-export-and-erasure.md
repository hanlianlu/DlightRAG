# One owner's data can be exported and erased

A hosted DlightRAG holds one person's conversations, transcripts, memory,
connections, Agent accounts, Skills, products and telemetry across several
stores. This decision adds two owner-scoped operations — an export that says
what we hold, and an erase that removes it — each with an append-only receipt
and a clock, because a data-subject request has to be answered with evidence
rather than by hand.

## Status

**Accepted and implemented in part.** `dlightrag-owner-data` runs both
operations against one deployment; the coverage is discovered from the schema,
the ledger is `dlightrag_owner_data_requests`, and the acceptance test erases a
seeded owner and then looks for them in every column of every table. Deferred,
and stated here rather than implied: a signed time-limited download (the command
writes a directory an operator hands over), a per-owner admission block for Runs
admitted after the refusal, and the self-service and customer-admin surfaces. The
B2C and B2B lines are separate deployments, each in one region (EU, APAC or
China), so one erase never spans regions: it runs where the data is.

Amended: the package sweep writes one ledger row per package it judged (§6b),
because the container log it leaned on rotates and a DPA audit reads durable
evidence; and the Corpus Workspace creator case is implemented (§3) — a Workspace
the requester created is covered by default, delivered as a Corpus Mutation Run
rather than deleted from here. Both replace an earlier answer in this ADR: "it
writes no ledger row, and cannot", and "the receipt names the ones the requester
created, and the customer decides those".

## Context

Today's deletion surfaces are per object, not per person:

- Deleting a conversation deletes its turns and their Runs, cascading events,
  transcript entries and Artifact references, releasing blobs no surviving Run
  references and reclaiming the Runs' workspace roots.
- Memory has forget (a tombstone), merge, and Clear Profile Memory (physical).
- Connections, Agent accounts, Skills and Corpus Workspaces each have their own
  delete.

What a conversation deletion leaves behind is the point of this ADR: Profile
Memory (deliberately — it is the owner's across conversations), sealed
connection credentials and their key ring, Agent accounts and browser leases,
published Skills, telemetry in ClickStack, and backups. None of that is
reachable by the owner today, and none of it is covered by the contribution
switch (`dlightrag_privacy_settings`), which is consent, not deletion.

The B2B line adds an asymmetry: the data subject is a person inside a customer,
and a Corpus Workspace's documents belong to that customer, not to the person.
An owner erase must remove the person's data without silently deleting the
customer's knowledge base.

## Decision

**Two operations, one receipt, one clock.**

1. **`export_owner(owner_id)`** produces a manifest and one JSONL stream per
   class — conversations and turns, Runs and their events, the agent transcript,
   memory records and their journal, connections and Agent accounts *metadata*,
   Skills, Artifact and attachment inventory, Corpus Workspace membership — plus
   the bytes of blobs the owner owns, addressed by digest. It returns a signed,
   time-limited download (7 days) and a receipt. Connection credentials, Agent
   account passwords and the key ring are **not** exported: they are sealed
   third-party credentials, and handing them over in clear would break the seal
   that protects the owner. The receipt says so, so the omission is a stated
   answer rather than a silent one.
2. **`erase_owner(owner_id)`** refuses while the owner has a Run queued or
   running, then deletes in dependency order: Runs and their cascades (events,
   turns, transcript, workspace inventory, Artifact references), unreferenced
   blobs, memory records and their journal and settings, connection rows and
   their sealed envelopes, Agent accounts and their browser leases, the owner's
   Skills, session-memory progress, workspace roots, and telemetry rows whose
   `user.id` is the owner's. It writes a tombstone receipt: what was covered,
   per-class counts, when, and by which request. It is idempotent and resumable,
   and it runs in bounded batches under the same lock order the existing
   cascade uses (Runs before turns, so it cannot deadlock against retention).
3. **Corpus documents are the customer's, and the creator's own Workspace is the
   exception.** When the requester created the Corpus Workspace, the erase covers
   it; when they did not, the erase removes the person's membership and
   contributions and the receipt names the documents left in place, whose deletion
   the customer's controller decides. Creation is the only relation that lets an
   erase act on a corpus: membership is not enumerable from here, and inferring it
   would let one person's request delete a colleague's knowledge base. Coverage is
   delivered, not performed — each covered Workspace becomes one `delete_workspace`
   Corpus Mutation Run, accepted through the same application service the browser's
   Workspace delete calls, so the Workspace write fence and the mutation window
   keep a concurrent ingest from writing back what the deletion removed. The erase
   therefore never runs `DELETE` against a corpus table itself. The deletion is
   asynchronous, and the receipt says so: `corpora_deletion_runs` holds the Run
   ids, the 30-day promise is checked against their terminal states, and until
   they run the Workspace row still names its creator. Two product rules sit on
   top. On the B2C line the creator is the data subject, so coverage is the
   default and `--keep-corpora` is the explicit exception for a Workspace a human
   judged to be shared. On the B2B line the customer controls the Workspace, so
   `--line b2b` requires `--customer-approval <reference>`, which the receipt
   records beside the line and the executor.
3b. **What "revoke" means here.** This deployment's authentication is stateless
   (JWT, edge identity or one deployment token) and it keeps no per-owner session
   store, so there is no session to revoke; what the erase does revoke is the
   owner's access to third parties, by deleting the sealed Connection grants, OAuth
   flows and Agent account envelopes, and their key-ring material with them. A
   deployment that later holds per-owner tokens covers them by discovery, like any
   other table that names an owner.

4. **Telemetry is erasable now, not in seven days.** ClickHouse rows carrying
   the owner's `user.id` are deleted by the erase rather than left to the TTL;
   the seven-day TTL stays as the backstop. This needs a delete-capable
   ClickHouse credential for the maintenance path, which the collector's
   write-only user does not have.
5. **The clock is stated, then met, and the backup horizon fits inside it.** The
   operation starts when the request is accepted and finishes within 30 days:
   active stores within the operation itself, and the last backup copy that
   still holds the data destroyed within the same 30 days. One promise, one
   number — a backup retained for R days keeps a copy alive until T+R, so a
   deployment that wants a longer restore window raises the promise with it,
   rather than keeping copies past what the DPA states. Deletion is not
   "scheduled for the retention sweep".
6. **One audit trail.** A DSR receipt table records the requester, the line, the
   region, the operation, the counts, the executor and the outcome; it is the
   evidence an auditor and a customer's DPA audit clause read.
6b. **The package sweep is maintenance, and it still leaves evidence.** An
   operator's timer runs `dlightrag-owner-data sweep-expired`, which removes a
   package past the deadline its own manifest states, names every package it
   judged on stdout, and exits non-zero when one survives its own removal. It
   writes one ledger row per package it judged — `operation = 'sweep'`, outcome
   `completed` or `failed`, counts carrying the package's own request id, its
   `expires_at` and whether the removal happened — because the container log
   rotates, and a DPA audit reads durable evidence rather than a log line. A
   `--dry-run` writes nothing: a removal that was not performed was not promised.
   The ledger's operation check names `sweep` beside `export` and `erase`, and a
   ledger that predates it has that one constraint rewritten in place — its rows
   are the evidence and are never migrated.

7. **Where it is reached from.** An operator CLI and an internal admin route
   first, so both lines can honour requests on day one; a self-service surface
   for the B2C line and a customer-admin surface for the B2B line follow, with
   the receipt as the shared contract either way.

## Consequences

- **The acceptance test is the deliverable's definition**: erase a fixture owner,
  then look for the owner in every text, uuid, array and JSON column of every
  table — not in the list the erase itself uses — and assert the filesystem roots
  are gone; export the same owner and rebuild their conversations and memory from
  the package alone. A covered Corpus Workspace is the stated exception the test
  pins: until its delivered Run executes, the Workspace row still names its
  creator, and the receipt's run id is what makes that wait bounded and checkable.
- **A covered Workspace is deleted by the lane, not by the erase**: the erase
  accepts one `delete_workspace` Run per Workspace the requester created, so a
  concurrent ingest cannot write back what a delete removed, and a Workspace
  Delete that needs repair is repaired by the lane's own machinery rather than by
  re-running the erase. The delivery is submitted by whoever ran the erase, never
  by the data subject: the Run outlives the request, and the subject's id must not
  survive in it.
- **A delivery is not a deletion until the lane runs it**: the operator process
  cannot know whether the deployment has a live coordinator, so what it can state
  is the delivery — `corpora_deletion_runs`, one Run id per covered Workspace, each
  Run naming its Workspace as its access scope. Until the Run executes, the
  Workspace row still carries `created_by`; when it executes, its own
  `unregister_workspace` step removes the row and the residue with it. An
  end-to-end check of a DSR therefore reads the delivered Runs' terminal states
  before `erase_due_at`, not the erase's exit status.
- **What an erase cannot reach, and says so**: ClickHouse rows carrying the
  `user.id` attribute are deleted by the operation; log records carry no such
  attribute today and session replay carries none at all, so those age out under
  the telemetry TTL. An orphan Agent Workspace tree whose Run row is already gone
  cannot be attributed to one owner — a shard is a hash prefix owners share — so
  it is named in the receipt and swept by the reclaimer.
- **The backup horizon is the promise**: the receipt records the deadline
  (`erase_due_at`, 30 days) that the backup policy must fit inside, because a copy
  kept for R days still holds the data at request plus R.
- **A live Run is the existing residual risk**: deleting a conversation while a
  worker is mid-flight already tolerates filesystem errors for one heartbeat
  ([ADR 0019](0019-turn-accurate-forking-and-the-carry-point.md)); an owner
  erase inherits that and should stop accepting new Runs for the owner first.
- **Blobs are shared by reference**: an erase releases only what nothing else
  references, which is the existing blob rule and not a new exception.
- **Two lines, three regions, no cross-region request**: a request is served by
  the deployment that holds the data; there is nothing to coordinate.
- **The subprocessor set bounds the promise**: an erase reaches our stores and
  our processors' copies only as far as those contracts allow; the DPA has to
  say which of them delete on request and which age out.

## Open questions

1. Whether credentials should ever be exported (a rotation flow instead?).
2. Whether the B2B line self-serves DSRs from a customer admin console or routes
   them through support with the receipt attached.
3. Whether any legal-hold case would suspend an erase; today nothing does.
