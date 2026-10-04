# Resource Materialization

A Run's Resources and its Agent Workspace are two planes, and nothing carries bytes
from the first to the second: `bash` cannot open a CSV the user uploaded or the
Agent downloaded. `materialize(resource_id, path)` copies a Resource's admitted
original bytes into the Agent Workspace as an ordinary file, accounted exactly like
`write`. The Resource stays immutable and citable; the copy is work. The tool exists
only with execution `trust`, every Child inherits it, and it is not done until a
real agent task has used it.

## Status

Accepted; implementation in progress. It lands as slice 3 of the sequence
[ADR 0032](0032-the-agent-browser.md) records, after the downloads it makes useful,
and passes the same review on four axes — Standards, Spec, correctness and security,
and performance — before slice 4 begins.

It amends nothing. It does not reopen
[ADR 0023](0023-a-published-product-is-a-resource.md)'s refusal to carry a product
into a workspace by itself: a copy happens only when the Agent asks for one, and
that decision's axis holds, with `resource_id` addressing immutable bytes and
`path` addressing mutable work.
[ADR 0024](0024-the-agent-sees-only-its-workspace.md)'s process view and
[ADR 0025](0025-a-child-inherits-capability-not-authority.md)'s capability line
apply to the new tool unchanged.

## Context

Two planes hold a Run's material. The Resource plane holds what the Run admitted —
uploads, fetched Web bodies, adopted Resources, and with ADR 0032 captures and
downloads — as owner-scoped, digest-addressed Blobs
([ADR 0010](0010-separate-run-blob-plane-from-operational-state.md)) behind one read
surface ([ADR 0016](0016-one-run-resource-read-surface.md)). Its bytes are
immutable, a Resource can be Evidence, and a later turn can adopt it
([ADR 0013](0013-lineage-adoption-of-earlier-run-resources.md)). The Agent Workspace
is the Run's mutable working copy: the only filesystem the Agent's processes can see
(ADR 0024), and the one publication reads
([ADR 0004](0004-structured-artifact-attachment-authority.md)).

Nothing crosses from the first plane to the second:

- `read` and `view` over a Resource (`engine/answer/tools/resources.py`) return
  bounded text windows and budgeted pixels to the model, never a file.
- `write` (`engine/agent/tools/files.py`) takes the whole file as UTF-8 `content`.
  The only way to move a Resource's bytes with it is for the model to re-type the
  Resource's text, one bounded window at a time, and a binary file cannot be moved
  at all.
- `bash` runs confined to the Agent Workspace, with no application credential in its
  environment (ADR 0024), so it cannot reach the Blob plane.
- Binding a Run's workspace lays down only the Session's notes
  ([ADR 0022](0022-session-owned-memory-and-run-owned-products.md)). The shell is
  told that earlier Artifacts and knowledge-base documents are never files in it,
  and an upload is a Resource, never a file.

So `bash` cannot process an uploaded or downloaded CSV. Re-typing its text would not
help: what `read` returns for a CSV is MarkItDown's view of it, which collapses line
breaks inside a cell and pads short rows
([resource reading](../resource-reading.md#conversion-routes)), not the file.
ADR 0032 widens the gap: a CSV the Agent downloads in order to compute with becomes
a Resource the computing tool cannot open.

## Decision

**One tool copies admitted bytes into the workspace.** `materialize(resource_id,
path)` writes the bytes a Resource admitted to a workspace path, byte for byte: an
upload's bytes, a Web snapshot's bytes, a browser download's bytes, a capture's
HTML. It never converts, renders, or fetches. A Web Resource that holds no snapshot
yet, such as a search-result link nothing has read, has nothing to copy, and the
refusal says to read it first.

**The planes stay separate.** The Resource remains the immutable, citable original;
the copy is mutable work with no provenance of its own. A citation names the
Resource, never the file. What the Agent computes from the copy is the Agent's work
over a cited source, as any computation is. A file built from the copy and
published is a new product with its own provenance under ADR 0004 and ADR 0023, not
a re-publication of the Resource.

**It is accounted exactly like `write`.** The destination resolves as a `write` path
does: rooted in the workspace, never through a symbolic link. The workspace
integrity latch refuses it as it refuses `write`, and it holds its path through the
same access scheduler. `ExecutionEnvironment.write_bytes` admits the bytes against
the workspace quota and refuses them with the quota error, and the settlement
carries the Workspace Inventory fact `write` emits: path, size, mode, and digest.
It replaces a file already at the path, as `write` does, and under `notes/` it
declares a Session Note by ADR 0022's path rule, inside that plane's own budget. It
is not read-only, so each call runs alone
([ADR 0029](0029-read-only-calls-run-at-once-and-settle-in-source-order.md)), and
its replay policy is `never`, as `write`'s is.

**A handle resolves as it does for `read` and `view`.** The same Resource Handles
and aliases resolve. A lazily held upload loads as it does on its first read and
spends what that read spends. An earlier turn's Resource is adopted first under
ADR 0013, which needs only its bytes, as `view` of a PDF does.

**It exists only with execution `trust`.** With `disabled` there is no Agent
Workspace, so the tool is not composed.

**Every Child inherits it.** Copying bytes into the Run's shared working copy is
capability, not authority (ADR 0025): the tool is not in the forbidden table, so the
computed default grants it. A Child copies into its own
`tmp/children/<child session id>/` scratch by the same convention as any other file
it writes.

**It is proven on a real agent task.** Tests at the tool seam do not finish it. It
counts as landed when a live Run, with a real model on a development deployment and
no doubles, downloads a CSV through the Agent Browser, materializes it, computes
with pandas in `bash`, and publishes a chart with `attach_artifact`.

## Considered options

- **Expose the Resource plane as read-only files.** Rejected. ADR 0024's allow-list
  is code, the Blob plane is chunked PostgreSQL `BYTEA` rather than a filesystem
  (ADR 0010), and a standing view would make every Resource a file whether the Agent
  needs it or not.
- **Copy every upload into the workspace when the Run binds.** Rejected. It is the
  automatic carry ADR 0023 refused for products, it spends quota and time on bytes
  most Runs never compute with, and uploads load lazily by design.
- **Let `write` take bytes, base64-encoded.** Rejected. The bytes would pass through
  model context, which [security](../security.md#answer-resources-and-execution)
  excludes ("Full bytes never enter model context"), cost output tokens in
  proportion to their size, and fail on anything large.
- **Give `bash` a way to fetch Resource bytes.** Rejected. An application credential
  or a local endpoint in the shell's environment is the authenticated call back
  into the application that ADR 0024 keeps out of the Agent's reach.
- **Make `read` write the file.** Rejected. `read` is read-only and replayable
  (ADR 0029); a workspace write would make it neither.
- **Download the source again with `curl`.** Rejected. It serves only public URLs,
  never an upload or a session-bound download, and it fetches content that may
  differ from the snapshot the answer cites.
- **Cite the workspace copy.** Rejected. A mutable file has no durable source
  identity; the Resource already has one.

## Consequences

Landing order, the sequence shared with ADR 0032 and ADR 0034:

1. The substrate and the Rendered Read (ADR 0032).
2. The interactive tool, capture, downloads, and Children (ADR 0032).
3. **`materialize`** (this decision): the tool, composed beside `write`; its
   description, which says to copy a Resource when a process needs its original
   bytes; the workspace fact the shell and `ls` state, which names the way in;
   tests at the product seam — the copy's digest equals the Resource's, the quota
   and latch refusals, the inventory fact, and the Child default; and the live proof
   above.
4. Accounts and the mailbox backend
   ([ADR 0034](0034-agent-accounts-and-the-agent-mailbox.md)).
5. The frontend co-design (ADR 0034).

Live documents to revise when slice 3 lands: [domain language](../domain-language.md)
(Resource Materialization, and the Agent Workspace term),
[resource reading](../resource-reading.md) (copying a Resource into the workspace),
[retrieval and answer](../retrieval-answer.md) (the Research tool list),
[security](../security.md) (the rooted file tools of the `trust` mode), and
[architecture](../architecture.md) (Answer Resources).

Residual risks, recorded rather than solved:

- A copy can be as large as the per-item admission bound (`max_attachment_bytes`,
  100 MiB by default). It counts against the workspace quota and, at publication,
  against the working-set limit (`publication.workspace_max_bytes`), so a Run that
  copies large inputs and then publishes can meet that limit at the terminal
  boundary.
- Nothing links a copy back to its Resource: a figure computed from an edited copy
  still cites the original.
- Parallel Children copying to one path are last-writer-wins, as ADR 0025 records
  for any write.
- An adoption made in order to copy an earlier turn's Resource spends this Run's
  attachment allowance, as one made to read it does.
- With `materialize`, an Agent steered by injected content can move an upload's whole
  bytes out in one step, through Bash's network access or the Agent Browser's `upload`,
  where before it had only `read`'s text windows to retype. This amplifies capabilities
  already accepted (Bash keeps network access,
  [security](../security.md#answer-resources-and-execution)), and the mitigations are
  the same.
- The word is overloaded. `ResourceRegistry.materialize` loads a Resource's bytes
  into the registry, and ADR 0013 calls adoption materialization; the glossary term
  Resource Materialization names only the copy into the Agent Workspace.
