# ADR 0051 — a report carries the fragment it was built from

`html-report` writes the fragment it read into the report's head, deflated and base64-encoded with
its own digest, and `html-report source report.html src.html` writes it back byte for byte. A
conversation that must change a report it published reopens that report and gets its source, rather
than a workspace that no longer exists.

## Status

Accepted and implemented in `html-report`. No migration: a report built before this has no block,
and `source` refuses it with that reason rather than inventing one.

## Context

Two planes hold a Run's material and neither of them can be edited into a newer version of itself
([ADR 0022](0022-session-owned-memory-and-run-owned-products.md), [ADR 0033](0033-resource-materialization.md)).
A Resource — an upload, a fetched page, a **published Artifact** — is digest-addressed, citable and
adoptable by a later Run of the same Session, and it is immutable, because its identity *is* its
bytes ([ADR 0013](0013-lineage-adoption-of-earlier-run-resources.md), [ADR 0023](0023-a-published-product-is-a-resource.md)).
The Agent Workspace is mutable and is not carried: a Run's tree is never read by the next one, and
the Session's `notes/` is the one exception ([composition.py](../../src/dlightrag/engine/answer/tools/composition.py)).

For a deliverable whose bytes *are* the product — a workbook, a Word file, a PNG — that is a
complete loop: `materialize` the published version, edit the copy, publish again. A generated
deliverable breaks it. The HTML report is a build output, the Skill's own command line builds it
from `report/src.html`, and the Skill said "a follow-up edits `report/src.html` and rebuilds" — a
sentence that stopped being true when the carry was deleted, because `report/` is scratch of the
Run that wrote it. The follow-up therefore held a published report it was forbidden to hand-patch
and no source to rebuild: the one deliverable the toolkit produces by compilation was the one
deliverable that could not be iterated on.

The source is small next to what it builds. The golden dashboard's fragment is 7,432 bytes against
a 1.2 MiB document, and the fragment is the compressible half — HTML with the data rows in JSON
blocks.

## Decision

**The document carries its source.** `build` writes the fragment it read into the document's head as
`<script type="application/json" id="report-source">`, holding the format version, the codec
(`zlib+base64`), the byte count, the SHA-256 of the fragment, the `--title` the report was built
with when one was given, and the deflated fragment. Deflated because the report has a size limit and
the fragment holds the rows; base64 because the fragment holds the end tag of a script and every
other encoding has to be escaped. The block sits in the head, so the body stays exactly the author's
fragment in one wrapper, and it is inert: the runtime reads `data-*` and `chart-*` blocks only.

**`source` writes it back, and refuses what it cannot trust.** `html-report source report.html
src.html` — `-` for standard output — verifies the block's format and codec, decodes it, checks the
digest, and refuses a document with no block, a block this build cannot read, or bytes that do not
match the digest the block carries, naming which. A round trip is therefore exact: rebuilding the
extracted fragment with the same `--title` reproduces the document byte for byte, and `source` says
on stderr which title the report was built with, so the one flag the fragment cannot carry is not
silently lost.

**The loop is stated where the author is.** The `interactive-html` Skill's Build section now says
the follow-up reopens the published report by the `resource_id` in its `artifact:` link,
materializes it, and runs `source`; the false sentence it replaces is gone.

## Consequences

- A report is a self-describing artifact: any later Run, any lane, any machine that holds the report
  can rebuild it. Iteration no longer depends on the Session's memory plane surviving, on `notes/`
  having room (256 KiB, and a data-heavy fragment would not fit), or on the Run that wrote it.
- The document grows by the block: 3,095 bytes for the golden dashboard's 7,432-byte fragment —
  2,155 deflated, 2,876 as base64 — which is 0.25% of the 1.2 MiB document and well inside the
  20 MiB artifact limit that already bounds what a report may hold.
- The embedded block is not evidence and not a second statement of the report: it is the input, and
  nothing reads it except `source`. It is invisible to the reader, and the body-composition test
  keeps it that way.
- A document edited after it was built is refused by `source` rather than silently rebuilt from
  stale input. That is the digest doing its job, and it is the one case where the remedy is to take
  the fragment from the published bytes instead.
