# A published product is a Resource, and memory is a path

Reading something again is one surface with two address forms. `read(path=…)` names a
file in the calling Run's working copy, which is where mutable memory lives;
`read(resource_id=…)` names immutable bytes, which is what everything else is. This
decision makes a Published Artifact the second kind, so a conversation can keep working
on one deliverable without products becoming editable files.

## Status

Accepted and implemented.

It extends [ADR 0013](0013-lineage-adoption-of-earlier-run-resources.md)'s adoption to
published products and records the model that keeps the extension from becoming its own
subsystem.

## Context

DlightRAG already had a re-readable vocabulary: an Evidence citation handle, a
committed Spill, and a fetched Web body are all digest-addressed bytes a later turn can
name, and the Context Projection renders them into the summary's `durable_handles` so
the handle survives the prefix that taught it. Adoption ([ADR 0013](0013-lineage-adoption-of-earlier-run-resources.md))
lets a later Run of the same Agent Session read an earlier Run's bytes, gated by owner
and Session stamp.

Published Artifacts were the one durable thing outside it. Their bytes were
content-addressed and their address already existed — publication mints
`artifact-<sha256(path)[:20]>`, and both the Answer's own references and the browser
read surface use it — but no Resource row recorded them, the adoptable declaration did
not include them, and the model was never given the handle: the Tool receipt named the
workspace path, which belongs to the Run that wrote it. So a second turn of the same
conversation could not open the deliverable the first turn had just published; it could
only produce a new one. Note also that nothing here needed a *new* address space: the
missing pieces were a registration, one declaration entry, and naming the handle.

The temptation this decision refuses is the obvious one: carry the Artifact's file into
the continuation's workspace, the way memory is materialized. That would make the
product a mutable copy of itself, whose provenance is the copy rather than the answer
that produced it, and it would put deliverables under last-settled-wins — the property
memory is allowed to have and a user-facing product is not.

## Decision

**A Published Artifact is a Resource.** Where its bytes land — the fenced publication
transaction, beside the blob and the Run's Artifact row — it also records one Resource
row of kind `published_artifact`, stamped with its Agent Session, carrying its blob
digest, presentation, and Artifact path. A publication with no Session stamp registers
no Resource: it is a Run-owned product that no later turn may adopt, which is exactly
what "no conversation owns it" means.

**Adoption is declared once.** The set of adoptable things is one engine-owned
declaration of (capability, resource kind) pairs, and both the adapter's lineage read
and the loader's gate derive from it. Adding a re-readable kind is therefore one line
where the kind is written, not a matching pair of edits in two layers — which is how
this extension cost one entry.

**A handle is whatever its minter declares it to be.** The handle families are
declared once, where handles are modeled: the registry mints `res-…` for prepared,
fetched, evidence-backed, and spilled bytes, publication mints `artifact-…` for a
product, and the alias binder that makes an adopted handle readable accepts every
declared family. Matching on one family's literal is what made a taught product handle
unreadable after adoption — the alias was bound to nothing — so the declaration is the
only place a family is named.

**The handle is the address publication already mints.** The deterministic Artifact
address is what the Tool returns in its receipt, what the answer references, what the
browser serves, and what a later turn names — so the model holds a usable handle as
soon as it attaches a deliverable, without asking whether the publication row exists
yet. The summary's handle list keeps one composer with a reserved share per
non-Evidence class (spills, then products, then Evidence), because the omitted handle
is the expensive failure and a retrieval-heavy Run must not crowd either class out.

**Memory stays the other leg, and the axis is mutability.** `path=` addresses mutable
memory: Session-owned bytes materialized into a Run's working copy and promoted at Tool
settlement, where last-settled-wins is the right semantics. `resource_id=` addresses
immutable bytes: adopted under the consuming Run's fence, where an adopted version is
the consuming Run's own copy and no writer can rewrite what a user already downloaded.
Ownership then needs no new rule — the Session decides reachability, the Run decides
authorship — and notes, Evidence, spills, fetched bodies, and products are all reached
through the one surface.

## Consequences

A conversation can iterate on one deliverable: read the published version by handle,
edit it in the working copy, attach again, and publish a new version, each keeping its
authoring Run, digest, Evidence, and place in the conversation. A fork shares its
Session, so it inherits that reachability; another conversation — or another owner —
cannot reach the bytes at all.

Retention still bounds the original: the Resource row cascades with the authoring Run,
so the handle stops resolving once that Run is pruned, and what survives is the copy a
later Run adopted before then. That is the same best-effort shape as every other
handle, and it is the reason the handle is rendered into the summary rather than left
in a receipt that compaction will cover.

The taught call depends on the product's type, and one function decides it: a product
whose type has no conversion route is read by decoding the adopted bytes, and one whose
type routes to a converter is reached by `view`, because the decision above refuses to
convert retrospectively. The receipt and the summary both render that function, so what
the model is told is a call that works rather than a call that refuses.

A read of an adopted resource demands the earlier Run's own conversion view only when
reading it here would convert it (a suffix or MIME with a conversion route). A published
Markdown report or a fetched text page has no route, so it is read by decoding the
adopted bytes; demanding a view for those would refuse the very call this decision
teaches, which is what the first implementation did.

Residual risks, recorded rather than solved: the receipt names a handle before the
publication exists, so a Run that attaches a deliverable and then fails or is cancelled
leaves a handle that resolves to nothing — which is honest but worth knowing; a product
published with no Session stamp (a caller that names no conversation) is registered as
no Resource and stays unadoptable; and adoption widens reach only for bytes, not for
the presentation, so a later turn can read a product's content and re-publish it but
cannot re-attach the earlier presentation without attaching again.

## Considered options

- **Carry the Artifact file into the continuation's workspace.** Rejected: it makes a
  product a mutable copy with copy provenance, and puts deliverables under last-settled-wins.
- **A separate Artifact read tool or endpoint for the agent.** Rejected: a second
  address space for something the existing read surface already addresses, and a second
  thing to keep in the summary's handle list.
- **A field on the Artifact row alone, read by a dedicated path.** Rejected: that is the
  bespoke wiring this decision removes; the product's bytes are already in the blob
  plane, so the only missing statement was that they are a Resource.
- **Widen adoption to every resource kind.** Deferred rather than rejected: Evidence
  handles already travel the citation path, and widening reach without a reader that
  needs it would make the adoptable set an accident of what happens to be registered.
