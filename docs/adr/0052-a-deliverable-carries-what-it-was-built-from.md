# ADR 0052 — a deliverable carries what it was built from

`attach_artifact` takes `sources`: the Artifact-root files a deliverable was produced from. They
are validated, published with the root as its dependency closure, carried on the product's Resource
row, and restored with it when a later Run reopens the published version. A product a program
produced no longer reaches the next turn without the program.

## Status

Accepted and implemented. Two additions and no migration of behaviour: an optional `sources` list
on the tool, and one `TEXT[]` column on `dlightrag_answer_artifact_attachments` (ADR 0022's
development-stage rule applies — an older database is reset, not migrated — but the column is added
in place anyway, so a running deployment upgrades on start).

## Context

Two planes hold a Run's material and neither can be edited into a newer version of itself
([ADR 0022](0022-session-owned-memory-and-run-owned-products.md)). A Resource is digest-addressed,
citable, adoptable by a later Run of the same Session and immutable
([ADR 0013](0013-lineage-adoption-of-earlier-run-resources.md), [ADR 0023](0023-a-published-product-is-a-resource.md));
the Agent Workspace is mutable and is not carried, with the Session's `notes/` as the one exception.
A later turn therefore reaches an earlier deliverable by `materialize`-ing it and attaching it again,
which is a complete loop for a deliverable whose bytes *are* the product — a workbook, a Word file,
a PNG. Editing those bytes is lossy at the edges but works.

A *generated* deliverable is different. The report was the first case and
[ADR 0051](0051-a-report-carries-the-fragment-it-was-built-from.md) answered it by making the
product carry its own source inside itself, because `html-report` is a tool the product owns and can
be made to write one. `office-documents` has no such tool: the agent writes an ad-hoc python-docx,
openpyxl or python-pptx program, and the program *is* the source. It was told to write that program
under `tmp/`, which is scratch of the Run that wrote it, so the next turn held the deliverable, was
forbidden from hand-patching it, and had nothing that could produce it again. The skill shipped the
loop's first half and no second half.

The product already had the carrier for this and had never used it for this. A published root and
its dependency closure travel together: dependencies are validated like roots, stored as Blobs,
registered as the Session's Resources, adoptable later. What a dependency could not be was
*unlinked*: the closure is discovered by walking `href`/`src`/`artifact:` references out of a
Markdown or HTML document, and a `.docx` or a `.xlsx` has no document to walk. The declaration
existed for the formats that can link and not for the ones that cannot — which is exactly backwards,
because a Markdown deliverable is the one that can carry its own source as text.

## Decision

**A root may declare the files it was built from.** `attach_artifact(path=…, sources=[…])` names
Artifact-root-relative paths: the generator, its inputs, anything the Run produced that the next Run
would need to produce the deliverable again. `prepare_artifact_attachment` validates each one the
way it validates a root — present, a real file under the Artifact root, inside the size limit, and
not one of a Skill's own documents ([ADR 0046](0046-a-skills-documents-are-its-authors-to-publish.md))
— and the tool says what the deliverable ships with. A source is published like any other member of
the closure, so the closure's own budget bounds it: a kit that would exceed the file count or the
total bytes is refused where its author is, not left behind at settlement, which is the one place a
product could otherwise ship without the program that produces it.

**They enter the closure as if the root had linked them.** `validate_publication` seeds the root's
children with its declared sources, so from there the walk, `_validate_file`, the admission budgets,
the Blob write, the Resource registration and the digest protections are the same code path a linked
dependency already takes. They are published, and they are never placed or listed: a source is not a
deliverable, nothing in the document points at it, and the reader receives the root the tool
attached rather than a script no sentence mentions. A file that the delivery *also* references is a
dependency like any other and is listed as one.

**The relation is recorded on the product, not in the transcript.** The plan carries
`sources_by_root`; the settlement writes `[{path, resource_id}, …]` into the published Artifact's
Resource capabilities; the attachment row carries the declared paths beside the root's own facts. A
later Run that reopens the product adopts its provenance with its bytes: the tool restores each
source into `artifacts/` under the path it was published at, leaves a path the workspace already
holds alone because that copy is the current Run's own work, and re-declares what it restored so the
next publication carries the kit forward again. A source retention has dropped is skipped, and the
product still opens.

## Consequences

- The loop closes for every generated deliverable, not only for the report: reopen the published
  file, get back what produced it, change it, run it again. Nothing about it depends on the
  Session's memory plane, on `notes/` having room, or on the Run that wrote the source still being
  reachable — the kit is as durable as the product it belongs to.
- What a user receives does not change. Sources are dependencies, so they are carried and never
  carded, listed, or counted against the `MAX_ARTIFACT_HANDLES` a summary spends on the products a
  conversation actually delivered.
- One more thing the model must decide at attach time, and the decision is co-located with the act
  it describes: the tool call that publishes a deliverable is the one that names what made it. The
  failure mode this replaces — a skill sentence asking the model to remember, one turn earlier, to
  put its script somewhere else — is the one this project keeps finding and removing.
- `office-documents` now writes its generator into `artifacts/` beside the file, not into `tmp/`,
  which is a change to what that Skill tells a model and the reason the column exists.
- The refusal at attach time is the ordinary path; the residual is a source the closure refuses
  later, at settlement, if the workspace changed after the tool returned. There the product
  publishes without it and the issue is reported beside the attachment, which is honest but is not
  the kit that was declared — an attach-time check cannot promise what a file will look like at the
  end of the Run.
