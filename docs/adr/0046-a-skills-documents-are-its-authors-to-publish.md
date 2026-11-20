# A Skill's documents are its author's to publish

A Run publishes what it produced, not the material it reasoned with. A Skill's own files — its
`SKILL.md`, its references, its scripts — are refused as Artifacts for every tier whose author is
not the Run's own user: a built-in is the product's, a global Skill the operator's. An owner's own
Skill is not protected at all, because withholding a user's own document from that user would be a
defect and not a protection.

Refusal is by content digest, so a copy under another name is still refused, and it holds for an
attached root and for every dependency a root links to. A Skill may declare a file `publishable`
in its frontmatter, which is how a document can be meant for the user while the rest stay the
author's.

## Status

Accepted and implemented. No migration: the rule reads the Skill roots the Run already resolves,
holds no state, and applies to Skills that already exist. It is enforced where a file can become
publishable and nowhere else, so nothing about a Run's own working copy changes: a Skill's script
still lives in `tmp/`, still runs, and its result is still delivered.

## Context

DlightRAG ships a Skill library — `charts`, `office-documents`, `interactive-html`
and the rest — and operators provision a global root beside it. Those
documents are the product's and the operator's work. A Run reads them freely: `load_skill` hands a
model a Skill's text to reason with.

Publication is the one boundary where a workspace file becomes something the user receives
([ADR 0004](0004-structured-artifact-attachment-authority.md)): an attached root, the files it
links to, and a download surface. Nothing there distinguished "bytes this Run computed" from
"bytes this Run was given", so an agent that wrote a Skill's own script into `artifacts/` could
hand the user the Skill itself.

Three properties of the existing design decide the shape of the answer:

- **A Skill's documents reach the workspace only through the agent.** `load_skill` is a read
  ([ADR 0022](0022-session-owned-memory-and-run-owned-products.md)); the packaged built-ins are not
  declared to the Execution Environment at all, so an agent's own processes cannot open them, while
  the global root and the Run owner's shard are declared read-only
  ([ADR 0024](0024-the-agent-sees-only-its-workspace.md)). Whatever a Run can publish of a Skill's,
  it obtained by being handed the text.
- **A file's identity in publication is its bytes.** `_validate_file` digests them, and the
  terminal boundary re-checks that digest against the attachment, so a name proves nothing and a
  path proves nothing.
- **Which Skill serves a name is precedence, not authorship.** A name is served by one Skill and a
  stronger tier keeps it ([ADR 0047](0047-a-skill-name-belongs-to-the-tier-that-owns-it.md)), so
  the served catalog is the product's and the operator's Skills plus whatever names they left. But
  a Skill that is hidden, or that lost a name, is still its author's document.

## Decision

**One set, computed from the Skill roots, and one check.** `SkillCatalog.protected_digests()`
returns the SHA-256 of every file of every Skill in a protected tier, minus the files that Skill
declares `publishable`. `_validate_file` refuses a file whose digest is in that set, with the issue
kind `protected_content`, which puts the check at the single place bytes are admitted to
publication and therefore covers a root and a dependency alike.

**A container is opened.** A document inside a delivery archive is the same document, and the
delivery archive is this rule's motivating case, so the members of a zip — which an Office file
also is — and of a tar, compressed or not, are each digested and checked, to a bounded depth. The
container is recognised by its own bytes as well as by its name, because a Run names its own
deliverables. A wrapper around a single file (gzip, bzip2, xz) is unwrapped and checked as that
file.

**The protected set is every discovered protected-tier Skill, not the served catalog.** A
deployment may hide a built-in with `disabled_builtin_skills`, and a global file may lose a name to
a built-in, and in both cases the document is still its author's. Protection is about authorship,
so it cannot be read off the list of Skills a Run is currently being served.

**An owner's Skill is never in the set.** Their documents are theirs; a Run may publish from them
freely. The declaration is parsed for every Skill and only means something where protection
applies.

**A Skill opts a file in, never out.** The frontmatter key `publishable` lists Skill-relative
paths, each checked to stay inside the Skill; the default is that nothing is publishable. A Skill's
author is the only one who can widen it, and a path that leaves the Skill is refused where the
declaration is read rather than at the boundary.

**Refusal is legible and early.** `attach_artifact` refuses before the publication intent is
recorded, naming the file and saying to deliver what the Run produced; the terminal boundary
refuses it again, because a dependency never passes through the tool.

**A document the framework cannot read is skipped, with a log; a Skill it cannot parse is
protected hardest.** `load_skill` reads through the same permissions, so a file this process
cannot read is one no Run was handed, and a permissions accident must not fail a Run. A symlink is
skipped for the same reason `load_skill` refuses one. A `SKILL.md` whose frontmatter is invalid
means the Skill is not served at all — and it stays in the protected set with nothing declared
`publishable`, because a typo must not be the thing that un-protects a Skill.

**Protection follows what `load_skill` can reach.** Every file the author put in the Skill
directory is protected, a dot-named directory included, because `read` reaches it and a rule
covering fewer bytes than the framework can hand a model covers nothing. What the framework writes
there itself is not a document: the marker that turns a Skill off, and the compiled caches a runtime
leaves behind, which `read` cannot read as a document anyway. A zero-byte file is left out too — it
is not a document, and its digest would refuse every empty Artifact.

## The limit

The check reads bytes: a document that has been *transformed* — re-encoded, base64-encoded,
excerpted, reformatted, or put in a container nothing here can open — is not detected. That limit is
stated rather than engineered around, because the case that must not happen is the ordinary one, an
agent writing out what it read and shipping it, and a content comparison would have to guess at what
counts as the same document. What a Run should and should not deliver stays the Skill's own
instruction to the model.

## Not building

- **Path-based refusal.** It would pass a rename and miss a copy.
- **Redaction of a Skill's text in the transcript.** That is a different question — what a reader
  of the trace sees — and it belongs to the projection surfaces, not here. This ADR is only about
  what a Run may hand over as a deliverable.
- **A per-deployment publishable list.** The Skill that owns the document is the one that knows
  whether the document is meant for a user, so the declaration lives with the document.
- **Deciding which Skill serves a name.** That is
  [ADR 0047](0047-a-skill-name-belongs-to-the-tier-that-owns-it.md); this ADR only decides whose
  document a file is.
- **Blocking the write.** A Skill's script in `tmp/` is how a Run uses the Skill; only the handing
  over is refused.

## Considered options

**Protect every tier, including the owner's.** Rejected: it makes a user's own Skill unusable as a
source of their own deliverables, and it protects the wrong party. A user who writes a Skill with a
template in it must be able to be handed that template.

**Protect by Skill name or by path prefix.** Rejected: publication sees a workspace file whose name
the agent chooses, and the digest is the only thing about it that is the same as the document.

**Refuse at the terminal boundary only.** Rejected: the agent would learn the rule from a failed
publication, after it had spent the turn assembling a deliverable it cannot ship, and the correction
pass would be the first place it heard of it. `attach_artifact` is where the intent is recorded, so
that is where the refusal is cheapest and clearest.

**Classify the served catalog.** Rejected: it fails open exactly when it matters, because the
product's most interesting Skills are the ones an owner is most likely to shadow.

## Consequences

A Run cannot hand a user a Skill's document, and a Skill that means to have one handed over says so
in its own frontmatter.

The check costs one walk of the Skill roots each time a Run decides publication, and one digest per
document — 23 files and about 150 KB for the built-ins as they stand, read again by the correction
pass when one runs. It is skipped entirely when a Run serves no Skills, when a Child is composed (a
Child never attaches) and when the Run has no Artifacts root to publish into. Opening a container
costs one more pass over that container's members, bounded by the same file-size limit the Artifact
itself passed.
