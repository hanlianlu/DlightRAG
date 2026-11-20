# A Skill name belongs to the tier that owns it

A Skill name is served by exactly one Skill. Tiers are authority, not override: a built-in keeps
its name, a global Skill keeps any name a built-in does not hold, and an owner's Skill keeps what
is left. A Skill whose name a stronger tier already serves is a conflict — it is not served, it is
logged, and `publish_skill` refuses to create one in the first place.

## Status

Accepted and implemented. It revises what `disabled_builtin_skills` was documented to allow
("Global and owner skills with the same names remain discoverable") and the precedence sentence in
the Agent Skill entry of [domain-language](../domain-language.md). No migration: discovery reads
the roots it already resolves, so an owner's Skill that collides today is simply not served from
the next Run.

## Context

Precedence was built-in, then global, then owner, with each tier replacing the one below it by
name, so an owner who published `charts` overrode the product's `charts` for themselves. The
intent was helpfulness: an owner knows what they want.

It is the wrong shape for a Skill library, for one reason that is not about authority at all. **A
Skill's name is how the agent names it.** The catalog is the list a model reads, `load_skill(name)`
is the call it makes, and the document it gets back is the whole basis on which it acts. When two
different documents can answer to one name — depending on what a *different* tier deployed, and
depending on *when* it was deployed — the model's picture of its own tools is not stable:

- The same conversation can mean different things across turns: a catalog the Run was started with
  is pinned into its prompt prefix ([ADR 0015](0015-prompt-prefix-stability-and-cache-anchored-accounting.md))
  while `load_skill` resolves names as they are when it is called, so a publication can change what
  a name resolves to underneath a Run that already read the old one.
- An agent that has read `charts` once may be handed a different `charts` later, and cannot tell.
- A user who overrides a product Skill has no way to know which document they replaced, and the
  operator who deployed the global version has no way to know it was ignored.
- Nothing about the name carries which tier it came from, so no instruction can disambiguate it.

The alternative — namespacing a shadowed Skill, or nesting tiers under a prefix — was rejected:
it makes every call site carry tier knowledge that only publication needs, and it changes the
names of Skills that were never in conflict.

## Decision

**Discovery serves one Skill per name, and the stronger tier keeps it.** `SkillCatalog.discover`
merges packaged built-ins, then global Skills, then the owner's enabled Skills, and a Skill whose
name is already served is skipped with a warning naming both tiers. A Skill that a deployment
hides with `disabled_builtin_skills` is not served, so its name is free and the next tier may serve
it: that is the operator saying they do not want the product's version, which is explicit rather
than implicit.

**Publication refuses a name another tier serves.** `publish_skill` takes the names the
built-in and global tiers serve and rejects a publication that would take one, saying the name
belongs to the tier that owns it and to publish under a name of their own. This is what makes the
rule explicit rather than a silent drop: the user learns it while they are naming the Skill, not
when they wonder why it never loads.

**Protection follows authorship, not service.** `protected_digests()` is built from every
discovered built-in and global Skill, including one that is hidden or that lost a name, so a
document is protected because of whose it is
([ADR 0046](0046-a-skills-documents-are-its-authors-to-publish.md)) and not because it happens to
be the version being served.

## Not building

- **Namespaced tiers at the call site.** A model should not have to know which tier a Skill came
  from; the name is the interface.
- **Refusing the deployment.** A global root is a filesystem an operator puts files in, so a
  collision there cannot be refused at write time; it is resolved deterministically and logged.
- **Renaming a conflicting Skill automatically.** A Skill's name is its author's choice, and the
  frontmatter requires the name to equal the directory: renaming one would invent a name nobody
  wrote.

## Considered options

**Keep the override, and mark the catalog line with its tier.** Rejected: it puts tier vocabulary
into every prompt that lists Skills, to solve a problem that only exists when two Skills share a
name — and it still leaves `load_skill(name)` ambiguous.

**Let the owner's Skill win, and refuse only built-in/global collisions among themselves.**
Rejected: the product's own Skills are the ones users read documentation about, and a name that
means the product's Skill in one account and something else in another is the ambiguity this ADR
exists to remove.

**Refuse the collision at publish time only.** Rejected as insufficient on its own: a global root
is provisioned outside the product, so discovery still has to resolve a collision it did not
create.

## Consequences

An owner keeps a directory of Skills whose names are theirs, and the names the product and the
operator serve are stable for every user of a deployment. `publish_skill` tells a user who reaches
for a taken name, and an operator sees one warning line naming both tiers when their global file
loses to a built-in — the two ways a collision can arise.

The cost is that an owner cannot replace a built-in Skill for themselves. Nobody could act on the
replacement's existence anyway: the catalog line, the documentation and the Skill's own
`description` all name the same Skill, and the product's version is the one the agent's instructions
elsewhere refer to.
