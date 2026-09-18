# The Agent's processes see only the Agent Workspace

A Run's Agent is given a shell. That shell is a child of the answering process,
so today it inherits everything that process can see: the corpus ingest tree, the
application's own files, and its credentials. This decision confines it to the
Agent Workspace, and removes the mode that pretended a stronger isolation the
distribution never shipped.

## Status

Proposed. Supersedes the `sandbox` clause of the Execution Environment term in
[domain language](../domain-language.md#current-execution-and-workspace-concepts)
and the mode list in [Configuration](../configuration.md); it does not touch
[ADR 0020](0020-uniform-environment-fast-inert-workspace.md)'s uniformity
decision, which this builds on.

The shape is borrowed from [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness),
whose capability seams separate a swappable seam from the policy attached to it
and whose `fs/*` capability events attach that policy "without importing the
loop"; its browser-use and computer-use capabilities arrive as provider plugins
rather than as kernel features, and its [safety notice](https://github.com/deepseek-ai/deepseek-harness/blob/main/SAFETY.md)
reaches the same conclusion this decision does about what a sandbox in the
application can and cannot promise. The vocabulary here stays DlightRAG's own.

## Context

An Agent Workspace is already deliberately outside the corpus plane: the
composition root refuses an `agent.workspace_root` that equals, contains, or is
contained by `working_dir`. That rule governs *where files are written*. It says
nothing about *what a process may read*, and the Agent's shell is an ordinary
process in the answering container, which mounts the corpus tree and has the
application's environment.

A live Run demonstrated the difference. Asked for page counts, it listed
`/app/dlightrag_storage/inputs/personel/`, walked the parsed sidecars, ran
`pdfinfo`, and `pip install`ed a PDF library into its workspace. Another Run
grepped the parse output to recover page labels — cheaper than the retrieval
path it was meant to use. Nothing here was a model misbehaving: the shell was
offered a filesystem and used it.

That matters for two reasons, and only one of them is about isolation.
Retrieval is the product's path to the corpus: it is ranked, budgeted, cited,
authorised per owner, and it is the only path that still exists when a corpus
lives somewhere other than the answering host. An Agent that can open the source
files is not using a worse path, it is using a path that will not be there at
scale. Second, the corpus ingest tree is not only bytes: it is the plane the
application itself parses, stages, and serves, and a shell that can read it can
also read whatever the application leaves there.

## Decision

**Two execution modes.** `disabled` exposes no Agent Workspace and no path or
Bash tools. `trust` runs the Agent's processes in the host user's authority,
confined to the Agent Workspace. There is no `sandbox` mode: per-Run isolation
stronger than the host user's own kernel is a property of the environment the
application is deployed in, not of the application, and keeping a seam whose
only implementation lives outside the distribution invited the belief that the
distribution ships one. A deployment that needs that isolation runs the
application inside it.

**`trust` means the deployment is trusted, not that the Agent is unconfined.**
The confinement is part of what an enabled execution environment *is*, not a
separate switch: a Run with an Agent Workspace gets a process view rooted in it.
There is no configuration value that widens what the Agent's children may see.

**The pipe is the seam, and the policy is attached to it.** Every Agent process
spawns through one call (`ExecutionEnvironment.run`), already rooted at one
admitted workspace. The local environment applies a kernel-enforced allow-list
to each child: the Agent Workspace with full rights, the runtime read-only
(`/usr`, `/lib`, `/lib64`, `/bin`, `/sbin`, `/etc`, `/proc`, `/dev` without its
create rights, and the interpreter's own prefix), and nothing else — no corpus
tree, no deployment configuration, no project tree, no `/tmp`, no other Run's
workspace. The interpreter prefix is granted because the toolchain lives in it: a
Run that cannot execute Python cannot do the work it was given, and the installed
package sits under the same prefix. What that costs is recorded below. The one mechanism this distribution relies on
is Landlock, which is unprivileged, has no false positives, and cannot be
bypassed by how a command is spelled; a command that names the corpus returns
nothing, because the kernel never let it look.

**The allow-list is code, and its deny set is an invariant.** Deployments may
not extend it. The paths the Agent may see are a product property: a Run that
could be configured into reading the corpus would make "retrieval is the path to
knowledge" a deployment preference rather than a guarantee. Capabilities that
need to see more declare it where they are composed, in code, and composition
fails at startup if a declared path equals, contains, or is contained by the
corpus working directory or the application tree. This is the same non-overlap
rule the composition root already applies to the Agent Workspace, moved from the
layout to the process view. Nothing is delayed by this that a deployment could
otherwise do at runtime: the only surface that can name a new root is a mount, and
a mount reaches a running process by recreating its container or pod. What lives
*under* a declared root is picked up per Run without a restart — a published Skill
is visible to the next Run — and the surfaces that are hot today (the runtime model
catalogue overlay, outbound Connections and their per-Run tool bindings) never
touch this list at all.

**Capabilities declare what they need; the core declares nothing.** Skills are
the one capability that needs more today: they are documents the Agent reads and
may reference executable assets, so the operator-global and owner skill roots are
granted read-only with execute. The packaged built-ins are not declared: the
serving process loads them through its own `load_skill` tool, their assets are
reachable where the install puts them under the runtime prefix, and a source
checkout keeps them in the project tree the Agent may not see. Capabilities whose state cannot live in a file — a browser session, a
desktop — are not paths to add: they are providers that run in their own
environment (a driver or an outbound MCP endpoint), reached through a tool, in
the shape DeepSeek Harness uses for browser-use and computer-use. The Agent's
process view therefore does not grow when such a capability arrives.

**Nothing here narrows how a Run organizes its agents.** A child session shares
its parent Run's working copy, so the confinement root is the same for a child as
for its parent: fan-out, swarms, and workflow-style compositions keep exactly the
file access they have today, and a Child spawned with the default read-only tool
set has no shell to confine at all. What changes for every agent of the Run is the
same single thing — the corpus is reached through retrieval, which composition
offers to a Child first — and no new boundary appears between a parent and its
children.

**Egress is the deployment's.** The Agent keeps the network authority the host
user has. The shell holds no application credentials — the child environment is an
explicit allow-list with no service secrets — so its reach does not include an
authenticated call back into the application. A deployment that wants no egress
at all expresses that with a network policy where egress can actually be
enforced, not with a path list that cannot enforce it.

**Unavailable enforcement is recorded, never silent and never fatal.** Landlock
needs a Linux kernel at 5.13 or newer with the syscalls reachable; a native
macOS development host or a hardened container profile may not have them. The
environment detects this before it spawns, applies what it can, and reports the
effective state as a closed value — `disabled`, `unavailable`, or `landlock:abiN` —
in `/health` and in the Run's own trace. The Run
proceeds unconfined, because a Run that cannot be confined is still a Run the
user asked for — but nothing about that is invisible afterwards.

**The Agent is not told.** The tool result stays clean: plumbing is not the
model's business, and a model that believes its sandbox is off is a model that
can be talked into reading files directly.

## Considered options

- **Isolate by mount: stop mounting the corpus into the answering container.**
  Rejected as the primary mechanism, because in a single-deployment installation
  the process that ingests documents is the process that answers questions: the
  corpus tree must be mounted for ingestion, and the Agent runs in that same
  view. Splitting ingestion from answering is a real topology and stays
  available, but it is a deployment shape rather than a guarantee this
  distribution can make.
- **Isolate by identity: run the Agent's children as another uid, with the corpus
  readable only by the application's uid.** Rejected: it needs `CAP_SETUID` or a
  root start, which a restricted Kubernetes pod security profile forbids, and it
  leaves the application's own readable files (`config.yaml`, the source tree)
  in the Agent's view. It also trades one enumerated list for a permission
  layout that a later packaging mistake can silently widen.
- **Isolate by namespace: run each command under `unshare`/bubblewrap with the
  corpus bound away.** Rejected for now: it needs user namespaces or
  `CAP_SYS_ADMIN` (denied in the current container, which has no capabilities at
  all and refuses `unshare`), so it would be unavailable in exactly the
  environments this distribution is run in, and its availability would have to
  be renegotiated per orchestrator.
- **Filter commands instead: reject reads of the corpus by inspecting the
  command text, or scan tool output for corpus paths.** Rejected. It is
  bypassable — a path can be assembled, encoded, or opened from inside another
  program — and it breaks legitimate commands (`find . -name '*.pdf'` over the
  Agent's own uploads). Domain language already lists "shell-command filtering as
  a security boundary" as a term to avoid; this decision makes that concrete.
- **Keep the allow-list configurable, with a validated deny set.** Rejected: it
  is the same product property as the previous option's counterpart, one
  configuration file away from being observable as a difference between two
  deployments of the same version.
- **Move corpus access behind a retrieval service, so the answering process holds
  no corpus credential.** Rejected *in this decision's scope*, not as a design:
  the Agent's shell does not inherit the application's environment (the child
  environment is an explicit allow-list with no service secrets), so the database
  is already out of its reach. Removing credentials from the answering *process*
  protects against a compromised application, which is the deployment's boundary
  to draw.
- **Keep `sandbox` as a documented no-op seam.** Rejected: a mode that always
  fails is a mode that will be patched on a Friday, and the failure was already
  the honest behaviour.

## Consequences

Landing order, one sequence:

1. The confined local environment: a helper that applies the allow-list to
   itself and then `exec`s the real command (a pre-exec callback is unsafe in a
   threaded event loop; the helper runs as a file, because `-m` warns on stderr for
   a module the package import chain already reached, and that warning would land
   in the model's tool output), the policy that composes the base layer and
   validates declared layers against the deny set, and the wiring at the one place
   the adapter is resolved.
2. The Skills capability declares the roots it serves — the operator-global root,
   and the shard of the owner whose Run is binding rather than the shared parent —
   and the composition root refuses every root it may serve, not only the ones this
   Run grants, when one overlaps the corpus working directory or the application
   tree.
3. Delete the `sandbox` mode: the mode literal, its resolver branch, its error
   type, the four validation sites, and the configuration description. A
   configuration that still names it fails validation; no alias is added.
4. Observability: the effective confinement in `/health` beside `service_role`,
   and in the Run trace.
5. Tests: the helper (applies, denies outside the workspace, masks by ABI,
   degrades without Landlock); an integration test asserting an Agent child
   cannot read the corpus tree while `search_knowledge_base`, retrieval image
   hydration, ingestion, and source download all still work; the degrade branch
   without depending on the platform; the `/health` value; a skill's bundled script
   runs; and an outbound MCP tool still works (it is an HTTP session, so
   confinement does not reach it).

Live documents to revise with the implementation:
[domain language](../domain-language.md) (the Execution Environment term),
[configuration](../configuration.md) (the mode list and the sandbox paragraph),
and [architecture](../architecture.md) (the mode list and the shared-mount note).

Residual risks, recorded rather than solved: on a host without Landlock the Agent
is unconfined and only `agent_shell_confinement` on `/health` and in the Run's trace says so, and a kernel that
offers Landlock while refusing it to this process is not distinguishable per
command — that one belongs to the deployment's own seccomp and capability setup,
which is also the only place it can be seen; the grant that keeps the toolchain
working is the interpreter prefix, so the installed package's source is readable
from inside the confinement even though the corpus, the deployment's configuration,
and the project tree are not; tools that hardcode
`/tmp` rather than honouring `TMPDIR` fail, which is the price of not handing
over a world-writable directory another Run — or another user's in-flight
upload — can write into; a capability that needs a display server or input
devices will arrive as a provider with its own environment rather than as a wider
list, and if that ever proves impossible the answer is a deployment that runs the
application inside such an environment, not a longer list; and the invariant this
decision protects is a product property — the Agent reaches the corpus through
retrieval — not a security boundary against a compromised application process.
