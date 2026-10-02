# A workspace's creator holds it

Each workspace records the owner that created it. Under `jwt_claims`, that
creator holds `editor`, `workspace.reset`, and `workspace.delete` on the workspace
beyond what Access Rules grant. Everything else still comes from the rules.

## Status

Accepted and implemented.

## Context

One deployment can now serve several people. Each person signs in at an edge
such as Cloudflare Access and becomes their own owner. Their Sessions, Profile
Memory, and Connections are already private. Workspaces were not: Access Rules
match claims to workspace patterns, so no rule could say "the workspaces this
person made". A person either saw every workspace a pattern covered, or an
operator wrote one rule per person per workspace.

The deployment that motivated this wants three things. An administrator holds
everything. Everyone reads the default workspace. Each person creates
workspaces that only they and the administrator can see or change.

## Decision

- **Record the creator.** The workspace registry records `created_by`, the
  owner id of the caller who created the workspace. REST, Web, and MCP all pass
  it. The engine keeps it as an opaque string beside the workspace identity. A
  workspace the deployment registers itself, such as the default, has no
  creator.
- **Grant the creator.** `JwtClaimsAccessControl` allows an action when a rule
  matches, or when the action is a creator action and the subject created the
  workspace. One registry lookup answers every unruled workspace in a filter.
- **Leave the rest to rules.** The grant never applies to deployment-wide
  actions or to operator storage facts. Creating a workspace stays a rule
  decision.
- **Keep a Run with its submitter.** A Corpus Mutation Run always shows to the
  person who submitted it, even after their workspace is deleted. A workspace
  its creator holds shows anyone else only the Runs its creator submitted, so a
  reused name never inherits an earlier holder's history. Seeing is not
  changing: cancelling or resuming a Run needs its action on the workspace now,
  so an old Run never writes with access its submitter has lost.
- **Compose the policy once.** The Application holds the policy, built from
  configuration with the catalog as its creator source. Transports read
  `application.access_control` instead of rebuilding it on every request.
- **Show the result in the Web.** Each workspace the Web lists carries the
  corpus changes its caller may make, and the Files panel offers only those.

## Considered options

- **Per-person rules in configuration:** rejected. Every new workspace would
  need an operator edit.
- **A membership store with roles and invitations:** rejected for now. The need
  is creator access plus one shared read. A membership store would add a user
  model DlightRAG deliberately does not own.
- **Namespacing workspace ids per owner:** rejected. Ids, cookies, and Run
  records would all have to carry the owner, and administrators would see
  mangled names.

## Consequences

- People see only their own workspaces and whatever rules grant them. A
  workspace without a creator, such as the default, is reached only through
  rules.
- Workspace ids stay deployment-wide. Creating a name someone else holds is
  refused as existing, which tells the caller that the name is taken.
- `allow_all` deployments are unchanged. Every caller there is the deployment
  owner and holds everything.
