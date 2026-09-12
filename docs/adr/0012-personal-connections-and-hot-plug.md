# Personal Connections with versioned Run bindings

## Status

Accepted and implemented in the owned worktree based on `main@be162182`; independent final validation/review pending.

## Context

The prior baseline composed one deployment-declared outbound MCP tool tuple at process startup. The accepted product instead lets each eligible user manage and authorize personal Streamable-HTTP MCP Connections whose tools appear automatically in that owner's future Research Runs. Durable Runs may execute or recover after a Connection changes, while disable/revoke must stop later external dispatch without duplicating existing effect authority.

## Decision

Application owns one owner-scoped Connections module with immutable Connection Generations and Credential Grants. Research acceptance atomically writes a normalized Run Connection Binding to each selected generation and pins the same tool definitions in the existing `AgentRunPlan`. Execution reconstructs those definitions through a neutral host-injected Engine Answer interface; Agent Core receives ordinary `AgentTool`s and no MCP, credential, owner, or Web identity.

Connection publication affects future Runs. Disable, deletion, and Grant revocation are checked again at a database-linearized gate immediately before network I/O; a gate that commits first is in flight, while revoke that commits first permits zero I/O. The existing Effect Intent/Effect Settlement, never-replay behavior, Run/Child Session leases, fencing, and cancellation remain the only durable effect authority. There is no `InvocationPermit` table or second runtime.

The reusable hot-plug mechanics remain private to Connections until a second real consumer proves a common seam. Models, Skills, Memory, and other integrations keep their existing owners.

See the [accepted implementation plan](../personal-mcp-connections.md) and [canonical vocabulary](../domain-language.md).

## Considered options

- **Global admin catalogue:** rejected because it grants one deployment-wide surface rather than representing each user's external account authorization.
- **Universal plugin/protocol framework:** rejected because MCP is the only real consumer and a driver registry would expose speculative, shallow abstractions.
- **Mutable current catalogue at execution:** rejected because recovery could silently change tool definitions or external identity.
- **A second durable permit/runtime:** rejected because existing pending-effect, fencing, cancellation, and never-replay semantics already own uncertain effects.

## Consequences

Every owner/FK path includes owner identity; JWT and local single-user `none` are eligible, while shared `simple` is not. Fast has no MCP. A Connection authorizes all of its current and future discovered tools, including possible writes, with no per-tool or per-call approval; OAuth scope expansion still requires provider consent.

Pins preserve local definitions, not remote code/data, availability, or external isolation. Product network, transport, quota, and secret policies may deny an otherwise authorized call but can never grant user authority. Old generation metadata remains until retained Run pins are gone; retired credentials can be erased earlier. Deployment-declared outbound MCP and Web-managed stdio paths are removed; old configuration is rejected without adapters or data reset. OAuth refresh uses a Grant-leased SDK token-only preflight followed by the complete effect gate, rather than letting SDK authentication replay a foreground effect. Writer maintenance re-encrypts by secret-version CAS and collects only unpinned metadata/expired inboxes. See the implementation contract for pending validation and operational limits.
