# ADR 0062 — MCP tools are declared whole, and resources are read on demand

## Status

Accepted and implemented: every enabled Connection's tool definitions are declared whole on every
Research request, and a Run reads its servers' resources on demand through three shared tools.
The three resource tools are pinned in the Agent Run Plan like every other tool, the server's
bounded `initialize` instructions are pinned with its generation and attributed in the prompt, and
the Connections projection states each published catalogue's tool count and declaration cost. No
`AgentRunPlan` field is added: a Plan accepted before this decision serializes and digests as it
did.

## Context

[ADR 0012](0012-personal-connections-and-hot-plug.md) made the Connection the authorization unit
and pinned every published Connection definition into the Agent Run Plan before the first provider
call, so a resumed or retried Run rebuilds the tool surface it was promised rather than whatever
the Connection has since become. [ADR 0015](0015-prompt-prefix-stability-and-cache-anchored-accounting.md)
made those definitions part of the request's fixed envelope rather than anything a Run composes per
turn: byte-stable for the Run, shared by every Run that composes the same tools, and billed once in
the reusable prefix. The model therefore sees every tool the Run may call, and no product surface
has to explain which tools are "active".

The MCP Client Best Practices page describes the opposite pattern as an option for hosts that
connect to many servers: keep the fixed surface small, defer tool definitions, expose a lightweight
`search_tools` meta-tool, and load definitions as needed — with a threshold around 1%–5% of the
context window and, where a provider offers it, the provider's own tool search
([Client Best Practices](https://modelcontextprotocol.io/docs/2026-07-28/develop/clients/client-best-practices)).
That pattern was built as a candidate for this product and declined. Tool selection by retrieval is
a small RAG over tool descriptions; for MCP catalogues it is overkill, it adds a load step the
model can skip, and it puts the declared surface, a per-Run budget, and a host-executed
meta-tool between the pinned plan and the request. Flat declaration is the shape the harness's MCP
client (DSH) uses, and it is the shape here: each server's tools are named `mcp__<server>__<tool>`
and declared whole, and resources are a shared package (`dsh-mcp-resources`) rather than a tool per
server.

Two concrete pressures named the limits of that shape. On 2026-10-22 Notion's remote MCP server
published 65 tools in a 342,733-byte catalogue, with a largest input schema of 160,357 bytes; the
shipped per-tool bound of 128 KiB refused it, so discovery published nothing, and a deployment that
admits it declares that whole catalogue on every request. The cost is real and is now visible in
Settings rather than hidden or capped. Separately, Notion's create-pages tools ask the model to
read the server's own Markdown specification resource
(`notion://docs/notion-flavored-markdown-spec`), which no tool declaration can carry; servers
document themselves as resources.

## Decision

**Declare every pinned definition whole.** A Research Run pins each enabled Connection's published
tool definitions in its Agent Run Plan and declares all of them on every request. There is no search
meta-tool, no ranking, no deferral, no per-Run declaration budget, and no per-tool selection: the
model sees every tool it may call, and the plan is the whole truth about authority. Fast still has
no MCP tools, and nothing about Grants, consent, enablement, the dispatch gate, failure behavior,
or retention changes.

**The product control is the Connection's switch.** Enabling a Connection authorizes every tool it
publishes; disabling it removes the catalogue from future Runs. A large catalogue is a reason to
leave a Connection off, not a reason to filter its tools at Run time.

**Flat declaration is chosen for reliability and the cache.** The model never has to succeed at a
load before it can call a tool, there is one declaration path rather than a plan plus a
transcript-derived load set, and the pinned array is the same bytes for every request of a Run, so
the provider prefix cache behaves as [ADR 0015](0015-prompt-prefix-stability-and-cache-anchored-accounting.md)
describes. The MCP guidance is recorded here as the source of the optional pattern this product
declines, not as a recommendation it follows.

**The cost is shown, not capped.** The Connections projection gains `tool_count` and
`declaration_tokens`: the published catalogue's size and the estimated tokens its declarations add
to one Research request, measured with the same estimator the runtime's accounting uses. Both are
null until a catalogue is published, and a disabled Connection's estimate is hypothetical and still
shown. The card states it as `≈ N tokens per request · M tools` (the zh copy renders the same
figures as `≈ N 个 token/请求 · M 个工具`), so an owner sees what switching a catalogue on costs
before the first Run pays it. The estimate covers the catalogue's own definitions; the three
shared resource tools and the server instructions are not part of it. Nothing truncates or defers
a definition, but the estimate is display, not a budget: the whole declared array is part of the
acceptance measurement, so a Run whose model cannot hold it is refused before admission rather
than silently overfull.

**Resources are read on demand.** A Research Run with at least one enabled Connection that has a
published catalogue (and, where that generation names a Grant, an active one) composes three
shared, pinned tools, one set for every
Connection rather than one set per server:

- `mcp_list_resources(connection, cursor?)` lists one page of the server's known resources and
  returns the cursor for the next page;
- `mcp_list_resource_templates(connection, cursor?)` does the same for the server's URI templates;
- `mcp_read_resource(connection, uri)` returns one resource's text, and describes a binary resource
  (URI, media type, base64 length) rather than passing bytes to the model.

`connection` is the token the Run's system prompt lists — the Connection's label as the prompt
renders it, with a duplicated label disambiguated by the Connection id's first eight hex characters
in parentheses (extended if those would still collide), guaranteed unique and refused if a pinned
input repeats one — and never the Connection id; it is bounded to 80 characters, a listing cursor to
2048, and a read URI to 4096. The tools are read-only and `replay_policy="replayable"`:
a read mutates nothing, so one may run beside other read-only calls, and a recovery may run a
pending read again rather than settling its outcome unknown. Each call opens its own SDK session,
there is no resource cache, and the transport owns timeouts, admission, header policy, credential
redaction, `max_result_parts`, and `max_result_bytes`.

**A read passes the live authority gate, and the content is live.** A resource call resolves the
token against the Run's pinned `connection_surface`, then passes the same Connection authority a
dispatch passes, without running discovery: the pinned generation's endpoint and catalogue digest,
the head enabled, not deleted, and consented at the pinned activation epoch, the generation's
Grant active for that endpoint when it names one (with the OAuth refresh preflight re-running the
gate), and the live Run or Child fence; a watcher re-checks the head, Grant, and fence while the
read is in flight, so a revoke or a moved lease cancels it. Only then does the request reach the
server. The generation fixed the definitions; what a
read returns is the server's content at that moment, like a fetched page, so it is never pinned,
never becomes a catalogue entry, and may differ on a later read. It enters the Run as the call's
tool result. This is how a server's guidance document becomes readable on demand — on the Notion Connection, the
Markdown specification its create-pages tools ask for — without being declared on every request.

**The server's instructions are pinned and attributed.** Discovery captures the MCP
`initialize` result's `instructions` text and stores it with the immutable generation, bounded by
`answer.agent.connections.max_instruction_bytes` (default 32768 bytes, matching the harness's MCP
client; 1024–131072). A text over the
bound rejects the catalogue candidate whole; nothing is truncated. A prepared Run input carries the
text independently bounded at the 131072-byte hard ceiling, so a corrupt or hand-built payload
cannot inflate a prompt. Credentials are redacted from the text, and a whitespace-only text is
treated as none. A Run pins the generation, so its prompt carries the pinned instructions of
each Connection as a separate reference message after the transcript, never as system-prompt
prose: one line naming the Connections available for resources when the Run composes those
tools, one paragraph stating that the block is data rather than instructions, and then one
product-marked block per Connection that states instructions (`[External server guidance —
<label>]` … `[End external server guidance — <label>]`). The framing is part of the contract:
the text is data the model may use, never an instruction from DlightRAG, never authority, and
a later publication reaches future Runs only. A Child that inherits no Connection tool and no
resource tool carries no external guidance at all; the pinned surface itself is not narrowed per
Connection. A deployment that does not serve the resource transport still pins the surface and
its instructions, but composes none of the three tools.

**Scope stays closed elsewhere.** MCP prompts, apps, and resource subscriptions are still not
supported; resources are listed and read one page at a time by the server's own cursors, and there
is no listing at discovery, no subscription, and no cache. Grants, OAuth scopes, retention,
enablement, and the pending-effect contract are unchanged.

## Considered options

- **Progressive disclosure behind a `search_tools` meta-tool.** Rejected: it was built as a
  candidate and withheld. Retrieval over tool descriptions is a small RAG for catalogues of this
  size, the model can fail to load a tool it needs, and the mechanism adds a per-Run declaration
  budget, a host-executed tool, and a transcript-derived load set that the flat plan does not need.
- **A provider's own tool search.** Rejected for the same reason the provider guidance is optional:
  it is provider-specific, and the model still has to be given the definitions the Run may call;
  one declaration path is more reliable and cheaper to reason about.
- **One resource tool per Connection, or per-server resource tools.** Rejected: the three shared
  tools keep the tool set constant across Runs and Connections, mirror the harness's shared
  resources package, and need no per-server tool naming.
- **List and pin resources at discovery time.** Rejected: resources are live and unbounded where
  tools are a bounded catalogue; discovery already owns a candidate and a size budget, and pinning
  document content would freeze what the server can update between reads.
- **Pin the resource content with the generation.** Rejected: content is live, like a fetched page.
  The pin fixes what authorizes a read and which definitions are declared, not what the server
  returns later.
- **Discard the `initialize` instructions as before.** Rejected: servers use them to explain
  themselves, and Notion's tool descriptions point the model at a resource document precisely
  because the tool schema cannot carry that guidance.
- **Truncate over-limit instructions.** Rejected: a catalogue candidate is accepted whole or
  rejected whole already; cutting a server's guidance mid-sentence would make the prompt state
  something the server did not say. The operator can raise `max_instruction_bytes` within its
  ceiling.
- **Add a declaration budget or auto-disable a large catalogue.** Rejected: the trade-off is shown
  in Settings and the owner's enable switch is the control; a second cap would decide for the owner
  which tools the model may see, which is the progressive-disclosure trade this decision declines.
- **Carry instructions without a Connections list in the prompt.** Rejected: the resource tools
  take a Connection token, so the prompt must state the tokens the model may use.

## Consequences

- A Research request carries the whole pinned tool array. A large catalogue therefore costs its
  declarations on every request; the Settings estimate makes the number visible before enabling,
  and nothing truncates or defers a definition to hide it.
- The declaration array and the external guidance are byte-stable for a Run: both are pinned at
  acceptance, a later publication reaches future Runs only, and the provider prefix cache sees a
  stable envelope ([ADR 0015](0015-prompt-prefix-stability-and-cache-anchored-accounting.md)).
- Resource reads are live and may differ between calls. A binary resource is described rather than
  returned; a fault fails the call, not the Run; the three tools run beside other read-only calls,
  and a pending read is replayed on recovery rather than settled unknown.
- The server instructions are bounded per Connection, so several enabled Connections cannot spend
  the answer's context on unbounded remote prose. A server that exceeds the bound rejects its
  catalogue candidate; nothing is truncated. Their content is remote-authored: it is framed,
  attributed, and carried as a reference message rather than system prose, but a hostile server
  can still write text that tries to persuade the model, so the framing and the bounded authority
  are mitigated, not eliminated.
- The Notion catalogue now publishes where the deployment raises `max_schema_bytes` to its 256 KiB
  ceiling, its declaration cost is stated in Settings, and the Markdown specification its tools
  ask for is readable through `mcp_read_resource` without a search meta-tool.
- Grants, OAuth scopes, enablement and consent, the dispatch gate, failure behavior, and retention
  are unchanged. The one durable addition is the per-generation `instructions` text; the
  projection adds two numbers, and no new table, register, or effect path is introduced.
- Stated limits, not covered: MCP prompts and apps; resource subscriptions; a server whose
  resources cannot be listed or read through its own cursors; a resource too large for
  `max_result_bytes` (the call refuses rather than truncating); and a server whose instructions
  exceed `max_instruction_bytes` (its catalogue is not published until the bound or the text
  changes).
