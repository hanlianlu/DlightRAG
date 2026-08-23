# Migrating to DlightRAG 3.0

DlightRAG 3.0 replaces the pre-3.0 Agent journal and control contracts. It does
not read old active Agent sessions or emulate their tool names. Preserve corpus
sources and backups, drain or cancel active Answer runs, deploy one 3.0 writer,
then start readers. Development installations should use the full reset in
[operations.md](operations.md#full-development-reset).

The 2.0 guide remains the historical 1.x-to-2.0 migration; it is not the current
upgrade path.

## Required changes

1. Upgrade both lockstep distributions to `3.0.0`:
   `dlightrag` depends on exactly `dlightrag-memory==3.0.0`.
2. Replace `answer.agent.execution_environment: local_trusted` with `trust`.
   Valid values are now exactly `disabled`, `trust`, and `sandbox`. `sandbox`
   requires a trusted execution adapter and fails explicitly when none is
   installed; it never falls back to host execution.
3. Remove Agent scopes, tool profiles, `delegate_research`, and any caller-side
   assumptions about the old journal. Every configured tool is available by
   default; child tool lists may only narrow inherited tools.
4. Recreate the development Answer/Agent/Memory schema. There is no compatibility
   reader for old run/session rows. Production operators must drain incompatible
   active runs before rolling the writer.
5. Update clients to the durable control surface. In addition to start, status,
   events, and cancel, REST, inbound MCP, and `AnswerRunClient` expose steer,
   follow-up, resume, fork, transcript tail, child roster, usage, evidence, and
   parent-run lineage. Web exposes the applicable controls and a minimal branch
   interaction.

## New optional configuration

`answer.agent.outbound_mcp` declares explicit stdio or streamable-HTTP endpoints
and the exact remote tools admitted from each endpoint. It is a thin client
adapter, not a registry or OAuth service.

Research discovers Skills progressively from `~/.agents/skills/` and the active
Agent Workspace's `.agents/skills/`; workspace metadata wins on name conflicts.
Only Skill metadata enters initial context. The `load_skill` tool reads
`SKILL.md` or a contained reference on demand and never executes Skill code.

Trusted Python composition has only three extension seams: tool registration,
Context Contribution, and ExecutionEnvironment adapter. Extensions are trusted
host code, not an end-user plugin or approval system.

## Profile Memory changes

`dlightrag-memory` now exposes the narrow host-facing Memory façade, Profile
models/errors, and store protocol. Remember formation can be split into a stable
proposal followed by idempotent commit. Forget is idempotent and records a
non-recallable tombstone. DlightRAG also supplies one stable local owner when
auth mode is `none`; simple shared-token mode remains ineligible for personal
Memory. Standalone Memory MCP clients must send a stable `idempotency_key` to
`memory_remember` and reuse it on retry. A Research run journals the exact
recalled Profile facts before model execution, so recovery never re-queries
mutable Memory.
