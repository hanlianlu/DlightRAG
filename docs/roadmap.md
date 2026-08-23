# Roadmap

No decided follow-up is pending. This page records completed commitments and
explicit non-goals so they are not mistaken for open product promises.

## Completed

- [x] **Agent 3.0 kernel and durable sessions.** Product-neutral event loop,
  lifecycle events, run-local ToolRegistry, typed linear Session journal with a
  parent-linked selected-head projection, Run Segments, Context Contributions, one active compaction
  summary, model-aware reserves, replay-stable controls, and deterministic
  citation finalization.
- [x] **Execution and tools.** Base read/write/edit/grep/Bash tools, durable
  oversized-result spill, `disabled | trust | sandbox` adapter semantics,
  Path/Workspace/External scheduling, and exactly three trusted Python
  extension seams.
- [x] **Foreground subagents.** Parallel one/many spawn, status/wait/cancel,
  lineage, context/model/tool selection, inclusive usage, default depth one,
  and transactional parent Evidence adoption.
- [x] **Profile Memory and Skills.** Independent narrow Memory package,
  proposal/commit, replay-stable recall, local owner, timeout fallback,
  idempotent tombstone forget, and progressive global/workspace Skills.
- [x] **MCP and run controls.** Thin deployment-allowlisted outbound MCP plus
  shared start/status/steer/follow-up/cancel/resume/fork, transcript, child,
  usage, Evidence, and lineage projections across REST, inbound MCP, Python,
  and the applicable Web controls.
- [x] **Browser identity UX.** Edge-asserted Web identity for Cloudflare, Azure,
  and AWS, with fail-closed verification, CSRF double-submit, and exact-Origin
  checks.
- [x] **Durable context accounting.** Provider-measured anchors, dynamic model
  reserves, episodic conversation continuation, and exactly one active
  compaction projection.

## Deliberate non-goals

DlightRAG does not promise detached/background agents, workflow languages,
missions, schedules, councils, worktrees, watchdogs, a bundled sandbox backend,
a tool approval/IAM platform, a Skill marketplace, or an MCP registry/OAuth
management service. Additions require a separately approved product contract.
