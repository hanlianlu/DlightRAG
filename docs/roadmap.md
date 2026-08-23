# Roadmap

## Decided follow-ups

These are decided but not scheduled; each lands with its own product contract
when picked up.

- **Answer output continuation.** When final answer generation stops with
  `finish_reason=length` because the resolved `max_output_tokens`
  underestimates the model, a continuation harness would issue a bounded
  "continue from here" follow-up call and stitch the result before citation
  finalization. Deferred: the fallback profile is deliberately generous on
  output, the existing Research length-stop semantics (skip tool preflight)
  must stay intact, and stitching across calls needs its own token budget and
  citation-boundary contract.

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
- [x] **Model capacity fallback.** Uncatalogued endpoints resolve to a shared
  generous fallback profile instead of failing, with uncatalogued-use logging;
  `supports_tools` was removed from the profile schema because every modern
  model is tool-capable.

## Deliberate non-goals

DlightRAG does not promise detached/background agents, workflow languages,
missions, schedules, councils, worktrees, watchdogs, a bundled sandbox backend,
a tool approval/IAM platform, a Skill marketplace, or an MCP registry/OAuth
management service. Additions require a separately approved product contract.
