# ADR 0085 — the Inspector's Run panel is an overview, and steps belong to the agent page

**Status**: Implemented on 2026-10-30 in answer to the owner's direction: the Run timeline and the
Agent timeline both listed steps, which read as duplication; the Run panel should carry higher-value
Run-level content, the panel is read by the web consumer, and the duplicated path is deleted rather
than kept. The Run overview, the two new reads, the deleted timeline wire and step-detail read, and
the web consumer's dossier are in place and accepted by the owner on 2026-10-31.

**Summary / 摘要**

The Inspector read a Run at two grains but drew both as step lists: the left "Run timeline" merged
every agent's steps into one index, and the right agent page's "Agent timeline" listed one agent's
steps. They render the same stored entries, so the wide dock showed the same work twice while the
facts a *Run* can state and an *agent page* cannot — what the Run delivered, who did what, what went
wrong, where the decisions were — were either absent or scattered through the rows.

This ADR moves the Run grain up one level. The left column becomes **Run overview**: the Run's
delivery and trust record, a composition bar that toggles by step kind or by agent, one ledger row
per agent with its span over the Run's clock (the cross-agent order and overlap the merged list was
the only place to see), and one Run-quality line in the reader's words whose technical details
fold behind a single disclosure. The only
step list in the Run panel is the search-results mode, because "find the words across every agent"
is a real Run-level task. Steps, their full text and their detail belong to the agent page, where
they already are. The retired wire — the paged head/older/delta timeline read and the on-demand
step-detail read — is deleted, so no second path to a step survives.

## Context

[ADR 0067](0067-the-inspector-keeps-two-read-grains.md) kept two read grains and made a timeline
step readable in place. [ADR 0070](0070-the-inspectors-run-timeline-is-a-bounded-index-followed-by-delta.md)
turned the Run timeline into a derived, paged, delta-followed index. Both decisions were right about
*what the two surfaces know*: the Run timeline was the only cross-agent order and the only surface
carrying the whole Run's mechanical marks; the agent page was the only readable transcript. But both
answered "what happened" with a list of steps, and the wide layout places those lists side by side.
ADR 0067 itself noted that "the two surfaces now overlap in appearance — both show steps".

The product's readers are web consumers, not operators of the trace pipeline. What a consumer needs
from a Run surface is a different set of questions: did this Run deliver anything, is the deliverable
attested, which sources back it, who did the work, and is there anything I must look at. A step list
answers none of those; it makes the reader do the aggregation. The data for the Run-level answers
already exists: the roster carries per-agent status, objective, tokens, timings and result; the
derived timeline index already holds every step's agent, instant, kind, tool duration and decision
marks; the Run result already holds root/child/inclusive usage; the answer presentation already holds
artifacts with their per-artifact attestation record and the artifact outcome. The compact form of
the one Run-level fact a list carried — order and overlap — is a span per agent, not a row per step.

The step list also costs the most to serve: on the measured Run, a head page was 133 KB before
reasoning was removed and a live watch read it repeatedly; the delta machinery exists only to keep
that list current. Once the Run panel stops listing steps, the delta, the head paging and the
on-demand step-detail read have no reader, and keeping them would be legacy rather than a contract.

## Decision

**1. The Run panel is a Run overview, and it never renders one row per step (search excepted).**
The left column states, top to bottom:

- a one-line header: Run state, duration, agents, steps, tokens;
- a composition bar with a two-mode toggle — by step kind (default) and by agent — under a legend
  that names every segment (colour alone states nothing). Every part states its share on hover;
  in the agent mode a part is a control that selects that agent's page. The agent on show is
  marked current for assistive technology in the bar, the legend and the ledger, but no element
  wears a permanent highlight there: the switcher line and the page beside it already state which
  agent is on show, and a surface where one agent is always current must not dress that fact as a
  selection (the selected-row token belongs to lists where selection is a mode);
- a delivery card: Run state, sources, artifacts with their attestation record, artifact-outcome
  issues, and the token composition when the Run states one. It states only what exists (a
  text-only answer says so in one quiet line instead of two zeroes) and no count of "delivery
  checks": a delivery step is a pre-delivery note for the quality details, not a tally;
- an agent ledger, one row per agent: name/objective, state, duration, tokens, step and tool counts,
  and its span over the Run's clock with the decision instants marked (the span and a mark state
  their own detail on hover). A row is one control that opens that agent's page, and the row of
  the page already on show carries only the accessible current mark — no fill, no chevron: a
  permanent highlight would read as an alert in a surface where that row is always current. This
  is what keeps the one fact only the merged list carried — cross-agent order and overlap — in a
  form that is not a second copy of the steps;
- a Run-quality line written for the Run's reader rather than its operator: a calm sentence says
  whether the Run recorded anything to flag, and the section head's trailing
  `dl-text-action--disclosure` folds the whole Run's mark counts and findings (failed, degraded,
  redirected, repeated, delivery notes) into a bounded, keyboard-focusable scroll region under it,
  so opening the detail never grows the dossier to the length of its list. Each finding still
  selects its agent and focuses that step in the Agent timeline; the ledger carries no mark
  vocabulary, and a clean Run states that confidently rather than drawing an empty box.

The rule is stated so later work cannot reintroduce the duplication: **the Run panel renders
aggregates, structure, outcome facts and links; a row that is one step belongs to the agent page.**

**2. Search is the one step list the Run panel keeps.** Searching words across every agent is a
Run-level task; typing switches the panel to matching steps, paged, with a count and the words it
read. Clearing returns to the overview. The search rows are an index, not a reading surface: they
carry a one-line excerpt and select the agent whose page holds the step; they do not expand in place.

**3. The Run grain gets its own reads, and the retired ones are deleted.**
- `GET /web/api/answer/{run_id}/overview` serves the Run's aggregates: step totals by kind, decision
  count and mark split, tool time, per-agent rollups, the bounded findings list with its uncapped
  total, and the Run result's usage triple. It is served from the same derived index the timeline
  used, with the same version/ETag discipline, and applies the reader content gate to every text
  field at emission.
- `GET /web/api/answer/{run_id}/steps/search` serves the search-results mode; `q` is required and
  the read excludes thinking steps, because a result names the actionable step and a list no longer
  ships reasoning.
- `GET /web/api/answer/{run_id}/timeline` (head, older-page, delta and search modes) and
  `GET /web/api/answer/{run_id}/timeline/step` are deleted with their client code, their wire fields
  (`totals`, `version`, `next_since`, `reset`, per-step `facts`/`call_id`/`index`) and their tests.
  The derived index, its version and the mark rules remain: they are what the overview and the
  search read are computed from.

**4. Missing facts are stated as missing.** The overview never invents a verdict: usage absent is
`null` and is not drawn as zero; a Run that has not settled says so instead of showing an empty
delivery, and an older payload with no attestation record draws no verdict. No dollar cost is shown:
the repository holds no price table, and only a provider-reported charge or a web-search cost exists.

**5. The Agent timeline is unchanged and stays the only full step surface.** The agent page keeps its
transcript, reasoning, tool results, brief, attachments, controls and questions; a finding or a search
result hands the reader to it, and the page focuses the step the Run panel named.

**The two read grains stay, at different levels:** the Run grain is the dossier (aggregates,
structure, outcome, attention) and the Agent grain is the record (steps, text, detail). This does not
perform ADR 0067's named "full merge": the merge would have made one surface carry message bodies,
which is exactly the payload this decision removes from the Run panel.

## Considered options

- **Keep the step list and add the dossier above it.** Rejected: the list is the duplication; a panel
  that keeps it keeps the payload, the paging, the delta read and the "where do I look" work the
  reader should not have to do.
- **Delete the cross-agent order outright.** Rejected: the merge is the one fact a per-agent page
  cannot state. The ledger's spans and marked instants carry it as a picture instead of a list.
- **Put the steps only in the agent page and drop search too.** Rejected: "find these words across
  every agent" is a Run-level read real readers use; without it a reader searches by opening agents.
- **Make the Run panel a monitor/dashboard** (cost, latency, percentiles, phase timeline, agent DAG).
  Rejected for this change: a single Run is one sample, most of it is fleet or operator content, and
  the parts worth having (phase history, per-call model latency, dollar cost) need reads or a price
  table this decision does not create. The overview is the surface those modules can join later.
- **Keep the on-demand step-detail read for the Run panel's search rows.** Rejected: a result row
  should take the reader to where the step happened, not grow a second reading surface in the Run
  panel; the agent page already reads the full step.
- **Serve the overview from the client's loaded pages.** Rejected: the client holds one page of
  steps, so any number computed there would describe the pages loaded, not the Run; the derived index
  is the only whole-Run read.
- **Amend ADR 0067 in place.** Rejected by the record convention: the two-grained decision stands;
  this decision changes what the Run grain renders and retires the list, so it is its own record.

## Consequences

- The wide dock reads as a dossier beside a record instead of two step lists: the Run's delivery and
  trust facts, its agents and its decisions first; the selected agent's steps where they belong.
- The Run timeline's paged head/older/delta reads, its `version`/`next_since`/`reset` wire and the
  step-detail read are deleted; the derived index, its version, the merge, the marks and the step
  rule remain and now serve the overview and the step search.
- Live following refreshes the overview with the roster (one coalesced read, browser-cache
  revalidated by ETag) instead of appending timeline deltas.
- Search results exclude thinking steps; a thought is read in the agent page's timeline, as before.
- The Run panel's facts are server-derived; no Run-level number is computed from a page the client
  happens to hold. The overview read is O(index) like the timeline read it replaces, and its findings
  read loads only the bounded entries it states.
- ADR 0067 is amended: the Run grain no longer opens a step in place, and its "Run timeline" is
  replaced by the overview. ADR 0070 is amended: the deltas, the page cursors and the step-detail read
  it defined are retired; the in-process index and the version token it defined stay.
- Documentation: `docs/interfaces.md` states the two reads and drops the two retired routes; the
  Inspector copy says "Run overview" in both languages.
- The feedback loop the old list made cheap — scan a Run's steps to find a failure — is replaced by
  the Run-quality line and its folded findings, which name what the Run's own marks state; a
  failure the marks cannot see is read where it happened, in the agent page. The line is the
  default view because the panel's reader is the Run's consumer, not its operator; the technical
  detail is one press away rather than the first thing read.
