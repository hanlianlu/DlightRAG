# The Inspector keeps two read grains, and a step opens where it stands

The Inspector reads a Run at two grains. The **Run timeline** is the whole Run as one index: every
agent's steps merged on the clock they share, one line each, with the whole Run's kind totals, its
mechanical decision points, and a cursor that pages across agents. The **agent page** is one
agent's readable record: its transcript at message grain, its brief, its evidence and attachments,
and the words that matched a search. The first answers *what happened, in what order, and where
were the decisions*; the second answers *what exactly did this agent see and say*. Neither is a
copy of the other, and the Inspector keeps both — both read, neither commands.

What changes is that a spine row stops being a jumping-off point: it opens **where it stands**.
Expanding a step reads that step's own content — the words it said or answered, the reasoning
behind it, the Tool result it settled, the facts and sources it stated — through a bounded,
masked, on-demand read. The reader reviews a Run without leaving the timeline; the agent page
remains beside it for its search and its full lane, and it reads rather than commands.
A full merge of the two grains is a later, evidence-gated move rather than this decision: it is
named below with its trigger, its cost, and what it would give up.

## Status

Accepted and implemented. The Inspector now reads at both grains, opens a spine step in place
(one expansion at a time, through `GET /web/api/answer/{run_id}/timeline/step`), offers no child
command (the server routes stay; this surface does not present them), and opens each panel at a
proportion of its split rather than at a fixed token.

**Amended by [ADR 0085](0085-the-run-panel-is-an-overview-and-steps-belong-to-the-agent-page.md):**
the two grains stay, but the Run grain is now an overview (delivery, structure, attention) rather
than a step index; it no longer opens a step in place, and the Run timeline's paged/delta read and
the step-detail read are retired.

## Context

The Inspector is the product's only trace surface. [ADR 0037](0037-a-runs-children-are-watched-in-the-inspector.md)
made it the place a Run's children are watched and steered; this ADR keeps the watching and retires
the steering, because acting belongs to the conversation. The trace/readability work added the
Run timeline as a cross-agent index with mechanical marks
([ADR 0065](0065-the-models-reasoning-is-a-trace-surface-bounded-and-masked.md) added reasoning to
both the agent page and the timeline). The two surfaces now overlap in appearance — both show
steps, both can show words — which is what prompts the question: why keep both?

They differ in grain, scope, content, paging, and operations:

| | Run timeline | Agent page |
| --- | --- | --- |
| Read | `GET /web/api/answer/{run_id}/timeline` | `GET /web/api/answer/{run_id}/transcript` (+ children roster, search) |
| Grain | one step, one line | one message, readable body |
| Scope | the whole Run, all agents merged by time | one agent's Session lane, including earlier Runs of its conversation |
| Content | kind, agent, subject, bounded excerpt, marks, Tool facts, reasoning folds | full masked message text, Tool results, instructions, delivery, brief, attachments/evidence, search hits |
| Paging | step cursor across agents | message sequence within a Session |
| Operations | none (read-only index) | feedback only; questions read-only, no child commands |
| Live | refresh on Run activity | `since` delta polling per agent, follow-the-tail |

The two also serve different moments. Before a reader knows where to look, the index is what lets
them see the shape of the Run — where a child worked, where a call failed, how many decisions the
Run stated — without loading a transcript. Once they know where to look, the agent page is what
lets them read and search. Merging them means choosing one of those grains for both jobs, and
paying for it in payload, paging, or loss of the page's reading space.

## Decision

**Keep the two grains; make a spine step readable in place.** The Run timeline's rows gain an
expandable body, fetched on demand, bounded and masked like every other reader projection:

- A step that states prose (a `say`, `answer`, `instruction`, `delivery`, or `thinking` step) opens
  its own masked words, not a 120-character excerpt. A Tool step opens the result it settled and
  the facts its card already states.
- The body comes from a step-detail read scoped to the Run and the step's identity
  (`agent`, `sequence`, `kind`), not from widening the page payload: a page of 200 steps would
  otherwise carry megabytes a reader never opens.
- The step keeps its identity and marks; expanding it never moves the cursor or the decision
  count, and the whole-Run totals remain the page's summary.
- **The Inspector reads; it does not operate.** The agent page loses its steer / continue /
  cancel / reply composer, the reauthorize box, and the command-outcome states that existed to
  drive them. An `ask_parent` question reads as a status line (who it asks, when it expires), not
  as a box; the parent agent answers it, and the owner who wants to steer continues in the
  conversation's own composer. A review surface that also writes has a second command surface to
  keep correct (ids, idempotency keys, outcomes, authorization), and the conversation is already
  the place a reader states what they want. The server-side child routes stay as they are; it is
  the Web Inspector that stops offering them.
- **One expansion at a time.** Opening another step closes the open one: the spine stays an
  index, and a reader never scrolls past a stack of expanded bodies to reach the next step.
- **The step's own content** may be a Tool result body; it is bounded and masked through the same
  reader content gate as the transcript it comes from. No new payload class crosses to a reader
  without that gate.

**Panel defaults are proportional, not fixed.** That rule is its own decision
([ADR 0068](0068-panel-widths-open-at-a-share-of-their-split.md)); this ADR's two-column layout
relies on it, because the traces dock has to reach its two-column width before it can be read as two
columns.

**The full merge is a named later option, with a trigger.** If, after the in-place expansion has
been in use, the agent page's reading is no longer what readers use it for — the page is opened
mainly to steer a running child or to search one agent's words — then the merge is the right
simplification: the spine becomes the single Run surface, search filters the spine, and the agent
page retires. Until that evidence exists, merging would trade a working reading surface for a
larger payload and a lost lane view.

## Considered options

- **Merge everything into one Run surface now.** The owner's first suggestion, and the simpler UI
  on paper. Rejected for now: the spine is step-grain and the transcript is message-grain, so one
  surface must either ship whole message bodies with every page (the page becomes the transcript,
  and the one-line index is lost) or reduce the agent page's readable content to a step line (the
  reading surface is lost). The operations, the per-agent search, and the lane replay would all
  have to move in the same change, and the Inspector is the product's only trace surface — the
  cost of getting it wrong is a reviewer with no place to read. The merge is preserved as the
  evidence-gated option above rather than rejected on taste.
- **Keep both exactly as they are, change nothing.** Rejected: the overlap is real — a reader who
  wants to know what a `say` actually said must leave the index, and a step's own content is the
  index's most obvious next read.
- **Expand steps by widening the timeline page payload** (full content for every step on the
  page). Rejected: it makes the common read pay for the rare expansion and turns the index into a
  transcript with extra steps.
- **Show only Tool steps' content in place** (leave prose to the agent page). Rejected: the
  question "what did it say here" is exactly as common as "what did the call return", and a
  surface that answers one but not the other keeps the jump in the middle of the most common read.
- **Keep the child controls in the Inspector.** The status quo, and the owner's answer was no:
  steering is a conversation act, and the Inspector should not grow a second composer with its own
  idempotency, reauthorization, and outcome vocabulary. The controls are removed rather than moved;
  the conversation's composer already states the intent.
- **Adopt the DSH pattern: Chat and Trajectory tabs per Session.** Not applicable as-is:
  DSH's trajectory is per-session and its subagents are separate sessions, so it never has to
  merge agents; DlightRAG's Run spans the main agent and its children, and the Inspector already
  merges them. The part worth copying is DSH's per-record inspector: a row opens on the record's
  own content and timing rather than sending the reader elsewhere — which is what this decision
  adds.

## Consequences

- The Reader can review a Run end to end in one surface; the agent page stops being on the critical
  path for reading and becomes a read-only detail surface.
- [ADR 0037](0037-a-runs-children-are-watched-in-the-inspector.md)'s monitoring half stands — the
  children are watched in the Inspector — while its steering half is retired: the Web Inspector
  offers no child command, and a reader who wants to act returns to the conversation. The API's
  child control routes keep their contract for clients that call them directly; only this surface
  stops presenting them.
- Panel defaults are proportions of their split rather than fixed tokens, with the open/drag/resize
  rules [ADR 0068](0068-panel-widths-open-at-a-share-of-their-split.md) states.
- One more reader read exists: a step-detail route scoped by `(run_id, agent, sequence, kind)`,
  through the same content gate, bounded per response. It must state no more than the transcript
  would; the transcript stays the wider of the two reads.
- The decision count and kind totals keep their meaning: expansion is a display action. A thinking
  step still opens on its own reasoning and still is not a decision point.
- A page's steps each hold a closed disclosure by default; the timeline's one-line-at-a-glance
  property survives because the index is still one line per step until a reader opens one.
- The full merge stays possible without another reversal: the step-detail read is exactly the
  content a merged surface would need; what would remain is deciding which surface keeps per-agent
  search and the lane view. The trigger for revisiting is stated above so this decision does not
  become an argument against simplifying.
