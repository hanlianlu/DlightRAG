# The Inspector's Run timeline is a bounded index, followed by delta

The Inspector's Run timeline is read from a derived index: one merged, marked, totalled view of a
Run, kept in process under the Run version it was built from, answering a page in O(page) and an
opened step in O(1). A live reader follows that index by the steps that landed after the cursor it
holds. The page states the index; a step states its own content, which is where the model's
reasoning is read.

## Status

Accepted and implemented. It retires the Run-timeline page's `reasoning` field (amending
[ADR 0065](0065-the-models-reasoning-is-a-trace-surface-bounded-and-masked.md)), extends
[ADR 0067](0067-the-inspector-keeps-two-read-grains.md)'s on-demand step read, and creates no
second durable truth ([ADR 0069](0069-the-web-reads-run-activity-from-the-runs-it-keeps.md)).

**Amended by [ADR 0085](0085-the-run-panel-is-an-overview-and-steps-belong-to-the-agent-page.md):**
the in-process derived index, its version token and the mark rules stand and now serve the Run
overview and the step search; the head/older/delta page reads, the `next_since`/`reset` wire, the
step-detail read and their client code are retired.

## Context

On the measured Run — 299 stored entries, 300 timeline steps — a production
`GET /web/api/answer/{run_id}/timeline` head page cost about 97–126 ms and answered 133 KB. The
work was not the step count. The same read with the reader content gate disabled was 22 ms;
roughly 80% of the request was `SkillProtection.mask` scanning the page, because one page's
working set is 149 unique strings while the process's mask cache held 128. The cache was smaller
than one page, so a warm read behaved like a cold one: 107.4 ms warm against 82.0 ms cold. Sizing
the cache to more than a page dropped the same read to 22.3 ms and the gate itself to 0.36 ms.

The page also carried the model's reasoning on every thinking step — 70% of its bytes — and no
list row renders it: 133,358 B became 41,510 B without it, the whole Run 404,981 B became
119,276 B, and the compressed page fell 79–80%. The text is real reader content, but it belongs
to the step a reader opens, not to the index every reader pages through.

Every read recomputed the entire Run. Page 1 cost 114 ms, page 2 97 ms, and a page requested past
the end still cost 16.9 ms, because the roster, all of the Run's steps, the Tool subjects, the
merge, the marks and the totals were rebuilt from canonical tables before a page was cut. Opening
one row cost 15.6 ms for the same reason: the detail read merged the whole Run to validate one
step. Paging was O(whole Run) per page and expansion O(whole Run) per step.

Live follow refetched the full head page about once a second, because the read had no delta
cursor, no ETag, and the client no page cache. Watching one Run for 11.8 minutes made 59 timeline,
412 transcript and 59 roster requests — 7.55 MB uncompressed, 7.37 MB of it timeline pages — for a
Run that gained a few dozen steps.

The step read also paid for the conversation rather than the Run: it walked the Session's
parent-entry ancestry from the lane head to the conversation root, recursively, and filtered by
`run_id` afterwards. A 2-step Run inside a 410-entry conversation performed 410 recursive loops
against an index on `(owner_id, run_id, sequence)` that could have answered the same rows directly.
The lane walk is the right read for a transcript that spans a conversation's Runs; it is the wrong
order for a read scoped to one Run.

These are performance findings about one read path, not a product change. ADR 0067's two read
grains stay, and the index is a projection of the same canonical tables the transcript reads. The
absolute tail measured during the same work came from host and event-loop stalls, not from this
path — a 175 B roster read also took 2.1 s and 12.7 s in those windows — so the decision below is
to make each read cheap and rare; it does not claim to fix a stalled process.

## Decision

**The timeline is a derived in-process index, keyed by a Run version.** `AnswerService` keeps, per
`(owner, run)`, the merged step index — agent identities, steps, Tool subjects, the whole Run's
marks and totals — and the version token it was built under. The token is the Run's
`durable_progress_version` (bumped once per durable entry transaction, never by a token or
reasoning batch), its lifecycle `status`, the child-roster fingerprint (each Child's id and
objective, in roster order), and each Session's highest stored sequence (which catches a recovery
write that appends without moving the progress version). A read compares the token the store states with the index's; a
mismatch is a rebuild, not a stale answer. The cache is a bounded LRU and it is disposable:
losing it costs one rebuild, never correctness, and it creates no durable second truth
([ADR 0069](0069-the-web-reads-run-activity-from-the-runs-it-keeps.md)). A page read is now
O(page) — O(limit) steps plus the projection they carry — and an opened step is O(1).

**Run-scoped reads use the `run_id` index, not the conversation-lane walk.** The parent-entry
ancestry CTE stays for `scope="lane"` transcript reads, where folding a conversation's Runs is the
point. A Run's timeline steps and a Run-scoped agent transcript read filter `run_id` directly: the
`(owner_id, run_id, sequence)` index already existed, and the row set is the one the walk reached
(verified equal across the Runs and entries of the development database). A short Run's cost no
longer scales with how long its conversation is.

**The page is an index; a step carries its own content.** The page states each step's identity and
shape — agent, `at`, `kind`, the Tool's `name`/`subject`/`outcome`/`duration_ms`, a bounded
`excerpt`, the already-masked Tool `facts`, its `sequence` and `call_id`, and the whole Run's
`marks` — beside the Run's totals and the paging cursors. It no longer carries the model's
`reasoning`: the list never renders it, and ADR 0067 already chose "a step opens where it stands"
over widening the page payload. A thinking step's `excerpt` is cut from its masked reasoning; the
full masked text is read on demand through `GET /web/api/answer/{run_id}/timeline/step`, where it
keeps ADR 0065's 4000-character bound and the same reader content gate
([ADR 0061](0061-a-protected-skills-text-is-masked-at-the-readers-edge.md)). The step read states
the step's own masked words (65,536 characters for a prose or Tool-result body), its reasoning and
its facts. This amends only the "both surfaces carry reasoning" half of ADR 0065: reasoning
remains a trace surface through the activity page and the on-demand step read.

**The timeline has a version, a delta and HTTP caching.** The answer states `version`, an opaque
token for the Run state it was read from, and `next_since`, a cursor naming the Run's newest step
(null while a search narrows the read). `since=<cursor>` returns only the steps after that step in
merge order, plus the current totals and version; `since` and `before` walk opposite directions
and are alternatives, and `since` does not combine with a search (either pairing is a 422, as is a
cursor the route did not state). A followed delta states `reset: false`; a valid cursor whose step
the Run no longer holds, or a delta longer than one page, answers a full head page with
`reset: true`. The client appends a delta and still performs a full head read every 15 seconds:
marks are whole-Run order rules, so a later duplicate can put a `repeated` mark on a step the
client already holds, and the periodic head read is the path that heals it. Responses carry an
`ETag` derived from the version and the request's cursor, search and limit, and `If-None-Match`
answers `304` with no body. Every answer is `private`: a head page is `no-cache`, an older
`before=` page may be reused for 60 seconds, and a delta or searched answer is `no-store`.

A live dock therefore stops re-reading the head page on every event: it follows with the delta and
keeps the 15-second reconcile. The 1 Hz whole-tree re-render is gone, with the clock moved into the
components that display elapsed time; the agent page mounts beside the roster rather than behind
it, and refreshes its transcript on Run events rather than as a render side effect.

**Keeping a page O(page) includes keeping the gate's work proportional to it.** The mask cache is
sized for several pages, not one transcript page, because its previous bound was smaller than a
single timeline page's working set; the Inspector projections no longer mask Tool-call arguments
they never emit; the protected `(Skill, path)` set is cached beside the existing 30-second text
index instead of rediscovered on every request; and page projections run off the event loop. These
are supporting fixes to the same read, not new reader surfaces.

## Considered options

- **Keep the per-read rebuild and only cut bytes** (retire the page's reasoning, add gzip and
  ETag). Rejected: the dead weight is a multiplier on a request that rebuilds the whole Run, so
  removing it lowers bytes, not the server's cost; an ETag helps an unchanged re-read while every
  page and every expanded step still pays a merge.
- **Cache the rendered page JSON instead of the merged index.** Rejected: one read has four entry
  points — head, older page, delta and search — plus step detail, and the marks and totals must be
  the Run's whatever page is asked for. The merge is the common input of all of them, so that is
  what is kept.
- **Make the derived index durable** (a cache-aside table keyed by `(run_id, version)`, or a
  projection written on entry append). Rejected for now: it adds a migration, retention, and a
  second durable record of facts the canonical tables own (ADR 0069), and a write-path projection
  would touch the Run's lease/fencing hot path (ADR 0007). Named as a later option with a trigger:
  revisit when reads run in more than one process or Runs reach thousands of steps, where a
  per-process rebuild becomes the remaining cost.
- **Use `updated_at` or the progress version alone as the cache key.** Rejected: the Run's lease
  heartbeat renews `updated_at`, so it would invalidate on every beat; the progress version does
  not move when a Child joins the roster, and a recovery write can append without moving it. The
  composite token costs one bounded read and makes a stale index impossible.
- **Push timeline steps over the durable event stream** (SSE) instead of a versioned GET.
  Rejected: the durable event payload does not carry the step's agent identity or kind, so the
  client would have to re-derive merge order, marks, excerpts and totals — duplicating server
  rules and the reader content gate on the reader's side. The versioned GET reuses the read that
  already decides those facts.
- **Merge the two read grains** ([ADR 0067](0067-the-inspector-keeps-two-read-grains.md)'s full
  merge) so one surface pays once. Rejected: its trigger is reading-behaviour evidence, not
  performance, and a merged surface would put message bodies on the index page — a larger payload
  for the common read, the opposite of this decision. This ADR leaves that trigger where it was.
- **Keep the reasoning on the page but cap it harder.** Rejected: it is still bytes every page
  reader pays for and the list never uses, and ADR 0067 already decided the expansion read is where
  a step's content belongs.

## Consequences

- A page read pays for the page and the index it is cut from; an opened step pays for one step.
  The first read after a Run changes rebuilds once (one merge and one protected-set check); every
  page of the same version is a lookup.
- A live watch moves kilobytes rather than megabytes: a delta carries only the steps that landed,
  and the 15-second full read is the only whole-page transfer.
- The Web wire changes: `steps[].reasoning` is gone from the timeline page and the client schema;
  `version`, `next_since`, `reset` and `since` are the follow contract; the step-detail response is
  unchanged. The field was Web-only — a REST transcript or a public Child observation never carried
  it — and [`docs/interfaces.md`](../interfaces.md) states the row.
- [`docs/security.md`](../security.md) states the trace's reasoning by its two read surfaces: the
  activity page, and the timeline through the on-demand step read. Both keep the 4000-character
  bound and the reader content gate.
- Masking stays a display action ([ADR 0061](0061-a-protected-skills-text-is-masked-at-the-readers-edge.md)):
  the index holds derived structure, not a durable masked copy, and its bound is a memory bound.
- The index is process-local. The API runs one uvicorn worker today; a deployment with several
  gives each worker its own copy — bounded, disposable and correct, but each rebuilds and the
  memory multiplies. Nothing in the wire contract depends on a shared cache.
- A delta does not re-state old steps, so a retroactive mark lands only through the 15-second head
  read. Dropping that read would need a marks patch, which is not decided here.
- An ETag hit still costs a round trip; it saves the body, not the network.
- ADR 0065 continues to hold for the live reasoning stream and the activity page; its timeline-page
  sentence is amended. ADR 0067 continues to hold: two grains, a step opens in place, and the
  step-detail read — now O(1) — is the only route to reasoning from the spine, which makes its
  contract more load-bearing, not less. Its full merge stays a named later option with its original
  trigger — reading-behaviour evidence, not performance.
- The reader content gate's warm cost for a page is ~0.3 ms instead of ~107 ms, and two concurrent
  Inspector reads no longer serialize the mask work on the event loop.
- Absolute latency did not move into this decision: the measured host stalls were larger than the
  read path. Making reads cheap and rare is what this does; isolating the read path from the Run's
  event loop remains a deployment question.
