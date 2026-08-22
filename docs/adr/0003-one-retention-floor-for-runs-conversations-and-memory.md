# One retention floor for runs, conversations, and memory

Durable state previously expired on three separate clocks: terminal Answer runs
and event logs at 30 days (with a conversation-linked exemption), Web
conversations on a 30-day inactivity TTL, and superseded Profile memory at 30
days. Physical storage is cheap while durable session history is valuable, so
we replaced all three with one deployment knob:
`answer.runtime.answer_run_retention_days`
(default 365), a best-effort floor counted from `finished_at`.

- Terminal runs prune after the floor regardless of conversation linkage; the
  turn cascade empties the conversation and an hourly sweep reclaims
  conversation rows with no turns left.
- Event logs trim on the same clock.
- Superseded Memory history purges after the same span. Active Memory Records
  never expire on a timer — growth is bounded by supersede folding, explicit
  forget/clear, and deployment storage quota, not auto-decay (peer products
  MemMachine and MemoraX ship no active-memory TTL either).
- The conversation `max_turns` trim that deleted old turns along with their
  runs is gone: the snapshot window is now a read bound only, and turn
  durability belongs to the retention clock, not a per-conversation cap.

## Considered options

- **Keep three clocks.** Rejected: overlapping expiry semantics made the
  conversation-linked exemption necessary and silently destroyed old turns of
  active conversations.
- **Auto-decay active memory.** Rejected: a still-true standing assertion
  vanishing after a year is data loss for a memory product; retrieval relevance
  (not time) should sink stale records.
- **Retention floor vs deadline.** Chose floor: the hourly sweep may reclaim
  later, never earlier.
