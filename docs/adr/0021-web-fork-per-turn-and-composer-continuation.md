# Web turns offer Fork, and the composer is the continuation path

The per-turn Follow-Up control is gone from the Web. Fork stays on every settled
turn, and continuing a conversation is what the composer does.

## Status

Accepted and implemented. Revises the control-placement clause of
[ADR 0019](0019-turn-accurate-forking-and-the-carry-point.md), which said the
per-turn Follow-Up control was wrong and that only the tip offers one.

## Context

ADR 0019 decided that Fork is a tree operation and Follow-Up is a line operation:
a Follow-Up appends to the selected Lane's tip, nothing can be inserted into the
middle of a Lane, and therefore "the per-turn Fork control is correct as a product
affordance and the per-turn Follow-Up control is not: only the tip offers
Follow-Up." Neither the gate nor its removal landed, and the browser kept
rendering both controls on every terminal turn.

That reproduced the divergence the decision exists to remove. Selecting an earlier
turn and following up there continued at the Lane tip: the transcript came from
the Lane's current ancestry, while the clicked Run supplied only its lineage, its
carried Run Notes, and its inherited retrieval parameters. The turn the control
named was not the turn the answer continued from.

The composer already appends to the Lane tip, which is what a Follow-Up is, so
gating the control to the tip would leave two controls for one operation, one of
them behind a dialog.

## Decision

**The Web offers no per-turn Follow-Up control.** Fork stays on every settled
turn: choosing the divergence point is its scenario, and it branches from the
state that turn settled at. A conversation continues through the composer, which
appends to the Lane tip with no dialog in front of it.

**The browser client has no Follow-Up path.** `forkAnswerRun` is the browser
client's only continuation call, the dialog has no second mode, and the turn
action, its message strings, and the tip gate ADR 0019's remedy implied are
removed rather than left unreachable. `POST /web/api/answer/{run_id}/follow-up`
stays a documented browser route, and the REST and MCP surfaces keep their own
endpoints: those callers name a Run explicitly and their contracts state what the
call appends to.

## Consequences

A Web turn submitted through the composer records no `parent_run_id`, so it
carries no Run Notes and records no continuation lineage. The removed control was
the only path that named a parent for a turn submitted into the same conversation;
a Fork still carries the notes of the Run it branches from into the new
conversation, and non-browser callers can still follow up. If the Web needs that
carry on the one control it has left, the composer should name the conversation's
tip Run as its parent — that changes the durable acceptance path, so it is its own
decision rather than a UI affordance.

## Considered options

- **Gate Follow-Up to the Lane tip.** Rejected: the composer *is* the tip's
  Follow-Up, so the control would duplicate it behind a dialog, and that
  duplication is part of what let the divergence survive review.
- **Keep Follow-Up on every turn and let it continue at the tip.** Rejected: the
  control names a turn it does not continue from, which is the untruth ADR 0019
  set out to remove.
- **Make an earlier turn's Follow-Up branch from that turn.** Rejected: that is
  Fork, and nothing can be inserted into the middle of a Lane.
- **Remove the browser route as well.** Rejected: it is a browser contract for a
  client that names a Run explicitly, not a compatibility alias for a replaced
  shape, and deleting it would remove the only Web-side path that carries Run
  Notes before deciding where that carry belongs.
