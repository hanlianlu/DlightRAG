# A Run's children are watched in the Inspector

The Web shows a Run's Child Sessions in a non-modal dock in the Inspector, beside
Sources and Files, and a stored turn records that it had children. Watching live
work does not take over the page that shows it, and a settled Run's children stay
reachable.

## Status

Accepted and implemented.

## Context

The Subagent Roster was a modal dialog opened by a text button under a live turn.
Four things were wrong with that, all of them visible to a person using it:

- **It blocked what it watched.** The dialog followed a Run that was still
  streaming, and its modality hid the answer being written.
- **It was reachable only while the Run was live.** Whether a turn had children came
  from live events, so a settled or reloaded turn offered no way back to its
  children, their Results or their Evidence, although the roster is durable.
- **It read as a log.** One line per child, then a stacked transcript of raw tool
  output, then three stacked forms.
- **It made a question look like the owner's to answer.** A child's `ask_parent`
  goes to the parent agent and expires after five minutes; the owner may answer
  instead, and usually does not need to.

## Decision

- **One more Inspector kind.** `children` joins `files` and `sources`: the same
  panel, split and phone sheet, closed with the conversation like Sources. It is
  not closed by a click on the chat, because it is a monitor rather than a lookup.
- **The panel's own width chooses the layout.** Narrow, the list and one child's
  detail alternate with a back control; from 40 rem the two sit side by side. The
  measure is the panel's, not the viewport's, since the user resizes the dock.
- **A turn knows it had children.** The stored turn carries `child_count`, so the
  turn offers its children after the Run settles and after a reload. The entry is a
  quiet "Child agents" button with a live dot while the Run runs; it shows no
  counts, which would need a store that fetched a roster for every turn.
- **The roster says when and how many.** A child's status carries `started_at`,
  `finished_at` (null while it runs) and `pending_questions`, from the latest
  Operation and the guidance table. They are additive; no migration.
- **A question is the parent's.** The card reads "Asking the parent" with its
  expiry and offers "Answer instead". It does not use an urgent colour; the product
  has none for it.
- **One composer.** A running child is steered; a settled one is continued, with
  the reauthorization box only for user-cancelled work. This keeps the Run's
  command contract (Idempotency-Key, `operationId`, outcomes) untouched.
- **A finished Run's children are read-only.** The server refuses every command to
  a child whose Run is over, so the Web children route says whether the Run is live
  (`run_status`) and the dock then offers no box and no Cancel, only a note. The
  same field makes the dock correct when the Run ends while it is open.
- **Activity is what the public projection holds.** The transcript projection still
  omits tool arguments, so a step is the tool's verb (the vocabulary the main
  turn's trace uses) and an excerpt of its result. Tokens show only for a settled
  child and only when `total_tokens` is present; usage keys are provider dialects
  and are attributed when an Operation settles.

## Considered options

- **The same modal in two columns:** rejected. It is cheaper, and it still hides the
  answer the children are working for.
- **Child cards inside the chat transcript:** rejected. The transcript is the
  answer; worker detail would grow it without bound and re-render with every tick.
- **Counts in the turn's entry through a shared roster store:** rejected for now. It
  costs a store and a fetch per turn to save one click.
- **Exposing tool arguments in the transcript projection:** rejected. The
  projection is public status only, and nothing here needs more.

## Consequences

- The roster dialog and its stacked forms are gone, and with them the editor
  capture and restore code that existed because the dialog re-rendered over a
  textarea.
- A later surface for live work, such as an Agent Page view, is an Inspector kind
  rather than another dialog.
- The turn's wire gains one integer, and the child status three fields. Both are
  optional for a client that predates them.
