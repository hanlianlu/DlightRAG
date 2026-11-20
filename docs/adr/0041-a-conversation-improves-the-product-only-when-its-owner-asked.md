# A conversation improves the product only when its owner asked

A conversation is used to evaluate and train models only when its owner asked
before it began, and still asks. The permission, Conversation Contribution, is
off until they turn it on, and it is one timestamp: turning it off ends it at
once, and turning it on again begins a new one that covers only the
conversations started after. It is stored and shown in Settings, and the
operator's trace export reads it.

## Status

Accepted; partly implemented. The setting is stored, served at
`/web/api/privacy/settings`, shown on the Privacy page of Settings, and honoured
by the operator's trace export, `scripts/export_traces.py`: it reads only Runs
of owners who contribute and only Sessions that began at or after the
permission, and it has no option that includes more. Its de-identification and
its packaging into a dataset are
[ADR 0044](0044-the-training-dataset-is-a-full-re-export-of-whole-consented-sessions.md).
Still to do: an archive that keeps a contributed conversation after its owner
deletes it.

It is separate from
[ADR 0039](0039-observability-is-always-on-and-the-seam-masks-private-data.md),
which stands as written, and this switch does not touch monitoring.

## Context

A deployment can serve several people, each their own owner
([ADR 0031](0031-a-workspace-creator-holds-it.md)). What the product keeps of an
owner's conversations is what evaluation and training want: the agent's
reasoning and tool calls, the Evidence it read, and the judgements the owner
gave an answer with their reasons. It is the owner's content, and whether it may
be used for that is theirs to say, not the operator's and not a default's.

A Run does not stand alone. It carries the earlier text of its Agent Session as
its context, so what one Run holds can be older than the Run. A conversation is
one Session, and everything forked or followed up from it stays in that Session,
on the Web or over REST or MCP: a follow-up continues the Session of the Run it
follows, and a fork opens a new Lane in the same one. A Run submitted without a
parent starts a new Session. The Session, not the Run and not the Web
conversation row, is what carries the earlier text.

Two neighbours look like this and are not:

- **Monitoring** is what keeps the service running, and it has no off state. Its
  traces keep conversation content: the Run's question (up to 2000 characters),
  its answer, each message's text (up to 4000 characters) and the Tools'
  summaries, with only secrets, bytes and images masked
  ([ADR 0039](0039-observability-is-always-on-and-the-seam-masks-private-data.md)).
  This switch changes none of that. "Off" is a promise about what DlightRAG's
  own export does, not about the terms of the model providers a deployment
  calls.
- **The `answer_feedback` Runtime Policy** decides whether an owner is *offered*
  the thumbs, and an administrator edits it. This decides whether what the owner
  gave, and the conversation it was given on, may be *used*, and only the owner
  decides.

## Decision

- **Off until the owner turns it on.** An owner with no stored setting has never
  been asked, and that reads as off everywhere. Nothing is assumed on anyone's
  behalf.
- **The permission is a timestamp.** `contribute_since` is when the current
  permission began, and it is null exactly when the switch is off; the table's
  check refuses a row where the two disagree. One atomic statement sets it from
  the database's clock when the switch turns on after being off or absent, keeps
  it when it is turned on again, and clears it when it turns off, so two tabs or
  a retry cannot move it. The clock is read as the statement writes and not as
  its transaction began, so a turn-on that waited behind a turn-off is dated
  after that turn-off committed. A write is made once and never replayed: a
  replay arriving after a later turn-off would give back a permission the owner
  had withdrawn.
- **The unit is the Agent Session.** A Run may be used only if its owner
  contributes now **and** the Run's Agent Session began at or after
  `contribute_since`. A Session begins at the earlier of two times: its own
  row's creation, `dlightrag_agent_sessions.created_at`, and the acceptance of
  its first remaining Run, `min(dlightrag_runs.created_at)` over the Runs
  whose `dlightrag_answer_run_routing.agent_session_id` is that Session. The
  row's time lasts as long as any Run of the Session does, so retention pruning
  a long Session's oldest Runs cannot move the start later; the Run's time
  covers the wait before the row is written, while a Run is queued to be
  claimed; and taking the earlier lets neither move it. All of them are on the
  database's clock. `contribute_since` is read as its statement writes, so it
  can come later than `NOW()` would have dated it and never earlier, and later
  covers less. The Run's Session is the one its routing row names, never a
  Child's. The owner's word for a Session stays "conversation", because a
  conversation and everything forked or followed up from it share one. The first
  half of the rule makes turning the switch off stop everything at once. The
  second keeps the permission from covering a conversation begun earlier, even
  one that goes on or is forked afterwards, and turning it off and on again
  covers neither the conversations begun in between nor those begun before. What
  a covered conversation itself brings in goes with it, as the Consequences say.
- **The owner's own state.** It is kept under the owner id and no access action
  guards it, like Memory and answer feedback: the actions guard the corpus and
  the deployment's own settings, not a person's own state. Settings is its only
  surface, so there is no REST or MCP route. `PUT` states the value the owner
  wants rather than a toggle, so a page that is out of date cannot flip it. In
  `none` and `simple` authentication every caller is the same owner, so one
  switch covers the whole deployment, REST and MCP clients included.
- **What it does not do yet.** The only export is the operator's host tool,
  which reads what the rule above allows and nothing else, and which can
  de-identify its records and pack them into a dataset
  ([ADR 0044](0044-the-training-dataset-is-a-full-re-export-of-whole-consented-sessions.md)).
  There is no archive of deleted conversations. Deleting a conversation deletes
  its Runs, and this setting does not change that; the export reads nothing of a
  Run that is gone, though the entries it wrote stay in a Session shared with its
  forks until that Session is deleted, and a surviving fork's record carries them
  as the context it ran on. An archive that would keep a contributed
  conversation after its owner deletes it is a separate decision, and the text
  shown to the owner will say so when it arrives. Until then the page says only
  what is true today.

## Considered options

- **On by default, with an opt-out.** Rejected. An owner's first conversations
  would be used before they had seen the question, and a default is not consent.
- **The unit is the Run.** Rejected, because a Run carries its Session's earlier
  text. A Run created after the permission began, in a Session begun before it,
  would put text the owner never covered into a training record.
- **The unit is the Web conversation.** Rejected. A fork on the Web opens a new
  conversation row, and a follow-up or fork over REST or MCP opens none, while
  all of them carry the parent Session's earlier text; dated by the row, or by
  their own creation, they would be covered although what they carry predates
  the permission.
- **Date a Session by its own row alone.** Rejected.
  `dlightrag_agent_sessions.created_at` is written when the Session's first
  Run first commits to it, after that Run has waited to be claimed, so a
  question submitted before the permission began and started after it would
  count as after.
- **Date a Session by its first remaining Run alone.** Rejected. Retention
  deletes a long Session's oldest Runs one by one while the Session lives on in
  its newer ones, so the oldest Run left drifts later and a Session begun before
  the permission could read as begun after it.
- **Stamp each Session, or each Run, when it starts.** Rejected. It writes the
  owner's choice a second time, on the path every Session takes, and it freezes
  the choice at the start. Turning the switch off would then reach the stamped
  ones only if the export also read the current setting, which is the timestamp
  rule with a copy of the choice beside it. And turning it on again would revive
  every Session stamped in an earlier window, which the timestamp rule leaves
  behind.
- **An epoch**, a counter bumped when the switch goes off, as Memory's
  capability keeps. Rejected. It needs the same stamp, and a counter cannot say
  *when*, so "began at or after" would need a second record to translate it.
- **A switch for each surface.** Rejected. The owner answers one question;
  separate switches would each need their own wording, could disagree, and would
  leave the export to take their intersection.
- **A history of every grant and withdrawal.** Rejected for now. Turning the
  switch off withdraws every earlier window, so nothing needs the past ones, and
  a log of when someone consented is itself a record to protect.
- **Fold it into the feedback policy.** Rejected. That policy is an
  administrator's, about what is offered, and this is the owner's, about what is
  used.
- **Strip Profile Memory bodies from the export.** Rejected, because the answer
  text would carry the same content anyway and the trace would lose the
  reasoning that used it.

## Consequences

- A conversation begun before the permission is never covered, however long it
  goes on and however it is forked or followed up. One begun while the switch
  was off is not covered by turning it on later, and one that was covered stops
  being covered the moment the switch goes off or begins a new permission. A
  Session is covered or not as a whole, and its Runs with it.
- A covered conversation is covered with what it brings in: an attachment, and
  the history a REST caller sends, in the `history` field, with a request made
  after the permission began, since the owner chose to send them then.
- A Profile Memory the agent recalls in a covered conversation is part of that
  conversation too: `recall_memory` returns its body as a Tool result, and the
  receipts of `remember` and `forget` carry it, so the export writes it and no
  filter removes it.
- Turning the switch off stops every later export of that owner's conversations.
  It does not recall a file an earlier export wrote.
- `scripts/export_traces.py`, the operator's host tool, honours the setting with
  default deny. Its selection query joins each Run to its owner's setting and to
  the start of its Session, so a Run of an owner who never chose, or of a
  Session that began before the permission, is never read, and no option
  includes it. A Run with no Session, a Retrieval or a Corpus Mutation, is not a
  conversation and is not exported. A Run's record holds what that Run wrote:
  every Session entry states the Run that committed it, and the export takes a
  Run's steps by that, never by time or by the Runs beside it. A fork shares its
  parent's Session, and deleting a conversation removes its Runs and leaves the
  entries they wrote in a Session another Run still routes to, so a Run's steps
  read off the clock would carry the words of a conversation its owner deleted.
  For a development database the owner turns the switch on for their own
  account.
- One case still reads late, a Session whose first Run was pruned and whose row
  was written after the permission began, and it is practically unreachable: it
  needs the permission turned on while that Run waited to be claimed and the
  Session to outlive that Run's retention.
- The setting adds one table, `dlightrag_privacy_settings`, in the Run store's
  schema, with one more migration appended. A database made before it gains the
  table at the next writer start.
- Tests hold the rule: an owner with no row reads as off; the moment is the
  database's, a second turn-on keeps it, a turn-off clears it, a new turn-on
  makes it later, and a turn-on that waited behind a turn-off is dated after it;
  writers at once, and a writer that waits on another, decide against what the
  other left; the table refuses a flag and a moment that disagree; a write is
  not tried again after a lost connection; a malformed body or a write the page
  did not make changes nothing; and the trace export reads one snapshot and only
  the Sessions that began at or after the permission, one whose oldest Runs
  retention pruned included, whatever options it is given, and a Run's record
  holds exactly the entries that Run wrote, in the order of their sequence, so
  the words of a fork its owner deleted are in no record and Runs that take
  turns in one Session each keep their own.
