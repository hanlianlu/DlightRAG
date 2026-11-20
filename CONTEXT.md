# DlightRAG

DlightRAG answers research questions with Runs that load packaged Agent Skills and publish the
files they produce; the platform derives from a Run's own records what its Skills asked it to do.
The terms below are the ones this product uses for that pipeline.

## Language

**agentspace**:
The Agent's model-visible files for one conversation, carried from one Run to the next; each settled
Run leaves a restorable snapshot of it, and the word never means a corpus workspace.
_Avoid_: Agent Workspace, agent workspace, Agent Space, agent-space, working directory, working dir, sandbox, workspace

**starter**:
A tested artifact copied verbatim into the Agentspace for the Agent to edit in place; its
structure is never retyped.
_Avoid_: template, scaffold, boilerplate

**in-loop gate**:
A check the Agent runs inside its Run before delivery that can fail and be corrected while the Run
is still going.
_Avoid_: preflight, validation step, delivery gate

**delivery record preview**:
The platform's mirror of the reader-facing delivery record, returned to the Agent in a Tool result
while the Run can still act on it.
_Avoid_: preflight warning, attestation badge

**pre-delivery checkpoint**:
An advisory check the harness runs when the Agent proposes a final response: it compares the Run's
delivery record with the Run's own logs and, when the record has a gap, injects those findings as a
warning so the Agent can correct the answer while the Run is still going. A delivery that loaded no
Skill is one such gap: the reader's record will publish its Artifacts unattested, and the check says
so before the Run settles. It is not a gate, and it is not a registered extension point.
_Avoid_: delivery gate, checkpoint provider, hook, callback, plugin

**gate card**:
A short whole-file document a Skill declares as required reading; it carries one step's gates and
points into the deep reference that stays lookup-only.
_Avoid_: checklist, summary, cheat sheet

**required read**:
A declared, whole-document read by `load_skill`; only coverage from the first character to the last
satisfies it, and a `write_to` copy never does.
_Avoid_: required reading, mandatory document

**goal**:
An optional, durable objective a Session carries across Rounds; the harness decides when another
Round starts, and the Agent decides when the objective is complete. A process takes up the goals
still active when it starts, so a restart continues the work rather than parking it behind a
Resume button.
_Avoid_: task, todo, plan

**goal round**:
The continuation Run the harness starts for an active goal, carrying the objective and its number.
_Avoid_: auto-continue, retry, loop

**goal bar**:
The conversation surface that shows a goal's phase, rounds, objective and blocked reason, and offers
pause, resume, edit and clear.
_Avoid_: goal widget, status chip

**goal command**:
A `/goal` message the reader writes in the composer: it changes the conversation's goal without
being sent to the model as a turn.
_Avoid_: slash command, admin action

**run activity**:
The reader's recent conversation-linked Answer Runs read as one bounded page: which conversations
are still running, and which finished since the reader last opened them. The page derives running
and unread from it; the server keeps no read state and creates no notification record.
_Avoid_: task list, queue, notification record
