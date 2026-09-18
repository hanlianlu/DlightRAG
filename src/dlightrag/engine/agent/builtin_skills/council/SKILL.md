---
name: council
description: Council recipe for independent Child Session investigations and one curated cross-examination. Load when independent scrutiny would materially improve a contested, high-stakes, or multi-source answer, or when the user asks for independent critique. Skip ordinary factual or trivial questions. User veto, cancellation, and scope constraints win.
---

# Council

Independent Child Session investigations plus one curated cross-examination. This Skill is a recipe over ordinary children. It creates no Council entity, Run, table, or approval action. Loading it grants no tools, permissions, or host capabilities.

## When to load

Load and follow this Skill when independent scrutiny would materially improve the answer, or when the user asked for independent critique. Ordinary factual or trivial questions stay with the parent.

Stop as soon as the user opts out, cancels, or narrows scope. User veto wins over this recipe and over an explicit `/skill:council` request. Explicit Skill selection is a convenience path, not a permission gate.

## Announce, then proceed

State in one or two sentences why independent scrutiny helps, then dispatch. Do not wait for confirmation.

## Bound

Declare the discussion plan before spawning: independent first-pass children, then at most one focused cross-examination round on those same Child Sessions. Do not force agreement or add polish rounds.

Completion: the declared plan is written, first-pass children are accepted, optional cross-examination (if used) has finished or been cancelled, and the parent is ready to synthesize.

## Steps

1. **Select investigations.** Choose two or three independent objectives that can change the answer. One child failure does not abort siblings; inspect the failure and decide whether to continue, replace the path, or cancel.
   Completion: each objective is unique, concrete, and inside the user's scope.

2. **First pass, read-only.** Call `spawn_agent` and continue useful parent work. **Pass an explicit `tools` list** — `search_knowledge_base`, `search_web`, `read`, `view`, `grep`, `find`, `ls`, `recall_memory`, `load_skill` — because a child with no `tools` runs with the parent's whole capability, `bash` and `write` included. Never list `attach_artifact`, and list `bash` or `edit` only when the investigation genuinely needs them. Host permission ceilings remain the real enforcement; this Skill cannot widen them.
   Completion: `spawn_agent` returned stable `child_session_id` handles without waiting for every child to finish.

3. **Optional cross-examination.** If a material dispute remains that child Evidence could settle, call `continue_subagent` on the original Child Sessions with a curated challenge packet. Do not dump peer transcripts. Do not spawn replacement children for a user-cancelled objective unless the user explicitly reauthorized that work.
   Completion: each continued child has a new Operation on the same Session, or the round was skipped because no material dispute remained.

4. **Parent synthesis.** Write the answer from accepted child Evidence. Admit Evidence separately from endorsing conclusions. Retain meaningful dissent and uncertainty. Important missing evidence stays visible.
   Completion: the parent answer states the recommendation, accepted and rejected claims with reasons, and remaining dissent.

## Hard limits

- Delegation is one level: children cannot spawn grandchildren or control siblings.
- A child inherits its parent's capability except the Run's authority (roster controls, durable owner memory, publication); read-only deliberation is therefore requested explicitly, by listing the tools.
- Steer, status, and recovery never revive a terminal or user-cancelled child.
- Waiting for `ask_parent` guidance is the child's correlated request path; completion results use the normal child result path.
- Cancel outstanding children you no longer need before parent finalization.
