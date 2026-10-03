---
name: council
description: Use when the user asks for a judgment, recommendation, or go/no-go call on something with real stakes (pricing, an acquisition, compliance or legal exposure, contract terms, a strategy choice) where the knowledge base may hold evidence on both sides; when sources disagree and must be reconciled; or when the user asks for an independent review, second opinion, or red-team critique. Runs two or three independent Child Sessions plus at most one cross-examination round. Skip lookups, summaries, and simple factual questions. User veto, cancellation, and scope constraints win.
---

# Council

A recipe, not a runtime: it grants no tools or permissions. You run it with ordinary children.

1. Tell the user in a sentence why independent scrutiny helps, then call `spawn_agent` with two or three children whose objectives are different and concrete, for example the strongest case for, the strongest case against, and a check of the key evidence. Pass `tools` for each child: every tool you have except `bash`, `write`, `edit` and `attach_artifact`, so they investigate and change nothing. Keep working on your own meanwhile.
2. Read the children with `wait_subagent`. One failed child does not stop the others; decide whether to replace it.
3. If a material dispute remains that child evidence could settle, send one focused challenge with `continue_subagent` to the same children. Skip this when nothing material is disputed.
4. Write the answer from what the children found: your recommendation, which claims you accepted or rejected and why, and the dissent that remains. Cancel children you no longer need.

The user's veto, cancellation or narrowed scope ends the recipe at once. Children cannot spawn further children.
