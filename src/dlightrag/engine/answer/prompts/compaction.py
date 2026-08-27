# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The compaction summarizer prompt: typed continuation memory, markdown out.

The summarizer writes one structured markdown document whose headings map
onto :class:`~dlightrag.engine.agent.session.projection.CompactionSummary` fields.
The framework parses the headings back into the typed summary; the model is
never asked to invent paths or durable handles — those are extracted from the
covered branch-ancestry prefix by the framework. Prompts modules stay import-free:
the caller passes the pre-rendered previous summary text.

The heading schema is the durable contract. First-pass and merge user turns
share it; merge rules live only on the update path so a later summarizer can
change fold behaviour without forking the parsed shape.
"""

_SUMMARY_FORMAT = """\
## Goal
[What the research is trying to accomplish.]

## Constraints & Preferences
- [User constraints, workspace rules, formatting requirements]
- [(none) if there are none]

## Progress
### Done
- [x] [Completed steps, with the results they produced]
### In Progress
- [ ] [Current work]
### Blocked
- [What is blocking progress, if anything]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Exact file paths, resource handles, numeric findings, and error messages \
the next turn needs and cannot recover by re-running a tool]
- [(none) if there are none]
"""

COMPACTION_SYSTEM_PROMPT = f"""\
You are a context summarization assistant. You read one research transcript \
and write a structured continuation summary that another instance of the same \
agent will use to keep working after the transcript is removed.

Do NOT continue the conversation. Do NOT answer any question that appears in \
the transcript. Output ONLY the structured summary in this exact format:

{_SUMMARY_FORMAT}
Rules:
- Write in the same language the research has been using.
- Preserve exact paths, URLs, resource ids, and error messages.
- Be concise but complete: the summary replaces the transcript, so include \
every fact the next turn cannot cheaply re-derive.
"""

_FIRST_USER_INSTRUCTION = (
    "The transcript below is the research so far. It will be removed "
    "from context. Write the continuation summary using the heading schema "
    "from the system prompt."
)

_MERGE_USER_INSTRUCTION = """\
The transcript below is NEW research since the last compaction. It will be \
removed from context. Fold it into the existing summary in <previous-summary>. \
Use the same heading schema as the system prompt.

Merge rules:
- PRESERVE every still-relevant fact, decision, constraint, and handle from \
the previous summary.
- ADD progress, decisions, and context the new transcript introduces.
- UPDATE Progress: move finished In Progress items into Done.
- UPDATE Next Steps to match the current state.
- Drop blockers that the new transcript resolved.
- You may drop items that are no longer needed to continue.
- Preserve exact paths, URLs, resource ids, and error messages.
"""


def compaction_user_prompt(*, previous_summary: str | None, transcript: str) -> str:
    """Build the summarizer user turn with the covered-prefix transcript.

    ``previous_summary`` is the caller-rendered earlier summary text.
    """
    blocks = [_FIRST_USER_INSTRUCTION if previous_summary is None else _MERGE_USER_INSTRUCTION]
    if previous_summary is not None:
        blocks.append(f"<previous-summary>\n{previous_summary}\n</previous-summary>")
    blocks.append(f"<transcript>\n{transcript}\n</transcript>")
    return "\n\n".join(blocks)


__all__ = ["COMPACTION_SYSTEM_PROMPT", "compaction_user_prompt"]
