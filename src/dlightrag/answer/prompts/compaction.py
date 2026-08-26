# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The compaction summarizer prompt: typed continuation memory, markdown out.

The summarizer writes one structured markdown document whose headings map
onto :class:`~dlightrag.agent.session.projection.CompactionSummary` fields.
The framework parses the headings back into the typed summary; the model is
never asked to invent paths or durable handles — those are extracted from the
covered branch-ancestry prefix by the framework. Prompts modules stay import-free:
the caller passes the pre-rendered previous summary text.
"""

COMPACTION_SYSTEM_PROMPT = """\
You are a context summarization assistant. You read one research transcript \
and write a structured continuation summary that another instance of the same \
agent will use to keep working after the transcript is removed.

Do NOT continue the conversation. Do NOT answer any question that appears in \
the transcript. Output ONLY the structured summary in this exact format:

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

Rules:
- Write in the same language the research has been using.
- Preserve exact paths, URLs, resource ids, and error messages.
- Be concise but complete: the summary replaces the transcript, so include \
every fact the next turn cannot cheaply re-derive.
"""


def compaction_user_prompt(*, previous_summary: str | None, transcript: str) -> str:
    """Build the summarizer user turn with the covered-prefix transcript.

    ``previous_summary`` is the caller-rendered earlier summary text.
    """
    blocks = [
        "The transcript below is the research so far. It will be removed "
        "from context. Write the continuation summary.",
    ]
    if previous_summary is not None:
        blocks.append(
            "An earlier compaction already summarized older work. Preserve its "
            "information while adding what the new transcript adds:\n\n"
            "<previous-summary>\n"
            f"{previous_summary}\n"
            "</previous-summary>"
        )
    blocks.append(f"<transcript>\n{transcript}\n</transcript>")
    return "\n\n".join(blocks)


__all__ = ["COMPACTION_SYSTEM_PROMPT", "compaction_user_prompt"]
