# Structured Artifact attachments authorize optional publication

The Answer is the default user-facing deliverable. A Published Artifact is an optional separate reading, presentation, or download surface—not a parallel Answer and not a routine by-product of Research. The Agent creates one only when the user explicitly requests a file, report, export, or separate presentation; when the complete deliverable is too long or structurally rich for one practical Answer; or when a separate visual, interactive, or downloadable surface materially improves use.

When a Published Artifact carries the complete deliverable, the Answer provides a concise orientation, key takeaways, and access to the Artifact instead of reproducing substantial portions of it. Deliberate duplication remains valid when the user explicitly requests both inline and file versions. Independent citation validation governs the evidentiary support of each surface; it does not require duplicated prose.

Publication authority is a settled parent-only `attach_artifact` tool call, not model-authored `artifact:` links. Each Root Artifact Attachment binds a normalized Agent Workspace path to its raw-content digest and label. `artifact:` remains readable placement syntax, and the Host appends omitted roots in attachment settlement order while leaving dependencies unplaced.

The selection policy belongs primarily to the capability-gated Agent prompt because the Agent must decide whether a separate deliverable is warranted before it writes or attaches a file. The `attach_artifact` Tool Interface repeats a local reminder, but it does not own the policy and the Host does not reject semantically similar outputs: intentional inline/file duplication is a legitimate request.

## Considered Options

- Create an Artifact whenever Research has a Workspace — rejected because tool availability is not user intent and produces redundant parallel Answers.
- Put the selection policy only in `attach_artifact` — rejected because the decision arrives after the Agent has usually written the file.
- Reject Answer/Artifact similarity in the Host — rejected because similarity is not publication authority and explicit dual-format requests may legitimately duplicate content.
- Treat final-answer links as authority — compact, but asks the Host to infer control state from prose and makes accidental links publish files.
- Return opaque attachment handles — strongest indirection, but forces files to know ids before they can express natural relative dependencies.
- Use relative-path receipts — selected because authorization remains structured while Markdown/HTML dependencies stay authorable before attachment.

## Consequences

Fast and Child Sessions do not receive publication authority. Attachment settlement must be durable and atomic with the model-visible tool result; final publication remains a fenced Host transaction. Editing an attached root requires reattachment, and citation validation still runs independently inside every published Markdown Artifact.

Prompt tests preserve the default-Answer, selective-Artifact, and non-duplication guidance. Tool tests preserve the local optionality reminder. Semantic similarity remains an observational quality signal rather than a hard runtime invariant.
