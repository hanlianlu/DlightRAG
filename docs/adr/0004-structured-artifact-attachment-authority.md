# Structured Artifact attachments authorize publication

Research publication authority is a settled parent-only `attach_artifact` tool call, not model-authored `artifact:` links. Each Root Artifact Attachment binds a normalized Workspace path to its raw-content digest and label; terminal publication fails closed if it is stale, then includes the valid root's safe Markdown/HTML dependency closure. `artifact:` remains readable placement syntax, and the Host appends omitted roots in attachment settlement order while leaving dependencies unplaced.

## Considered Options

- Treat final-answer links as authority — compact, but asks the Host to infer control state from prose and makes accidental links publish files.
- Return opaque attachment handles — strongest indirection, but forces files to know IDs before they can express natural relative dependencies.
- Use relative-path receipts — selected because authorization remains structured while Markdown/HTML dependencies stay authorable before attachment.

## Consequences

Fast and Child Sessions do not receive publication authority. Attachment settlement must be durable and atomic with the model-visible tool result; final publication remains a fenced Host transaction. Editing an attached root requires reattachment, and citation validation still runs independently inside every published Markdown Artifact.
