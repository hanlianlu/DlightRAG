# Structured Artifact attachments authorize optional publication

The Answer is the default user-facing deliverable. A Published Artifact is an optional separate reading, presentation, or download surface—not a parallel Answer and not a routine by-product of Research. The Agent creates one only when the user explicitly requests a file, report, export, or separate presentation; when the complete deliverable is too long or structurally rich for one practical Answer; or when a separate visual, interactive, or downloadable surface materially improves use.

When a Published Artifact carries the complete deliverable, the Answer provides a concise orientation, key takeaways, and access to the Artifact instead of reproducing substantial portions of it. Deliberate duplication remains valid when the user explicitly requests both inline and file versions. Independent citation validation governs the evidentiary support of each surface; it does not require duplicated prose.

Publication authority is a settled parent-only `attach_artifact` tool call, not model-authored `artifact:` links. Each Root Artifact Attachment binds a normalized Agent Workspace path to its raw-content digest and label. The attachment receipt returns an `artifact:` URI containing the stable resource id. Relative `artifact:` paths remain authorable inside documents before attachment. The Host appends omitted roots in attachment settlement order while leaving dependencies unplaced. If incomplete Markdown would hide those trailing affordances, the Host places them before the original Answer without rewriting its source.

The selection policy belongs primarily to the capability-gated Agent prompt because the Agent must decide whether a separate deliverable is warranted before it writes or attaches a file. The `attach_artifact` Tool Interface repeats a local reminder, but it does not own the policy and the Host does not reject semantically similar outputs: intentional inline/file duplication is a legitimate request.

## Considered Options

- Create an Artifact whenever Research has a Workspace — rejected because tool availability is not user intent and produces redundant parallel Answers.
- Put the selection policy only in `attach_artifact` — rejected because the decision arrives after the Agent has usually written the file.
- Reject Answer/Artifact similarity in the Host — rejected because similarity is not publication authority and explicit dual-format requests may legitimately duplicate content.
- Treat final-answer links as authority — compact, but asks the Host to infer control state from prose and makes accidental links publish files.
- Require opaque handles everywhere — rejected because documents must be able to express natural relative dependencies before attachment.
- Return stable resource-id receipts and resolve relative document dependencies at publication — selected because attachment identity is explicit while Markdown/HTML dependencies stay authorable before attachment.

## Consequences

Fast and Child Sessions do not receive publication authority. Attachment settlement must be durable and atomic with the model-visible tool result; final publication remains a fenced Host transaction. Editing an attached root requires reattachment, and citation validation still runs independently inside every published Markdown Artifact.

The shared Answer grammar identifies actual links and images, excluding code examples, math, and unused reference definitions. Answer and Markdown Artifact links that cannot be resolved as Artifact, Evidence, supported external, or embedded resources are invalid references, including workspace paths and unknown schemes. These references enter the existing bounded publication correction pass, where the Agent must attach the file and use the returned URI. Detection does not authorize publication. If correction still fails, a document-scoped binding places the unavailable resource at the actual reference occurrence.

Artifact reference resolution preserves authored Markdown and HTML instead of replacing source text. Existing citation cleanup and public-source projection prepare Markdown publication bytes once, after the root attachment's original content digest has been checked and before dependency discovery, binding, and admission budgets. If citation preparation changes actual Artifact or Evidence reference identities, publication requires correction instead of attaching bindings to a different document. Staging stores the validated bytes without running citation preparation again.

Each document carries an `artifact_bindings` map from its parser-normalized target to an available or unavailable resource id. The Answer has its own map; every published text Artifact carries its own map in the descriptor. This scope allows identical relative targets in different document directories to resolve independently. Repeated occurrences retain their own labels and image/link presentation while sharing one resolution. Consumers parse the complete document and resolve only actual reference tokens.

Only structured roots and their `artifact:` dependency closure can publish files. HTML dependency discovery uses actual `href` and `src` attributes; comments and script/style source examples are inert. Existing self-contained HTML and media validation still apply. Resolution metadata does not change source bytes or their content digest. Bindings are additional result metadata and require no database schema migration.

Prompt tests preserve the default-Answer, selective-Artifact, and non-duplication guidance. Tool tests preserve the local optionality reminder. Semantic similarity remains an observational quality signal rather than a hard runtime invariant.
