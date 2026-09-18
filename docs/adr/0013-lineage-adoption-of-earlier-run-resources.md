# Lineage adoption of earlier Runs' Resources

A later Answer Run on the same Agent Session may re-materialize a Resource an earlier Run registered, but only through the lineage rule and only under the consuming Run's own fence: bytes are copied forward, the stored conversion snapshot is reused, the handle is minted per Run, and the earlier handle survives as an alias. A Resource Handle is never an authority by itself, and a cursor never crosses a Run.

## Status

Adoption's reach is extended by
[ADR 0023](0023-a-published-product-is-a-resource.md): a Published Artifact registers a
Resource of its Agent Session, so a later turn reads the version the conversation
published. The owner-and-Session admission rule, the alias semantics, and the
filesystem-backed rejection recorded here are unchanged.

Accepted and implemented for the handle-triggered adoption path. `bfc1c5ae` (a reused historical handle fails as a typed refusal) and `c4ee5f5b` (an attachment image may not cut a tool batch) are its landed prerequisites. This ADR narrows nothing already decided: the run-scoped meaning of a Resource Handle, the lineage authorization of attachment replay, and `replay`/never-reparse semantics all stand.

## Context

An earlier turn's tool text carries `[resource: res-…]`, and nothing at that surface says the handle belonged to a Run that has ended. A follow-up turn therefore reuses it — observed in production: `view(resource_id=res-…, locator="3")` and `view(resource_id=res-…, cursor="overview.…")` on a conversation whose Durable history was intact — and the contract answered no, because [the Resource reading contract](../resource-reading.md) states that hydration "does not register historical source handles or cursors as capabilities in the new Run, reparse old sources, or add legacy schema compatibility".

That refusal is correct but incomplete in one direction: the Run *already declares* what it may carry. `SubmissionSeed`/`CarriedAttachment` carries each successful prior turn's attachment metadata to a new Run (`run_id`, `source_ordinal`, digest, filename, mime type, byte size), and session-entry attachment occurrences already adopt tool-produced pixels under the consuming Run's fence. What is missing is materialization: bytes and the adopted conversion snapshot are never re-published as this Run's Resources, so the model cannot read an earlier document or render a page it has not seen.

The alignment principle comes from Pi's durable runtime: identifiers embedded in message text are not references ("the harness never tracks those references and they may go stale. Copy content, don't reference it"), tool identity is the reserved result entry rather than a provider-batch id, a settled tool outcome never re-executes, and context "never reads past a compaction" so compaction is the one deliberate cache invalidation. DlightRAG's equivalent durable identity already exists as the Entry position (`attachment-occurrence:<entry_id>:<part_index>`), and its blob plane plus `dlightrag_answer_resources` rows already carry `session_id`, `intent_id`, `result_ordinal`, digests, and `capabilities.resource_aliases`.

## Decision

A Run adopts its lineage Resources **lazily on first use**, through one Answer-owned rule:

- **Selection.** The trigger is a handle the model names, and admission is the durable row's own authorization: same owner, same Agent Session, adoptable kind, and bytes still retained. A row another owner or another Session registered is never returned, so naming a foreign handle reaches nothing. Selection never parses message text and never trusts the handle itself. Deciding this by Session stamp rather than by re-deriving the declared carries and the retained entry suffix keeps one rule for both the already-declared attachments and the documents a tool produced.
- **Materialization.** Under the consuming Run's fence: stream and verify the origin blob, reuse the stored `conversion_snapshot` (never reparse), register the content as **this Run's** Resource, record the earlier handle in `capabilities.resource_aliases`, and pin the blob reference so origin-Run cleanup cannot invalidate it.
- **Resolution.** `read`/`view` accept this Run's handles and aliases of Resources this Run materialized; anything else keeps failing as the typed refusal that names the rule and the remedies. A cursor remains this Run's projection state and a stale one is refused and re-derived by reading again.
- **Publication.** This Run's Resource manifest lists this Run's own Resources, including anything already adopted. Listing a *not-yet-adopted* lineage-available set is deliberately deferred: the model reaches the adoptable handles through its own context, which is also what keeps the manifest bounded, and surfacing more would need its own bound and its own measurement before it earns a place in a prompt.
- **Accounting.** Adopted bytes charge the consuming Run's aggregate text and image budgets on use; no new budget is opened.

## Considered options

- **Session-stable Resource handles.** Rejected: it promotes an identifier embedded in message text into an authority, which is the opposite of "copy content, don't reference it", and it widens retention and cleanup ownership without a new authorization story.
- **Rewriting replayed history to mark historical handles.** Rejected: it breaks the append-only provider context and duplicates the authority that compaction already owns as the single invalidation point; the manifest and the typed refusal carry that information instead.
- **Eager adoption of every lineage Resource at Run start.** Rejected as the default: it spends blob reads and budget on documents the turn never uses and inflates the manifest. Kept as a configurable alternative if measurement shows lazy adoption is too late.
- **Nothing beyond the typed refusal.** Rejected: it leaves the model unable to work with a document it can still see in its own context, which is the user-visible defect this decision fixes.

## Consequences

A follow-up turn can read an earlier document and render pages that were never rendered, while no earlier Run's capability is silently extended: every resolution re-checks the lineage rule and re-pins bytes under the consuming Run. Compaction naturally shrinks the adoptable set, and `history_attachments` stays bounded by the existing attachment allowance. `resource-reading.md` and the domain-language Resource Handle entry must be revised with the implementation, and the new behavior is owned by configuration (`answer.generation.lineage_adoption`) per ADR 0006.

Adopted Resources count against the consuming Run's caller attachment allowance and aggregate budgets, so adoption cannot outgrow the allowance the Run already has. Each adoption that the Session stamp refuses, or whose stored conversion view is unusable, keeps failing explicitly rather than repairing by re-parsing; bytes that are no longer retained refuse the same way instead of failing the tool internally. Pixels and text differ on purpose: `view` adopts a document the earlier Run never converted, while a text `read` refuses it with the remedy, because converting it here would record a view that Run never had. The stored conversion view keeps the identity the earlier Run wrote for it — recovery matches a snapshot to its parent by that identity, and the adopting Run reaches the parent through the alias — so this Run's handle and the earlier handle both stay resolvable after a resume. Adoption is observable durably: each adopted Resource is a `lineage_adoption` row of the consuming Run, so counts and bytes per Run need no new metric, and the adoption log names the origin Run, file, and source URL.

Stop conditions, to revisit this decision rather than extend it silently: adoption sets routinely exceeding a small bound; adopted content consuming the budgets of current work; compaction-pruned documents still being requested; material change in multi-host blob retention cost; or a future decision to share Resources across Sessions.
