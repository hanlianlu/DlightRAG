# One read surface for a Run's stored bytes

Every byte one Answer Run stores is reachable through a single owner-scoped resource address: an accepted upload, a publication, and an image the run fetched, rendered, or adopted are all named by the `resource_id` their own registry recorded, and one reader serves them from the content-addressed blob plane. A conversation addresses the images its answers show through that surface, so a third-party URL expiring no longer breaks a stored answer. The positional `/runs/{run_id}/attachments/{ordinal}` address is replaced, not aliased.

## Status

Accepted and implemented. Supersedes nothing in [ADR 0010](0010-separate-run-blob-plane-from-operational-state.md) or [ADR 0013](0013-lineage-adoption-of-earlier-run-resources.md); it adds the read surface those decisions implied but never built.

## Context

A Run records bytes in two registries over one blob plane: `dlightrag_answer_run_artifacts` (the accepted input uploads and the publications) and `dlightrag_answer_resources` (what a worker fetched, rendered, or adopted). The browser could reach only the first, and only positionally: `/web/api/runs/{run_id}/attachments/{ordinal}`, which is why an uploaded image keeps working forever while a fetched one does not.

Observed in production: an answer that rendered two WolframAlpha figures wrote `![…](https://public6.wolframalpha.com/files/GIF_….gif)` into its own text. The run had stored both images — `dlightrag_answer_resources` rows with a blob digest, the recorded `source_locator` naming that exact URL — and the browser still displayed the third-party hotlink, which later returned 404. The screenshots in the conversation broke while the bytes sat intact in Postgres, because the *address* in the answer text was never ours. A service restart is not the cause; URL expiry is, and the same expiry would break the conversation with no restart at all.

Two facts shaped the repair. First, the publication path already refuses a non-self-contained document ("Active HTML must be a self-contained single file"), so the artifact surface has no such leak — the answer body simply never had the rule. Second, the id spaces do not collide: uploads are `attachment-<ordinal>`, carried-forward uploads `history-attachment-<ordinal>`, and resources `res-…`, `res-view-…`, or `attachment-occurrence:<entry>:<part>`, so one reader can resolve all of them without inventing new identity.

## Decision

**One reader, one address family.** `AnswerService.run_resource` resolves an owned `(run_id, resource_id)` across both registries — accepted artifacts first, with a current upload taking precedence over a carried-forward one sharing its ordinal, then run resources — and `open_run_resource` / `run_resource_size` / `read_run_resource` stream, size, and read the digest it resolves. `open_artifact` and `artifact_size` delegate to the same primitive, so a published artifact is no longer read by a second implementation.

**The positional attachment address is replaced.** `GET /web/api/runs/{run_id}/resources/{resource_id}` and its `/thumbnail` variant serve the bytes, keep the download disposition for non-images, and keep thumbnail derivation for image resources. The old `/attachments/{ordinal}` shape is removed rather than aliased, following this repository's existing rule that old browser data paths have no compatibility alias; the frontend never constructed those URLs itself, they were projected per turn.

**Stored answer images are addressed on our origin.** For a run, `run_external_source_map` maps each recorded external URL (normalized the way admission normalized it, plus recorded aliases) to its resource id, and the browser projection rewrites `<img src>` in an answer to the run-resource address. Only public HTTP(S) URLs that this run holds bytes for are rewritten: an unknown URL, a `data:` payload, a relative path, and a plain link the answer cites are all left exactly as the answer wrote them.

**No fetch on publication.** A URL the model wrote but never had fetched keeps pointing at the third party. The engine does not go and get it on the answer's behalf: a stored answer must stay a record of what the run actually read, and adding network work to the read path would put a third-party failure between a user and their own history.

**Rendered images are addressable too.** `view` output (`res-view-…`) was previously reachable only by the model; it is now served by the same surface, which is what makes a conversation's images one system rather than two.

## Considered options

- **Widen the artifact route to serve resources.** Rejected: `/artifacts/{id}` names publications, and a canvas that lists artifacts would then advertise bytes the answer never published as its own.
- **Register fetched images as publications.** Rejected: it promotes run state into the answer's deliverables, changing what the artifact canvas lists and what a publication budget counts.
- **Keep `attachments/{ordinal}` and add a third family.** Rejected on the operator's call: two address families for one blob plane is the split that caused this defect, and the positional address stops meaning anything once a resource id exists.
- **Rewrite markdown at answer time so the stored answer is self-contained.** Rejected for now: it mutates durable answer text and the run's result contract, where a projection-time rewrite is reversible and leaves the record untouched. Revisit if exports or other viewers need self-contained answers.
- **Fetch missing images when the answer is published.** Rejected: it invents reads the run never performed, spends the deployment's reputation on third-party requests, and turns a missing image into a new failure mode on the write path.

## Consequences

A conversation's images survive on our own origin for as long as the Run's bytes are retained, which is the same promise its uploaded attachments always had; a third-party URL can no longer break a stored answer, and the browser stops leaking the reader's address to an image host. Rendered pages become reachable in the UI, and the artifact route and the resource route read through one implementation, so a future registry joins by resolving an id rather than by adding a route.

Retention is unchanged: a resource is pruned with its Run (the `run_retention_days` floor), and an answer whose bytes aged out shows the same broken image it would have shown before. The rewrite is bounded work on read: only a succeeded answer that may name an external image costs a registry lookup, and only the rows of that one run are read.

Stop conditions, to revisit this decision rather than extend it silently: a stored answer rewritten to a resource that no longer resolves without the conversation saying so; the read path becoming a source of new third-party traffic; or a second registry appearing whose ids cannot be resolved by this reader.
