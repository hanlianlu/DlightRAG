# ADR 0056 — an Answer places a held image by the Resource id its Tool named

Status: accepted. Author: the multimodal-answer work.

## Context

An Answer could draw exactly one kind of picture: the visual asset retrieval had selected for a
corpus chunk it cited. `evidence_images` was derived only from cited sources
(`evidence_images_from_sources`, `src/dlightrag/engine/answer/media.py`), so on a deployment whose
corpus is empty the list was always empty. The product had meanwhile become an agent product: a
Research Run fetches pages with the Agent Browser and views the pictures on them, and those pixels
arrive in its context as tool-result attachments.

The result was an Answer that could read a dozen photographs and show none of them. A real
conversation asked for the day's photographs from a publisher's page; the Agent fetched the page,
viewed the images, and — with nowhere to put a picture — described them in prose for three turns
running, until the user said so and the Agent worked around it by copying the images into Artifacts
it could attach. That workaround was the only route: `attach_artifact` hands the Model the placement
syntax for what it attaches (`artifact_link()`, `engine/answer/publication.py`), and nothing gave a
viewed image the equivalent, so a picture the Run held had no address an Answer could name.

The plumbing was already there and unused. `answer_parts_from_markdown` resolved
`![label](evidence:<id>)` into an inline placement, `build_answer_presentation` rendered it, and the
run-resource route (`/web/api/runs/{run_id}/resources/{resource_id}`) already served any Resource a
Run holds. What was missing was a decision about **what makes a held image placeable**, and the
first attempt at one was wrong.

## Decision

**1. Placement is the selection rule, and it is the Model's.** A held image enters `evidence_images`
only where the Answer names it as `![label](evidence:<resource id>)`. A Run holds every image on
every page it read — logos, thumbnails, furniture — so admitting them all would bury the Answer in
furniture the reader never asked for. Requiring the Model to name one keeps the judgement where it
is: the Model knows which picture the reader asked to see.

**2. Admission does not ask the Resource registry whether the id exists.** This is the part the first
attempt got wrong, and it is the reason this record exists. The registry
(`engine/answer/resources/registry.py`) is the in-process model of what a Run may **read and view**;
it is not the record of what a Run **holds**. Tool attachments are durable resources the registry was
never told about: `attach()` prepares pixels and returns whether it attached them, and the caller
turns those into `attached_resources`, which settle as resource rows the run-resource route serves —
while the registry is never asked to admit them. Gating
placement on `registry.manifest()` therefore denied the Answer exactly the images the Agent had gone
and looked at, and it denied screenshots permanently, because recovery skips
`tool_attachment` rows when rebuilding the registry (`executor.py`, `kind in {"tool_attachment",
"conversion_asset"}: continue`). The invariant is one sentence: **a run-resource address resolves iff
the Run's own rows hold that resource, and that is the route's question, not finalization's.**

**3. The id a Tool names is the id the Answer places, and it is the attachment's.** `view` returns
the placement for the pixels it hands over — the prepared attachment — so the reader is shown what
the Model saw, at the size the Run decided, rather than the original the Run happened to fetch. The
same rule holds for every branch of that tool: no branch invents an id it cannot serve.

`screenshot` does not advertise a placement. Its pixels are a durable resource the run-resource
route would serve, so the capability is one line away; what stops it is that a screenshot's job in
this product is to put a page in front of the *model*, immediately, and the pages a reader should be
shown are the ones the Run fetched and viewed. A rule that named it too would make every screenshot
taken while working a candidate for the reading column. That is a judgement about prompting, not a
limit of the mechanism, and it is stated here so the next reader knows which it is.

**4. A transport composes the address, never the Model.** `project_answer_result` gained
`run_resource_prefix`, the run-resource counterpart of `artifact_url_prefix`; a transport that
publishes no such route (MCP keeps its calls cookie-free) leaves the address unset rather than
inventing one. Nothing a Model writes becomes an address a browser follows, which is the invariant
`retrieval-answer.md` already held — it now names both provenances instead of only the corpus one.

**5. Placing a picture does not make it evidence for a sentence.** A held image's pixels are an
observation of what the Agent looked at. A claim still cites the page's own text, and the
`evidence:` scheme keeps its name because the placement grammar is shared with corpus visual
evidence, not because a screenshot became citable. `domain-language.md` carries the distinction as a
**Held Image**.

## Consequences

- A Research Run on an empty corpus can show what it found. The reported Answer's shape — the Agent
  attaching copies of photographs to get them in front of the reader — is no longer the only route.
- Nothing changes for the corpus path: a chunk image still carries workspace and chunk and is still
  gated by the visual-workspace ACL, and it still asks its own route for a size.
- A held image names no thumbnail. The run-resource route declares no `size` parameter, so borrowing
  the corpus convention (`?size=thumb`) served the full original while claiming to be a thumbnail;
  and the route's own `/thumbnail` sibling would re-derive a copy from pixels the Run already
  prepared to display size. The record therefore carries the full address and no second one, and a
  client falls back to it.
- An id the Model invents resolves to nothing the Run holds, so the route answers not-found and the
  reader sees no picture. That is the same honest degradation as an Artifact link that does not
  resolve, and it is preferred to finalization guessing about bytes it cannot see.
- Placement is still unproven against a live Agent Run: no test drives a real browser-backed Run to
  the point of placing a screenshot. The seams are pinned; the end-to-end path is not.
