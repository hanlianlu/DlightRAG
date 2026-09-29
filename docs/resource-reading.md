# Answer Resource Reading and Viewing

This document owns how an Answer Run reads and views its Resources: the `read`
and `view` tools, extraction status and visual discovery, conversion routes,
conversion snapshots, and adoption of an earlier Run's Resources. Public URL
admission follows [ADR 0005](adr/0005-public-web-resource-acquisition.md),
stored bytes share one read surface under
[ADR 0016](adr/0016-one-run-resource-read-surface.md), and adoption follows
[ADR 0013](adr/0013-lineage-adoption-of-earlier-run-resources.md).
[Domain Language](domain-language.md) defines Resource Handle, Web Resource, and
Blob.

## Scope

- A Resource belongs to one Answer Run: an accepted upload, an earlier upload
  re-registered for a follow-up, a caller link, a search result link, a public
  URL the Agent chose, or an earlier Run's Resource adopted by this Run.
- Corpus ingestion is separate: Answer Resources never invoke MinerU or Docling
  and never become corpus documents, chunks, vectors, BM25 rows, or graph data.
  Workspace files are not converted; `read(path)` decodes UTF-8 or BOM-tagged
  UTF-16 text, and `view(path)` accepts standalone images only.
- `read` and `view` are Research tools. Fast sends current-turn images to the
  model directly and cannot represent any other attachment, so an Answer that
  carries one must resolve to Research. An image attachment in either mode needs
  a vision-capable query model; there is no text-only fallback.

## Tools

### `read`

- Takes exactly one of a workspace `path`, a `resource_id` registered in this
  Run, or an anonymous public `url`. `offset` and `limit` apply only to paths
  and `focus` only to Resources. A cursor continues a directory or a Resource; a
  URL read continues with the `resource_id` it returns. Optional
  `http.user_agent`, `http.accept`, and `http.accept_language` apply only to the
  first direct acquisition of a URL.
- Returns bounded text only. A Resource page starts with
  `[resource: <id> | lines <a>-<b> | extraction_status=<status>]`, holds one
  text window whose whole envelope fits the model's remaining text allowance,
  and ends with notes, visual handles, and a `[more; cursor=…]` continuation.
  `focus` starts the page order at the most relevant window; the continuation
  still covers the whole text.
- Never attaches pixels, runs OCR, renders a document, or calls another model.
  Reading an image returns its media type and size and points to `view`.

### `view`

- Takes exactly one of a `resource_id`, an admitted public `url`, or a workspace
  `path`. `locator` and `cursor` are mutually exclusive, a path takes neither,
  and a URL view continues with its returned `resource_id`. `view(url)` shares
  the acquisition of `read(url)` and does not need text conversion to succeed.
- Attaches pixels as tool-result attachments for the vision-capable model of the
  Agent or Child Session that called it; no separate VLM describes them. A Child
  Session's result reaches its parent as text, so pixels are not forwarded.

| Target | Result |
|---|---|
| Verified image | The source image; no locator or cursor |
| PDF with `locator=<n>` | Physical page `n`, rendered at 2× (144 dpi) |
| PDF without a locator | Thumbnails of up to 8 consecutive physical pages, longest side at most 900 px, labeled by physical page, stating the covered range (`covers physical pages a-b of N only`) and that thumbnails are not reliable small-text transcription, with a `cursor` for the next pages |
| Converted document with `locator=vis-…` | One embedded image from the adopted conversion |
| Anything else, including hosted-extraction text | An error naming what can be viewed |

- PDFium renders one page at a time on demand, under a process-wide lock and a
  40-megapixel limit, independently of text extraction. Office pages and slides
  are never rendered.
- Every image is charged to the Run's answering-model image budget (image count,
  bytes, and pixels), which tool calls and replayed attachments share. An
  overview stops early when the budget runs out, and a view that can attach
  nothing fails without pixels. A model without image support cannot view.

## Registration and acquisition

- Resource handles (`res-…`) are opaque and minted with per-Run secrets. Cursors
  are HMAC-signed and bound to their Resource and focus; a cursor never crosses
  a Run.
- Caller attachments and links count against `answer.generation.max_attachments`
  (6). An upload larger than `max_attachment_bytes` (100 MiB) is refused, and a
  fetched URL body larger than it fails the direct fetch; uploads together may
  not exceed `max_total_attachment_bytes` (128 MiB).
- A public URL is checked for scheme and embedded credentials when it is
  registered, and for DNS and redirect policy when it is fetched. Within a Run,
  one normalized URL resolves to its first successfully admitted snapshot.
  Fetched bytes are persisted before the tool result settles and are never
  fetched again during recovery.
- Direct anonymous HTTP runs first. When it fails or yields no text for a
  textual resource, the configured Extract chain supplies text once. A URL the
  local policy rejects never reaches an external provider.
- The model sees an inventory of registered Resources with a kind for each:
  `image; view`, `PDF; read text or view physical pages`,
  `DOCX|PPTX|XLSX; read extracted text and embedded-image inventory`, or the
  MIME type.

## Extraction status and discovery

| `extraction_status` | Meaning |
|---|---|
| `text` | Decoded text of an unconverted resource, or text from the Extract chain |
| `usable_text_unverified_coverage` | The converter produced text; completeness is unverified |
| `known_incomplete` | The converter reported an omission: OCR required (with the known pages when reported), a source it cannot represent, or external, unsupported, or unmapped images. No partial text is invented |
| `no_extracted_text` | Conversion produced no text, which does not prove the document is blank |
| `conversion_failed` | An ordinary conversion failure; no text evidence |
| `safety_refused` | An archive, resource-limit, deadline, admission, or memory refusal |
| `image` | The resource is an image; use `view` |
| `unavailable` | A public URL yielded no citable text from direct HTTP or the Extract chain |

- A converted read says it is an extracted text view. For a PDF it adds the
  physical page count when PDFium can read it and says that pages are not mapped
  to text lines; for other formats it says coverage is unverified and that a
  handle is an embedded image, not a page screenshot.
- Visual handles are listed up to 8 per read, within a quarter of the text
  allowance, with a `cursor` to page through the rest as a visual inventory that
  is not text evidence.
- Handles are opaque `vis-…` identifiers. A label shows only a supported
  location: an XLSX `Sheet!Cell` anchor, an image's alt text, or for DOCX
  `package part <part>; location unknown`. No text-line-to-page or part-to-page
  mapping is inferred.
- A refused read returns
  `extraction_status=safety_refused; no evidence admitted` and tells the model
  not to retry another parser or renderer around the restriction.

## Conversion routes

Routing uses the filename suffix, then the declared MIME type. The converted
formats are PDF, DOCX, XLSX, PPTX, CSV, and HTML (`.html`, `.htm`); other
non-image resources are decoded as text.

| Format | Text | Images | Fallback |
|---|---|---|---|
| PDF | AnyDoc | PDFium page rendering, independent of text | MarkItDown once. `NeedsOcr` and `Unsupported` results are terminal `known_incomplete` |
| DOCX | AnyDoc | AnyDoc's typed image occurrences bound to verified package parts, whether or not the document has images | MarkItDown once, with its images bound to package parts the same way |
| XLSX | AnyDoc display values; cached formula results, no recalculation | openpyxl images with `Sheet!Cell` anchors; external links are not loaded | MarkItDown once |
| PPTX | MarkItDown | Embedded data-URI images | None |
| CSV | MarkItDown | None | None |
| HTML | MarkItDown | Embedded data-URI images | None |

- AnyDoc is exactly `firecrawl-anydoc` 0.2.4 (import `anydoc`); any other
  installed version is a configuration error, not a fallback. It runs offline on
  admitted bytes with OCR rejected. MarkItDown runs with plugins disabled on
  admitted bytes and an explicit stream type, never fetches, and uses a fresh
  converter per call.
- OOXML archives (DOCX, PPTX, XLSX) pass a central-directory preflight before
  any converter opens them: no duplicate or encrypted entries, at most 10,000
  entries, 100 MiB per entry, 512 MiB uncompressed in total, and a 100×
  expansion ratio.
- One 120-second deadline covers preflight, AnyDoc, and any fallback. It stops
  new work from starting or being adopted; it cannot interrupt native code that
  is already running, and neither can cancelling the Python caller.
- Only an AnyDoc initialization, malformed-input, or missing-part failure falls
  back: once, after AnyDoc's work has ended, within the same deadline, with the
  fallback reason recorded. An empty AnyDoc result is `no_extracted_text`, not a
  fallback trigger. Resource limits, unsafe archives, failed image binding,
  deadline exhaustion, cancellation, and memory exhaustion are `safety_refused`
  and never fall back. An encrypted document is `conversion_failed`, and so is
  an XLSX whose image extraction fails, because text is never adopted without
  its images.

Why the other formats stay on MarkItDown, and what the AnyDoc routes do not
claim:

- HTML: AnyDoc 0.2.4 does not support HTML.
- PPTX: AnyDoc 0.2.4 drops a slide whose part is missing from the package
  without an error, and DlightRAG has no OPC completeness check or typed image
  binding for PPTX that would detect the loss.
- CSV: neither engine is complete. AnyDoc 0.2.4 mis-decodes Shift-JIS without an
  error; MarkItDown decodes it but truncates over-wide rows and keeps BOMs and
  raw cell newlines. DlightRAG has no host decoding and normalization step for
  CSV.
- PDF: AnyDoc returns `Unsupported`, without page metadata, for some PDFs that
  do carry text, such as a tested text-plus-raster page. That stays
  `known_incomplete` rather than adopting partial fallback text, which omits the
  raster content too; physical-page viewing remains available.
- DOCX: AnyDoc 0.2.4's Markdown omits image links, so images come from its typed
  assets.
- XLSX: the generated fixtures cover percent, date, currency, custom, and merged
  cells, cached and uncached formulas, and one image repeated across cells and
  sheets; other number formats and drawing types are unverified.

These are current routes, not permanent bans. `scripts/anydoc_pilot.py`,
`scripts/docx_integration_bench.py`, and `scripts/format_route_bench.py` rerun
the PDF, DOCX, XLSX, and PPTX comparisons offline on generated fixtures. They
carry only a UTF-8 CSV control, so the CSV findings above have no fixture there.
Their small samples do not establish service latency, arbitrary-document
completeness, or untested platforms.

## Conversion snapshots and recovery

- A Resource's first conversion in a Run is adopted as its snapshot: text,
  images with their locators, extraction status, converter and version, fallback
  reason, known OCR pages and page count, note, input digest (SHA-256 of the
  source bytes), and output digest (SHA-256 of the text). A failure snapshot
  records status and converter with empty text.
- The snapshot settles with the tool result through the existing owner-scoped
  resource and Blob effect settlement, as `conversion.json` plus one
  `conversion_asset` per image. There is no cross-owner cache or second store.
- Later reads, cursors, and recovery reuse the adopted snapshot and never rerun
  parser selection or switch engines after a dependency change. Recovery checks
  the output and asset digests, re-reads the source bytes, and checks the input
  digest before adopting the snapshot again.
- Concurrent reads of one Resource share one conversion. A cancelled read cannot
  start a late fallback, and closing the registry waits for native conversion to
  finish before releasing storage.
- A rendered page or viewed image settles as a derivative attachment with its
  source Resource, its exact provenance (physical page and overview flag, or
  embedded-image handle, anchor, and package part), and its digest. The
  attachment identity hashes provenance and bytes, so identical bytes at two
  locations stay two occurrences. Replay restores the settled bytes; nothing is
  pre-rendered.

## Earlier Runs

- A follow-up or fork Run hydrates earlier image attachments only from its
  selected Session lineage. Before hydration, the executor retains the exact
  selected occurrences under its own Run lease and fence, then checks each Blob
  digest; missing, mismatched, or unauthorized bytes fail explicitly. Replayed
  images are charged to the consuming model's image budget. Hydration alone
  registers no earlier handle.
- Earlier uploads re-registered for a follow-up load their bytes only when read
  or viewed.
- Lineage adoption is on by default (`answer.generation.lineage_adoption`). When
  `read` or `view` names an unknown handle, the loader looks in the same owner
  and Agent Session for a retained row with that handle and an adoptable kind: a
  fetched Web body, a tool attachment, or a Published Artifact. Its Blob digest
  must match.
- A stored conversion view is checked before anything is registered: it must
  decode, name the same handle, and match the input digest of the bytes. A view
  that fails any check refuses the adoption ("the document was not converted
  again"), and because nothing was registered, asking again refuses again.
- The bytes then become a Resource of this Run under a new canonical handle, the
  earlier handle becomes its alias, and the stored view is adopted verbatim. The
  bytes, view, and images settle as this Run's own Resources under its fence, so
  cleanup of the origin Run cannot invalidate them. The adoption row is located
  by the canonical handle and carries every alias bound to it, so two earlier
  handles for the same file (the same file name, declared MIME type, and
  SHA-256) settle one Resource with merged aliases.
- The adoption settles even when the retried `read` or `view` then fails, for
  example on a stale cursor, a document with no viewable target, or a refused
  view: the failure returns as a typed refusal that carries the adoption. Once
  adopted the handle is held, so an embedded-image handle the call cannot find
  is named as the missing one.
- An adoption whose call was cancelled, or whose result the Session could not
  record, leaves no adoption row. Recovery then skips any view later settled
  under that earlier handle, with a warning, instead of failing the Run; naming
  the handle again adopts it again under the same canonical handle.
- Newly adopted bytes are registered stored-view-only, and this Run never
  converts them. A convertible document (PDF, DOCX, XLSX, PPTX, CSV, or HTML)
  reads text only through the view adopted or restored with it; other formats,
  such as a Markdown Published Artifact or a fetched text page, are decoded from
  the adopted bytes. `read` of a convertible document whose earlier Run never
  extracted text refuses, and names the remedy: re-read it from its URL or a
  fresh attachment, or, for a PDF only (by file name or declared type), view its
  pages as pixels. `view` can
  still adopt such a document for pixels that need no conversion, such as PDF
  pages, but a later `read` through the earlier handle or this Run's handle
  refuses the same way, and recovery keeps it so.
- Adopted bytes that match a Resource this Run already holds by file name,
  declared MIME type, and SHA-256, such as the same file attached again, keep
  that Resource's state from its first admission. A view it already has stays
  and no second view is adopted; a Resource without a view adopts the stored
  view, if there is one. Follow-up uploads re-registered from an earlier Run
  load lazily and never match adopted bytes.
- Newly adopted bytes take one `answer.generation.max_attachments` slot and
  count toward the upload byte limits. An adoption past the allowance refuses as
  a tool error: "too many attachments; the earlier document was not adopted into
  this run".
- The Resource manifest tells the model that a resource id printed by an earlier
  turn may still resolve, and that only a refusal means attaching the document
  again. A handle that this Run neither holds nor can adopt is refused with that
  remedy. Cursors stay per Run: an earlier turn's cursor is refused, and calling
  `read` or `view` on the Resource again returns a current one.

## Verification

- `tests/unit/test_resource_tools.py`: the `read` and `view` seams, inventories,
  identity, and image budgets.
- `tests/unit/test_resource_converters.py`,
  `tests/unit/test_docx_conversion.py`: routes, OOXML preflight, fallback and
  terminal classifications, and DOCX occurrences.
- `tests/unit/test_resource_visual.py`: bounded PDF rendering independent of
  text.
- `tests/unit/test_resource_registry.py`, `tests/unit/test_resource_text.py`,
  `tests/unit/test_resource_lexical.py`,
  `tests/unit/test_resource_review_regressions.py`: registration, text windows,
  focus ordering, cursors, and manifest classification.
- `tests/unit/test_resource_snapshot_runtime.py`: settled snapshots and located
  pixels restored through the Answer host and Agent runtime.
- `tests/unit/test_resource_lineage_adoption.py` and the recovery test in
  `tests/unit/test_answer_executor.py`: adoption checks, stored-view-only
  adopted bytes, decoding of unconverted formats, the same file under two
  earlier handles, settlement when the retried call fails, the attachment
  allowance, and the manifest wording.
- `tests/integration/test_resource_lineage_pg.py`,
  `tests/integration/test_agent_session_pg.py` (one adoption row with both
  aliases), `tests/integration/test_attachment_replay_pg.py`,
  `tests/integration/test_resource_review_regressions_pg.py`: adoption,
  selected-lineage replay, and durable settlement against PostgreSQL.
