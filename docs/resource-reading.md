# Answer Resource Reading and Viewing

This document owns how an Answer Run reads and views its Resources: the `read`
and `view` tools, extraction status and visual discovery, conversion routes,
conversion snapshots, and adoption of an earlier Run's Resources. Public URL
admission follows [ADR 0005](adr/0005-public-web-resource-acquisition.md), a page
read as a browser renders it follows
[ADR 0032](adr/0032-the-agent-browser.md), stored bytes share one read surface under
[ADR 0016](adr/0016-one-run-resource-read-surface.md), and adoption follows
[ADR 0013](adr/0013-lineage-adoption-of-earlier-run-resources.md).
[Domain Language](domain-language.md) defines Resource Handle, Web Resource,
Rendered Read, and Blob.

## Scope

- A Resource belongs to one Answer Run: an accepted upload, an earlier upload
  re-registered for a follow-up or fork, a caller link, a search result link, a
  public URL the Agent chose, or an earlier Run's Resource adopted by this Run.
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
  Run, or an anonymous public `url`. `offset` and `limit` apply only to paths,
  and `focus` only to `resource_id` and `url` reads. A cursor continues a
  directory or a Resource; a URL read continues with the `resource_id` it
  returns. Optional `http.user_agent`, `http.accept`, and `http.accept_language`
  apply only to the first direct acquisition of a URL; passing them once the URL
  was fetched or has a text view is refused.
- A deployment with an Agent Browser also declares `rendered`, which reads a `url`
  or a Web Resource as a browser renders it ([Rendered reads](#rendered-reads)). It
  is refused with a `path` or `http` options, and it is not declared otherwise.
- Returns bounded text only. A Resource page starts with
  `[resource: <id> | lines <a>-<b> | extraction_status=<status>]`, holds one
  text window sized so that the page and the label the runtime puts before it fit
  the observation capacity (the room the compaction trigger leaves below the input
  limit, which is also where the runtime cuts a Tool result), and ends with notes,
  visual handles, and a `[more; cursor=…]` continuation. A Resource longer than one
  window is therefore always read on by cursor, never cut. The window is sized for the
  Run's own model: a Child on a role whose configured context window is much smaller
  has the page cut at its own capacity, with no cursor for the cut part.
  `focus` starts the page order at the most relevant window; the continuation
  runs to the end, wraps to the start, and stops before the focus window, so
  every character is returned once.
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
  40-megapixel limit, independently of text extraction; only a conversion that
  ended `safety_refused` also refuses `view` for the rest of the Run. Office
  pages and slides are never rendered.
- Every image is charged to the Run's answering-model image budget (image count,
  bytes, and pixels), which tool calls and replayed attachments share, and is
  prepared under its
  [image rules](retrieval-answer.md#answer-input-and-packing), so a rendered
  page can reach the model downscaled. An overview stops early when the budget
  runs out, and a view that can attach nothing fails without pixels. A model
  without image support cannot view.

## Registration and acquisition

- Resource handles (`res-…`, and `vis-…` for an embedded image) are opaque and
  minted from a random identity the Run draws at acceptance and records with its
  prepared input. No deployment secret takes part, so a resume, also after the
  database credentials rotate, mints the handles the Run already printed.
  Cursors carry text offsets and are HMAC-signed, not encrypted, with a key
  derived from the same identity and bound to their Resource and focus; a cursor
  never crosses a Run.
- An upload is admitted only when a Run can read it, by one rule in
  `engine/answer/resources/admission.py`: its type admits an image, a document a
  converter turns into text, a listed text type, or a `text/*` media type, and
  any other upload is decided by its bytes, admitted when they verify as an image
  or decode as text. Every transport applies that rule at acceptance and refuses
  any other upload as `UNSUPPORTED_ATTACHMENT_TYPE`; the Web composer offers the
  listed types. A link is admitted whatever its address names, because what it
  serves is known only once it is fetched. A caller link whose address or
  declared type names an image is the exception: acceptance fetches and verifies
  it within the image limits and pins it as an attachment, or fails the request.
- Current uploads, caller links, re-registered earlier uploads, and adopted
  Resources share the `answer.generation.max_attachments` slots; Web Search
  links and URLs the Agent chooses take none. An upload larger than
  `max_attachment_bytes` is refused, and a fetched URL body larger than it fails
  the direct fetch; uploads together may not exceed `max_total_attachment_bytes`
  ([defaults](configuration.md#answer-generation-and-attachments)).
- A public URL is checked for scheme and embedded credentials when it is
  registered, and for DNS and redirect policy when it is fetched, under the
  [egress boundary](security.md#answer-resources-and-execution). Within a Run,
  one normalized URL resolves to its first successfully admitted snapshot.
  Fetched bytes are persisted before the tool result settles and are never
  fetched again during recovery.
- Direct anonymous HTTP runs first. When it fails or yields no text for a
  textual resource, the configured Extract chain supplies text once. Its steps run
  in order and the first usable result wins: the hosted providers (Exa, Tavily),
  then the Agent Browser, which a Research Run with a browser appends unless the
  configuration places it elsewhere
  ([Public Web Sources](configuration.md#public-web-sources)). A URL the local
  policy rejects never reaches an external provider or the browser.
- Extract text becomes the snapshot only when the fetch failed. When the fetch
  succeeded but its bytes hold no text, the bytes stay the snapshot and the
  Extract text is their text view, recorded and restored with them, so `view`
  and `read` of the same URL agree in any order. The Agent Browser's rendering is
  not Extract text: it is a second representation of the Resource
  ([Rendered reads](#rendered-reads)).
- The model sees an inventory of registered Resources with a kind for each:
  `image; view`, `PDF; read text or view physical pages`,
  `DOCX|PPTX|XLSX; read extracted text and embedded-image inventory`, the MIME
  type, or `resource; type verified on acquisition` when none is declared.

## Rendered reads

A page whose content a script builds reaches `read` as a shell: the fetch succeeds
and the text is a loading or enable-JavaScript notice. A Research Run in a deployment
with an [Agent Browser](security.md#agent-browser-boundary) can read such a page as
a browser renders it. The Fast path never does.

- **Asking.** `read(url=…, rendered=true)` and `read(resource_id=…, rendered=true)`
  render the page in the Run's browser, serialize the DOM after its scripts ran, and
  convert that HTML by the route direct HTML takes, so both text views come from one
  converter. `rendered` applies only to a URL or a Web Resource. A workspace `path`,
  `http` options, a spill handle, and a Resource that is not a Web Resource (an
  upload, for one) are refused, never silently ignored.
- **A second representation.** The rendering is appended to the same Web Resource
  with acquisition `browser_render`. It never replaces the admitted snapshot, so
  `view` and earlier citations keep reading what they read, and it has no handle of
  its own: results print the Resource's handle, a header
  `[resource: <id> | rendered | lines <a>-<b> | …]` says the text is the rendering,
  and its note names the final URL when the page ended somewhere else. It is not in
  the manifest and takes no attachment slot. Its evidence cites the Resource's URL
  with the acquisition `browser_render`. What the browser returned is the browser's
  assertion, not an attestation that an anonymous GET serves the same page.
- **Cursors.** A cursor names the representation it pages. A rendered continuation
  starts with `r.` and a rendered visual inventory with `rvisual.`; a cursor alone
  selects its representation, flagged or not. `rendered=true` with a cursor of the
  direct text is refused, as is a rendered cursor on any other Resource.
- **Images.** The HTML route lists only the rendering's embedded data-URI images, as
  `vis-…` handles. Viewing one makes no fetch and no render.
- **A plain read.** Without `rendered` and without a cursor, `read` of a Web Resource
  returns the first of these, which depend only on what the Run holds and so are the
  same after recovery:
  1. the Resource's own text view, from its fetch or from hosted Extract text;
  2. its rendering, when the Run holds one and the Resource has no bytes of its own
     or its bytes held no text, without a fetch and without walking the chain;
  3. otherwise the acquisition of any URL: a direct fetch, then the Extract chain
     when the fetch failed or yielded no text.
- **The chain.** When the chain reaches the browser, the render is automatic. For a
  fetch that failed, the rendering is the Resource's only representation. For a
  fetch that succeeded with bytes holding no text, the bytes stay the snapshot and
  settle beside the rendering, which a plain read then returns. An automatic render
  never raises: its failure is folded into the `unavailable` note as
  `Agent Browser: <reason sentence>`.
- **Once per Run.** The first successful rendering of a Resource serves every later
  read of the Run, Child Sessions included, and concurrent reads share one render. A
  render that fails, or yields no text, pins nothing, so a later read tries again.
- **Bounds.** Each render uses a temporary context that holds no cookies, storage, or
  service workers, accepts no downloads, and closes when the render ends; a Run's
  browser serves at most four at once. The page loads within
  `navigation_timeout_seconds`, then settles for at most `settle_timeout_seconds` and
  is read as it stands if it never goes quiet
  ([Agent Browser](configuration.md#agent-browser)). The serialized page may not
  exceed `answer.generation.max_attachment_bytes`.
- **Where the check ends.** DlightRAG checks the first URL as a direct read does
  (scheme, no embedded credentials or credential query parameters, a host that
  resolves to public addresses only) and the form of the final URL the page ended at.
  Everything the page loads in between, redirects and subresources included, is
  confined by the deployment's network, not by this check
  ([Agent Browser Boundary](security.md#agent-browser-boundary)).

An explicit rendered read that fails is a tool error with no effects, and the
sentence it carries is fixed per reason: page URLs and driver error text never enter
it.

| Reason | What the model reads |
|---|---|
| `not_configured` | The Agent Browser is not available in this Run. |
| `busy` | Every Agent Browser is in use by other Runs; try again later or work from the direct read |
| `unreachable` | The Agent Browser is unreachable, so this page was not rendered |
| `disconnected` | The browser disconnected while rendering; the next rendered read starts a fresh one |
| `timeout` | The page did not finish loading within the configured seconds |
| `navigation_failed` | The browser could not load the page, with its network error token when it has one |
| `http_status` | The page answered HTTP 4xx or 5xx to the browser |
| `download` | The URL starts a download, not a page |
| `too_large` | The rendered page exceeds the byte limit |
| `no_text` | The rendered page produced no text |
| `final_url_refused` | The page ended at a URL this deployment does not admit; nothing was admitted |

A rendering settles with the read that produced it, and recovery restores it without
a browser. The settlement is one `web_render` row, the rendered HTML with its URL,
final URL, and admission origin, plus the conversion snapshot of its text and images
that every converted Resource settles
([Conversion snapshots](#conversion-snapshots-and-recovery)). A resumed Run restores
the rendering under the Resource's handle, re-registering a search or agent Resource
that the Run knew only by its rendering, and then reads it with zero renders. A later
turn's lineage adoption does not carry a rendering: it adopts the Resource's
admitted bytes, if there are any, and renders the page again if it needs to.

## Extraction status and discovery

| `extraction_status` | Meaning |
|---|---|
| `text` | Decoded text of an unconverted resource, or text from the Extract chain |
| `usable_text_unverified_coverage` | The converter produced text; completeness is unverified |
| `known_incomplete` | The converter or the DOCX package audit reported an omission: OCR required (with the known pages when reported), a source it cannot represent, or visuals the extraction does not cover. No partial text is invented |
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
  allowance (at least 32 tokens), with a `cursor` to page through the rest as a
  visual inventory that is not text evidence.
- Handles are opaque `vis-…` identifiers. A label shows only a supported
  location: an XLSX `Sheet!Cell` anchor, an image's alt text, or for DOCX
  `package part <part>; location unknown`. No text-line-to-page or part-to-page
  mapping is inferred.
- A refused read returns
  `extraction_status=safety_refused; no evidence admitted` and tells the model
  not to retry another parser or renderer around the restriction.

## Conversion routes

Routing uses the filename suffix, then the declared MIME type. The converted
formats are PDF, DOCX, XLSX, PPTX, CSV, and HTML (`.html`, `.htm`). Other
non-image resources decode strictly: a byte-order mark, then a declared
charset, then, for bytes that do not look binary, a detected encoding.
Undecodable bytes fail the read instead of becoming replacement characters; a
textual URL then falls back to the Extract chain.

| Format | Text | Images | Fallback |
|---|---|---|---|
| PDF | AnyDoc | PDFium page rendering, independent of text | MarkItDown once. `NeedsOcr` and `Unsupported` results are terminal `known_incomplete` |
| DOCX | AnyDoc | AnyDoc's typed image occurrences bound to verified package parts, whether or not the document has images | MarkItDown once, each image bound to a referenced package part by SHA-256 |
| XLSX | AnyDoc display values; cached formula results, no recalculation | openpyxl images with `Sheet!Cell` anchors; external links are not loaded | MarkItDown once |
| PPTX | MarkItDown | Embedded data-URI images | None |
| CSV | MarkItDown | None | None |
| HTML | MarkItDown | Embedded data-URI images | None |

- AnyDoc is exactly `firecrawl-anydoc` 0.2.4 (import `anydoc`); any other
  installed version is a configuration error, not a fallback. It runs offline on
  admitted bytes with OCR rejected. MarkItDown runs with plugins disabled on
  admitted bytes and an explicit stream type, never fetches, and uses a fresh
  converter per call. A charset the declared media type names, when Python knows it,
  decodes the bytes ahead of any `<meta charset>` the markup carries, as the HTML
  specification orders; a rendered page is UTF-8 whatever its own meta says.
- OOXML archives (DOCX, PPTX, XLSX) pass a central-directory preflight before
  any converter opens them: no duplicate or encrypted entries, at most 10,000
  entries, 100 MiB per entry, 512 MiB uncompressed in total, and a 100×
  expansion ratio.
- One 120-second deadline covers preflight, AnyDoc, and any fallback. It stops
  new work from starting or being adopted; it bounds neither process memory nor
  native code that is already running, which cancelling the Python caller cannot
  interrupt either.
- Only an AnyDoc initialization, malformed-input, or missing-part failure falls
  back: once, after AnyDoc's work has ended, within the same deadline, with the
  fallback reason recorded. An empty AnyDoc result is `no_extracted_text`, not a
  fallback trigger. Resource limits, unsafe archives (including encrypted
  entries), failed image binding, deadline exhaustion, cancellation, and memory
  exhaustion are `safety_refused` and never fall back. An encrypted PDF, a
  password-protected Office file (which is not a ZIP archive), and an XLSX whose
  image extraction fails are `conversion_failed`, the last because text is never
  adopted without its images.

Why PPTX, CSV, and HTML use MarkItDown, and what the AnyDoc routes do not
claim:

- HTML: AnyDoc does not accept HTML.
- PPTX: AnyDoc drops a slide whose part is missing from the package without an
  error, and DlightRAG has no OPC completeness check or typed image binding for
  PPTX that would detect the loss.
- CSV: AnyDoc mis-decodes Shift-JIS without an error. MarkItDown detects the
  encoding, strips a leading BOM, and pads short rows to the widest row, but
  collapses line breaks inside a cell into spaces. DlightRAG adds no decoding or
  normalization step of its own for CSV.
- PDF: AnyDoc returns `Unsupported`, without page metadata, for some PDFs that
  do carry text, such as a page holding both text and a raster image. That stays
  `known_incomplete` rather than adopting partial fallback text, which omits the
  raster content too; the physical-page overview of `view` locates the page
  instead.
- DOCX: AnyDoc's Markdown omits image links, so images come from its typed
  assets. A package audit marks the read `known_incomplete` when the source
  references visuals the typed occurrences do not cover, such as header, VML,
  external, chart, or embedded-object visuals; it never fetches them or switches
  engines. An asset whose bytes are not a supported image or differ from its
  package part refuses the conversion as `safety_refused`. A fallback image that
  matches no referenced part is refused the same way, and one that matches
  several parts stays unmapped and makes the read `known_incomplete`.
- XLSX: an uncached formula cell stays empty. Number formats other than
  percent, date, currency, and custom ones, and drawing types other than
  embedded images, are unverified.

`scripts/anydoc_pilot.py`, `scripts/docx_integration_bench.py`, and
`scripts/format_route_bench.py` rerun these comparisons offline on generated
fixtures; their small samples establish neither service latency nor
completeness for arbitrary documents.

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
- Concurrent reads of one Resource share one conversion. Cancelling one of them
  cancels that conversion for every reader: no fallback starts, and an
  unfinished conversion settles as a `safety_refused` failure snapshot for the
  rest of the Run. Closing the registry joins running native conversions,
  including publication views, before it drops its in-memory views.
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
- A Child Session started with its parent's context receives only the image
  occurrences pinned in that context, from bytes this Run already holds, under
  the Run's lease. It adopts nothing from earlier Runs and registers no handle;
  its model must accept images, and each inherited image is charged once to the
  Run's image budget.
- Earlier uploads re-registered for a follow-up or fork load their bytes only
  when read or viewed. A follow-up or fork through the Run API re-registers its
  parent Run's uploads and links. A Web follow-up re-registers the uploads of
  the conversation's succeeded turns, newest turn first, up to the attachment
  allowance its own uploads leave.
- Lineage adoption is on by default (`answer.generation.lineage_adoption`). When
  `read` or `view` names an unknown handle, the loader looks in the same owner
  and Agent Session for a retained row with that handle and an adoptable kind: a
  fetched Web body, a tool attachment, a Published Artifact, or a Resource an
  earlier turn adopted. Its Blob digest must match. A rendering is not an adoptable
  kind ([Rendered reads](#rendered-reads)).
- A stored conversion view is checked before anything is registered: it must
  decode, name the same handle, and match the input digest of the bytes. A view
  that fails any check refuses the adoption ("the document was not converted
  again"), and because nothing was registered, asking again refuses again.
- The bytes then become a Resource of this Run under a new canonical handle, and
  the earlier handle becomes its alias. The stored view becomes the view of that
  Resource: its text, images, and conversion facts are the earlier Run's,
  unchanged, and only the Resource it names becomes this Run's canonical handle.
  That is the handle the call prints, so a later turn adopts the Resource again
  by it.
- Before the earlier handle resolves, the adoption is written as this Run's own
  Resources: the adoption row (the canonical handle, the bytes, and the earlier
  handle as an alias), the view, and its images, in one transaction under the
  Run's lease. The rows belong to the conversation's Agent Session also when a
  Child's call adopted, so a later turn can adopt them in turn. Cleanup of the
  origin Run cannot invalidate them, also while they are written: the write
  holds the Blobs it names until it commits, so a concurrent purge skips them.
  The adoption holds whatever the call that asked for it does next: a cancelled
  call, or a retried `read` or `view` that fails on a stale cursor, a document
  with no viewable target, or a refused view, leaves the adoption in place, and
  the failure is a typed refusal of its own. Once adopted the handle is held, so
  an embedded-image handle the call cannot find is named as the missing one.
- If that write fails, the running Run keeps nothing of it: the handle stays
  unknown, no attachment slot is spent, and asking again tries again. A write
  whose outcome is unknown, because the connection or the call ended while it
  committed, may still have landed; the next resume then restores it like any
  recorded adoption. A lost lease stops the call without writing anything; the
  Run's next claimant adopts afresh.
- A Run adopts one earlier Resource at a time. Two earlier handles for the same
  file (the same file name, declared MIME type, and SHA-256) are one Resource
  with one adoption row whose aliases are merged, and the first stored view
  adopted for it is its only view; the store refuses a different second view.
- A resume restores adopted Resources from those rows as the Run's durable
  state, under the handle each was recorded with: both handles resolve and the
  view is back, without reading the lineage. A restored adoption takes its
  attachment slot and bytes but is never refused, even when an adoption whose
  write landed unseen puts the Run past its allowance. A stored view whose
  Resource no restored row names is a real inconsistency and fails the resume.
- Newly adopted bytes are registered stored-view-only, and this Run never
  converts them. A convertible document (PDF, DOCX, XLSX, PPTX, CSV, or HTML)
  reads text only through a stored view: the one adopted or restored with it, or
  one this Run already holds for the same bytes through another handle. Other
  formats, such as a Markdown Published Artifact or a fetched text page, are
  decoded from the adopted bytes. `read` of a convertible document with no such
  view refuses without adopting it, and names the remedy: re-read it from its
  URL or a fresh attachment, or, for a PDF only (by file name or declared type),
  view its pages as pixels. `view` can still adopt such a document for pixels
  that need no conversion, such as PDF pages, but a later `read` through the
  earlier handle or this Run's handle refuses the same way, and recovery keeps
  it so.
- Publication stores that view for a convertible Published Artifact. The
  publishing Run converts the product with the converters and limits a read
  uses, mints its image handles as a read would, and records the snapshot and
  its images as rows of that Run, named by the product's handle and stamped with
  the conversation's Agent Session, in the transaction that publishes the
  product; that transaction writes every Blob it names in one sorted pass. With
  lineage adoption off, publication builds no view, so a convertible product
  published then refuses `read` even after adoption is turned on. The products
  of one publication share one conversion's 120-second budget: once it is spent
  no further conversion starts or is adopted, though a native step already
  running finishes first, as it does for a read. A product whose conversion
  fails, is refused, or finds the budget spent still publishes without a view,
  and a later `read` of it refuses as above. So every product is reached by
  `read(resource_id='artifact-…')`, the call the attaching Tool teaches. Every
  version of one Artifact path shares that handle; a later turn adopts the
  newest version with only the view its own Run stored, never an older
  version's.
- Adopted bytes that match a Resource this Run already holds by file name,
  declared MIME type, and SHA-256, such as the same image attached again, keep
  that Resource's state from its first admission: bytes this Run can convert
  itself keep their own conversion and never take an earlier view, and a
  Resource adopted without a view takes the first stored view adopted for the
  same bytes. Uploaded documents, current or re-registered from an earlier Run,
  load lazily and never match adopted bytes.
- Newly adopted bytes take an attachment slot and count toward the upload byte
  limits. An adoption past the allowance refuses as a tool error: "too many
  attachments; the earlier document was not adopted into this run".
- When a Run holds at least one Resource, its manifest tells the model that a
  resource id printed by an earlier turn may still resolve, and that only a
  refusal means attaching the document again. A handle that this Run neither
  holds nor can adopt is refused with that remedy. Cursors stay per Run: an
  earlier turn's cursor is refused, and calling `read` or `view` on the Resource
  again returns a current one.
