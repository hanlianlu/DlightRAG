# Answer Resource Reading and Viewing

Status: approved implementation contract. Implementation and evaluation results must be reported separately from these requirements.

Scope: Answer Run attachments and Resources only. This change does not replace LightRAG corpus ingestion, expand admitted formats, or add local workspace PDF/Office conversion. The project is in development: remove superseded code and contracts, not compatibility aliases or legacy replay branches. This does not authorize deleting user data or unrelated VLM capabilities.

## Tool responsibilities

- `read` returns bounded text, honest extraction status, and discoverable visual targets. It never attaches image pixels or secretly invokes OCR, document rendering, or a second model to judge whether the text is sufficient.
- `view` returns verified source images, rendered PDF pages, or extracted embedded images as tool-result attachments to the vision-capable model of the Agent or Child Session that called `view`. Pixels are not automatically forwarded from a Child Session to its parent. It never asks a separate VLM to describe them.
- Default/query answering profiles support vision. Remove the inspect-based text-only answering fallback; capability validation should fail honestly rather than invent a fallback.
- Delete the `inspect` tool, its dedicated VLM invocation/prompt/result wrappers, obsolete configuration/setup guidance, allowlist entries, and superseded tests. Do not retain an alias, unused `focus` argument, or legacy output decoder. Keep unrelated retrieval-planning and corpus-sidecar VLM usage.
- Standalone images use `view`, including workspace images formerly attached by `read(path)`. A mistaken image `read` returns bounded identifying information and actionable `view` guidance, not image bytes. Existing automatic user-upload image input need not take a tool round trip.
- `view` accepts exactly one of a registered `resource_id`, an admitted anonymous public `url`, or a workspace `path`. Paths in this delivery support standalone images only. Resource locators support physical PDF pages and embedded-image handles; cursors continue bounded overviews. Public URLs share the existing Resource Registry acquisition, anonymous request policy, header allowlist, canonical identity, settled byte snapshot, and owner/security controls with `read`; there is no second downloader. URL continuation uses the returned resource identity. A `view(url)` does not first require successful text conversion.
- A PDF `view` with no page locator returns a bounded, low-resolution overview with physical page labels and an explicit actual coverage range/continuation. A selected page is rendered for detail. Never render or attach an entire document implicitly, silently skip pages, claim overview coverage is whole-document coverage, or treat thumbnails as reliable small-text transcription.
- Keep PDF rendering independent of text extraction. Current pypdfium2 rendering is not replaced by AnyDoc. Office embedded-image extraction is not a promise of whole-page/whole-slide screenshots.

## Discovery and evidence

- Resource inventory identifies useful resource types so the model can choose `read` or `view` directly.
- PDF/Office text results identify themselves as extracted text views, expose bounded visual discovery and valid viewing instructions, and retain normal text continuation.
- Expose physical PDF page count where available and known embedded-image handles/anchors without making unreliable text-line-to-page or asset-part-to-page/cell mappings. Do not rebuild page-aware PDF Markdown in this delivery merely to provide discovery. Use page overviews when the target page is unknown.
- Bound and paginate long visual inventories rather than exhausting text windows. Do not hide all remaining visuals behind truncated output.
- Nonempty Markdown or no parser exception does not prove completeness. Unknown coverage remains unknown. No extracted text is not proof that the document is blank.
- Distinguish usable text with unverified coverage, known OCR/incomplete extraction (with truthful known pages/omissions), ordinary conversion failure, and admission/safety/resource refusal. Do not fabricate partial text when the converter supplies none.
- Known OCR/incomplete extraction exposes the original viewing route where independently allowed. The agent chooses whether to view relevant pages; no automatic hosted OCR, whole-document rendering, or fallback that launders known omissions into a complete success.
- Safety/admission/resource refusals are not permission to retry another parser or renderer around the same restriction.
- Keep evidence source identity and exact visual provenance (parent resource, physical page or supported image occurrence/anchor) through tool effects, citations, durable settlement and replay. A digest deduplicates bytes, not distinct source occurrences. Embedded assets must belong to the requested resource.
- Use the actual tool-calling/consuming model's image/context budget across tool calls and replay, not fresh per-call budgets that bypass aggregate safety. Preserve existing image quality floors, bounded work and explicit failures/continuation.

## Conversion snapshots and recovery

- Persist the adopted conversion snapshot: text view, selected assets and source locators, extraction status, converter/version, fallback reason when applicable, and input/output digests.
- Subsequent reads, cursors, and recovery reuse that adopted snapshot. They do not re-run parser selection, opportunistically switch engines, or silently rebuild a different view after a dependency change.
- Reuse existing owner-scoped blob/effect settlement and retention mechanisms; no cross-owner cache or second storage platform. Do not persist failed speculative output as evidence.
- Render PDF pages on demand and settle returned derivative bytes with their source locator and digest, without eagerly pre-rendering every page. Restore those returned bytes honestly on replay.
- Legacy data/run compatibility is out of scope. Correct atomicity, concurrent access, cancellation, ownership, cleanup and recovery for new runs remain required.
- Clarification approved during Phase A: new follow-up/fork Runs replay prior-Run image attachments only through exact owner/selected-Session-lineage and originating-Run/blob-reference authorization. Adopt retained references atomically under the consuming Run fence so origin-Run cleanup cannot invalidate replay. Same owner or a shared Session alone is insufficient. Preserve original/derivative provenance and charge the consuming model's aggregate budget; missing, mismatched or unauthorized snapshots fail explicitly. This hydration does not register historical source handles or cursors as capabilities in the new Run, reparse old sources, or add legacy schema compatibility.

## AnyDoc-first adoption

AnyDoc-first with a qualified MarkItDown fallback is an explicit implementation goal, not merely a hypothetical future plugin. Adoption is gated independently by format and must not block the `read`/`view` simplification if a candidate fails.

User clarification after the paired pilot supersedes image-free-only DOCX routing: adapt text and image assets properly rather than introducing an image/no-image engine-selection branch. This authorizes the DOCX asset adaptation and its additional qualification tests; it does not retroactively claim that the earlier text-only pilot qualified those assets.

- Evaluate and integrate the exact `firecrawl-anydoc` 0.2.4 candidate (Python import `anydoc`), using already-admitted bytes and local OCR rejection. No hosted OCR, provider/model calls, private documents or telemetry-based evaluation.
- DOCX uses one intended AnyDoc-first text-and-assets route, with or without images. Adapt AnyDoc's structured image assets and distinct occurrences independently of its Markdown image links, which the pilot found missing. Remove the provisional image-free selector and its unused paths; do not substitute another heuristic image-presence router. Qualify the actual integrated contract with both image-bearing and image-free DOCX fixtures before claiming adoption.
- Preserve real asset membership, bytes/media/digests, repeated occurrences and bounded visual discovery. Expose only supported locators: package-part provenance or a null anchor is not a physical page, slide, or cell. Do not invent inline placement when linkage is unavailable, fetch external image relationships automatically, or silently drop required unmapped/unsupported visual evidence. Text and assets must belong to the same admitted and adopted source snapshot. Measure any extra structured-document parse cost honestly.
- Further user clarification requires evidence-based reassessment of PDF, PPTX, XLSX, HTML and CSV before final route decisions. Their incumbent routes are current/provisional, not proof that AnyDoc is inferior. Separate official format support, valid-document text fidelity, malformed-input/known-omission handling, missing host image adaptation, and actual combined-path cost. In particular, a partial incumbent result is not automatically better than an explicit candidate refusal, and a missing candidate image adapter does not require retaining its incumbent text engine. Record reproducible per-format evidence and the final primary/fallback recommendation; do not silently enable additional formats before that decision. PDFium page rendering remains independent; whole Office-page rendering and expansion of the admitted-format set remain out of scope.
- Run a small reproducible paired pilot, approximately 16–20 generated or publicly redistributable fixtures with semantic gold: text/scanned/mixed PDF, text-plus-image regions, figures and repeated images, DOCX numbering/tables, PPTX slides/notes, XLSX display values/anchors, and malformed/empty/safety/error cases. Check critical text/numbers/page or slide coverage/tables/assets/provenance before measuring complete conversion/read latency and process peak memory. The incumbent output is not the gold standard.
- Verify supported current-platform CPython 3.14 import/conversion and report any platform not actually tested. Do not claim Linux/platform support from wheel metadata alone, widen platform promises, or claim ONNX removal while retaining MarkItDown.
- Only demonstrated, quality-qualified formats may become AnyDoc-first. A quality failure keeps the direct old route, not success-dependent guessing or a fake fallback. Report enabled and deferred formats and actual evidence. Do not invent performance thresholds or statistically reliable p95 from a small sample. If costs/quality leave adoption ambiguous, ask the supervisor before enabling.
- At most one MarkItDown fallback for a recoverable ordinary parsing/init failure (or qualified empty result), after the first operation has ended and within remaining total limits. No reverse loop or speculative parallel parsing.
- Preserve typed terminal classifications: NeedsOcr/known incomplete cannot become complete via fallback; ResourceLimit/UnsafeArchive/admission refusal/total budget exhaustion/OOM do not fall back. Timeout while native work is still running does not launch another parser. Synchronous native work releasing the GIL is not cancellable by cancelling a Python await.
- Keep candidate resource caps plus current host preflight, not a replacement for host safety. Do not broaden URL/path access or let hosted text extraction masquerade as original visual bytes.
- MarkItDown is intentional retained functionality, not a legacy compatibility alias. Do not remove pypdfium2/openpyxl on the assumption AnyDoc replaces their responsibilities.

### Approved bounded route decision after per-format reassessment

The supervisor approved PDF and XLSX adoption after the independent 17-fixture
reassessment; this is implementation scope, not whole-contract acceptance.
Current production routes are:

| Format | Primary text / independent visuals | Ordinary fallback / pending qualification |
|---|---|---|
| DOCX | AnyDoc 0.2.4 text + verified typed occurrences, with or without images | One qualified MarkItDown fallback; no image-presence dispatch |
| PDF | AnyDoc 0.2.4 text; PDFium physical-page viewing stays independent | One ordinary fallback; NeedsOcr and Unsupported are terminal known-incomplete results, never rescue-parsed |
| XLSX | AnyDoc 0.2.4 text + existing openpyxl embedded images with actual Sheet!Cell anchors | One ordinary fallback; no formula execution/recalculation or external-link fetch |
| PPTX | Direct MarkItDown, current bounded deferral | Candidate valid slides/notes/table facts pass, but referenced missing/shape-less slides can silently disappear. Adoption needs separately qualified secure OPC completeness checking and typed occurrence binding/coverage, including unusable Markdown image targets; no prototype regex checker is approved for production |
| CSV | Direct MarkItDown, current bounded deferral | Candidate preserves over-wide cells and improves BOM/multiline table shape, but silently mojibakes tested Shift-JIS. Incumbent decodes that case but truncates over-wide rows and retains BOM/raw cell newlines. A separately validated uniform host decoding/normalization policy (evaluate existing `resources/text.py::decode_text`) is required before candidate adoption |
| HTML | Direct MarkItDown | Candidate 0.2.4 does not support HTML; no guaranteed-failing candidate call |

PDF qualification includes valid multipage and local-only embedded-font
Cyrillic/CJK facts, two scanned pages, and a text-plus-raster page. The latter
provably has text, yet the candidate returns Unsupported without page metadata:
adopt no fabricated partial text and do not infer OCR pages from exception prose.
Keep unknown details unknown; independent physical-page inventory and bounded
`view` overviews remain available. The incumbent's partial text on this fixture
also omits raster facts, so it is not proof of superior completeness.

XLSX qualification includes percent/date/currency/custom display values, merged
and empty cells, authored cached formulas and uncached formulas (empty, not
recalculated), plus identical image bytes at distinct cells across sheets.
Candidate text and openpyxl images share the same admitted snapshot. Display
coverage outside these generated cases is unverified, not a promise of every
Excel number format, formula or drawing type.

These decisions are not permanent bans or a general AnyDoc-inferiority claim.
No new formats, whole Office-page renders, corpus changes or image/no-image
engine selectors are authorized. Integrated tests and measurements must be
reported separately from the earlier direct-call prototypes; small synthetic
samples do not prove production latency, arbitrary-document completeness, or
untested platforms. The full local assessment with fixed-source citations is
supplemental at `docs/research/format-route-reassessment.md`; the route table
above remains authoritative even when ignored research files are unavailable.

## Validation and review

Implement destination-shaped slices, update authoritative documentation and test the new contract rather than retaining tests for deleted behavior. Include focused coverage for schema/tool composition/children, images and URL classification, physical page discovery and pagination, conversion/error routing, asset membership and duplicate occurrences, real durable settlement/replay, snapshot reuse, cancellation and aggregate image budgets, and provider attachment projections.

Run independent fresh-context reviews on four axes: Standards/simplicity, Spec compliance, Runtime correctness/durability, and Security/evidence/budgets. Review against the pre-change `main` baseline `521dc33ae105e0e1f502b835af5eb152d9382b6e`, including added files and unstaged changes. Fix accepted findings, rerun affected checks and re-review before claiming completion. No commit, push, deployment, public publication or destructive data cleanup is authorized by this implementation contract.

Supplemental pre-implementation research is available locally at `docs/research/anydoc-markitdown-evaluation.md`; its findings are evidence and proposals, not proof of successful adoption. Its references to old `inspect` behavior are historical context, not a compatibility requirement.
