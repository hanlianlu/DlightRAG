# A video Artifact plays, and a video link becomes a card

An Agent that produces or receives a video has to be able to show it. This
decision states what "show" means on the Answer surface — the file plays in a
native player, a link to someone else's video becomes a card — and what it
deliberately does not mean: no ingestion, no embed, no loosened sanitizer.

## Status

Accepted and implemented: publication, delivery, both Answer surfaces, address
auto-detection, and video link cards. See [Consequences](#consequences) for the
bounds and the accepted residuals.

## Context

A video is the first Artifact class the product cannot read. The binary
converters admit HTML, CSV, PDF, DOCX, PPTX, and XLSX for text; images are
compared pixel by pixel against a model's image budget; every other media type
is published as `application/octet-stream` with `download` presentation, which
is what a video was until now. Three facts decided the shape of this slice:

1. **The browser is the only decoder we have.** No ffmpeg, no container parser,
   and no plan to add either to the service image. Whatever a `<video>`
   element cannot play is not playable here, and that is a fact to present, not
   a gap to close.
2. **The player must not live inside the sandboxed HTML frame.** The Active HTML
   Artifact boundary already declares `media-src data: blob:` yet forces
   `fullscreen`, `autoplay`, and `picture-in-picture` to `'none'`. The Answer
   surfaces render a player in the main document instead, where the native
   controls keep fullscreen.
3. **A remote image already loads in the Answer.** Verification showed that
   `![x](https://example.com/a.png)` reaches the browser as an `<img>` and is
   fetched from that origin, so "the surface never makes a third-party request"
   was never true. Any later decision about link cards must start from that
   fact rather than from an invariant that does not hold.

The alternative to this decision is teaching the Agent to understand video
(frame extraction, transcripts, model analysis). That was rejected for now:
it depends on an ingestion path this product does not have and on a scope
(screen capture, recorder consent, per-user recording) that is not defined.

## Decision

### A published video is a `video` presentation

Publication admits `.mp4`, `.m4v`, `.mov`, and `.webm`, mapping each to the
container the extension names: `video/mp4`, `video/quicktime`, or `video/webm`.
All four publish as `presentation: "video"`. Placement stays the Answer's own
statement, exactly as it is for images: the Model writes the reference, and a
captionless `![…](artifact:clip.mp4)` is what makes it play in place. A video the
Answer attaches without placing keeps the Artifact card, because the framework's
trailing affordance is not a request to spend the reading column on a player.

### Identification is content-based and intentionally weaker than parsing

The bytes must identify as the declared media type. Identification is
`magika`, already present as the binary-converter dependency and now a declared
one. This is a weaker bar than published images (Pillow `verify()`), Office
documents (OPC entry preflight), and PDFs (page count): a container whose header
survives truncation is admitted. What a viewer then sees is the element's own
error path, and the caption keeps an escape hatch. A structural container parser
is deliberately not introduced.

### Delivery serves video inline, and the player streams

Video joins images and PDFs as a `safe_inline` media type, so the Artifact
endpoint answers with its own media type, `Content-Disposition: inline`, and
`Accept-Ranges: bytes`. The player points at that authenticated same-origin URL;
bytes are never inlined into the Answer document, and seeking costs one range
request. `?download=1` still forces `application/octet-stream` with
`Content-Disposition: attachment`.

### The Answer renders two different things, chosen by the Model

An inline placement (`![](artifact:clip.mp4)`) renders a native
`<video controls preload="metadata" playsinline>` inside a figure, sized by the
answer column with the same 70vh ceiling images already use; fullscreen belongs
to the browser. The figure's caption always carries "open in a new tab" and
"download", so a container this browser cannot decode still has a path. A video
the Answer never places renders the Artifact card, and the framework's trailing
affordance for it is a plain reference rather than an inline one — deliberately
unlike images, whose affordance is inline. The Canvas plays the same URL in its
own player.

### A linked video becomes a card, not an embed

A link whose page declares itself a video — Open Graph `og:type` of `video.*`,
or the presence of `og:video` — becomes a card built from the page's own
`og:title`, `og:image`, and `og:description`. This is a scope, not a site
allowlist: YouTube (`og:video:url` to `youtube.com/embed/…`) and Bilibili
(`og:video` to `player.bilibili.com/player.html`) declare it identically, while
an ordinary page (`og:type: website`) stays a plain link. A bare address the Model
writes becomes an ordinary hyperlink whether or not the card rule matches it, so
the two features compose rather than compete: every address is readable and
clickable, and only a verified video link is additionally presented as a card.
Citation links and source links never become cards.

Autolinking is deliberately narrower than GFM. It admits an address that names
its own scheme and refuses to guess one from a top-level domain, because this
product's own vocabulary is full of words that end in a real one — `report.md`,
`build.sh`, `clip.mov`, `archive.zip` — and each would otherwise become a link to
`http://report.md`. Email autolinking is off as well: the fragment sanitiser
admits only `http`, `https`, `data`, and `blob`, so a `mailto:` href would be
dropped and leave an inert link behind. The renderer also bounds an address at
CJK text, which is what a linkify rule reads straight through: CJK punctuation
always ends the address, a CJK run that opens a path or query component belongs
to it, and a CJK run that merely continues a segment leaves the address
ambiguous — an ambiguous address stays text, because a wrong link is worse than
a plain one. Autolinking applies to what the Model writes (answers and Markdown
Artifacts); a source chunk quotes a document, so a bare address inside a
quotation stays as the parser produced it. One edge is accepted rather than
solved: an address that continues a preceding token (`blob:https://…`,
`see:https://…`) leaves the outer token as text and links the address it
contains, because refusing it needs the pass that reads that context, and the
Chinese sentence above needs the pass that does not.

Clicking a card leaves the application and plays at the source. No third-party
`<iframe>` is introduced, and the answer-sanitizing allowlists are unchanged. The
card carries the page's own cover image, title, description, and site name; the
browser reads that image from the platform's CDN, exactly as it already reads a
remote image an answer writes.

Reading the pages happens during the Answer's own settlement, under bounds: at
most three addresses an answer writes are read, each with a four-second deadline
and a 256 KiB ceiling, and they are read together so an answer with three links
costs about one deadline rather than three. An address that is unreachable, too
slow, declares nothing, or answers with anything but HTML stays the plain link
the Model wrote — a card is never a reason to fail an answer. A cited source is
this Answer's authority rather than someone else's page, so it never becomes a
card.

## Considered options

- **Loosen the sanitizer to allow `<video>`/`<iframe>` in answer HTML.** Rejected:
  the typed-part renderer already exists for exactly this (a non-markdown part
  is rendered by Lit from validated data), so loosening the HTML allowlist buys
  nothing and costs the invariant that sanitized HTML is the only same-DOM sink.
- **Reuse the Active HTML Artifact frame for playback.** Rejected: it is a
  deliberate consent boundary with `frame-src`, `object-src`, and `fullscreen`
  closed; a self-contained `data:video/mp4` document does pass publication, and
  that remains a demo path rather than the product surface.
- **Generate frames, transcripts, or model descriptions of the video.** Out of
  scope here; recorded above as the deliberately deferred alternative.
- **A site allowlist for video links.** Rejected in favour of the declaration
  the page itself publishes; a provider that declares nothing visible (Bilibili
  served to a plain HTTP client one moment and OG the next) degrades to a link.
- **A fixed height for the inline player.** Rejected: the ceiling that images
  already use (70vh) expresses "fits the reading column", and fullscreen is the
  user's control, not ours.

## Consequences

- **Link autolinking is implemented; link cards are not.** An answer and a
  Markdown Artifact autolink an address that names its own scheme
  (`linkify-it-py`, the engine markdown-it-py's own rule requires), bounded at
  CJK text as described above. A source chunk keeps its quotation as written. No
  card surface exists yet, so a video link is still presented as the plain link
  the autolinker or the Model produced.
- **Weaker video validation is a recorded residual,** not an oversight; the
  escape hatch in the caption is what a truncated or undecodable file gets.
- **A browser that cannot decode the codec** (for example HEVC where the user's
  platform lacks it) shows an empty player with the caption's open/download
  links. That is accepted: choosing a codec is the producer's decision.
- **Non-browser clients** (MCP, REST with bearer credentials) receive the
  download URL rather than a stream, because the inline path relies on the
  browser's cookie session. This mirrors images and PDFs.
- Publication, delivery, wire validation (REST and browser), the frontend
  Artifact union, both Answer surfaces, and the catalogs are updated together:
  a `video` presentation that any one of them rejected would be unusable.
- Video still has no reading path: `read(url)` classifies it as opaque, so a
  video the Agent obtains is a deliverable, never citable evidence.
