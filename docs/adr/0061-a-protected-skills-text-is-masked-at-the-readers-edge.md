# A protected Skill's text is masked at the reader's edge

A passage that is a protected Skill document's own text is replaced with `[...]` on every
surface whose reader is a user or an outside party. The Run's own model, the Session the model
re-reads, the run events as written, the observability plane, and the training export are not
touched: the bytes stay whole in the record, and only the projection a reader is handed is
masked.

The key is the text, never the carrier. Fingerprints of every protected document are compared
against each reader-facing projection, and a candidate match is verified character by character,
so what is replaced is a real run of the document's own words with its real span. The same bytes
are reached through `load_skill`, through `bash cat`, through a copy read back, through a child
agent's conclusion, through a search hit, through a retrieval passage, through a tool's
arguments, and through the model's own answer; a rule keyed on a tool name catches one of them.

## Status

Accepted and implemented for the surfaces that carry prose: the answer projection (history, SSE
`done` frame, REST, MCP), its retrieval contexts **and its source chunks**, the transcript and
its full-text search, the transcript tail REST and MCP serve, the SSE `token` stream on both the
browser and REST transports, the live `reasoning` stream on both transports and the Inspector
activity page's bounded reasoning, Artifact publication's words, Profile Memory's reader
projection, a child agent's summary, questions and control text, and a conversation's goal text.

Not yet: the data-subject export, byte-shaped surfaces (held artifact bytes, the live browser's
frames, corpus retrieval results), and cross-turn accumulation accounting. They are the next slice;
until they land, the rule is stated here so no surface is added against it.

## Context

[ADR 0046](0046-a-skills-documents-are-its-authors-to-publish.md) decided that a Skill's files
belong to their author: a Run publishes what it produced, not the material it reasoned with. It
enforced that at publication, by content digest, on the bytes a workspace file holds.

The reader-facing transcript grew a narrower rule of its own: a `load_skill` result whose header
named a protected document was replaced with a placeholder. The key was the tool name, so the
same bytes reached a reader unmasked by every other road — a command's output, a copy read back,
a child agent's conclusion, a search hit, a retrieval passage rendered in the source panel, the
model's own answer, and its arguments to the next tool.

Two decisions from [ADR 0043](0043-a-step-records-the-model-parameters-and-prompt-digests-it-ran-on.md)
bound what masking may do. The transcript routes project an entry by whitelist, so the record is
already wider than any reader's view; and a Run's session is what the model continues from, so
the record must keep the text a Run reasoned with.

## Decision

**Masking is a display action.** It runs when a projection is rendered for a reader and changes
nothing else. A masked span must never reach the model, the stored result, the run events, or the
training export, because a surface that could hand a Run a masked document would change what the
Run does next — and a product whose safety policy silently degrades its answers has made a worse
trade than the leak it was avoiding. One consequence is accepted and stated: publication is the
one place the Run learns of the gate, as a refused tool call that names the reason, exactly as
the digest rule already refuses an attached Skill document (ADR 0046).

**Detection is by content fingerprint.** `ProtectedTextIndex` indexes every protected document as
one rolling hash per 40-character window of a normalization both sides share, then verifies each
candidate by direct character comparison. The fingerprint proposes, the comparison disposes: a
collision cannot mask a passage that is not there, and the verified run gives the exact span to
replace. The normalization folds whitespace and case, unifies quotes and dashes, resolves HTML
character references, and removes the invisible characters a copier can insert between letters,
so a reformatted or an encoded reproduction is the same document.

**Replacement says nothing.** A verified run becomes `[...]`. The retired placeholder explained
the product's authorization policy inside a reader's pane; a reader surface states what a Run did
and, where the text is not the reader's, says nothing at all. The row that names the document
keeps its name, so a reader still knows what was read.

**The gate is a list of recipients, not a page.** Every projection whose reader is not us passes
the same function: the answer, its contexts, its source chunks, the transcript, the search hits,
the tail REST and MCP serve, the streaming frames on both transports, Profile Memory, a child
agent's conclusion and questions, a goal's text, and Artifact publication. The observability
plane, the training export, the Session and the run events are ours and keep the text.

**The stream releases only what it has cleared.** Answer text arrives as frames, so a run of
protected text can straddle two of them. The streaming projection holds back the last
`FINGERPRINT_CHARS + 2` **visible** characters — visible because the index matches normalized
text, where whitespace has collapsed and invisibles are gone, and a raw-character bound can
shrink below one window — and releases a prefix only once no match can still begin in it. A run
that begins inside the held text is held whole until the text shows where it ends; only then is
it replaced, and a run shorter than `MIN_MASK_CHARS` is released as the words it is. Detection
therefore uses 40 characters (a run must be seen early enough to be held) while replacement uses
80 (the reader-facing minimum), which keeps the live draft and the finished answer saying the
same thing.

**What counts as a reproduction differs by surface, deliberately.** A reader surface replaces a
run of `MIN_MASK_CHARS` (80) normalized characters. A deliverable is refused at
`MIN_PUBLISHED_REPRODUCTION_CHARS` (320): publication ships a file whole or not at all, so the
bar is higher and the cost of a false positive — a withheld delivery — is carried by a tool
result that says why.

**Publication reads words, not just bytes.** The digest gate reads a file's bytes; the content
gate reads what a file says. A PDF's text is extracted through the renderer the image already
ships, an Office Open XML file's parts are read as the words their runs spell, and a text file is
decoded as UTF-8 or UTF-16 (with or without a byte-order mark). A tagged page's markup is
removed before its text is read, because a reader of the rendered page never sees the tags. The
legacy OLE formats (`.doc`, `.xls`, `.ppt`) have no reader in this process and are named as a
limit below rather than pretended to be covered.

**Quotation is legitimate and stays.** An answer may name a Skill, repeat its description, use
its terms, or quote a short line; a reader surface replaces a reproduction — a paragraph, a table
row, a copied script body — not a citation. A `SKILL.md`'s frontmatter is the routing surface the
catalog publishes, so the index starts after it, and where that block ends is the parser's own
rule rather than a second copy of it.

## Considered options

- **Mask before persistence** (the `flush_tokens` seam). Rejected. It would rewrite
  `result.answer`, which the next Round of a Goal continues from and which the training export is
  built on; the Run would answer differently because a display rule fired. It also cannot be
  undone, while a projection can.
- **Correction rewrite** — detect a reproduction in the answer, hand the model an instruction to
  rephrase, resend. Rejected for the same reason: it turns a display fact into a behavioral one,
  spends a model call, and on a false positive rewrites an answer that was correct.
- **Key masking on the tool name.** Rejected: it is what the current rule does, and it catches one
  of eight roads.
- **Semantic or embedding similarity.** Rejected. It is probabilistic where this must be exact,
  needs a model in the read path, and its false positives land where a reader sees `[...]` for no
  reason. The product's rule is that a Skill's *text* is the author's; the method's idea is the
  product.
- **Store only fingerprints and drop the text.** Rejected: a Run continues from its Session, so
  the document it read must be there for the next turn.
- **Explain the omission in the product.** Rejected: a reader surface that explains an
  authorization policy has made the policy the subject. `[...]` is the whole sentence.
- **Refuse every binary deliverable.** Rejected: a report a Run produced is a delivery, and its
  format is the user's choice. The gate reads the formats whose words it can reach; the formats
  it cannot are stated as a limit.

## Consequences

- A reader who used to see text can now see `[...]`. The cost of a false positive is exactly that
  and nothing more: no answer is changed, no Run behaves differently, no event is lost.
- The index costs one build per process — about 0.27 s for the 2.2 M characters of the built-in
  library — paid on the first reader projection or the first publication that reads words, then
  amortized. A reader projection scans only the text it is about to show: about 2.5 ms for a 4 KB
  answer and about 250 ms for a 400 KB one that reproduces a passage (44 ms when it does not).
  The operator-global root is re-read at most every 30 s, a root that cannot be read keeps the
  last index that could be built rather than unmasking every operator document, and a followed
  transcript page is served from a bounded cache after its first projection.
- A text with no reproduction is answered without building one: a publication reads a file in
  64 KB overlapping windows and stops at the first verified reproduction, so a thirty-megabyte
  deliverable costs a bounded walk rather than memory proportional to the file.
- A text that matches everywhere is bounded rather than quadratic: a query run of identical
  windows is one question to the index, and one window is compared against at most 64 document
  postings.
- The record keeps the text, so a database that is copied out hands over what masking protected.
  This is the boundary every display DLP shares: the rule defends the product's surfaces, not the
  disk. Static encryption is where that threat is met, and it is not this decision.
- **Stated limits, not covered:** a paraphrase, a translation, or a functional reimplementation of
  the method; a protected document shorter than two fingerprint windows; a text segmented into
  runs shorter than the reader minimum by an inserted character the normalization does not drop
  (some combining marks, a decomposed accent that canonical composition would fold); a legacy OLE
  deliverable; a PDF whose text objects split a passage below the minimum inside the extraction
  the renderer returns, or an Office Open XML file whose words sit past the extraction bound; a
  page whose text is only reachable through script; an answer that writes a passage as HTML
  character references while it streams (the settled answer's gate reads the references as their
  characters, and the streaming holdback counts raw visible characters, so a deliberately encoded
  passage can be released piece by piece before the settled gate sees it); and a client that
  reconnects mid-stream and therefore loses the carried tail, bounded by the replacement minimum
  (a run whose first eighty characters were the part before the reconnect is shown as a finished
  short run by the draft, and the settled answer still replaces it).
- The 80/320-character minima and the 40-character window are starting values, to be calibrated on
  the deployment's own traffic; the mechanism records nothing about a match, so that calibration
  is a follow-up decision.
