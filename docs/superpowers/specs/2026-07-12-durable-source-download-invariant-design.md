# Durable Source Download Invariant

Date: 2026-07-12
Status: approved for implementation

## 1. Decision

Every successfully ingested document must have both:

- a stable provenance identity (`source_uri`); and
- a server-resolvable download locator (`download_locator`).

`retain_source_file` selects where the locator points. It never selects whether
the document is downloadable.

- Retained sources use the workspace-local retained file as their locator.
- Non-retained sources use a validated durable remote locator.
- If neither is available, that document is rejected before parser ingestion.

The system must not silently retain a source when the caller explicitly chose
`retain_source_file=false`. It must not accept a provenance-only source and make
the Web UI hide the missing action.

## 2. Current defects

### 2.1 Custom connector provenance is mistaken for a download locator

`RAGService.aingest_source()` currently accepts arbitrary stable identities such
as `bynder://assets/asset-123`. With remote retention disabled, that URI is
written into metadata as `file_path`, while the temporary parser file is
deleted. `SourceUrlResolver` intentionally rejects the unknown scheme, so the
source has provenance but no download path.

### 2.2 Signed HTTPS fetch URLs are mistaken for durable download locations

URL ingestion strips query and fragment from the fetch URL before persisting it.
That protects credentials but can remove the token required to fetch the source
again. The resulting URL is a useful identity, not necessarily a usable
download locator.

### 2.3 One field currently carries three meanings

`PreparedIngestFile.metadata_path`, document metadata `file_path`,
`SourceReference.path`, and `SourceReference.url` blur parser location,
provenance identity, storage locator, and HTTP projection. Optional URL handling
then leaks into the Source panel as an apparently valid no-download state.

## 3. Goals

- Fail closed before ingesting a document that cannot later be downloaded.
- Preserve provenance independently from where original bytes are served.
- Never persist an implicit secret-bearing or expiring HTTPS query string.
- Keep local, S3, Azure, HTTPS, and custom SDK sources behind one download
  projection boundary.
- Preserve existing ingested local and supported remote data through an
  additive metadata migration.
- Make the Web Source panel render Download for every authorized source.
- Remove stale `path`/`url` SourceReference aliases and the conditional download
  branch rather than retaining compatibility shims.

## 4. Non-goals

- No retrieval, ranking, citation-index, gallery, or image-budget changes.
- No Files panel download action in this iteration.
- No generic plugin registry for arbitrary URI schemes.
- No silent local-retention fallback.
- No promise that a third-party object can never be deleted or access credentials
  can never be revoked. The invariant is that DlightRAG persists a valid locator
  and can attempt the authorized download through its canonical endpoint.
- No storage of signed fetch tokens as a shortcut for durability.

## 5. Canonical model

### 5.1 SourceDocument

`SourceDocument` gains:

```python
source_uri: str | None
download_uri: str | None
```

`source_uri` is identity/provenance and may use a connector-specific scheme.
`download_uri` is an explicit durable remote locator and is restricted to the
server-supported download schemes.

`aingest_source()` also accepts an optional `download_uri_for_key` callback for
connectors that stream lightweight document descriptors rather than populating
each `SourceDocument.download_uri`.

The per-document value wins over the callback. There is no alias from
`source_uri` to `download_uri` for custom connectors.

### 5.2 PreparedIngestFile

Delete the ambiguous `metadata_path` field. Replace it with:

```python
source_uri: str
download_locator: str
```

`parser_path` remains the temporary or retained local file LightRAG parses.

### 5.3 Stored metadata

Add DlightRAG-owned PostgreSQL metadata columns:

```text
source_uri TEXT
download_locator TEXT
```

`file_path` remains because LightRAG and existing delete/retry workflows use it;
new product code must not infer provenance or download semantics from it.

The schema migration backfills both new columns from `file_path` for existing
rows. New ingests always write the explicit values. Retrieval enrichment reads
the new columns only; it does not keep a runtime fallback to `file_path`.

Existing rows whose backfilled locator uses an unsupported scheme are invalid
and must be re-ingested. The Web projection surfaces an invariant error instead
of hiding Download.

### 5.4 SourceReference

The public source fields become:

```python
source_uri: str
download_url: str | None
```

Remove the ambiguous `path` and `url` fields without deprecated aliases.

The domain object retains an excluded internal `download_locator` so HTTP
adapters can project a principal-authorized `/files/raw/...` URL without leaking
raw local paths or provider locators. `download_url` may remain absent in a
transport-neutral in-process result, but it is required at the Web rendering
boundary.

## 6. Ingest validation

### 6.1 Retained source

When `retain_source_file=true`:

- materialize into `__remote_sources__/<source_type>/...`;
- set `download_locator` to that contained local path;
- keep the independent remote `source_uri` for provenance.

### 6.2 Non-retained built-in source

When `retain_source_file=false`:

- S3 uses its canonical `s3://bucket/key` locator;
- Azure uses its canonical `azure://container/blob` locator;
- URL ingestion uses a public HTTPS locator only when it is explicitly durable.

A queryless, fragmentless public HTTPS fetch URL is implicitly durable. A fetch
URL containing query or fragment is treated as ephemeral. It requires either:

- `retain_source_file=true`; or
- an explicit queryless, fragmentless `download_uri`.

`source_uri` never substitutes for the missing explicit locator in this case.

### 6.3 Non-retained custom connector

A custom `AsyncDataSource` must provide `SourceDocument.download_uri` or
`download_uri_for_key`. The locator must use a supported scheme. Otherwise that
document fails before `amaterialize_document()` and before LightRAG parser
ingestion.

The per-document error is:

```text
retain_source_file=false requires a durable download_uri for source <safe source id>; provide download_uri/download_uri_for_key or enable retain_source_file
```

Batch ingest preserves its current bounded, per-document error aggregation;
invalid documents are not partially inserted.

## 7. Durable locator validation

One focused validator is the source of truth for non-local locators.

Accepted:

- well-formed `s3://bucket/key`;
- well-formed `azure://container/blob`;
- public HTTPS without credentials, query, or fragment.

Rejected:

- unknown schemes such as `bynder://`;
- `file://`;
- HTTP;
- HTTPS credentials, query, or fragment;
- malformed provider URIs;
- local absolute/traversal paths supplied as remote locators.

`source_uri` is validated only as a non-empty, NUL-free provenance identity and
may use a connector-specific scheme.

## 8. URL ingest contract

REST, MCP, CLI, and Python request models add:

```text
download_uri
download_uris
documents[].download_uri
```

They follow the same single/batch cardinality rules as `source_uri` and
`source_uris`. There are no deprecated names or aliases.

`URLDataSource` keeps fetch URLs private to the materialization step. It emits:

- stable provenance via `source_uri`;
- a separate validated locator via `download_uri`.

It never persists a signed fetch URL or silently converts a stripped URL into a
download guarantee.

## 9. Retrieval and Web projection

Metadata enrichment injects dedicated internal keys:

```text
source_uri
source_download_locator
```

Source building uses only those keys. The HTTP adapter resolves the locator to
the existing authenticated `/files/raw/{path}` endpoint.

The Source panel:

- always renders a real Download link for every source it receives;
- does not inspect extension, filename, workspace selection, or locator scheme;
- does not render a disabled or missing-download state;
- keeps the download link as a sibling of the accordion button;
- preserves `aria-label="Download source"` through sanitization.

An unresolved locator at the Web boundary is a structured invariant failure and
must be logged/traced. It is not converted into a source row without Download.

## 10. Authorization

`/files/raw` remains the enforcement point for
`workspace.download_source`. The default reader/editor/admin presets already
include that action.

Download URLs for federated results must encode the source's actual workspace;
they must not rely on Search-in scope, Files-in target, or configured default.

Authorization denial is distinct from a missing locator. The ingestion record
remains valid even when a particular principal cannot download it.

## 11. Observability

Add structured outcomes without high-cardinality URI labels:

```text
source_download_locator_kind=local|s3|azure|https
source_download_locator_outcome=accepted|missing|unsupported|ephemeral
source_download_projection_outcome=resolved|unauthorized|invalid
```

Logs may include a safe document display name and source scheme. They must not
log HTTPS query strings, credentials, signed URLs, or full sensitive locators.

## 12. Cleanup

Delete:

- `PreparedIngestFile.metadata_path`;
- `SourceReference.path`;
- `SourceReference.url`;
- source-template `{% if src.url %}` handling;
- tests and documentation that treat `source_uri` as a download locator;
- examples that ingest a signed/custom source with retention disabled but no
  durable `download_uri`;
- runtime fallbacks from missing new metadata to legacy `file_path`.

Keep:

- LightRAG-facing `file_path` storage and delete/retry behavior;
- local containment checks;
- S3/Azure signed redirect generation;
- HTTPS redirect safety validation;
- current citation ordering, chunks, highlights, images, and lightbox behavior.

## 13. Acceptance tests

### Contract

- Request schemas expose `download_uri`/`download_uris` with strict cardinality.
- Unknown stale field names fail through `extra="forbid"`.
- `.md`, unknown extensions, PDFs, and Office files have identical download
  projection rules.

### Ingestion

- Local and Web upload ingests store local download locators.
- Retained S3/Azure/URL/custom sources store local download locators and remote
  provenance URIs.
- Non-retained S3/Azure store provider locators.
- Non-retained queryless public HTTPS stores the public locator.
- Non-retained signed/query-bearing HTTPS without explicit locator fails before
  parser materialization.
- The same URL succeeds with retention enabled.
- A custom `bynder://` source without `download_uri` fails before parser
  materialization.
- The same custom source succeeds with retention enabled.
- The same custom source succeeds with a valid separate HTTPS/S3/Azure
  `download_uri`.
- Unsupported, secret-bearing, malformed, or traversal locators fail closed.

### Metadata and migration

- Fresh metadata stores distinct source identity and download locator.
- Existing rows are backfilled once by an idempotent migration.
- Retrieval enrichment uses the new columns and never parser basename fallback.

### Projection and UI

- REST retrieve, REST streaming answer, REST non-stream answer, and Web done
  projection resolve downloads from the same locator.
- Federated sources encode their actual workspace.
- Web rendering fails loudly on an unresolved source locator.
- Every rendered source row contains one keyboard-reachable Download link,
  including Markdown.
- Sanitized HTML preserves the accessible name.
- The link remains outside the document accordion button.

### Regression

- Source citation filtering, accordion behavior, semantic highlights, images,
  MathJax, gallery/lightbox, and answer streaming remain unchanged.
- File deletion and retry continue using LightRAG-compatible `file_path`.
- No raw local download locator appears in public JSON.
