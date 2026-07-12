# Durable Source Download Invariant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every successfully ingested document carry an independent provenance identity and a durable server-resolvable download locator, rejecting custom or signed-URL sources that cannot satisfy the invariant.

**Architecture:** Separate `source_uri` from `download_uri` at the source boundary and from `download_locator` in stored metadata. A focused source-contract module owns provenance and durable remote-locator policy; ingestion fails before materialization when retention is disabled and no valid locator exists. Citation-domain sources retain required internal workspace/locator fields, while adapters convert them to a distinct public `SourceReferencePayload` with an authenticated workspace-scoped `download_url`; the Web Source panel treats missing HTTP projection as an invariant failure.

**Tech Stack:** Python 3.14, Pydantic v2, FastAPI/Starlette, PostgreSQL/asyncpg migrations, Jinja2, pytest/pytest-asyncio, TypeScript frontend static contracts.

## Global Constraints

- Preserve retrieval, ranking, citation indexing, semantic highlights, Gallery/lightbox, image budgeting, and answer streaming behavior.
- Do not add a Files panel download action.
- Do not add a generic custom-URI plugin registry.
- Never silently retain a remote source when `retain_source_file=false`.
- Never persist an implicit HTTPS query string, fragment, credential, or signed token as a download locator.
- Keep LightRAG parser `file_path` and PostgreSQL legacy `file_path` for compatibility, but route failed-document retry through the new source/download metadata contract.
- Remove `PreparedIngestFile.metadata_path`, `SourceReference.path`, and `SourceReference.url` without deprecated aliases or compatibility shims; never ask FastAPI to revalidate a locator-free dump against the internal model.
- Keep all public request models `extra="forbid"`.
- Stage and commit only task-owned files; do not include local `.env`, config, Visual Companion, or unrelated worktree changes.

---

## File Structure

- Create `src/dlightrag/sourcing/source_contract.py`: one policy boundary for validating provenance identities and durable S3, Azure, and queryless public HTTPS download URIs.
- Modify `src/dlightrag/sourcing/base.py`: add `SourceDocument.download_uri`.
- Modify `src/dlightrag/sourcing/url.py`: keep fetch URLs private and expose a separate durable locator.
- Modify `src/dlightrag/core/client_contracts.py` and `src/dlightrag/core/client_requests.py`: strict `download_uri`/`download_uris` request contract.
- Modify `src/dlightrag/core/ingestion/engine.py`: replace `metadata_path` with explicit provenance/download fields.
- Modify `src/dlightrag/core/service.py`: resolve and enforce the invariant before remote materialization.
- Modify `src/dlightrag/core/retrieval/metadata_fields.py` and `src/dlightrag/storage/pg_metadata_index.py`: additive source/download columns and idempotent backfill.
- Modify `src/dlightrag/citations/schemas.py` and `src/dlightrag/citations/source_builder.py`: separate the internal source identity from the public source payload.
- Modify `src/dlightrag/core/client_payloads.py`, REST routes/events, and Web answer projection: adapter-owned `download_url` generation.
- Modify `src/dlightrag/web/templates/partials/source_panel.html` and `src/dlightrag/web/safe_html.py`: unconditional authorized Download action with preserved accessible name.
- Modify MCP/CLI/docs and targeted tests to expose one canonical contract and remove stale examples.

---

### Task 1: Source Identity and Durable Download Policy

**Files:**
- Create: `src/dlightrag/sourcing/source_contract.py`
- Modify: `src/dlightrag/sourcing/base.py:14-25`
- Test: `tests/unit/test_download_locator.py`
- Test: `tests/unit/test_url_source.py`

**Interfaces:**
- Consumes: existing `parse_remote_uri()` and `validate_public_https_url()` source policies.
- Produces: `SourceDownloadContractError`, `validate_source_uri(value: str) -> str`, `local_source_uri(workspace: str, relative_path: str | Path) -> str`, `validate_download_uri(value: str) -> str`, `implicit_https_download_uri(fetch_url: str) -> str | None`, and `SourceDocument.download_uri`.

- [ ] **Step 1: Write failing locator-policy tests**

```python
from __future__ import annotations

import pytest

from dlightrag.sourcing.source_contract import (
    implicit_https_download_uri,
    local_source_uri,
    validate_download_uri,
    validate_source_uri,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("s3://bucket/docs/report.pdf", "s3://bucket/docs/report.pdf"),
        ("azure://container/docs/report.pdf", "azure://container/docs/report.pdf"),
        ("https://cdn.example.com/docs/report.pdf", "https://cdn.example.com/docs/report.pdf"),
    ],
)
def test_validate_download_uri_accepts_durable_supported_locators(value: str, expected: str) -> None:
    assert validate_download_uri(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "bynder://asset/report.pdf",
        "file:///tmp/report.pdf",
        "http://example.com/report.pdf",
        "https://user:secret@example.com/report.pdf",
        "https://example.com/download?id=1",
        "https://example.com/report.pdf#page=2",
        "s3://bucket",
        "azure://container",
        "s3://bucket/../secret.pdf",
        "s3://bucket/%2e%2e/secret.pdf",
        "azure://container/folder/%2E%2E/secret.pdf",
        "s3://user:secret@bucket/report.pdf",
    ],
)
def test_validate_download_uri_rejects_non_durable_or_unsupported_locators(value: str) -> None:
    with pytest.raises(ValueError, match="durable download_uri"):
        validate_download_uri(value)


def test_implicit_https_download_uri_never_persists_query_or_fragment() -> None:
    assert implicit_https_download_uri("https://example.com/report.pdf") == (
        "https://example.com/report.pdf"
    )
    assert implicit_https_download_uri("https://example.com/report.pdf?sig=secret") is None
    assert implicit_https_download_uri("https://example.com/report.pdf#page=2") is None


@pytest.mark.parametrize("value", ["", "\x00", "cms://asset/\x00secret"])
def test_validate_source_uri_rejects_empty_or_nul_values(value: str) -> None:
    with pytest.raises(ValueError, match="source_uri is invalid"):
        validate_source_uri(value)


def test_validate_source_uri_allows_connector_specific_identity() -> None:
    assert validate_source_uri("bynder://asset/1") == "bynder://asset/1"


def test_local_source_uri_is_workspace_relative_and_path_safe() -> None:
    assert local_source_uri("research", Path("reports/Q2 results.md")) == (
        "local://research/reports/Q2%20results.md"
    )
```

- [ ] **Step 2: Run the tests and verify the new policy module is absent**

Run: `uv run pytest tests/unit/test_download_locator.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'dlightrag.sourcing.source_contract'`.

- [ ] **Step 3: Implement the focused validator and SourceDocument field**

```python
# src/dlightrag/sourcing/source_contract.py
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlsplit

from dlightrag.sourcing.uri import parse_remote_uri
from dlightrag.utils import normalize_workspace


class SourceDownloadContractError(ValueError):
    """A safe client-visible source/download invariant failure."""


def validate_source_uri(value: str) -> str:
    if not value or "\x00" in value:
        raise ValueError("source_uri is invalid")
    return value


def local_source_uri(workspace: str, relative_path: str | Path) -> str:
    safe_workspace = normalize_workspace(workspace)
    path = PurePosixPath(Path(relative_path).as_posix())
    if (
        not safe_workspace
        or not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or "\x00" in path.as_posix()
    ):
        raise ValueError("local source identity is invalid")
    return f"local://{quote(safe_workspace, safe='')}/{quote(path.as_posix(), safe='/')}"


def validate_download_uri(value: str) -> str:
    candidate = value.strip()
    try:
        parsed = urlsplit(candidate)
        if parsed.scheme in {"s3", "azure"}:
            parse_remote_uri(candidate)
            decoded_path = PurePosixPath(unquote(parsed.path))
            if (
                parsed.query
                or parsed.fragment
                or parsed.username
                or parsed.password
                or ".." in decoded_path.parts
                or "\\" in unquote(parsed.path)
            ):
                raise ValueError
            return candidate
        if parsed.scheme == "https":
            from dlightrag.sourcing.url import validate_public_https_url

            validate_public_https_url(candidate)
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError
            return candidate
    except ValueError as exc:
        raise ValueError(
            "durable download_uri must be a well-formed s3://, azure://, "
            "or credential-free queryless public https:// URI"
        ) from exc
    raise ValueError(
        "durable download_uri must be a well-formed s3://, azure://, "
        "or credential-free queryless public https:// URI"
    )


def implicit_https_download_uri(fetch_url: str) -> str | None:
    parsed = urlsplit(fetch_url)
    if parsed.query or parsed.fragment:
        return None
    return validate_download_uri(fetch_url)
```

```python
# src/dlightrag/sourcing/base.py
@dataclass(frozen=True)
class SourceDocument:
    key: str
    source_uri: str | None = None
    download_uri: str | None = None
    display_filename: str | None = None
    title: str | None = None
    author: str | None = None
    metadata: Mapping[str, Any] | None = None
    metadata_policy: MetadataIngestPolicy | None = None
```

- [ ] **Step 4: Run policy and source-model tests**

Run: `uv run pytest tests/unit/test_download_locator.py tests/unit/test_url_source.py -q`

Expected: all tests pass; existing URL source tests remain green.

- [ ] **Step 5: Commit the policy boundary**

```bash
git add src/dlightrag/sourcing/source_contract.py src/dlightrag/sourcing/base.py tests/unit/test_download_locator.py tests/unit/test_url_source.py
git commit -m "Add durable source download locator policy"
```

---

### Task 2: Strict URL and Client Ingest Contracts

**Files:**
- Modify: `src/dlightrag/core/client_contracts.py:46-144`
- Modify: `src/dlightrag/core/client_requests.py:54-110`
- Modify: `src/dlightrag/sourcing/url.py:25-153`
- Modify: `src/dlightrag/core/service.py:54-63,1285-1349`
- Modify: `src/dlightrag/mcp/server.py:461-520`
- Modify: `scripts/cli.py:130-165,572-590`
- Test: `tests/unit/test_client_requests.py`
- Test: `tests/unit/test_url_source.py`
- Test: `tests/unit/test_mcp_workspace_tools.py`
- Test: `tests/unit/test_cli.py`

**Interfaces:**
- Consumes: Task 1 `validate_download_uri`, `implicit_https_download_uri`, and `SourceDocument.download_uri`.
- Produces: strict `download_uri`, `download_uris`, `documents[].download_uri`, and `URLDataSource.download_uri_for_key()` across SDK/REST/MCP/CLI projection.

- [ ] **Step 1: Write failing transport and URL-source tests**

```python
def test_url_ingest_projects_download_uri_fields() -> None:
    spec = IngestSpec(
        source_type="url",
        urls=["https://fetch.example.com/a.pdf", "https://fetch.example.com/b.pdf"],
        source_uris=["cms://a", "cms://b"],
        download_uris=["https://cdn.example.com/a.pdf", "https://cdn.example.com/b.pdf"],
    )
    kwargs = ingest_kwargs_from_payload(spec)
    assert kwargs["download_uris"] == [
        "https://cdn.example.com/a.pdf",
        "https://cdn.example.com/b.pdf",
    ]


def test_url_ingest_download_uri_cardinality_is_strict() -> None:
    with pytest.raises(ValidationError, match="download_uris"):
        IngestSpec(
            source_type="url",
            urls=["https://example.com/a.pdf", "https://example.com/b.pdf"],
            download_uris=["https://cdn.example.com/a.pdf"],
        )


async def test_url_data_source_separates_fetch_identity_and_download_uri() -> None:
    source = URLDataSource(
        urls=["https://fetch.example.com/download?sig=secret"],
        filename="asset.pdf",
        source_uri="bynder://asset/1",
        download_uri="https://cdn.example.com/assets/1.pdf",
        client=_Client(),
    )
    document = (await source.alist_documents())[0]
    assert document.source_uri == "bynder://asset/1"
    assert document.download_uri == "https://cdn.example.com/assets/1.pdf"


async def test_url_data_source_does_not_derive_download_uri_from_signed_fetch_url() -> None:
    source = URLDataSource(
        urls=["https://fetch.example.com/download?sig=secret"],
        filename="asset.pdf",
        client=_Client(),
    )
    document = (await source.alist_documents())[0]
    assert document.download_uri is None
```

- [ ] **Step 2: Run focused tests and verify fields are rejected or absent**

Run: `uv run pytest tests/unit/test_client_requests.py tests/unit/test_url_source.py tests/unit/test_mcp_workspace_tools.py tests/unit/test_cli.py -q`

Expected: failures mention unknown `download_uri`/`download_uris` fields and missing `SourceDocument.download_uri` projection.

- [ ] **Step 3: Add strict request fields and cardinality validation**

```python
# src/dlightrag/core/client_contracts.py
class IngestDocument(ClientContractModel):
    path: str | None = None
    key: str | None = None
    url: str | None = None
    filename: str | None = None
    source_uri: str | None = None
    download_uri: str | None = None
    title: str | None = None
    author: str | None = None
    metadata: dict[str, Any] | None = None
    metadata_policy: MetadataPolicy | None = None


class IngestSpec(ClientContractModel):
    download_uri: str | None = None
    download_uris: list[str] | None = None

    def _validate_download_fields(self) -> None:
        top_level_present = self.download_uri is not None or self.download_uris is not None
        document_values = [
            document.download_uri
            for document in self.documents or []
            if document.download_uri is not None
        ]
        if self.source_type != "url":
            if top_level_present or document_values:
                raise ValueError("download_uri fields are only valid for URL ingestion")
            return
        if self.documents is not None:
            if top_level_present:
                raise ValueError(
                    "'download_uri'/'download_uris' and 'documents' are mutually exclusive"
                )
            return

        url_count = int(self.url is not None) + len(self.urls or [])
        if self.download_uri is not None and self.download_uris is not None:
            raise ValueError("'download_uri' and 'download_uris' are mutually exclusive")
        if self.download_uri is not None and url_count != 1:
            raise ValueError("'download_uri' can only be used with a single url")
        if self.download_uris is not None and len(self.download_uris) != url_count:
            raise ValueError("'download_uris' must match the number of urls")
```

Call `self._validate_download_fields()` as the first statement of the existing `_validate_source_fields()` validator, before any source-type manifest branch can return. Extend `ingest_kwargs_from_payload()` to copy `download_uri` and `download_uris` only for URL ingestion, testing presence with `is not None` so an explicit empty value reaches the canonical validator and fails instead of disappearing. Let `documents` serialize the per-document field through the existing strict Pydantic model.

- [ ] **Step 4: Make URLDataSource own separate fetch, provenance, and download maps**

```python
# src/dlightrag/sourcing/url.py constructor additions
self._download_uri_by_key: dict[str, str | None] = {}

if document.download_uri is not None:
    explicit_download_uri = document.download_uri
elif download_uri is not None:
    explicit_download_uri = download_uri
elif download_uris is not None:
    explicit_download_uri = download_uris[index]
else:
    explicit_download_uri = None
resolved_download_uri = (
    validate_download_uri(explicit_download_uri)
    if explicit_download_uri is not None
    else implicit_https_download_uri(url)
)
self._download_uri_by_key[key] = resolved_download_uri
self._document_by_key[key] = SourceDocument(
    key=key,
    source_uri=stable_source_uri,
    download_uri=resolved_download_uri,
    display_filename=document.display_filename,
    title=document.title,
    author=document.author,
    metadata=document.metadata,
    metadata_policy=document.metadata_policy,
)

def download_uri_for_key(self, key: str) -> str | None:
    return self._download_uri_by_key[key]
```

Update `_source_document_from_manifest()` to copy `download_uri`, pass top-level URL fields into `URLDataSource`, and add MCP/CLI arguments named exactly `download_uri`/`download_uris`. Replace URLDataSource's private `_validate_source_uri()` with the shared `validate_source_uri()` import. Do not add deprecated spellings.

- [ ] **Step 5: Run all client-contract tests**

Run: `uv run pytest tests/unit/test_client_requests.py tests/unit/test_url_source.py tests/unit/test_mcp_workspace_tools.py tests/unit/test_cli.py tests/unit/test_api_server.py -q`

Expected: all tests pass, and generated MCP/OpenAPI request schemas contain the two canonical field names.

- [ ] **Step 6: Commit the public contract**

```bash
git add src/dlightrag/core/client_contracts.py src/dlightrag/core/client_requests.py src/dlightrag/sourcing/url.py src/dlightrag/core/service.py src/dlightrag/mcp/server.py scripts/cli.py tests/unit/test_client_requests.py tests/unit/test_url_source.py tests/unit/test_mcp_workspace_tools.py tests/unit/test_cli.py tests/unit/test_api_server.py
git commit -m "Separate URL fetch and download contracts"
```

---

### Task 3: Fail-Fast Remote Ingestion and Prepared Item Cleanup

**Files:**
- Modify: `src/dlightrag/core/ingestion/engine.py:40-60,200-335`
- Modify: `src/dlightrag/core/service.py:1019-1278,1351-1420`
- Modify: `src/dlightrag/core/servicemanager.py:463-490`
- Test: `tests/unit/test_service.py`
- Test: `tests/unit/test_unified_ingestion_engine.py`

**Interfaces:**
- Consumes: Task 1 locator validator; Task 2 `SourceDocument.download_uri` and `URLDataSource.download_uri_for_key()`.
- Produces: `PreparedIngestFile(source_uri: str, download_locator: str)` and matching `download_uri_for_key: Callable[[str], str | None] | None` parameters on both `RAGService.aingest_source()` and `RAGServiceManager.aingest_source()`.

- [ ] **Step 1: Write failing service tests for the two defects**

```python
async def test_custom_non_retained_source_without_download_uri_fails_before_materialize(
    test_config: DlightragConfig,
) -> None:
    materialized = False

    class BynderSource(AsyncDataSource):
        async def aiter_documents(self, prefix: str | None = None):
            yield SourceDocument(key="asset/report.pdf", source_uri="bynder://asset/1")

        async def amaterialize_document(self, document: SourceDocument, destination: Path) -> None:
            nonlocal materialized
            materialized = True

    service = RAGService(config=test_config)
    service._initialized = True
    service._ingestion_engine = MagicMock()
    service._ingestion_engine.aingest_files = AsyncMock(
        return_value={"processed": 1, "errors": [], "results": []}
    )
    result = await service.aingest_source(
        BynderSource(),
        source_type="bynder",
        retain_source_file=False,
    )
    assert result["processed"] == 0
    assert result["errors"] == [
        "retain_source_file=false requires a durable download_uri for source report.pdf; "
        "provide download_uri/download_uri_for_key or enable retain_source_file"
    ]
    assert materialized is False


async def test_signed_url_requires_retention_or_explicit_download_uri(
    test_config: DlightragConfig,
) -> None:
    class SignedUrlSource(AsyncDataSource):
        async def aiter_documents(self, prefix: str | None = None):
            yield SourceDocument(
                key="report.pdf",
                source_uri="https://fetch.example.com/report.pdf",
            )

        async def amaterialize_document(
            self, document: SourceDocument, destination: Path
        ) -> None:
            raise AssertionError("materialization must not run")

    service = RAGService(config=test_config)
    service._initialized = True
    service._ingestion_engine = MagicMock()
    service._ingestion_engine.aingest_files = AsyncMock(
        return_value={"processed": 1, "errors": [], "results": []}
    )
    result = await service.aingest_source(
        SignedUrlSource(),
        source_type="url",
        retain_source_file=False,
    )
    assert result["processed"] == 0
    assert "durable download_uri" in result["errors"][0]


async def test_custom_non_retained_source_accepts_separate_download_uri(
    test_config: DlightragConfig,
) -> None:
    class BynderSource(AsyncDataSource):
        async def aiter_documents(self, prefix: str | None = None):
            yield SourceDocument(
                key="asset/report.pdf",
                source_uri="bynder://asset/1",
                download_uri="https://cdn.example.com/assets/1.pdf",
            )

        async def amaterialize_document(
            self, document: SourceDocument, destination: Path
        ) -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"%PDF-fake")

    service = RAGService(config=test_config)
    service._initialized = True
    service._ingestion_engine = MagicMock()
    service._ingestion_engine.aingest_files = AsyncMock(
        return_value={"processed": 1, "errors": [], "results": []}
    )
    source = BynderSource()
    result = await service.aingest_source(source, source_type="bynder", retain_source_file=False)
    assert result["processed"] == 1
    prepared = service._ingestion_engine.aingest_files.await_args.args[0][0]
    assert prepared.source_uri == "bynder://asset/1"
    assert prepared.download_locator == "https://cdn.example.com/assets/1.pdf"
```

- [ ] **Step 2: Run the defect tests and verify current ingestion accepts invalid documents**

Run: `uv run pytest tests/unit/test_service.py -q -k 'custom_non_retained or signed_url_requires'`

Expected: tests fail because custom `source_uri` is accepted as `metadata_path` and the materializer runs.

- [ ] **Step 3: Replace PreparedIngestFile.metadata_path**

```python
@dataclass(frozen=True)
class PreparedIngestFile:
    parser_path: Path
    source_uri: str
    download_locator: str
    display_filename: str | None = None
    title: str | None = None
    author: str | None = None
    metadata: Mapping[str, Any] | None = None
    metadata_policy: MetadataIngestPolicy | None = None
```

Cover every local preparation flow explicitly:

- In `_aingest_local_files()` and `_aingest_local_manifest()`, wrap each staged file in `PreparedIngestFile`. Set `source_uri=local_source_uri(self.config.workspace, staged.relative_to(self._workspace_input_root()))` and `download_locator=str(staged)`.
- Change `_prepare_ingest_item(path)` to `_prepare_ingest_item(path, workspace=...)`; its raw-path fallback uses a non-reversible hash of the resolved input path plus the display basename inside `local_source_uri(...)`, preventing collisions without exposing parent directories. Pass `self._workspace` from `aingest_files()`.
- `aingest_file()` does not use `_prepare_ingest_item()`. Add optional keyword-only `source_uri` and `download_locator` arguments and pass them into both `_prepare_metadata_record()` calls, including the metadata-only idempotency branch. `_aingest_local_file()` computes the exact workspace-relative `local://` identity from the staged path and supplies both arguments, so nested service ingests never use the hashed fallback.

The public local identity is therefore `local://<workspace>/<workspace-relative-path>`, never an absolute server path. Update test constructors explicitly and delete every production/test reference to `metadata_path`.

- [ ] **Step 4: Validate the locator before materialization**

Resolve each remote document as:

```python
source_uri = validate_source_uri(document.source_uri or source_uri_for_key(document.key))
safe_source_id = document.display_filename or Path(document.key).name or "document"
download_uri = document.download_uri
if download_uri is None and download_uri_for_key is not None:
    download_uri = download_uri_for_key(document.key)

if retain_source_files:
    download_locator = str(
        retained_remote_source_path(
            input_root=self._workspace_input_root(),
            source_type=source_type,
            source_uri=source_uri,
            key=document.key,
        )
    )
else:
    if download_uri is None:
        raise SourceDownloadContractError(
            "retain_source_file=false requires a durable download_uri "
            f"for source {safe_source_id}; "
            "provide download_uri/download_uri_for_key or enable retain_source_file"
        )
    try:
        download_locator = validate_download_uri(download_uri)
    except ValueError:
        raise SourceDownloadContractError(
            f"invalid durable download_uri for source {safe_source_id}; "
            "provide a supported durable URI or enable retain_source_file"
        ) from None
```

Pass `source_uri` and `download_locator` into `_download_remote_to_prepared_item()`. Validation must happen before `source.amaterialize_document()`.

When a bounded download returns an exception, aggregate it under `document.display_filename or Path(document.key).name` instead of prefixing `source_uri`. The dedicated missing/invalid-locator exception must carry the complete safe message asserted above, so a signed URL or custom provenance URI never appears in client-visible batch errors.

Wire S3 and Azure with identical identity/download callbacks. Wire URL with separate `source.source_uri_for_key` and `source.download_uri_for_key`. A custom connector receives no implicit alias.

Mirror `download_uri_for_key` on the public `RAGServiceManager.aingest_source()` SDK method and pass it unchanged to the workspace service. For remote replace, call `_purge_existing_for_replace(stored_file_path=prepared_item.download_locator)`, never `source_uri`; add a custom `bynder://` + HTTPS locator regression proving the old document is found and removed before replacement.

Emit structured, low-cardinality locator outcomes (`accepted`, `missing`, `unsupported`, `ephemeral`) with only locator kind and a safe display filename. URLDataSource reports `ephemeral` when it refuses to derive a locator from a query/fragment fetch URL; service validation reports the remaining outcomes. Never put the URI, query string, credentials, or signed token in log fields/messages.

- [ ] **Step 5: Update engine and service tests for explicit fields**

Replace assertions such as:

```python
assert item.metadata_path == "s3://bucket/report.pdf"
```

with:

```python
assert item.source_uri == "s3://bucket/report.pdf"
assert item.download_locator == "s3://bucket/report.pdf"
```

For retained sources assert the remote `source_uri` and local contained `download_locator` differ. Add a test proving `retain_source_file=true` allows a signed fetch URL and uses the retained local path.

- [ ] **Step 6: Run ingestion tests**

Run: `uv run pytest tests/unit/test_service.py tests/unit/test_unified_ingestion_engine.py tests/unit/test_ingest_jobs.py tests/unit/test_servicemanager.py -q`

Expected: all pass; no test or source file contains `metadata_path`.

- [ ] **Step 7: Commit the invariant enforcement**

```bash
git add src/dlightrag/core/ingestion/engine.py src/dlightrag/core/service.py src/dlightrag/core/servicemanager.py tests/unit/test_service.py tests/unit/test_unified_ingestion_engine.py tests/unit/test_ingest_jobs.py tests/unit/test_servicemanager.py
git commit -m "Enforce durable downloads before ingestion"
```

---

### Task 4: Metadata Separation and Idempotent Backfill

**Files:**
- Modify: `src/dlightrag/core/retrieval/metadata_fields.py:147-168,179-242`
- Modify: `src/dlightrag/core/ingestion/engine.py:300-336`
- Modify: `src/dlightrag/storage/pg_metadata_index.py:54-90`
- Modify: `src/dlightrag/core/service.py:1680-1748`
- Modify: `src/dlightrag/api/routes/metadata.py:17-31`
- Test: `tests/unit/test_metadata_fields.py`
- Test: `tests/unit/test_metadata_index.py`
- Test: `tests/unit/test_unified_ingestion_engine.py`
- Test: `tests/unit/test_service.py`

**Interfaces:**
- Consumes: Task 3 `PreparedIngestFile.source_uri` and `.download_locator`.
- Produces: stored `source_uri`, stored `download_locator`, and retrieval metadata keys `source_uri`/`source_download_locator`.

- [ ] **Step 1: Write failing metadata and migration tests**

```python
def test_metadata_registry_has_source_identity_and_download_locator() -> None:
    ids = {field.field_id for field in METADATA_FIELDS}
    assert {"source_uri", "download_locator"} <= ids


def test_metadata_migrations_backfill_new_source_columns() -> None:
    migrations = list(pg_metadata_index._SCHEMA_MIGRATIONS)
    versions = [migration.version for migration in migrations]
    backfill_index = versions.index("backfill_source_download_contract")
    assert versions.index("column_source_uri") < backfill_index
    assert versions.index("column_download_locator") < backfill_index
    statement = "\n".join(migrations[backfill_index].statements)
    assert "source_uri = COALESCE(source_uri, CASE" in statement
    assert "download_locator = COALESCE(download_locator, file_path)" in statement
    assert "WHERE source_uri IS NULL OR download_locator IS NULL" in statement


def test_extract_system_metadata_stores_distinct_source_and_download_fields() -> None:
    metadata = extract_system_metadata(
        "https://cdn.example.com/assets/1.pdf",
        ingest_strategy="mineru",
        display_filename="report.pdf",
        source_uri="bynder://asset/1",
        download_locator="https://cdn.example.com/assets/1.pdf",
    )
    assert metadata["source_uri"] == "bynder://asset/1"
    assert metadata["download_locator"] == (
        "https://cdn.example.com/assets/1.pdf"
    )
```

- [ ] **Step 2: Run metadata tests and verify fields/migration are missing**

Run: `uv run pytest tests/unit/test_metadata_fields.py tests/unit/test_metadata_index.py tests/unit/test_unified_ingestion_engine.py -q`

Expected: failures show missing registry fields and system metadata keys.

- [ ] **Step 3: Store the dedicated fields**

Add:

```python
MetadataFieldDef("source_uri", "TEXT"),
MetadataFieldDef("download_locator", "TEXT"),
```

Change `extract_system_metadata()` to accept keyword-only `source_uri` and `download_locator`. Pass `PreparedIngestFile.download_locator` as the positional path, so PostgreSQL's legacy `file_path` column remains equal to the durable locator for metadata lookup/deletion compatibility. This is separate from LightRAG `doc_status.file_path`, which continues to receive `parser_path` from `apipeline_enqueue_documents`; do not claim the PostgreSQL assignment changes retry behavior. Store the explicit provenance and locator in their dedicated columns; do not derive either from `parser_path`:

```python
return {
    "filename": file_name.name,
    "filename_stem": file_name.stem,
    "file_path": raw_path,
    "source_uri": source_uri,
    "download_locator": download_locator,
    # existing fields
}
```

Pass the explicit prepared-item fields through `_prepare_metadata_record()`.

- [ ] **Step 4: Add one idempotent data migration after column creation**

```python
Migration(
    "backfill_source_download_contract",
    "Backfill source identity and download locator from existing file_path",
    (
        "UPDATE dlightrag_doc_metadata "
        "SET source_uri = COALESCE(source_uri, CASE "
        "WHEN file_path LIKE 's3://%' OR file_path LIKE 'azure://%' "
        "OR file_path LIKE 'https://%' THEN file_path "
        "ELSE 'local://legacy/' || md5(workspace || ':' || doc_id) || '/' || "
        "regexp_replace(COALESCE(filename, 'source'), '[^A-Za-z0-9._-]', '_', 'g') END), "
        "download_locator = COALESCE(download_locator, file_path) "
        "WHERE source_uri IS NULL OR download_locator IS NULL",
    ),
)
```

Append it only after the generated `column_source_uri` and `column_download_locator` migrations. The `COALESCE` assignments plus null-only `WHERE` make reruns idempotent. Legacy local rows receive a non-path-leaking synthetic provenance identity while retaining their existing local locator. Do not add a runtime fallback in retrieval.

- [ ] **Step 5: Enrich chunks from the new fields only**

```python
source_uri = meta.get("source_uri")
download_locator = meta.get("download_locator")
if isinstance(source_uri, str) and source_uri:
    fetched_meta["source_uri"] = source_uri
if isinstance(download_locator, str) and download_locator:
    fetched_meta["source_download_locator"] = download_locator
```

Delete creation and consumption of `source_file_path`. Keep `source_file_name` for presentation.

Add both `source_uri` and `download_locator` to `_enrich_chunks_with_metadata()`'s `_SKIP` set before injecting only the two dedicated aliases above. Otherwise `download_locator` would be copied wholesale into general chunk metadata. In `aget_metadata()`, project out `workspace`, `doc_id`, `file_path`, and `download_locator`; in `PGMetadataIndex._INTERNAL_COLS`, exclude `file_path`, `source_uri`, and `download_locator` from LLM field-schema hints. Add REST/service tests proving `/metadata/{doc_id}` never returns the raw locator.

Repair failed-document retry at the same boundary instead of continuing to interpret LightRAG's parser path as provenance. Upsert each pending document's system metadata immediately before `apipeline_enqueue_documents()` (the successful finalization upsert remains and enriches that row). Then `aretry_failed_docs()` reads `source_uri` and `download_locator` from the metadata index before deleting the failed LightRAG row. A private `_aingest_download_locator(source_uri, download_locator)` dispatches the validated locator to local, URL, S3, or Azure materialization while preserving the stored provenance on the new `SourceDocument`; it never parses `doc_status.file_path` as a source URI. Add a regression where `doc_status.file_path` is an already-deleted `__remote_ingest__` parser path but retry succeeds from the metadata locator.

- [ ] **Step 6: Run metadata, service, deletion, and retry tests**

Run: `uv run pytest tests/unit/test_metadata_fields.py tests/unit/test_metadata_index.py tests/unit/test_unified_ingestion_engine.py tests/unit/test_service.py tests/unit/test_cleanup.py tests/unit/test_file_routes.py -q`

Expected: all pass; `rg -n 'source_file_path|metadata_path' src tests` returns no product references.

- [ ] **Step 7: Commit the metadata migration**

```bash
git add src/dlightrag/core/retrieval/metadata_fields.py src/dlightrag/core/ingestion/engine.py src/dlightrag/storage/pg_metadata_index.py src/dlightrag/core/service.py src/dlightrag/api/routes/metadata.py tests/unit/test_metadata_fields.py tests/unit/test_metadata_index.py tests/unit/test_unified_ingestion_engine.py tests/unit/test_service.py tests/unit/test_cleanup.py tests/unit/test_file_routes.py tests/unit/test_api_server.py
git commit -m "Store source identity and download locator separately"
```

---

### Task 5: Clean SourceReference and HTTP Projection

**Files:**
- Modify: `src/dlightrag/citations/schemas.py:23-32`
- Modify: `src/dlightrag/citations/finalization.py:32-65`
- Modify: `src/dlightrag/citations/source_builder.py:45-166`
- Modify: `src/dlightrag/core/answer.py:145-180`
- Modify: `src/dlightrag/core/client_payloads.py:102-144`
- Modify: `src/dlightrag/core/retrieval/source_url_resolver.py:14-120`
- Modify: `src/dlightrag/api/routes/rag.py:145-225`
- Modify: `src/dlightrag/api/routes/files.py:98-181`
- Modify: `src/dlightrag/api/models.py:104-121`
- Modify: `src/dlightrag/api/events.py:23-25`
- Modify: `src/dlightrag/mcp/server.py:190-340`
- Test: `tests/unit/citations/test_schemas.py`
- Test: `tests/unit/test_source_builder.py`
- Test: `tests/unit/test_client_payloads.py`
- Test: `tests/unit/test_source_url_resolver.py`
- Test: `tests/unit/test_api_server.py`
- Test: `tests/unit/test_mcp_workspace_tools.py`
- Test: `tests/unit/test_servicemanager.py`

**Interfaces:**
- Consumes: Task 4 retrieval metadata keys and `SourceUrlResolver.resolve(locator, workspace=...)`.
- Produces: internal `SourceReference(source_uri, workspace, download_locator)`, public `SourceReferencePayload(source_uri, download_url)`, and `project_source_payloads()` for every adapter.

- [ ] **Step 1: Write failing clean-schema and projection tests**

```python
def test_source_reference_and_public_payload_have_distinct_contracts() -> None:
    source = SourceReference(
        id="1",
        title="report.pdf",
        source_uri="bynder://asset/1",
        workspace="finance",
        download_locator="https://cdn.example.com/assets/1.pdf",
    )
    assert {"workspace", "download_locator"} <= set(SourceReference.model_fields)
    assert {"path", "url", "download_url"}.isdisjoint(SourceReference.model_fields)

    public = project_source_payloads([source], resolver=None)[0]
    payload = public.model_dump()
    assert payload["source_uri"] == "bynder://asset/1"
    assert payload["download_url"] is None
    assert "workspace" not in payload
    assert "download_locator" not in payload
    assert "path" not in payload
    assert "url" not in payload


def test_source_builder_uses_dedicated_metadata_contract() -> None:
    chunk = _chunk("c1", "ref-1", file_path="parser-name.pdf")
    chunk["metadata"] = {
        "source_uri": "bynder://asset/1",
        "source_download_locator": "https://cdn.example.com/assets/1.pdf",
        "source_file_name": "report.pdf",
    }
    chunk["_workspace"] = "finance"
    source = build_sources({"chunks": [chunk]})[0]
    assert source.source_uri == "bynder://asset/1"
    assert source.workspace == "finance"
    assert source.download_locator == "https://cdn.example.com/assets/1.pdf"


def test_project_source_payloads_resolves_and_hides_raw_locator() -> None:
    source = SourceReference(
        id="1",
        source_uri="s3://bucket/report.pdf",
        workspace="finance",
        download_locator="s3://bucket/report.pdf",
    )
    projected = project_source_payloads([source], resolver=SourceUrlResolver())[0]
    assert projected.download_url == "/files/raw/s3://bucket/report.pdf?workspace=finance"
    payload = projected.model_dump()
    assert "download_locator" not in payload
    assert "workspace" not in payload


def test_public_context_projection_strips_internal_source_metadata() -> None:
    contexts = {
        "chunks": [
            {
                "chunk_id": "c1",
                "reference_id": "r1",
                "file_path": "/srv/dlightrag/inputs/finance/report.pdf",
                "content": "text",
                "metadata": {
                    "source_uri": "bynder://asset/1",
                    "source_download_locator": "https://cdn.example.com/assets/1.pdf",
                    "category": "research",
                },
            }
        ],
        "entities": [],
        "relationships": [],
    }
    projected = project_contexts_for_client(contexts)
    assert projected["chunks"][0]["metadata"] == {"category": "research"}
    assert projected["chunks"][0]["file_path"] == "report.pdf"


async def test_remote_source_download_enforces_its_workspace(
    tmp_working_dir: Path,
) -> None:
    class DenyFinanceWorkspace:
        async def check(self, user, action, *, workspace=None):
            if workspace == "finance":
                raise AccessDeniedError("denied")

        async def filter_workspaces(self, user, action, workspaces):
            return [workspace for workspace in workspaces if workspace != "finance"]

    config = DlightragConfig(
        working_dir=str(tmp_working_dir),
        llm=LLMConfig(default=ModelConfig(model="gpt-5.4-mini", api_key="test")),
        embedding=_embedding_config(),
    )
    set_config(config)
    with patch("dlightrag.api.server.RAGServiceManager.acreate", new_callable=AsyncMock):
        app = create_app(include_web=False)
        app.state.access_control = DenyFinanceWorkspace()
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            follow_redirects=False,
        ) as client:
            response = await client.get(
                "/files/raw/s3://bucket/report.pdf",
                params={"workspace": "finance"},
            )
    assert response.status_code == 403
    assert response.json()["detail"] == "denied"
```

- [ ] **Step 2: Run source tests and verify legacy fields are still required**

Run: `uv run pytest tests/unit/citations/test_schemas.py tests/unit/test_source_builder.py tests/unit/test_client_payloads.py -q`

Expected: failures mention missing `source_uri`/`download_locator` and existing `path`/`url` payload keys.

- [ ] **Step 3: Replace the SourceReference fields**

```python
from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    id: str
    title: str | None = None
    type: str | None = None
    source_uri: str
    workspace: str = Field(exclude=True, repr=False)
    download_locator: str = Field(exclude=True, repr=False)
    cited_chunk_ids: list[str] | None = None
    chunks: list[ChunkSnippet] | None = None

    model_config = {"extra": "forbid"}


class SourceReferencePayload(BaseModel):
    id: str
    title: str | None = None
    type: str | None = None
    source_uri: str
    download_url: str | None = None
    cited_chunk_ids: list[str] | None = None
    chunks: list[ChunkSnippet] | None = None

    model_config = {"extra": "forbid"}
```

Do not add `path` or `url` properties, aliases, validators, or deprecation warnings. `SourceReference` is never a FastAPI response field; API response/event models use `SourceReferencePayload`, so required internal fields are not lost and then revalidated.

- [ ] **Step 4: Build and project sources through dedicated functions**

In `source_builder.py`, require non-empty metadata `source_uri` and `source_download_locator`. Set the internal workspace from the source catalog when present, otherwise from the first chunk's `_workspace`, otherwise from `default_workspace`; fail the source invariant if none exists. Construct the new fields and preserve current citation/chunk ordering. Remove `source_url_resolver` from `build_sources()`/`build_sources_from_chunks()` and remove both `source_contexts` and `source_url_resolver` from `finalize_answer()`; source finalization always consumes raw contexts, while adapters project a separate public context copy. Delete catalog `url` handling: the domain/finalization layers never resolve HTTP URLs. In `core/answer.py`, replace `s.title or s.path` with `s.title or "Source"` when producing lightweight references.

Change `SourceUrlResolver.resolve()` to accept an optional keyword-only per-call workspace. It must append the effective workspace as an encoded `workspace` query parameter for local and remote locators; a per-call value overrides the constructor default. This makes authorization for federated S3/Azure/HTTPS sources independent of current Search-in, Files-in, or configured-default state.

Add to `core/client_payloads.py`:

```python
class SourceDownloadInvariantError(RuntimeError):
    pass


def project_source_payloads(
    sources: list[SourceReference],
    *,
    resolver: SourceUrlResolver | None,
) -> list[SourceReferencePayload]:
    projected: list[SourceReferencePayload] = []
    for source in sources:
        download_url = None
        if resolver is not None:
            download_url = resolver.resolve(
                source.download_locator,
                workspace=source.workspace,
            )
        if resolver is not None and not download_url:
            raise SourceDownloadInvariantError(
                f"Could not project download locator for source {source.id}"
            )
        projected.append(
            SourceReferencePayload(
                id=source.id,
                title=source.title,
                type=source.type,
                source_uri=source.source_uri,
                download_url=download_url,
                cited_chunk_ids=source.cited_chunk_ids,
                chunks=source.chunks,
            )
        )
    return projected
```

Use safe source IDs in errors; never log the raw locator. Before public chunk projection, remove `source_uri` and `source_download_locator` from chunk `metadata`; the source-level payload is their sole public representation. Preserve the existing public chunk `file_path` key only as a display basename (`Path(...).name` for local paths and the decoded URL/provider path basename for URIs), never as an absolute path or locator.

Emit `source_download_projection_outcome=resolved|invalid` with the source ID and no locator. Keep `/files/raw` and `workspace.download_source` as the authorization enforcement point; on a 403 it records `source_download_projection_outcome=unauthorized` with the source's actual workspace only, never the locator.

- [ ] **Step 5: Make REST projections consistent**

Build retrieval/finalized sources from raw contexts before calling `project_contexts_for_client()`, because the public projection intentionally strips internal source metadata. Delete every streaming/API call that passes `source_contexts=public_contexts`:

```python
sources = build_sources(result.contexts)
source_payloads = project_source_payloads(
    sources,
    resolver=source_url_resolver,
)
contexts = project_contexts_for_client(result.contexts, image_url_prefix=image_url_prefix)
```

Change `answer_payload()` to accept `source_url_resolver: SourceUrlResolver | None` and always convert `result.sources` to `SourceReferencePayload` before dumping. Pass a resolver in non-stream REST answer, retrieve, streaming answer, and the Web done payload. SDK/MCP call the same public projector with `resolver=None`; they receive the clean public schema with `download_url=None`, never a dumped internal model.

Change `RetrievalResponse.sources`, `AnswerResponse.sources`, and `AnswerSourcesStreamEvent.data` to `list[SourceReferencePayload]`. Update response/event tests to assert `source_uri` and `download_url`, and assert raw `workspace`, `download_locator`, `path`, and `url` are absent. Use the same source fixture (`workspace="finance"`, `download_locator="s3://bucket/report.pdf"`) in four regressions: `/retrieve`, non-stream `/answer`, the streaming `type="sources"` event, and direct `answer_payload()`. Every HTTP payload must contain `/files/raw/s3://bucket/report.pdf?workspace=finance`; transport-neutral `answer_payload()` without a resolver must keep `download_url=None`. Add one FastAPI response-validation regression to prove these payloads return 200 rather than failing because excluded internal fields are required.

- [ ] **Step 6: Run source/API/MCP tests**

Run: `uv run pytest tests/unit/citations tests/unit/test_source_builder.py tests/unit/test_client_payloads.py tests/unit/test_api_server.py tests/unit/test_mcp_workspace_tools.py tests/unit/test_answer_engine.py -q`

Expected: all pass; source payloads have the clean contract and no raw local locator leaks.

- [ ] **Step 7: Commit the source projection boundary**

```bash
git add src/dlightrag/citations/schemas.py src/dlightrag/citations/finalization.py src/dlightrag/citations/source_builder.py src/dlightrag/core/answer.py src/dlightrag/core/client_payloads.py src/dlightrag/core/retrieval/source_url_resolver.py src/dlightrag/api/routes/rag.py src/dlightrag/api/routes/files.py src/dlightrag/api/models.py src/dlightrag/api/events.py src/dlightrag/mcp/server.py tests/unit/citations tests/unit/test_source_builder.py tests/unit/test_client_payloads.py tests/unit/test_source_url_resolver.py tests/unit/test_api_server.py tests/unit/test_file_routes.py tests/unit/test_mcp_workspace_tools.py tests/unit/test_answer_engine.py tests/unit/test_servicemanager.py
git commit -m "Project source downloads through HTTP adapters"
```

---

### Task 6: Web Source Panel Invariant and Accessibility

**Files:**
- Modify: `src/dlightrag/web/answer_events.py:84-125`
- Modify: `src/dlightrag/web/templates/partials/answer_done.html:15-25`
- Modify: `src/dlightrag/web/templates/partials/source_panel.html:1-20`
- Modify: `src/dlightrag/web/safe_html.py:45-100,155-180`
- Test: `tests/unit/test_web_frontend_static.py`
- Test: `tests/unit/test_web_routes.py`
- Test: `tests/e2e/test_source_panel.py`

**Interfaces:**
- Consumes: Task 5 `project_source_payloads()` and `SourceReferencePayload.download_url`.
- Produces: Web source rows with exactly one unconditional, sanitized, keyboard-reachable Download link.

- [ ] **Step 1: Write failing render and sanitizer tests**

```python
def test_source_panel_requires_download_for_every_source() -> None:
    from dlightrag.citations.schemas import SourceReferencePayload
    from dlightrag.web.deps import templates

    source = SourceReferencePayload(
        id="1",
        title="notes.md",
        source_uri="local://default/notes.md",
        download_url="/files/raw/default/notes.md?workspace=default",
        chunks=[],
    )
    html = templates.env.get_template("partials/source_panel.html").render(
        sources=[source]
    )
    assert html.count('class="source-dl-icon"') == 1
    assert 'href="/files/raw/default/notes.md?workspace=default"' in html

    source_panel_text = (
        Path(__file__).resolve().parents[2]
        / "src/dlightrag/web/templates/partials/source_panel.html"
    ).read_text(encoding="utf-8")
    assert "{% if src." not in source_panel_text


def test_sanitized_source_download_preserves_accessible_name() -> None:
    from dlightrag.citations.schemas import SourceReferencePayload
    from dlightrag.web.safe_html import safe_source_panel

    source = SourceReferencePayload(
        id="1",
        title="notes.md",
        source_uri="local://default/notes.md",
        download_url="/files/raw/default/notes.md?workspace=default",
        chunks=[],
    )
    html = safe_source_panel(sources=[source])
    assert 'aria-label="Download source"' in html
    assert 'download=""' in html or " download" in html
```

- [ ] **Step 2: Run Web tests and verify the conditional/ARIA stripping failures**

Run: `uv run pytest tests/unit/test_web_frontend_static.py tests/unit/test_web_routes.py -q -k 'source or download'`

Expected: tests fail because the template still checks `src.url` and sanitizer removes `aria-label`.

- [ ] **Step 3: Project before rendering and remove the optional UI state**

In `_build_answer_done_payload()`, call transport-neutral `finalize_answer()` first. Use internal sources for `answer_images_from_sources()`, then call `project_source_payloads(finalized.sources, resolver=resolver)` and use that public list for answer HTML, the highlights event, and `_AnswerPayload.sources`. Let `SourceDownloadInvariantError` propagate to the existing structured answer error path. Add a Web done-payload test whose source chunk has `_workspace="finance"` and assert both rendered download anchors contain `workspace=finance`.

Replace the template branch with:

```html
<a href="{{ src.download_url }}"
   class="source-dl-icon"
   title="Download source"
   aria-label="Download source"
   download>
  <svg class="source-dl-icon-svg" viewBox="0 0 24 24" fill="none"
       stroke="currentColor" stroke-width="1.5" stroke-linecap="round"
       stroke-linejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
    <polyline points="7 10 12 15 17 10"/>
    <line x1="12" y1="15" x2="12" y2="3"/>
  </svg>
</a>
```

Keep the anchor as a sibling of `.source-doc-toggle`. Replace title fallbacks in both `source_panel.html` and `answer_done.html` from `src.title or src.path` to `src.title or "Source"`; no template may consume a legacy path. Do not add disabled, tooltip-only, hover-only, extension-specific, or missing-link states.

- [ ] **Step 4: Preserve accessibility attributes through sanitization**

Add `aria-label` to the sanitizer's global or anchor-specific allowed attributes without allowing arbitrary event handlers, styles, targets, or unsafe schemes. Keep existing URL sanitization for `href`.

- [ ] **Step 5: Add Markdown route coverage**

```python
async def test_local_markdown_download_is_attachment(
    client: AsyncClient,
    tmp_working_dir: Path,
) -> None:
    source = tmp_working_dir / "inputs" / "default" / "notes.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("# Notes", encoding="utf-8")
    response = await client.get("/files/raw/default/notes.md")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert 'attachment; filename="notes.md"' in response.headers["content-disposition"]
```

- [ ] **Step 6: Run Web and E2E source tests**

Run: `uv run pytest tests/unit/test_web_frontend_static.py tests/unit/test_web_routes.py tests/unit/test_file_routes.py tests/e2e/test_source_panel.py -q`

Expected: all pass; citation keyboard navigation, download sibling structure, and Markdown download work together.

- [ ] **Step 7: Commit the Web invariant**

```bash
git add src/dlightrag/web/answer_events.py src/dlightrag/web/templates/partials/answer_done.html src/dlightrag/web/templates/partials/source_panel.html src/dlightrag/web/safe_html.py tests/unit/test_web_frontend_static.py tests/unit/test_web_routes.py tests/unit/test_file_routes.py tests/e2e/test_source_panel.py
git commit -m "Require downloads in the Web source panel"
```

---

### Task 7: Documentation, Stale-Path Removal, and Full Verification

**Files:**
- Modify: `README.md:280-290`
- Modify: `docs/configuration.md:289-309`
- Modify: `docs/interfaces.md:172-199,500-585`
- Modify: `docs/security.md:245-270`
- Modify: `scripts/cli.py`
- Test: `tests/unit/test_config.py`
- Test: `tests/unit/test_client_requests.py`
- Test: `tests/unit/test_web_frontend_static.py`

**Interfaces:**
- Consumes: all prior task contracts.
- Produces: one documented, searchable, legacy-free source download model and repository-wide validation evidence.

- [ ] **Step 1: Add static stale-contract tests**

```python
def test_source_download_contract_has_no_legacy_public_names() -> None:
    partials = ROOT / "src/dlightrag/web/templates/partials"
    template_text = "\n".join(
        (partials / name).read_text(encoding="utf-8")
        for name in ("source_panel.html", "answer_done.html")
    )
    internal_fields = set(SourceReference.model_fields)
    public_fields = set(SourceReferencePayload.model_fields)
    assert "metadata_path" not in (ROOT / "src/dlightrag/core/ingestion/engine.py").read_text()
    assert {"path", "url", "download_url"}.isdisjoint(internal_fields)
    assert {"workspace", "download_locator", "source_uri"} <= internal_fields
    assert {"path", "url", "workspace", "download_locator"}.isdisjoint(public_fields)
    assert {"source_uri", "download_url"} <= public_fields
    assert "src.url" not in template_text
    assert "src.path" not in template_text
    assert "src.download_url" in template_text
```

- [ ] **Step 2: Run stale-contract tests and repository searches**

Run:

```bash
uv run pytest tests/unit/test_client_requests.py tests/unit/test_web_frontend_static.py -q
rg -n "metadata_path|source_file_path|src\.(path|url)|s\.path|catalog\.(path|url)|source\.(path|url)" src tests README.md docs scripts
rg -n -U "SourceReference\((?s:.{0,500}?)(path|url)\s*=" src tests
```

Expected: pytest passes after earlier tasks; both `rg` commands return only historical design discussion, never active product code, tests, examples, or public docs.

- [ ] **Step 3: Rewrite public documentation with exact failure semantics**

Document:

```text
source_uri identifies the source; download_uri tells DlightRAG how to retrieve
the original when no local copy is retained. Signed/query-bearing HTTPS fetch
URLs are ephemeral and require either retain_source_file=true or a separate
queryless public download_uri. A non-retained custom AsyncDataSource must set
SourceDocument.download_uri or download_uri_for_key. Invalid documents are
rejected before parser materialization; DlightRAG never silently retains them.
```

Update REST/MCP/CLI examples for single URL, URL batch, signed URL with retention, and custom SDK connector. Remove examples that use `source_uri` as a download address.

- [ ] **Step 4: Run formatting, typing, architecture, and unit gates**

Run:

```bash
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
uv run pyright
uv run lint-imports
uv run pytest tests/unit -q
```

Expected: all commands exit 0.

- [ ] **Step 5: Run frontend and targeted E2E gates**

Run:

```bash
npm --prefix frontend run typecheck
npm --prefix frontend run build
npm --prefix frontend run lint:css
uv run pytest tests/e2e/test_source_panel.py -q
```

Expected: all commands exit 0; Vite production build succeeds and the Source panel E2E passes.

- [ ] **Step 6: Inspect the final diff for scope and secrets**

Run:

```bash
git diff --check
git status --short
git diff --stat HEAD~6..HEAD
git log -7 --oneline
```

Expected: no whitespace errors, no `.env`/local config/Visual Companion files, and only source download invariant files are changed.

- [ ] **Step 7: Commit documentation and final cleanup**

```bash
git add README.md docs/configuration.md docs/interfaces.md docs/security.md scripts/cli.py tests/unit/test_config.py tests/unit/test_client_requests.py tests/unit/test_web_frontend_static.py
git commit -m "Document durable source download behavior"
```

---

## Final Review Checklist

- [ ] Signed/query-bearing HTTPS with `retain_source_file=false` and no explicit durable locator is rejected before bytes are materialized.
- [ ] Custom `bynder://` provenance with no durable locator is rejected before bytes are materialized.
- [ ] Both cases succeed with local retention.
- [ ] Both cases succeed with a separately validated supported download URI where applicable.
- [ ] Existing local/S3/Azure/queryless-HTTPS ingestion still succeeds.
- [ ] PostgreSQL metadata stores and migrates separate source/download fields.
- [ ] Local provenance uses stable `local://` identities; no absolute staged path appears as public `source_uri`.
- [ ] Failed-document retry consumes `source_uri`/`download_locator`, never LightRAG parser `file_path`, as source/download policy.
- [ ] No public source payload leaks `download_locator` or internal local paths.
- [ ] Public chunk `file_path` is display-only basename and `/metadata`/field-schema projections expose no internal locator.
- [ ] REST retrieve, REST streaming answer, REST non-stream answer, and Web project the same source contract; SDK/MCP remain transport-neutral.
- [ ] Federated download URLs encode the source's actual workspace and `/files/raw` enforces `workspace.download_source` against it.
- [ ] Structured `source_download_locator_outcome` and `source_download_projection_outcome` logs contain no URI, query string, credentials, or signed tokens.
- [ ] Every authorized Web source row, including `.md`, has one accessible Download link.
- [ ] No deprecated aliases, silent fallbacks, or no-download UI states remain.
- [ ] Full Python, TypeScript, CSS, architecture, unit, and targeted E2E gates pass.
