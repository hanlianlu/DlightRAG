# Eliminate sources/ Directory — Design Document

**Date:** 2026-03-05
**Status:** Approved

## Goal

Remove the `sources/` copy step from the ingestion pipeline. Record original source URIs in metadata instead of local copy paths. Eliminate redundant file storage.

## Problem

The current ingestion pipeline copies every file to `{working_dir}/sources/{type}/` before parsing:

```
Original file ──copy──> sources/local/file.pdf ──parse──> artifacts/local/file/
Azure blob    ──dl──>   sources/azure_blobs/c/file.pdf ──parse──> artifacts/azure_blobs/c/file/
```

Three issues:
1. **Redundant storage** — local files are duplicated (original + sources/ copy)
2. **Wrong metadata** — `hash_index` and LightRAG `doc_status` record `sources/local/file.pdf` instead of the true original path
3. **Fragile path parsing** — `_extract_relative_source_path()` relies on string-matching `/sources/` marker

## Design

### New Ingestion Flow

```
[Local file]
  Original path → hash check → [Excel? convert in .tmp/] → parse from original/.tmp path
  → artifacts/ → LightRAG insert with source_uri = original path

[Azure blob]
  Download to .tmp/ → hash check → [Excel? convert in .tmp/] → parse from .tmp path
  → artifacts/ → LightRAG insert with source_uri = azure://container/blob

[Snowflake]
  Unchanged — already uses snowflake://{label} with no file copy
```

### Two Distinct Path Concepts

| Concept | Purpose | Example |
|---------|---------|---------|
| `parse_path` | Actual local file path for the parser to read | `/Users/me/report.pdf` or `{working_dir}/.tmp/abc123/report.pdf` |
| `source_uri` | Original source identifier stored in metadata | `/Users/me/report.pdf`, `azure://container/data/report.pdf` |

These are passed as separate parameters and must not be confused:

```python
rag.parse_document(file_path=str(parse_path), ...)       # parser reads from here
rag.insert_content_list(file_path=source_uri, ...)        # recorded in doc_status
hash_index.register(content_hash, doc_id, source_uri)     # recorded in hash_index
```

### Temp Directory Management

Temp files live in `{working_dir}/.tmp/` (not system `/tmp`), sharing the same storage volume (important for K8s PVC deployments).

```
working_dir/
├── artifacts/     # Parse output (persistent)
├── .tmp/          # Temporary downloads/conversions (auto-cleaned)
└── ...            # Graph, vector, KV storage
```

Config addition:

```python
@property
def temp_dir(self) -> Path:
    return self.working_dir_path / ".tmp"
```

**Unified temp pattern** — every backend wraps single-file ingestion the same way:

```python
async def _ingest_local_single(self, path: Path, ...) -> IngestionResult:
    tmpdir = self._create_temp_dir()
    try:
        working_path = await self._prepare_for_parsing(path, tmpdir)
        return await self._ingest_single_file_with_policy(
            parse_path=working_path,
            source_uri=str(path.resolve()),
            ...
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

async def _ingest_azure_single(self, source, container, blob_path, ...) -> IngestionResult:
    tmpdir = self._create_temp_dir()
    try:
        downloaded = await self._download_blob(source, blob_path, tmpdir)
        working_path = await self._prepare_for_parsing(downloaded, tmpdir)
        return await self._ingest_single_file_with_policy(
            parse_path=working_path,
            source_uri=f"azure://{container}/{blob_path}",
            ...
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
```

- Non-Excel files: `_prepare_for_parsing` returns the input path unchanged; tmpdir is empty
- Excel files: converted PDF written to tmpdir
- Azure downloads: blob written to tmpdir
- After ingestion: `shutil.rmtree(tmpdir)` cleans everything

### Modules to Change

#### 1. `src/dlightrag/ingestion/pipeline.py` (main changes)

**Remove:**
- `_acopy_to_sources_local()` — no more copying local files
- `_download_blob_to_storage_async()` — replaced with temp-based download
- `_extract_relative_source_path()` — no more sources/ path parsing
- `_resolve_source_file()` — no more sources/ file resolution

**Add:**
- `_create_temp_dir() -> Path` — creates unique subdir under `config.temp_dir`
- `_prepare_for_parsing(file_path, tmpdir) -> Path` — Excel conversion or pass-through
- `_download_blob_to_temp(source, blob_path, tmpdir) -> Path` — download blob to temp

**Modify:**
- `_ingest_single_file_with_policy()` — accept both `parse_path` and `source_uri`
  - `parse_document(file_path=str(parse_path))`
  - `insert_content_list(file_path=source_uri)`
  - `hash_index.register(..., source_uri)`
- `aingest_from_local()` — pass original path directly, use temp only for Excel
- `aingest_from_azure_blob()` — use temp downloads, build `azure://` URI
- `adelete_files()` — remove source file deletion logic; only clean artifacts + LightRAG + hash_index

#### 2. `src/dlightrag/config.py`

- Remove `sources_dir` property
- Add `temp_dir` property → `working_dir_path / ".tmp"`

#### 3. `src/dlightrag/ingestion/cleanup.py`

- Update `collect_deletion_context()` to not search sources/ directory

#### 4. `src/dlightrag/ingestion/hash_index.py`

- No code changes needed — `file_path` field naturally receives source_uri

### Deletion Logic Changes

**Before:**
```
adelete_files → find doc_ids → lightrag.adelete_by_doc_id → hash_index.remove
             → delete source file from sources/
             → delete artifacts/
```

**After:**
```
adelete_files → find doc_ids → lightrag.adelete_by_doc_id → hash_index.remove
             → delete artifacts/
             (no source file deletion — originals are not ours to manage)
```

We don't own the original files. Local files belong to the user; Azure blobs belong to the cloud. We only clean up what we created: artifacts, LightRAG data, hash_index entries.

### Backward Compatibility

- Existing `sources/` directories are NOT auto-deleted (users can clean up manually)
- Existing hash_index entries with `sources/` paths still work for dedup (hash-based matching; path is metadata only)
- New ingestions record true source URIs
- `adelete_files()` still attempts artifact cleanup based on stored metadata

### K8s Deployment

Only one PVC needed, mounted at `working_dir`:
```
PVC mount → /app/dlightrag_storage/
├── artifacts/     (persistent parse output)
├── .tmp/          (ephemeral, auto-cleaned per ingestion)
├── graph/vector/kv storage
```

No separate volume for temp. `.tmp/` shares the PVC with artifacts.
