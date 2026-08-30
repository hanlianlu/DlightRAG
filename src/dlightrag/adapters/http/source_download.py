# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared HTTP response projection for authorized source downloads."""

from fastapi import HTTPException
from starlette.responses import FileResponse, RedirectResponse

from dlightrag.application.corpus_admin import (
    CorpusAdmin,
    LocalDownloadTarget,
    RedirectDownloadTarget,
    SourceDownloadInvalidError,
    SourceDownloadNotFoundError,
    SourceDownloadUnavailableError,
)


async def source_download_response(
    corpora: CorpusAdmin,
    *,
    workspace: str,
    document_id: str,
) -> FileResponse | RedirectResponse:
    """Prepare one source and map its transport-neutral outcome to HTTP."""
    try:
        target = await corpora.prepare_source_download(workspace, document_id)
    except SourceDownloadInvalidError as exc:
        raise HTTPException(400, str(exc)) from exc
    except SourceDownloadNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except SourceDownloadUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc

    if isinstance(target, LocalDownloadTarget):
        return FileResponse(
            target.path,
            media_type=target.media_type,
            filename=target.filename,
        )
    if isinstance(target, RedirectDownloadTarget):
        return RedirectResponse(url=target.url, status_code=302)
    raise TypeError("Unsupported source download target")


__all__ = ["source_download_response"]
