# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Name a document-parser outage at LightRAG's parser transport boundary.

LightRAG's MinerU and Docling clients make every parser HTTP call, and its parse
worker records whatever they raise as the document's failure. A refused or reset
connection, a timed-out socket operation, or a retryable HTTP status from the
parser service therefore arrived as an arbitrary client error that nothing could
tell apart from a document the parser rejected.

This patch wraps the active client's ``download_into`` so exactly those failures
raise :class:`ParserUnavailableError`, chained to the original, which is logged.
LightRAG records the document with that error's fixed, secret-free message, and
:func:`parser_unavailable_recorded` reads the verdict back from the document
status. Every other failure stays the error LightRAG raised: a 4xx response, a
conversion the parser reports as failed, an exhausted polling budget or download
deadline (a large document can exhaust them on every attempt), and an oversized
or malformed bundle.

LightRAG reports a non-2xx parser response as a plain ``RuntimeError``; the patch
stamps the response status on that same error so the shared request
classification reads the status instead of parsing message text.

Delete this module once LightRAG types its parser transport failures itself.
"""

import logging
from collections.abc import Callable, Mapping
from functools import wraps
from typing import Any

from dlightrag.engine.dependencies import ParserUnavailableError, is_transient_request_failure

logger = logging.getLogger(__name__)

_PATCH_ATTR = "_dlightrag_reports_parser_outages"
_RECORDED_VERDICT = str(ParserUnavailableError())


def apply_parser_outage_reporting(*, docling_active: bool) -> bool:
    """Patch the active parser client to name its outages. Idempotent.

    Returns True when it installs.
    """
    if docling_active:
        from lightrag.parser.external.docling import client as client_module

        client_class: Any = client_module.DoclingRawClient
        engine = "Docling"
    else:
        from lightrag.parser.external.mineru import client as client_module

        client_class = client_module.MinerURawClient
        engine = "MinerU"

    original = client_class.download_into
    if getattr(original, _PATCH_ATTR, False):
        return False

    @wraps(original)
    async def download_into(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await original(self, *args, **kwargs)
        except Exception as exc:
            if not is_transient_request_failure(exc):
                raise
            logger.warning(
                "%s parser request failed transiently; recording the parser as unavailable",
                engine,
                exc_info=True,
            )
            raise ParserUnavailableError() from exc

    setattr(download_into, _PATCH_ATTR, True)
    client_module.raise_for_status_with_detail = _stamping_status(
        client_module.raise_for_status_with_detail
    )
    client_class.download_into = download_into
    logger.info("Applied LightRAG %s parser outage reporting", engine)
    return True


def parser_unavailable_recorded(doc_status: Any) -> bool:
    """Return whether a document status records the parser-outage verdict."""
    if isinstance(doc_status, Mapping):
        error = doc_status.get("error_msg")
    else:
        error = getattr(doc_status, "error_msg", None)
    return error == _RECORDED_VERDICT


def _stamping_status(original: Callable[..., None]) -> Callable[..., None]:
    @wraps(original)
    def raise_for_status_with_detail(
        resp: Any,
        operation: str,
        *,
        body: bytes | None = None,
    ) -> None:
        try:
            original(resp, operation, body=body)
        except RuntimeError as exc:
            status = getattr(resp, "status_code", None)
            if isinstance(status, int) and not isinstance(status, bool):
                exc.status_code = status  # pyright: ignore[reportAttributeAccessIssue]
            raise

    return raise_for_status_with_detail


__all__ = ["apply_parser_outage_reporting", "parser_unavailable_recorded"]
