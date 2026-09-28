# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Name a document-parser outage at LightRAG's parser transport boundary.

LightRAG's MinerU and Docling clients make every parser HTTP call, and its parse
worker records whatever they raise as the document's failure. A refused or reset
connection, a timed-out socket operation, or a retryable HTTP status from the
parser service therefore arrived as an arbitrary client error that nothing could
tell apart from a document the parser rejected.

This patch wraps both clients' ``download_into`` (a per-file parser directive can
route a document to the engine that is not the configured default) so exactly
those failures raise :class:`ParserUnavailableError`. The verdict reads only the
parser's response status and the client's transport error type, never message
text: LightRAG's error text names the user's file and quotes the parser's
response body, so a word such as "schema" in either must not turn an outage into
a rejection. The client error is logged, and the verdict carries no cause, so
nothing that classifies it later reads that text either.

LightRAG records the document with the verdict's fixed, secret-free message, and
:func:`parser_unavailable_recorded` reads it back from the document status.
Every other failure stays the error LightRAG raised: a 4xx response, a
conversion the parser reports as failed, an exhausted polling budget or download
deadline (a large document can exhaust them on every attempt), an oversized or
malformed bundle, and a misconfigured endpoint.

LightRAG reports a non-2xx parser response as a plain ``RuntimeError``; the patch
stamps the response status on that same error.

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


def apply_parser_outage_reporting() -> bool:
    """Patch both parser clients to name their outages. Idempotent.

    Returns True when it installs on either client.
    """
    from lightrag.parser.external.docling import client as docling_client
    from lightrag.parser.external.mineru import client as mineru_client

    installed = [
        _install(engine, module, client_class)
        for engine, module, client_class in (
            ("MinerU", mineru_client, mineru_client.MinerURawClient),
            ("Docling", docling_client, docling_client.DoclingRawClient),
        )
    ]
    return any(installed)


def parser_unavailable_recorded(doc_status: Any) -> bool:
    """Return whether a document status records the parser-outage verdict."""
    if isinstance(doc_status, Mapping):
        error = doc_status.get("error_msg")
    else:
        error = getattr(doc_status, "error_msg", None)
    return error == _RECORDED_VERDICT


def _install(engine: str, client_module: Any, client_class: Any) -> bool:
    original = client_class.download_into
    if getattr(original, _PATCH_ATTR, False):
        return False

    @wraps(original)
    async def download_into(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await original(self, *args, **kwargs)
        except Exception as exc:
            if not is_transient_request_failure(exc, text_vetoes=False):
                raise
            logger.warning(
                "%s parser request failed transiently; recording the parser as unavailable",
                engine,
                exc_info=True,
            )
        # Raised outside the handler so the verdict carries no cause.
        raise ParserUnavailableError()

    setattr(download_into, _PATCH_ATTR, True)
    client_module.raise_for_status_with_detail = _stamping_status(
        client_module.raise_for_status_with_detail
    )
    client_class.download_into = download_into
    logger.info("Applied LightRAG %s parser outage reporting", engine)
    return True


def _stamping_status(original: Callable[..., None]) -> Callable[..., None]:
    @wraps(original)
    def raise_for_status_with_detail(*args: Any, **kwargs: Any) -> None:
        # Everything passes through untouched, so a change to the upstream
        # signature can only lose the stamp, never break a parse.
        try:
            original(*args, **kwargs)
        except RuntimeError as exc:
            response = args[0] if args else kwargs.get("resp")
            status = getattr(response, "status_code", None)
            if isinstance(status, int) and not isinstance(status, bool):
                exc.status_code = status  # pyright: ignore[reportAttributeAccessIssue]
            raise

    return raise_for_status_with_detail


__all__ = ["apply_parser_outage_reporting", "parser_unavailable_recorded"]
