# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""asyncpg connection arguments derived from the typed PostgreSQL settings.

Configuration holds the validated endpoint, credentials, and TLS file paths; what
asyncpg connects with, a TLS context included, is built here. Every connection
DlightRAG opens itself comes from this one function (the domain pool, the
notification hub's listener, the corpus coordination and maintenance adapters, and
the workspace write gate), and the LightRAG environment bridge reads its endpoint
from it too, so a managed deployment configures TLS once.
"""

import ssl
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dlightrag.application.config import PostgresSettings


def pg_connection_kwargs(settings: PostgresSettings) -> dict[str, Any]:
    """The endpoint, credentials, and TLS argument asyncpg connects with.

    ``ssl`` is left out when the mode keeps asyncpg's own default (unset or
    ``allow``). A TLS file that cannot be loaded fails here, naming the error.
    """
    kwargs: dict[str, Any] = {
        "host": settings.host,
        "port": settings.port,
        "user": settings.user,
        "password": settings.password,
        "database": settings.database,
    }
    if (ssl_value := _ssl_argument(settings)) is not None:
        kwargs["ssl"] = ssl_value
    return kwargs


def _ssl_argument(settings: PostgresSettings) -> ssl.SSLContext | bool | None:
    mode = settings.ssl_mode
    if mode is None:
        return None
    if mode in {"require", "prefer"}:
        return True
    if mode == "disable":
        return False
    if mode == "allow":
        return None
    try:
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        context.check_hostname = mode == "verify-full"
        if settings.ssl_root_cert and Path(settings.ssl_root_cert).exists():
            context.load_verify_locations(cafile=settings.ssl_root_cert)
        if (
            settings.ssl_cert
            and settings.ssl_key
            and Path(settings.ssl_cert).exists()
            and Path(settings.ssl_key).exists()
        ):
            context.load_cert_chain(settings.ssl_cert, settings.ssl_key)
        if settings.ssl_crl and Path(settings.ssl_crl).exists():
            context.verify_flags |= ssl.VERIFY_CRL_CHECK_LEAF
            context.load_verify_locations(cafile=settings.ssl_crl)
        return context
    except Exception as exc:
        raise ValueError(f"PostgreSQL SSL configuration error: {exc}") from exc


__all__ = ["pg_connection_kwargs"]
