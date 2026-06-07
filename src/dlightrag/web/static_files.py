"""Static file serving for the Web UI."""

from __future__ import annotations

from starlette.responses import Response
from starlette.staticfiles import StaticFiles


class NoCacheStaticFiles(StaticFiles):
    """Serve Web UI assets without browser persistence.

    The Web UI uses native ES modules. Cache-busting only the entry module is
    not enough because nested static imports keep their own URLs. Serving the
    small UI asset bundle as no-store avoids stale module graphs after rebuilds.
    """

    async def get_response(self, path: str, scope) -> Response:  # type: ignore[override]
        response = await super().get_response(path, scope)
        if response.status_code < 400:
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


__all__ = ["NoCacheStaticFiles"]
