"""Static file serving for the Web UI."""

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
        # Vendored third-party assets (e.g. MathJax under vendor/) are immutable
        # and multi-megabyte, so keep their default validators (ETag/Last-Modified)
        # to allow 304 revalidation instead of re-downloading them every load.
        if response.status_code < 400 and not path.startswith("vendor/"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


__all__ = ["NoCacheStaticFiles"]
