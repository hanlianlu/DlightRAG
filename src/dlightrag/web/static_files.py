"""Static file serving and cache policy for the Vite-owned Web UI."""

from pathlib import Path

from starlette.responses import Response
from starlette.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent / "static"
APP_DIR = STATIC_DIR / "app"


class WebStaticFiles(StaticFiles):
    """Cache hashed Vite assets forever and keep mutable support files fresh."""

    async def get_response(self, path: str, scope) -> Response:  # type: ignore[override]
        response = await super().get_response(path, scope)
        if response.status_code >= 400:
            return response
        if path.startswith("app/assets/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            return response
        # Vendored third-party assets are large and versioned with the package;
        # their default ETag/Last-Modified validators allow cheap 304 responses.
        if path.startswith("vendor/"):
            return response
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


__all__ = ["APP_DIR", "STATIC_DIR", "WebStaticFiles"]
