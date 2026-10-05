# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Open a built report the way the product shows an HTML Artifact.

The product inserts the Artifact's source into one ``<iframe srcdoc sandbox="allow-scripts">``,
under a Content-Security-Policy, after the reader activates it. ``frontend/ui/active-artifact-frame.ts``
is the source of truth for that wrapper, so this module parses the policy, the permissions and the
wrapper's shape from it at run time: a change to the product's boundary changes these tests with it.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from playwright.sync_api import Browser, ConsoleMessage, Frame, Page, Request, Response

_FRAME_SOURCE = Path(__file__).resolve().parents[2] / "frontend/ui/active-artifact-frame.ts"

# What the wrapper and the iframe are built from; a failed match means the product changed shape.
_SHAPE = (
    "[BASE_CSP[0], script, ...BASE_CSP.slice(1)].join('; ')",
    "\"script-src 'unsafe-inline'\"",
    "document.currentScript.remove();",
    "sandbox=${this.active ? 'allow-scripts' : ''}",
    '\'<meta name="color-scheme" content="light dark"></head><body>\'',
    "escapeBridge + source + '</body></html>'",
)

_VIOLATIONS = """
(() => {
  window.__violations = [];
  document.addEventListener('securitypolicyviolation', (event) => {
    window.__violations.push(`${event.violatedDirective} ${event.blockedURI}`);
  });
})();
"""


_STRING = re.compile(r""""((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)'""")


def _constant(name: str, source: str) -> list[str]:
    """The string literals of ``const NAME = [...]``, whichever quote each is written in."""
    body = re.search(rf"const {name} = \[(.*?)\]", source, re.DOTALL)
    assert body, f"{name} is gone from active-artifact-frame.ts"
    return [double or single for double, single in _STRING.findall(body.group(1))]


def frame_source() -> str:
    """The product's wrapper source, checked for the shape the helpers below rely on."""
    source = _FRAME_SOURCE.read_text(encoding="utf-8")
    for fragment in _SHAPE:
        assert fragment in source, f"the artifact wrapper changed: {fragment!r} is gone"
    return source


def policy() -> str:
    """The Content-Security-Policy of an active Artifact, as the product joins it."""
    csp = _constant("BASE_CSP", frame_source())
    return "; ".join([csp[0], "script-src 'unsafe-inline'", *csp[1:]])


def permissions() -> str:
    """The Permissions-Policy list of the iframe's ``allow`` attribute."""
    return "; ".join(f"{name} 'none'" for name in _constant("PERMISSIONS", frame_source()))


def wrapper_document(report: str, token: str = "report-test") -> str:
    """The srcdoc the product builds for an active HTML Artifact."""
    source = frame_source()
    message = re.search(r"const ESCAPE_MESSAGE = '([^']+)'", source)
    assert message, "ESCAPE_MESSAGE is gone from active-artifact-frame.ts"
    bridge = (
        f"<script>(()=>{{const token={json.dumps(token)};document.currentScript.remove();"
        'window.addEventListener("keydown",event=>{'
        f'if(event.key==="Escape")parent.postMessage({{type:"{message.group(1)}",token}},"*");'
        "},true);})();</script>"
    )
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        f'<meta http-equiv="Content-Security-Policy" content="{policy()}">'
        '<meta name="referrer" content="no-referrer">'
        '<meta name="color-scheme" content="light dark"></head><body>'
        + bridge
        + report
        + "</body></html>"
    )


@dataclass
class Observed:
    """Everything the page reported while the report ran."""

    console: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def clean(self) -> bool:
        return not (self.console or self.errors or self.failed)


class ReportPage:
    """One built report open in the product's sandbox at a width and colour scheme."""

    def __init__(
        self,
        browser: Browser,
        report: Path,
        *,
        width: int,
        height: int = 900,
        scheme: Literal["light", "dark"] = "light",
    ) -> None:
        self.context = browser.new_context(
            viewport={"width": width, "height": height}, color_scheme=scheme, device_scale_factor=1
        )
        self.context.add_init_script(_VIOLATIONS)
        self.page: Page = self.context.new_page()
        self.page.set_default_timeout(15000)
        self.observed = Observed()
        self.page.on("console", self._console)
        self.page.on("pageerror", lambda error: self.observed.errors.append(str(error)))
        self.page.on("requestfailed", self._failed)
        self.page.on("response", self._response)
        self.page.set_content(
            "<!doctype html><meta charset=utf-8><style>html,body{margin:0;background:white}"
            "iframe{border:0;display:block;width:100vw;height:100vh;background:white}</style>"
            '<iframe id="artifact" title="report" sandbox="allow-scripts" '
            f'referrerpolicy="no-referrer" allow="{permissions()}"></iframe>'
        )
        self.page.evaluate(
            "(html) => { document.getElementById('artifact').srcdoc = html; }",
            wrapper_document(report.read_text(encoding="utf-8")),
        )
        handle = self.page.wait_for_selector("#artifact")
        frame = handle.content_frame() if handle else None
        if frame is None:
            raise RuntimeError("the artifact iframe did not open")
        self.frame: Frame = frame
        self.frame.wait_for_function(
            "() => document.documentElement.dataset.reportReady === 'true'"
        )

    def _console(self, message: ConsoleMessage) -> None:
        # The host page's own ``allow`` attribute lists features this Chromium does not know.
        if message.type in ("error", "warning") and "Unrecognized feature" not in message.text:
            self.observed.console.append(f"{message.type}: {message.text[:200]}")

    def _response(self, response: Response) -> None:
        if response.status >= 400:
            self.observed.failed.append(f"{response.status} {response.url[:80]}")

    def _failed(self, request: Request) -> None:
        self.observed.failed.append(f"{request.url[:80]} {request.failure}")

    def violations(self) -> list[str]:
        return self.frame.evaluate("() => window.__violations || []")

    def eval(self, script: str, arg: Any = None) -> Any:
        return self.frame.evaluate(script, arg)

    def screenshot(self, path: Path) -> None:
        """A full-page shot: the frame grows to its document so the whole report is in view."""
        height = self.frame.evaluate("() => document.documentElement.scrollHeight")
        self.page.set_viewport_size({"width": self.page.viewport_size["width"], "height": height})  # type: ignore[index]
        self.page.wait_for_timeout(250)
        self.page.screenshot(path=str(path))

    def close(self) -> None:
        self.context.close()
