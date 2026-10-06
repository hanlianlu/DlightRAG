"""The product's active-artifact frame, derived from its own source at run time.

An HTML artifact is shown in `<dl-active-artifact-frame>` (frontend/ui/active-artifact-frame.ts): one
`<iframe srcdoc sandbox="allow-scripts">` whose document is `wrapperDocument(source, active, token)`,
i.e. a CSP meta, a referrer meta, a colour-scheme meta, an Escape bridge script and then the artifact
source inserted into <body>. Nothing of that is copied here. This module reads the TypeScript file,
cuts out `PERMISSIONS`, `ESCAPE_MESSAGE`, `BASE_CSP`, `wrapperDocument` and the iframe's template
attributes and style rules, strips the type annotations with Node's own `stripTypeScriptTypes`, and
evaluates them, so a change to the product's frame changes what the checker judges. When the file no
longer has the shape this expects, parsing raises FrameParseError saying which piece to update: a
checker that silently kept an old copy would judge a frame nobody ships.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # scripts/skill-evals/frame.py -> the repository root
DEFAULT_FRAME_TS = REPO / "frontend" / "ui" / "active-artifact-frame.ts"
_PLACEHOLDER = "\u0000ARTIFACT-SOURCE\u0000"
_ESCAPE_SENTINEL = "eval-harness-escape-token"


class FrameParseError(RuntimeError):
    """active-artifact-frame.ts no longer has the shape the checker reads."""


@dataclass(frozen=True)
class ProductFrame:
    """Everything needed to rebuild the product's frame around an artifact."""

    source_path: str
    csp: str  # the active frame's Content-Security-Policy
    sandbox: str  # the iframe's sandbox attribute when active
    referrerpolicy: str
    allow: str  # the iframe's `allow` (permissions policy) attribute
    css: str  # the component's style rules, with :host mapped to .host
    wrapper_head: str  # wrapperDocument() up to the artifact source
    wrapper_tail: str  # ... and after it
    fingerprint: str  # sha256 over all of the above: changes when the product's frame changes

    def wrap(self, artifact_source: str) -> str:
        """The srcdoc the product would give the iframe for this artifact."""
        return self.wrapper_head + artifact_source + self.wrapper_tail

    def page_html(self, width: int, height: int) -> str:
        """The embedding page: the component's host and boundary around its iframe, same CSS."""
        return (
            '<!doctype html><html><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            "<style>html,body{margin:0;padding:0;width:100%;height:100%;background:#fff}"
            f"{self.css}</style></head><body>"
            '<div class="host"><div class="boundary">'
            f'<iframe title="Artifact HTML preview" sandbox="{_attr(self.sandbox)}" '
            f'referrerpolicy="{_attr(self.referrerpolicy)}" allow="{_attr(self.allow)}"></iframe>'
            "</div></div></body></html>"
        )

    def describe(self) -> dict[str, str]:
        return {
            "source": self.source_path,
            "sandbox": self.sandbox,
            "referrerpolicy": self.referrerpolicy,
            "csp": self.csp,
            "allow_head": self.allow[:80] + "...",
            "fingerprint": self.fingerprint,
        }


def _attr(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;")


def _scan_statement(src: str, start: int, *, end_on_brace: bool = False) -> int:
    """Return the index just past the JS/TS statement (or block) that starts at `start`.

    A small scanner that knows strings, template literals (with `${}` nesting) and bracket depth,
    which is all the pieces cut out of active-artifact-frame.ts need.
    """
    depth = 0
    i = start
    n = len(src)
    while i < n:
        c = src[i]
        if c in "'\"":
            quote = c
            i += 1
            while i < n and src[i] != quote:
                i += 2 if src[i] == "\\" else 1
        elif c == "`":
            i = _skip_template(src, i)
            continue
        elif src.startswith("//", i):
            i = src.find("\n", i)
            i = n if i < 0 else i
            continue
        elif src.startswith("/*", i):
            i = src.find("*/", i) + 2
            continue
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if end_on_brace and depth == 0 and c == "}":
                return i + 1
        elif c == ";" and depth == 0 and not end_on_brace:
            return i + 1
        i += 1
    raise FrameParseError(f"unterminated statement starting at offset {start}")


def _skip_template(src: str, i: int) -> int:
    """`i` is at an opening backtick; return the index after the matching closing one."""
    i += 1
    while i < len(src):
        c = src[i]
        if c == "\\":
            i += 2
        elif c == "`":
            return i + 1
        elif c == "$" and src.startswith("${", i):
            depth = 1
            i += 2
            while i < len(src) and depth:
                if src[i] == "`":
                    i = _skip_template(src, i)
                    continue
                if src[i] in "'\"":
                    q = src[i]
                    i += 1
                    while i < len(src) and src[i] != q:
                        i += 2 if src[i] == "\\" else 1
                elif src[i] == "{":
                    depth += 1
                elif src[i] == "}":
                    depth -= 1
                i += 1
        else:
            i += 1
    raise FrameParseError("unterminated template literal")


def _cut(src: str, pattern: str, what: str, *, block: bool = False) -> str:
    match = re.search(pattern, src, flags=re.MULTILINE)
    if match is None:
        raise FrameParseError(f"cannot find {what} in the frame source (pattern {pattern!r})")
    start = match.start()
    if block:
        brace = src.index("{", src.index(")", start))  # past the parameter list
        end = _scan_statement(src, brace, end_on_brace=True)
    else:
        end = _scan_statement(src, start)
    return src[start:end]


def _first(src: str, pattern: str, what: str) -> str:
    match = re.search(pattern, src)
    if match is None:
        raise FrameParseError(
            f"cannot find {what} in the frame source (pattern {pattern!r}); the product's template changed"
        )
    return match.group(1)


_NODE_EVAL = r"""
import { stripTypeScriptTypes } from 'node:module';
import { readFileSync } from 'node:fs';
const { ts, placeholder, token } = JSON.parse(readFileSync(0, 'utf8'));
const js = stripTypeScriptTypes(ts);
const run = new Function(
  js + '\nreturn { PERMISSIONS, ESCAPE_MESSAGE, BASE_CSP, wrapperDocument };'
);
const { PERMISSIONS, ESCAPE_MESSAGE, BASE_CSP, wrapperDocument } = run();
const doc = wrapperDocument(placeholder, true, token);
const at = doc.indexOf(placeholder);
if (at < 0) throw new Error('wrapperDocument() does not insert its source verbatim');
const csp = /http-equiv="Content-Security-Policy" content="([^"]*)"/.exec(doc);
if (!csp) throw new Error('wrapperDocument() carries no CSP meta');
process.stdout.write(JSON.stringify({
  permissions: PERMISSIONS,
  escapeMessage: ESCAPE_MESSAGE,
  baseCsp: BASE_CSP,
  csp: csp[1],
  head: doc.slice(0, at),
  tail: doc.slice(at + placeholder.length),
}));
"""


def load_product_frame(path: Path | str = DEFAULT_FRAME_TS) -> ProductFrame:
    """Parse the product's frame source into a ProductFrame (raises FrameParseError on drift)."""
    path = Path(path)
    src = path.read_text(encoding="utf-8")
    ts = "\n".join(
        [
            _cut(src, r"^const PERMISSIONS =", "PERMISSIONS"),
            _cut(src, r"^const ESCAPE_MESSAGE =", "ESCAPE_MESSAGE"),
            _cut(src, r"^const BASE_CSP =", "BASE_CSP"),
            _cut(src, r"^function wrapperDocument\(", "wrapperDocument()", block=True),
        ]
    )
    sandbox = _first(
        src, r"sandbox=\$\{this\.active \? '([^']*)' : ''\}", "the iframe's sandbox attribute"
    )
    referrerpolicy = _first(src, r'referrerpolicy="([^"]*)"', "the iframe's referrerpolicy")
    if "allow=${PERMISSIONS}" not in src:
        raise FrameParseError(
            "the iframe's allow attribute is no longer ${PERMISSIONS}; update frame.py"
        )
    if ".srcdoc=${wrapperDocument(this.source, this.active," not in src:
        raise FrameParseError(
            "the iframe's srcdoc is no longer wrapperDocument(source, active, token)"
        )
    css_raw = _first(src, r"static styles = css`([\s\S]*?)`;", "the component's static styles")
    if "iframe {" not in css_raw or ".boundary {" not in css_raw:
        raise FrameParseError("the component's styles no longer define .boundary and iframe rules")
    css = css_raw.replace(":host", ".host")
    argv = ["node", "--input-type=module", "-e", _NODE_EVAL]
    try:
        done = subprocess.run(  # noqa: S603 - fixed node argv; the TypeScript source goes in on stdin
            argv,
            input=json.dumps({"ts": ts, "placeholder": _PLACEHOLDER, "token": _ESCAPE_SENTINEL}),
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    except FileNotFoundError as exc:
        raise FrameParseError(
            "node is required to evaluate the product's wrapperDocument()"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise FrameParseError(
            f"evaluating the frame source failed: {exc.stderr.strip()[:600]}"
        ) from exc
    data = json.loads(done.stdout)
    if "script-src 'unsafe-inline'" not in data["csp"]:
        raise FrameParseError(
            "the active frame's CSP no longer allows inline script; the checker's premise is gone"
        )
    fingerprint = hashlib.sha256(
        "\u0001".join(
            [data["head"], data["tail"], sandbox, referrerpolicy, data["permissions"], css]
        ).encode()
    ).hexdigest()[:16]
    return ProductFrame(
        source_path=str(path),
        csp=data["csp"],
        sandbox=sandbox,
        referrerpolicy=referrerpolicy,
        allow=data["permissions"],
        css=css,
        wrapper_head=data["head"],
        wrapper_tail=data["tail"],
        fingerprint=fingerprint,
    )


if __name__ == "__main__":
    frame = load_product_frame(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FRAME_TS)
    print(json.dumps(frame.describe(), indent=2, ensure_ascii=False))
    print("wrapper head:", frame.wrapper_head)
    print("wrapper tail:", frame.wrapper_tail)
    print("css:", frame.css.strip()[:400])
