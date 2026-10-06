"""The checker: judge ONE HTML file the way the product shows it.

The file is opened in a page that reproduces the product's active-artifact frame exactly (frame.py parses
frontend/ui/active-artifact-frame.ts at run time): an `<iframe srcdoc sandbox="allow-scripts">` of the
viewport's width, whose document is the product's wrapper with the artifact inserted into its <body>, under
the product's CSP. Playwright inspects the iframe's frame from outside; nothing in the artifact is patched.

The checks are generic: a hand-rolled baseline report has no known structure, a toolkit report has
`window.Report` and ECharts instances, and the same code must judge both. Every check returns pass / fail /
n/a with a one-line reason carrying the measured numbers; `n/a` always says why (never a silent pass).
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import boilerplate
from frame import ProductFrame, load_product_frame
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import Browser, BrowserContext, Dialog, Frame, Page, sync_playwright
from playwright.sync_api import Error as PlaywrightError

HERE = Path(__file__).resolve().parent
PROBES_JS = (HERE / "probes.js").read_text(encoding="utf-8")
# Bumped whenever a check's definition or measurement changes: a saved report judged by another version is judged again.
#   1  the first harness
#   2  html_published; timeline_or_events and assumptions_stated as defined in cases.json; chart_min_size min(viewport - 32, 280)
#   3  assumptions_stated: a lead-in introduces the assumptions (the word in the first 30 characters, or a short block ending in a colon);
#      a page describing itself ("... and all the assumptions.") no longer counts, and the strongest evidence is reported
#   4  slicer_changes_chart, controls_change_output, slicer_coverage: after a control is operated (and after it is reset) the page is given
#      time to take the effect (frames flushed, then quiet) before it is measured; before, the first difference ended the wait, and a slider's
#      value label differs at once while its chart follows a frame or a debounce later: sliders and scenario buttons "changed nothing"
CHECKER_VERSION = "4"
PHONE_MAX = 480
DEFAULT_VIEWPORTS = (360, 390, 820, 1280)
DEFAULT_HEIGHTS = (800, 1000)
MAX_SHOT_DEVICE_PX = 15000  # a screenshot taller than this is cut (and says so)

INIT_JS = r"""
(() => {
  if (window.__evalEvents) return;
  const ev = (window.__evalEvents = []);
  try {
    document.addEventListener('securitypolicyviolation', (e) => ev.push({ type: 'csp', directive: e.violatedDirective, blocked: e.blockedURI, source: e.sourceFile, line: e.lineNumber }), true);
    window.addEventListener('error', (e) => { const t = e.target; if (t && t !== window && t.tagName) ev.push({ type: 'resource-error', tag: t.tagName, src: t.currentSrc || t.src || t.href || '' }); }, true);
  } catch (e) { /* a page without a document yet */ }
})();
"""

# Chromium's own words when the sandbox refuses a call the document made.
SANDBOX_REFUSAL = re.compile(
    r"sandbox|allow-(?:modals|popups|forms|downloads|same-origin|top-navigation|pointer-lock)|Ignored call to|"
    r"Blocked (?:a )?(?:form|frame|opening|script)|is disallowed|SecurityError",
    re.IGNORECASE,
)
CSP_CONSOLE = re.compile(
    r"Refused to|violates the following Content Security Policy", re.IGNORECASE
)
BENIGN_ERRORS = re.compile(r"ResizeObserver loop|favicon", re.IGNORECASE)


# ---------------------------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------------------------


@dataclass
class CheckResult:
    status: str  # pass | fail | n/a
    reason: str
    details: dict[str, Any] = field(default_factory=dict)
    heuristic: bool = False

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status, "reason": self.reason}
        if self.heuristic:
            out["heuristic"] = True
        if self.details:
            out["details"] = self.details
        return out


def ok(reason: str, **details: Any) -> CheckResult:
    return CheckResult("pass", reason, details)


def bad(reason: str, **details: Any) -> CheckResult:
    return CheckResult("fail", reason, details)


def na(reason: str, **details: Any) -> CheckResult:
    return CheckResult("n/a", reason, details)


def as_list(value: Any) -> list[Any]:
    """A probe's list result; an error dict (the probe threw) or None is an empty list, the error is recorded by Session.call."""
    return value if isinstance(value, list) else []


def slug(text: str, fallback: str = "page") -> str:
    out = re.sub(r"[^\w]+", "-", text.strip().lower(), flags=re.UNICODE).strip("-")
    return (out or fallback)[:32]


def _round(x: float | None, n: int = 1) -> float | None:
    return None if x is None else round(x, n)


# ---------------------------------------------------------------------------------------------
# One load of the artifact at one viewport
# ---------------------------------------------------------------------------------------------


@dataclass
class FrameInfo:
    frame: Frame
    key: str  # stable across srcdoc swaps: the chain of iframe selectors
    depth: int
    scale: float  # cumulative visual scale of this frame (CSS transform on an ancestor iframe)


class Session:
    def __init__(
        self,
        browser: Browser,
        product: ProductFrame,
        width: int,
        height: int,
        *,
        color_scheme: str = "light",
    ):
        self.product, self.width, self.height = product, width, height
        self.phone = width <= PHONE_MAX
        self.dpr = 2 if self.phone else 1
        self.ctx: BrowserContext = browser.new_context(
            viewport={"width": width, "height": height},
            device_scale_factor=self.dpr,
            is_mobile=self.phone,
            has_touch=self.phone,
            locale="zh-CN",
            color_scheme=color_scheme,  # type: ignore[arg-type]
        )
        self.ctx.add_init_script(INIT_JS)
        self.ctx.set_default_timeout(20000)
        self.page: Page = self.ctx.new_page()
        self.events: list[dict[str, Any]] = []
        self.probe_errors: list[dict[str, str]] = []
        self.phase = "load"
        self._attach()

    # -- event capture -----------------------------------------------------------------
    def _attach(self) -> None:
        page = self.page
        page.on(
            "pageerror", lambda e: self._event("pageerror", str(getattr(e, "message", e))[:400])
        )
        page.on("console", self._on_console)
        page.on(
            "requestfailed",
            lambda r: self._event("requestfailed", f"{r.method} {r.url[:160]} {r.failure}"),
        )
        page.on(
            "response",
            lambda r: (
                self._event("http-error", f"{r.status} {r.url[:160]}") if r.status >= 400 else None
            ),
        )
        page.on("dialog", self._on_dialog)
        page.on("popup", lambda p: self._event("popup", p.url[:160]))
        page.on("download", lambda d: self._event("download", d.suggested_filename))
        page.on("crash", lambda _page: self._event("crash", "the page crashed"))

    def _on_dialog(self, dialog: Dialog) -> None:
        self._event("dialog", f"{dialog.type}: {dialog.message[:120]}")
        dialog.dismiss()

    def _event(self, kind: str, text: str) -> None:
        self.events.append(
            {"kind": kind, "text": text, "phase": self.phase, "viewport": self.width}
        )

    def _on_console(self, msg: Any) -> None:
        if msg.type not in ("error", "warning"):
            return
        text = msg.text
        if BENIGN_ERRORS.search(text):
            return
        loc = msg.location or {}
        where = (
            f" ({loc.get('url', '')[:60]}:{loc.get('lineNumber', '')})" if loc.get("url") else ""
        )
        if CSP_CONSOLE.search(text):
            self._event("csp-console", text[:300])
        elif SANDBOX_REFUSAL.search(text):
            self._event("sandbox-refusal", text[:300] + where)
        elif msg.type == "error":
            self._event("console-error", text[:300] + where)
        else:
            self._event("console-warning", text[:200])

    def drain_page_events(self) -> None:
        """Pull the in-page CSP-violation / resource-error records of every frame."""
        for info in self.frames():
            try:
                for rec in info.frame.evaluate("(window.__evalEvents || []).splice(0)"):
                    if rec["type"] == "csp":
                        self._event(
                            "csp", f"{rec['directive']} blocked {str(rec['blocked'])[:100]}"
                        )
                    else:
                        self._event("resource-error", f"{rec['tag']} {str(rec['src'])[:120]}")
            except PlaywrightError:
                pass

    # -- loading -----------------------------------------------------------------------
    def load(self, source: str) -> None:
        self.page.set_content(self.product.page_html(self.width, self.height))
        self.page.evaluate(
            "(src) => { document.querySelector('iframe').srcdoc = src; }",
            self.product.wrap(source),
        )
        deadline = time.time() + 20
        while time.time() < deadline:
            frame = self.artifact_frame()
            if frame is not None:
                break
            time.sleep(0.1)
        frame = self.artifact_frame()
        if frame is None:
            raise RuntimeError("the artifact frame did not appear")
        try:
            frame.wait_for_load_state("load", timeout=30000)
        except PlaywrightError as exc:
            self._event("load-timeout", str(exc)[:200])
        self.settle()
        self.drain_page_events()

    def artifact_frame(self) -> Frame | None:
        main = self.page.main_frame
        for frame in self.page.frames:
            if frame.parent_frame is main and not frame.is_detached():
                return frame
        return None

    def top_frame(self) -> Frame:
        """The artifact's frame, for the checks that cannot go on without it."""
        frame = self.artifact_frame()
        if frame is None:
            raise RuntimeError("the artifact frame is not in the page")
        return frame

    def frames(self) -> list[FrameInfo]:
        top = self.artifact_frame()
        if top is None:
            return []
        out: list[FrameInfo] = []

        def walk(frame: Frame, key: str, depth: int, parent_scale: float) -> None:
            scale = parent_scale
            if depth > 0:
                try:
                    handle = frame.frame_element()
                    parent = frame.parent_frame
                    if parent is None:
                        return
                    self.call(parent, "info")  # makes sure the probes are in the parent
                    ratio = parent.evaluate("(e) => window.__evalProbe.scaleOf(e)", handle)
                    sel = handle.evaluate(
                        "e => { const p = []; let c = e; while (c && c.nodeType === 1 && c !== document.documentElement) { let i = 1, s = c; while ((s = s.previousElementSibling)) i++; p.unshift(c.tagName.toLowerCase() + ':nth-child(' + i + ')'); c = c.parentElement; } return p.join('>'); }"
                    )
                    scale = parent_scale * float(ratio)
                    key = f"{key}/{sel}"
                except PlaywrightError:
                    return
            out.append(FrameInfo(frame, key, depth, scale))
            for child in frame.child_frames:
                if not child.is_detached():
                    walk(child, key, depth + 1, scale)

        walk(top, "top", 0, 1.0)
        return out

    # -- probes ------------------------------------------------------------------------
    def call(self, frame: Frame, name: str, arg: Any = None) -> Any:
        if not frame.evaluate("typeof window.__evalProbe !== 'undefined'"):
            frame.evaluate(PROBES_JS)
        result = frame.evaluate(f"(a) => window.__evalProbe[{json.dumps(name)}](a)", arg)
        if isinstance(result, dict) and set(result) == {"error"}:
            # the probe threw inside the page: say so (a check built on it must not pass silently)
            self.probe_errors.append(
                {"probe": name, "error": str(result["error"])[:200], "viewport": str(self.width)}
            )
        return result

    def top_call(self, name: str, arg: Any = None) -> Any:
        frame = self.artifact_frame()
        return self.call(frame, name, arg) if frame is not None else None

    def gather(self, name: str, arg: Any = None) -> list[tuple[FrameInfo, Any]]:
        out = []
        for info in self.frames():
            try:
                out.append((info, self.call(info.frame, name, arg)))
            except PlaywrightError:
                continue
        return out

    # -- settling ----------------------------------------------------------------------
    def _signature(self) -> str:
        """A cheap fingerprint of chart sizes, page size and frame count; no per-frame scale lookups."""
        parts = []
        top = self.artifact_frame()
        if top is None:
            return "no-frame"
        stack = [top]
        while stack:
            frame = stack.pop()
            if frame.is_detached():
                parts.append("detached")
                continue
            try:
                parts.append(
                    frame.evaluate(
                        """() => { const r = [];
                          if (window.echarts && window.echarts.getInstanceByDom) for (const el of document.querySelectorAll('*')) {
                            let i = null; try { i = window.echarts.getInstanceByDom(el); } catch (e) {}
                            if (i && !i.isDisposed()) r.push([el.clientWidth, el.clientHeight, i.getWidth(), i.getHeight()]); }
                          return JSON.stringify([r, document.documentElement.scrollHeight, document.documentElement.scrollWidth, document.querySelectorAll('iframe').length]); }"""
                    )
                )
            except PlaywrightError:
                parts.append("detached")
            stack.extend(frame.child_frames)
        return "|".join(parts)

    def settle(self, timeout: float = 6.0, quiet: float = 0.45, animation: float = 0.0) -> None:
        """Wait until chart sizes and the page height stop changing, then for ECharts' animation."""
        start = time.time()
        last, stable = None, None
        while time.time() - start < timeout:
            sig = self._signature()
            if sig == last:
                stable = stable or time.time()
                if time.time() - stable >= quiet:
                    break
            else:
                last, stable = sig, None
            time.sleep(0.1)
        if animation:
            time.sleep(animation)

    # -- interaction -------------------------------------------------------------------
    def click(self, info: FrameInfo, sel: str) -> str:
        """A real click; falls back to element.click() when the control is covered or off-screen."""
        loc = info.frame.locator(sel).first
        try:
            loc.click(timeout=2500)
            return "click"
        except PlaywrightError:
            try:
                loc.evaluate("e => e.click()")
                return "js-click"
            except PlaywrightError as exc:
                return f"failed: {str(exc)[:80]}"

    def resize(self, height: int) -> None:
        self.page.set_viewport_size({"width": self.width, "height": height})

    def screenshot(self, path: Path, label: str = "") -> dict[str, Any]:
        """The whole page as one image: the iframe is made as tall as its document (up to a cap)."""
        top = self.artifact_frame()
        content_h = (
            int(top.evaluate("document.documentElement.scrollHeight")) if top else self.height
        )
        cap = int(MAX_SHOT_DEVICE_PX / self.dpr)
        target = max(self.height, min(content_h, cap))
        if target != self.height:
            self.resize(target)
            self.settle(timeout=4.0, quiet=0.3, animation=0.9)
        else:
            time.sleep(0.5)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.page.screenshot(path=str(path), full_page=False)
        if target != self.height:
            self.resize(self.height)
            self.settle(timeout=4.0, quiet=0.3)
        return {
            "file": path.name,
            "content_height": content_h,
            "captured_height": target,
            "truncated": content_h > cap,
            "dpr": self.dpr,
        }

    def fold_shot(self, path: Path) -> None:
        """What a person sees without scrolling: the iframe at its own height."""
        self.page.screenshot(path=str(path), full_page=False)

    def close(self) -> None:
        try:
            self.ctx.close()
        except PlaywrightError:
            pass


# ---------------------------------------------------------------------------------------------
# The measurements of one page state (a tab shown, at one viewport and height)
# ---------------------------------------------------------------------------------------------


def measure(sess: Session) -> dict[str, Any]:
    """Everything the per-viewport checks read, in the page's current state."""
    top = sess.top_frame()
    m: dict[str, Any] = {"viewport": sess.width, "height": sess.height}
    m["info"] = sess.call(top, "info")
    m["overflow"] = sess.call(top, "overflow")
    charts: list[dict[str, Any]] = []
    fonts: list[dict[str, Any]] = []
    iframes: list[dict[str, Any]] = []
    for info, data in sess.gather("charts"):
        if not isinstance(data, dict) or "error" in data:
            continue
        for c in data["echarts"]:
            c["frame"], c["frame_scale"] = info.key, info.scale
            charts.append(c)
        for h in data["hand"]:
            h["frame"], h["frame_scale"] = info.key, info.scale
            charts.append({**h, "hand": True})
        for im in data.get("images", []):
            im["frame"], im["frame_scale"] = info.key, info.scale
            charts.append({**im, "hand": True, "image": True, "visible": True})
        if data["hasLibrary"]:
            m.setdefault("frames_with_library", []).append(info.key)
    for info in sess.frames():
        try:
            data = sess.call(info.frame, "fonts", {"scale": info.scale})
        except PlaywrightError:
            continue
        if isinstance(data, dict) and "error" not in data:
            for item in data["below"]:
                item["frame"], item["frame_scale"] = info.key, info.scale
                fonts.append(item)
            m["font_min"] = min(
                [v for v in (m.get("font_min"), data["min"]) if v is not None], default=None
            )
            m["font_text_count"] = m.get("font_text_count", 0) + data["count"]
    for info, data in sess.gather("iframes"):
        if isinstance(data, list):
            for f in data:
                f["frame"] = info.key
                iframes.append(f)
    m["charts"], m["small_text"], m["iframes"] = charts, fonts, iframes
    if sess.phone:
        m["tap"] = sess.call(top, "tapTargets")
    m["frame_count"] = len(sess.frames())
    return m


# ---------------------------------------------------------------------------------------------
# Tabs and pages
# ---------------------------------------------------------------------------------------------


@dataclass
class Page_:
    label: str
    item: dict[str, Any] | None  # the tab item {sel, label, ...}; None for the single page
    frame_key: str = "top"


def _lines_changed(before: list[str], after: list[str]) -> float:
    if not before and not after:
        return 0.0
    b, a = set(before), set(after)
    return 1 - len(b & a) / max(len(b), len(a), 1)


def find_tabs(sess: Session) -> tuple[list[Page_], dict[str, Any]]:
    """The report's pages: ARIA tabs, or a row of buttons that switches most of the visible text."""
    top = sess.top_frame()
    cands = sess.top_call("tabCandidates") or {}
    info: dict[str, Any] = {
        "candidates": [
            {"kind": g["kind"], "n": len(g["items"]), "labels": [i["label"] for i in g["items"]]}
            for g in cands.get("groups", [])
        ],
        "data_page_sections": cands.get("dataPageSections"),
        "tabpanels": cands.get("tabpanels"),
        "sections": cands.get("sections"),
    }
    for group in cands.get("groups", []):
        items = group["items"]
        if group["kind"] == "aria":
            info.update(chosen=group["kind"], verified="aria roles", group=group)
            return [Page_(i["label"] or f"page{n + 1}", i) for n, i in enumerate(items)], info
        # a nav of buttons: it is a set of tabs only if clicking another one switches most of the visible text
        active = next((n for n, i in enumerate(items) if i["active"]), 0)
        other = (active + 1) % len(items)
        before = as_list(sess.call(top, "lines"))
        sess.click(FrameInfo(top, "top", 0, 1.0), items[other]["sel"])
        sess.settle(timeout=2.5, quiet=0.25)
        after = as_list(sess.call(top, "lines"))
        fraction = _lines_changed(before, after)
        sess.click(FrameInfo(top, "top", 0, 1.0), items[active]["sel"])
        sess.settle(timeout=2.5, quiet=0.25)
        info.setdefault("probe", []).append(
            {"group": group.get("desc"), "lines_changed": round(fraction, 2)}
        )
        if fraction >= TAB_SWITCH_FRACTION:
            info.update(
                chosen=group["kind"],
                verified=f"clicking '{items[other]['label']}' changed {fraction:.0%} of the visible lines",
                group=group,
            )
            return [Page_(i["label"] or f"page{n + 1}", i) for n, i in enumerate(items)], info
    info.update(chosen=None)
    return [Page_("main", None)], info


def activate(sess: Session, page: Page_) -> str:
    if page.item is None:
        return "single page"
    top = sess.top_frame()
    how = sess.click(FrameInfo(top, "top", 0, 1.0), page.item["sel"])
    sess.settle(timeout=4.0, quiet=0.35)
    time.sleep(
        0.45
    )  # a hidden page's charts are sized after the click: give a ResizeObserver its turn
    return how


def keyboard_tabs(sess: Session, group: dict[str, Any]) -> dict[str, Any]:
    """Can the tabs be switched without a pointer? Arrow keys on ARIA tabs, Enter/Space on buttons."""
    top = sess.top_frame()
    fi = FrameInfo(top, "top", 0, 1.0)
    items = group["items"]
    result: dict[str, Any] = {"focusable": [i["tabIndex"] >= 0 for i in items], "tried": []}
    if len(items) < 2:
        return result
    active_idx = next((n for n, i in enumerate(items) if i["active"]), 0)
    start, target = items[active_idx], items[(active_idx + 1) % len(items)]
    sess.click(fi, start["sel"])
    sess.settle(timeout=2.0, quiet=0.2)

    def active_of(sel: str) -> bool:
        cands = sess.call(top, "tabCandidates") or {}
        for g in cands.get("groups", []):
            for item in g["items"]:
                if item["sel"] == sel:
                    return bool(item["active"])
        return False

    def try_keys(label: str, sel: str, keys: list[str], expect: str) -> bool:
        """Focus `sel`, press `keys`; True when `expect` (a tab selector) became active or the page's text switched."""
        try:
            top.locator(sel).first.focus(timeout=2000)
        except PlaywrightError:
            result["tried"].append({"how": label, "focus": False})
            return False
        before_lines, before_active = as_list(sess.call(top, "lines")), active_of(expect)
        for k in keys:
            sess.page.keyboard.press(k)
        sess.settle(timeout=2.0, quiet=0.25)
        changed = _lines_changed(before_lines, as_list(sess.call(top, "lines")))
        flipped = active_of(expect) and not before_active
        result["tried"].append(
            {"how": label, "lines_changed": round(changed, 2), "became_active": flipped}
        )
        return flipped or changed >= TAB_SWITCH_FRACTION

    def reset() -> None:
        sess.click(fi, start["sel"])
        sess.settle(timeout=2.0, quiet=0.2)

    switched = try_keys(
        f"focus '{target['label']}' + Enter", target["sel"], ["Enter"], target["sel"]
    )
    if not switched:
        reset()
        switched = try_keys(
            f"focus '{target['label']}' + Space", target["sel"], ["Space"], target["sel"]
        )
    if not switched:
        reset()
        switched = try_keys(
            f"focus '{start['label']}' + ArrowRight", start["sel"], ["ArrowRight"], target["sel"]
        )
        if not switched:
            switched = try_keys(
                "then Enter after ArrowRight", start["sel"], ["Enter"], target["sel"]
            )
    result["switched"] = switched
    reset()
    return result


# ---------------------------------------------------------------------------------------------
# Interactions: slicers and scenario controls
# ---------------------------------------------------------------------------------------------

TAB_SWITCH_FRACTION = 0.2  # a tab switch changes at least this share of the visible lines (a filter changes far fewer)


RESET_LABEL = re.compile(
    r"重置|复位|清除|全部|所有|默认|reset|clear|all|default|restore", re.IGNORECASE
)


def _snapshot(sess: Session) -> dict[str, Any]:
    snap: dict[str, Any] = {"charts": {}, "svgs": {}, "text": "", "lines": []}
    texts = []
    for info, data in sess.gather("snapshot"):
        if not isinstance(data, dict) or "error" in data:
            continue
        for c in data["charts"]:
            snap["charts"][f"{info.key}|{c['key']}"] = (c["data"], c["title"])
        for s in data["svgs"]:
            snap["svgs"][f"{info.key}|{s['key']}"] = s["data"]
        texts.append(data["text"])
    snap["text"] = "|".join(texts)
    top = sess.top_frame()
    snap["lines"] = as_list(sess.call(top, "lines"))
    return snap


def _diff(base: dict[str, Any], now: dict[str, Any]) -> dict[str, Any]:
    keys = set(base["charts"]) | set(now["charts"])
    data_changed = [
        k
        for k in keys
        if (base["charts"].get(k) or (None,))[0] != (now["charts"].get(k) or (None,))[0]
    ]
    title_only = [
        k for k in keys if k not in data_changed and base["charts"].get(k) != now["charts"].get(k)
    ]
    svg_changed = [
        k for k in set(base["svgs"]) | set(now["svgs"]) if base["svgs"].get(k) != now["svgs"].get(k)
    ]
    return {
        "changed_keys": sorted(data_changed + svg_changed)[:40],
        "chart_data_changed": len(data_changed),
        "chart_title_changed": len(title_only),
        "svg_changed": len(svg_changed),
        "text_changed": _lines_changed(base["lines"], now["lines"]),
        "same": not data_changed
        and not title_only
        and not svg_changed
        and base["text"] == now["text"],
        "charts_same": not data_changed and not title_only and not svg_changed,
    }


# A control's effect is not instant. The value label of a slider changes at once, but a chart follows later: the runtime sends a slider's event
# once per animation frame, a handler may debounce, a chart is redrawn from rows. Two frames and a task later the page has run what a control
# scheduled; the quiet period then covers a debounce and an update that comes in steps.
FLUSH_FRAMES_JS = """() => new Promise((resolve) => {
  let n = 0;
  const tick = () => { if (++n >= 3) setTimeout(resolve, 0); else requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
  setTimeout(resolve, 600); // a page whose frames are paused must not hang the checker
})"""


def flush_frames(sess: Session) -> None:
    """Let every frame run the animation-frame work its controls scheduled."""
    for info in sess.frames():
        try:
            info.frame.evaluate(FLUSH_FRAMES_JS)
        except PlaywrightError:
            pass


def _signature(snap: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(sorted((k, v[0], v[1]) for k, v in snap["charts"].items())),
        tuple(sorted(snap["svgs"].items())),
        snap["text"],
    )


def settled_snapshot(
    sess: Session,
    min_wait: float = 0.45,
    quiet: float = 0.3,
    timeout: float = 2.5,
    poll: float = 0.08,
) -> dict[str, Any]:
    """The page's state once it has taken what a control did: frames flushed, at least `min_wait` seconds gone, and nothing it plots or says
    changed for `quiet` seconds (or `timeout`, for a page that never stops, e.g. an animation loop).

    Returning at the first difference measured the control's own value label and called the chart unchanged: the label differs at once, the chart
    a frame (or a debounce) later. Resetting a control before its chart had followed could even keep the chart from ever changing, because a
    slider's events are coalesced to the latest value.
    """
    start = time.time()
    flush_frames(sess)
    last = _snapshot(sess)
    last_sig = _signature(last)
    stable_since = time.time()
    while True:
        time.sleep(poll)
        now = _snapshot(sess)
        t = time.time()
        sig = _signature(now)
        if sig != last_sig:
            last_sig, stable_since = sig, t
        elif t - stable_since >= quiet and t - start >= min_wait:
            return now
        if t - start >= timeout:
            return now


# The second look: a control that changed what the page says but no chart yet may have a slow handler (a long debounce): it is given one more,
# longer wait before the checker says it changed no chart.
SLOW_SETTLE = {"min_wait": 1.0, "quiet": 0.4, "timeout": 3.0}


def run_interactions(
    sess: Session, pages: list[Page_], tab_selectors: set[str], limit: int = 36
) -> list[dict[str, Any]]:
    """Operate every non-tab control of every page and record what it changes; then try to reset it."""
    log: list[dict[str, Any]] = []
    budget = limit * max(1, len(pages))
    for page in pages:
        activate(sess, page)
        for finfo in sess.frames():
            controls = as_list(sess.call(finfo.frame, "controls", {"max": 90}))
            controls = [
                c
                for c in controls
                if isinstance(c, dict)
                and c["sel"] not in tab_selectors
                and c["kind"] not in ("tab", "link", "summary")
            ]
            controls = controls[:limit]
            groups: dict[str, list[dict[str, Any]]] = {}
            for c in controls:
                groups.setdefault(c.get("group") or c["sel"], []).append(c)
            base = settled_snapshot(sess)
            for c in controls:
                if budget <= 0:
                    break
                budget -= 1
                entry = {
                    "page": page.label,
                    "control": c["desc"],
                    "label": c["label"],
                    "kind": c["kind"],
                    "group": c.get("groupDesc"),
                    "was_active": c["active"],
                }
                original = None
                try:
                    if c["kind"] == "range":
                        original = c["value"]
                        finfo.frame.locator(c["sel"]).first.focus(timeout=2000)
                        sess.page.keyboard.press(
                            "End" if float(c["value"]) < float(c["max"]) else "Home"
                        )
                        entry["method"] = "keyboard"
                    elif c["kind"] == "select":
                        original = c["value"]
                        other = next((o for o in c.get("options", []) if o != c["value"]), None)
                        if other is None:
                            continue
                        finfo.frame.locator(c["sel"]).first.select_option(other, timeout=2500)
                        entry["method"] = "select"
                    else:
                        if c["active"] and c["kind"] in ("button", "radio"):
                            entry["skipped"] = "already the active choice"
                            log.append(entry)
                            continue
                        entry["method"] = sess.click(finfo, c["sel"])
                except PlaywrightError as exc:
                    entry["method"] = f"failed: {str(exc)[:80]}"
                    log.append(entry)
                    continue
                now = settled_snapshot(sess)
                delta = _diff(base, now)
                if delta["charts_same"] and not delta["same"]:
                    # the page's text moved (a slider's own value label does, at once) but no chart has followed: look again after a longer wait
                    now = settled_snapshot(sess, **SLOW_SETTLE)
                    delta = _diff(base, now)
                    if not delta["charts_same"]:
                        entry["late_effect"] = (
                            True  # the chart followed more than half a second after the control
                        )
                entry.update(delta)
                entry["reset"] = _try_reset(
                    sess, finfo, c, groups.get(c.get("group") or c["sel"], []), original, base
                )
                log.append(entry)
                after = _snapshot(sess)
                if not _diff(base, after)["charts_same"]:
                    base = after  # the reset did not restore it: continue from where the page is
    return log


def _try_reset(
    sess: Session,
    finfo: FrameInfo,
    c: dict[str, Any],
    siblings: list[dict[str, Any]],
    original: Any,
    base: dict[str, Any],
) -> dict[str, Any]:
    """Put the control back, by what a person would do, and say whether the page came back to its base."""
    tried = []

    def settled_same() -> bool:
        if _diff(base, settled_snapshot(sess))["charts_same"]:
            return True
        return _diff(base, settled_snapshot(sess, **SLOW_SETTLE))[
            "charts_same"
        ]  # a slow handler may still be drawing the page back

    if c["kind"] == "range" and original is not None:
        finfo.frame.locator(c["sel"]).first.evaluate(
            "(e, v) => { e.value = v; e.dispatchEvent(new Event('input', {bubbles: true})); e.dispatchEvent(new Event('change', {bubbles: true})); }",
            original,
        )
        tried.append("restore value")
        if settled_same():
            return {"restored": True, "by": tried}
    elif c["kind"] == "select" and original is not None:
        try:
            finfo.frame.locator(c["sel"]).first.select_option(original, timeout=2500)
        except PlaywrightError:
            pass
        tried.append("select original")
        if settled_same():
            return {"restored": True, "by": tried}
    elif c["kind"] == "checkbox":
        # a checkbox is put back by toggling it, whatever else shares its toolbar
        sess.click(finfo, c["sel"])
        tried.append("click again (toggle)")
        if settled_same():
            return {"restored": True, "by": tried}
    else:
        # the initially active choice among controls of the same kind (a chip row, a radio group)
        same_kind = [s_ for s_ in siblings if s_["kind"] == c["kind"]]
        active_sibling = next(
            (s_ for s_ in same_kind if s_["active"] and s_["sel"] != c["sel"]), None
        )
        if active_sibling is not None:
            sess.click(finfo, active_sibling["sel"])
            tried.append(f"click initially active '{active_sibling['label']}'")
            if settled_same():
                return {"restored": True, "by": tried}
    # a reset control anywhere on the page
    for control in as_list(sess.call(finfo.frame, "controls", {"max": 90})):
        if (
            control["kind"] in ("button", "link")
            and RESET_LABEL.search(control["label"] or "")
            and control["sel"] != c["sel"]
        ):
            sess.click(finfo, control["sel"])
            tried.append(f"click '{control['label']}'")
            if settled_same():
                return {"restored": True, "by": tried}
            break
    if c["kind"] in ("button", "radio") and "click again (toggle)" not in tried:
        sess.click(finfo, c["sel"])
        tried.append("click again (toggle)")
        if settled_same():
            return {"restored": True, "by": tried}
    return {"restored": False, "by": tried}


# ---------------------------------------------------------------------------------------------
# The checker
# ---------------------------------------------------------------------------------------------


class Checker:
    def __init__(
        self,
        out_dir: Path | None = None,
        viewports: tuple[int, ...] = DEFAULT_VIEWPORTS,
        heights: tuple[int, ...] = DEFAULT_HEIGHTS,
        product: ProductFrame | None = None,
        color_scheme: str = "light",
        interact_viewport: int | None = None,
        log: Callable[[str], None] | None = None,
    ):
        self.out_dir = out_dir
        self.viewports = tuple(viewports)
        self.heights = tuple(heights)
        self.product = product or load_product_frame()
        self.color_scheme = color_scheme
        wide = [v for v in self.viewports if v > PHONE_MAX]
        self.interact_viewport = interact_viewport or (max(wide) if wide else max(self.viewports))
        self.log = log or (lambda s: None)

    # -- the whole report ----------------------------------------------------------------
    def run(self, html_path: Path, query: str | None = None) -> dict[str, Any]:
        source = html_path.read_text(encoding="utf-8", errors="replace")
        t0 = time.time()
        doc: dict[str, Any] = {
            "file": str(html_path),
            "sha256": hashlib.sha256(source.encode()).hexdigest()[:16],
            "bytes": len(source.encode()),
            "checker_version": CHECKER_VERSION,
            "frame": self.product.describe(),
            "viewports": list(self.viewports),
            "heights": list(self.heights),
            "color_scheme": self.color_scheme,
        }
        events: list[dict[str, Any]] = []
        probe_errors: list[dict[str, str]] = []
        run_errors: list[dict[str, str]] = []
        page_runs: list[dict[str, Any]] = []
        blocks: list[dict[str, Any]] = []
        static: dict[str, Any] = {}
        tabs_info: dict[str, Any] = {}
        keyboard: dict[str, Any] | None = None
        interactions: list[dict[str, Any]] = []
        shots: list[dict[str, Any]] = []
        heur: dict[str, Any] = {}
        tab_cache: dict[int, list[Page_]] = {}

        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                for width in self.viewports:
                    for height in self.heights:
                        primary = height == self.heights[0]
                        sess = Session(
                            browser, self.product, width, height, color_scheme=self.color_scheme
                        )
                        try:
                            self.log(f"  load {width}x{height}")
                            sess.load(source)
                            if primary:
                                pages, tinfo = find_tabs(sess)
                                tab_cache[width] = pages
                            else:
                                pages, tinfo = tab_cache.get(width) or find_tabs(sess)[0], {}
                            if primary and not tabs_info:
                                tabs_info = tinfo
                            if not static:
                                static = self._static(sess, source)
                            for index, page in enumerate(pages):
                                how = activate(sess, page) if index else "initial"
                                sess.phase = "walk"
                                m = measure(sess)
                                m.update(page=page.label, page_index=index, activation=how)
                                page_runs.append(m)
                                if not primary:
                                    continue
                                blocks.extend(self._blocks(sess))
                                if self.out_dir:
                                    name = f"{width}-" + (
                                        slug(page.label, f"p{index + 1}") if page.item else "main"
                                    )
                                    shot = sess.screenshot(self.out_dir / "shots" / f"{name}.png")
                                    shot.update(viewport=width, page=page.label, page_index=index)
                                    shots.append(shot)
                                    sess.fold_shot(self.out_dir / "shots" / f"{name}-fold.png")
                            sess.drain_page_events()
                            events.extend(sess.events)
                        except Exception as exc:  # noqa: BLE001 - one broken load must not lose the other viewports
                            run_errors.append(
                                {
                                    "stage": f"load {width}x{height}",
                                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                                }
                            )
                            self.log(f"  ! load {width}x{height} failed: {exc}")
                            events.extend(sess.events)
                        finally:
                            probe_errors.extend(sess.probe_errors)
                            sess.close()
                # interactions at one viewport, in a fresh load
                height = self.heights[0]
                sess = Session(
                    browser,
                    self.product,
                    self.interact_viewport,
                    height,
                    color_scheme=self.color_scheme,
                )
                try:
                    self.log(f"  interact {self.interact_viewport}x{height}")
                    sess.load(source)
                    sess.events.clear()
                    pages, tinfo = find_tabs(sess)
                    sess.phase = "interaction"
                    tab_sels = (
                        {i["sel"] for i in tinfo.get("group", {}).get("items", [])}
                        if tinfo.get("group")
                        else set()
                    )
                    if tinfo.get("group"):
                        keyboard = keyboard_tabs(sess, tinfo["group"])
                    interactions = run_interactions(sess, pages, tab_sels)
                    activate(sess, pages[0])
                    first_controls = sess.top_call("controls", {"max": 120})
                    heur["initial_controls"] = as_list(first_controls)
                    heur["tab_sels"] = sorted(tab_sels)
                    heur["timeline"] = sess.top_call("timeline")
                    heur["assumption_blocks"] = self._plain_blocks(sess)
                    sess.drain_page_events()
                    events.extend(sess.events)
                except Exception as exc:  # noqa: BLE001
                    run_errors.append(
                        {
                            "stage": "interactions",
                            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                        }
                    )
                    self.log(f"  ! interactions failed: {exc}")
                finally:
                    probe_errors.extend(sess.probe_errors)
                    sess.close()
                # the same page in a dark browser: does its text stay legible?
                sess = Session(
                    browser,
                    self.product,
                    self.interact_viewport,
                    self.heights[0],
                    color_scheme="dark",
                )
                try:
                    self.log(f"  dark scheme {self.interact_viewport}x{self.heights[0]}")
                    sess.load(source)
                    heur["dark"] = sess.top_call("contrast")
                    if self.out_dir:
                        sess.fold_shot(
                            self.out_dir / "shots" / f"dark-{self.interact_viewport}-fold.png"
                        )
                except Exception as exc:  # noqa: BLE001
                    run_errors.append(
                        {"stage": "dark scheme", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
                    )
                finally:
                    probe_errors.extend(sess.probe_errors)
                    sess.close()
            finally:
                browser.close()

        doc["tabs"] = {k: v for k, v in tabs_info.items() if k != "group"}
        doc["tab_group"] = tabs_info.get("group")
        doc["keyboard"] = keyboard
        doc["static"] = {k: v for k, v in static.items() if k not in ("scripts_text",)}
        doc["interactions"] = interactions
        doc["measurements"] = page_runs
        doc["shots"] = shots
        doc["events"] = events
        unique_errors = {(e["probe"], e["error"]): e for e in probe_errors}
        doc["probe_errors"] = list(unique_errors.values())
        doc["run_errors"] = run_errors
        doc["boilerplate_blocks"] = len(blocks)
        checks = self._checks(
            doc, static, tabs_info, keyboard, interactions, page_runs, events, blocks, heur, query
        )
        doc["checks"] = {k: v.to_json() for k, v in checks.items()}
        doc["elapsed_s"] = round(time.time() - t0, 1)
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.contact_sheet(doc)
        return doc

    # -- gathering -----------------------------------------------------------------------
    def _static(self, sess: Session, source: str) -> dict[str, Any]:
        """Facts about the document itself: scripts, references, iframes (from the first load)."""
        out: dict[str, Any] = {"scripts": [], "references": [], "iframes": []}
        for info, scripts in sess.gather("scripts"):
            if isinstance(scripts, list):
                for s in scripts:
                    s["frame"] = info.key
                    out["scripts"].append(s)
        for info, refs in sess.gather("references"):
            if isinstance(refs, list):
                for r in refs:
                    r["frame"] = info.key
                    out["references"].append(r)
        for info, ifr in sess.gather("iframes"):
            if isinstance(ifr, list):
                out["iframes"].extend({**f, "frame": info.key} for f in ifr)
        out["source_bytes"] = len(source.encode())
        out["frames"] = len(sess.frames())
        return out

    def _plain_blocks(self, sess: Session) -> list[dict[str, Any]]:
        """The artifact's own text blocks in document order (hidden tab panels included), without the nested frames'."""
        blocks = sess.top_call("textBlocks")
        return [b for b in blocks if isinstance(b, dict)] if isinstance(blocks, list) else []

    def _blocks(self, sess: Session) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for info, blocks in sess.gather("textBlocks"):
            if isinstance(blocks, list):
                found.extend(boilerplate.scan_blocks(blocks, info.key))
        return found

    # -- the checks ----------------------------------------------------------------------
    def _checks(
        self, doc, static, tabs_info, keyboard, interactions, runs, events, blocks, heur, query
    ) -> dict[str, CheckResult]:
        out: dict[str, CheckResult] = {}
        out["html_published"] = na(
            "a property of the Run (what it attached), not of the file: evaluate.py tasks fills it"
        )
        out["load_ok"] = self.check_load_ok(events)
        out["no_external"] = self.check_no_external(static)
        out["charts_via_echarts_runtime"] = self.check_charts_runtime(static, runs)
        out["no_horizontal_overflow"] = self.check_overflow(runs)
        out["pages_or_sections"] = self.check_pages(tabs_info, keyboard, runs)
        out["slicer_changes_chart"] = self.check_slicer(interactions, runs)
        out["slicer_coverage"] = self.check_slicer_coverage(interactions, runs)
        out["legend_not_dominant"] = self.check_legend(runs)
        out["chart_min_size"] = self.check_min_size(runs)
        out["min_font"] = self.check_min_font(runs)
        out["tap_targets"] = self.check_tap_targets(runs)
        out["restraint_no_tabs_no_slicers"] = self.check_restraint(tabs_info, heur)
        out["controls_change_output"] = self.check_controls(interactions, runs)
        out["no_boilerplate"] = self.check_boilerplate(blocks)
        out["timeline_or_events"] = self.check_timeline(heur, blocks)
        out["assumptions_stated"] = self.check_assumptions(heur)
        out["no_text_overlap"] = self.check_text_overlap(runs)
        out["dark_scheme_legible"] = self.check_dark(heur)
        out["skills_loaded"] = na(
            "needs a Run's tool calls (evaluate.py tasks fills it); the checker sees only the file"
        )
        return out

    # load_ok -----------------------------------------------------------------------
    def check_load_ok(self, events: list[dict[str, Any]]) -> CheckResult:
        def uniq(kind: str) -> list[str]:
            seen: list[str] = []
            for e in events:
                if e["kind"] == kind and e["text"] not in seen:
                    seen.append(e["text"])
            return seen

        pageerrors = uniq("pageerror")
        console = uniq("console-error")
        csp = uniq("csp")
        failed_raw = uniq("requestfailed") + uniq("http-error") + uniq("resource-error")
        failed, seen_urls = [], set()
        for text in failed_raw:
            url = re.search(r"https?://\S+|data:\S+", text)
            key = url.group(0).rstrip(" ,;") if url else text
            if key not in seen_urls:
                seen_urls.add(key)
                failed.append(text)
        sandbox = uniq("sandbox-refusal") + uniq("dialog") + uniq("popup") + uniq("download")
        crash = uniq("crash") + uniq("load-timeout")
        problems = {
            "uncaught_errors": pageerrors,
            "console_errors": console,
            "csp_violations": csp,
            "failed_requests": failed,
            "sandbox_refusals": sandbox,
            "crash_or_timeout": crash,
        }
        counts = {k: len(v) for k, v in problems.items() if v}
        warnings = len(uniq("console-warning"))
        if not counts:
            return ok(
                f"no uncaught error, CSP violation, failed request or sandbox refusal at any viewport ({warnings} console warning(s))",
                warnings=warnings,
            )
        first = next(iter(next(v for v in problems.values() if v)))
        return bad(
            "; ".join(f"{n} {k.replace('_', ' ')}" for k, n in counts.items())
            + f". First: {first[:160]}",
            **problems,
        )

    # no_external -------------------------------------------------------------------
    def check_no_external(self, static: dict[str, Any]) -> CheckResult:
        refs = static.get("references", [])
        if not refs:
            return ok("no src/href/url()/@import/<link> leaves the document")
        sample = "; ".join(f"{r['el']} {r['attr']}={r['value'][:60]}" for r in refs[:3])
        return bad(f"{len(refs)} reference(s) leave the document: {sample}", references=refs[:20])

    # charts_via_echarts_runtime ------------------------------------------------------
    HAND_PATTERNS = (
        ("niceTicks", re.compile(r"nice_?ticks?\b", re.IGNORECASE)),
        (
            "createElementNS svg",
            re.compile(r"createElementNS\s*\(\s*['\"`]http://www\.w3\.org/2000/svg"),
        ),
        ("canvas getContext", re.compile(r"\bgetContext\s*\(")),
        (
            "hand-written chart function",
            re.compile(
                r"function\s+(?:line|bar|pie|hbars?|area|scatter|spark(?:line)?|donut|radar)Chart\b|function\s+hbars?\b|const\s+(?:line|bar|pie)Chart\s*=",
                re.IGNORECASE,
            ),
        ),
    )

    def check_charts_runtime(
        self, static: dict[str, Any], runs: list[dict[str, Any]]
    ) -> CheckResult:
        scripts = static.get("scripts", [])
        libs = [s for s in scripts if s["isLibrary"]]
        # distinct charts over every page and viewport run: by frame key + selector
        echarts: dict[str, dict[str, Any]] = {}
        hand: dict[str, dict[str, Any]] = {}
        for run in runs:
            for c in run["charts"]:
                key = f"{c['frame']}|{c.get('sel')}"
                (hand if c.get("hand") else echarts)[key] = c
        nested = [f for f in static.get("iframes", [])] + [
            f for run in runs for f in run["iframes"]
        ]
        nested_unique = {(f.get("frame"), f.get("sel")) for f in nested}
        author = [
            s
            for s in scripts
            if not s["isLibrary"] and s["type"] in ("", "text/javascript", "module")
        ]
        hits: dict[str, list[str]] = {}
        for s in author:
            for name, rx in self.HAND_PATTERNS:
                if rx.search(s["text"]):
                    hits.setdefault(name, []).append(s["id"] or s["frame"])
        library_frames = sorted({k for run in runs for k in run.get("frames_with_library", [])})
        library_copies = len(libs)
        details = {
            "echarts_instances": len(echarts),
            "hand_drawn_elements": len(hand),
            "library_copies": library_copies,
            "frames_with_library": library_frames,
            "nested_iframes": len(nested_unique),
            "hand_drawn_code": {k: len(v) for k, v in hits.items()},
        }
        problems = []
        if not echarts and not hand:
            return bad(
                "no chart found: 0 ECharts instances, no chart-like svg/canvas, no large image (a chart drawn from HTML/CSS boxes alone would not be seen)",
                **details,
            )
        images = [c for c in hand.values() if c.get("image")]
        drawn = [c for c in hand.values() if not c.get("image")]
        if images:
            c0 = images[0]
            problems.append(
                f"{len(images)} static chart image(s) (e.g. a {c0['natural'][0]}x{c0['natural'][1]} {c0.get('kind', 'image')[11:].split(';')[0] or 'image'}, shown at {c0['rect']['w']:.0f}px wide; probably an echarts-render output) instead of live charts"
            )
        if drawn:
            problems.append(
                f"{len(drawn)} chart-like svg/canvas element(s) that are not ECharts instances"
            )
        if hits:
            problems.append(
                "hand-built chart code in the report's own script: " + ", ".join(sorted(hits))
            )
        if nested_unique:
            problems.append(f"{len(nested_unique)} nested iframe(s) (chart pages embedded)")
        if library_copies != 1:
            problems.append(
                f"ECharts library carried {library_copies}× in the document"
                + (
                    f" and loaded in {len(library_frames)} frame(s)"
                    if len(library_frames) > 1
                    else ""
                )
            )
        elif len(library_frames) > 1:
            problems.append(f"ECharts loaded in {len(library_frames)} frames")
        if not echarts:
            problems.append("no ECharts instance")
        if problems:
            return bad(f"{len(echarts)} ECharts instance(s); " + "; ".join(problems), **details)
        return ok(
            f"{len(echarts)} chart(s), all ECharts instances; library carried once; no hand-built chart code; no nested iframes",
            **details,
        )

    # no_horizontal_overflow --------------------------------------------------------
    def check_overflow(self, runs: list[dict[str, Any]]) -> CheckResult:
        failures = []
        worst = 0.0
        pages = 0
        for run in runs:
            ov = run["overflow"]
            if not isinstance(ov, dict) or "error" in ov:
                continue
            pages += 1
            w = run["viewport"]
            scroll = ov["scrollWidth"]
            over = scroll - w
            worst = max(worst, over)
            cut = ov.get("cutOffByBodyOverflow") or []
            offenders = ov.get("offenders") or []
            if over > 1 or cut:
                what = (
                    f"scrollWidth {scroll} > {w}"
                    if over > 1
                    else "content cut off by overflow-x:hidden"
                )
                who = (offenders or cut)[:1]
                failures.append(
                    {
                        "viewport": w,
                        "page": run["page"],
                        "height": run["height"],
                        "what": what,
                        "offender": (
                            f"{who[0]['sel']} right edge {who[0]['right']}" if who else None
                        ),
                    }
                )
        if not pages:
            return na("no measurable page")
        if not failures:
            return ok(
                f"scrollWidth <= viewport on all {pages} page state(s) (max overshoot {worst:.0f}px)"
            )
        seen = {}
        for f in failures:
            seen.setdefault((f["viewport"], f["page"], f["what"]), f)
        first = list(seen.values())[:3]
        text = "; ".join(
            f"{f['viewport']}px/{f['page']}: {f['what']}"
            + (f" ({f['offender']})" if f["offender"] else "")
            for f in first
        )
        return bad(
            f"{len(seen)} overflowing page state(s): {text}", failures=list(seen.values())[:12]
        )

    # pages_or_sections -------------------------------------------------------------
    def check_pages(
        self, tabs_info: dict[str, Any], keyboard: dict[str, Any] | None, runs: list[dict[str, Any]]
    ) -> CheckResult:
        group = tabs_info.get("group")
        sections = tabs_info.get("sections")
        if not group:
            return bad(
                f"no tabs: one scrolling page ({sections} section element(s), {len(tabs_info.get('candidates', []))} nav-like group(s) tried, none switches the visible text)",
                tabs=tabs_info.get("candidates"),
            )
        n = len(group["items"])
        problems = []
        if n < 2:
            problems.append("fewer than two tabs")
        if keyboard is not None and not keyboard.get("switched"):
            problems.append(
                "keyboard cannot switch the tabs ("
                + "; ".join(
                    f"{t['how']}: {'no focus' if t.get('focus') is False else str(t.get('lines_changed')) + ' of the lines changed, tab ' + ('active' if t.get('became_active') else 'not active')}"
                    for t in keyboard.get("tried", [])[:3]
                )
                + ")"
            )
        # each page's charts at the right size after switching
        bad_sizes = []
        for run in runs:
            for c in run["charts"]:
                if c.get("hand") or not c.get("visible"):
                    continue
                if (
                    abs(c["instW"] - (c["clientW"] - c.get("padX", 0))) > 3
                    or c["instH"] < 10
                    or c["instW"] < 10
                ):
                    bad_sizes.append(
                        f"{run['viewport']}px/{run['page']}: {c['desc']} instance {c['instW']}x{c['instH']} in a {c['clientW']}px box"
                    )
        if bad_sizes:
            problems.append(
                f"{len(bad_sizes)} chart(s) not sized to their box after switching, e.g. {bad_sizes[0]}"
            )
        details = {
            "tabs": [i["label"] for i in group["items"]],
            "kind": group["kind"],
            "verified": tabs_info.get("verified"),
            "keyboard": keyboard,
        }
        if problems:
            return bad(f"{n} tabs ({group['kind']}) but " + "; ".join(problems), **details)
        how = "; ".join(
            f"{t['how']} switched"
            for t in (keyboard or {}).get("tried", [])
            if t.get("became_active") or t.get("lines_changed", 0) >= TAB_SWITCH_FRACTION
        )[:90]
        return ok(
            f"{n} tabs ({group['kind']}): {tabs_info.get('verified')}; keyboard ok ({how}); charts sized to their boxes after switching",
            **details,
        )

    # slicer_changes_chart ----------------------------------------------------------
    def _charts_exist(self, runs: list[dict[str, Any]]) -> bool:
        return any(run["charts"] for run in runs)

    @staticmethod
    def _chart_keys(runs: list[dict[str, Any]]) -> dict[str, str]:
        keys: dict[str, str] = {}
        for run in runs:
            for c in run["charts"]:
                if c.get("visible", True):
                    keys.setdefault(
                        f"{c['frame']}|{c.get('sel')}", f"{c.get('desc')} ({run['page']})"
                    )
        return keys

    def coverage(
        self, interactions: list[dict[str, Any]], runs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Which charts some control changes, and which no control touches."""
        keys = self._chart_keys(runs)
        responsive = set()
        for i in interactions:
            responsive |= set(i.get("changed_keys", []))
        responsive &= set(keys) | responsive
        static = [label for k, label in keys.items() if k not in responsive]
        return {
            "charts": len(keys),
            "responsive": len([k for k in keys if k in responsive]),
            "static": static[:20],
        }

    def check_slicer(
        self, interactions: list[dict[str, Any]], runs: list[dict[str, Any]]
    ) -> CheckResult:
        if not self._charts_exist(runs):
            return bad("no chart found, so nothing for a slicer to change")
        tried = [i for i in interactions if "chart_data_changed" in i]
        if not tried:
            return bad(
                "no slicer or filter control found (no non-tab button, select, range or checkbox)"
            )
        changing = [i for i in tried if i["chart_data_changed"] or i["svg_changed"]]
        kinds = sorted({i["kind"] for i in tried})
        cov = self.coverage(interactions, runs)
        if not changing:
            title = [i for i in tried if i["chart_title_changed"]]
            extra = f"; {len(title)} changed only a chart title" if title else ""
            text = [i for i in tried if i["text_changed"] > 0.05]
            extra += f"; {len(text)} changed other text" if text else ""
            return bad(
                f"{len(tried)} control(s) tried ({', '.join(kinds)}), none changed any chart's data{extra}",
                coverage=cov,
                tried=[{k: i[k] for k in ("page", "control", "label", "kind")} for i in tried[:12]],
            )
        restored = [i for i in changing if i["reset"]["restored"]]
        sample = changing[0]
        label = f"'{sample['label']}' ({sample['kind']}) changed {sample['chart_data_changed'] or sample['svg_changed']} chart(s)"
        coverage = f"{cov['responsive']} of {cov['charts']} chart(s) respond to some control"
        if not restored:
            return bad(
                f"{len(changing)} control(s) change chart data (e.g. {label}) but reset did not restore it; {coverage}",
                coverage=cov,
                changing=[
                    {k: i[k] for k in ("page", "control", "label", "reset")} for i in changing[:8]
                ],
            )
        return ok(
            f"{len(changing)} of {len(tried)} control(s) change chart data (e.g. {label}); reset restores it for {len(restored)}; {coverage}",
            coverage=cov,
            changing=[
                {k: i[k] for k in ("page", "control", "label", "kind", "chart_data_changed")}
                for i in changing[:12]
            ],
        )

    # slicer_coverage (extra, not in cases.json): the owner's "a chart cannot be filtered" -------------
    def check_slicer_coverage(
        self, interactions: list[dict[str, Any]], runs: list[dict[str, Any]]
    ) -> CheckResult:
        keys = self._chart_keys(runs)
        tried = [i for i in interactions if "chart_data_changed" in i]
        if not keys:
            return na("no chart found")
        if not tried:
            return na("no slicer or filter control found, so no chart is filterable by design")
        cov = self.coverage(interactions, runs)
        share = cov["responsive"] / cov["charts"]
        static = cov["static"]
        text = f"{cov['responsive']} of {cov['charts']} chart(s) respond to some control" + (
            f"; static: {', '.join(static[:4])}" + ("..." if len(static) > 4 else "")
            if static
            else ""
        )
        res = CheckResult("pass" if share >= 0.8 else "fail", text, {"coverage": cov})
        res.details["extra"] = True
        return res

    # legend_not_dominant -----------------------------------------------------------
    def check_legend(self, runs: list[dict[str, Any]]) -> CheckResult:
        seen: dict[tuple, dict[str, Any]] = {}
        evaluated = 0
        for run in runs:
            for c in run["charts"]:
                if c.get("hand") or not c.get("visible") or c.get("describeError"):
                    continue
                evaluated += 1
                axes = [
                    a["labelSize"]
                    for a in c.get("axes", [])
                    if a.get("labelShown") and a.get("labelSize")
                ]
                body = c.get("bodySizes") or []
                ref = max(axes) if axes else (statistics.median(body) if body else 12.0)
                refsrc = (
                    "axis labels" if axes else ("series/body text" if body else "the 12px default")
                )
                title_sizes = [
                    t["textSize"]
                    for t in c.get("titles", [])
                    if t["show"] and t["text"] and t.get("textSize")
                ] + [
                    t["subtextSize"]
                    for t in c.get("titles", [])
                    if t["show"] and t["subtext"] and t.get("subtextSize")
                ]
                legend_sizes = [
                    legend["size"]
                    for legend in c.get("legends", [])
                    if legend["show"] and legend["size"]
                ]
                legend_h = max(
                    [
                        (legend["rect"]["h"] / c["instH"])
                        for legend in c.get("legends", [])
                        if legend["show"] and legend["rect"] and c["instH"]
                    ]
                    or [0]
                )
                problems = []
                if title_sizes and max(title_sizes) > ref * 1.25 + 0.01:
                    problems.append(
                        f"title {max(title_sizes):g}px vs {ref:g}px {refsrc} (x{max(title_sizes) / ref:.2f})"
                    )
                if legend_sizes and max(legend_sizes) > ref * 1.25 + 0.01:
                    problems.append(
                        f"legend {max(legend_sizes):g}px vs {ref:g}px {refsrc} (x{max(legend_sizes) / ref:.2f})"
                    )
                if legend_h > 0.30:
                    problems.append(f"legend takes {legend_h:.0%} of the chart height")
                if problems:
                    key = (c["frame"], c["sel"], tuple(problems))
                    seen.setdefault(
                        key,
                        {
                            "viewport": run["viewport"],
                            "page": run["page"],
                            "chart": c["desc"],
                            "problems": problems,
                            "chart_size": f"{c['instW']}x{c['instH']}",
                        },
                    )
        if not evaluated:
            return na(
                "no ECharts instance to read legend, title and axis fonts from (hand-rolled or absent charts)"
            )
        if not seen:
            return ok(
                f"title and legend within 1.25x of the axis labels and under 30% of the chart height on {evaluated} chart state(s)"
            )
        first = list(seen.values())[0]
        return bad(
            f"{len(seen)} chart state(s) too dominant, e.g. {first['viewport']}px {first['chart']} ({first['chart_size']}): "
            + "; ".join(first["problems"]),
            offenders=list(seen.values())[:10],
        )

    # chart_min_size ----------------------------------------------------------------
    def check_min_size(self, runs: list[dict[str, Any]]) -> CheckResult:
        seen: dict[tuple, dict[str, Any]] = {}
        evaluated = 0
        for run in runs:
            vw = run["viewport"]
            for c in run["charts"]:
                if not c.get("visible", True):
                    continue
                rect = c["rect"]
                w, h = rect["w"] * c.get("frame_scale", 1.0), rect["h"] * c.get("frame_scale", 1.0)
                if w < 1 or h < 1:
                    continue
                evaluated += 1
                need_w = min(vw - 32, 280)
                problems = []
                if w < need_w - 0.5:
                    problems.append(f"{w:.0f}px wide (< {need_w}px)")
                if h < 200 - 0.5:
                    problems.append(f"{h:.0f}px tall (< 200px)")
                if not c.get("hand") and c.get("grids") and c.get("instW"):
                    left = min(g["x"] for g in c["grids"])
                    right = max(g["x"] + g["w"] for g in c["grids"])
                    ratio = (right - left) / c["instW"]
                    if ratio < 0.5:
                        problems.append(f"plot area {ratio:.0%} of the chart width (< 50%)")
                if problems:
                    key = (c["frame"], c.get("sel"), vw, tuple(problems))
                    seen.setdefault(
                        key,
                        {
                            "viewport": vw,
                            "page": run["page"],
                            "chart": c.get("desc"),
                            "problems": problems,
                        },
                    )
        if not evaluated:
            return na("no visible chart to measure")
        if not seen:
            return ok(
                f"{evaluated} chart state(s): none narrower than min(viewport - 32px, 280px), shorter than 200px, or with a plot area under 50% of its width"
            )
        first = list(seen.values())[0]
        return bad(
            f"{len(seen)} chart state(s) too small, e.g. {first['viewport']}px/{first['page']} {first['chart']}: "
            + "; ".join(first["problems"]),
            offenders=list(seen.values())[:12],
        )

    # min_font ----------------------------------------------------------------------
    def check_min_font(self, runs: list[dict[str, Any]]) -> CheckResult:
        worst: dict[tuple, dict[str, Any]] = {}
        measured = 0
        for run in runs:
            for item in run["small_text"]:
                worst.setdefault(
                    (item["sel"], round(item["size"], 1), run["viewport"]),
                    {**item, "viewport": run["viewport"], "page": run["page"], "source": "dom"},
                )
            for c in run["charts"]:
                if c.get("image"):
                    # text inside an image chart: the house theme's smallest text is 13px at the drawing's own size;
                    # echarts-render writes 2 PNG pixels per drawing pixel unless told otherwise, an SVG is 1:1
                    per_px = 1.0 if "svg" in c.get("kind", "") else 2.0
                    size = 13 * per_px * c["scale"] * c.get("frame_scale", 1.0)
                    measured += 1
                    if size < 10.95:
                        worst.setdefault(
                            (c["desc"], round(size, 1), run["viewport"]),
                            {
                                "sel": c["desc"],
                                "size": round(size, 1),
                                "text": "(chart image: 13px theme text x scale; a PNG is assumed to be echarts-render --scale 2)",
                                "viewport": run["viewport"],
                                "page": run["page"],
                                "source": "static chart image",
                            },
                        )
                    continue
                if c.get("hand") or not c.get("visible"):
                    continue
                measured += 1
                scale = c.get("frame_scale", 1.0) * c.get("scale", 1.0)
                smallest = c.get("minText")
                if smallest and smallest["size"] * scale < 10.95:
                    size = smallest["size"] * scale
                    worst.setdefault(
                        (c["desc"], round(size, 1), run["viewport"]),
                        {
                            "sel": c["desc"],
                            "size": round(size, 1),
                            "text": smallest["text"],
                            "viewport": run["viewport"],
                            "page": run["page"],
                            "source": "chart text",
                        },
                    )
            measured += run.get("font_text_count", 0) and 1
        if not measured:
            return na("no visible text measured")
        if not worst:
            return ok(
                "no visible text under 11px (DOM text and ECharts-drawn text, after CSS scaling)"
            )
        items = sorted(worst.values(), key=lambda i: i["size"])
        first = items[0]
        return bad(
            f"{len(items)} distinct text style(s) under 11px, smallest {first['size']}px ({first['source']}: {first['sel']} '{first['text'][:20]}' at {first['viewport']}px)",
            offenders=items[:12],
        )

    # tap_targets -------------------------------------------------------------------
    def check_tap_targets(self, runs: list[dict[str, Any]]) -> CheckResult:
        phone = [r for r in runs if r["viewport"] <= PHONE_MAX and r.get("tap")]
        if not phone:
            return na(
                f"no phone viewport (<= {PHONE_MAX}px) among {sorted({r['viewport'] for r in runs})}"
            )
        total = sum(
            r["tap"]["total"] for r in phone if isinstance(r["tap"], dict) and "total" in r["tap"]
        )
        if not total:
            return ok("no interactive control at phone widths")
        small: dict[tuple, dict[str, Any]] = {}
        for r in phone:
            t = r["tap"]
            if not isinstance(t, dict):
                continue
            for it in t.get("items", []):
                small.setdefault(
                    (it["sel"], round(it["h"]), r["viewport"]),
                    {**it, "viewport": r["viewport"], "page": r["page"]},
                )
        if not small:
            return ok(f"all {total} control(s) at phone widths are at least 36px high")
        first = list(small.values())[0]
        return bad(
            f"{len(small)} control style(s) under 36px high at phone widths, e.g. {first['sel']} '{first['label']}' {first['h']}px at {first['viewport']}px",
            offenders=list(small.values())[:12],
        )

    # restraint_no_tabs_no_slicers ---------------------------------------------------
    def check_restraint(self, tabs_info: dict[str, Any], heur: dict[str, Any]) -> CheckResult:
        tab_sels = set(heur.get("tab_sels", []))
        controls = [
            c
            for c in heur.get("initial_controls", [])
            if c["kind"] in ("button", "select", "range", "checkbox", "radio", "tab")
            and c["sel"] not in tab_sels
        ]
        tabs = tabs_info.get("group")
        found = []
        if tabs:
            found.append(f"{len(tabs['items'])} tabs")
        if controls:
            found.append(
                f"{len(controls)} control(s): "
                + ", ".join(f"{c['kind']} '{c['label'] or c['desc']}'" for c in controls[:4])
            )
        if tabs_info.get("data_page_sections", 0) >= 2:
            found.append(f"{tabs_info['data_page_sections']} data-page sections")
        if found:
            return bad("rendered " + "; ".join(found), controls=controls[:8])
        return ok("no tabs, no slicers, no button/select/range/checkbox rendered")

    # controls_change_output --------------------------------------------------------
    def check_controls(
        self, interactions: list[dict[str, Any]], runs: list[dict[str, Any]]
    ) -> CheckResult:
        if not self._charts_exist(runs):
            return bad("no chart found, so no control can change what is plotted")
        tried = [i for i in interactions if "chart_data_changed" in i]
        if not tried:
            return bad("no slider, scenario selector or other control found")
        changing = [i for i in tried if i["chart_data_changed"] or i["svg_changed"]]
        sliders = [i for i in tried if i["kind"] == "range"]
        slider_changing = [i for i in sliders if i["chart_data_changed"] or i["svg_changed"]]
        if not changing:
            return bad(
                f"{len(tried)} control(s) tried ({', '.join(sorted({i['kind'] for i in tried}))}); none changed the plotted series"
                + (
                    f"; {len(sliders)} slider(s) moved to an end stop changed nothing"
                    if sliders
                    else ""
                ),
                tried=[{k: i[k] for k in ("page", "control", "label", "kind")} for i in tried[:12]],
            )
        if sliders and not slider_changing:
            return bad(
                f"{len(sliders)} slider(s) present but moving them changed no chart; {len(changing)} other control(s) did",
                sliders=[i["control"] for i in sliders],
            )
        sample = (slider_changing or changing)[0]
        return ok(
            f"{len(changing)} control(s) change the plotted series (e.g. {sample['kind']} '{sample['label']}' changed {sample['chart_data_changed'] or sample['svg_changed']} chart(s)); {len(slider_changing)} of {len(sliders)} slider(s) do",
            changing=[
                {k: i[k] for k in ("page", "control", "label", "kind", "chart_data_changed")}
                for i in changing[:12]
            ],
        )

    # no_boilerplate ----------------------------------------------------------------
    def check_boilerplate(self, blocks: list[dict[str, Any]]) -> CheckResult:
        hits = boilerplate.dedupe(blocks)
        high = [h for h in hits if h["confidence"] == "high"]
        covered = {(h["path"], h.get("sentence")) for h in high}
        soft = [
            h
            for h in hits
            if h["confidence"] != "high" and (h["path"], h.get("sentence")) not in covered
        ]
        sentences = sorted({(h["path"], h.get("sentence", h["phrase"])) for h in high})
        details = {
            "hits": high,
            "soft_hits": soft,
            "by_category": {},
            "sentences": [s_[1] for s_ in sentences],
        }
        for h in high:
            details["by_category"][h["category"]] = details["by_category"].get(h["category"], 0) + 1
        if not high:
            tail = f" ({len(soft)} soft hit(s) listed)" if soft else ""
            return ok(
                "no AI/tool credit, disclaimer, privacy/compliance/copyright notice, generated-on/powered-by footer or call to action in the visible text"
                + tail,
                **details,
            )
        summary = "; ".join(
            f"'{h['phrase']}' [{h['category']}] in {h['element']}" for h in high[:4]
        )
        more = f" (+{len(high) - 4} more)" if len(high) > 4 else ""
        return bad(
            f"{len(high)} boilerplate hit(s) in {len(sentences)} sentence(s): {summary}{more}",
            **details,
        )

    # dark_scheme_legible (extra, not in cases.json) ----------------------------------------------------
    def check_dark(self, heur: dict[str, Any]) -> CheckResult:
        d = heur.get("dark")
        if not isinstance(d, dict) or "error" in d:
            return na("the dark-scheme probe did not run")
        if not d.get("chars"):
            return na("no visible text sampled in the dark scheme")
        share = d["low"] / d["chars"]
        details = {
            "extra": True,
            "schemeDark": d["schemeDark"],
            "canvas": d["canvas"],
            "worst": d["worst"],
        }
        rgb = [int(x) for x in re.findall(r"\d+", d["canvas"])[:3]] or [255, 255, 255]
        light_canvas = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) > 128
        mode = (
            "the page paints a light canvas in a dark browser"
            if light_canvas
            else "the page shows a dark canvas in a dark browser"
        ) + ("" if d["schemeDark"] else " (it opts out of the dark scheme)")
        if share > 0.02:
            w = d["worst"][0] if d["worst"] else {}
            return CheckResult(
                "fail",
                f"{share:.0%} of the text has contrast under 3:1 in a dark browser ({mode}; canvas {d['canvas']}); e.g. '{w.get('text', '')}' {w.get('color')} on {w.get('bg')} = {w.get('ratio')}:1",
                details,
            )
        return CheckResult(
            "pass",
            f"text stays legible in a dark browser ({mode}; {d['chars']} characters sampled, {share:.1%} under 3:1)",
            details,
        )

    # no_text_overlap (extra, not in cases.json) ------------------------------------------------------
    def check_text_overlap(self, runs: list[dict[str, Any]]) -> CheckResult:
        seen: dict[tuple, dict[str, Any]] = {}
        evaluated = 0
        for run in runs:
            for c in run["charts"]:
                if c.get("image") or not c.get("visible", True) or c.get("describeError"):
                    continue
                evaluated += 1
                pairs = c.get("overlaps") or []
                if pairs:
                    key = (c["frame"], c.get("sel"), run["viewport"])
                    seen.setdefault(
                        key,
                        {
                            "viewport": run["viewport"],
                            "page": run["page"],
                            "chart": c.get("desc"),
                            "pairs": pairs,
                        },
                    )
        if not evaluated:
            return na("no live chart (ECharts or hand-drawn svg) whose text boxes can be read")
        if not seen:
            return CheckResult(
                "pass",
                f"no two drawn texts cover each other on {evaluated} chart state(s)",
                {"extra": True},
            )
        first = list(seen.values())[0]
        p = first["pairs"][0]
        return CheckResult(
            "fail",
            f"{len(seen)} chart state(s) with overlapping text, e.g. {first['viewport']}px {first['chart']}: '{p['a']}' over '{p['b']}' ({len(first['pairs'])} pair(s))",
            {"extra": True, "offenders": list(seen.values())[:10]},
        )

    # timeline_or_events (heuristic, as cases.json defines it) --------------------------------------
    def check_timeline(self, heur: dict[str, Any], blocks: list[dict[str, Any]]) -> CheckResult:
        """At least five entries that begin with a year or a date, inside a list, a table or a .timeline element."""
        t = heur.get("timeline")
        if not isinstance(t, dict) or "error" in t:
            return na("the timeline probe did not run", probe=t)
        total, containers = t.get("total", 0), t.get("containers", [])
        details = {
            "total": total,
            "containers": containers,
            "extra_note": "`tl` is accepted as .timeline; a data table whose rows begin with a year also qualifies",
        }
        if total >= 5:
            best = containers[0]
            sample = " | ".join(best["sample"][:2])
            return CheckResult(
                "pass",
                f"{total} entries begin with a year or a date; largest group: {best['desc']} ({best['kind']}, {best['n']} entries, e.g. '{sample}')",
                details,
                heuristic=True,
            )
        return CheckResult(
            "fail",
            f"{total} entr{'y' if total == 1 else 'ies'} beginning with a year or a date inside a list, a table or a .timeline element (need 5)",
            details,
            heuristic=True,
        )

    # assumptions_stated (heuristic, as cases.json defines it) --------------------------------------
    ASSUME_RX = re.compile(r"假设|assumptions?", re.IGNORECASE)
    HEADINGISH = re.compile(r"^(h[1-6]|summary|legend|caption|figcaption|dt)$")

    LEAD_IN_WINDOW = 30  # a lead-in introduces the assumptions: the word is within the first characters of its block
    LEAD_IN_SHORT = 80  # or the block is short and ends with a colon, introducing what follows

    def check_assumptions(self, heur: dict[str, Any]) -> CheckResult:
        """A heading or lead-in containing 假设 or assumption(s), followed by at least one sentence with a number.

        Heading: h1-h6, summary, legend, caption, figcaption or dt holding the word anywhere. Lead-in: a block with the word in its first 30
        characters, or a block of at most 80 characters that ends with a colon. A page that merely DESCRIBES itself ("... and all the
        assumptions.") is not a lead-in, and neither is a mention inside a long paragraph. The verdict reports the strongest evidence it
        found: a passing heading before a passing lead-in, in document order.
        """
        blocks = [
            b
            for b in (heur.get("assumption_blocks") or [])
            if isinstance(b, dict) and not str(b.get("el", "")).startswith("echarts:")
        ]

        def tag_of(block: dict[str, Any]) -> str:
            return re.split(r"[.#]", str(block.get("el", "")), maxsplit=1)[0].lower()

        def numbered(text: str) -> list[str]:
            return [
                part.strip()
                for part in re.split(r"[。！？!?；;\n]", text)
                if re.search(r"\d", part)
            ]

        seen: list[dict[str, Any]] = []
        for i, block in enumerate(blocks):
            text = str(block["text"])
            match = self.ASSUME_RX.search(text)
            if not match:
                continue
            tag = tag_of(block)
            if self.HEADINGISH.match(tag):
                kind = "heading"
            elif match.start() <= self.LEAD_IN_WINDOW or (
                len(text) <= self.LEAD_IN_SHORT and text.rstrip().endswith((":", "\uff1a"))
            ):
                kind = "lead-in"
            else:
                continue
            following = [text[match.end() :]]
            for nxt in blocks[i + 1 : i + 9]:
                if self.HEADINGISH.match(tag_of(nxt)):
                    break
                following.append(str(nxt["text"]))
            sentences = [sentence for part in following for sentence in numbered(part)]
            seen.append(
                {"kind": kind, "block": text[:60], "tag": tag, "numbered_sentences": sentences[:2]}
            )
        passing = [f for f in seen if f["numbered_sentences"]]
        if passing:
            best = min(
                passing, key=lambda f: 0 if f["kind"] == "heading" else 1
            )  # min() keeps the first of equals: document order
            return CheckResult(
                "pass",
                f"{best['kind']} '{best['block'][:40]}' ({best['tag']}) is followed by a sentence with a number: "
                f"'{best['numbered_sentences'][0][:60]}'",
                {"found": seen},
                heuristic=True,
            )
        if seen:
            return CheckResult(
                "fail",
                f"{len(seen)} heading(s)/lead-in(s) mention 假设 or assumption(s), e.g. '{seen[0]['block']}' ({seen[0]['tag']}), "
                "but no sentence with a number follows",
                {"found": seen},
                heuristic=True,
            )
        return CheckResult(
            "fail",
            "no heading or lead-in containing 假设 or assumption(s) (a mention inside a long paragraph or a description "
            "of the page is not a lead-in)",
            {},
            heuristic=True,
        )

    # -- the contact sheet -----------------------------------------------------------------
    def contact_sheet(self, doc: dict[str, Any]) -> None:
        """One image per report: each viewport's first page, cropped to its top, side by side."""
        if self.out_dir is None:
            raise ValueError("the contact sheet needs an output directory")
        shots = [s for s in doc.get("shots", []) if s.get("page_index") == 0]
        if not shots:
            return
        target_h, crop_css = 1500, 2400
        tiles = []
        for s in sorted(shots, key=lambda s: s["viewport"]):
            path = self.out_dir / "shots" / s["file"]
            if not path.exists():
                continue
            im = Image.open(path).convert("RGB")
            dpr = s["dpr"]
            crop_h = min(im.height, int(crop_css * dpr))
            im = im.crop((0, 0, im.width, crop_h))
            scale = target_h / (crop_css * dpr) if crop_h >= crop_css * dpr else target_h / crop_h
            scale = (
                min(scale, 1.0 / dpr * (1.0 if s["viewport"] > PHONE_MAX else 1.0))
                if False
                else scale
            )
            tiles.append(
                (
                    s["viewport"],
                    im.resize(
                        (max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                        Image.Resampling.LANCZOS,
                    ),
                )
            )
        if not tiles:
            return
        pad, label_h = 14, 28
        width = sum(t.width for _, t in tiles) + pad * (len(tiles) + 1)
        height = max(t.height for _, t in tiles) + label_h + pad * 2
        sheet = Image.new("RGB", (width, height), (236, 236, 232))
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default()
        x = pad
        for vw, tile in tiles:
            draw.text((x, pad // 2 + 4), f"{vw}px", fill=(30, 30, 30), font=font)
            sheet.paste(tile, (x, label_h + pad // 2))
            x += tile.width + pad
        sheet.save(self.out_dir / "contact.png")
        # a second sheet with every page of the phone and the desktop view, when there are tabs
        pages = sorted({s["page_index"] for s in doc.get("shots", [])})
        if len(pages) > 1:
            for vw in (v for v in (min(doc["viewports"]), max(doc["viewports"]))):
                row = [s for s in doc["shots"] if s["viewport"] == vw]
                imgs = []
                for s in sorted(row, key=lambda s: s["page_index"]):
                    p = self.out_dir / "shots" / s["file"]
                    if p.exists():
                        im = Image.open(p).convert("RGB")
                        crop = im.crop((0, 0, im.width, min(im.height, int(1600 * s["dpr"]))))
                        sc = 1000 / crop.height
                        imgs.append(
                            crop.resize(
                                (max(1, int(crop.width * sc)), 1000), Image.Resampling.LANCZOS
                            )
                        )
                if imgs:
                    w = sum(i.width for i in imgs) + pad * (len(imgs) + 1)
                    sheet2 = Image.new("RGB", (w, 1000 + pad * 2), (236, 236, 232))
                    xx = pad
                    for i in imgs:
                        sheet2.paste(i, (xx, pad))
                        xx += i.width + pad
                    sheet2.save(self.out_dir / f"contact-pages-{vw}.png")


def check_file(
    path: Path,
    out_dir: Path | None,
    viewports: tuple[int, ...] = DEFAULT_VIEWPORTS,
    heights: tuple[int, ...] = DEFAULT_HEIGHTS,
    query: str | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    checker = Checker(out_dir=out_dir, viewports=viewports, heights=heights, log=log)
    return checker.run(path, query=query)
