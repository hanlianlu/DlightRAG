# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The ``browser`` tool: an Agent Session drives a page of the Run's Agent Browser (ADR 0032).

It is one tool with an ``action`` because its actions share one piece of state, the active
page and the refs of its latest snapshot. Actions that change the page answer with a bounded
snapshot; ``capture`` and a file a page downloads become Resources, and everything else a
page yields is context, never evidence. The tool is not read-only and never replays: each
call runs alone, and a call pending at a crash settles its outcome as unknown.
"""

from __future__ import annotations

import asyncio
import functools
import mimetypes
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Annotated, Any, Literal, Self, cast, get_args
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    create_model,
    model_validator,
)

from dlightrag.engine.agent.environment import AccessScheduler
from dlightrag.engine.agent.environment.access import PathAccess
from dlightrag.engine.agent.environment.errors import (
    TOOL_RESULT_MAX_BYTES,
    TOOL_RESULT_MAX_LINES,
    PathRejected,
)
from dlightrag.engine.agent.environment.execution import ExecutionEnvironment
from dlightrag.engine.agent.tool_content import ToolResourceAttachmentPart, ToolTextPart
from dlightrag.engine.agent.tools import (
    AgentTool,
    ResourceAttachmentBytes,
    ToolDeclaration,
    ToolEffects,
    ToolResult,
    ToolRuntime,
)
from dlightrag.engine.agent.tools.contracts import CommittedOutput
from dlightrag.engine.agent.tools.files import (
    ImagePreparer,
    ResourceReader,
    ResourceReadRequest,
    SpillWriter,
    head_excerpt,
    within_result_bounds,
    workspace_integrity_refusal,
)
from dlightrag.engine.agent.tools.listing import escape_path
from dlightrag.engine.answer.agent_browser import (
    MAX_DOWNLOADS_PER_CALL,
    AgentBrowserError,
    AgentPage,
    DownloadRefusal,
    PageEvents,
    PageObservation,
    PageState,
    RunAgentBrowser,
    UploadFile,
)
from dlightrag.engine.answer.resources.models import ResourceRegistryError
from dlightrag.engine.answer.resources.registry import (
    BROWSER_CAPTURE,
    BROWSER_DOWNLOAD,
    BrowserResourceInput,
    ResourceEffectOwner,
    ResourceRegistry,
    browser_capture_filename,
    browser_download_filename,
    browser_download_media_type,
)
from dlightrag.engine.public_http import (
    PublicHttpPolicyError,
    avalidate_public_http_url,
    validate_public_http_url,
)

BrowserAction = Literal[
    "navigate",
    "snapshot",
    "find",
    "back",
    "wait",
    "click",
    "type",
    "select",
    "press",
    "scroll",
    "upload",
    "screenshot",
    "capture",
]
BROWSER_ACTIONS: tuple[BrowserAction, ...] = get_args(BrowserAction)

_ACTION_LINES: dict[str, str] = {
    "navigate": "navigate (url): open a public http(s) URL in this session's page.",
    "snapshot": "snapshot: the current page's accessibility snapshot.",
    "find": "find (query): elements whose role, name, or text contains query, with their refs.",
    "back": "back: the previous page in this page's history.",
    "wait": "wait (text | text_gone | seconds): until text appears, text_gone disappears, or "
    "seconds pass.",
    "click": "click (ref): click the element.",
    "type": "type (ref, text, submit): replace the field's value with text; submit=true then "
    "presses Enter.",
    "select": "select (ref, values): choose options by value or label.",
    "press": "press (key, ref): press a key on the element, or on the page without ref.",
    "scroll": "scroll (direction, ref): scroll the page, or the container under the element, "
    "up or down.",
    "upload": "upload (ref, files): put workspace files into a file input.",
    "screenshot": "screenshot (full_page): attach the page's pixels; spends the Run's image "
    "budget.",
    "capture": "capture: admit the current page as a new citable Web Resource and return its "
    "first window.",
}

_REF = Annotated[str, StringConstraints(pattern=r"^(f[0-9]+)?e[0-9]+$", max_length=32)]
_KEY = Annotated[str, StringConstraints(pattern=r"^[\x21-\x7e]{1,64}$")]
_URL = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8192)]
_QUERY = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
_GONE = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
#: Typed text keeps its spaces, so nothing strips it.
_TEXT = Annotated[str, StringConstraints(max_length=10_000)]
_VALUE = Annotated[str, StringConstraints(max_length=1000)]
_PATH = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)]
_SECONDS = Annotated[float, Field(gt=0, le=30, allow_inf_nan=False)]

#: Each argument: its annotation, its field arguments, and the actions that read it. The
#: order is the schema's. An action reads only what it names, so every argument is optional.
_FIELDS: dict[str, tuple[Any, dict[str, Any], frozenset[str]]] = {
    "url": (
        _URL,
        {"description": "navigate: the public http(s) URL to open."},
        frozenset({"navigate"}),
    ),
    "query": (
        _QUERY,
        {"description": "find: text to look for in element roles, names, and text."},
        frozenset({"find"}),
    ),
    "ref": (
        _REF,
        {
            "description": "The element's ref from the latest snapshot or find of this page, such as e12."
        },
        frozenset({"click", "type", "select", "press", "scroll", "upload"}),
    ),
    "text": (
        _TEXT,
        {"description": "type: the text that replaces the field's value. wait: text to wait for."},
        frozenset({"type", "wait"}),
    ),
    "submit": (
        bool,
        {"description": "type: press Enter after the text."},
        frozenset({"type"}),
    ),
    "values": (
        list[_VALUE],
        {
            "min_length": 1,
            "max_length": 20,
            "description": "select: option values or labels to choose.",
        },
        frozenset({"select"}),
    ),
    "key": (
        _KEY,
        {"description": "press: a key such as Enter, Tab, Escape, ArrowDown, or Control+A."},
        frozenset({"press"}),
    ),
    "direction": (
        Literal["up", "down"],
        {"description": "scroll: up or down; down when omitted."},
        frozenset({"scroll"}),
    ),
    "text_gone": (
        _GONE,
        {"description": "wait: text to wait to disappear."},
        frozenset({"wait"}),
    ),
    "seconds": (
        _SECONDS,
        {"description": "wait: seconds to wait."},
        frozenset({"wait"}),
    ),
    "full_page": (
        bool,
        {"description": "screenshot: the whole page instead of its visible part."},
        frozenset({"screenshot"}),
    ),
    "files": (
        list[_PATH],
        {
            "min_length": 1,
            "max_length": 10,
            "description": "upload: workspace paths of the files to put into the file input.",
        },
        frozenset({"upload"}),
    ),
}
_REQUIRED: dict[str, tuple[str, ...]] = {
    "navigate": ("url",),
    "find": ("query",),
    "click": ("ref",),
    "type": ("ref", "text"),
    "select": ("ref", "values"),
    "press": ("key",),
    "upload": ("ref", "files"),
}

_DESCRIPTION = (
    "Drive this Run's Agent Browser for a multi-step task that one read cannot do: a search "
    "form, a filter, pagination behind a button, or a file behind a download control. Read "
    "pages with read, and pass rendered=true only when a read returned a JavaScript shell; "
    "use browser only for interaction. Each Agent Session has its own anonymous page for this "
    "Run. Actions that change the page return a depth-limited accessibility snapshot whose "
    "[ref=eN] markers name elements for the next action; refs come from the latest snapshot "
    "or find, and find locates what the snapshot does not show. Page text and screenshots are "
    "context, never evidence: capture admits the page as a citable Web Resource, and a file "
    "the page downloads becomes a Resource to read. If a page shows a CAPTCHA or any other "
    "human-verification check, stop that path and report it; never solve, bypass, or "
    "outsource it."
)


class BrowserArgs(BaseModel):
    """The arguments of one ``browser`` call; the model for the actions a Run offers extends it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: str

    @model_validator(mode="after")
    def _fields_match_action(self) -> Self:
        values = {name: getattr(self, name) for name in _FIELDS if name in type(self).model_fields}
        given = {name for name, value in values.items() if value is not None}
        if stray := sorted(name for name in given if self.action not in _FIELDS[name][2]):
            raise ValueError(f"browser {self.action} does not take {', '.join(stray)}")
        if missing := [name for name in _REQUIRED.get(self.action, ()) if name not in given]:
            raise ValueError(f"browser {self.action} requires {', '.join(missing)}")
        if self.action == "wait":
            if len(given & {"text", "text_gone", "seconds"}) != 1:
                raise ValueError("browser wait takes exactly one of text, text_gone, or seconds")
            if "text" in given and not values["text"].strip():
                raise ValueError("browser wait text cannot be blank")
        return self


@functools.cache
def browser_input_model(actions: tuple[str, ...]) -> type[BrowserArgs]:
    """The arguments of the actions a Run offers: only their fields, and their lines."""
    offered = set(actions)
    fields: dict[str, Any] = {
        "action": (
            cast(Any, Literal)[actions],
            Field(
                description="One action per call:\n" + "\n".join(_ACTION_LINES[a] for a in actions)
            ),
        ),
        **{
            name: (annotation | None, Field(default=None, **arguments))
            for name, (annotation, arguments, readers) in _FIELDS.items()
            if readers & offered
        },
    }
    return create_model("BrowserArgs", __base__=BrowserArgs, **fields)


def browser_declaration(*, upload: bool) -> ToolDeclaration:
    """The tool a Run offers. ``upload`` needs an Agent Workspace, so only ``trust`` has it.

    No configured value appears in the description or the schema, so changing a timeout or
    the depth never changes the plan a Run is pinned to.
    """
    actions = tuple(action for action in BROWSER_ACTIONS if upload or action != "upload")
    return ToolDeclaration(
        name="browser",
        description=_DESCRIPTION,
        input_model=browser_input_model(actions),
        replay_policy="never",
        read_only=False,
        contract_version=1,
    )


@dataclass(frozen=True, slots=True)
class BrowserRequest:
    """One validated call, with the defaults its actions assume."""

    action: BrowserAction
    url: str | None = None
    query: str | None = None
    ref: str | None = None
    text: str | None = None
    submit: bool = False
    values: tuple[str, ...] = ()
    key: str | None = None
    direction: Literal["up", "down"] = "down"
    text_gone: str | None = None
    seconds: float | None = None
    full_page: bool = False
    files: tuple[str, ...] = ()

    @classmethod
    def from_args(cls, args: BrowserArgs) -> BrowserRequest:
        # A model that does not offer an action has none of its fields.
        given = {name: getattr(args, name, None) for name in _FIELDS}
        return cls(
            action=cast(BrowserAction, args.action),
            url=given["url"],
            query=given["query"],
            ref=given["ref"],
            text=given["text"],
            submit=bool(given["submit"]),
            values=tuple(given["values"] or ()),
            key=given["key"],
            direction=given["direction"] or "down",
            text_gone=given["text_gone"],
            seconds=given["seconds"],
            full_page=bool(given["full_page"]),
            files=tuple(given["files"] or ()),
        )


@dataclass(frozen=True, slots=True)
class BrowserToolHost:
    """What the Run's ``browser`` tool reaches, shared by its Parent and every Child."""

    browser: RunAgentBrowser
    registry: ResourceRegistry
    read_resource: ResourceReader
    """The Run's own reader, which a capture is read through as ``read`` would read it."""


NEW_PAGE = "A new tab opened and is now the active page."
RETURNED = "The active page closed; the previous page is active again."
CLOSED = "The page closed itself; navigate to open a new one."
NAVIGATE_REFUSED = (
    "browser navigate refused this URL: {exc}. It opens public http(s) URLs without embedded "
    "credentials."
)
FOUND = '{total} element(s) match "{query}"'
FOUND_CUT = " (first {shown} shown)"
NOTHING_FOUND = 'No element matches "{query}".'
SPILLED = "browser snapshot exceeded {max_bytes} UTF-8 bytes or {max_lines} lines ({size} bytes); its head follows."
UNAVAILABLE = (
    "The action completed, but its full snapshot is unavailable: it exceeds {max_bytes} UTF-8 "
    "bytes or {max_lines} lines and this Run cannot keep it. Do not repeat the action; use find "
    "to locate an element. Its head follows."
)
UPLOAD_TOO_BIG = "upload sends at most {limit} MiB in one call."
UPLOAD_NOT_FILE = "upload needs regular workspace files: {path} is not one."
DOWNLOADED = (
    "Downloaded {filename} ({mime}, {size:,} bytes) as {resource_id} (browser_download); "
    "read(resource_id='{resource_id}') reads it."
)
NOT_ADMITTED = "The download {name} was not admitted: {why}."
CAPTURED = "Captured this page as Web Resource {resource_id} (browser_capture)."
NO_IMAGE = (
    "No screenshot was attached: it does not fit the remaining image budget, or this model "
    "does not accept images."
)
SHOT = (
    "Screenshot of the {extent} page ({media}, {size:,} model bytes); its pixels are context, "
    "not evidence."
)

_FOUND_SHOWN = 30
_FRAME_URL_CHARS = 500
#: Playwright's own limit on the files one call hands a page.
_MAX_UPLOAD_MIB = 50


def browser_tool(
    host: BrowserToolHost,
    *,
    environment: ExecutionEnvironment | None,
    scheduler: AccessScheduler,
    spill: SpillWriter | None,
    image_preparer: ImagePreparer | None,
) -> AgentTool:
    """Bind the ``browser`` declaration to the Run's browser, Resources, and workspace.

    Without an ``image_preparer`` the model takes no images, so a screenshot is refused.
    """

    async def execute(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        call = _Call(
            host,
            BrowserRequest.from_args(cast(BrowserArgs, raw)),
            runtime,
            environment=environment,
            scheduler=scheduler,
            spill=spill,
            image_preparer=image_preparer,
        )
        return await call.run()

    return browser_declaration(upload=environment is not None).bind(execute)


class _Call:
    """One ``browser`` call: the action asked for, and the pieces of the Run it uses."""

    def __init__(
        self,
        host: BrowserToolHost,
        request: BrowserRequest,
        runtime: ToolRuntime,
        *,
        environment: ExecutionEnvironment | None,
        scheduler: AccessScheduler,
        spill: SpillWriter | None,
        image_preparer: ImagePreparer | None,
    ) -> None:
        self._host = host
        self._request = request
        self._runtime = runtime
        self._environment = environment
        self._scheduler = scheduler
        self._spill = spill
        self._image_preparer = image_preparer
        self._scope = runtime.execution_scope
        self._owner = ResourceEffectOwner(self._scope, runtime.intent_id)

    async def run(self) -> ToolResult:
        if subject := self._subject():
            await self._runtime.emit_update(ToolResult.text("", subject=subject))
        try:
            return await self._act()
        except AgentBrowserError as exc:
            return ToolResult.text(exc.public_message, is_error=True)
        except ResourceRegistryError as exc:
            return ToolResult.text(str(exc), is_error=True)

    def _subject(self) -> str:
        """What the call acts on, for the live row: never typed text."""
        request = self._request
        if request.action == "navigate":
            return _label(request.url)
        if request.action == "find":
            return request.query or ""
        current = self._host.browser.current_url(self._scope)
        if current is None:
            return ""
        if request.ref is not None:
            return f"{request.ref} · {_label(current)}"
        return _label(current)

    async def _act(self) -> ToolResult:
        request = self._request
        match request.action:
            case "navigate":
                url = cast(str, request.url)
                try:
                    # Only the first URL is checked here; the deployment's network confines
                    # what the page loads after it.
                    validate_public_http_url(url)
                    await avalidate_public_http_url(url)
                except PublicHttpPolicyError as exc:
                    return ToolResult.text(NAVIGATE_REFUSED.format(exc=exc), is_error=True)
                return await self._acting(lambda p: p.navigate(url), open_page=True)
            case "back":
                return await self._acting(lambda p: p.back())
            case "snapshot":
                return await self._acting(lambda p: p.snapshot())
            case "wait":
                return await self._acting(
                    lambda p: p.wait(
                        text=request.text, text_gone=request.text_gone, seconds=request.seconds
                    )
                )
            case "click":
                return await self._acting(lambda p: p.click(cast(str, request.ref)))
            case "type":
                return await self._acting(
                    lambda p: p.type_text(
                        cast(str, request.ref), cast(str, request.text), submit=request.submit
                    )
                )
            case "select":
                return await self._acting(
                    lambda p: p.select(cast(str, request.ref), request.values)
                )
            case "press":
                return await self._acting(
                    lambda p: p.press(cast(str, request.key), ref=request.ref)
                )
            case "scroll":
                return await self._acting(
                    lambda p: p.scroll(direction=request.direction, ref=request.ref)
                )
            case "upload":
                return await self._upload()
            case "find":
                return await self._find()
            case "screenshot":
                return await self._screenshot()
            case "capture":
                return await self._capture()

    async def _on_page[T](
        self, call: Callable[[AgentPage], Awaitable[T]], *, open_page: bool = False
    ) -> T:
        return await self._host.browser.with_page(self._scope, call, open_page=open_page)

    async def _acting(
        self,
        call: Callable[[AgentPage], Awaitable[PageObservation]],
        *,
        open_page: bool = False,
    ) -> ToolResult:
        """A call that answers with the page and its snapshot."""
        observation = await self._on_page(call, open_page=open_page)
        report = await self._report(observation.page, observation.events)
        if observation.snapshot is None:
            return ToolResult.text(report)
        return await self._bounded(report, observation.snapshot)

    async def _find(self) -> ToolResult:
        query = cast(str, self._request.query)
        found = await self._on_page(lambda p: p.find(query, limit=_FOUND_SHOWN))
        report = await self._report(found.page, found.events)
        if not found.lines:
            return ToolResult.text("\n".join((report, NOTHING_FOUND.format(query=query))))
        count = FOUND.format(total=found.total, query=query)
        if found.total > len(found.lines):
            count += FOUND_CUT.format(shown=len(found.lines))
        return ToolResult.text("\n".join((report, f"{count}:", *found.lines)))

    async def _screenshot(self) -> ToolResult:
        request = self._request
        shot = await self._on_page(lambda p: p.screenshot(full_page=request.full_page))
        report = await self._report(shot.page, shot.events)
        label = _label(shot.page.url)
        prepared = (
            None
            if self._image_preparer is None
            else await asyncio.to_thread(self._image_preparer, shot.png, f"screenshot of {label}")
        )
        if prepared is None:
            return ToolResult.text("\n".join((report, NO_IMAGE)), is_error=True)
        digest = sha256(prepared.data).hexdigest()
        resource_id = (
            "res-shot-"
            + sha256(
                f"{self._scope}\0{self._runtime.intent_id.value}\0{digest}".encode()
            ).hexdigest()[:32]
        )
        text = SHOT.format(
            extent="whole" if request.full_page else "visible",
            media=prepared.media_type,
            size=len(prepared.data),
        )
        return ToolResult(
            parts=(
                ToolTextPart("\n".join((report, text))),
                ToolResourceAttachmentPart(
                    resource_id,
                    "screenshot",
                    prepared.media_type,
                    digest,
                    len(prepared.data),
                    data=prepared.data,
                    source=None,
                ),
            ),
            effects=ToolEffects(
                attached_resources=(
                    ResourceAttachmentBytes(
                        resource_id,
                        "screenshot",
                        prepared.media_type,
                        source_locator=label or resource_id,
                        content=prepared.data,
                    ),
                )
            ),
        )

    async def _capture(self) -> ToolResult:
        host = self._host
        cap = await self._on_page(lambda p: p.capture())
        report = await self._report(cap.page, cap.events)
        try:
            resource_id = await host.registry.admit_browser_resource(
                BrowserResourceInput(
                    BROWSER_CAPTURE,
                    cap.html,
                    browser_capture_filename(cap.page.url),
                    "text/html; charset=utf-8",
                    cap.page.url,
                ),
                effect_owner=self._owner,
                index=len(cap.events.downloads),
            )
        except ResourceRegistryError as exc:
            return ToolResult.text("\n".join((report, str(exc))), is_error=True)
        read = await host.read_resource(
            ResourceReadRequest(resource_id=resource_id, url=None, focus=None, cursor=None),
            self._runtime,
        )
        # What a read of the capture settles and cites is kept; only its words are framed.
        text = "\n".join((report, CAPTURED.format(resource_id=resource_id), read.text_content))
        return replace(read, parts=(ToolTextPart(text),))

    async def _upload(self) -> ToolResult:
        environment = cast(ExecutionEnvironment, self._environment)
        files = await self._workspace_files(environment)
        if isinstance(files, ToolResult):
            return files
        return await self._acting(lambda p: p.upload(cast(str, self._request.ref), files))

    async def _workspace_files(
        self, environment: ExecutionEnvironment
    ) -> tuple[UploadFile, ...] | ToolResult:
        """The workspace files an upload sends, or the result that says why it cannot."""
        if blocked := workspace_integrity_refusal(environment):
            return blocked
        files: list[UploadFile] = []
        total = 0
        for name in self._request.files:
            try:
                path = environment.resolve(name)
            except PathRejected as exc:
                return ToolResult.text(str(exc), is_error=True)
            async with self._scheduler.hold(PathAccess(path=str(path), kind="read")):
                if environment.stat_kind(path) != "file":
                    return ToolResult.text(
                        UPLOAD_NOT_FILE.format(path=escape_path(name)), is_error=True
                    )
                # A file is measured before it is read, so one that is too big is never loaded.
                total += path.stat().st_size
                if total > _MAX_UPLOAD_MIB * 1024 * 1024:
                    return ToolResult.text(
                        UPLOAD_TOO_BIG.format(limit=_MAX_UPLOAD_MIB), is_error=True
                    )
                content = environment.read_bytes(path)
            media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            files.append(UploadFile(path.name, media_type, content))
        return tuple(files)

    async def _report(self, page: PageState | None, events: PageEvents) -> str:
        """The frame that names the page, then a note for everything the pages did.

        The files they downloaded are admitted as Resources of this call as they are noted.
        """
        action = self._request.action
        if page is None:
            frame = f"[browser: {action} | the page closed]"
        else:
            url = page.url
            if len(url) > _FRAME_URL_CHARS:
                url = url[: _FRAME_URL_CHARS - 1] + "…"
            status = ""
            if events.http_status is not None and events.http_status >= 400:
                status = f" | HTTP {events.http_status}"
            frame = f"[browser: {action} | page: {url} | title: {page.title}{status}]"
        notes = [
            note
            for flag, note in (
                (events.new_page, NEW_PAGE),
                (events.returned, RETURNED),
                (events.closed, CLOSED),
            )
            if flag
        ]
        notes.extend(events.dialogs)
        notes.extend(await self._admit_downloads(events))
        notes.extend(self._refusals(events.refused_downloads))
        return "\n".join((frame, *notes))

    async def _admit_downloads(self, events: PageEvents) -> list[str]:
        """Admit what the pages downloaded as Resources of this call, one note for each."""
        notes = []
        for index, file in enumerate(events.downloads):
            filename = browser_download_filename(file.suggested_filename)
            media_type = browser_download_media_type(filename, file.content)
            try:
                resource_id = await self._host.registry.admit_browser_resource(
                    BrowserResourceInput(
                        BROWSER_DOWNLOAD, file.content, filename, media_type, file.url
                    ),
                    effect_owner=self._owner,
                    index=index,
                )
            except ResourceRegistryError as exc:
                notes.append(NOT_ADMITTED.format(name=filename, why=exc))
                continue
            notes.append(
                DOWNLOADED.format(
                    filename=filename,
                    mime=media_type,
                    size=len(file.content),
                    resource_id=resource_id,
                )
            )
        return notes

    def _refusals(self, refused: tuple[DownloadRefusal, ...]) -> list[str]:
        limits = self._host.browser.limits
        reasons = {
            "too_large": f"it exceeds {limits.max_download_bytes} bytes",
            "timeout": f"it did not finish within {limits.navigation_timeout:g} seconds",
            "failed": "it could not be downloaded",
            "limit": f"one call admits at most {MAX_DOWNLOADS_PER_CALL} downloads",
        }
        return [
            NOT_ADMITTED.format(
                name=browser_download_filename(refusal.suggested_filename),
                why=reasons[refusal.reason],
            )
            for refusal in refused
        ]

    async def _bounded(self, report: str, snapshot: str) -> ToolResult:
        """The report and the snapshot, with the snapshot kept in full when it is too big."""
        whole = "\n".join((report, snapshot))
        if within_result_bounds(whole):
            return ToolResult.text(whole)
        bounds = {"max_bytes": TOOL_RESULT_MAX_BYTES, "max_lines": TOOL_RESULT_MAX_LINES}
        excerpt = head_excerpt(snapshot).rstrip()
        if (receipt := await self._spilled(snapshot)) is None:
            # The action completed, so this is not a failure: reporting one would invite a
            # repeat, a second submit among them, and the head carries refs to act on.
            return ToolResult.text("\n".join((report, UNAVAILABLE.format(**bounds), excerpt)))
        protected = f"Full output: read(resource_id={receipt.resource_id!r}, cursor=...)"
        notice = SPILLED.format(size=len(snapshot.encode("utf-8")), **bounds)
        return ToolResult.text(
            "\n".join((report, notice, excerpt, protected)),
            protected_text=protected,
            effects=ToolEffects(committed_outputs=(receipt,)),
        )

    async def _spilled(self, snapshot: str) -> CommittedOutput | None:
        """The snapshot kept in full in the workspace, or None where nothing can keep it."""
        if self._spill is None:
            return None
        try:
            return await self._spill(snapshot)
        except OSError:
            return None


def _label(url: str | None) -> str:
    """A page by where it is, never by its query: the host and path, or the scheme of a non-page."""
    if not url:
        return ""
    try:
        parts = urlsplit(url)
    except ValueError:
        return ""
    if parts.scheme not in {"http", "https"}:
        return parts.scheme
    return f"{parts.hostname or ''}{parts.path}"


__all__ = [
    "BROWSER_ACTIONS",
    "BrowserArgs",
    "BrowserRequest",
    "BrowserToolHost",
    "browser_declaration",
    "browser_input_model",
    "browser_tool",
]
