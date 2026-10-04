# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""materialize: a Resource's admitted bytes become a workspace file, and nothing else happens."""

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from dlightrag.engine.agent.environment import AccessScheduler, local
from dlightrag.engine.agent.environment.local import LocalExecutionEnvironment
from dlightrag.engine.agent.session.ids import IntentId
from dlightrag.engine.agent.tools import AgentTool, ToolEffects, ToolResult, ToolRuntime
from dlightrag.engine.agent.tools.contracts import WorkspaceInventoryFacts, WorkspacePathFact
from dlightrag.engine.agent.tools.files import materialize_tool
from dlightrag.engine.answer.resources.models import ResourceInput
from dlightrag.engine.answer.resources.registry import (
    BrowserResourceInput,
    FetchedResourceBytes,
    ResourceEffectOwner,
    ResourceRegistry,
)
from dlightrag.engine.answer.tools.resources import make_admitted_bytes_reader
from tests.support.agent_browser import RecordingRenderer
from tests.support.dns import public_dns
from tests.support.public_http import serve_public_http
from tests.support.resources import call, tools
from tests.tool_helpers import recording_tool_runtime

# CRLF line ends and a quoted newline: what MarkItDown's text view of a CSV does not keep.
CSV = b'"Month", "1958"\r\n"JAN", 340\r\n"NOTE", "two\r\nlines"\r\n'
URL = "https://example.com/airtravel.csv"
DESTINATION = "tmp/data.csv"


@pytest.fixture
def web(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """The public Web, serving the CSV, and every request a Run sent it."""
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    requests: list[httpx.Request] = []

    def serve(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=CSV, headers={"content-type": "text/csv"})

    serve_public_http(monkeypatch, serve)
    return requests


class Run:
    """One Run's registry, what its sink made durable, and what its uploads cost to load."""

    def __init__(self, **limits: Any) -> None:
        self.stored: list[tuple[FetchedResourceBytes, ResourceEffectOwner | None]] = []
        self.loads = 0
        self.registry = ResourceRegistry(fetched_bytes_sink=self._store, **limits)

    async def __aenter__(self) -> Run:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.registry.aclose()

    async def _store(
        self, fetched: FetchedResourceBytes, owner: ResourceEffectOwner | None
    ) -> None:
        self.stored.append((fetched, owner))

    async def _load(self) -> bytes:
        self.loads += 1
        return CSV

    async def download(self) -> str:
        return await self.registry.admit_browser_resource(
            BrowserResourceInput("browser_download", CSV, "airtravel.csv", "text/csv", URL),
            effect_owner=ResourceEffectOwner("parent", IntentId.new()),
            index=0,
        )

    async def upload(self) -> str:
        return self.registry.register(
            ResourceInput(filename="airtravel.csv", declared_mime="text/csv", loader=self._load)
        )

    async def fetched(self) -> str:
        """A URL the Run read, as the model reads one before it copies it."""
        resource_id = self.registry.register_agent_url(URL)
        read, _ = tools(self.registry)
        await call(read, resource_id=resource_id)
        return resource_id

    async def restored(self) -> str:
        self.registry.restore_fetched_resource(
            resource_id="res-restored",
            ordinal=0,
            filename="airtravel.csv",
            mime_type="text/csv",
            url=URL,
            content=CSV,
            admission_origin="agent",
            acquisition="direct_http",
        )
        return "res-restored"


def workspace(tmp_path: Path) -> LocalExecutionEnvironment:
    """A workspace whose destination already holds a file, which a copy replaces."""
    (tmp_path / "tmp").mkdir()
    (tmp_path / DESTINATION).write_bytes(b"old")
    return LocalExecutionEnvironment(tmp_path)


def copying(env: LocalExecutionEnvironment, registry: ResourceRegistry) -> AgentTool:
    return materialize_tool(
        env, AccessScheduler(), admitted_bytes_reader=make_admitted_bytes_reader(registry)
    )


async def copy(
    tool: AgentTool,
    resource_id: str,
    path: str = DESTINATION,
    *,
    runtime: ToolRuntime | None = None,
) -> ToolResult:
    arguments = tool.input_model.model_validate({"resource_id": resource_id, "path": path})
    return await tool.execute(arguments, runtime or recording_tool_runtime([]))


@pytest.mark.parametrize(
    ("admit", "settles_again"),
    [(Run.download, False), (Run.upload, False), (Run.fetched, True), (Run.restored, False)],
    ids=["download", "upload", "fetched", "restored"],
)
async def test_a_resource_copies_byte_for_byte_and_is_accounted_like_write(
    tmp_path: Path, web: list[httpx.Request], admit: Any, settles_again: bool
) -> None:
    async with Run() as run:
        resource_id = await admit(run)
        tool = copying(workspace(tmp_path), run.registry)
        requests, settlements = len(web), len(run.stored)
        updates: list[ToolResult] = []
        runtime = recording_tool_runtime(updates, tool_name="materialize")

        result = await copy(tool, resource_id, runtime=runtime)

        assert result.is_error is False, result.text_content
        assert (tmp_path / DESTINATION).read_bytes() == CSV
        assert result.text_content == (
            f"materialized {resource_id} to {DESTINATION} (text/csv, {len(CSV)} bytes)"
        )
        mode = (tmp_path / DESTINATION).stat().st_mode
        assert result.effects == ToolEffects(
            workspace_inventory=WorkspaceInventoryFacts(
                upserts=(
                    WorkspacePathFact(
                        DESTINATION, "file", len(CSV), mode, hashlib.sha256(CSV).hexdigest()
                    ),
                )
            )
        )
        assert updates[0].subject == DESTINATION
        # Copying never sends a request. Web bytes a Run holds settle again with the copying
        # call, as with a repeated read, and bytes that are already durable do not.
        assert len(web) == requests
        assert len(run.stored) == settlements + settles_again
        if settles_again:
            fetched, owner = run.stored[-1]
            assert fetched.content == CSV
            assert owner == ResourceEffectOwner(runtime.execution_scope, runtime.intent_id)


async def test_bytes_with_no_declared_type_are_reported_as_opaque(tmp_path: Path) -> None:
    async with ResourceRegistry() as registry:
        resource_id = registry.register(ResourceInput(filename="blob.bin", content=b"\x00\x01"))

        result = await copy(copying(workspace(tmp_path), registry), resource_id, "tmp/blob.bin")

    assert result.text_content == (
        f"materialized {resource_id} to tmp/blob.bin (application/octet-stream, 2 bytes)"
    )
    assert (tmp_path / "tmp/blob.bin").read_bytes() == b"\x00\x01"


async def test_an_upload_loads_once_however_often_it_is_copied_and_read(tmp_path: Path) -> None:
    async with Run() as run:
        resource_id = await run.upload()
        tool = copying(workspace(tmp_path), run.registry)
        read, _ = tools(run.registry)

        await copy(tool, resource_id)
        await copy(tool, resource_id, "tmp/again.csv")
        await call(read, resource_id=resource_id)

        assert run.loads == 1
        assert (tmp_path / "tmp/again.csv").read_bytes() == CSV


async def test_it_never_fetches_or_renders(tmp_path: Path, web: list[httpx.Request]) -> None:
    renderer = RecordingRenderer({URL: "<html><body>rendered</body></html>"})
    async with ResourceRegistry(page_renderer=renderer) as registry:
        tool = copying(workspace(tmp_path), registry)
        link = registry.register_discovered_link(URL)
        assert link is not None

        unread = await copy(tool, link)

        assert unread.is_error is True
        assert f"read(resource_id={link!r})" in unread.text_content
        assert unread.effects == ToolEffects()

        # A Rendered Read gives the page a rendering and no bytes of its own.
        await registry.read(link, rendered=True, max_window_tokens=2000)
        rendering = await copy(tool, link)

        assert rendering.is_error is True
        assert 'browser(action="capture")' in rendering.text_content
        assert rendering.effects == ToolEffects()

    assert web == []
    assert renderer.calls == [URL], "only the explicit Rendered Read rendered"
    assert (tmp_path / DESTINATION).read_bytes() == b"old"


async def test_a_page_with_a_snapshot_and_a_rendering_copies_the_snapshot(
    tmp_path: Path, web: list[httpx.Request]
) -> None:
    renderer = RecordingRenderer({URL: "<html><body>rendered</body></html>"})
    async with ResourceRegistry(page_renderer=renderer) as registry:
        read, _ = tools(registry, rendered=True)
        resource_id = registry.register_agent_url(URL)
        await call(read, resource_id=resource_id)
        await call(read, resource_id=resource_id, rendered=True)

        result = await copy(copying(workspace(tmp_path), registry), resource_id)

    assert result.is_error is False, result.text_content
    assert (tmp_path / DESTINATION).read_bytes() == CSV
    assert (len(web), renderer.calls) == (1, [URL])


def sound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The workspace is as a Run finds it."""


def latch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A symbolic link left in the workspace latches it."""
    (tmp_path / "link").symlink_to("elsewhere")


def fill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A workspace whose quota is eight bytes."""
    monkeypatch.setattr(local, "WORKSPACE_MAX_BYTES", 8)


def occupy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A directory where the file is to go."""
    (tmp_path / DESTINATION).mkdir(parents=True)


def entries(root: Path) -> list[str]:
    """Everything a workspace holds, as a refused call must leave it."""
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))


@pytest.mark.parametrize(
    ("prepare", "path", "limits", "complaint", "loads"),
    [
        pytest.param(latch, DESTINATION, {}, "workspace integrity latched", 0, id="latched"),
        pytest.param(fill, DESTINATION, {}, "workspace quota exceeded", 1, id="full"),
        pytest.param(sound, "../x", {}, "path must not escape the workspace", 0, id="escaping"),
        pytest.param(occupy, DESTINATION, {}, "cannot overwrite a directory", 0, id="directory"),
        pytest.param(
            sound,
            DESTINATION,
            {"max_total_attachment_bytes": 8},
            "total attachment bytes exceeded",
            1,
            id="allowance",
        ),
    ],
)
async def test_a_refused_copy_writes_nothing_and_loads_only_what_it_must(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepare: Callable[[Path, pytest.MonkeyPatch], None],
    path: str,
    limits: dict[str, int],
    complaint: str,
    loads: int,
) -> None:
    prepare(tmp_path, monkeypatch)
    env = LocalExecutionEnvironment(tmp_path)
    before = entries(tmp_path)
    async with Run(**limits) as run:
        resource_id = await run.upload()

        result = await copy(copying(env, run.registry), resource_id, path)

        assert result.is_error is True
        assert complaint in result.text_content
        assert result.effects == ToolEffects()
        assert run.loads == loads
    assert entries(tmp_path) == before
    assert not (tmp_path.parent / "x").exists()
    assert env.quota_violation is None, "a refused copy latches nothing"
