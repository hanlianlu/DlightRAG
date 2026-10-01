# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Connection tool names: readable, provider-safe, and unique across an owner's Connections."""

import hashlib
import re
from collections.abc import Iterable

import pytest

from dlightrag.application.connections.models import ConnectionsError, RemoteTool
from dlightrag.application.connections.naming import MAX_TOOL_NAME, name_catalogue
from dlightrag.engine.answer.execution.connection_binding import is_connection_tool

#: Letters, digits, and `_`, starting with a letter, at most 64 characters: the subset the
#: OpenAI-compatible, Response, Anthropic, and Gemini tool-name rules all accept.
_PROVIDER_SAFE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")
_ID = "a" * 32
_OTHER_ID = "b" * 32


def _names(
    *remote: str, label: str = "Notion", connection_id: str = _ID, others: Iterable[str] = ()
) -> list[str]:
    catalogue = tuple(
        RemoteTool(name, f"{name} description", {"type": "object"}) for name in remote
    )
    named = name_catalogue(catalogue, label=label, connection_id=connection_id, others=others)
    return [tool.local_name for tool in named]


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:6]


def test_a_name_says_which_connection_and_which_server_tool() -> None:
    assert _names("notion-search", "fetch") == ["mcp__Notion__notion_search", "mcp__Notion__fetch"]
    assert _names("hf_doc_search", label="Hugging Face") == ["mcp__Hugging_Face__hf_doc_search"]
    (tool,) = name_catalogue(
        (RemoteTool("notion-search", "Search the workspace.", {"type": "object"}),),
        label="Notion",
        connection_id=_ID,
    )
    # Only the local name is added; the server's own definition passes through untouched.
    assert (tool.remote_name, tool.description, tool.input_schema) == (
        "notion-search",
        "Search the workspace.",
        {"type": "object"},
    )


@pytest.mark.parametrize(
    ("label", "part"),
    [
        ("  Work / Notion!! ", "Work_Notion"),
        ("Café Zürich", "Cafe_Zurich"),
        ("我的 Notion", "Notion"),
        ("snake__case_", "snake_case"),
        ("1Password", "1Password"),
        ("A label far longer than any tool name should carry", "A_label_far_longer_than"),
    ],
)
def test_a_label_keeps_only_its_letters_and_digits(label: str, part: str) -> None:
    assert _names("search", label=label) == [f"mcp__{part}__search"]


def test_a_label_with_nothing_readable_takes_a_short_hash() -> None:
    assert _names("search", label="飞书") == [f"mcp__{_digest(_ID)}__search"]
    assert _names("search", label="飞书", connection_id=_OTHER_ID) == [
        f"mcp__{_digest(_OTHER_ID)}__search"
    ]


@pytest.mark.parametrize("label", ["Notion", "飞书", "x" * 100, "Work / Notion"])
@pytest.mark.parametrize("remote", ["search", "a.b-c", "-", "9lives", "y" * 128])
def test_every_name_is_provider_safe_and_a_connection_tool(label: str, remote: str) -> None:
    (name,) = _names(remote, label=label)
    assert _PROVIDER_SAFE.fullmatch(name)
    assert is_connection_tool(name)


def test_a_name_that_does_not_fit_keeps_a_readable_head_and_a_hash() -> None:
    room = MAX_TOOL_NAME - len("mcp__Notion__")
    exact, over = "a" * room, "get_" + "x" * room
    assert _names(exact) == [f"mcp__Notion__{exact}"]
    (name,) = _names(over)
    assert len(name) == MAX_TOOL_NAME
    assert name == f"mcp__Notion__{over[: room - 7]}_{_digest(over)}"
    # Names that differ only past the cut stay apart.
    twins = _names(over + "_one", over + "_two")
    assert len(set(twins)) == 2 and all(len(name) == MAX_TOOL_NAME for name in twins)


def test_a_hash_appears_only_where_sanitizing_would_collide() -> None:
    # The server's own `get_page` keeps its name; `get-page`, which sanitizing merged into
    # it, gains a hash; `get.item` merges into nothing and stays readable.
    assert _names("get_page", "get-page", "get.item") == [
        "mcp__Notion__get_page",
        f"mcp__Notion__get_page_{_digest('get-page')}",
        "mcp__Notion__get_item",
    ]
    # When no server name is already clean, each merged one gains its own hash.
    assert _names("a.b", "a-b") == [
        f"mcp__Notion__a_b_{_digest('a.b')}",
        f"mcp__Notion__a_b_{_digest('a-b')}",
    ]


def test_a_catalogue_built_to_collide_is_rejected_whole() -> None:
    with pytest.raises(ConnectionsError, match="catalogue"):
        _names("a_b", "a-b", f"a_b_{_digest('a-b')}")


def test_connections_that_share_a_label_never_share_a_name() -> None:
    first = _names("search", "fetch")
    second = _names("search", "fetch", connection_id=_OTHER_ID, others=first)
    assert first == ["mcp__Notion__search", "mcp__Notion__fetch"]
    assert second == [
        f"mcp__Notion_{_digest(_OTHER_ID)}__search",
        f"mcp__Notion_{_digest(_OTHER_ID)}__fetch",
    ]
    # Labels that sanitize alike collide the same way.
    assert _names("search", label="Notion!", others=first, connection_id=_OTHER_ID) == [
        f"mcp__Notion_{_digest(_OTHER_ID)}__search"
    ]
    # A label that happens to spell out the hashed part leaves only the Connection id,
    # which no other Connection's part can be.
    taken = [*first, f"mcp__Notion_{_digest(_ID)}__other"]
    assert _names("search", others=taken) == [f"mcp__{_ID}__search"]


def test_names_are_stable_across_refreshes_of_an_unchanged_catalogue() -> None:
    first = _names("search", "fetch", others=["mcp__Jira__search"])
    assert _names("search", "fetch", others=["mcp__Jira__search"]) == first
    # Names of another shape, such as an older release's, take nothing.
    assert _names("search", "fetch", others=["mcp_" + "c" * 32 + "_" + "d" * 24]) == first
    # A tool the server adds leaves the others' names alone, unless sanitizing merges it with
    # one of them: then the server's own `get_page` takes the readable name from `get.page`,
    # which gains a hash, so one name can stand for another tool in a later generation.
    assert _names("search", "fetch", "create", others=["mcp__Jira__search"])[:2] == first
    assert _names("get.page") == ["mcp__Notion__get_page"]
    assert _names("get.page", "get_page") == [
        f"mcp__Notion__get_page_{_digest('get.page')}",
        "mcp__Notion__get_page",
    ]


def test_a_renamed_connection_publishes_new_names() -> None:
    assert _names("search", label="Notion") == ["mcp__Notion__search"]
    assert _names("search", label="Work Notion") == ["mcp__Work_Notion__search"]
