# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Provider adapters, ordered failover, and Web Evidence projection."""

import asyncio
import inspect
import json
import logging
import re
from dataclasses import replace

import httpx
import pytest
from pydantic import Field, ValidationError

from dlightrag.engine.agent.tools.contracts import (
    ToolDeclaration,
    ToolResult,
)
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.tools.search import (
    SearchInput,
    WebSearchInput,
    _searched_workspaces,
    knowledge_base_search_declaration,
    knowledge_base_search_tool,
    web_search_declaration,
    web_search_tool,
)
from dlightrag.engine.answer.tools.web_search import web_context_rows
from dlightrag.engine.answer.web_sources.contracts import (
    WebExtractResult,
    WebSearchHit,
    WebSearchRequest,
    WebSearchResult,
    WebSourceUnavailable,
)
from dlightrag.engine.answer.web_sources.deepseek import DeepSeekWebSource, _search_query
from dlightrag.engine.answer.web_sources.exa import ExaWebSource
from dlightrag.engine.answer.web_sources.service import WebSourceService
from dlightrag.engine.answer.web_sources.tavily import TavilyWebSource
from dlightrag.engine.rag.retrieval.results import RetrievalResult
from tests.support.tool_helpers import recording_tool_runtime, tool_runtime

_PAGE = {
    "url": "https://example.org/taylor",
    "title": "The Taylor rule",
    "publishedDate": "2026-01-02T00:00:00.000Z",
    "image": "https://example.org/figure-1.png",
    "highlights": ["a is usually about 1.5", "b is usually about 0.5"],
}


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch):
    """Answer a provider adapter's own HTTP client from a mock transport."""

    def install(provider: str, handler) -> None:
        monkeypatch.setattr(
            f"dlightrag.engine.answer.web_sources.{provider}._default_client",
            lambda: _client(handler),
        )

    return install


def _responds(payload: dict, status: int = 200):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return handler


def test_web_search_schema_exposes_provider_neutral_controls() -> None:
    async def unused(_request: WebSearchRequest) -> WebSearchResult:
        return WebSearchResult(())

    tool = web_search_tool(
        search=unused,
        evidence=EvidenceLedger(),
        trace={"web_search_cost_dollars": 0.0},
        register_web_source=None,
    )

    assert set(tool.input_model.model_fields) == {
        "query",
        "max_results",
        "include_domains",
        "exclude_domains",
        "start_date",
        "end_date",
        "effort",
    }
    assert "source page, document, image, or file" in tool.description
    assert "Results are page excerpts." in tool.description
    # Every control says what it does, in the schema the model is given.
    properties = tool.definition.parameters["properties"]
    assert set(properties) == set(tool.input_model.model_fields)
    assert all(schema.get("description") for schema in properties.values())
    parsed = tool.input_model.model_validate(
        {
            "query": "policy",
            "max_results": 20,
            "include_domains": ["EXAMPLE.ORG"],
            "start_date": "2026-01-01",
            "end_date": "2026-01-31",
            "effort": "balanced",
        }
    )
    assert parsed.model_dump()["include_domains"] == ("example.org",)
    # The provider's own research tier is not a depth the model can ask for.
    with pytest.raises(ValidationError):
        tool.input_model.model_validate({"query": "q", "effort": "deep"})
    with pytest.raises(ValidationError):
        tool.input_model.model_validate({"query": "q", "max_results": 21})
    with pytest.raises(ValidationError):
        tool.input_model.model_validate(
            {"query": "q", "include_domains": ["a.example"], "exclude_domains": ["a.example"]}
        )


def test_knowledge_base_tool_states_its_mechanics_only() -> None:
    # Workspaces are created at runtime and each request selects its own, while a
    # tool definition is provider prefix-cache input the Run pins at acceptance. The
    # declaration therefore takes no input a workspace name or content could enter by.
    assert inspect.signature(knowledge_base_search_declaration).parameters == {}
    description = knowledge_base_search_declaration().description

    assert "Search the knowledge-base workspaces selected for this conversation" in description
    assert "Each passage it returns names the document it came from." in description


async def test_a_knowledge_base_result_names_what_the_search_covered() -> None:
    """The description names no workspace, so the result says which were searched."""

    async def retrieve(_query: str) -> RetrievalResult:
        return RetrievalResult(
            trace={
                "workspaces": ["hlyu", "default"],
                "per_workspace": {
                    "hlyu": {},
                    "default": {"workspace": "default", "workspace_empty": True},
                },
                "per_workspace_chunk_count": {"hlyu": 2},
                "failed_workspaces": ["finance"],
            }
        )

    result = await knowledge_base_search_tool(
        retrieve=retrieve, evidence=EvidenceLedger(), trace={}
    ).execute(SearchInput(query="q"), recording_tool_runtime([]))

    assert result.text_content == (
        "Knowledge base added 0 new passages. Searched hlyu (2 passages), "
        "default (no published documents), finance (search failed)."
    )


def test_one_searched_workspace_is_named_with_what_it_returned() -> None:
    assert _searched_workspaces({"workspaces": ["hlyu"]}, [{}, {}, {}]) == (
        "Searched hlyu (3 passages)."
    )
    assert _searched_workspaces(
        {"workspaces": ["default"], "workspace": "default", "workspace_empty": True}, []
    ) == ("Searched default (no published documents).")
    assert _searched_workspaces({}, [{}]) == ""


async def test_both_search_tools_report_the_query_as_their_subject_live() -> None:
    updates: list[ToolResult] = []
    query = "quarterly revenue 2026"

    async def retrieve(_query: str) -> RetrievalResult:
        return RetrievalResult()

    async def search(_request: WebSearchRequest) -> WebSearchResult:
        return WebSearchResult(())

    await knowledge_base_search_tool(
        retrieve=retrieve,
        evidence=EvidenceLedger(),
        trace={},
    ).execute(SearchInput(query=query), recording_tool_runtime(updates))
    await web_search_tool(
        search=search,
        evidence=EvidenceLedger(),
        trace={"web_search_cost_dollars": 0.0},
        register_web_source=None,
    ).execute(WebSearchInput(query=query), recording_tool_runtime(updates))

    assert [update.subject for update in updates if update.subject] == [
        query,
        query,
    ]


async def test_a_web_search_states_its_provider_cost_and_sources_as_facts() -> None:
    """The step card needs the search's own facts, not a second copy of its passages."""
    hits = (
        WebSearchHit(
            url="https://example.org/taylor",
            title="The Taylor rule",
            text="a is usually about 1.5",
            published_date="2026-01-02",
        ),
        WebSearchHit(url="https://other.example/b", title="", text="b is usually about 0.5"),
    )

    async def search(_request: WebSearchRequest) -> WebSearchResult:
        return WebSearchResult(hits=hits, cost_dollars=0.0123, provider="exa")

    result = await web_search_tool(
        search=search,
        evidence=EvidenceLedger(),
        trace={"web_search_cost_dollars": 0.0},
        register_web_source=None,
    ).execute(WebSearchInput(query="taylor rule"), tool_runtime())

    assert result.facts == {
        "provider": "exa",
        "hits": 2,
        "degraded": False,
        "cost": 0.0123,
        # Where a hit came from, not the page's words: a title reaches a reader through the
        # passage the result carries and the reader's content gate, not as a fact.
        "sources": [
            {
                "url": "https://example.org/taylor",
                "domain": "example.org",
                "published_at": "2026-01-02",
            },
            {"url": "https://other.example/b", "domain": "other.example"},
        ],
    }


async def test_an_unavailable_web_search_states_its_failure_as_facts() -> None:
    """A failed search is a whole fact row: no provider, no hits, degraded, no cost."""

    async def search(_request: WebSearchRequest) -> WebSearchResult:
        raise WebSourceUnavailable("exa", "search", "not configured")

    result = await web_search_tool(
        search=search,
        evidence=EvidenceLedger(),
        trace={"web_search_cost_dollars": 0.0},
        register_web_source=None,
    ).execute(WebSearchInput(query="q"), tool_runtime())

    assert result.is_error is True
    assert result.facts == {"provider": "", "hits": 0, "degraded": True, "cost": 0.0}


async def test_a_knowledge_base_search_states_what_it_added_as_facts() -> None:
    async def retrieve(_query: str) -> RetrievalResult:
        return RetrievalResult()

    result = await knowledge_base_search_tool(
        retrieve=retrieve,
        evidence=EvidenceLedger(),
        trace={},
    ).execute(SearchInput(query="quarterly revenue"), tool_runtime())

    assert result.facts == {"new_chunks": 0, "dropped_rows": 0}


async def test_searches_running_beside_each_other_admit_evidence_in_source_order() -> None:
    """Citation numbers follow the batch, not which search happened to return first."""
    ledger = EvidenceLedger()
    first_may_return = asyncio.Event()
    second_retrieved = asyncio.Event()

    async def retrieve(query: str) -> RetrievalResult:
        if query == "first":
            await first_may_return.wait()
        else:
            second_retrieved.set()
        row = {
            "chunk_id": f"{query}-chunk",
            "reference_id": f"{query}-doc",
            "file_path": f"{query}.pdf",
            "content": f"{query} passage",
            "metadata": {"source_uri": f"file:///{query}.pdf"},
        }
        return RetrievalResult(contexts={"chunks": [row], "entities": [], "relationships": []})

    tool = knowledge_base_search_tool(retrieve=retrieve, evidence=ledger, trace={})
    first_returned = asyncio.Event()

    async def first() -> ToolResult:
        try:
            return await tool.execute(SearchInput(query="first"), tool_runtime())
        finally:
            first_returned.set()

    async def after_first() -> None:
        await first_returned.wait()

    second_runtime = replace(tool_runtime(), _in_source_order=after_first)
    searches = asyncio.gather(first(), tool.execute(SearchInput(query="second"), second_runtime))
    await second_retrieved.wait()
    # The second search has its passage but admits nothing before the first returns.
    assert ledger.row_count == 0

    first_may_return.set()
    await searches

    assert [(row["content"], row["reference_id"]) for row in ledger.contexts["chunks"]] == [
        ("first passage", "1"),
        ("second passage", "2"),
    ]


async def test_exa_maps_all_search_controls_and_passages(serve) -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"results": [{**_PAGE, "text": "Full body."}], "costDollars": {"total": 0.007}},
        )

    serve("exa", handler)
    provider = ExaWebSource("k")
    result = await provider.search(
        WebSearchRequest(
            "coefficients",
            max_results=7,
            include_domains=("example.org",),
            exclude_domains=("bad.example",),
            start_date="2026-01-01",
            end_date="2026-02-01",
            effort="balanced",
        )
    )

    assert requests == [
        {
            "query": "coefficients",
            "type": "auto",
            "numResults": 7,
            "contents": {"highlights": {"maxCharacters": 4000}},
            "includeDomains": ["example.org"],
            "excludeDomains": ["bad.example"],
            "startPublishedDate": "2026-01-01",
            "endPublishedDate": "2026-02-01",
        }
    ]
    assert [hit.text for hit in result.hits] == [*_PAGE["highlights"], "Full body."]
    assert result.hits[0].acquisition == "exa_search"
    assert result.cost_dollars == 0.007


@pytest.mark.parametrize(("effort", "expected"), [("fast", "fast"), ("balanced", "auto")])
async def test_exa_search_never_asks_for_a_research_tier(serve, effort, expected) -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"results": []})

    serve("exa", handler)

    await ExaWebSource("k").search(WebSearchRequest("q", effort=effort))

    assert requests[0]["type"] == expected
    assert not requests[0]["type"].startswith("deep")


async def test_exa_extract_and_malformed_partial_results(serve) -> None:
    serve(
        "exa",
        _responds(
            {
                "results": [
                    {**_PAGE, "text": "  Extracted body.\n"},
                    {"url": "https://other.example/page", "text": "other body"},
                    {"title": "missing locator"},
                    {"url": "http://127.0.0.1/admin", "text": "private"},
                ]
            }
        ),
    )
    provider = ExaWebSource("k")

    result = await provider.extract("https://example.org/start")

    assert result.url == _PAGE["url"]
    assert result.text == "  Extracted body.\n"
    assert result.acquisition == "exa_extract"
    assert result.dropped_results == 3


async def test_exa_extract_asks_for_the_whole_page(serve) -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"results": [{**_PAGE, "text": "Extracted body."}]})

    serve("exa", handler)

    await ExaWebSource("k").extract("https://example.org/start")

    # Read windows a page by the reader's own limits; a provider-side cap would cut it
    # silently, with no cursor to continue from.
    assert requests[0]["text"] is True
    assert requests[0]["livecrawl"] == "fallback"


async def test_exa_missing_results_is_provider_failure_not_empty_success(serve) -> None:
    serve("exa", _responds({"unexpected": []}))
    provider = ExaWebSource("k")

    with pytest.raises(WebSourceUnavailable) as failure:
        await provider.search(WebSearchRequest("q"))

    assert failure.value.reason == "invalid_response"


async def test_exa_auth_failure_is_not_parked(serve) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={})

    serve("exa", handler)
    provider = ExaWebSource("k")
    for _ in range(2):
        with pytest.raises(WebSourceUnavailable) as failure:
            await provider.search(WebSearchRequest("q"))
        assert failure.value.reason == "unauthorized"
    assert calls == 2


async def test_tavily_maps_effort_filters_and_drops_only_bad_items(serve) -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "results": [
                    {"url": "https://a.example/x", "title": "A", "content": "body"},
                    {"url": "https://bad.example"},
                    {"url": "https://bad.example/?token=secret", "content": "credential"},
                ]
            },
        )

    serve("tavily", handler)
    provider = TavilyWebSource("tk")
    result = await provider.search(
        WebSearchRequest(
            "q",
            max_results=5,
            include_domains=("a.example",),
            exclude_domains=("b.example",),
            effort="balanced",
        )
    )

    assert requests[0]["api_key"] == "tk"
    assert requests[0]["search_depth"] == "advanced"
    assert "chunks_per_source" not in requests[0]
    assert "auto_parameters" not in requests[0]
    assert requests[0]["include_answer"] is False
    assert result.provider == "tavily"
    assert result.dropped_results == 2
    assert result.hits[0].acquisition == "tavily_search"


async def test_tavily_extract_keeps_one_exact_representation(serve) -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "results": [
                    {"url": "https://a.example/page", "raw_content": "  exact body\n"},
                    {"url": "https://b.example/page", "raw_content": "wrong body"},
                ]
            },
        )

    serve("tavily", handler)
    provider = TavilyWebSource("tk")

    result = await provider.extract("https://a.example/start")

    assert requests[0]["extract_depth"] == "advanced"

    assert result.url == "https://a.example/page"
    assert result.text == "  exact body\n"
    assert result.dropped_results == 1


_DEEPSEEK = "dlightrag.engine.answer.web_sources.deepseek"
_DEEPSEEK_SEARCH = "https://api.deepseek.com/v1/tools/search"
_DEEPSEEK_FETCH = "https://api.deepseek.com/v1/tools/fetch"
_DEEPSEEK_HIT = {
    "rerank_score": 0.93,
    "title": "The Taylor rule",
    "url": "https://example.org/taylor",
    "snippet": "A short snippet of the Taylor rule.",
    "summary": "A long, query-relevant summary of the Taylor rule and its coefficients.",
    "content": None,
    "site_name": "Example",
    "site_icon": None,
    "date": "2026-01-02T00:00:00Z",
    "authority": 0.5,
}


@pytest.fixture
def no_throttle_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer a 429 with no wait, so a retry costs the test nothing."""
    monkeypatch.setattr(f"{_DEEPSEEK}._THROTTLE_WAIT_SECONDS", 0.0)
    monkeypatch.setattr(f"{_DEEPSEEK}._THROTTLE_JITTER_SECONDS", 0.0)


def _deepseek_requests(serve, response: dict | None = None) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=response if response is not None else {"results": []})

    serve("deepseek", handler)
    return seen


async def test_deepseek_maps_search_controls_and_results(serve) -> None:
    seen = _deepseek_requests(serve, {"results": [_DEEPSEEK_HIT]})

    result = await DeepSeekWebSource("dk").search(
        WebSearchRequest(
            "coefficients",
            max_results=7,
            include_domains=("example.org",),
            exclude_domains=("bad.example",),
            start_date="2026-01-01",
            end_date="2026-02-01",
        )
    )

    assert str(seen[0].url) == _DEEPSEEK_SEARCH
    assert seen[0].headers["authorization"] == "Bearer dk"
    assert json.loads(seen[0].content) == {
        "query": "coefficients site:example.org -site:bad.example after:2026-01-01 before:2026-02-01",
        "count": 7,
        "need_content": False,
    }
    assert result.hits == (
        WebSearchHit(
            url="https://example.org/taylor",
            title="The Taylor rule",
            text=_DEEPSEEK_HIT["summary"],
            published_date="2026-01-02T00:00:00Z",
            image_url=None,
            acquisition="deepseek_search",
        ),
    )
    assert (result.provider, result.cost_dollars, result.dropped_results) == ("deepseek", 0.0, 0)


async def test_deepseek_has_no_depth_so_effort_changes_nothing(serve) -> None:
    seen = _deepseek_requests(serve)
    provider = DeepSeekWebSource("dk")

    await provider.search(WebSearchRequest("q", effort="fast"))
    await provider.search(WebSearchRequest("q", effort="balanced"))

    assert seen[0].content == seen[1].content
    assert json.loads(seen[0].content) == {"query": "q", "count": 10, "need_content": False}


async def test_deepseek_hit_text_is_the_summary_then_the_snippet_and_the_title_the_url(
    serve,
) -> None:
    serve(
        "deepseek",
        _responds(
            {
                "results": [
                    {"url": "https://a.example/1", "summary": "  summary  ", "snippet": "snippet"},
                    {"url": "https://a.example/2", "summary": " ", "snippet": " snippet only "},
                    {"url": "https://a.example/3", "snippet": "no summary key", "date": None},
                ]
            }
        ),
    )

    result = await DeepSeekWebSource("dk").search(WebSearchRequest("q"))

    assert [hit.text for hit in result.hits] == ["summary", "snippet only", "no summary key"]
    assert [hit.title for hit in result.hits] == [hit.url for hit in result.hits]
    assert [hit.published_date for hit in result.hits] == [None, None, None]
    assert result.dropped_results == 0


async def test_deepseek_search_drops_only_unusable_items_and_counts_them(serve, caplog) -> None:
    serve(
        "deepseek",
        _responds(
            {
                "results": [
                    _DEEPSEEK_HIT,
                    "not an object",
                    {"title": "no locator", "summary": "text"},
                    {"url": "https://a.example/empty", "summary": " ", "snippet": None},
                    {"url": "http://127.0.0.1/admin", "summary": "private"},
                    {"url": "https://bad.example/?token=secret", "summary": "credential"},
                ]
            }
        ),
    )

    with caplog.at_level(logging.WARNING, logger=_DEEPSEEK):
        result = await DeepSeekWebSource("dk").search(WebSearchRequest("q"))

    assert [hit.url for hit in result.hits] == [_DEEPSEEK_HIT["url"]]
    assert result.dropped_results == 5
    assert "DeepSeek search dropped 5 malformed or unusable result(s)" in caplog.text


_DOMAIN_RESULTS = [
    "https://example.org/a",
    "https://sub.example.org/b",
    "https://notexample.org/c",
    "https://example.org.attacker.net/d",
    "https://other.net/e",
    "https://Docs.Example.ORG/f",
]


@pytest.mark.parametrize("operators_fit", [True, False], ids=["in-query", "left-out-of-query"])
@pytest.mark.parametrize(
    ("include", "exclude", "expected"),
    [
        pytest.param(("example.org",), (), ["a", "b", "f"], id="include"),
        pytest.param((), ("example.org",), ["c", "d", "e"], id="exclude"),
        pytest.param(("example.org",), ("sub.example.org",), ["a", "f"], id="both"),
        pytest.param(("EXAMPLE.org.",), (), ["a", "b", "f"], id="case-and-dot"),
    ],
)
async def test_deepseek_holds_every_result_to_the_domains_either_way(
    serve, operators_fit, include, exclude, expected
) -> None:
    # Filler domains, added to each group in use, push the operators past what the query holds.
    filler = () if operators_fit else tuple(f"filler-domain-{i}.example" for i in range(8))
    seen = _deepseek_requests(
        serve,
        {"results": [{"url": url, "summary": f"text {url}"} for url in _DOMAIN_RESULTS]},
    )

    result = await DeepSeekWebSource("dk").search(
        WebSearchRequest(
            "q",
            include_domains=(*include, *filler) if include else (),
            exclude_domains=(*exclude, *filler) if exclude else (),
        )
    )

    assert ("site:" in json.loads(seen[0].content)["query"]) is (
        operators_fit and bool(include or exclude)
    )
    assert [hit.url.rsplit("/", 1)[1] for hit in result.hits] == expected
    # A result off the requested domains is filtered, not malformed.
    assert result.dropped_results == 0


def _query(text: str, **kwargs) -> str:
    return _search_query(WebSearchRequest(text, **kwargs))


def test_deepseek_query_carries_the_dates_and_the_domains_that_fit() -> None:
    query = _query(
        "coefficients",
        include_domains=("a.example", "b.example"),
        exclude_domains=("x.example", "y.example"),
        start_date="2026-01-01",
        end_date="2026-02-01",
    )

    assert query == (
        "coefficients site:a.example OR site:b.example -site:x.example -site:y.example "
        "after:2026-01-01 before:2026-02-01"
    )
    assert _query("q", start_date="2026-01-01") == "q after:2026-01-01"
    assert _query("q", end_date="2026-02-01") == "q before:2026-02-01"
    assert _query("  q  ") == "q"


def test_deepseek_query_keeps_the_dates_and_leaves_out_domains_that_squeeze_the_text() -> None:
    text = "t" * 50
    dates = {"start_date": "2026-01-01", "end_date": "2026-02-01"}
    wide = tuple(f"domain-{i}.example.org" for i in range(5))
    long_exclude = tuple(f"-long-domain-name-{i}.example.org" for i in range(3))

    assert (
        _query(text, include_domains=wide, **dates) == f"{text} after:2026-01-01 before:2026-02-01"
    )
    # One group may fit where the next does not, and the one that fits is kept.
    kept = _query(text, include_domains=("a.example",), exclude_domains=long_exclude, **dates)
    assert kept == f"{text} site:a.example after:2026-01-01 before:2026-02-01"


def test_deepseek_query_leaves_the_text_at_least_32_characters_for_a_domain_group() -> None:
    fits = "a" * 86 + ".com"  # "site:" + 90 characters: 95, which leaves exactly 32 and a space
    squeezes = "a" * 87 + ".com"
    text = "t" * 50

    assert _query(text, include_domains=(fits,)) == f"{'t' * 32} site:{fits}"
    assert _query(text, include_domains=(squeezes,)) == text
    # A text shorter than that floor needs only its own length.
    assert _query("NVDA", include_domains=(squeezes,)) == f"NVDA site:{squeezes}"


def test_deepseek_query_cuts_the_text_at_a_word_when_that_keeps_half_the_budget() -> None:
    words = " ".join(["word"] * 40)

    assert _query(words) == " ".join(["word"] * 25)
    assert _query("w" * 127 + " tail") == "w" * 127
    assert _query("w" * 128 + " tail") == "w" * 128
    # The only space is too early to keep half, so the cut falls where the budget ends.
    assert _query("ab " + "c" * 200) == ("ab " + "c" * 200)[:128]
    dated = _query(words, start_date="2026-01-01", end_date="2026-02-01")
    assert dated.endswith(" after:2026-01-01 before:2026-02-01")
    assert len(dated) <= 128


def test_deepseek_query_hard_cuts_text_without_spaces_by_code_points() -> None:
    cjk = "检索" * 100

    assert _query(cjk) == cjk[:128]
    dated = _query(cjk, start_date="2026-01-01", end_date="2026-02-01")
    assert dated == f"{cjk[:93]} after:2026-01-01 before:2026-02-01"
    assert len(dated) == 128


# One character, the 32-character floor kept beside an operator, either side of the 128 budget,
# and well past it.
@pytest.mark.parametrize("length", [1, 32, 128, 129, 200])
@pytest.mark.parametrize(
    "operators",
    [
        {},
        {"start_date": "2026-01-01", "end_date": "2026-02-01"},
        {"include_domains": ("a.example", "b.example"), "start_date": "2026-01-01"},
        {"exclude_domains": tuple(f"x{i}.example.org" for i in range(6))},
        {
            "include_domains": tuple(f"inc{i}.example.org" for i in range(3)),
            "exclude_domains": tuple(f"exc{i}.example.org" for i in range(3)),
            "start_date": "2026-01-01",
            "end_date": "2026-02-01",
        },
    ],
)
def test_deepseek_query_never_exceeds_the_budget_or_squeezes_the_text(length, operators) -> None:
    text = "t" * length

    query = _query(text, **operators)

    assert 0 < len(query) <= 128
    head = re.split(r" (?=-?site:|after:|before:)", query, maxsplit=1)[0]
    assert text.startswith(head)
    if "site:" in query:
        assert len(head) >= min(length, 32)
    for date in ("after:", "before:"):
        assert (date in query) is (
            operators.get("start_date" if date == "after:" else "end_date") is not None
        )


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (401, "unauthorized"),
        (402, "payment_required"),
        (403, "error"),
        (404, "error"),
        (408, "error"),
        (413, "error"),
        (500, "error"),
        (504, "error"),
    ],
)
async def test_deepseek_search_failures_say_why(serve, status, reason) -> None:
    serve("deepseek", _responds({"error": {"message": "no"}}, status))

    with pytest.raises(WebSourceUnavailable) as failure:
        await DeepSeekWebSource("dk").search(WebSearchRequest("q"))

    assert (failure.value.provider, failure.value.operation) == ("deepseek", "search")
    assert (failure.value.reason, failure.value.status) == (reason, status)


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (401, "unauthorized"),
        (402, "payment_required"),
        (403, "refused"),
        (404, "empty"),
        (408, "error"),
        (500, "error"),
        (504, "error"),
    ],
)
async def test_deepseek_fetch_failures_say_why(serve, status, reason) -> None:
    serve("deepseek", _responds({"error": {"message": "no"}}, status))

    with pytest.raises(WebSourceUnavailable) as failure:
        await DeepSeekWebSource("dk").extract("https://example.org/page")

    assert (failure.value.provider, failure.value.operation) == ("deepseek", "extract")
    assert (failure.value.reason, failure.value.status) == (reason, status)


async def test_deepseek_asks_again_once_after_a_429_and_returns_the_second_answer(
    serve, no_throttle_wait
) -> None:
    answers = [429, 200]
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        status = answers.pop(0)
        return httpx.Response(status, json={"results": [_DEEPSEEK_HIT]} if status == 200 else {})

    serve("deepseek", handler)

    result = await DeepSeekWebSource("dk").search(WebSearchRequest("q"))

    assert calls == 2
    assert [hit.url for hit in result.hits] == [_DEEPSEEK_HIT["url"]]


@pytest.mark.parametrize("operation", ["search", "extract"])
async def test_deepseek_gives_up_on_a_second_429_as_throttled(
    serve, no_throttle_wait, operation
) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    serve("deepseek", handler)
    provider = DeepSeekWebSource("dk")

    with pytest.raises(WebSourceUnavailable) as failure:
        if operation == "search":
            await provider.search(WebSearchRequest("q"))
        else:
            await provider.extract("https://example.org/page")

    assert calls == 2, "one retry, not a loop"
    assert (failure.value.reason, failure.value.status) == ("throttled", 429)


async def test_deepseek_waits_before_the_retry(serve, monkeypatch) -> None:
    waits: list[float] = []

    async def record(delay: float) -> None:
        waits.append(delay)

    monkeypatch.setattr(f"{_DEEPSEEK}.asyncio.sleep", record)
    answers = [429, 200]
    serve("deepseek", lambda _request: httpx.Response(answers.pop(0), json={"results": []}))

    await DeepSeekWebSource("dk").search(WebSearchRequest("q"))

    assert len(waits) == 1
    assert 1.0 <= waits[0] <= 1.25


@pytest.mark.parametrize("operation", ["search", "extract"])
async def test_deepseek_retry_stays_inside_the_operations_timeout(
    serve, monkeypatch, operation
) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, json={})

    serve("deepseek", handler)
    monkeypatch.setattr(f"{_DEEPSEEK}._THROTTLE_WAIT_SECONDS", 30.0)
    monkeypatch.setattr(f"{_DEEPSEEK}._SEARCH_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(f"{_DEEPSEEK}._FETCH_TIMEOUT_SECONDS", 0.05)
    provider = DeepSeekWebSource("dk")

    with pytest.raises(WebSourceUnavailable) as failure:
        if operation == "search":
            await provider.search(WebSearchRequest("q"))
        else:
            await provider.extract("https://example.org/page")

    assert failure.value.reason == "timeout"
    assert calls == 1, "the wait outlasted the timeout, so the retry never went out"


async def test_deepseek_search_and_fetch_have_their_own_timeouts(serve, monkeypatch) -> None:
    delays: list[float] = []
    real_timeout = asyncio.timeout

    def recording(delay: float | None):
        delays.append(delay)  # type: ignore[arg-type]
        return real_timeout(delay)

    monkeypatch.setattr(asyncio, "timeout", recording)
    serve(
        "deepseek",
        lambda request: httpx.Response(
            200, json={"results": []} if request.url.path.endswith("search") else {"content": "x"}
        ),
    )
    provider = DeepSeekWebSource("dk")

    await provider.search(WebSearchRequest("q"))
    await provider.extract("https://example.org/page")

    assert delays == [20.0, 30.0]


async def test_deepseek_fetch_is_extract_under_the_url_that_was_asked_for(serve) -> None:
    seen = _deepseek_requests(serve, {"title": " Page ", "content": "# Heading\n\nBody.\n"})

    result = await DeepSeekWebSource("dk").extract("https://example.org/start")

    assert str(seen[0].url) == _DEEPSEEK_FETCH
    assert seen[0].headers["authorization"] == "Bearer dk"
    assert json.loads(seen[0].content) == {"url": "https://example.org/start"}
    assert result == WebExtractResult(
        url="https://example.org/start",
        text="# Heading\n\nBody.\n",
        title="Page",
        provider="deepseek",
        acquisition="deepseek_extract",
    )


async def test_deepseek_fetch_without_a_title_has_none(serve) -> None:
    _deepseek_requests(serve, {"content": "Body."})

    assert (await DeepSeekWebSource("dk").extract("https://example.org/page")).title is None


async def test_deepseek_fetch_with_blank_content_is_empty(serve) -> None:
    serve("deepseek", _responds({"title": "Page", "content": "  \n"}))

    with pytest.raises(WebSourceUnavailable) as failure:
        await DeepSeekWebSource("dk").extract("https://example.org/page")

    assert failure.value.reason == "empty"


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param({"title": "Page"}, id="no-content"),
        pytest.param({"content": ["Body."]}, id="content-not-text"),
        pytest.param([], id="not-an-object"),
    ],
)
async def test_deepseek_fetch_with_a_malformed_answer_is_a_provider_failure(serve, answer) -> None:
    serve("deepseek", _responds(answer))

    with pytest.raises(WebSourceUnavailable) as failure:
        await DeepSeekWebSource("dk").extract("https://example.org/page")

    assert failure.value.reason == "invalid_response"


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param({"unexpected": []}, id="no-results"),
        pytest.param({"results": {"0": {}}}, id="results-not-a-list"),
        pytest.param([], id="not-an-object"),
    ],
)
async def test_deepseek_search_with_a_malformed_answer_is_not_an_empty_success(
    serve, answer
) -> None:
    serve("deepseek", _responds(answer))

    with pytest.raises(WebSourceUnavailable) as failure:
        await DeepSeekWebSource("dk").search(WebSearchRequest("q"))

    assert failure.value.reason == "invalid_response"


async def test_deepseek_answer_that_is_not_json_is_a_provider_failure(serve) -> None:
    serve("deepseek", lambda _request: httpx.Response(200, content=b"<html>gateway</html>"))

    with pytest.raises(WebSourceUnavailable) as failure:
        await DeepSeekWebSource("dk").search(WebSearchRequest("q"))

    assert failure.value.reason == "invalid_response"


async def test_deepseek_unreachable_host_is_a_provider_failure(serve) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    serve("deepseek", handler)

    with pytest.raises(WebSourceUnavailable) as failure:
        await DeepSeekWebSource("dk").search(WebSearchRequest("q"))

    assert failure.value.reason == "unreachable"


async def test_deepseek_fetch_refuses_a_url_past_its_limit_without_asking(serve) -> None:
    seen = _deepseek_requests(serve, {"content": "Body."})
    provider = DeepSeekWebSource("dk")
    at_limit = "https://example.org/" + "a" * (8192 - len("https://example.org/"))

    with pytest.raises(WebSourceUnavailable) as failure:
        await provider.extract(at_limit + "a")

    assert failure.value.reason == "unsupported_url"
    assert seen == []
    assert (await provider.extract(at_limit)).url == at_limit
    assert len(seen) == 1


async def test_deepseek_response_bytes_are_bounded(monkeypatch: pytest.MonkeyPatch, serve) -> None:
    monkeypatch.setattr(f"{_DEEPSEEK}._MAX_RESPONSE_BYTES", 8)
    serve("deepseek", _responds({"results": []}))

    with pytest.raises(WebSourceUnavailable) as failure:
        await DeepSeekWebSource("dk").search(WebSearchRequest("q"))

    assert failure.value.reason == "response_too_large"


async def test_provider_response_bytes_are_bounded(monkeypatch: pytest.MonkeyPatch, serve) -> None:
    monkeypatch.setattr("dlightrag.engine.answer.web_sources.exa._MAX_RESPONSE_BYTES", 8)
    serve("exa", _responds({"results": []}))
    provider = ExaWebSource("k")

    with pytest.raises(WebSourceUnavailable) as failure:
        await provider.search(WebSearchRequest("q"))

    assert failure.value.reason == "response_too_large"


class _StubProvider:
    def __init__(
        self,
        name: str,
        *,
        search: WebSearchResult | Exception,
        extract: WebExtractResult | Exception,
    ) -> None:
        self.name = name
        self._search = search
        self._extract = extract
        self.search_calls = 0
        self.extract_calls = 0

    async def search(self, request: WebSearchRequest) -> WebSearchResult:
        self.search_calls += 1
        if isinstance(self._search, Exception):
            raise self._search
        return self._search

    async def extract(self, url: str) -> WebExtractResult:
        self.extract_calls += 1
        if isinstance(self._extract, Exception):
            raise self._extract
        return self._extract

    def answer_search(self, result: WebSearchResult | Exception) -> None:
        """Change what this provider answers with, as one that came back would."""
        self._search = result

    async def aclose(self) -> None:
        return None


async def test_search_and_extract_fail_over_in_independent_orders() -> None:
    exa = _StubProvider(
        "exa",
        search=WebSourceUnavailable("exa", "search", "timeout"),
        extract=WebExtractResult(
            "https://a.example/final", "exa text", provider="exa", acquisition="exa_extract"
        ),
    )
    tavily = _StubProvider(
        "tavily",
        search=WebSearchResult(
            (WebSearchHit("https://a.example", "A", "fact", acquisition="tavily_search"),),
            provider="tavily",
        ),
        extract=WebSourceUnavailable("tavily", "extract", "timeout"),
    )
    service = WebSourceService(
        search_providers=(exa, tavily),
        extract_providers=(tavily, exa),
    )

    searched = await service.search(WebSearchRequest("q"))
    extracted = await service.extract("https://a.example", providers=("exa", "tavily"))

    assert searched.provider == "tavily"
    assert searched.degradation == "Provider fallback: exa (timeout); used tavily."
    assert extracted.provider == "exa"
    assert extracted.degradation == "Provider fallback: tavily (timeout); used exa."
    assert (exa.search_calls, tavily.search_calls) == (1, 1)
    assert (tavily.extract_calls, exa.extract_calls) == (1, 1)


async def test_a_provider_that_says_it_is_out_is_left_alone_for_a_minute() -> None:
    """Exa out of credit must not be asked on every call of a burst: the chain says it is cooling."""
    now = [100.0]
    exa = _StubProvider(
        "exa",
        search=WebSourceUnavailable("exa", "search", "payment_required", status=402),
        extract=WebExtractResult("https://a.example", "text", provider="exa"),
    )
    tavily = _StubProvider(
        "tavily",
        search=WebSearchResult(
            (WebSearchHit("https://a.example", "A", "fact", acquisition="tavily_search"),),
            provider="tavily",
        ),
        extract=WebExtractResult("https://a.example", "text", provider="tavily"),
    )
    service = WebSourceService(
        search_providers=(exa, tavily),
        extract_providers=(exa, tavily),
        clock=lambda: now[0],
    )

    first = await service.search(WebSearchRequest("q"))
    second = await service.search(WebSearchRequest("q"))

    assert (first.provider, second.provider) == ("tavily", "tavily")
    assert first.degradation == "Provider fallback: exa (payment_required); used tavily."
    assert second.degradation == "Provider fallback: exa (cooling); used tavily."
    assert exa.search_calls == 1, "the dead provider was asked once, not once per call"
    # Extraction is a different leg: Exa's search credit says nothing about its contents call.
    await service.extract("https://a.example", providers=("exa", "tavily"))
    assert exa.extract_calls == 1


async def test_a_cooling_provider_is_asked_again_once_the_window_passes(caplog) -> None:
    now = [100.0]
    exa = _StubProvider(
        "exa",
        search=WebSourceUnavailable("exa", "search", "rate_limited", status=429),
        extract=WebExtractResult("https://a.example", "text", provider="exa"),
    )
    tavily = _StubProvider(
        "tavily",
        search=WebSearchResult((), provider="tavily"),
        extract=WebExtractResult("https://a.example", "text", provider="tavily"),
    )
    service = WebSourceService(search_providers=(exa, tavily), clock=lambda: now[0])

    with caplog.at_level("WARNING"):
        await service.search(WebSearchRequest("q"))
    now[0] += 61.0
    exa.answer_search(
        WebSearchResult(
            (WebSearchHit("https://a.example", "A", "fact", acquisition="exa_search"),),
            provider="exa",
        )
    )
    recovered = await service.search(WebSearchRequest("q"))

    assert recovered.provider == "exa"
    assert recovered.degradation is None, "a recovered provider leaves no fallback behind"
    assert exa.search_calls == 2
    # The failed attempt is on the operator's log once, with what a dashboard groups by.
    record = next(
        item for item in caplog.records if item.message == "Web search provider unavailable"
    )
    assert (record.provider, record.operation, record.reason, record.status) == (
        "exa",
        "search",
        "rate_limited",
        429,
    )


async def test_when_every_provider_is_cooling_one_probe_goes_out() -> None:
    """A provider that came back must be noticed without a restart: one probe, not the whole chain.

    Both windows end at the same moment here, and the chain's own order settles the tie — the
    primary is the one asked, which is also the one most likely to be the deployment's account.
    """
    now = [100.0]
    exa = _StubProvider(
        "exa",
        search=WebSourceUnavailable("exa", "search", "payment_required", status=402),
        extract=WebExtractResult("https://a.example", "text", provider="exa"),
    )
    tavily = _StubProvider(
        "tavily",
        search=WebSourceUnavailable("tavily", "search", "rate_limited", status=429),
        extract=WebExtractResult("https://a.example", "text", provider="tavily"),
    )
    service = WebSourceService(search_providers=(exa, tavily), clock=lambda: now[0])

    with pytest.raises(WebSourceUnavailable) as refused:
        await service.search(WebSearchRequest("q"))
    assert "all search unavailable: exa:payment_required,tavily:rate_limited" in str(refused.value)

    now[0] += 5.0
    exa.answer_search(WebSearchResult((), provider="exa"))
    probed = await service.search(WebSearchRequest("q"))

    assert probed.provider == "exa"
    assert (exa.search_calls, tavily.search_calls) == (2, 1), "one probe, not the chain again"


async def test_a_failure_about_this_call_does_not_cool_the_provider() -> None:
    """A timeout or one malformed body is about the call; the next call may well answer."""
    now = [100.0]
    exa = _StubProvider(
        "exa",
        search=WebSourceUnavailable("exa", "search", "timeout"),
        extract=WebExtractResult("https://a.example", "text", provider="exa"),
    )
    tavily = _StubProvider(
        "tavily",
        search=WebSearchResult((), provider="tavily"),
        extract=WebExtractResult("https://a.example", "text", provider="tavily"),
    )
    service = WebSourceService(search_providers=(exa, tavily), clock=lambda: now[0])

    await service.search(WebSearchRequest("q"))
    await service.search(WebSearchRequest("q"))

    assert exa.search_calls == 2


async def test_the_search_tool_reads_an_outage_as_one_failed_call_with_somewhere_to_go() -> None:
    """The model is told what else this Run can reach, and the step still counts as failed."""

    async def refused(_request: WebSearchRequest) -> WebSearchResult:
        raise WebSourceUnavailable("all", "search", "exa:payment_required,tavily:payment_required")

    async def call(*, browser_available: bool) -> ToolResult:
        tool = web_search_tool(
            search=refused,
            evidence=EvidenceLedger(),
            trace={"web_search_cost_dollars": 0.0},
            register_web_source=None,
            browser_available=browser_available,
        )
        return await tool.execute(WebSearchInput(query="q"), recording_tool_runtime([]))

    with_browser = await call(browser_available=True)
    without_browser = await call(browser_available=False)

    assert with_browser.is_error is True
    assert "Open-web search is unavailable" in with_browser.text_content
    assert "Agent Browser is available in this Run" in with_browser.text_content
    assert "Answer from the evidence you already have" in without_browser.text_content
    assert "Agent Browser" not in without_browser.text_content
    # What a dashboard groups by is on both, and an outage bills nothing.
    assert without_browser.telemetry == {
        "web_search_provider": "",
        "web_search_hits": 0,
        "web_search_degraded": True,
        "web_search_cost_dollars": 0.0,
    }


async def test_extract_asks_only_the_providers_it_is_given() -> None:
    """The Extract chain splits around the Agent Browser, so each run of hosted providers is its own call."""

    def provider(name: str) -> _StubProvider:
        return _StubProvider(
            name,
            search=WebSourceUnavailable(name, "search", "unused"),
            extract=WebExtractResult(
                "https://a.example/final", f"{name} text", provider=name, acquisition="exa_extract"
            ),
        )

    exa, tavily = provider("exa"), provider("tavily")
    service = WebSourceService(extract_providers=(exa, tavily))

    extracted = await service.extract("https://a.example", providers=("tavily",))

    assert extracted.provider == "tavily"
    assert (exa.extract_calls, tavily.extract_calls) == (0, 1)


async def test_the_chains_end_with_deepseek() -> None:
    def provider(name: str, *, works: bool) -> _StubProvider:
        failure = lambda operation: WebSourceUnavailable(name, operation, "timeout")  # noqa: E731
        return _StubProvider(
            name,
            search=WebSearchResult(
                (WebSearchHit("https://a.example", "A", "fact", acquisition="deepseek_search"),),
                provider=name,
            )
            if works
            else failure("search"),
            extract=WebExtractResult(
                "https://a.example", "text", provider=name, acquisition="deepseek_extract"
            )
            if works
            else failure("extract"),
        )

    exa, tavily, deepseek = (
        provider("exa", works=False),
        provider("tavily", works=False),
        provider("deepseek", works=True),
    )
    service = WebSourceService(
        search_providers=(exa, tavily, deepseek), extract_providers=(exa, tavily, deepseek)
    )

    searched = await service.search(WebSearchRequest("q"))
    extracted = await service.extract("https://a.example", providers=("exa", "tavily", "deepseek"))

    expected = "Provider fallback: exa (timeout), tavily (timeout); used deepseek."
    assert (searched.provider, searched.degradation) == ("deepseek", expected)
    assert (extracted.provider, extracted.degradation) == ("deepseek", expected)
    assert [(p.search_calls, p.extract_calls) for p in (exa, tavily, deepseek)] == [(1, 1)] * 3


@pytest.mark.parametrize(
    ("status", "second_attempt_reaches_deepseek", "second_degradation"),
    [
        pytest.param(402, False, "deepseek (cooling)", id="payment-required-cools"),
        pytest.param(401, False, "deepseek (cooling)", id="unauthorized-cools"),
        pytest.param(429, True, "deepseek (throttled)", id="throttled-does-not-cool"),
    ],
)
async def test_deepseek_is_left_alone_for_credit_but_not_for_a_per_second_limit(
    serve, no_throttle_wait, status, second_attempt_reaches_deepseek, second_degradation
) -> None:
    asked = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal asked
        asked += 1
        return httpx.Response(status, json={})

    serve("deepseek", handler)
    exa = _StubProvider(
        "exa",
        search=WebSearchResult((), provider="exa"),
        extract=WebExtractResult("https://a.example", "text", provider="exa"),
    )
    service = WebSourceService(search_providers=(DeepSeekWebSource("dk"), exa), clock=lambda: 100.0)
    per_attempt = 2 if status == 429 else 1  # a 429 is asked for once more before it is final

    first = await service.search(WebSearchRequest("q"))
    asked_by_first = asked
    second = await service.search(WebSearchRequest("q"))

    assert first.provider == second.provider == "exa"
    assert asked_by_first == per_attempt
    assert (asked - asked_by_first == per_attempt) is second_attempt_reaches_deepseek
    assert first.degradation is not None and second.degradation is not None
    assert second_degradation in second.degradation
    await service.aclose()


def test_the_search_tool_asks_for_a_short_query_of_key_terms() -> None:
    query = web_search_declaration().definition.parameters["properties"]["query"]

    assert "short search query of key terms" in query["description"]
    assert "under about 100 characters" in query["description"]
    assert "one angle per call" in query["description"].lower()
    assert query["minLength"] == 1
    assert "maxLength" not in query
    # The knowledge base keeps its own wording.
    knowledge = knowledge_base_search_declaration().definition.parameters["properties"]["query"]
    assert "natural language" in knowledge["description"]


def test_rewording_the_query_field_leaves_the_search_tools_contract_digest_alone() -> None:
    class _FormerWording(WebSearchInput):
        query: str = Field(
            min_length=1,
            description=(
                "One concrete unresolved question or fact to search for, in natural "
                "language. Search one angle per call rather than combining several."
            ),
        )

    current = web_search_declaration()
    former = ToolDeclaration(
        current.name,
        current.description,
        _FormerWording,
        contract_version=current.contract_version,
        read_only=current.read_only,
    )

    assert former.definition.parameters != current.definition.parameters
    assert former.input_schema_digest == current.input_schema_digest


async def test_empty_search_is_success_not_quality_based_fallback() -> None:
    first = _StubProvider(
        "exa",
        search=WebSearchResult((), provider="exa"),
        extract=WebSourceUnavailable("exa", "extract", "unused"),
    )
    second = _StubProvider(
        "tavily",
        search=WebSearchResult(
            (WebSearchHit("https://a.example", "A", "fact", acquisition="tavily_search"),),
            provider="tavily",
        ),
        extract=WebSourceUnavailable("tavily", "extract", "unused"),
    )
    result = await WebSourceService(search_providers=(first, second)).search(WebSearchRequest("q"))

    assert result.hits == ()
    assert second.search_calls == 0


def _hit(url: str, text: str, **kwargs) -> WebSearchHit:
    return WebSearchHit(url=url, title=kwargs.pop("title", "T"), text=text, **kwargs)


def test_web_context_rows_use_final_url_and_web_resource_metadata() -> None:
    rows = web_context_rows(
        [
            _hit("https://a/x#one", "one"),
            _hit("https://a/x#two", "two"),
            _hit("https://a/x", "one"),
        ]
    )

    assert len(rows) == 2
    assert len({row["reference_id"] for row in rows}) == 1
    metadata = rows[0]["metadata"]
    assert metadata["source_uri"] == "https://a/x"
    assert metadata["resource_kind"] == "web"
    assert metadata["admission_origin"] == "search"
    assert metadata["acquisition"] == "exa_search"
    assert rows[0]["_workspace"] == "__web_search__"


async def test_search_tool_reports_partial_drop_and_provider_degradation() -> None:
    async def search(_request: WebSearchRequest) -> WebSearchResult:
        return WebSearchResult(
            (WebSearchHit("https://a.example", "A", "fact"),),
            provider="tavily",
            dropped_results=2,
            degradation="Provider fallback: exa (timeout); used tavily.",
        )

    evidence = EvidenceLedger()
    result = await web_search_tool(
        search=search,
        evidence=evidence,
        trace={"web_search_cost_dollars": 0.0},
        register_web_source=lambda _url: "res-1",
    ).execute(WebSearchInput(query="q"), recording_tool_runtime([]))

    assert "Dropped 2 malformed result(s)." in result.text_content
    assert "Provider fallback" in result.text_content
    # The page's handle is printed once, on its document heading, not listed here too.
    assert "res-1" not in result.text_content
    assert evidence.contexts["chunks"][0]["metadata"]["resource_id"] == "res-1"


async def test_a_web_search_states_each_source_page_once() -> None:
    """A provider may cut several passages from one page; a step card states the page once."""
    hits = (
        WebSearchHit(url="https://example.org/report", title="Report", text="first"),
        WebSearchHit(url="https://example.org/report", title="Report", text="second"),
        WebSearchHit(url="https://example.org/other", title="Other", text="third"),
    )

    async def search(_request: WebSearchRequest) -> WebSearchResult:
        return WebSearchResult(hits=hits, cost_dollars=0.0, provider="exa")

    result = await web_search_tool(
        search=search,
        evidence=EvidenceLedger(),
        trace={"web_search_cost_dollars": 0.0},
        register_web_source=None,
    ).execute(WebSearchInput(query="report"), tool_runtime())

    assert result.facts is not None
    assert [source["url"] for source in result.facts["sources"]] == [
        "https://example.org/report",
        "https://example.org/other",
    ]
