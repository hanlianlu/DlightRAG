# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Provider adapters, ordered failover, and Web Evidence projection."""

import asyncio
import inspect
import json
from dataclasses import replace

import httpx
import pytest
from pydantic import ValidationError

from dlightrag.engine.agent.tools import ToolResult
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.tools.search import (
    SearchInput,
    WebSearchInput,
    _searched_workspaces,
    knowledge_base_search_declaration,
    knowledge_base_search_tool,
    web_search_tool,
)
from dlightrag.engine.answer.tools.web_search import web_context_rows
from dlightrag.engine.answer.web_sources import (
    ExaWebSource,
    TavilyWebSource,
    WebEffort,
    WebExtractResult,
    WebSearchHit,
    WebSearchRequest,
    WebSearchResult,
    WebSourceService,
    WebSourceUnavailable,
)
from dlightrag.engine.rag.retrieval import RetrievalResult
from tests.tool_helpers import recording_tool_runtime, tool_runtime

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
            "effort": "deep",
        }
    )
    assert parsed.model_dump()["include_domains"] == ("example.org",)
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
            effort="deep",
        )
    )

    assert requests == [
        {
            "query": "coefficients",
            "type": "deep",
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

    result = await provider.extract("https://example.org/start", effort="balanced")

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

    await ExaWebSource("k").extract("https://example.org/start", effort="balanced")

    # Read windows a page by the reader's own limits; a provider-side cap would cut it
    # silently, with no cursor to continue from.
    assert requests[0]["text"] is True


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
            effort="deep",
        )
    )

    assert requests[0]["api_key"] == "tk"
    assert requests[0]["search_depth"] == "advanced"
    assert requests[0]["chunks_per_source"] == 3
    assert requests[0]["include_answer"] is False
    assert result.provider == "tavily"
    assert result.dropped_results == 2
    assert result.hits[0].acquisition == "tavily_search"


async def test_tavily_extract_keeps_one_exact_representation(serve) -> None:
    serve(
        "tavily",
        _responds(
            {
                "results": [
                    {"url": "https://a.example/page", "raw_content": "  exact body\n"},
                    {"url": "https://b.example/page", "raw_content": "wrong body"},
                ]
            }
        ),
    )
    provider = TavilyWebSource("tk")

    result = await provider.extract("https://a.example/start", effort="balanced")

    assert result.url == "https://a.example/page"
    assert result.text == "  exact body\n"
    assert result.dropped_results == 1


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

    async def extract(self, url: str, *, effort: WebEffort) -> WebExtractResult:
        self.extract_calls += 1
        if isinstance(self._extract, Exception):
            raise self._extract
        return self._extract

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
    extracted = await service.extract("https://a.example", effort="fast")

    assert searched.provider == "tavily"
    assert searched.degradation == "Provider fallback: exa (timeout); used tavily."
    assert extracted.provider == "exa"
    assert extracted.degradation == "Provider fallback: tavily (timeout); used exa."
    assert (exa.search_calls, tavily.search_calls) == (1, 1)
    assert (tavily.extract_calls, exa.extract_calls) == (1, 1)


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
