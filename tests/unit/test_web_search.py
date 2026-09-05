# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Provider adapters, ordered failover, and Web Evidence projection."""

import json

import httpx
import pytest
from pydantic import ValidationError

from dlightrag.engine.agent.session.ids import IntentId
from dlightrag.engine.agent.tools import ToolResult, ToolRuntime
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.tools.search import (
    SearchInput,
    WebSearchInput,
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

_PAGE = {
    "url": "https://example.org/taylor",
    "title": "The Taylor rule",
    "publishedDate": "2026-01-02T00:00:00.000Z",
    "image": "https://example.org/figure-1.png",
    "highlights": ["a is usually about 1.5", "b is usually about 0.5"],
}


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _responds(payload: dict, status: int = 200):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return handler


def _label_capture_runtime(updates: list[ToolResult]) -> ToolRuntime:
    async def sink(result: ToolResult) -> None:
        updates.append(result)

    return ToolRuntime(
        call_id="test-call",
        tool_name="test-tool",
        intent_id=IntentId.new(),
        execution_scope="test-scope",
        _update_sink=sink,
    )


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


async def test_both_search_tools_report_the_query_as_object_label_live() -> None:
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
    ).execute(SearchInput(query=query), _label_capture_runtime(updates))
    await web_search_tool(
        search=search,
        evidence=EvidenceLedger(),
        trace={"web_search_cost_dollars": 0.0},
        register_web_source=None,
    ).execute(WebSearchInput(query=query), _label_capture_runtime(updates))

    assert [update.details["object_label"] for update in updates if update.details] == [
        query,
        query,
    ]


async def test_exa_maps_all_search_controls_and_passages() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"results": [{**_PAGE, "text": "Full body."}], "costDollars": {"total": 0.007}},
        )

    provider = ExaWebSource("k", client=_client(handler))
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


async def test_exa_extract_and_malformed_partial_results() -> None:
    provider = ExaWebSource(
        "k",
        client=_client(
            _responds(
                {
                    "results": [
                        {**_PAGE, "text": "  Extracted body.\n"},
                        {"url": "https://other.example/page", "text": "other body"},
                        {"title": "missing locator"},
                        {"url": "http://127.0.0.1/admin", "text": "private"},
                    ]
                }
            )
        ),
    )

    result = await provider.extract("https://example.org/start", effort="balanced")

    assert result.url == _PAGE["url"]
    assert result.text == "  Extracted body.\n"
    assert result.acquisition == "exa_extract"
    assert result.dropped_results == 3


async def test_exa_missing_results_is_provider_failure_not_empty_success() -> None:
    provider = ExaWebSource("k", client=_client(_responds({"unexpected": []})))

    with pytest.raises(WebSourceUnavailable) as failure:
        await provider.search(WebSearchRequest("q"))

    assert failure.value.reason == "invalid_response"


async def test_exa_auth_failure_is_not_parked() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={})

    provider = ExaWebSource("k", client=_client(handler))
    for _ in range(2):
        with pytest.raises(WebSourceUnavailable) as failure:
            await provider.search(WebSearchRequest("q"))
        assert failure.value.reason == "unauthorized"
    assert calls == 2


async def test_tavily_maps_effort_filters_and_drops_only_bad_items() -> None:
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

    provider = TavilyWebSource("tk", client=_client(handler))
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


async def test_tavily_extract_keeps_one_exact_representation() -> None:
    provider = TavilyWebSource(
        "tk",
        client=_client(
            _responds(
                {
                    "results": [
                        {
                            "url": "https://a.example/page",
                            "raw_content": "  exact body\n",
                        },
                        {"url": "https://b.example/page", "raw_content": "wrong body"},
                    ]
                }
            )
        ),
    )

    result = await provider.extract("https://a.example/start", effort="balanced")

    assert result.url == "https://a.example/page"
    assert result.text == "  exact body\n"
    assert result.dropped_results == 1


async def test_provider_response_bytes_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dlightrag.engine.answer.web_sources.exa._MAX_RESPONSE_BYTES", 8)
    provider = ExaWebSource("k", client=_client(_responds({"results": []})))

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

    result = await web_search_tool(
        search=search,
        evidence=EvidenceLedger(),
        trace={"web_search_cost_dollars": 0.0},
        register_web_source=lambda _url: "res-1",
    ).execute(WebSearchInput(query="q"), _label_capture_runtime([]))

    assert "Dropped 2 malformed result(s)." in result.text_content
    assert "Provider fallback" in result.text_content
    assert "[resource: res-1]" in result.text_content
