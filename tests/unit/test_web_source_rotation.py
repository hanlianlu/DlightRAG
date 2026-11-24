# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One Web source behind a rotating pool of API keys."""

import asyncio

import pytest

from dlightrag.engine.answer.web_sources.contracts import (
    WebExtractResult,
    WebSearchHit,
    WebSearchRequest,
    WebSearchResult,
    WebSourceUnavailable,
)
from dlightrag.engine.answer.web_sources.rotation import RotatingWebSource

_HIT = WebSearchHit("https://a.example", "A", "fact", acquisition="exa_search")
_RESULT = WebSearchResult((_HIT,), provider="exa")
_EXTRACTED = WebExtractResult(
    "https://a.example", "page text", provider="exa", acquisition="exa_extract"
)


class _KeyedStub:
    """One credential's adapter: answers, refuses, or records how often it was asked."""

    def __init__(
        self,
        secret: str,
        *,
        search: WebSearchResult | Exception = _RESULT,
        extract: WebExtractResult | Exception = _EXTRACTED,
    ) -> None:
        self.secret = secret
        self.search_outcome = search
        self.extract_outcome = extract
        self.search_calls = 0
        self.extract_calls = 0
        self.closed = 0

    async def search(self, request: WebSearchRequest) -> WebSearchResult:
        self.search_calls += 1
        if isinstance(self.search_outcome, Exception):
            raise self.search_outcome
        return self.search_outcome

    async def extract(self, url: str) -> WebExtractResult:
        self.extract_calls += 1
        if isinstance(self.extract_outcome, Exception):
            raise self.extract_outcome
        return self.extract_outcome

    async def aclose(self) -> None:
        self.closed += 1


def _refused(reason: str, status: int | None = None) -> WebSourceUnavailable:
    return WebSourceUnavailable("exa", "search", reason, status=status)


def _source(
    stubs: dict[str, _KeyedStub],
    *,
    secrets: tuple[str, ...],
    clock=None,
) -> RotatingWebSource:
    return RotatingWebSource(
        name="exa",
        secrets=secrets,
        build=lambda secret: stubs[secret],
        clock=clock,
    )


async def test_the_second_key_answers_when_the_first_is_out_of_credit() -> None:
    """Rotation is inside the provider: the chain sees one source that answered."""
    first = _KeyedStub("one", search=_refused("payment_required", status=402))
    second = _KeyedStub("two")
    source = _source({"one": first, "two": second}, secrets=("one", "two"))

    result = await source.search(WebSearchRequest("q"))

    assert (result.provider, result.degradation) == ("exa", None)
    assert (first.search_calls, second.search_calls) == (1, 1)


async def test_the_pool_raises_exhausted_once_every_key_has_refused() -> None:
    now = [100.0]
    first = _KeyedStub("one", search=_refused("payment_required", status=402))
    second = _KeyedStub("two", search=_refused("unauthorized", status=401))
    source = _source({"one": first, "two": second}, secrets=("one", "two"), clock=lambda: now[0])

    with pytest.raises(WebSourceUnavailable) as refused:
        await source.search(WebSearchRequest("q"))

    assert refused.value.reason == "exhausted"
    # All keys are out, so the next call spends one probe, not the whole pool again.
    with pytest.raises(WebSourceUnavailable):
        await source.search(WebSearchRequest("q"))
    assert (first.search_calls, second.search_calls) == (2, 1)


async def test_a_pool_of_one_reads_exhausted_when_its_key_is_out() -> None:
    """One key is a pool of one: the provider is out, and the chain parks it."""
    first = _KeyedStub("only", search=_refused("payment_required", status=402))
    source = _source({"only": first}, secrets=("only",))

    with pytest.raises(WebSourceUnavailable) as refused:
        await source.search(WebSearchRequest("q"))

    assert (refused.value.provider, refused.value.reason) == ("exa", "exhausted")


async def test_a_mixed_pool_reports_the_reasons_instead_of_hiding_them() -> None:
    """A call-level failure may answer next time, so the provider must not park on it."""
    first = _KeyedStub("one", search=_refused("payment_required", status=402))
    second = _KeyedStub("two", search=_refused("timeout"))
    source = _source({"one": first, "two": second}, secrets=("one", "two"))

    with pytest.raises(WebSourceUnavailable) as refused:
        await source.search(WebSearchRequest("q"))

    assert refused.value.reason == "payment_required+timeout"


async def test_a_call_level_failure_moves_to_the_next_key() -> None:
    first = _KeyedStub("one", search=_refused("timeout"))
    second = _KeyedStub("two")
    source = _source({"one": first, "two": second}, secrets=("one", "two"))

    result = await source.search(WebSearchRequest("q"))

    assert result.provider == "exa"
    assert (first.search_calls, second.search_calls) == (1, 1)


async def test_extract_rotates_independently_of_search() -> None:
    """A provider out of search credit still serves extraction, as the chain already assumes."""
    first = _KeyedStub("one", search=_refused("payment_required", status=402))
    second = _KeyedStub("two")
    source = _source({"one": first, "two": second}, secrets=("one", "two"))

    await source.search(WebSearchRequest("q"))
    extracted = await source.extract("https://a.example")

    assert extracted.provider == "exa"
    assert (first.extract_calls, second.extract_calls) == (1, 0)


async def test_every_credential_builds_one_adapter_and_close_closes_them_all() -> None:
    stubs = {"one": _KeyedStub("one"), "two": _KeyedStub("two")}
    source = _source(stubs, secrets=("one", "two"))

    await source.aclose()
    await source.aclose()

    assert [stub.closed for stub in stubs.values()] == [1, 1]


async def test_rotation_spreads_successive_calls_across_the_pool() -> None:
    """Without a failure, two keys alternate: quota and a per-key limit spread, never drain."""
    stubs = {"one": _KeyedStub("one"), "two": _KeyedStub("two")}
    source = _source(stubs, secrets=("one", "two"))

    await source.search(WebSearchRequest("q"))
    await source.search(WebSearchRequest("q"))

    assert (stubs["one"].search_calls, stubs["two"].search_calls) == (1, 1)


async def test_a_cancelled_call_is_not_a_credential_failure() -> None:
    class _Cancelled(_KeyedStub):
        async def search(self, request: WebSearchRequest) -> WebSearchResult:
            raise asyncio.CancelledError

    cancelled = _Cancelled("one")
    second = _KeyedStub("two")
    source = _source({"one": cancelled, "two": second}, secrets=("one", "two"))

    with pytest.raises(asyncio.CancelledError):
        await source.search(WebSearchRequest("q"))

    assert second.search_calls == 0
