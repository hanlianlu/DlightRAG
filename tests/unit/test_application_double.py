# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The strict Application double refuses what the real Application would.

Every transport suite relies on these refusals to catch a call to a member a
service no longer has, or with arguments the real one rejects. Loosening the
double (dropping spec_set, or the per-method autospec) would leave those suites
green while hiding exactly that drift, so these tests pin it.
"""

from typing import Any

import pytest

from dlightrag.application.config import DlightragConfig
from dlightrag.application.corpus_admin import FilePanelCursorCodec
from tests.support.application_double import application_double, delegate


@pytest.fixture
def application(test_config: DlightragConfig) -> Any:
    return application_double(test_config)


def test_the_root_admits_only_applications_public_attributes(application: Any) -> None:
    with pytest.raises(AttributeError):
        _ = application.corpra
    with pytest.raises(AttributeError):
        application.corpra = object()
    with pytest.raises(AttributeError):
        _ = application._open


def test_a_service_admits_only_the_members_of_its_class(application: Any) -> None:
    with pytest.raises(AttributeError):
        _ = application.corpora.list_workspace
    with pytest.raises(AttributeError):
        application.corpora.list_workspace = object()


async def test_a_call_the_real_signature_rejects_raises_and_is_not_recorded(
    application: Any,
) -> None:
    get_visual_asset = application.corpora.get_visual_asset

    with pytest.raises(TypeError):
        await get_visual_asset("default", chunk="chunk-1")
    with pytest.raises(TypeError):
        application.retrieval.project_stored({}, projection=None, unexpected=True)

    assert get_visual_asset.call_count == 0
    assert get_visual_asset.await_count == 0
    await get_visual_asset("default", "chunk-1", size="thumb")
    get_visual_asset.assert_awaited_once_with("default", "chunk-1", size="thumb")


async def test_the_lifecycle_keeps_applications_signatures(application: Any) -> None:
    with pytest.raises(TypeError):
        await application.aclose(True)

    await application.aclose()

    application.aclose.assert_awaited_once_with()


def test_a_service_is_built_once_and_a_real_one_is_kept(test_config: DlightragConfig) -> None:
    health = object()
    application = application_double(test_config, health=health)

    assert application.corpora is application.corpora
    assert application.health is health
    assert application.config is test_config
    with pytest.raises(TypeError, match="no service named corpus"):
        application_double(test_config, corpus=object())


def test_a_property_returning_an_application_class_is_autospecced(application: Any) -> None:
    corpora = application.corpora

    with pytest.raises(AttributeError):
        _ = corpora.file_panel_cursor_codec.encod
    with pytest.raises(TypeError):
        corpora.file_panel_cursor_codec.encode()
    # Last, because the assertion narrows the attribute's static type.
    assert isinstance(corpora.file_panel_cursor_codec, FilePanelCursorCodec)


class _Assets:
    """A fake that accepts any arguments, so only the autospec can refuse a call."""

    def __init__(self) -> None:
        self.reads: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def get_visual_asset(self, *args: object, **kwargs: object) -> str:
        self.reads.append((args, kwargs))
        return "asset"

    async def warm(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


async def test_delegate_answers_only_calls_the_real_signature_accepts(application: Any) -> None:
    assets = _Assets()
    delegate(application.corpora, assets, "get_visual_asset")

    with pytest.raises(TypeError):
        await application.corpora.get_visual_asset("default", chunk="chunk-1")
    answer = await application.corpora.get_visual_asset("default", "chunk-1")

    assert answer == "asset"
    assert assets.reads == [(("default", "chunk-1"), {})]


def test_delegate_refuses_what_could_never_answer_a_call(application: Any) -> None:
    assets = _Assets()

    with pytest.raises(AttributeError):
        delegate(application.corpora, assets, "get_visual_assets")
    with pytest.raises(TypeError, match="is a property"):
        delegate(application.corpora, assets, "file_panel_cursor_codec")
    with pytest.raises(TypeError, match="is sync"):
        delegate(application.retrieval, assets, "warm")
