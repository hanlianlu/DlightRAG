# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the offline workspace BM25 rebuild command."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest


def _config(*, enabled: bool = True, reader: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        bm25_enabled=enabled,
        is_reader=reader,
        workspace="research",
        bm25_profiles=[
            SimpleNamespace(
                name="en",
                text_config="english",
                languages=["en"],
                fallback=False,
            ),
            SimpleNamespace(
                name="simple",
                text_config="simple",
                languages=[],
                fallback=True,
            ),
        ],
        bm25_k1=1.4,
        bm25_b=0.65,
    )


def test_bm25_rebuild_parser_defaults() -> None:
    from dlightrag.tools.rebuild_bm25 import build_parser

    args = build_parser().parse_args([])

    assert args.yes is False
    assert args.batch_size == 500


def test_bm25_rebuild_requires_yes() -> None:
    from dlightrag.tools.rebuild_bm25 import build_parser, validate_args

    args = build_parser().parse_args([])

    with pytest.raises(SystemExit, match="--yes is required"):
        validate_args(args)


def test_pyproject_exposes_bm25_rebuild_console_script() -> None:
    import tomllib

    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["dlightrag-rebuild-bm25"] == (
        "dlightrag.tools.rebuild_bm25:main"
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_config(enabled=False), "bm25_enabled=true"),
        (_config(reader=True), "writer service role"),
    ],
)
async def test_bm25_rebuild_rejects_incompatible_config(
    config: SimpleNamespace,
    message: str,
) -> None:
    from dlightrag.tools.rebuild_bm25 import run_rebuild_bm25

    with pytest.raises(SystemExit, match=message):
        await run_rebuild_bm25(config=cast(Any, config), assume_yes=True)


async def test_bm25_prerequisites_provision_extensions_and_close_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.core.retrieval.bm25 import BM25Profile
    from dlightrag.tools import rebuild_bm25 as module

    conn = AsyncMock()
    connect = AsyncMock(return_value=conn)
    ensure_major = AsyncMock()
    ensure_extensions = AsyncMock()
    monkeypatch.setattr(module.asyncpg, "connect", connect)
    monkeypatch.setattr(module, "ensure_postgres_major", ensure_major)
    monkeypatch.setattr(module, "ensure_postgres_extensions", ensure_extensions)
    config = SimpleNamespace(
        pg_connection_kwargs=MagicMock(return_value={"host": "db", "port": 5432})
    )
    profiles = (BM25Profile(name="en", text_config="english", languages=("en",)),)

    await module._ensure_bm25_prerequisites(cast(Any, config), profiles)

    connect.assert_awaited_once_with(host="db", port=5432)
    ensure_major.assert_awaited_once_with(conn)
    ensure_extensions.assert_awaited_once_with(conn, ("pg_textsearch",))
    conn.close.assert_awaited_once()


async def test_bm25_rebuild_provisions_indexes_then_relabels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.tools import rebuild_bm25 as module

    config = _config()
    events: list[object] = []
    prerequisites = AsyncMock(side_effect=lambda *_args, **_kwargs: events.append("prerequisites"))
    monkeypatch.setattr(module, "_ensure_bm25_prerequisites", prerequisites)

    async def rebuild(config: Any, **kwargs: Any) -> dict[str, int]:
        events.append(("rebuild", config, kwargs))
        return {"processed_chunks": 3, "updated_chunks": 3}

    monkeypatch.setattr(module, "rebuild_postgres_bm25", rebuild)
    fake_pool = SimpleNamespace(bind=MagicMock(), close=AsyncMock())
    monkeypatch.setattr(module, "pg_pool", fake_pool)

    stats = await module.run_rebuild_bm25(
        config=cast(Any, config),
        assume_yes=True,
        batch_size=25,
    )

    assert stats == {"processed_chunks": 3, "updated_chunks": 3}
    assert events[0] == "prerequisites"
    rebuild_event = cast(tuple[str, Any, dict[str, Any]], events[1])
    assert rebuild_event[0] == "rebuild"
    assert rebuild_event[1] is config
    assert rebuild_event[2] == {"pool": fake_pool, "batch_size": 25}
    fake_pool.bind.assert_called_once_with(config)
    fake_pool.close.assert_awaited_once()
