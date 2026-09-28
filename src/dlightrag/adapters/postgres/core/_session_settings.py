# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL session settings derived from the typed configuration."""

from dlightrag.application.config import DlightragConfig


def setting_text(value: str | int | float | bool | None) -> str | None:
    """Render one configured value as PostgreSQL and LightRAG read it; None when unset."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip() or None


def domain_pool_server_settings(config: DlightragConfig) -> dict[str, str]:
    """Session GUCs for DlightRAG's domain pool: HNSW search breadth, then configured ones."""
    pg, vector = config.storage.postgres, config.storage.lightrag
    settings = {"hnsw.ef_search": str(vector.hnsw_ef_search)}
    for key, value in pg.session_settings.items():
        if (rendered := setting_text(value)) is not None:
            settings[str(key)] = rendered
    return settings


def lightrag_pool_server_settings(config: DlightragConfig) -> dict[str, str]:
    """The domain settings; a reader's corpus pool also runs read-only transactions.

    The reader invariant is applied last, so no configured session setting can
    switch a reader's LightRAG sessions back to writable.
    """
    settings = domain_pool_server_settings(config)
    if config.is_reader:
        settings["default_transaction_read_only"] = "on"
    return settings


__all__ = ["domain_pool_server_settings", "lightrag_pool_server_settings", "setting_text"]
