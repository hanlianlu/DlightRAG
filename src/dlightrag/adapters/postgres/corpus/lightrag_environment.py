# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""LightRAG's environment interface, written from the typed configuration.

LightRAG reads its PostgreSQL, Milvus, parser-sidecar, and runtime settings from
process environment variables. The PostgreSQL corpus adapter is what constructs
LightRAG, so the bridge lives here and runs when that adapter is built;
configuration only holds the settings and never mutates the environment.
"""

import os
from urllib.parse import urlencode

from dlightrag.adapters.postgres.core._session_settings import (
    lightrag_pool_server_settings,
    setting_text,
)
from dlightrag.application.config import DlightragConfig


def lightrag_backend_env(config: DlightragConfig) -> dict[str, str]:
    """LightRAG's PostgreSQL and vector-store variables for this configuration.

    ``POSTGRES_WORKSPACE`` is always cleared: DlightRAG multiplexes WorkspaceRag
    instances in one process, and LightRAG's process-global workspace override
    would bind every storage instance to one workspace. An optional binding that
    is unset (SSL files, Milvus fields) is omitted, leaving an inherited value.
    """
    pg, vector = config.storage.postgres, config.storage.lightrag
    endpoint = config.pg_connection_kwargs()
    values: dict[str, str | int | float | None] = {
        "POSTGRES_WORKSPACE": "",
        "POSTGRES_HOST": endpoint["host"],
        "POSTGRES_PORT": endpoint["port"],
        "POSTGRES_USER": endpoint["user"],
        "POSTGRES_PASSWORD": endpoint["password"],
        "POSTGRES_DATABASE": endpoint["database"],
        "POSTGRES_VECTOR_INDEX_TYPE": vector.vector_index_type,
        "POSTGRES_HNSW_M": vector.hnsw_m,
        "POSTGRES_HNSW_EF": vector.hnsw_ef_construction,
        "POSTGRES_MAX_CONNECTIONS": pg.lightrag_pool_max_size,
        "POSTGRES_CONNECTION_RETRIES": pg.connection_retries,
        "POSTGRES_CONNECTION_RETRY_BACKOFF": pg.connection_retry_backoff,
        "POSTGRES_CONNECTION_RETRY_BACKOFF_MAX": pg.connection_retry_backoff_max,
        "POSTGRES_POOL_CLOSE_TIMEOUT": pg.pool_close_timeout,
        "POSTGRES_SERVER_SETTINGS": urlencode(lightrag_pool_server_settings(config)),
    }
    if pg.statement_cache_size is not None:
        values["POSTGRES_STATEMENT_CACHE_SIZE"] = pg.statement_cache_size
    rendered = {key: str(value) for key, value in values.items()}
    for key, value in {
        "POSTGRES_SSL_MODE": pg.ssl_mode,
        "POSTGRES_SSL_CERT": pg.ssl_cert,
        "POSTGRES_SSL_KEY": pg.ssl_key,
        "POSTGRES_SSL_ROOT_CERT": pg.ssl_root_cert,
        "POSTGRES_SSL_CRL": pg.ssl_crl,
        "MILVUS_URI": vector.milvus_uri,
        "MILVUS_TOKEN": vector.milvus_token,
        "MILVUS_DB_NAME": vector.milvus_db_name,
    }.items():
        if (text := setting_text(value)) is not None:
            rendered[key] = text
    return rendered


def lightrag_sidecar_env(config: DlightragConfig) -> dict[str, str]:
    """The VLM and active parser sidecar variables; unset optional fields are omitted."""
    sidecars = config.corpus.sidecars
    parser = sidecars.mineru if sidecars.mineru is not None else sidecars.docling
    raw = {
        env: getattr(settings, field)
        for settings in (sidecars.vlm, parser)
        if settings is not None
        for field, env in settings._ENV_MAP.items()
    }
    return {key: text for key, value in raw.items() if (text := setting_text(value)) is not None}


def lightrag_runtime_env(config: DlightragConfig) -> dict[str, str]:
    """The parser routing rules and the directory LightRAG reads parser inputs from.

    That is the service's own corpus directory, never the folder operators place
    local sources in: LightRAG looks a document up there by its basename alone.
    """
    return {"LIGHTRAG_PARSER": config.parser_rules, "INPUT_DIR": str(config.corpus_dir_path)}


def apply_lightrag_environment(config: DlightragConfig) -> None:
    """Bridge typed host settings to LightRAG's environment interface.

    Resolved bindings win over inherited variables; an optional binding that is
    unset leaves inherited LightRAG behaviour in place.
    """
    os.environ.update(lightrag_backend_env(config))
    os.environ.update(lightrag_sidecar_env(config))
    os.environ.update(lightrag_runtime_env(config))


__all__ = [
    "apply_lightrag_environment",
    "lightrag_backend_env",
    "lightrag_runtime_env",
    "lightrag_sidecar_env",
]
