# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""DlightRAG defaults must never sit below LightRAG's own defaults.

Every knob here is one DlightRAG hands to LightRAG, so a lower value silently
narrows upstream behaviour instead of configuring it. The mapping is written
out by hand rather than matched on name: ``chunk_p_token_size`` answers to
``DEFAULT_CHUNK_P_SIZE``, and a name match would pair ``timeout`` with
``DEFAULT_TIMEOUT``, which is upstream's Gunicorn worker timeout and unrelated
to an LLM call budget.
"""

import lightrag.constants as lightrag_constants
import pytest

from dlightrag.config import DlightragConfig, LLMConfig

# DlightRAG default -> the LightRAG constant it must not undercut.
UPSTREAM_FLOORS: dict[str, str] = {
    "top_k": "DEFAULT_TOP_K",
    "chunk_top_k": "DEFAULT_CHUNK_TOP_K",
    "chunk_p_token_size": "DEFAULT_CHUNK_P_SIZE",
    "rag_pipeline_max_async": "DEFAULT_MAX_ASYNC",
    "embedding_func_max_async": "DEFAULT_EMBEDDING_FUNC_MAX_ASYNC",
    "embedding_batch_num": "DEFAULT_EMBEDDING_BATCH_NUM",
    "max_total_tokens": "DEFAULT_MAX_TOTAL_TOKENS",
    "max_entity_tokens": "DEFAULT_MAX_ENTITY_TOKENS",
    "max_relation_tokens": "DEFAULT_MAX_RELATION_TOKENS",
    "max_parallel_analyze": "DEFAULT_MAX_PARALLEL_ANALYZE",
    "max_parallel_insert": "DEFAULT_MAX_PARALLEL_INSERT",
    "max_parallel_parse_docling": "DEFAULT_MAX_PARALLEL_PARSE_DOCLING",
    "max_parallel_parse_mineru": "DEFAULT_MAX_PARALLEL_PARSE_MINERU",
    "max_parallel_parse_native": "DEFAULT_MAX_PARALLEL_PARSE_NATIVE",
    "queue_size_analyze": "DEFAULT_QUEUE_SIZE_ANALYZE",
    "queue_size_insert": "DEFAULT_QUEUE_SIZE_INSERT",
    "queue_size_parse": "DEFAULT_QUEUE_SIZE_PARSE",
}


@pytest.mark.parametrize(("field_name", "constant_name"), sorted(UPSTREAM_FLOORS.items()))
def test_default_is_not_below_lightrag(field_name: str, constant_name: str) -> None:
    ours = DlightragConfig.model_fields[field_name].default
    upstream = getattr(lightrag_constants, constant_name)

    assert ours >= upstream, (
        f"{field_name}={ours} is below LightRAG {constant_name}={upstream}. "
        "Raise it to at least the upstream value."
    )


def test_shipped_llm_timeout_is_not_below_lightrag() -> None:
    """The default role's timeout becomes LightRAG's ``default_llm_timeout``."""
    assert LLMConfig().default.timeout >= lightrag_constants.DEFAULT_LLM_TIMEOUT
