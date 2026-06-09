#!/usr/bin/env python3
"""RAGAS evaluation adapter for DlightRAG.

Reuses LightRAG's built-in :class:`RAGEvaluator` — RAGAS metrics
(Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision),
two-stage pipeline concurrency, progress bars, and CSV/JSON export.

Only :meth:`generate_rag_response` is overridden to call DlightRAG's
``/api/answer`` instead of LightRAG's ``/query``.

When ``EVAL_LLM_BINDING_API_KEY`` is not set, the adapter auto-resolves
eval credentials from DlightRAG's own config (query role → default LLM,
cascading down to the embedding config when provider-compatible).
No extra ``.env`` entries needed in the common case.

Usage::

    uv pip install ragas datasets langchain-openai
    uv run python scripts/ragas_eval.py --api http://localhost:8100

See `docs/ragas-evaluation.md <../docs/ragas-evaluation.md>`_ for full guide.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from lightrag.evaluation.eval_rag_quality import RAGEvaluator
from lightrag.utils import logger

# Load DlightRAG .env so DLIGHTRAG_* vars are available for fallback resolution.
load_dotenv(dotenv_path=".env", override=False)


# ═══════════════════════════════════════════════════════════════════
# Auto-resolve eval credentials from DlightRAG config
# ═══════════════════════════════════════════════════════════════════

# OpenAI-compatible embedding providers — their api_key + base_url work
# with langchain's OpenAIEmbeddings. Native-SDK providers (voyage, gemini,
# jina, dashscope_qwen) are excluded because their keys don't work with
# the OpenAI embeddings API.
_OPENAI_COMPATIBLE_EMBED_PROVIDERS = frozenset(
    {"openai_compatible", "qwen_openai_compatible", "ollama"}
)


def _resolve_eval_env() -> None:
    """Set EVAL_* env vars from DlightRAG config when not explicitly configured.

    Cascade (each level only applies when the env var is unset):

    Eval LLM:
      1. EVAL_LLM_BINDING_API_KEY  ← config.llm.roles.query.api_key
                                    ← config.llm.default.api_key
      2. EVAL_LLM_MODEL            ← query role model  ← default model
      3. EVAL_LLM_BINDING_HOST     ← query role base_url ← default base_url

    Eval embeddings:
      4. EVAL_EMBEDDING_BINDING_API_KEY  ← EVAL_LLM_BINDING_API_KEY
                                          ← DlightRAG embedding key (if OpenAI-compatible)
      5. EVAL_EMBEDDING_BINDING_HOST     ← EVAL_LLM_BINDING_HOST
                                          ← DlightRAG embedding base_url (if OpenAI-compatible)

    DlightRAG connection:
      6. DLIGHTRAG_API_TOKEN             ← config.api_auth_token
    """
    # If both eval keys are already set, nothing to do
    llm_key_set = bool(os.getenv("EVAL_LLM_BINDING_API_KEY"))
    embed_key_set = bool(os.getenv("EVAL_EMBEDDING_BINDING_API_KEY"))
    if llm_key_set and embed_key_set:
        return

    try:
        from dlightrag.config import DlightragConfig
    except ImportError:
        logger.warning("DlightRAG not importable — skipping eval credential auto-resolution")
        return

    try:
        config = DlightragConfig()  # pyright: ignore[reportCallIssue]
    except Exception:
        logger.warning(
            "DlightRAG config failed to load — eval credentials must be set explicitly "
            "via EVAL_LLM_BINDING_API_KEY. Run from the repo root where config.yaml exists.",
            exc_info=True,
        )
        return
    query_cfg = config.llm.roles.query or config.llm.default

    # -- Eval LLM --
    if not llm_key_set:
        if query_cfg.api_key:
            os.environ["EVAL_LLM_BINDING_API_KEY"] = query_cfg.api_key
            logger.info("Eval LLM key: auto-resolved from DlightRAG query/default role")
        elif os.getenv("OPENAI_API_KEY"):
            logger.info("Eval LLM key: using OPENAI_API_KEY")
        else:
            logger.warning(
                "No eval LLM key found — set EVAL_LLM_BINDING_API_KEY, "
                "DLIGHTRAG_LLM__DEFAULT__API_KEY, or OPENAI_API_KEY"
            )

    if not os.getenv("EVAL_LLM_MODEL"):
        os.environ["EVAL_LLM_MODEL"] = query_cfg.model
        logger.info("Eval LLM model: %s (from DlightRAG config)", query_cfg.model)

    if not os.getenv("EVAL_LLM_BINDING_HOST") and query_cfg.base_url:
        os.environ["EVAL_LLM_BINDING_HOST"] = query_cfg.base_url
        logger.info("Eval LLM host: %s (from DlightRAG config)", query_cfg.base_url)

    # -- Eval Embeddings --
    if not embed_key_set:
        # Cascade: EVAL_LLM key → DlightRAG embedding key (if OpenAI-compatible provider)
        resolved_embed_key = os.getenv("EVAL_LLM_BINDING_API_KEY")
        if (
            not resolved_embed_key
            and config.embedding.provider in _OPENAI_COMPATIBLE_EMBED_PROVIDERS
        ):
            resolved_embed_key = config.embedding.api_key
        if resolved_embed_key:
            os.environ["EVAL_EMBEDDING_BINDING_API_KEY"] = resolved_embed_key
            logger.info("Eval embedding key: cascaded from eval LLM or DlightRAG embedding config")

    if not os.getenv("EVAL_EMBEDDING_BINDING_HOST"):
        # Cascade: EVAL_LLM host → DlightRAG embedding host (if OpenAI-compatible)
        llm_host = os.getenv("EVAL_LLM_BINDING_HOST")
        embed_cfg = config.embedding
        if llm_host:
            os.environ["EVAL_EMBEDDING_BINDING_HOST"] = llm_host
        elif embed_cfg.provider in _OPENAI_COMPATIBLE_EMBED_PROVIDERS and embed_cfg.base_url:
            os.environ["EVAL_EMBEDDING_BINDING_HOST"] = embed_cfg.base_url

    # -- DlightRAG API token --
    if not os.getenv("DLIGHTRAG_API_TOKEN") and config.api_auth_token:
        os.environ["DLIGHTRAG_API_TOKEN"] = config.api_auth_token
        logger.info("DlightRAG API token: auto-resolved from config.api_auth_token")


# Run once at import time — sets env vars before RAGEvaluator.__init__ reads them.
_resolve_eval_env()


# ═══════════════════════════════════════════════════════════════════
# Adapter
# ═══════════════════════════════════════════════════════════════════


class DlightRAGAdapterEvaluator(RAGEvaluator):
    """RAGEvaluator wired to a DlightRAG ``/api/answer`` endpoint.

    Inherits everything — RAGAS metrics, concurrency, tqdm, CSV/JSON export —
    and only overrides the API-call method to speak DlightRAG's response format.
    """

    def __init__(
        self,
        test_dataset_path: str | None = None,
        rag_api_url: str | None = None,
        *,
        api_key: str | None = None,
    ) -> None:
        self._dlightrag_api_key = api_key
        # Parent RAGEvaluator defaults to its own sample_dataset.json and
        # localhost:9621 when passed None. Our CLI guarantees `api` is set,
        # but at the type level we forward the values as-is — the parent
        # constructor handles None defaults internally.
        super().__init__(  # pyright: ignore[reportArgumentType]
            test_dataset_path=test_dataset_path,  # type: ignore[arg-type]
            rag_api_url=rag_api_url,  # type: ignore[arg-type]
        )

    # ---------------------------------------------------------------- #
    #  The ONLY overridden method — format translation                 #
    # ---------------------------------------------------------------- #

    async def generate_rag_response(
        self,
        question: str,
        client: httpx.AsyncClient,
    ) -> dict[str, Any]:
        """Call DlightRAG ``POST /api/answer`` and translate to LightRAG format.

        DlightRAG response::

            {"answer": "...",
             "contexts": {"chunks": [{"content": "...", "chunk_id": "..."}]},
             "references": [...], "sources": [...], "trace": {...}}

        Translated to LightRAG RAGEvaluator format::

            {"answer": "...",
             "contexts": ["chunk text 1", "chunk text 2", ...]}
        """
        try:
            payload: dict[str, Any] = {
                "query": question,
                "mode": "mix",
                "stream": False,
                "top_k": int(os.getenv("EVAL_QUERY_TOP_K", "10")),
                "chunk_top_k": int(os.getenv("EVAL_QUERY_TOP_K", "10")) * 3,
            }

            headers: dict[str, str] = {}
            if self._dlightrag_api_key:
                headers["Authorization"] = f"Bearer {self._dlightrag_api_key}"

            response = await client.post(
                f"{self.rag_api_url}/api/answer",
                json=payload,
                headers=headers if headers else None,
            )
            response.raise_for_status()
            result = response.json()

            answer = result.get("answer", "No response generated")
            chunks = result.get("contexts", {}).get("chunks", [])

            # Extract content strings from DlightRAG's structured chunks
            contexts: list[str] = []
            for chunk in chunks:
                content = chunk.get("content", "")
                if isinstance(content, str) and content.strip():
                    contexts.append(content)

            if not contexts:
                logger.warning("Eval query returned no chunk content for: %s", question[:80])

            return {
                "answer": answer,
                "contexts": contexts,
            }

        except httpx.ConnectError as exc:
            raise Exception(
                f"Cannot connect to DlightRAG API at {self.rag_api_url}/api/answer\n"
                f"  Make sure DlightRAG is running: docker compose up -d\n"
                f"  Error: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise Exception(
                f"DlightRAG API error {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.ReadTimeout as exc:
            raise Exception(
                f"Request timeout waiting for DlightRAG response\n"
                f"  Question: {question[:100]}...\n"
                f"  Error: {exc}"
            ) from exc
        except Exception as exc:
            raise Exception(f"Error calling DlightRAG API: {type(exc).__name__}: {exc}") from exc


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAGAS evaluation for DlightRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Zero-config — eval creds auto-resolved from DlightRAG config
  python scripts/ragas_eval.py --api http://localhost:8100

  # Custom dataset
  python scripts/ragas_eval.py --api http://localhost:8100 --dataset my_questions.json

  # With auth
  python scripts/ragas_eval.py --api https://dlightrag.example.com --api-key "$TOKEN"

  # Explicit eval model overrides (overrides auto-resolution)
  EVAL_LLM_MODEL=gpt-4o EVAL_EMBEDDING_MODEL=text-embedding-3-large \\
    python scripts/ragas_eval.py --api http://localhost:8100
        """,
    )

    parser.add_argument(
        "--api",
        type=str,
        default=os.getenv("DLIGHTRAG_API_URL"),
        help="DlightRAG API base URL (default: $DLIGHTRAG_API_URL). Example: http://localhost:8100",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("DLIGHTRAG_API_TOKEN"),
        help="Bearer token when auth_mode is 'simple' or 'jwt' (default: $DLIGHTRAG_API_TOKEN).",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        help="Path to test dataset JSON file (default: bundled sample). "
        'Format: {"test_cases": [{"question": "...", "ground_truth": "..."}]}',
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Directory for CSV/JSON results (default: ./ragas_eval_results/).",
    )

    return parser


# ═══════════════════════════════════════════════════════════════════
# Shared env reference (informational — shown on startup)
# ═══════════════════════════════════════════════════════════════════

_EXPECTED_ENV_VARS: dict[str, str] = {
    "EVAL_LLM_MODEL": "LLM for RAGAS scoring (default: from DlightRAG config, or gpt-4o-mini)",
    "EVAL_LLM_BINDING_API_KEY": "API key for eval LLM (auto-resolved from DlightRAG config)",
    "EVAL_LLM_BINDING_HOST": "Custom endpoint for eval LLM (auto-resolved from DlightRAG config)",
    "EVAL_EMBEDDING_MODEL": "Embedding model for RAGAS (default: text-embedding-3-large)",
    "EVAL_EMBEDDING_BINDING_API_KEY": "API key for eval embeddings (cascaded from eval LLM key)",
    "EVAL_EMBEDDING_BINDING_HOST": "Custom endpoint for eval embeddings (cascaded from eval LLM host)",
    "DLIGHTRAG_API_URL": "DlightRAG API base URL (default for --api)",
    "DLIGHTRAG_API_TOKEN": "Bearer token (auto-resolved from config.api_auth_token)",
    "EVAL_QUERY_TOP_K": "top_k sent to DlightRAG /api/answer (default: 10)",
    "EVAL_MAX_CONCURRENT": "RAGAS evaluation concurrency (default: 2)",
    "EVAL_LLM_MAX_RETRIES": "Max retries for eval LLM calls (default: 5)",
    "EVAL_LLM_TIMEOUT": "Timeout per eval LLM call, seconds (default: 180)",
}


def _check_env() -> None:
    """Print configured environment variables (informational)."""
    logger.info("Environment variables (set → value, unset → <auto>):")
    for var, description in _EXPECTED_ENV_VARS.items():
        value = os.getenv(var)
        if value:
            display = value if "KEY" not in var and "TOKEN" not in var else "***"
            logger.info("  %-34s = %s  # %s", var, display, description)
        else:
            logger.info("  %-34s   <auto>  # %s", var, description)
    logger.info("")


async def _run() -> None:
    args = _build_parser().parse_args()

    if not args.api:
        print(
            "DlightRAG API URL is required. Set --api or DLIGHTRAG_API_URL.",
            file=sys.stderr,
        )
        sys.exit(1)

    _check_env()

    evaluator = DlightRAGAdapterEvaluator(
        test_dataset_path=args.dataset,
        rag_api_url=args.api.rstrip("/"),
        api_key=args.api_key,
    )

    # Override results directory if requested
    if args.output_dir:
        evaluator.results_dir = Path(args.output_dir)
        evaluator.results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("DlightRAG API: %s/api/answer", evaluator.rag_api_url)
    logger.info("Eval LLM:     %s", evaluator.eval_model)
    logger.info("Eval Embed:   %s", evaluator.eval_embedding_model)
    logger.info("Results dir:  %s", evaluator.results_dir.absolute())
    logger.info("")

    await evaluator.run()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
