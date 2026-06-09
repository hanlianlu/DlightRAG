#!/usr/bin/env python3
"""RAGAS evaluation adapter for DlightRAG.

Reuses LightRAG's built-in :class:`RAGEvaluator` — RAGAS metrics
(Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision),
two-stage pipeline concurrency, progress bars, and CSV/JSON export.

Only :meth:`generate_rag_response` is overridden to call DlightRAG's
``/api/answer`` instead of LightRAG's ``/query``.

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

# Load .env so DLIGHTRAG_API_AUTH_TOKEN and EVAL_LLM_* vars are available.
load_dotenv(dotenv_path=".env", override=False)


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
  # Default dataset + local DlightRAG API
  python scripts/ragas_eval.py --api http://localhost:8100

  # Custom dataset
  python scripts/ragas_eval.py --api http://localhost:8100 --dataset my_questions.json

  # With auth
  python scripts/ragas_eval.py --api https://dlightrag.example.com --api-key "$TOKEN"

  # Remote DlightRAG instance
  DLIGHTRAG_API_TOKEN=... \\
  python scripts/ragas_eval.py --api https://your-server.example.com
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
# Shared env guidance (mirrors LightRAG's RAGEvaluator conventions)
# ═══════════════════════════════════════════════════════════════════

_EXPECTED_ENV_VARS: dict[str, str] = {
    "EVAL_LLM_MODEL": "LLM for RAGAS scoring (default: gpt-4o-mini)",
    "EVAL_EMBEDDING_MODEL": "Embedding model for RAGAS (default: text-embedding-3-large)",
    "EVAL_LLM_BINDING_API_KEY": "API key for eval LLM (falls back to OPENAI_API_KEY)",
    "EVAL_LLM_BINDING_HOST": "Custom endpoint for eval LLM (optional)",
    "EVAL_EMBEDDING_BINDING_API_KEY": "API key for eval embeddings (falls back to EVAL_LLM_BINDING_API_KEY)",
    "EVAL_EMBEDDING_BINDING_HOST": "Custom endpoint for eval embeddings (falls back to EVAL_LLM_BINDING_HOST)",
    "DLIGHTRAG_API_URL": "DlightRAG API base URL (default for --api)",
    "DLIGHTRAG_API_TOKEN": "Bearer token when auth is enabled on DlightRAG",
    "EVAL_QUERY_TOP_K": "top_k sent to DlightRAG /api/answer (default: 10)",
    "EVAL_MAX_CONCURRENT": "RAGAS evaluation concurrency (default: 2)",
    "EVAL_LLM_MAX_RETRIES": "Max retries for eval LLM calls (default: 5)",
    "EVAL_LLM_TIMEOUT": "Timeout per eval LLM call, seconds (default: 180)",
}


def _check_env() -> None:
    """Print configured environment variables (informational)."""
    logger.info("Environment variables (set → value, unset → <not set>):")
    for var, description in _EXPECTED_ENV_VARS.items():
        value = os.getenv(var)
        if value:
            display = value if "KEY" not in var and "TOKEN" not in var else "***"
            logger.info("  %-34s = %s  # %s", var, display, description)
        else:
            logger.info("  %-34s   <not set>  # %s", var, description)
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
