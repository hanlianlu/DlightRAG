#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""CLI for dlightrag — ingestion runs locally, queries go through the REST API.

Usage:
    # Local ingestion (runs directly via RAGService, no API server needed)
    uv run scripts/cli.py ingest ./docs
    uv run scripts/cli.py ingest ./docs --replace
    uv run scripts/cli.py ingest ./docs --workspace project-a

    # Azure Blob ingestion
    uv run scripts/cli.py ingest --source azure_blob --container my-container
    uv run scripts/cli.py ingest --source azure_blob --container c --blob-path docs/report.pdf
    uv run scripts/cli.py ingest --source azure_blob --container c --prefix reports/

    # S3 ingestion
    uv run scripts/cli.py ingest --source s3 --bucket my-bucket --key docs/report.pdf
    uv run scripts/cli.py ingest --source s3 --bucket my-bucket --prefix docs/

    # Query & answer (requires API server: docker compose up dlightrag-api)
    uv run scripts/cli.py query "What are the key findings?"
    uv run scripts/cli.py query "findings?" --workspaces project-a project-b
    uv run scripts/cli.py answer "What are the key findings?"
    uv run scripts/cli.py chat
    uv run scripts/cli.py chat --workspaces project-a project-b

    # RAGAS evaluation
    uv run scripts/cli.py ragas_eval --api http://localhost:8100
    uv run scripts/cli.py ragas_eval --api http://localhost:8100 --dataset my_tests.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

import httpx

DEFAULT_API_URL = "http://localhost:8100"
DEFAULT_QUERY_TIMEOUT = 120
DEFAULT_INGEST_TIMEOUT = 600


def _get_timeout(for_ingest: bool = False) -> int:
    env_val = os.environ.get("DLIGHTRAG_REQUEST_TIMEOUT")
    if env_val:
        return int(env_val)
    return DEFAULT_INGEST_TIMEOUT if for_ingest else DEFAULT_QUERY_TIMEOUT


def _get_api_url() -> str:
    return os.environ.get("DLIGHTRAG_API_URL", DEFAULT_API_URL)


def _get_auth_token() -> str | None:
    """Resolve API bearer token: env → DlightRAG config (simple/jwt)."""
    token = os.environ.get("DLIGHTRAG_API_TOKEN") or os.environ.get("DLIGHTRAG_API_AUTH_TOKEN")
    if token:
        return token

    try:
        from dlightrag.config import DlightragConfig

        config = DlightragConfig()  # pyright: ignore[reportCallIssue]
        if config.api_auth_token:
            return config.api_auth_token
        if config.jwt_secret and config.auth_mode == "jwt":
            import time

            import jwt  # PyJWT ≥2.8.0

            now = int(time.time())
            return jwt.encode(
                {"sub": "dlightrag-cli", "iat": now, "exp": now + 86400},
                config.jwt_secret,
                algorithm=config.jwt_algorithm or "HS256",
            )
    except Exception:
        pass
    return None


def _headers() -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    token = _get_auth_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


def _die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(2)


def _validate_ingest_args(args: argparse.Namespace) -> None:
    source = args.source_type
    if source == "local":
        if not args.path:
            _die("local source requires a path. Usage: ingest <path> [--replace]")
        if args.container_name or args.blob_path or args.prefix:
            _die("--container, --blob-path, --prefix are only for azure_blob source")
        if args.bucket or args.key:
            _die("--bucket, --key are only for s3 source")
    elif source == "azure_blob":
        if args.path:
            _die("positional path is not used with azure_blob. Use --blob-path or --prefix.")
        if not args.container_name:
            _die("azure_blob requires --container")
        if args.blob_path and args.prefix:
            _die("--blob-path and --prefix are mutually exclusive")
        if args.bucket or args.key:
            _die("--bucket, --key are only for s3 source")
    elif source == "s3":
        if args.path:
            _die("positional path is not used with s3")
        if not args.bucket:
            _die("s3 requires --bucket")
        if args.key and args.prefix:
            _die("--key and --prefix are mutually exclusive for s3")
        if args.container_name or args.blob_path:
            _die("--container, --blob-path are only for azure_blob source")


# ═══════════════════════════════════════════════════════════════════
# ingest
# ═══════════════════════════════════════════════════════════════════


async def _run_ingest(args: argparse.Namespace) -> None:
    from dlightrag.config import get_config
    from dlightrag.core.service import RAGService

    source = args.source_type
    kwargs: dict[str, Any] = {}

    if source == "local":
        kwargs["path"] = args.path
        kwargs["replace"] = args.replace
        print(f"Ingesting: {args.path} (replace={args.replace})")
    elif source == "azure_blob":
        kwargs["container_name"] = args.container_name
        if args.blob_path:
            kwargs["blob_path"] = args.blob_path
        if args.prefix is not None:
            kwargs["prefix"] = args.prefix
        kwargs["replace"] = args.replace
        target = args.blob_path or (f"prefix={args.prefix}" if args.prefix else "entire container")
        print(
            f"Ingesting Azure Blob: container={args.container_name}, {target} (replace={args.replace})"
        )
    elif source == "s3":
        kwargs["bucket"] = args.bucket
        if args.key:
            kwargs["key"] = args.key
        if args.prefix is not None:
            kwargs["prefix"] = args.prefix
        kwargs["replace"] = args.replace
        target = args.key or (f"prefix={args.prefix}" if args.prefix else "entire bucket")
        print(f"Ingesting S3: bucket={args.bucket}, {target} (replace={args.replace})")

    config = get_config()
    workspace = args.workspace or config.workspace
    if args.workspace:
        config = config.model_copy(update={"workspace": workspace})
    print(f"Workspace: {workspace}\n")

    service = await RAGService.create(config=config)
    try:
        result = await service.aingest(source_type=source, **kwargs)
        _print_json(result)
    except KeyboardInterrupt:
        print("\nIngestion cancelled.")
    except Exception as e:
        print(f"Ingestion failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        await service.close()


def cmd_ingest(args: argparse.Namespace) -> None:
    _validate_ingest_args(args)
    asyncio.run(_run_ingest(args))


# ═══════════════════════════════════════════════════════════════════
# query / answer / chat
# ═══════════════════════════════════════════════════════════════════


def cmd_query(args: argparse.Namespace) -> None:
    url = f"{_get_api_url()}/retrieve"
    payload: dict[str, Any] = {"query": args.query}
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.workspaces:
        payload["workspaces"] = args.workspaces

    print(f"Query: {args.query}")
    if args.workspaces:
        print(f"Workspaces: {', '.join(args.workspaces)}")
    print(f"API: {url}\n")

    resp = httpx.post(url, json=payload, headers=_headers(), timeout=_get_timeout())
    resp.raise_for_status()
    _print_json(resp.json())


def cmd_answer(args: argparse.Namespace) -> None:
    url = f"{_get_api_url()}/answer"
    payload: dict[str, Any] = {"query": args.query, "stream": False}
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.workspaces:
        payload["workspaces"] = args.workspaces

    print(f"Question: {args.query}")
    if args.workspaces:
        print(f"Workspaces: {', '.join(args.workspaces)}")
    print(f"API: {url}\n")

    resp = httpx.post(url, json=payload, headers=_headers(), timeout=_get_timeout())
    resp.raise_for_status()
    data = resp.json()

    # Print answer first, then references, then sources
    answer = data.get("answer") or "(no answer)"
    print(f"Answer:\n{answer}\n")

    references = data.get("references") or []
    if references:
        print(f"References ({len(references)}):")
        for ref in references:
            print(f"  [{ref.get('id', '?')}] {ref.get('title', '')}")


def cmd_chat(args: argparse.Namespace) -> None:
    url = f"{_get_api_url()}/answer"
    history: list[dict[str, str]] = []

    ws_info = f", workspaces={','.join(args.workspaces)}" if args.workspaces else ""
    print(f"dlightrag chat (API={_get_api_url()}{ws_info})")
    print("Type your question, or: /clear to reset history, /quit to exit\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question in ("/quit", "/exit", "/q"):
            print("Bye!")
            break
        if question == "/clear":
            history.clear()
            print("-- history cleared --\n")
            continue

        payload: dict[str, Any] = {
            "query": question,
            "stream": False,
        }
        if history:
            payload["conversation_history"] = [
                {"role": m["role"], "content": m["content"]} for m in history[-20:]
            ]
        if args.top_k is not None:
            payload["top_k"] = args.top_k
        if args.workspaces:
            payload["workspaces"] = args.workspaces

        try:
            resp = httpx.post(url, json=payload, headers=_headers(), timeout=_get_timeout())
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            print(f"[error] HTTP {e.response.status_code}: {e.response.text}\n")
            continue
        except httpx.ConnectError:
            print(f"[error] Connection failed: {_get_api_url()}\n")
            continue

        data = resp.json()
        answer_text = data.get("answer") or "(no answer)"

        print(f"\nAssistant: {answer_text}")

        sources = data.get("sources") or []
        if sources:
            titles = {s.get("title") for s in sources if s.get("title")}
            if titles:
                print(f"  Sources: {', '.join(sorted(titles))}")
        print()

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer_text})


# ═══════════════════════════════════════════════════════════════════
# ragas_eval
# ═══════════════════════════════════════════════════════════════════


def cmd_ragas_eval(args: argparse.Namespace) -> None:
    """Delegate to ragas_eval.py with the given args."""
    import subprocess

    eval_script = os.path.join(os.path.dirname(__file__), "ragas_eval.py")
    cmd: list[str] = [
        sys.executable,
        eval_script,
        "--api",
        args.api or _get_api_url(),
    ]
    if args.dataset:
        cmd.extend(["--dataset", args.dataset])
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])

    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=False)


# ═══════════════════════════════════════════════════════════════════
# parser
# ═══════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dlightrag-cli",
        description="dlightrag CLI — ingestion runs locally, queries go through the REST API",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- ingest --
    p_ingest = sub.add_parser(
        "ingest",
        help="Ingest documents from local, Azure Blob, or S3 sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Ingest documents into the RAG knowledge base.\n\n"
            "Source types:\n"
            "  local (default)  Ingest from local filesystem (file or directory)\n"
            "  azure_blob       Ingest from Azure Blob Storage container\n"
            "  s3               Ingest from AWS S3 bucket\n\n"
            "Examples:\n"
            "  %(prog)s ./docs                                          # local file/dir\n"
            "  %(prog)s ./docs --replace                                # local with replace\n"
            "  %(prog)s --source azure_blob --container my-container    # entire container\n"
            "  %(prog)s --source azure_blob --container c --prefix rpt/ # by prefix\n"
            "  %(prog)s --source s3 --bucket my-bucket --key doc.pdf    # S3 single object\n"
            "  %(prog)s --source s3 --bucket my-bucket --prefix docs/   # S3 by prefix"
        ),
    )
    p_ingest.add_argument("path", nargs="?", default=None, help="Path to file or directory (local)")
    p_ingest.add_argument(
        "--source",
        choices=["local", "azure_blob", "s3"],
        default="local",
        dest="source_type",
        help="Data source type (default: local)",
    )
    p_ingest.add_argument("--container", dest="container_name", help="Azure Blob container name")
    p_ingest.add_argument("--blob-path", dest="blob_path", help="Specific blob (azure_blob)")
    p_ingest.add_argument("--prefix", help="Blob prefix filter (azure_blob/s3)")
    p_ingest.add_argument("--bucket", help="S3 bucket name")
    p_ingest.add_argument("--key", help="S3 object key")
    p_ingest.add_argument("--replace", action="store_true", help="Replace existing documents")
    p_ingest.add_argument("--workspace", default=None, help="Target workspace")

    # -- query --
    p_query = sub.add_parser("query", help="Retrieve contexts and sources (no answer)")
    p_query.add_argument("query", help="Search query")
    p_query.add_argument("--top-k", type=int, default=None, dest="top_k")
    p_query.add_argument("--workspaces", nargs="+", default=None, help="Workspaces (federation)")

    # -- answer --
    p_answer = sub.add_parser("answer", help="LLM-generated answer with contexts and sources")
    p_answer.add_argument("query", help="Question to answer")
    p_answer.add_argument("--top-k", type=int, default=None, dest="top_k")
    p_answer.add_argument("--workspaces", nargs="+", default=None, help="Workspaces (federation)")

    # -- chat --
    p_chat = sub.add_parser("chat", help="Interactive multi-turn conversation")
    p_chat.add_argument("--top-k", type=int, default=None, dest="top_k")
    p_chat.add_argument("--workspaces", nargs="+", default=None, help="Workspaces (federation)")

    # -- ragas_eval --
    p_eval = sub.add_parser("ragas_eval", help="Run RAGAS evaluation against a DlightRAG API")
    p_eval.add_argument(
        "--api",
        default=None,
        help="DlightRAG API base URL (default: $DLIGHTRAG_API_URL or http://localhost:8100)",
    )
    p_eval.add_argument(
        "--dataset",
        "-d",
        default=None,
        help="Test dataset JSON (default: LightRAG bundled sample)",
    )
    p_eval.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Results directory (default: ./ragas_eval_results/)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "answer": cmd_answer,
        "chat": cmd_chat,
        "ragas_eval": cmd_ragas_eval,
    }

    try:
        dispatch[args.command](args)
    except httpx.HTTPStatusError as e:
        print(f"HTTP {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except httpx.TimeoutException:
        print(f"Request timed out: {_get_api_url()}", file=sys.stderr)
        sys.exit(1)
    except httpx.ConnectError:
        print(f"Connection failed: {_get_api_url()}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
