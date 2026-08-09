#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""CLI for dlightrag — ingestion runs locally, queries go through the REST API.

Usage:
    # Local ingestion (runs directly via RAGService, no API server needed)
    uv run scripts/cli.py ingest ./docs
    uv run scripts/cli.py ingest ./docs --replace
    uv run scripts/cli.py ingest ./docs --workspace project-a
    uv run scripts/cli.py ingest ./report.pdf --title "Quarterly Report" --metadata-json '{"department":"finance"}'

    # Azure Blob ingestion
    uv run scripts/cli.py ingest --source azure_blob --container my-container
    uv run scripts/cli.py ingest --source azure_blob --container c --blob-path docs/report.pdf
    uv run scripts/cli.py ingest --source azure_blob --container c --prefix reports/

    # S3 ingestion
    uv run scripts/cli.py ingest --source s3 --bucket my-bucket --s3-key docs/report.pdf --s3-region us-east-1
    uv run scripts/cli.py ingest --source s3 --bucket my-bucket --prefix docs/

    # URL ingestion
    uv run scripts/cli.py ingest --source url --url https://example.com/doc.pdf --filename doc.pdf

    # Query & answer (requires API server: docker compose up dlightrag-api)
    uv run scripts/cli.py query "What are the key findings?" --chunk-top-k 30
    uv run scripts/cli.py query "findings?" --workspaces project-a project-b
    uv run scripts/cli.py query "findings?" --filters-json '{"author":"Ada"}'
    uv run scripts/cli.py answer "What are the key findings?"
    uv run scripts/cli.py answer "summarize report" --attach ./report.pdf --attach-url https://example.com/appendix.pdf
    uv run scripts/cli.py chat
    uv run scripts/cli.py chat --workspaces project-a project-b

"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from dlightrag.core.client_requests import (
    ingest_kwargs_from_payload,
    ingest_spec_from_payload,
    query_image_blocks_from_urls,
)

DEFAULT_API_URL = "http://localhost:8100"
DEFAULT_QUERY_TIMEOUT = 120


def _get_timeout() -> int:
    env_val = os.environ.get("DLIGHTRAG_REQUEST_TIMEOUT")
    if env_val:
        return int(env_val)
    return DEFAULT_QUERY_TIMEOUT


def _get_api_url() -> str:
    return os.environ.get("DLIGHTRAG_API_URL", DEFAULT_API_URL)


def _get_auth_token() -> str | None:
    """Resolve API bearer token from env or simple-auth config."""
    token = os.environ.get("DLIGHTRAG_API_TOKEN") or os.environ.get("DLIGHTRAG_API_AUTH_TOKEN")
    if token:
        return token

    from dlightrag.config import DlightragConfig

    config = DlightragConfig()  # pyright: ignore[reportCallIssue]
    if config.auth_mode == "simple" and config.api_auth_token:
        return config.api_auth_token
    return None


def _headers() -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    token = _get_auth_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


def _json_object_arg(value: str) -> dict[str, Any]:
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"expected JSON object: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError("expected JSON object")
    return data


def _die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(2)


def _validate_ingest_args(args: argparse.Namespace) -> None:
    """Reject flags aimed at the wrong source, then defer to the shared spec."""
    if args.source_type != "local" and args.path:
        _die(f"positional path is not used with {args.source_type}")
    if args.source_type != "azure_blob" and (args.container_name or args.blob_path):
        _die("--container, --blob-path are only for azure_blob source")
    if args.source_type != "s3" and (args.bucket or args.s3_key):
        _die("--bucket, --s3-key are only for s3 source")
    try:
        ingest_spec_from_payload(args)
    except ValidationError as exc:
        _die("; ".join(error["msg"].removeprefix("Value error, ") for error in exc.errors()))


def _metadata_filter_payload(args: argparse.Namespace) -> dict[str, Any] | None:
    filters = dict(args.filters_json or {})
    custom = args.filter_custom
    if custom is not None:
        existing = filters.get("custom")
        filters["custom"] = {**existing, **custom} if isinstance(existing, dict) else custom

    return filters or None


def _apply_common_options(
    payload: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if getattr(args, "chunk_top_k", None) is not None:
        payload["chunk_top_k"] = args.chunk_top_k
    if args.workspaces:
        payload["workspaces"] = args.workspaces

    filters = _metadata_filter_payload(args)
    if filters is not None:
        payload["filters"] = filters

    return payload


def _apply_query_options(
    payload: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    _apply_common_options(payload, args)

    if args.query_images:
        payload["query_images"] = query_image_blocks_from_urls(args.query_images)

    return payload


def _build_answer_payload(
    args: argparse.Namespace,
    *,
    query: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": query, "stream": False}
    _apply_common_options(payload, args)
    attachment_urls = getattr(args, "attachment_urls", None)
    if attachment_urls:
        payload["attachments"] = [{"url": url} for url in attachment_urls]
    return payload


def _answer_images_by_id(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    images = data.get("answer_images") or []
    if not isinstance(images, list):
        return {}
    return {
        str(image["id"]): image for image in images if isinstance(image, dict) and image.get("id")
    }


def _format_answer_image_ref(image: dict[str, Any] | None, image_id: str) -> str:
    if image is None:
        return f"[image {image_id or '?'}]"

    source_ref = str(image.get("source_ref") or image.get("id") or image_id or "?")
    text = f"[image {source_ref}]"
    label = str(image.get("label") or "").strip()
    if label:
        text = f"{text} {label}"
    url = str(image.get("thumbnail_url") or image.get("url") or "").strip()
    if url:
        text = f"{text} {url}"
    return text


def _render_answer_for_terminal(data: dict[str, Any]) -> str:
    """Render structured answer blocks in a terminal-friendly text form."""
    fallback = str(data.get("answer") or "(no answer)")
    blocks = data.get("answer_blocks") or []
    if not isinstance(blocks, list) or not blocks:
        images = _answer_images_by_id(data).values()
        image_refs = [
            _format_answer_image_ref(image, str(image.get("id") or "")) for image in images
        ]
        return "\n".join([fallback, *image_refs]) if image_refs else fallback

    images_by_id = _answer_images_by_id(data)
    rendered: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "markdown":
            rendered.append(str(block.get("text") or ""))
        elif block.get("type") == "image_ref":
            image_id = str(block.get("image_id") or "")
            rendered.append(f"\n{_format_answer_image_ref(images_by_id.get(image_id), image_id)}\n")

    return "".join(rendered).strip() or fallback


# ═══════════════════════════════════════════════════════════════════
# ingest
# ═══════════════════════════════════════════════════════════════════


async def _run_ingest(args: argparse.Namespace) -> None:
    from dlightrag.config import get_config
    from dlightrag.core.service import RAGService

    source = args.source_type
    kwargs = ingest_kwargs_from_payload(args)

    if source == "local":
        print(f"Ingesting: {args.path} (replace={args.replace})")
    elif source == "azure_blob":
        target = args.blob_path or (f"prefix={args.prefix}" if args.prefix else "entire container")
        print(
            f"Ingesting Azure Blob: container={args.container_name}, {target} (replace={args.replace})"
        )
    elif source == "s3":
        target = args.s3_key or (f"prefix={args.prefix}" if args.prefix else "entire bucket")
        print(f"Ingesting S3: bucket={args.bucket}, {target} (replace={args.replace})")

    config = get_config()
    workspace = args.workspace or config.workspace
    if args.workspace:
        config = config.model_copy(update={"workspace": workspace})
    print(f"Workspace: {workspace}\n")

    service = await RAGService.acreate(config=config)
    try:
        result = await service.aingest(source_type=source, **kwargs)
        _print_json(result)
    except KeyboardInterrupt:
        print("\nIngestion cancelled.")
    except Exception as e:
        print(f"Ingestion failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        await service.aclose()


def cmd_ingest(args: argparse.Namespace) -> None:
    _validate_ingest_args(args)
    asyncio.run(_run_ingest(args))


# ═══════════════════════════════════════════════════════════════════
# query / answer / chat
# ═══════════════════════════════════════════════════════════════════


def cmd_query(args: argparse.Namespace) -> None:
    url = f"{_get_api_url()}/retrieve"
    payload = _apply_query_options({"query": args.query}, args)

    print(f"Query: {args.query}")
    if args.workspaces:
        print(f"Workspaces: {', '.join(args.workspaces)}")
    print(f"API: {url}\n")

    resp = httpx.post(url, json=payload, headers=_headers(), timeout=_get_timeout())
    resp.raise_for_status()
    _print_json(resp.json())


def _post_answer(
    url: str, payload: dict[str, Any], attachment_paths: list[str] | None
) -> httpx.Response:
    """Send an answer request as JSON, or multipart when local files are attached."""
    if attachment_paths:
        files = [
            (
                "attachments",
                (Path(path).name, Path(path).read_bytes(), "application/octet-stream"),
            )
            for path in attachment_paths
        ]
        return httpx.post(
            url,
            data={"request": json.dumps(payload)},
            files=files,
            headers=_headers(),
            timeout=_get_timeout(),
        )
    return httpx.post(url, json=payload, headers=_headers(), timeout=_get_timeout())


def cmd_answer(args: argparse.Namespace) -> None:
    url = f"{_get_api_url()}/answer"
    payload = _build_answer_payload(args, query=args.query)

    print(f"Question: {args.query}")
    if args.workspaces:
        print(f"Workspaces: {', '.join(args.workspaces)}")
    print(f"API: {url}\n")

    resp = _post_answer(url, payload, getattr(args, "attachment_paths", None))
    resp.raise_for_status()
    data = resp.json()

    # Print answer first, then validated references.
    answer = _render_answer_for_terminal(data)
    print(f"Answer:\n{answer}\n")

    references = data.get("references") or []
    if references:
        print(f"References ({len(references)}):")
        for ref in references:
            print(f"  [{ref.get('id', '?')}] {ref.get('title', '')}")


def cmd_chat(args: argparse.Namespace) -> None:
    url = f"{_get_api_url()}/answer"

    ws_info = f", workspaces={','.join(args.workspaces)}" if args.workspaces else ""
    print(f"dlightrag chat (API={_get_api_url()}{ws_info})")
    print("Type your question, or /quit to exit. Each request is stateless.\n")

    while True:
        try:
            question = input("You: ").strip()
        except EOFError, KeyboardInterrupt:
            print("\nBye!")
            break

        if not question:
            continue
        if question in ("/quit", "/exit", "/q"):
            print("Bye!")
            break
        payload = _build_answer_payload(
            args,
            query=question,
        )

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
        rendered_answer = _render_answer_for_terminal(data)

        print(f"\nAssistant: {rendered_answer}")

        sources = data.get("sources") or []
        if sources:
            titles = {s.get("title") for s in sources if s.get("title")}
            if titles:
                print(f"  Sources: {', '.join(sorted(titles))}")
        print()


def _add_filter_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--filters-json",
        type=_json_object_arg,
        default=None,
        help="Full metadata filters JSON object sent to the API",
    )
    parser.add_argument(
        "--filter-custom-json",
        type=_json_object_arg,
        default=None,
        dest="filter_custom",
        help="Custom metadata filter JSON object",
    )


def _add_common_options(
    parser: argparse.ArgumentParser,
    *,
    include_chunk_top_k: bool = False,
) -> None:
    parser.add_argument("--top-k", type=int, default=None, dest="top_k")
    if include_chunk_top_k:
        parser.add_argument("--chunk-top-k", type=int, default=None, dest="chunk_top_k")
    parser.add_argument("--workspaces", nargs="+", default=None, help="Workspaces (federation)")
    _add_filter_options(parser)


def _add_retrieval_options(
    parser: argparse.ArgumentParser,
    *,
    include_chunk_top_k: bool = False,
) -> None:
    _add_common_options(parser, include_chunk_top_k=include_chunk_top_k)
    parser.add_argument(
        "--query-image",
        action="append",
        default=None,
        dest="query_images",
        help="User-attached image URL or data URI; repeat up to 3 times",
    )


def _add_answer_options(parser: argparse.ArgumentParser) -> None:
    _add_common_options(parser, include_chunk_top_k=True)
    parser.add_argument(
        "--attach",
        action="append",
        default=None,
        dest="attachment_paths",
        help="Local file to attach as an answer resource; repeatable",
    )
    parser.add_argument(
        "--attach-url",
        action="append",
        default=None,
        dest="attachment_urls",
        help="HTTPS URL to attach as an answer resource; repeatable",
    )


# ═══════════════════════════════════════════════════════════════════
# parser
# ═══════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dlightrag-cli",
        description="dlightrag CLI — ingestion runs locally, queries go through the REST API",
        suggest_on_error=True,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- ingest --
    p_ingest = sub.add_parser(
        "ingest",
        help="Ingest documents from local, Azure Blob, S3, or URL sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Ingest documents into the RAG knowledge base.\n\n"
            "Source types:\n"
            "  local (default)  Ingest from local filesystem (file or directory)\n"
            "  azure_blob       Ingest from Azure Blob Storage container\n"
            "  s3               Ingest from AWS S3 bucket\n"
            "  url              Ingest from public or signed HTTPS URLs\n\n"
            "Examples:\n"
            "  %(prog)s ./docs                                          # local file/dir\n"
            "  %(prog)s ./docs --replace                                # local with replace\n"
            "  %(prog)s --source azure_blob --container my-container    # entire container\n"
            "  %(prog)s --source azure_blob --container c --prefix rpt/ # by prefix\n"
            "  %(prog)s --source s3 --bucket my-bucket --s3-key doc.pdf    # S3 single object\n"
            "  %(prog)s --source s3 --bucket my-bucket --prefix docs/   # S3 by prefix\n"
            "  %(prog)s --source url --url https://example.com/doc.pdf  # URL single document\n"
            "  %(prog)s --source url --urls https://example.com/a.pdf https://example.com/b.pdf\n"
            "  %(prog)s --source url --url 'https://fetch.example.com/doc?sig=...' --retain-source-file\n"
            "  %(prog)s --source url --url 'https://fetch.example.com/doc?sig=...' "
            "--download-uri https://cdn.example.com/doc.pdf"
        ),
    )
    p_ingest.add_argument("path", nargs="?", default=None, help="Path to file or directory (local)")
    p_ingest.add_argument(
        "--source",
        choices=["local", "azure_blob", "s3", "url"],
        default="local",
        dest="source_type",
        help="Data source type (default: local)",
    )
    p_ingest.add_argument("--container", dest="container_name", help="Azure Blob container name")
    p_ingest.add_argument("--blob-path", dest="blob_path", help="Specific blob (azure_blob)")
    p_ingest.add_argument("--prefix", help="Blob prefix filter (azure_blob/s3)")
    p_ingest.add_argument("--bucket", help="S3 bucket name")
    p_ingest.add_argument("--s3-region", dest="s3_region", help="S3 region name")
    p_ingest.add_argument("--s3-key", dest="s3_key", help="S3 object key")
    p_ingest.add_argument("--url", help="Public or signed HTTPS document URL")
    p_ingest.add_argument("--urls", nargs="+", help="Public or signed HTTPS document URLs")
    p_ingest.add_argument("--filename", help="Parser filename for a single URL")
    p_ingest.add_argument(
        "--source-uri", dest="source_uri", help="Stable source URI for a single URL"
    )
    p_ingest.add_argument(
        "--source-uris",
        nargs="+",
        dest="source_uris",
        help="Stable source URIs for URL batches",
    )
    p_ingest.add_argument(
        "--download-uri",
        dest="download_uri",
        help="Queryless durable download URI for a single URL (required for signed URLs unless retained)",
    )
    p_ingest.add_argument(
        "--download-uris",
        nargs="+",
        dest="download_uris",
        help="Queryless durable download URIs for URL batches",
    )
    p_ingest.add_argument(
        "--retain-source-file",
        action="store_true",
        default=None,
        dest="retain_source_file",
        help="Retain fetched bytes for later download (including signed URL fetches)",
    )
    p_ingest.add_argument("--replace", action="store_true", help="Replace existing documents")
    p_ingest.add_argument("--workspace", default=None, help="Target workspace")
    p_ingest.add_argument("--title", default=None, help="Optional document title metadata")
    p_ingest.add_argument("--author", default=None, help="Optional document author metadata")
    p_ingest.add_argument(
        "--metadata-json",
        type=_json_object_arg,
        default=None,
        dest="metadata",
        help="User metadata JSON object to attach to ingested documents",
    )

    # -- query --
    p_query = sub.add_parser("query", help="Retrieve contexts and sources (no answer)")
    p_query.add_argument("query", help="Search query")
    _add_retrieval_options(p_query, include_chunk_top_k=True)

    # -- answer --
    p_answer = sub.add_parser("answer", help="LLM-generated answer with contexts and sources")
    p_answer.add_argument("query", help="Question to answer")
    _add_answer_options(p_answer)

    # -- chat --
    p_chat = sub.add_parser("chat", help="Interactive multi-turn conversation")
    _add_common_options(p_chat, include_chunk_top_k=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "answer": cmd_answer,
        "chat": cmd_chat,
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
