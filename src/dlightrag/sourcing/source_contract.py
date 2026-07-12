# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Source identity and durable download locator policies."""

from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlsplit

from dlightrag.sourcing.uri import parse_remote_uri
from dlightrag.utils import normalize_workspace


class SourceDownloadContractError(ValueError):
    """A safe client-visible source/download invariant failure."""


def validate_source_uri(value: str) -> str:
    if not value or "\x00" in value:
        raise ValueError("source_uri is invalid")
    return value


def local_source_uri(workspace: str, relative_path: str | Path) -> str:
    safe_workspace = normalize_workspace(workspace)
    path = PurePosixPath(Path(relative_path).as_posix())
    if (
        not safe_workspace
        or not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or "\x00" in path.as_posix()
    ):
        raise ValueError("local source identity is invalid")
    return f"local://{quote(safe_workspace, safe='')}/{quote(path.as_posix(), safe='/')}"


def validate_download_uri(value: str) -> str:
    candidate = value.strip()
    try:
        parsed = urlsplit(candidate)
        if parsed.scheme in {"s3", "azure"}:
            parse_remote_uri(candidate)
            decoded_path = PurePosixPath(unquote(parsed.path))
            if (
                parsed.query
                or parsed.fragment
                or parsed.username
                or parsed.password
                or ".." in decoded_path.parts
                or "\\" in unquote(parsed.path)
            ):
                raise ValueError
            return candidate
        if parsed.scheme == "https":
            from dlightrag.sourcing.url import validate_public_https_url

            validate_public_https_url(candidate)
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError
            return candidate
    except ValueError as exc:
        raise ValueError(
            "durable download_uri must be a well-formed s3://, azure://, "
            "or credential-free queryless public https:// URI"
        ) from exc
    raise ValueError(
        "durable download_uri must be a well-formed s3://, azure://, "
        "or credential-free queryless public https:// URI"
    )


def implicit_https_download_uri(fetch_url: str) -> str | None:
    parsed = urlsplit(fetch_url)
    if parsed.query or parsed.fragment:
        return None
    return validate_download_uri(fetch_url)
