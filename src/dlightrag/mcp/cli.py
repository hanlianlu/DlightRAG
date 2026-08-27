# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""CLI bootstrap that loads configuration before constructing the MCP server."""

import argparse

from dlightrag.application.config import load_config, set_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DlightRAG MCP server",
        suggest_on_error=True,
    )
    parser.add_argument("--env-file", help="Path to .env configuration file")
    args = parser.parse_args()
    if args.env_file:
        set_config(load_config(args.env_file))

    from dlightrag.mcp.server import run

    run()
