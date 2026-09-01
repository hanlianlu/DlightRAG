# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""CLI bootstrap that loads configuration before constructing the MCP server."""

import argparse
import os
from pathlib import Path

from dlightrag.application.config import load_config, set_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DlightRAG MCP server",
        suggest_on_error=True,
    )
    parser.add_argument("--env-file", help="Path to .env configuration file")
    args = parser.parse_args()
    if args.env_file:
        # config.yaml is discovered from the current working directory, and MCP
        # hosts may launch this server from an arbitrary directory. The env-file
        # location is the configuration root: adopt it as the working directory
        # so config.yaml (and relative deployment paths) resolve consistently.
        env_file = Path(args.env_file).expanduser().resolve()
        os.chdir(env_file.parent)
        set_config(load_config(env_file))

    from dlightrag.adapters.mcp.server import run

    run()
