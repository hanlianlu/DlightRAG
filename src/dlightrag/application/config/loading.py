# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Configuration loading and process-local configuration state."""

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .sections import DlightragConfig

# Singleton for standalone mode (MCP/API server)
_config: DlightragConfig | None = None


def load_config(env_file: str | Path | None = None, **overrides: Any) -> DlightragConfig:
    """Build config from an optional .env file without globally loading dotenv."""
    try:
        if env_file is not None:
            return DlightragConfig(_env_file=env_file, **overrides)  # type: ignore[call-arg]
        return DlightragConfig(**overrides)
    except ValidationError as exc:
        # Pydantic echoes the rejected input, which here is the settings mapping
        # holding API keys. Field locations alone say what to fix.
        detail = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors(include_input=False, include_url=False)
        )
    # Raised outside the handler so the original error is not chained onto it.
    raise ValueError(f"Invalid dlightrag configuration: {detail}")


def get_config() -> DlightragConfig:
    """Get global dlightrag configuration (singleton).

    For standalone use (MCP/API server). When used as a library,
    construct DlightragConfig directly and pass it to WorkspaceRag.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: DlightragConfig) -> None:
    """Set the global config singleton. Useful for testing."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global config singleton. Useful for testing."""
    global _config
    _config = None
