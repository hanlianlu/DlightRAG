from __future__ import annotations

from typing import Any

from dlightrag.config import DlightragConfig


def clone_config(config: DlightragConfig) -> DlightragConfig:
    """Revalidate a frozen config into an independent test instance."""
    return DlightragConfig.model_validate(config.model_dump())


def replace_config(config: DlightragConfig, path: str, value: Any) -> DlightragConfig:
    """Test-only setup for collaborators already holding the config."""
    mutate_config(config, path, value)
    return config


def mutate_config(config: DlightragConfig, path: str, value: Any) -> None:
    """Test-only in-place setup for collaborators already holding the config."""
    parts = path.split(".")
    owner: Any = config
    for name in parts[:-1]:
        owner = getattr(owner, name)
    object.__setattr__(owner, parts[-1], value)
