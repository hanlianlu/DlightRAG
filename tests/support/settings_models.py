# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Walk the settings model graph that hangs off the root configuration."""

from collections.abc import Mapping
from typing import Any, get_args, get_origin

from pydantic import BaseModel

from dlightrag.application.config import DlightragConfig


def model_types(annotation: Any) -> list[type[BaseModel]]:
    """Every settings model an annotation names, through unions and containers."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    return [model for arg in get_args(annotation) for model in model_types(arg)]


def settings_models(root: type[BaseModel] = DlightragConfig) -> set[type[BaseModel]]:
    """Every settings model reachable from ``root``, including ``root``."""
    seen: set[type[BaseModel]] = set()
    pending = [root]
    while pending:
        model = pending.pop()
        if model not in seen:
            seen.add(model)
            for field in model.model_fields.values():
                pending.extend(model_types(field.annotation))
    return seen


def _is_mapping(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if isinstance(origin, type) and issubclass(origin, Mapping):
        return True
    return any(_is_mapping(arg) for arg in get_args(annotation))


def names_a_config_field(name: str) -> bool:
    """Whether ``DLIGHTRAG_A__B__C`` reaches a configuration field or a mapping entry."""
    model: type[BaseModel] = DlightragConfig
    parts = name.removeprefix("DLIGHTRAG_").lower().split("__")
    for index, part in enumerate(parts):
        field = model.model_fields.get(part)
        if field is None:
            return False
        if index == len(parts) - 1 or _is_mapping(field.annotation):
            return True
        nested = model_types(field.annotation)
        if not nested:
            return False
        model = nested[0]
    return False
