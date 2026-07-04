# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Structured-output contracts for small LLM planning tasks."""

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from dlightrag.utils.text import extract_json


def _strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return an OpenAI strict-mode friendly JSON schema copy."""
    normalized = json.loads(json.dumps(schema))

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                properties = node.get("properties")
                if isinstance(properties, dict):
                    node["required"] = list(properties)
                    node.setdefault("additionalProperties", False)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(normalized)
    return normalized


@dataclass(frozen=True)
class StructuredOutput:
    """Pydantic-backed structured-output request and validation contract."""

    name: str
    schema: type[BaseModel]
    strict: bool = True

    def response_format_for_provider(self, provider: str) -> dict[str, Any]:
        """Return the response_format shape supported by the configured provider."""
        if provider == "openai":
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": self.name,
                    "schema": _strict_json_schema(self.schema.model_json_schema()),
                    "strict": self.strict,
                },
            }
        return {"type": "json_object"}

    def parse(self, raw: Any) -> BaseModel:
        """Validate raw model output against the contract."""
        if isinstance(raw, self.schema):
            return raw
        if isinstance(raw, BaseModel):
            return self.schema.model_validate(raw.model_dump())
        if isinstance(raw, dict):
            return self.schema.model_validate(raw)
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("empty structured output")
        return self.schema.model_validate_json(extract_json(raw))
