# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Abstract base for multimodal embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbedProvider(ABC):
    """Strategy for multimodal embedding API protocols.

    Inputs to build_payload are text strings or ``data:image/...;base64,...`` URIs.
    """

    @property
    @abstractmethod
    def endpoint(self) -> str:
        """API endpoint path relative to base_url."""

    @abstractmethod
    def build_payload(self, model: str, inputs: list[str]) -> dict:
        """Build request payload for the embedding API."""

    @property
    def max_images_per_request(self) -> int:
        """Hard API limit on images per request."""
        return 128

    def parse_response(self, data: dict) -> list[list[float]]:
        """Extract vectors from JSON response."""
        return [item["embedding"] for item in data["data"]]
