# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Startup vision capability probe for chat/answer models.

Sends one 1×1 pixel PNG image to the model and checks whether the
response is a valid text completion (``"ok"``) or an error.
The result is cached on ``provider.supports_vision``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Minimal 1×1 white PNG — base64-encoded so no filesystem dependency.
_ONE_PIXEL_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+"
    "hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
)
_ONE_PIXEL_DATA_URI = f"data:image/png;base64,{_ONE_PIXEL_PNG_B64}"


async def probe_vision_support(provider: Any, *, model: str) -> bool:
    """Probe whether *model* accepts ``image_url`` blocks.

    Sends a single-turn request with one 1×1 pixel image and expects
    the model to respond with ``"ok"``.  Returns ``True`` if the model
    handles images, ``False`` otherwise.

    The probe is idempotent — callers normally run it once at startup
    and cache the result on ``provider.supports_vision``.
    """
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Reply with exactly 'ok' and nothing else."},
                {"type": "image_url", "image_url": {"url": _ONE_PIXEL_DATA_URI}},
            ],
        }
    ]
    try:
        response = await provider.complete(messages, model=model, max_tokens=5, temperature=0)
        text = str(response).strip().lower()
        return "ok" in text
    except Exception:
        logger.debug("Vision probe failed for model %s", model, exc_info=True)
        return False


__all__ = ["probe_vision_support"]
