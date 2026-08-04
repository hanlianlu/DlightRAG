# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Startup shim for the MinerU sidecar interpreter (and its spawned workers).

MinerU loads document scans with Pillow, whose default decompression-bomb guard
warns at ~89.5MP and raises ``DecompressionBombError`` at ~179MP. Large
multi-page composite scans (e.g. a ~205MP certificate) therefore fail to load
*before* MinerU can parse them, surfacing as::

    MinerU local parse failed: Failed to load file <name>: Image size (N pixels)
    exceeds limit of 178956970 pixels, could be decompression bomb DOS attack.

DlightRAG targets 32GB+ hosts and intentionally accepts large scans, so raise
the ceiling to match DlightRAG's own ``MAX_DECODE_IMAGE_PIXELS`` (250MP: no
warning up to 250MP; hard error only above 500MP). CPython's ``site`` machinery
imports this module automatically because ``scripts/mineru/api.sh`` puts this
directory on ``PYTHONPATH`` — which MinerU's spawned worker processes inherit.
A failure here cannot break the interpreter: ``site`` catches sitecustomize
errors, warns on stderr, and continues startup with Pillow's default ceiling.
"""

from PIL import Image

Image.MAX_IMAGE_PIXELS = 250_000_000
