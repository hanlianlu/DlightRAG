# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The tools the chart renderer's PNG path needs beyond node, and what a test does without them."""

import os
import shutil
from pathlib import Path

import pytest

FONT_DIR = Path(os.environ.get("ECHARTS_RENDER_FONT_DIR", "/usr/local/share/fonts/noto-sans-sc"))


def png_tools_missing() -> str | None:
    """Return what is missing for ``echarts-render`` to write a PNG, or None when it has it all."""
    if shutil.which("resvg") is None:
        return "resvg is not on PATH"
    if not (FONT_DIR / "NotoSansSC-Regular.otf").is_file():
        return f"Noto Sans SC is not in {FONT_DIR}"
    return None


def require_png_tools() -> None:
    """Skip a test that needs the PNG tools when a developer machine lacks them; fail it in CI.

    A gate that can silently not run is no gate, so CI, which installs the tools with
    scripts/install-chart-tools.sh, treats their absence as a failure.
    """
    missing = png_tools_missing()
    if missing is None:
        return
    message = f"{missing}; scripts/install-chart-tools.sh installs both"
    if os.environ.get("CI"):
        pytest.fail(message)
    pytest.skip(message)
