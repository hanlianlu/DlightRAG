# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The tools the chart renderer's PNG path needs beyond node, and what a test does without them."""

import os
import shutil
from pathlib import Path

import pytest

FONT_DIR = Path(os.environ.get("FONTS_DIR", "/usr/local/share/fonts/dlightrag"))


def png_tools_missing() -> str | None:
    """Return what is missing for ``echarts-render`` to write a PNG, or None when it has it all."""
    if shutil.which("resvg") is None:
        return "resvg is not on PATH"
    if not (FONT_DIR / "NotoSansSC-Regular.otf").is_file():
        return f"the font set is not in {FONT_DIR}"
    return None


def require_png_tools() -> None:
    """Skip a test that needs the PNG tools when a developer machine lacks them; fail it in CI.

    A gate that can silently not run is no gate, so CI, which installs the tools with
    scripts/install-chart-tools.sh, treats their absence as a failure. `make fonts` installs the
    font set alone, on any platform.
    """
    missing = png_tools_missing()
    if missing is None:
        return
    message = f"{missing}; make fonts installs the font set, scripts/install-chart-tools.sh both"
    if os.environ.get("CI"):
        pytest.fail(message)
    pytest.skip(message)


def require_echarts_library() -> None:
    """Skip a test that builds a report page when chart-render has no ECharts build; fail it in CI.

    The page build inlines the ECharts library, which a checkout gets from `npm ci` in chart-render
    (`make chart-render-install`). Like the PNG tools, its absence is a failure in CI.
    """
    from echarts_render import RenderError, echarts_library

    try:
        echarts_library()
    except RenderError as error:
        if os.environ.get("CI"):
            pytest.fail(str(error))
        pytest.skip(str(error))
