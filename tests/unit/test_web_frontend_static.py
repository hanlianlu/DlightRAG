# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Static regression checks for browser-side modules."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_chat_module_imports_close_panel_before_using_it() -> None:
    chat_js = (ROOT / "src/dlightrag/web/static/js/chat.js").read_text(encoding="utf-8")

    assert "closePanel();" in chat_js
    assert "import {closePanel} from './panel.js';" in chat_js


def test_chat_streaming_is_split_into_transport_and_renderer_modules() -> None:
    static_js = ROOT / "src/dlightrag/web/static/js"
    chat_js = (static_js / "chat.js").read_text(encoding="utf-8")

    assert (static_js / "sse.js").is_file()
    assert (static_js / "chat_renderer.js").is_file()
    assert "from './sse.js'" in chat_js
    assert "from './chat_renderer.js'" in chat_js
    assert "response.body.getReader()" not in chat_js
    assert "contentDiv.innerHTML" not in chat_js


def test_htmx_behavior_lives_in_dedicated_module() -> None:
    static_js = ROOT / "src/dlightrag/web/static/js"
    main_js = (static_js / "main.js").read_text(encoding="utf-8")
    panel_js = (static_js / "panel.js").read_text(encoding="utf-8")

    assert (static_js / "htmx.js").is_file()
    assert "setupHtmxInteractions" in main_js
    assert "htmx:afterRequest" not in panel_js


def test_unused_ingest_progress_frontend_contract_is_removed() -> None:
    web_root = ROOT / "src/dlightrag/web"
    checked = [
        web_root / "routes" / "files.py",
        *(web_root / "static" / "js").glob("*.js"),
        *(web_root / "templates").rglob("*.html"),
    ]

    offenders = [path for path in checked if "ingest/progress" in path.read_text(encoding="utf-8")]
    assert offenders == []
