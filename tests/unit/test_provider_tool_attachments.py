# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tool-message attachment projections for each native provider."""

import base64
import json

from dlightrag.ai.providers.anthropic_native import _anthropic_tool_messages
from dlightrag.ai.providers.gemini_native import _gemini_tool_contents
from dlightrag.ai.providers.openai_compatible import _openai_tool_messages

_PNG = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()
DATA_URL = f"data:image/png;base64,{_PNG}"


def _tool_message() -> dict[str, object]:
    return {
        "role": "tool",
        "tool_call_id": "call-1",
        "name": "read",
        "content": "image attachment: chart.png",
        "attachments": [
            {
                "resource_id": "att_1",
                "safe_name": "chart.png",
                "media_type": "image/png",
                "content_digest": "a" * 64,
                "size_bytes": 15,
                "data_url": DATA_URL,
            }
        ],
        "is_error": False,
    }


def test_anthropic_inlines_the_attachment_in_the_tool_result() -> None:
    converted = _anthropic_tool_messages([{"role": "user", "content": "look"}, _tool_message()])

    assert converted[-1]["role"] == "user"
    (block,) = converted[-1]["content"]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "call-1"
    content = block["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "image attachment: chart.png"}
    assert content[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": _PNG},
    }


def test_gemini_puts_the_image_in_the_same_user_turn() -> None:
    contents = _gemini_tool_contents([{"role": "user", "content": "look"}, _tool_message()])

    assert contents[-1]["role"] == "user"
    parts = contents[-1]["parts"]
    assert parts[0]["function_response"]["name"] == "read"
    assert parts[0]["function_response"]["response"]["output"] == "image attachment: chart.png"
    assert parts[1] == {"inline_data": {"mime_type": "image/png", "data": _PNG}}


def test_openai_compatible_appends_untrusted_multimodal_user_message() -> None:
    converted = _openai_tool_messages([{"role": "user", "content": "look"}, _tool_message()])

    assert converted[1]["role"] == "tool"
    assert "attachments" not in converted[1]
    follow_up = converted[2]
    assert follow_up["role"] == "user"
    assert follow_up["untrusted_tool_data"] is True
    parts = follow_up["content"]
    assert parts[0] == {"type": "text", "text": "image attachment: chart.png"}
    assert parts[1] == {"type": "image_url", "image_url": {"url": DATA_URL}}


def test_plain_tool_message_projects_without_any_user_turn() -> None:
    plain = {
        "role": "tool",
        "tool_call_id": "call-1",
        "name": "grep",
        "content": "matches",
        "is_error": False,
    }
    assert json.dumps(_openai_tool_messages([plain])) is not None
    converted = _openai_tool_messages([plain])
    assert len(converted) == 1
    assert converted[0]["role"] == "tool"
