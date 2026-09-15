# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tool-message attachment projections for each native provider."""

import base64
import json

from dlightrag.engine.ai.providers.anthropic_native import _anthropic_tool_messages
from dlightrag.engine.ai.providers.gemini_native import _gemini_tool_contents
from dlightrag.engine.ai.providers.openai_compatible import _openai_tool_messages

_PNG = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()
DATA_URL = f"data:image/png;base64,{_PNG}"
_PAGE_ONE = b"\x89PNG\r\n\x1a\npage-one"
_PAGE_TWO = b"\x89PNG\r\n\x1a\npage-two"
_PAGE_ONE_URL = f"data:image/png;base64,{base64.b64encode(_PAGE_ONE).decode()}"
_PAGE_TWO_URL = f"data:image/png;base64,{base64.b64encode(_PAGE_TWO).decode()}"


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


def test_anthropic_serializes_every_attachment_in_declared_order() -> None:
    message = _tool_message()
    message["attachments"] = [
        {
            "resource_id": "att_1",
            "safe_name": "page-1.png",
            "media_type": "image/png",
            "content_digest": "a" * 64,
            "size_bytes": len(_PAGE_ONE),
            "data_url": _PAGE_ONE_URL,
        },
        {
            "resource_id": "att_2",
            "safe_name": "page-2.png",
            "media_type": "image/png",
            "content_digest": "b" * 64,
            "size_bytes": len(_PAGE_TWO),
            "data_url": _PAGE_TWO_URL,
        },
    ]
    converted = _anthropic_tool_messages([{"role": "user", "content": "look"}, message])

    (block,) = converted[-1]["content"]
    content = block["content"]
    assert content[0] == {"type": "text", "text": "image attachment: chart.png"}
    assert content[1:] == [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(_PAGE_ONE).decode(),
            },
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(_PAGE_TWO).decode(),
            },
        },
    ]


def test_gemini_carries_the_image_beside_its_function_response() -> None:
    contents = _gemini_tool_contents([{"role": "user", "content": "look"}, _tool_message()])

    assert contents[0] == {"role": "user", "parts": [{"text": "look"}]}
    assert contents[-1]["role"] == "user"
    parts = contents[-1]["parts"]
    assert parts[0]["function_response"]["name"] == "read"
    assert parts[0]["function_response"]["response"]["output"] == "image attachment: chart.png"
    # Pixels stay top-level: google-genai serializes a nested FunctionResponsePart
    # with Python field names, which the REST API cannot bind to the response.
    assert "parts" not in parts[0]["function_response"]
    assert parts[1] == {
        "inline_data": {
            "mime_type": "image/png",
            "data": base64.b64decode(_PNG),
        }
    }


def test_gemini_answers_one_tool_batch_in_one_turn() -> None:
    """Every result of a model turn answers it in the same user turn.

    The API matches one functionResponse per functionCall, so a turn per tool
    message leaves the batch unmatched and strands the trailing result behind an
    extra user turn.
    """
    second = {
        "role": "tool",
        "tool_call_id": "call-2",
        "name": "read",
        "content": "no attachment here",
        "is_error": False,
    }
    contents = _gemini_tool_contents(
        [
            {"role": "user", "content": "look"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call-1", "function": {"name": "read", "arguments": "{}"}},
                    {"id": "call-2", "function": {"name": "read", "arguments": "{}"}},
                ],
            },
            _tool_message(),
            second,
        ]
    )

    assert [content["role"] for content in contents] == ["user", "model", "user"]
    answers = contents[-1]["parts"]
    responses = [part["function_response"] for part in answers if "function_response" in part]
    assert [response["id"] for response in responses] == ["call-1", "call-2"]
    assert [part for part in answers if "inline_data" in part] == [
        {"inline_data": {"mime_type": "image/png", "data": base64.b64decode(_PNG)}}
    ]


def test_gemini_keeps_two_call_batches_in_separate_turns() -> None:
    """A later user turn closes the batch, so the next batch answers its own turn."""
    contents = _gemini_tool_contents(
        [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1"}]},
            {"role": "tool", "tool_call_id": "call-1", "name": "read", "content": "one"},
            {"role": "user", "content": "again"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call-2"}]},
            {"role": "tool", "tool_call_id": "call-2", "name": "read", "content": "two"},
        ]
    )

    assert [content["role"] for content in contents] == [
        "model",
        "user",
        "user",
        "model",
        "user",
    ]
    assert contents[1]["parts"][0]["function_response"]["id"] == "call-1"
    assert contents[2]["parts"] == [{"text": "again"}]
    assert contents[4]["parts"][0]["function_response"]["id"] == "call-2"


def test_openai_compatible_appends_untrusted_multimodal_user_message() -> None:
    converted = _openai_tool_messages([{"role": "user", "content": "look"}, _tool_message()])

    assert converted[1]["role"] == "tool"
    assert "attachments" not in converted[1]
    follow_up = converted[2]
    assert follow_up["role"] == "user"
    assert "untrusted_tool_data" not in follow_up
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
