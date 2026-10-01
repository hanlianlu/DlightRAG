# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tool-message attachment projections for each native provider."""

import base64

from dlightrag.engine.ai.providers.anthropic_native import _anthropic_tool_messages
from dlightrag.engine.ai.providers.openai_compatible import _openai_tool_messages
from dlightrag.engine.ai.providers.openai_response import response_input

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


def _call(call_id: str) -> dict[str, object]:
    return {"id": call_id, "type": "function", "function": {"name": "read", "arguments": "{}"}}


def _tool_result(call_id: str, *, attachment: bool) -> dict[str, object]:
    message: dict[str, object] = {
        "role": "tool",
        "tool_call_id": call_id,
        "name": "read",
        "content": f"result for {call_id}",
    }
    if attachment:
        message["attachments"] = [
            {
                "resource_id": f"att_{call_id}",
                "safe_name": "page-one.png",
                "media_type": "image/png",
                "content_digest": "b" * 64,
                "size_bytes": 3,
                "data_url": DATA_URL,
            }
        ]
    return message


def _batch_violations(messages: list[dict[str, object]]) -> list[object]:
    """DeepSeek's rule: an assistant with tool_calls is answered by exactly those
    tool messages, contiguously, before any other role."""
    violations: list[object] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        tool_calls = message.get("tool_calls")
        calls = [call["id"] for call in (tool_calls if isinstance(tool_calls, list) else ())]
        if message.get("role") == "assistant" and calls:
            seen: list[object] = []
            cursor = index + 1
            while cursor < len(messages) and messages[cursor].get("role") == "tool":
                seen.append(messages[cursor].get("tool_call_id"))
                cursor += 1
            if seen != calls:
                following = messages[cursor].get("role") if cursor < len(messages) else None
                violations.append((index, calls, seen, following))
        index += 1
    return violations


def test_openai_compatible_rides_images_of_one_batch_in_one_user_message() -> None:
    converted = _openai_tool_messages(
        [
            {"role": "user", "content": "look"},
            {"role": "assistant", "content": "", "tool_calls": [_call("call-1"), _call("call-2")]},
            _tool_result("call-1", attachment=True),
            _tool_result("call-2", attachment=True),
            {"role": "assistant", "content": "next"},
        ]
    )

    assert _batch_violations(converted) == []
    assert [message["role"] for message in converted] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "user",
        "assistant",
    ]
    assert converted[4]["content"] == [
        {"type": "image_url", "image_url": {"url": DATA_URL}},
        {"type": "image_url", "image_url": {"url": DATA_URL}},
    ]


def test_openai_compatible_never_invents_a_user_turn_from_a_non_image_attachment() -> None:
    """A durable replay whose attachment bytes were not hydrated has nothing to
    send, so the tool result must not be duplicated into a user message."""
    unhydrated = _tool_result("call-1", attachment=False)
    unhydrated["attachments"] = [
        {
            "resource_id": "att_call-1",
            "safe_name": "page-one.png",
            "media_type": "image/png",
            "content_digest": "b" * 64,
            "size_bytes": 3,
        }
    ]

    converted = _openai_tool_messages(
        [
            {"role": "user", "content": "look"},
            {"role": "assistant", "content": "", "tool_calls": [_call("call-1")]},
            unhydrated,
        ]
    )

    assert [message["role"] for message in converted] == ["user", "assistant", "tool"]


def _viewed_question(*, hydrated: bool = True) -> dict[str, object]:
    """A question carrying the images its turn viewed, as Fast's conversation sends it."""
    (attachment,) = _tool_message()["attachments"]  # type: ignore[misc]
    if not hydrated:
        attachment = {key: value for key, value in attachment.items() if key != "data_url"}
    return {"role": "user", "content": "what does page 3 show?", "attachments": [attachment]}


def test_every_provider_shows_a_questions_attachments_as_its_named_images() -> None:
    """Each image follows its own name, as the tool result that took it printed it:
    two pages are otherwise two unnamed pictures the user seems to have sent."""
    question = "what does page 3 show?"
    assert _anthropic_tool_messages([_viewed_question()]) == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "text", "text": "[chart.png]"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": _PNG},
                },
            ],
        }
    ]
    assert _openai_tool_messages([_viewed_question()]) == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "text", "text": "[chart.png]"},
                {"type": "image_url", "image_url": {"url": DATA_URL}},
            ],
        }
    ]
    assert response_input([_viewed_question()]) == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": question},
                {"type": "input_text", "text": "[chart.png]"},
                {"type": "input_image", "image_url": DATA_URL},
            ],
        }
    ]


def test_an_attachment_without_bytes_leaves_the_question_as_written() -> None:
    """An attachment the Run has not hydrated has nothing to send, in any provider."""
    question = _viewed_question(hydrated=False)
    assert _anthropic_tool_messages([question]) == [
        {"role": "user", "content": "what does page 3 show?"}
    ]
    assert _openai_tool_messages([question]) == [
        {"role": "user", "content": "what does page 3 show?"}
    ]
    assert response_input([question]) == [{"role": "user", "content": "what does page 3 show?"}]
