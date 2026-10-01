# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Web localizes only error kinds the server can send.

`frontend/lib/run-errors.ts` keys localized copy by the error kind a Run stores
or an Answer submission is rejected with, with the server message as its English
source. A key the server never sends is dead copy, and a case slip silently
falls back to English, so the keys and their English sources are locked to the
server vocabulary here.
"""

import re
from pathlib import Path

from dlightrag.engine.answer import errors as answer_errors
from dlightrag.engine.answer import image_capability
from dlightrag.engine.answer.errors import (
    AnswerResourceAdmissionError,
    InvalidToolConfigurationError,
    UnsupportedResourceCapabilityError,
)

_RUN_ERRORS = Path(__file__).resolve().parents[2] / "frontend/lib/run-errors.ts"


def _frontend_copy() -> dict[str, str]:
    source = _RUN_ERRORS.read_text(encoding="utf-8")
    match = re.search(r"RUN_ERROR_KIND_COPY[^{]*\{(.*?)\n\};", source, re.S)
    assert match is not None, "RUN_ERROR_KIND_COPY catalog missing"
    keys = re.findall(r"^\s*(\w+):", match.group(1), re.M)
    entries = re.findall(r"(\w+):((?:\s*\+?\s*'[^']*')+),", match.group(1))
    copy = {key: "".join(re.findall(r"'([^']*)'", value)) for key, value in entries}
    assert keys and sorted(copy) == sorted(keys), "an entry's copy could not be parsed"
    return copy


def _server_kinds() -> set[str]:
    return {
        value
        for name, value in vars(answer_errors).items()
        if name in answer_errors.__all__
        and name.isupper()
        and not name.endswith("_MESSAGE")
        and isinstance(value, str)
    }


def test_every_localized_kind_is_one_the_server_sends() -> None:
    unknown = set(_frontend_copy()) - _server_kinds()

    assert unknown == set()


def test_localized_sources_are_the_server_messages_verbatim() -> None:
    copy = _frontend_copy()
    expected = {
        answer_errors.UNSUPPORTED_RESOURCE_CAPABILITY: (
            UnsupportedResourceCapabilityError().public_message
        ),
        answer_errors.ANSWER_RESOURCE_INVALID: AnswerResourceAdmissionError().public_message,
        answer_errors.INVALID_TOOL_CONFIGURATION: InvalidToolConfigurationError(
            ("read",)
        ).public_message,
        # The image kinds' public messages carry a bracketed marker the copy omits.
        answer_errors.ANSWER_IMAGE_CAPABILITY_UNKNOWN: image_capability._ERROR_CAPABILITY_UNKNOWN,
        answer_errors.CURRENT_IMAGES_UNSUPPORTED: image_capability._ERROR_IMAGES_NOT_SUPPORTED,
    }

    assert set(copy) == set(expected)
    assert {kind: copy.get(kind) for kind in expected} == expected
