# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Which uploaded attachments an Answer Run can read, decided once.

A Run reads an upload one of three ways: an image it views, a document one of its
converters turns into text (``converters.py``), or text it decodes directly
(``text.py``). Every transport admits uploads by the one rule here, and the Web
composer offers the types listed here.

A link is not an upload: what it serves is known only once it is fetched, and
reading it follows the public Web acquisition contract, so this rule never refuses
a link.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from dlightrag.engine.ai.media import verify_web_image_bytes
from dlightrag.engine.answer.errors import UnsupportedAttachmentTypeError
from dlightrag.engine.answer.mode import resource_role
from dlightrag.engine.answer.resources.converters import CONVERTED_EXTENSIONS, conversion_format
from dlightrag.engine.answer.resources.models import ResourceDecodeError
from dlightrag.engine.answer.resources.text import declared_charset, decode_text

#: Document types a Run decodes as text, admitted by their file extension alone. The
#: bytes behind one must still decode; a binary file under a text extension is
#: refused when it is read, as any undecodable text is.
TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {
        "conf",
        "css",
        "ini",
        "js",
        "json",
        "log",
        "md",
        "properties",
        "py",
        "rtf",
        "scss",
        "sh",
        "sql",
        "tex",
        "ts",
        "txt",
        "xml",
        "yaml",
        "yml",
    }
)

#: The document types admitted by their file extension, sorted: what a converter
#: turns into text and the listed text types. The Web composer offers these.
READABLE_DOCUMENT_EXTENSIONS: tuple[str, ...] = tuple(
    sorted(CONVERTED_EXTENSIONS | TEXT_EXTENSIONS)
)

_EXTENSION = re.compile(r"[a-z0-9]{1,16}")
_MEDIA_TYPE = re.compile(r"[a-z0-9][a-z0-9!#$&^_.+-]{0,63}/[a-z0-9][a-z0-9!#$&^_.+-]{0,63}")


def require_readable_attachment(
    filename: str | None, declared_mime: str | None, content: bytes
) -> None:
    """Refuse an upload no Run could read, naming its type.

    Its type admits it first: an image (verified as one before acceptance), a file a
    conversion route takes, a listed text type, or a ``text/*`` media type. Any
    other upload is decided by its bytes, as a Run would read them: bytes that
    verify as an image or decode as text are admitted, so a source file sent as
    ``application/octet-stream`` is, and a packaged format that does neither, such
    as a zip archive, is refused. Deciding by the bytes decodes the whole upload,
    so a caller on an event loop runs this in a thread.
    """
    extension = PurePosixPath(filename or "").suffix.lower().removeprefix(".")
    media_type = (declared_mime or "").split(";", 1)[0].strip().lower()
    if (
        resource_role(filename=filename, mime_type=declared_mime) == "image"
        or conversion_format(filename, declared_mime) is not None
        or extension in TEXT_EXTENSIONS
        or media_type.startswith("text/")
        or _readable_bytes(content, declared_mime)
    ):
        return
    raise UnsupportedAttachmentTypeError(
        _type_label(extension, media_type), READABLE_DOCUMENT_EXTENSIONS
    )


def _readable_bytes(content: bytes, declared_mime: str | None) -> bool:
    """Whether a Run reading these bytes finds an image or decodable text."""
    try:
        verify_web_image_bytes(content)
    except ValueError:
        pass
    else:
        return True
    try:
        decode_text(content, declared_charset=declared_charset(declared_mime))
    except ResourceDecodeError:
        return False
    return True


def _type_label(extension: str, media_type: str) -> str | None:
    """Name an upload's type without echoing an arbitrary caller-supplied string."""
    if _EXTENSION.fullmatch(extension):
        return f".{extension}"
    if _MEDIA_TYPE.fullmatch(media_type):
        return media_type
    return None


__all__ = [
    "READABLE_DOCUMENT_EXTENSIONS",
    "TEXT_EXTENSIONS",
    "require_readable_attachment",
]
