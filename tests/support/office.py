# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The smallest Office containers publication accepts as what their name claims.

Both a publication test and the attachment tool's test need a real ``.docx``: publication
recognises a zip by its first bytes and insists an Office document carry ``[Content_Types].xml``
and its own root part, so a fixture that is merely bytes named ``.docx`` is refused.
"""

import zipfile
from io import BytesIO

_MEMBERS = {
    "[Content_Types].xml": b'<?xml version="1.0"?><Types/>',
    "word/document.xml": b"<w:document/>",
}


def minimal_docx() -> bytes:
    """A Word document with nothing in it, and every part publication insists on."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in _MEMBERS.items():
            archive.writestr(name, content)
    return buffer.getvalue()
