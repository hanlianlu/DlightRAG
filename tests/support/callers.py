# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Who submits, for a test that needs a caller and no particular one.

``ANONYMOUS`` is the caller a deployment that tells no one apart gives every request. ``signed_in``
is one a verified token names, which is the only kind a Runtime Policy can tell from another.
"""

from dlightrag.application.access.principal import UserContext

ANONYMOUS = UserContext(user_id="anonymous", auth_mode="none")


def signed_in(email: str) -> UserContext:
    """A caller with a verified token whose ``email`` claim is the one given."""
    return UserContext(user_id=email, auth_mode="jwt", claims={"email": email})
