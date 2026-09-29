# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Every PostgreSQL NOTIFY channel DlightRAG uses, declared once.

Stores NOTIFY and subscribe by these names only. A payload is a wake hint, never
authority: whoever it wakes re-reads the rows it names.
"""

# The Run's wake digest; wakes what waits on that Run's children, such as guidance.
RUN_ACTIVITY_CHANNEL = "dlightrag_run_activity"
# The Run's wake digest; wakes every worker's cancel-pending rescan.
RUN_CANCEL_CHANNEL = "dlightrag_run_cancel"
# The owner; wakes Connection refresh loops and in-flight dispatch watchers.
CONNECTIONS_CHANGED_CHANNEL = "dlightrag_connections_changed"
# The flow owner's worker id; wakes that worker's OAuth callback wait.
CONNECTION_OAUTH_CHANNEL = "dlightrag_connection_oauth"
# The published revision; wakes every process's model catalogue reload.
MODEL_CATALOGUE_CHANNEL = "dlightrag_model_catalogue_changed"

CHANNELS = (
    RUN_ACTIVITY_CHANNEL,
    RUN_CANCEL_CHANNEL,
    CONNECTIONS_CHANGED_CHANNEL,
    CONNECTION_OAUTH_CHANNEL,
    MODEL_CATALOGUE_CHANNEL,
)

__all__ = [
    "CHANNELS",
    "CONNECTIONS_CHANGED_CHANNEL",
    "CONNECTION_OAUTH_CHANNEL",
    "MODEL_CATALOGUE_CHANNEL",
    "RUN_ACTIVITY_CHANNEL",
    "RUN_CANCEL_CHANNEL",
]
