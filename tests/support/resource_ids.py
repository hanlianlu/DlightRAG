# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The resource ids the run-resource doubles of the unit and e2e suites both serve."""


def attachment_ordinal(resource_id: str) -> int | None:
    """Read the ordinal one accepted upload's resource id spells."""
    prefix = "attachment-"
    if not resource_id.startswith(prefix):
        return None
    try:
        return int(resource_id[len(prefix) :])
    except ValueError:
        return None
