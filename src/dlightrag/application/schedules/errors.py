# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What the Schedule service tells a caller it refused, in the Application's error families."""

from __future__ import annotations

from dlightrag.application.errors import ApplicationConflictError, ApplicationInputError


class ScheduleInputError(ApplicationInputError):
    """A Schedule the owner must change before it can be saved."""


class ScheduleStateError(ApplicationConflictError):
    """The Schedule is in a state that does not allow this change."""


class ScheduleLimitError(ApplicationConflictError):
    """The owner already holds as many active Schedules as the deployment allows."""

    def __init__(self, limit: int) -> None:
        super().__init__(f"an owner may hold at most {limit} active Schedules")
        self.limit = limit


__all__ = ["ScheduleInputError", "ScheduleLimitError", "ScheduleStateError"]
