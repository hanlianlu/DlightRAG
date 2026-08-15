# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable-run lifecycle policy shared by Runtime and its store adapter."""

#: Terminal rows and every terminal run's event log expire 30 days after finish.
RUN_RETENTION_SECONDS = 30 * 24 * 3600
#: A worker renews this window while it holds a run; expiry makes the row reclaimable.
ANSWER_RUN_LEASE_SECONDS = 60
#: Consecutive expired-lease reclaims allowed without a committed checkpoint.
MAX_CONSECUTIVE_RECOVERIES = 4
RUN_ABANDONED_ERROR_KIND = "run_abandoned"

__all__ = [
    "ANSWER_RUN_LEASE_SECONDS",
    "MAX_CONSECUTIVE_RECOVERIES",
    "RUN_ABANDONED_ERROR_KIND",
    "RUN_RETENTION_SECONDS",
]
