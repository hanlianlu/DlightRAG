# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for request ID log record enrichment."""

from __future__ import annotations

import logging

from dlightrag.api.middleware import install_request_id_log_record_factory, request_id_var


def test_request_id_log_record_factory_adds_default_request_id() -> None:
    previous_factory = logging.getLogRecordFactory()
    try:
        install_request_id_log_record_factory()
        record = logging.getLogger("dlightrag.test").makeRecord(
            "dlightrag.test",
            logging.INFO,
            __file__,
            1,
            "message",
            (),
            None,
        )
        assert record.request_id == "-"
    finally:
        logging.setLogRecordFactory(previous_factory)


def test_request_id_log_record_factory_uses_active_request_id() -> None:
    previous_factory = logging.getLogRecordFactory()
    token = request_id_var.set("rid-test")
    try:
        install_request_id_log_record_factory()
        record = logging.getLogger("dlightrag.test").makeRecord(
            "dlightrag.test",
            logging.INFO,
            __file__,
            1,
            "message",
            (),
            None,
        )
        assert record.request_id == "rid-test"
    finally:
        request_id_var.reset(token)
        logging.setLogRecordFactory(previous_factory)
