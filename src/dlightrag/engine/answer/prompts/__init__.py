# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Centralized prompts for DlightRAG: one module per consumer.

A prompt lives in the module named after the call that sends it; `identity.py` holds the
self-conception (identity and stance) every answer shares, and `answer.py` the evidence,
link, and citation rules Research takes from Fast.
"""
