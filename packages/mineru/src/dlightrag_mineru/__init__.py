# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Manage the host-side MinerU sidecar DlightRAG parses through.

The application never imports MinerU's client from here: it reaches the running
service over HTTP. This distribution owns the host half — the dedicated service
virtualenv, the MinerU release range derived from the pinned LightRAG client, the
interpreter-startup guards, and the launchd/systemd units that keep it running
(ADR 0054).

Everything here is standard library and states no product vocabulary, so the tool
is usable by any host that wants a MinerU sidecar; the one piece of DlightRAG
knowledge — which reasoning dialect this deployment's title model speaks — is
resolved by the application and passed in.
"""

from __future__ import annotations

__all__ = ["__version__"]

try:
    from importlib.metadata import version as _version

    __version__ = _version("dlightrag-mineru")
except Exception:  # pragma: no cover - a source checkout without the wheel
    __version__ = "0.0.0"
