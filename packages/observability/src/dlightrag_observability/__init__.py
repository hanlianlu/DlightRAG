# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The privacy classifier, the OTLP export, and the ClickStack stack.

One rule decides what leaves the process, and it lives here so that the trace
seam, the log handler, and the browser integration answer it the same way
([ADR 0039](../../../../docs/adr/0039-observability-is-always-on-and-the-seam-masks-private-data.md)):
a **private** value is masked to its presence, length, and digest, an
**exportable** value is exported. There is no switch that turns monitoring off.
"""

from dlightrag_observability.privacy import PrivacyClass, classify, mask_payload

__all__ = [
    "PrivacyClass",
    "classify",
    "mask_payload",
]
