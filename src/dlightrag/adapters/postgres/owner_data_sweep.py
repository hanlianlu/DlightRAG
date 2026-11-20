# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Removing handover packages past their expiry (ADR 0049).

An export writes a copy of a person's life in cleartext to a directory an
operator hands over. The copy needs a bounded life, and this is the bound: a
timer runs `dlightrag-owner-data sweep-expired`, which reads each package's own
deadline and removes the ones past it.

The rules are narrow on purpose. A package is one this command wrote — a
directory carrying a manifest that says `export`. Anything else under the root is
left alone, because a delivery root may be shared and a sweep is not a licence to
delete a stranger's directory. A deadline that cannot be read is not a licence
either: a malformed or half-written manifest stays, and the next sweep looks
again. A timestamp without a zone is read as UTC — the export writes UTC — so a
hand-edited file is judged rather than ignored. A removal that fails is reported rather than
swallowed, because a package outliving its expiry is exactly the cleartext copy
this exists to bound.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SweptPackage:
    """One package the sweep judged, and what it did about it."""

    path: Path
    request_id: str
    owner_id: str
    expires_at: datetime
    removed: bool


def _deadline(package: Path) -> tuple[datetime, str, str] | None:
    """The package's own deadline, or None when this is not a package we wrote."""
    manifest = package / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or str(raw.get("operation")) != "export":
            return None
        deadline = datetime.fromisoformat(str(raw["expires_at"]))
        request_id = str(raw.get("request_id") or package.name)
        owner_id = str(raw.get("owner_id") or "?")
    except ValueError, KeyError, OSError, TypeError:
        # Unreadable is not expired. It stays, and it is said out loud.
        logger.warning("Package %s states no readable deadline; left in place", package)
        return None
    if deadline.tzinfo is None:  # a hand-written file may omit the zone
        deadline = deadline.replace(tzinfo=UTC)
    return deadline, request_id, owner_id


def sweep_expired_exports(
    out_dir: Path, *, now: datetime | None = None, dry_run: bool = False
) -> list[SweptPackage]:
    """Judge every package under ``out_dir``, removing the ones past their expiry."""
    moment = now or datetime.now(UTC)
    swept: list[SweptPackage] = []
    if not out_dir.is_dir():
        return swept
    for package in sorted(out_dir.iterdir()):
        if not package.is_dir():
            continue
        judged = _deadline(package)
        if judged is None:
            continue
        deadline, request_id, owner_id = judged
        if deadline > moment:
            continue
        if dry_run:
            swept.append(SweptPackage(package, request_id, owner_id, deadline, removed=False))
            continue
        shutil.rmtree(package, ignore_errors=True)
        if package.exists():
            logger.warning("Package %s outlived its expiry: removal failed", package)
            swept.append(SweptPackage(package, request_id, owner_id, deadline, removed=False))
            continue
        swept.append(SweptPackage(package, request_id, owner_id, deadline, removed=True))
    return swept


__all__ = ["SweptPackage", "sweep_expired_exports"]
