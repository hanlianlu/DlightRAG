# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Actual approved PDF/XLSX conversion + Registry/effects/recovery costs, offline.

Reuse the DOCX harness's format-neutral measurement operation, not its engine
control: production routing is unmodified. These are not Host/PG/service latency
measurements. Each invocation measures one fixture (first + five warm samples).
Optional non-Latin input must be a separately generated local synthetic PDF;
no font or PDF from the local-only probe is bundled with the project.
"""

import argparse
import asyncio
import hashlib
import json
import platform
import sys
from pathlib import Path

from scripts.anydoc_pilot import deny_network, peak_mib
from scripts.docx_integration_bench import measure
from scripts.format_route_fixtures import (
    PDF_FACTS,
    XLSX_ANCHORS,
    XLSX_FACTS,
    pdf_fixture,
    xlsx_fixture,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", choices=["multi", "scan", "mixed", "xlsx", "nonlatin"], required=True
    )
    parser.add_argument("--nonlatin-pdf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output == repo or repo in output.parents:
        parser.error("benchmark output must be outside repository")
    sys.addaudithook(deny_network)
    kind = args.fixture
    asset_digest = None
    if kind == "xlsx":
        data, png = xlsx_fixture()
        asset_digest = hashlib.sha256(png).hexdigest()
        facts = [fact.replace("\\|", "|") for fact in XLSX_FACTS] + ["$1,234.50", "0007"]
    elif kind == "nonlatin":
        if args.nonlatin_pdf is None:
            parser.error("nonlatin needs the explicitly supplied generated local PDF")
        data = args.nonlatin_pdf.read_bytes()
        facts = ["Привет мир 42.5", "Москва столица", "中文測試 7.5", "東京駅"]
    else:
        data = pdf_fixture(kind)
        facts = list(PDF_FACTS) if kind == "multi" else []
    fixture = {
        "name": "gold.xlsx" if kind == "xlsx" else "gold.pdf",
        "facts": facts,
        "patterns": [],
        "expected": "text" if kind not in {"scan", "mixed"} else "empty",
        "asset_occurrences": 3 if kind == "xlsx" else 0,
        "anchors": list(XLSX_ANCHORS) if kind == "xlsx" else [],
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    if kind == "xlsx":
        fixture["asset_digest"] = asset_digest

    async def run():
        samples = [await measure(fixture, data) for _ in range(6)]
        status = (
            "known_incomplete" if kind in {"scan", "mixed"} else "usable_text_unverified_coverage"
        )
        if any(s["status"] != status or s["converter"] != "firecrawl-anydoc" for s in samples):
            raise RuntimeError("integrated classification/route differs from gold")
        return samples

    samples = asyncio.run(run())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "platform": platform.platform(),
                "python": sys.version,
                "fixture": kind,
                "gold": fixture,
                "first": samples[0],
                "warm": samples[1:],
                "process_peak_mib": peak_mib(),
                "engine": "unmodified-production-route",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
