# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Offline full DOCX adapter/read measurements; outputs must be outside the repo.

Integrated uses unmodified production routing (Markdown + structured parse +
asset verification/audit). Incumbent is a counterfactual control: only the DOCX
candidate call is replaced by the retained MarkItDown adapter, including its
asset audit. Neither is a live service/Host/PG latency measurement.
"""

import argparse
import asyncio
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

from scripts.anydoc_pilot import deny_network, evaluate, peak_mib


async def measure(fixture: dict[str, Any], data: bytes) -> dict[str, Any]:
    from dlightrag.engine.answer.resources.converters import convert_resource
    from dlightrag.engine.answer.resources.models import ResourceInput
    from dlightrag.engine.answer.resources.registry import ResourceRegistry
    from dlightrag.engine.answer.resources.snapshots import ConversionSnapshot

    name = fixture["name"]
    start = time.perf_counter()
    converted = await convert_resource(data, filename=name, declared_mime=None)
    conversion_ms = (time.perf_counter() - start) * 1000
    async with ResourceRegistry(
        resource_secret=b"bench", cursor_secret=b"bench-cursor"
    ) as registry:
        start = time.perf_counter()
        resource = registry.register(ResourceInput(filename=name, content=data))
        first = await registry.read(resource, max_window_tokens=4000)
        effects = registry.conversion_effects(resource)
        first_read_effect_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        cached = await registry.read(resource, max_window_tokens=4000)
        cache_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        snapshot = ConversionSnapshot.restore(
            effects[-1].content, {e.resource_id: e.content for e in effects[:-1]}
        )
        async with ResourceRegistry(
            resource_secret=b"bench", cursor_secret=b"bench-cursor"
        ) as restored:
            restored.register(ResourceInput(filename=name, content=data))
            restored.adopt_conversion_snapshot(snapshot)
            replay = await restored.read(resource, max_window_tokens=4000)
            for asset in snapshot.visuals:
                if (await restored.visual_asset(resource, asset.handle_id)).data != asset.data:
                    raise RuntimeError("restored asset differs")
        restore_read_ms = (time.perf_counter() - start) * 1000
    if not first == cached == replay:
        raise RuntimeError("cache/recovery view differs")
    pure_quality = evaluate(
        fixture,
        {
            "text": converted.text,
            "assets": [
                {"sha256": hashlib.sha256(v.data).hexdigest(), "anchor": v.anchor}
                for v in converted.visuals
            ],
            "visual_references": len(converted.visuals),
        },
    )
    if not pure_quality["passed"]:
        raise RuntimeError(f"full conversion gold failed: {pure_quality}")
    result = {
        "text": snapshot.text,
        "assets": [
            {
                "sha256": hashlib.sha256(v.data).hexdigest(),
                "anchor": v.anchor,
                "origin_part": v.origin_part,
            }
            for v in snapshot.visuals
        ],
        "visual_references": len(first.visual_handles),
    }
    quality = evaluate(fixture, result)
    if not quality["passed"]:
        raise RuntimeError(f"integrated source-authored gold failed: {quality}")
    return {
        "conversion_ms": conversion_ms,
        "first_read_effect_ms": first_read_effect_ms,
        "cache_ms": cache_ms,
        "restore_read_ms": restore_read_ms,
        "converter": snapshot.converter,
        "version": snapshot.converter_version,
        "status": snapshot.extraction_status,
        "quality": quality,
        "result": result,
        "effect_count": len(effects),
        "process_peak_mib": peak_mib(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--engine", choices=["integrated", "incumbent"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output == repo or repo in output.parents:
        parser.error("measurement output must be outside repository")
    sys.addaudithook(deny_network)
    start = time.perf_counter()
    from dlightrag.engine.answer.resources import converters

    import_ms = (time.perf_counter() - start) * 1000
    if args.engine == "incumbent":

        def control(content, route, budget):
            budget.check()
            result = converters._convert_markitdown(content, route)
            budget.check()
            return result

        converters._convert_anydoc = control
    gold = json.loads((args.fixtures / "gold.json").read_text())
    fixture = next(g for g in gold if g["name"] == args.fixture)
    if not fixture["name"].endswith(".docx") or fixture["expected"] not in {
        "text",
        "rich",
        "empty",
    }:
        parser.error("only qualified DOCX cases may be timed")
    data = (args.fixtures / fixture["name"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != fixture["sha256"]:
        raise RuntimeError("fixture differs from independent gold")
    baseline = peak_mib()

    async def run():
        first = await measure(fixture, data)
        warm = [await measure(fixture, data) for _ in range(5)]
        return first, warm

    first, warm = asyncio.run(run())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "platform": platform.platform(),
                "python": sys.version,
                "engine": args.engine,
                "fixture": fixture,
                "import_ms": import_ms,
                "baseline_peak_mib": baseline,
                "first_operation_after_support_import": first,
                "warm": warm,
                "process_peak_mib": peak_mib(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
