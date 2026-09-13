# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Offline paired pilot, NOT a production router or dependency installer.

Generate with the project Python; run candidate in its isolated 3.14 venv.
See --help. Output is confined to a caller-supplied scratch directory. Python
socket access is denied before converter imports; AnyDoc always uses OCR reject.
This guard is not an OS sandbox for arbitrary native binaries.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import importlib.metadata
import io
import json
import platform
import re
import resource
import sys
import time
import zipfile
from pathlib import Path
from typing import Any


def deny_network(event: str, args: tuple[Any, ...]) -> None:
    if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.sendto"}:
        raise RuntimeError(f"offline pilot denied {event}")


def peak_mib() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


def evaluate(gold: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Facts and relational regexes are source-authored, never incumbent-derived.

    Known-incomplete/error cases do not qualify merely because text is nonempty.
    Assets are compared by occurrence and bytes, not by unique digest count.
    """
    text = result.get("text", "").replace("\\|", "|")
    missing = [fact for fact in gold["facts"] if fact not in text]
    patterns = [pattern for pattern in gold["patterns"] if not re.search(pattern, text)]
    assets = result.get("assets", [])
    asset_match = len(assets) == gold["asset_occurrences"]
    if gold.get("asset_digest"):
        asset_match = asset_match and all(a["sha256"] == gold["asset_digest"] for a in assets)
    anchors_match = all(anchor in [a.get("anchor") for a in assets] for anchor in gold["anchors"])
    visual_references_match = result.get("visual_references", 0) >= gold["asset_occurrences"]
    expected = gold["expected"]
    error = result.get("error")
    if expected == "host_safety_refusal":
        passed = error == "UnsafeArchiveError" and result.get("converter_calls") == 0
    elif expected == "malformed":
        passed = error in {
            "MalformedError",
            "MissingPartError",
            "FileConversionException",
            "ResourceConversionError",
        }
    elif expected == "ocr":
        passed = (
            error == "NeedsOcrError"
            and result.get("pages") == gold["ocr_pages"]
            and result.get("page_count") == gold["pages"]
        )
    elif expected == "incomplete":
        passed = error == "NeedsOcrError"
    elif expected == "empty":
        passed = not error and not text.strip()
    else:
        passed = not error and not missing and not patterns and asset_match and anchors_match
        if expected == "rich":
            passed = passed and visual_references_match
    return {
        "passed": bool(passed),
        "missing_facts": missing,
        "missing_patterns": patterns,
        "asset_occurrences_match": asset_match,
        "anchors_match": anchors_match,
        "visual_references_match": visual_references_match,
    }


def candidate_assets(document: Any) -> list[dict[str, Any]]:
    assets = {a.id: a for a in document.assets}
    occurrences = []

    def visit(node: Any) -> None:
        if getattr(node, "kind", None) == "image":
            source = node.source
            if source.kind == "asset":
                asset = assets[source.asset_id]
                occurrences.append(
                    {
                        "id": asset.id,
                        "origin_part": asset.origin_part,
                        "anchor": node.anchor,
                        "media_type": asset.media_type,
                        "sha256": hashlib.sha256(asset.data).hexdigest(),
                        "bytes": len(asset.data),
                    }
                )
        for name in ("blocks", "content", "items", "rows", "cells"):
            children = getattr(node, name, None)
            if isinstance(children, list):
                for child in children:
                    if isinstance(child, list):
                        for nested in child:
                            visit(nested)
                    else:
                        visit(child)
        for name in ("list", "table"):
            child = getattr(node, name, None)
            if child is not None:
                visit(child)

    for block in document.blocks:
        visit(block)
    return occurrences


class Converter:
    def __init__(self, engine: str, support_site: Path | None) -> None:
        self.engine = engine
        start = time.perf_counter()
        self.candidate: Any = None
        if engine == "anydoc":
            # Import before adding incumbent site packages: never import unrelated anydoc.
            if importlib.metadata.version("firecrawl-anydoc") != "0.2.4":
                raise RuntimeError("exact firecrawl-anydoc 0.2.4 required")
            self.candidate = importlib.import_module("anydoc")
        self.engine_import_ms = (time.perf_counter() - start) * 1000
        if support_site:
            sys.path.append(str(support_site.resolve()))
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        start = time.perf_counter()
        self.host: Any = importlib.import_module("dlightrag.engine.answer.resources.converters")
        self.support_import_ms = (time.perf_counter() - start) * 1000
        self.version = importlib.metadata.version(
            "firecrawl-anydoc" if self.candidate else "markitdown"
        )

    def pure(self, data: bytes, name: str, *, assets: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {"converter_calls": 0, "assets": []}
        started = time.perf_counter()
        try:
            extension = Path(name).suffix
            if extension in {".docx", ".pptx", ".xlsx"}:
                self.host._preflight_ooxml(data)
            result["preflight_ms"] = (time.perf_counter() - started) * 1000
            start = time.perf_counter()
            result["converter_calls"] = 1
            if self.candidate:
                text = self.candidate.to_markdown_bytes(data, format=extension[1:], ocr="reject")
            else:
                converter = self.host.MarkItDown(enable_plugins=False)
                result["init_ms"] = (time.perf_counter() - start) * 1000
                route = self.host._resolve_route(name, None)
                text = converter.convert_stream(
                    io.BytesIO(data),
                    stream_info=self.host.StreamInfo(mimetype=route.mimetype, extension=extension),
                    keep_data_uris=True,
                ).markdown
            result["convert_including_init_ms"] = (time.perf_counter() - start) * 1000
            result["text"] = text
            result["visual_references"] = len(
                re.findall(r"!\[[^\]]*\]\((?:data:image/|visual://)", text)
            )
            start = time.perf_counter()
            if assets and self.candidate and extension in {".docx", ".pptx", ".xlsx"}:
                document = self.candidate.to_document(data, format=extension[1:])
                result["assets"] = candidate_assets(document)
                result["unique_assets"] = [
                    {
                        "id": a.id,
                        "origin_part": a.origin_part,
                        "sha256": hashlib.sha256(a.data).hexdigest(),
                    }
                    for a in document.assets
                ]
            elif assets and not self.candidate:
                _, visuals = self.host._extract_embedded_visuals(text)
                if extension == ".xlsx":
                    visuals.extend(self.host._extract_xlsx_visuals(data))
                result["assets"] = [
                    {
                        "anchor": v.anchor,
                        "sha256": hashlib.sha256(v.data).hexdigest(),
                        "media_type": v.media_type,
                        "bytes": len(v.data),
                    }
                    for v in visuals
                ]
            result["asset_phase_ms"] = (time.perf_counter() - start) * 1000
        except Exception as exc:
            # Measurement records an error, never fabricates text or retries.
            result.update(
                error=type(exc).__name__,
                message=str(exc),
                pages=getattr(exc, "pages", None),
                page_count=getattr(exc, "page_count", None),
                limit=getattr(exc, "limit", None),
            )
        result["total_ms"] = (time.perf_counter() - started) * 1000
        result["process_peak_mib"] = peak_mib()
        return result

    async def read(self, data: bytes, name: str) -> dict[str, Any]:
        """Actual incumbent Registry read; candidate text-only shim is PROVISIONAL.

        Does not prove candidate error routing, durable Host settlement, or native
        cancellation. Those require the integration slice. No shim for rich Office.
        """
        registry_module: Any = importlib.import_module("dlightrag.engine.answer.resources.registry")
        models: Any = importlib.import_module("dlightrag.engine.answer.resources.models")
        original = registry_module.convert_resource
        calls = 0

        async def convert(content: bytes, *, filename: str, declared_mime: str | None) -> Any:
            nonlocal calls
            calls += 1
            if self.candidate:
                if Path(filename).suffix in {".docx", ".pptx"}:
                    self.host._preflight_ooxml(content)
                text = await asyncio.to_thread(
                    self.candidate.to_markdown_bytes,
                    content,
                    format=Path(filename).suffix[1:],
                    ocr="reject",
                )
                return self.host.ConvertedResource(
                    text=text,
                    visuals=(),
                    converter="firecrawl-anydoc",
                    converter_version=self.version,
                )
            return await original(content, filename=filename, declared_mime=declared_mime)

        registry_module.convert_resource = convert
        try:
            start = time.perf_counter()
            async with registry_module.ResourceRegistry(
                resource_secret=b"synthetic-pilot", cursor_secret=b"synthetic-cursor"
            ) as registry:
                source = models.ResourceInput(filename=name, content=data)
                resource_id = registry.register(source)
                first = await registry.read(resource_id, max_window_tokens=4000)
                first_ms = (time.perf_counter() - start) * 1000
                start = time.perf_counter()
                cached = await registry.read(resource_id, max_window_tokens=4000)
                cached_ms = (time.perf_counter() - start) * 1000
                effects = registry.conversion_effects(resource_id)
                from dlightrag.engine.answer.resources.snapshots import ConversionSnapshot

                snapshot_effect = next(
                    e for e in effects if e.resource_kind == "conversion_snapshot"
                )
                asset_bytes = {
                    e.resource_id: e.content
                    for e in effects
                    if e.resource_kind == "conversion_asset"
                }
                snapshot = ConversionSnapshot.restore(snapshot_effect.content, asset_bytes)
                async with registry_module.ResourceRegistry(
                    resource_secret=b"synthetic-pilot", cursor_secret=b"synthetic-cursor"
                ) as restored:
                    start = time.perf_counter()
                    recovered_id = restored.register(source)
                    restored.adopt_conversion_snapshot(snapshot)
                    recovered = await restored.read(recovered_id, max_window_tokens=4000)
                    recovery_ms = (time.perf_counter() - start) * 1000
                if not (first == cached == recovered) or calls != 1:
                    raise RuntimeError("snapshot/cache replay differs or reparsed")
                return {
                    "first_read_ms": first_ms,
                    "cached_read_ms": cached_ms,
                    "snapshot_restore_read_ms": recovery_ms,
                    "converter_calls": calls,
                    "text": snapshot.text,
                    "status": snapshot.extraction_status,
                    "effect_count": len(effects),
                    "process_peak_mib": peak_mib(),
                    "scope": "provisional candidate text shim"
                    if self.candidate
                    else "production incumbent registry",
                }
        finally:
            registry_module.convert_resource = original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="scratch output directory, never repository root"
    )
    parser.add_argument("--mode", choices=["generate", "quality", "bench"], required=True)
    parser.add_argument("--engine", choices=["anydoc", "markitdown"], default="markitdown")
    parser.add_argument(
        "--support-site",
        type=Path,
        help="existing project site-packages, added after candidate import",
    )
    parser.add_argument("--fixture", help="single quality-qualified text fixture for bench")
    parser.add_argument(
        "--repeats", type=int, default=5, help="warm samples after first sample; not p95"
    )
    parser.add_argument("--label", default="run")
    args = parser.parse_args()
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[1]
    if root == repo or repo in root.parents:
        parser.error("outputs must be outside the repository")
    if not 1 <= args.repeats <= 10:
        parser.error("repeats must be in 1..10")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("label must be a plain identifier")
    root.mkdir(parents=True, exist_ok=True)
    sys.addaudithook(deny_network)
    if args.mode == "generate":
        from anydoc_pilot_fixtures import generate

        generate(root / "fixtures")
        return
    gold = json.loads((root / "fixtures" / "gold.json").read_text())
    converter = Converter(args.engine, args.support_site)
    output: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "engine": args.engine,
        "version": converter.version,
        "engine_import_ms": converter.engine_import_ms,
        "support_import_ms": converter.support_import_ms,
        "baseline_peak_mib": peak_mib(),
        "records": [],
    }
    if args.mode == "bench":
        selected = [g for g in gold if g["name"] == args.fixture]
        if len(selected) != 1 or selected[0]["expected"] != "text":
            parser.error("bench requires one named text candidate")
    else:
        selected = gold
    for fixture in selected:
        data = (root / "fixtures" / fixture["name"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != fixture["sha256"]:
            raise RuntimeError("fixture differs from semantic gold manifest")
        if args.mode == "quality":
            result = converter.pure(data, fixture["name"])
            output["records"].append(
                {"name": fixture["name"], **result, "quality": evaluate(fixture, result)}
            )
        else:
            qualification = converter.pure(data, fixture["name"])
            if not evaluate(fixture, qualification)["passed"]:
                raise RuntimeError("quality failed; refusing performance qualification")
            output["first_operation_in_fresh_process"] = qualification
            # Qualification warms converter modules. First timed sample below is
            # explicitly NOT a cold import/conversion, even in a fresh process.
            for index in range(args.repeats):
                pure = converter.pure(data, fixture["name"], assets=False)
                read = asyncio.run(converter.read(data, fixture["name"]))
                output["records"].append(
                    {"name": fixture["name"], "sample": index, "pure_no_assets": pure, "read": read}
                )
    if args.mode == "quality":
        missing_part = io.BytesIO()
        with (
            zipfile.ZipFile(root / "fixtures" / "docx-text.docx") as source,
            zipfile.ZipFile(missing_part, "w") as target,
        ):
            for name in source.namelist():
                if name != "word/document.xml":
                    target.writestr(name, source.read(name))
        output["missing_part_policy_probe"] = converter.pure(
            missing_part.getvalue(), "missing-part.docx"
        )
    if args.mode == "quality" and converter.candidate:
        # A few KiB of nested XML exercises the real native cap, not an OOM/bomb.
        probe = io.BytesIO()
        with (
            zipfile.ZipFile(root / "fixtures" / "docx-text.docx") as source,
            zipfile.ZipFile(probe, "w") as target,
        ):
            for name in source.namelist():
                data = source.read(name)
                if name == "word/document.xml":
                    data = (
                        b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
                        + b"<w:p>" * 260
                        + b"<w:r><w:t>bounded depth probe</w:t></w:r>"
                        + b"</w:p>" * 260
                        + b"</w:body></w:document>"
                    )
                target.writestr(name, data)
        output["native_depth_policy_probe"] = converter.pure(probe.getvalue(), "depth-probe.docx")
        if output["native_depth_policy_probe"].get("error") != "ResourceLimitError":
            raise RuntimeError("native depth cap was not preserved")
    (root / f"{args.mode}-{args.engine}-{args.label}.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
