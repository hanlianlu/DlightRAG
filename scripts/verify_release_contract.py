#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Verify the release contract: the product version's places, and the platform's pins."""

from __future__ import annotations

import argparse
import ast
import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The configuration schema a deployment's config.yaml is written against. It is not the
#: product version: a version reset (2.x to 0.x, say) says nothing about the configuration
#: schema, so this names it and a release never rewrites it.
CONFIG_BASELINE = "2.0"


def _project(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))["project"]


def _verify_playwright_pins(root: Path, dependencies: list[str]) -> None:
    """The Agent Browser client and its pool image run one Playwright version (ADR 0032).

    The pool's server refuses a client whose major or minor version differs, and the
    pin fixes the patch too, so every place that names the version must agree.
    """
    pinned = [dependency for dependency in dependencies if dependency.startswith("playwright==")]
    if len(pinned) != 1:
        raise ValueError("root distribution must pin playwright== exactly once")
    version = pinned[0].removeprefix("playwright==")

    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    locked = {package["name"]: package["version"] for package in lock["package"]}.get("playwright")
    if locked != version:
        raise ValueError(f"uv.lock resolves playwright {locked}, not {version}")

    pool = root / "agent-browser/browser"
    declared = re.search(
        r"^ARG PLAYWRIGHT_VERSION=(\S+)$", (pool / "Dockerfile").read_text(encoding="utf-8"), re.M
    )
    if declared is None or declared.group(1) != version:
        raise ValueError(f"the pool Dockerfile does not pin Playwright {version}")
    manifest = json.loads((pool / "package.json").read_text(encoding="utf-8"))
    if manifest.get("dependencies", {}).get("playwright") != version:
        raise ValueError(f"the pool package.json does not pin Playwright {version}")
    package_lock = json.loads((pool / "package-lock.json").read_text(encoding="utf-8"))["packages"]
    if (
        package_lock[""].get("dependencies", {}).get("playwright") != version
        or package_lock.get("node_modules/playwright", {}).get("version") != version
    ):
        raise ValueError(f"the pool package-lock.json does not lock Playwright {version}")
    image = re.search(
        r"^\s*image: dlightrag-agent-browser:(\S+)$",
        (root / "docker-compose.yml").read_text(encoding="utf-8"),
        re.M,
    )
    if image is None or image.group(1) != version:
        raise ValueError(f"the Compose pool image is not tagged {version}")


def verify_versions(root: Path = ROOT) -> None:
    """The product version in the four places that name it.

    Every other place derives it — each workspace runtime reads its installed distribution,
    the frontend names no version, and ``uv.lock`` is resolved from the pins below — so a
    release edits three files and ``make release`` does that.
    """
    manifests = (
        root / "pyproject.toml",
        root / "packages/memory/pyproject.toml",
        root / "packages/mineru/pyproject.toml",
        root / "packages/observability/pyproject.toml",
    )
    projects = [_project(path) for path in manifests]
    versions = {str(project["version"]) for project in projects}
    if len(versions) != 1:
        raise ValueError(f"workspace versions are not lockstep: {sorted(versions)}")
    (version,) = versions
    for name in ("dlightrag-memory", "dlightrag-observability"):
        dependency = f"{name}=={version}"
        if dependency not in projects[0].get("dependencies", []):
            raise ValueError(f"root distribution must depend on {dependency}")


#: The parser tables in configuration.md and the settings class each one documents.
#: A deployment reads the table to decide what it may leave out, so a documented
#: default that drifts from the code is a wrong answer, not a stale sentence.
_PARSER_DOCUMENTATION = {
    "MinerU (default)": "MinerUSidecarSettings",
    "Docling": "DoclingSidecarSettings",
}
_SETTINGS_SOURCE = Path("src/dlightrag/engine/rag/workspace/settings.py")
_CONFIGURATION_DOC = Path("docs/configuration.md")


def _documented_parser_defaults(root: Path) -> dict[str, dict[str, str]]:
    """The Default column of each parser table, by section."""
    documented: dict[str, dict[str, str]] = {}
    section = ""
    for line in (root / _CONFIGURATION_DOC).read_text(encoding="utf-8").splitlines():
        if line.startswith("### "):
            section = line.removeprefix("### ").strip()
            continue
        if section not in _PARSER_DOCUMENTATION or not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        field = cells[0].strip("`").strip()
        # The header row and the separator carry no default; a field name is never
        # prose, so anything with a space in it is not one.
        if not field or field == "Field" or " " in field or set(field) <= {"-", ":"}:
            continue
        documented.setdefault(section, {})[field] = cells[1].strip("`")
    return documented


def _literal_defaults(node: ast.expr) -> object:
    """The value of ``X = "y"`` or ``Field(default=5)``.

    A factory or a computed default is refused rather than guessed: the check exists to
    compare documented values, and it cannot invent one.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Call):
        for keyword in node.keywords:
            if keyword.arg == "default":
                return _literal_defaults(keyword.value)
    raise ValueError(f"unsupported default: {ast.dump(node)[:60]}")


def _settings_parser_defaults(root: Path) -> dict[str, dict[str, str]]:
    """Each sidecar class's literal field defaults, read from the source."""
    classes = {
        node.name: node
        for node in ast.walk(ast.parse((root / _SETTINGS_SOURCE).read_text(encoding="utf-8")))
        if isinstance(node, ast.ClassDef)
    }
    defaults: dict[str, dict[str, str]] = {}
    for section, class_name in _PARSER_DOCUMENTATION.items():
        node = classes.get(class_name)
        if node is None:
            raise ValueError(f"{_SETTINGS_SOURCE} has no {class_name}")
        fields: dict[str, str] = {}
        for statement in node.body:
            if not isinstance(statement, ast.AnnAssign) or statement.value is None:
                continue
            name = getattr(statement.target, "id", "")
            if not name or name.startswith("_"):
                continue
            value = _literal_defaults(statement.value)
            fields[name] = "unset" if value is None else str(value).lower()
        defaults[section] = fields
    return defaults


def _verify_documented_parser_defaults(root: Path = ROOT) -> None:
    documentation = root / _CONFIGURATION_DOC
    settings = root / _SETTINGS_SOURCE
    if not documentation.is_file() or not settings.is_file():
        missing = documentation if not documentation.is_file() else settings
        raise ValueError(f"{missing} is missing, so its documented defaults cannot be checked")
    documented = _documented_parser_defaults(root)
    actual = _settings_parser_defaults(root)
    for section, fields in documented.items():
        # Both directions: a field renamed, dropped, or added on either side is drift.
        if set(fields) != set(actual[section]):
            only_documented = sorted(set(fields) - set(actual[section]))
            only_settings = sorted(set(actual[section]) - set(fields))
            raise ValueError(
                f"{_CONFIGURATION_DOC}: {section} documents {only_documented or '—'} and "
                f"omits {only_settings or '—'} relative to {_PARSER_DOCUMENTATION[section]}"
            )
        for field, shown in fields.items():
            if shown.lower() != actual[section][field]:
                raise ValueError(
                    f"{_CONFIGURATION_DOC}: {section} documents {field}={shown}, "
                    f"but {_PARSER_DOCUMENTATION[section]} defaults to "
                    f"{actual[section][field]}"
                )


def verify_platform(root: Path = ROOT) -> None:
    """What holds on every push, whatever the product version is."""
    dependencies = _project(root / "pyproject.toml").get("dependencies", [])
    _verify_playwright_pins(root, dependencies)
    _verify_documented_parser_defaults(root)

    config_text = (root / "config.yaml").read_text(encoding="utf-8")
    header = f"# DlightRAG {CONFIG_BASELINE} canonical configuration"
    if not re.search(rf"^{re.escape(header)}$", config_text, re.MULTILINE):
        raise ValueError(f"config.yaml does not name the {CONFIG_BASELINE} baseline")
    if "execution_environment: trust" not in config_text or "connections:" not in config_text:
        raise ValueError("config.yaml does not expose the canonical Agent defaults")


def verify_repository(root: Path = ROOT, *, scope: str = "all") -> None:
    if scope not in ("all", "versions", "platform"):
        raise ValueError(f"unknown scope: {scope}")
    if scope in ("all", "versions"):
        verify_versions(root)
    if scope in ("all", "platform"):
        verify_platform(root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the release contract.")
    parser.add_argument(
        "--scope",
        choices=("all", "versions", "platform"),
        default="all",
        help="versions: the product version's places; platform: the pins and config every push holds",
    )
    args = parser.parse_args()
    verify_repository(scope=args.scope)
    print(f"release contract verification passed ({args.scope})")


if __name__ == "__main__":
    main()
