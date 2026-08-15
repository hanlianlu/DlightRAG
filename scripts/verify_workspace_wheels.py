# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Verify DlightRAG workspace wheel ownership and dependency direction."""

from __future__ import annotations

import argparse
import ast
import email.parser
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_EXPECTED_PACKAGES = {
    "dlightrag": "dlightrag",
    "dlightrag-agent-core": "dlightrag_agent",
    "dlightrag-ai": "dlightrag_ai",
    "dlightrag-rag-core": "dlightrag_rag",
}
_EXPECTED_DLIGHTRAG_DEPENDENCIES = {
    "dlightrag": {"dlightrag-agent-core", "dlightrag-ai", "dlightrag-rag-core"},
    "dlightrag-agent-core": {"dlightrag-ai"},
    "dlightrag-ai": set(),
    "dlightrag-rag-core": {"dlightrag-ai"},
}
_CONCRETE_LIGHTRAG_BACKEND = "lightrag.kg.postgres_impl"
# import-linter rejects external submodules as contract targets, so the built
# artifact gate owns this one exact LightRAG implementation prohibition.
_SPECIFIC_FORBIDDEN_IMPORTS = {
    "dlightrag-rag-core": (_CONCRETE_LIGHTRAG_BACKEND,),
}
_REQUIRED_EXTERNAL_PROHIBITIONS = {
    "dlightrag": set(),
    "dlightrag-agent-core": {"lightrag", "asyncpg", "fastapi", "mcp"},
    "dlightrag-ai": {"lightrag", "asyncpg", "fastapi", "mcp"},
    "dlightrag-rag-core": {
        _CONCRETE_LIGHTRAG_BACKEND,
        "asyncpg",
        "fastapi",
        "mcp",
    },
}
_DLIGHTRAG_DISTRIBUTIONS = frozenset(_EXPECTED_PACKAGES)
_NORMALIZE_RE = re.compile(r"[-_.]+")

_ABSENT_HELPER = """
def absent(name):
    try:
        return importlib.util.find_spec(name) is None
    except ModuleNotFoundError:
        return True
"""

_AI_SMOKE = (
    """
import importlib
import importlib.util
import pkgutil
import sys
from PIL import Image

pillow_max_image_pixels = Image.MAX_IMAGE_PIXELS
import dlightrag_ai

optional_modules = {
    'dlightrag_ai.providers.anthropic_native',
    'dlightrag_ai.providers.gemini_native',
    'dlightrag_ai.providers.openai_compatible',
}
for module in pkgutil.walk_packages(dlightrag_ai.__path__, prefix='dlightrag_ai.'):
    if module.name not in optional_modules:
        importlib.import_module(module.name)

assert optional_modules.isdisjoint(sys.modules)
assert Image.MAX_IMAGE_PIXELS == pillow_max_image_pixels
"""
    + _ABSENT_HELPER
    + """
assert all(absent(name) for name in (
    'dlightrag', 'dlightrag_agent', 'dlightrag_rag', 'lightrag', 'asyncpg',
    'openai', 'anthropic', 'google.genai'
))
"""
)

_AI_ALL_SMOKE = """
import importlib

for module in (
    'dlightrag_ai.providers.anthropic_native',
    'dlightrag_ai.providers.gemini_native',
    'dlightrag_ai.providers.openai_compatible',
):
    importlib.import_module(module)
"""

_AGENT_SMOKE = (
    """
import asyncio
import importlib
import importlib.util
import pkgutil
import dlightrag_agent
from dlightrag_agent.tools import ToolTurnExecutor
from dlightrag_ai.messages import AssistantTurn

for module in pkgutil.walk_packages(dlightrag_agent.__path__, prefix='dlightrag_agent.'):
    importlib.import_module(module.name)

async def model(**kwargs):
    return AssistantTurn(text='done', tool_calls=(), stop_reason='model_stop')

async def main():
    result = await ToolTurnExecutor(model).run_turn([], [])
    assert result.assistant.text == 'done'

asyncio.run(main())
"""
    + _ABSENT_HELPER
    + """
assert all(absent(name) for name in (
    'dlightrag', 'dlightrag_rag', 'lightrag', 'asyncpg',
    'openai', 'anthropic', 'google.genai'
))
"""
)

_RAG_SMOKE = (
    """
import importlib
import importlib.util
import pkgutil
import dlightrag_rag
from dlightrag_rag.retrieval import MetadataFilter, rrf_fuse

for module in pkgutil.walk_packages(dlightrag_rag.__path__, prefix='dlightrag_rag.'):
    importlib.import_module(module.name)

rows = rrf_fuse([[{'chunk_id': 'a'}], [{'chunk_id': 'a'}]])
assert rows[0]['chunk_id'] == 'a'
assert abs(rows[0]['score'] - 2 / 61) < 1e-12
assert MetadataFilter(filename=' example.pdf ').filename == 'example.pdf'
"""
    + _ABSENT_HELPER
    + """
assert all(absent(name) for name in (
    'dlightrag', 'dlightrag_agent', 'asyncpg'
))
"""
)


@dataclass(frozen=True, slots=True)
class WheelFacts:
    distribution: str
    version: str
    dependencies: frozenset[str]
    requirements: tuple[str, ...]
    top_level_packages: frozenset[str]
    license_files: frozenset[str]
    legal_hashes: tuple[str, str]
    has_py_typed: bool
    has_frontend: bool
    has_model_catalog: bool


@dataclass(frozen=True, slots=True)
class SdistFacts:
    distribution: str
    version: str
    top_level_packages: frozenset[str]
    license_files: frozenset[str]
    legal_hashes: tuple[str, str]
    has_py_typed: bool
    has_frontend: bool
    has_model_catalog: bool


@dataclass(frozen=True, slots=True)
class ImportRule:
    source: str
    forbidden: tuple[str, ...]


def _normalize_distribution(value: str) -> str:
    return _NORMALIZE_RE.sub("-", value).lower()


def _requirement_name(value: str) -> str:
    name = re.split(r"[\s<>=!~;\[(]", value, maxsplit=1)[0]
    return _normalize_distribution(name)


def _metadata_facts(
    raw: bytes,
    *,
    artifact: str,
) -> tuple[str, str, tuple[str, ...], frozenset[str]]:
    metadata = email.parser.Parser().parsestr(raw.decode("utf-8"))
    distribution = _normalize_distribution(str(metadata.get("Name") or ""))
    version = str(metadata.get("Version") or "")
    requirements = tuple(metadata.get_all("Requires-Dist", []))
    license_files = frozenset(metadata.get_all("License-File", []))
    if not distribution or not version:
        raise ValueError(f"{artifact}: missing distribution name or version")
    return distribution, version, requirements, license_files


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _required_member_hashes(
    members: dict[str, bytes],
    *,
    artifact: str,
) -> tuple[str, str]:
    try:
        return _sha256(members["LICENSE"]), _sha256(members["NOTICE"])
    except KeyError as exc:
        raise ValueError(f"{artifact}: must contain LICENSE and NOTICE") from exc


def _top_level_from_wheel(names: list[str]) -> frozenset[str]:
    top_level: set[str] = set()
    for name in names:
        if name.startswith((".", "/")) or ".dist-info/" in name or ".data/" in name:
            continue
        if "/" in name:
            top_level.add(name.split("/", 1)[0])
        elif name.endswith(".py"):
            top_level.add(Path(name).stem)
    return frozenset(top_level)


def _wheel_facts(
    path: Path,
    *,
    rules_by_distribution: dict[str, tuple[ImportRule, ...]],
) -> WheelFacts:
    with zipfile.ZipFile(path) as wheel:
        metadata_paths = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise ValueError(f"{path.name}: expected one METADATA file")
        distribution, version, requirements, license_files = _metadata_facts(
            wheel.read(metadata_paths[0]), artifact=path.name
        )
        dependencies = frozenset(_requirement_name(requirement) for requirement in requirements)
        top_level = _top_level_from_wheel(wheel.namelist())
        legal_members = {
            Path(name).name: wheel.read(name)
            for name in wheel.namelist()
            if ".dist-info/licenses/" in name and Path(name).name in {"LICENSE", "NOTICE"}
        }
        legal_hashes = _required_member_hashes(legal_members, artifact=path.name)
        expected_package = _EXPECTED_PACKAGES.get(distribution, "")
        has_py_typed = f"{expected_package}/py.typed" in wheel.namelist()
        has_frontend = {
            "dlightrag/web/static/generated/style.css",
            "dlightrag/web/static/generated/js/main.js",
        }.issubset(wheel.namelist())
        has_model_catalog = "dlightrag_ai/model_catalog.json" in wheel.namelist()
        sources = (
            (name, wheel.read(name))
            for name in wheel.namelist()
            if name.endswith(".py") and ".dist-info/" not in name and ".data/" not in name
        )
        _validate_source_imports(
            sources,
            path=path,
            distribution=distribution,
            rules=rules_by_distribution.get(distribution, ()),
        )
    return WheelFacts(
        distribution,
        version,
        dependencies,
        requirements,
        top_level,
        license_files,
        legal_hashes,
        has_py_typed,
        has_frontend,
        has_model_catalog,
    )


def _source_module(name: str) -> tuple[str, bool]:
    module_parts = list(Path(name).with_suffix("").parts)
    is_package = module_parts[-1] == "__init__"
    if is_package:
        module_parts.pop()
    return ".".join(module_parts), is_package


def _import_candidates(
    node: ast.Import | ast.ImportFrom,
    *,
    module: str,
    is_package: bool,
) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)

    if node.level == 0:
        base = node.module or ""
    else:
        package = module if is_package else module.rpartition(".")[0]
        try:
            base = importlib.util.resolve_name(f"{'.' * node.level}{node.module or ''}", package)
        except ImportError:
            return ()
    if not base:
        return ()
    imported_members = tuple(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
    return (base, *imported_members)


def _dynamic_import_candidates(
    node: ast.AST,
    *,
    static_strings: tuple[str, ...],
) -> tuple[str, ...]:
    if not isinstance(node, ast.Call) or not node.args:
        return ()
    function_name = (
        node.func.attr
        if isinstance(node.func, ast.Attribute)
        else node.func.id
        if isinstance(node.func, ast.Name)
        else ""
    )
    is_import_module = function_name == "import_module" or function_name == "__import__"
    if not is_import_module:
        return ()

    first_arg = node.args[0]
    values = (
        (first_arg.value,)
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str)
        else static_strings
    )
    candidates = set(values)
    if function_name == "import_module" and any(value.startswith(".") for value in values):
        package_arg = (
            node.args[1]
            if len(node.args) > 1
            else next(
                (keyword.value for keyword in node.keywords if keyword.arg == "package"),
                None,
            )
        )
        packages = (
            (package_arg.value,)
            if isinstance(package_arg, ast.Constant) and isinstance(package_arg.value, str)
            else static_strings
        )
        for value in values:
            if not value.startswith("."):
                continue
            for package in packages:
                if package.startswith("."):
                    continue
                try:
                    candidates.add(importlib.util.resolve_name(value, package))
                except ImportError:
                    continue
    return tuple(sorted(candidates))


def _forbidden_import(imported: str, rules: tuple[ImportRule, ...]) -> bool:
    return any(
        imported == prefix or imported.startswith(f"{prefix}.")
        for rule in rules
        for prefix in rule.forbidden
    )


def _validate_source_imports(
    sources: Iterable[tuple[str, bytes]],
    *,
    path: Path,
    distribution: str,
    rules: tuple[ImportRule, ...],
) -> None:
    for name, raw in sources:
        module, is_package = _source_module(name)
        applicable_rules = tuple(
            rule for rule in rules if module == rule.source or module.startswith(f"{rule.source}.")
        )
        tree = ast.parse(raw, filename=f"{path.name}:{name}")
        static_strings = tuple(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        )
        for node in ast.walk(tree):
            imports = (
                _import_candidates(node, module=module, is_package=is_package)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                else ()
            )
            for imported in imports:
                if _forbidden_import(imported, applicable_rules):
                    raise ValueError(f"{distribution}: forbidden import {imported} in {name}")
            for imported in _dynamic_import_candidates(node, static_strings=static_strings):
                if _forbidden_import(imported, applicable_rules):
                    raise ValueError(f"{distribution}: forbidden import {imported} in {name}")


def _import_rules(config_path: Path) -> dict[str, tuple[ImportRule, ...]]:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    contracts = config["tool"]["importlinter"]["contracts"]
    rules_by_root: dict[str, list[ImportRule]] = {root: [] for root in _EXPECTED_PACKAGES.values()}
    for contract in contracts:
        if contract.get("type") != "forbidden":
            continue
        forbidden = tuple(contract.get("forbidden_modules") or ())
        sources = contract.get("source_modules") or ()
        for source in sources:
            for root in rules_by_root:
                if source == root or source.startswith(f"{root}."):
                    rules_by_root[root].append(ImportRule(source, forbidden))
                    break

    for distribution, forbidden in _SPECIFIC_FORBIDDEN_IMPORTS.items():
        root = _EXPECTED_PACKAGES[distribution]
        rules_by_root[root].append(ImportRule(root, forbidden))

    rules_by_distribution = {
        distribution: tuple(rules_by_root[root])
        for distribution, root in _EXPECTED_PACKAGES.items()
    }
    all_roots = set(_EXPECTED_PACKAGES.values())
    for distribution, root in _EXPECTED_PACKAGES.items():
        allowed_roots = {
            _EXPECTED_PACKAGES[dependency]
            for dependency in _EXPECTED_DLIGHTRAG_DEPENDENCIES[distribution]
        }
        required = (
            all_roots - {root} - allowed_roots | _REQUIRED_EXTERNAL_PROHIBITIONS[distribution]
        )
        root_forbidden = {
            forbidden
            for rule in rules_by_distribution[distribution]
            if rule.source == root
            for forbidden in rule.forbidden
        }
        if missing := required - root_forbidden:
            raise ValueError(
                f"{distribution}: import policy is missing prohibitions {sorted(missing)}"
            )
    return rules_by_distribution


def _sdist_facts(
    path: Path,
    *,
    rules_by_distribution: dict[str, tuple[ImportRule, ...]],
) -> SdistFacts:
    with tarfile.open(path, "r:gz") as sdist:
        members = [member for member in sdist.getmembers() if member.isfile()]
        metadata_members = [member for member in members if member.name.endswith("/PKG-INFO")]
        if len(metadata_members) != 1:
            raise ValueError(f"{path.name}: expected one PKG-INFO file")
        metadata_file = sdist.extractfile(metadata_members[0])
        if metadata_file is None:
            raise ValueError(f"{path.name}: could not read PKG-INFO")
        distribution, version, _, license_files = _metadata_facts(
            metadata_file.read(), artifact=path.name
        )
        sdist_root = Path(metadata_members[0].name).parts[0]
        expected_py_typed = f"{sdist_root}/src/{_EXPECTED_PACKAGES.get(distribution, '')}/py.typed"

        top_level: set[str] = set()
        legal_members: dict[str, bytes] = {}
        has_py_typed = False
        frontend_members: set[str] = set()
        has_model_catalog = False
        sources: list[tuple[str, bytes]] = []
        for member in members:
            parts = Path(member.name).parts
            if len(parts) >= 4 and parts[1] == "src" and member.name.endswith(".py"):
                top_level.add(parts[2])
            elif len(parts) >= 3 and member.name.endswith(".py"):
                top_level.add(parts[1])
            if Path(member.name).name in {"LICENSE", "NOTICE"} and len(parts) == 2:
                legal_file = sdist.extractfile(member)
                if legal_file is not None:
                    legal_members[Path(member.name).name] = legal_file.read()
            if member.name == expected_py_typed:
                has_py_typed = True
            if member.name == f"{sdist_root}/src/dlightrag_ai/model_catalog.json":
                has_model_catalog = True
            if len(parts) > 1:
                frontend_members.add("/".join(parts[1:]))
            relative_parts = parts[1:]
            if relative_parts[:1] == ("src",):
                relative_parts = relative_parts[1:]
            if relative_parts[:1] == (
                _EXPECTED_PACKAGES.get(distribution, ""),
            ) and member.name.endswith(".py"):
                source_file = sdist.extractfile(member)
                if source_file is not None:
                    sources.append(("/".join(relative_parts), source_file.read()))
        legal_hashes = _required_member_hashes(legal_members, artifact=path.name)
        _validate_source_imports(
            sources,
            path=path,
            distribution=distribution,
            rules=rules_by_distribution.get(distribution, ()),
        )
    return SdistFacts(
        distribution,
        version,
        frozenset(top_level),
        license_files,
        legal_hashes,
        has_py_typed,
        {
            "src/dlightrag/web/static/generated/style.css",
            "src/dlightrag/web/static/generated/js/main.js",
        }.issubset(frontend_members),
        has_model_catalog,
    )


def verify_dist(dist_dir: Path, *, config_path: Path) -> None:
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 4 or len(sdists) != 4:
        raise ValueError(
            f"expected four wheels and four sdists, found {len(wheels)} wheels and {len(sdists)} sdists"
        )

    facts_by_distribution: dict[str, WheelFacts] = {}
    legal_hashes = (
        _sha256((config_path.parent / "LICENSE").read_bytes()),
        _sha256((config_path.parent / "NOTICE").read_bytes()),
    )
    rules_by_distribution = _import_rules(config_path)
    for wheel in wheels:
        facts = _wheel_facts(
            wheel,
            rules_by_distribution=rules_by_distribution,
        )
        if facts.distribution in facts_by_distribution:
            raise ValueError(f"duplicate wheel for {facts.distribution}")
        facts_by_distribution[facts.distribution] = facts

    if set(facts_by_distribution) != set(_EXPECTED_PACKAGES):
        raise ValueError(
            "workspace wheel set differs from expected distributions: "
            f"{sorted(facts_by_distribution)}"
        )

    versions = {facts.version for facts in facts_by_distribution.values()}
    if len(versions) != 1:
        raise ValueError(f"workspace versions are not lockstep: {sorted(versions)}")
    (version,) = versions
    expected_sdists = {
        f"{distribution.replace('-', '_')}-{version}.tar.gz" for distribution in _EXPECTED_PACKAGES
    }
    actual_sdists = {path.name for path in sdists}
    if actual_sdists != expected_sdists:
        raise ValueError(
            f"sdist set does not match workspace wheels: expected {sorted(expected_sdists)}, "
            f"found {sorted(actual_sdists)}"
        )

    for distribution, facts in facts_by_distribution.items():
        expected_top_level = {_EXPECTED_PACKAGES[distribution]}
        if set(facts.top_level_packages) != expected_top_level:
            raise ValueError(
                f"{distribution}: expected top-level package {sorted(expected_top_level)}, "
                f"found {sorted(facts.top_level_packages)}"
            )
        actual_dlightrag_dependencies = set(facts.dependencies) & _DLIGHTRAG_DISTRIBUTIONS
        expected_dependencies = _EXPECTED_DLIGHTRAG_DEPENDENCIES[distribution]
        if actual_dlightrag_dependencies != expected_dependencies:
            raise ValueError(
                f"{distribution}: expected DlightRAG dependencies {sorted(expected_dependencies)}, "
                f"found {sorted(actual_dlightrag_dependencies)}"
            )
        if distribution == "dlightrag" and not any(
            requirement.lower().startswith("dlightrag-ai[all]")
            for requirement in facts.requirements
        ):
            raise ValueError("dlightrag: dlightrag-ai dependency must request the all extra")
        compact_requirements = {
            requirement.replace(" ", "").lower() for requirement in facts.requirements
        }
        for dependency in expected_dependencies:
            expected_requirement = f"{dependency}=={version}"
            if dependency == "dlightrag-ai" and distribution == "dlightrag":
                expected_requirement = f"dlightrag-ai[all]=={version}"
            if expected_requirement not in compact_requirements:
                raise ValueError(
                    f"{distribution}: dependency must be pinned as {expected_requirement}"
                )
        if facts.license_files != {"LICENSE", "NOTICE"} or facts.legal_hashes != legal_hashes:
            raise ValueError(f"{distribution}: wheel must contain repository LICENSE and NOTICE")
        if not facts.has_py_typed:
            raise ValueError(f"{distribution}: wheel must contain py.typed")
        if distribution == "dlightrag-ai" and not facts.has_model_catalog:
            raise ValueError("dlightrag-ai: wheel must contain model_catalog.json")
        if distribution == "dlightrag" and not facts.has_frontend:
            raise ValueError("dlightrag: wheel must contain generated frontend assets")

    parsed_sdists = [
        _sdist_facts(path, rules_by_distribution=rules_by_distribution) for path in sdists
    ]
    sdist_facts = {facts.distribution: facts for facts in parsed_sdists}
    for distribution, facts in sdist_facts.items():
        if facts.version != version or facts.top_level_packages != {
            _EXPECTED_PACKAGES[distribution]
        }:
            raise ValueError("sdist metadata or top-level packages do not match workspace wheels")
        if facts.license_files != {"LICENSE", "NOTICE"} or facts.legal_hashes != legal_hashes:
            raise ValueError(f"{distribution}: sdist must contain repository LICENSE and NOTICE")
        if not facts.has_py_typed:
            raise ValueError(f"{distribution}: sdist must contain py.typed")
        if distribution == "dlightrag-ai" and not facts.has_model_catalog:
            raise ValueError("dlightrag-ai: sdist must contain model_catalog.json")
        if distribution == "dlightrag" and not facts.has_frontend:
            raise ValueError("dlightrag: sdist must contain generated frontend assets")


def _run_checked(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    completed = subprocess.run(  # noqa: S603 - verifier builds fixed argv from validated artifacts
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"installed wheel smoke failed: {' '.join(command)}\n{detail}")


def _wheel_path(dist_dir: Path, distribution: str) -> Path:
    matches = tuple(dist_dir.glob(f"{distribution.replace('-', '_')}-*.whl"))
    if len(matches) != 1:
        raise ValueError(f"expected one wheel for {distribution}, found {len(matches)}")
    return matches[0]


def smoke_installed(dist_dir: Path) -> None:
    dist_dir = dist_dir.resolve()
    smoke_cases = (
        ("ai", ("dlightrag-ai",), _AI_SMOKE),
        ("ai-all", ("dlightrag-ai[all]",), _AI_ALL_SMOKE),
        ("agent", ("dlightrag-ai", "dlightrag-agent-core"), _AGENT_SMOKE),
        ("rag", ("dlightrag-ai", "dlightrag-rag-core"), _RAG_SMOKE),
    )
    required_distributions = {
        requirement.partition("[")[0]
        for _, requirements, _ in smoke_cases
        for requirement in requirements
    }
    wheel_paths = {
        distribution: _wheel_path(dist_dir, distribution) for distribution in required_distributions
    }
    base_env = dict(os.environ)
    base_env.pop("PYTHONPATH", None)
    base_env["PYTHONSAFEPATH"] = "1"

    for label, distributions, code in smoke_cases:
        with tempfile.TemporaryDirectory(prefix=f"dlightrag-{label}-wheel-") as raw_temp:
            temp = Path(raw_temp)
            venv = temp / ".venv"
            _run_checked(
                ["uv", "venv", "--python", "3.14", str(venv)],
                cwd=temp,
                env=base_env,
            )
            python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            _run_checked(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    *(
                        f"{requirement} @ {wheel_paths[requirement.partition('[')[0]].as_uri()}"
                        if "[" in requirement
                        else str(wheel_paths[requirement])
                        for requirement in distributions
                    ),
                ],
                cwd=temp,
                env=base_env,
            )
            _run_checked([str(python), "-I", "-c", code], cwd=temp, env=base_env)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist", type=Path, required=True, help="Directory containing built artifacts"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("pyproject.toml"),
        help="Import-linter configuration used as the source policy",
    )
    parser.add_argument(
        "--smoke-installed",
        action="store_true",
        help="Install core wheels into isolated Python 3.14 environments and run interface smokes",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        verify_dist(args.dist, config_path=args.config)
        if args.smoke_installed:
            smoke_installed(args.dist)
    except (OSError, SyntaxError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        print(f"workspace wheel verification failed: {exc}", file=sys.stderr)
        return 1
    print("workspace wheel verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
