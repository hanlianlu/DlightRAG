# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Behavioral checks for root plus independently installable Memory artifacts."""

import importlib
import io
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path
from typing import Any

import pytest

from tests.support.report_examples import examples

_REPO = Path(__file__).resolve().parents[2]
_PROJECT = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
_VERSION = _PROJECT["version"]
#: The root wheel's third-party requirements, as pyproject states them.
_BATTERIES = tuple(
    requirement
    for requirement in _PROJECT["dependencies"]
    if not requirement.startswith("dlightrag-")
)
_ROOT_REQUIRES = (
    f"dlightrag-memory=={_VERSION}",
    f"dlightrag-observability=={_VERSION}",
    *_BATTERIES,
)
_REPORT_EXAMPLES = "engine/agent/builtin_skills/interactive-html/references"
#: What the observability source distribution has to carry for its own commands to run:
#: the Compose file, the collector configurations, the credentials template, and the
#: alert and dashboard JSON the CLI reads. The wheel cannot carry them — uv's build
#: backend includes only the module root — so `source-include` is their only route into
#: an artifact, and this is what says the route still works.
_OBSERVABILITY_SDIST_STACK = {
    "compose.yaml": "services: {}\n",
    "stack/.env.example": "HYPERDX_API_KEY=\nEXPRESS_SESSION_SECRET=\n",
    "stack/collector.yaml": "receivers: {}\n",
    "stack/collector-standalone.yaml": "receivers: {}\n",
    "stack/alerts/channel.json": '{"name": "webhook"}',
    "stack/dashboards/dlightrag.json": '{"name": "DlightRAG"}',
}


@pytest.mark.parametrize("package_name", ["dlightrag", "dlightrag_memory"])
def test_workspace_package_root_imports(package_name: str) -> None:
    importlib.import_module(package_name)


def _write_wheel(
    dist_dir: Path,
    *,
    distribution: str,
    package: str,
    requires: tuple[str, ...] = (),
    provides_extras: tuple[str, ...] = (),
    source: str = "",
    version: str = _VERSION,
    include_legal: bool = True,
    include_frontend: bool = False,
    additional_sources: dict[str, str] | None = None,
    sdist_source: str | None = None,
    sdist_without: tuple[str, ...] = (),
    sdist_stack: dict[str, str] | None = None,
) -> None:
    wheel_name = distribution.replace("-", "_")
    dist_info = f"{wheel_name}-{version}.dist-info"
    metadata = [
        "Metadata-Version: 2.3",
        f"Name: {distribution}",
        f"Version: {version}",
        *(("License-File: LICENSE", "License-File: NOTICE") if include_legal else ()),
        *(f"Requires-Dist: {requirement}" for requirement in requires),
        *(f"Provides-Extra: {extra}" for extra in provides_extras),
        "",
    ]
    sources = additional_sources or {}
    with zipfile.ZipFile(dist_dir / f"{wheel_name}-{version}-py3-none-any.whl", "w") as wheel:
        wheel.writestr(f"{package}/__init__.py", source)
        for relative_path, content in sources.items():
            wheel.writestr(f"{package}/{relative_path}", content)
        wheel.writestr(f"{package}/py.typed", "")
        wheel.writestr(f"{dist_info}/METADATA", "\n".join(metadata))
        wheel.writestr(f"{dist_info}/RECORD", "")
        if include_legal:
            wheel.write(_REPO / "LICENSE", f"{dist_info}/licenses/LICENSE")
            wheel.write(_REPO / "NOTICE", f"{dist_info}/licenses/NOTICE")
        if include_frontend:
            wheel.writestr(
                f"{package}/adapters/http/browser/static/app/index.html", "<dl-app></dl-app>"
            )
            wheel.writestr(
                f"{package}/adapters/http/browser/static/app/login.html", "<form></form>"
            )
            wheel.writestr(
                f"{package}/adapters/http/browser/static/app/assets/app-test.css", "body {}"
            )
            wheel.writestr(
                f"{package}/adapters/http/browser/static/app/assets/app-test.js", "export {}"
            )
            wheel.writestr(
                f"{package}/adapters/http/browser/static/app/assets/theme-init-test.js", ""
            )

    sdist_root = f"{wheel_name}-{version}"
    packaged_source = source if sdist_source is None else sdist_source
    members = {
        f"{sdist_root}/PKG-INFO": "\n".join(metadata),
        f"{sdist_root}/src/{package}/__init__.py": packaged_source,
        **{
            f"{sdist_root}/src/{package}/{relative_path}": content
            for relative_path, content in sources.items()
            if relative_path not in sdist_without
        },
        f"{sdist_root}/src/{package}/py.typed": "",
    }
    if include_legal:
        members.update(
            {
                f"{sdist_root}/LICENSE": (_REPO / "LICENSE").read_text(encoding="utf-8"),
                f"{sdist_root}/NOTICE": (_REPO / "NOTICE").read_text(encoding="utf-8"),
            }
        )
    if include_frontend:
        members.update(
            {
                f"{sdist_root}/src/{package}/adapters/http/browser/static/app/index.html": "<dl-app></dl-app>",
                f"{sdist_root}/src/{package}/adapters/http/browser/static/app/login.html": "<form></form>",
                f"{sdist_root}/src/{package}/adapters/http/browser/static/app/assets/app-test.css": "body {}",
                f"{sdist_root}/src/{package}/adapters/http/browser/static/app/assets/app-test.js": "export {}",
                f"{sdist_root}/src/{package}/adapters/http/browser/static/app/assets/theme-init-test.js": "",
            }
        )
    # The stack files only the source distribution carries: uv's build backend has no
    # wheel includes, so they reach an artifact through `source-include` or not at all.
    members.update(
        {f"{sdist_root}/{member}": content for member, content in (sdist_stack or {}).items()}
    )
    with tarfile.open(dist_dir / f"{wheel_name}-{version}.tar.gz", "w:gz") as sdist:
        for name, content in members.items():
            payload = content.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            sdist.addfile(info, io.BytesIO(payload))


def _write_workspace_artifacts(
    tmp_path: Path,
    *,
    root_source: str = "",
    root_requires: tuple[str, ...] = _ROOT_REQUIRES,
    root_version: str = _VERSION,
    root_package: str = "dlightrag",
    root_additional_sources: dict[str, str] | None = None,
    root_sdist_source: str | None = None,
    root_include_legal: bool = True,
    root_include_model_catalog: bool = True,
    root_include_builtin_skill_creator: bool = True,
    root_include_builtin_skill_council: bool = True,
    root_include_report_examples: bool = True,
    root_sdist_without: tuple[str, ...] = (),
    root_include_frontend: bool = True,
    root_extras: tuple[str, ...] = ("milvus",),
    memory_source: str = "",
    memory_sdist_source: str | None = None,
    memory_version: str = _VERSION,
    observability_sdist_stack: dict[str, str] | None = _OBSERVABILITY_SDIST_STACK,
) -> None:
    root_sources = dict(root_additional_sources or {})
    if root_include_model_catalog:
        root_sources["engine/ai/model_catalog.json"] = '{"models":[]}'
    if root_include_builtin_skill_creator:
        root_sources["engine/agent/builtin_skills/skill-creator/SKILL.md"] = (
            "---\nname: skill-creator\ndescription: Create skills.\n---\n# Skill Creator"
        )
    if root_include_builtin_skill_council:
        root_sources["engine/agent/builtin_skills/council/SKILL.md"] = (
            "---\nname: council\ndescription: Independent critique.\n---\n# Council"
        )
    if root_include_report_examples:
        for name in examples().values():
            root_sources[f"{_REPORT_EXAMPLES}/{name.name}"] = (
                "<!-- an example -->\n<header></header>"
            )
    _write_wheel(
        tmp_path,
        distribution="dlightrag",
        package=root_package,
        requires=root_requires,
        provides_extras=root_extras,
        source=root_source,
        sdist_source=root_sdist_source,
        version=root_version,
        include_legal=root_include_legal,
        include_frontend=root_include_frontend,
        additional_sources=root_sources,
        sdist_without=root_sdist_without,
    )
    _write_wheel(
        tmp_path,
        distribution="dlightrag-memory",
        package="dlightrag_memory",
        requires=("asyncpg>=0.31.0", "mcp>=2.0.0"),
        source=memory_source,
        sdist_source=memory_sdist_source,
        version=memory_version,
    )
    _write_wheel(
        tmp_path,
        distribution="dlightrag-mineru",
        package="dlightrag_mineru",
        requires=(),
        version=memory_version,
    )
    _write_wheel(
        tmp_path,
        distribution="dlightrag-observability",
        package="dlightrag_observability",
        requires=(),
        version=memory_version,
        sdist_stack=observability_sdist_stack,
    )


def _verify_wheels(
    dist_dir: Path,
    *,
    config_path: Path = _REPO / "pyproject.toml",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(_REPO / "scripts" / "verify_workspace_wheels.py"),
            "--config",
            str(config_path),
            "--workspace-root",
            str(_REPO),
            "--dist",
            str(dist_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_workspace_wheel_verifier_accepts_root_and_memory(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path)

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 0, completed.stderr


def test_workspace_wheel_verifier_requires_the_stack_in_the_observability_sdist(
    tmp_path: Path,
) -> None:
    """The stack reaches a distributable only through the source distribution.

    A command that cannot find `compose.yaml` fails rather than degrading, so an sdist
    that lost the stack is a deployment that cannot start it — which is what this holds
    out, one member at a time.
    """
    stack = dict(_OBSERVABILITY_SDIST_STACK)
    del stack["stack/collector.yaml"]
    _write_workspace_artifacts(tmp_path, observability_sdist_stack=stack)

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "sdist must contain the stack its commands read" in completed.stderr


@pytest.mark.parametrize(
    "memory_source",
    ["import fastapi\n", "import dlightrag\n", "from dlightrag.engine import ai\n"],
)
def test_workspace_wheel_verifier_rejects_memory_host_imports(
    tmp_path: Path, memory_source: str
) -> None:
    _write_workspace_artifacts(tmp_path, memory_source=memory_source)

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "forbidden import" in completed.stderr


def test_workspace_wheel_verifier_scans_sdist_sources(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path, memory_sdist_source="import fastapi\n")

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "forbidden import fastapi" in completed.stderr


@pytest.mark.parametrize(
    "rag_source",
    [
        "import lightrag.kg.postgres_impl\n",
        "from lightrag.kg import postgres_impl\n",
        "import importlib\nimportlib.import_module('lightrag.kg.postgres_impl')\n",
    ],
)
def test_workspace_wheel_verifier_rejects_concrete_rag_backend_imports(
    tmp_path: Path, rag_source: str
) -> None:
    _write_workspace_artifacts(
        tmp_path,
        root_additional_sources={"engine/rag/example.py": rag_source},
    )

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "forbidden import lightrag.kg.postgres_impl" in completed.stderr


def test_workspace_wheel_verifier_rejects_rag_composition_imports(tmp_path: Path) -> None:
    _write_workspace_artifacts(
        tmp_path,
        root_additional_sources={
            "engine/rag/corpus/rebuild_bm25.py": (
                "from dlightrag.adapters.postgres.core._pool import pg_pool\n"
            )
        },
    )

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "forbidden import dlightrag.adapters.postgres" in completed.stderr


def test_workspace_wheel_verifier_rejects_answer_import_of_postgres_adapter(
    tmp_path: Path,
) -> None:
    _write_workspace_artifacts(
        tmp_path,
        root_additional_sources={
            "engine/answer/example.py": "from dlightrag.adapters.postgres import answer_runs\n"
        },
    )

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "forbidden import dlightrag.adapters.postgres" in completed.stderr


def test_workspace_wheel_verifier_lets_a_wildcard_ignore_cover_nested_modules(
    tmp_path: Path,
) -> None:
    _write_workspace_artifacts(
        tmp_path,
        root_additional_sources={"adapters/postgres/core/_errors.py": "import asyncpg\n"},
    )

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 0, completed.stderr


def test_workspace_wheel_verifier_still_refuses_a_module_outside_the_wildcard_ignore(
    tmp_path: Path,
) -> None:
    _write_workspace_artifacts(
        tmp_path,
        root_additional_sources={"engine/rag/example.py": "import asyncpg\n"},
    )

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "forbidden import asyncpg in dlightrag/engine/rag/example.py" in completed.stderr


@pytest.mark.parametrize(
    "config_source",
    ["import dlightrag.engine.answer\n", "from dlightrag.engine import answer\n"],
)
def test_workspace_wheel_verifier_enforces_root_source_contract(
    tmp_path: Path, config_source: str
) -> None:
    _write_workspace_artifacts(
        tmp_path,
        root_additional_sources={"application/config/probe.py": config_source},
    )

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert (
        "forbidden import dlightrag.engine.answer in dlightrag/application/config/probe.py"
        in completed.stderr
    )


def test_workspace_wheel_verifier_rejects_mismatched_sdist_set(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path)
    (tmp_path / f"dlightrag_memory-{_VERSION}.tar.gz").rename(
        tmp_path / f"unexpected-{_VERSION}.tar.gz"
    )

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "sdist set does not match" in completed.stderr


def test_workspace_wheel_verifier_rejects_version_drift(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path, memory_version="2.0.1")

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "versions are not lockstep" in completed.stderr


def test_workspace_wheel_verifier_rejects_wrong_top_level_package(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path, root_package="wrong_package")

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "expected top-level package" in completed.stderr


@pytest.mark.parametrize("junk_path", [".DS_Store", "__pycache__/probe.pyc", "probe.pyo"])
def test_workspace_wheel_verifier_rejects_generated_junk(tmp_path: Path, junk_path: str) -> None:
    _write_workspace_artifacts(tmp_path, root_additional_sources={junk_path: "junk"})

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "contains forbidden generated files" in completed.stderr


def test_workspace_wheel_verifier_requires_memory_dependency(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path, root_requires=_BATTERIES)

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "expected DlightRAG dependencies" in completed.stderr


def test_workspace_wheel_verifier_requires_lockstep_memory_pin(tmp_path: Path) -> None:
    _write_workspace_artifacts(
        tmp_path,
        root_requires=(
            f"dlightrag-memory>={_VERSION}",
            f"dlightrag-observability=={_VERSION}",
            *_BATTERIES,
        ),
    )

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert f"dependency must be pinned as dlightrag-memory=={_VERSION}" in completed.stderr


def test_workspace_wheel_verifier_requires_batteries_in_wheel_metadata(tmp_path: Path) -> None:
    _write_workspace_artifacts(
        tmp_path,
        root_requires=tuple(item for item in _ROOT_REQUIRES if not item.startswith("anthropic")),
    )

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "missing batteries-included dependencies ['anthropic']" in completed.stderr


def test_workspace_wheel_verifier_rejects_corrupt_sdist(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path)
    (tmp_path / f"dlightrag_memory-{_VERSION}.tar.gz").write_bytes(b"not a tar archive")

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "workspace wheel verification failed" in completed.stderr


@pytest.mark.parametrize(
    ("without", "message"),
    [
        ({"root_include_legal": False}, "LICENSE and NOTICE"),
        ({"root_include_model_catalog": False}, "model_catalog.json"),
        ({"root_include_builtin_skill_creator": False}, "built-in skill-creator SKILL.md"),
        ({"root_include_builtin_skill_council": False}, "built-in council SKILL.md"),
        ({"root_include_frontend": False}, "generated frontend"),
    ],
    ids=["legal-files", "model-catalog", "skill-creator", "skill-council", "frontend"],
)
def test_workspace_wheel_verifier_requires_each_shipped_root_file(
    tmp_path: Path, without: dict[str, Any], message: str
) -> None:
    _write_workspace_artifacts(tmp_path, **without)

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert message in completed.stderr


def test_workspace_wheel_verifier_requires_every_interactive_html_example_in_the_wheel(
    tmp_path: Path,
) -> None:
    _write_workspace_artifacts(tmp_path, root_include_report_examples=False)

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "wheel must contain the interactive-html report examples" in completed.stderr
    assert all(path.name in completed.stderr for path in examples().values())


def test_workspace_wheel_verifier_requires_every_interactive_html_example_in_the_sdist(
    tmp_path: Path,
) -> None:
    first = f"{_REPORT_EXAMPLES}/{next(iter(examples().values())).name}"
    _write_workspace_artifacts(tmp_path, root_sdist_without=(first,))

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "sdist must contain the interactive-html report examples" in completed.stderr
    assert Path(first).name in completed.stderr


def test_workspace_wheel_verifier_rejects_unexpected_root_extras(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path, root_extras=("all",))

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "expected extras" in completed.stderr
