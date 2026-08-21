# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Behavioral checks for independently installable workspace packages."""

import importlib
import io
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "package_name", ["dlightrag_ai", "dlightrag_agent", "dlightrag_memory", "dlightrag_rag"]
)
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
    version: str = "1.9.0",
    include_legal: bool = True,
    include_frontend: bool = False,
    additional_sources: dict[str, str] | None = None,
    sdist_source: str | None = None,
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
    with zipfile.ZipFile(dist_dir / f"{wheel_name}-{version}-py3-none-any.whl", "w") as wheel:
        wheel.writestr(f"{package}/__init__.py", source)
        for relative_path, content in (additional_sources or {}).items():
            wheel.writestr(f"{package}/{relative_path}", content)
        wheel.writestr(f"{package}/py.typed", "")
        wheel.writestr(f"{dist_info}/METADATA", "\n".join(metadata))
        wheel.writestr(f"{dist_info}/RECORD", "")
        if include_legal:
            wheel.write(_REPO / "LICENSE", f"{dist_info}/licenses/LICENSE")
            wheel.write(_REPO / "NOTICE", f"{dist_info}/licenses/NOTICE")
        if include_frontend:
            wheel.writestr(f"{package}/web/static/app/index.html", "<dl-app></dl-app>")
            wheel.writestr(f"{package}/web/static/app/login.html", "<form></form>")
            wheel.writestr(f"{package}/web/static/app/assets/style-test.css", "body {}")
            wheel.writestr(f"{package}/web/static/app/assets/app-test.js", "export {}")
            wheel.writestr(f"{package}/web/static/app/assets/theme-init-test.js", "")
    sdist_root = f"{wheel_name}-{version}"
    packaged_sdist_source = source if sdist_source is None else sdist_source
    with tarfile.open(dist_dir / f"{wheel_name}-{version}.tar.gz", "w:gz") as sdist:
        for name, content in {
            f"{sdist_root}/PKG-INFO": "\n".join(metadata),
            f"{sdist_root}/src/{package}/__init__.py": packaged_sdist_source,
            **{
                f"{sdist_root}/src/{package}/{relative_path}": content
                for relative_path, content in (additional_sources or {}).items()
            },
            f"{sdist_root}/src/{package}/py.typed": "",
            **(
                {
                    f"{sdist_root}/LICENSE": (_REPO / "LICENSE").read_text(encoding="utf-8"),
                    f"{sdist_root}/NOTICE": (_REPO / "NOTICE").read_text(encoding="utf-8"),
                }
                if include_legal
                else {}
            ),
            **(
                {
                    f"{sdist_root}/src/{package}/web/static/app/index.html": "<dl-app></dl-app>",
                    f"{sdist_root}/src/{package}/web/static/app/login.html": "<form></form>",
                    f"{sdist_root}/src/{package}/web/static/app/assets/style-test.css": "body {}",
                    f"{sdist_root}/src/{package}/web/static/app/assets/app-test.js": "export {}",
                    f"{sdist_root}/src/{package}/web/static/app/assets/theme-init-test.js": "",
                }
                if include_frontend
                else {}
            ),
        }.items():
            payload = content.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            sdist.addfile(info, io.BytesIO(payload))


def _write_workspace_artifacts(
    tmp_path: Path,
    *,
    rag_source: str = "",
    rag_version: str = "1.9.0",
    rag_package: str = "dlightrag_rag",
    rag_requires: tuple[str, ...] = ("dlightrag-ai==1.9.0", "pydantic>=2.11.0"),
    rag_sdist_source: str | None = None,
    root_requires: tuple[str, ...] = (
        "dlightrag-ai[all]==1.9.0",
        "dlightrag-agent-core==1.9.0",
        "dlightrag-memory==1.9.0",
        "dlightrag-rag-core==1.9.0",
    ),
    ai_include_legal: bool = True,
    ai_include_model_catalog: bool = True,
    ai_extras: tuple[str, ...] = ("all", "anthropic", "gemini", "openai"),
    root_include_frontend: bool = True,
    root_source: str = "",
    root_additional_sources: dict[str, str] | None = None,
) -> None:
    _write_wheel(
        tmp_path,
        distribution="dlightrag-ai",
        package="dlightrag_ai",
        requires=("pydantic>=2.11.0",),
        provides_extras=ai_extras,
        include_legal=ai_include_legal,
        additional_sources=(
            {"model_catalog.json": '{"revision":"test","models":[]}'}
            if ai_include_model_catalog
            else None
        ),
    )
    _write_wheel(
        tmp_path,
        distribution="dlightrag-agent-core",
        package="dlightrag_agent",
        requires=("dlightrag-ai==1.9.0",),
    )
    _write_wheel(
        tmp_path,
        distribution="dlightrag-memory",
        package="dlightrag_memory",
        requires=(),
    )
    _write_wheel(
        tmp_path,
        distribution="dlightrag-rag-core",
        package=rag_package,
        requires=rag_requires,
        source=rag_source,
        sdist_source=rag_sdist_source,
        version=rag_version,
    )
    _write_wheel(
        tmp_path,
        distribution="dlightrag",
        package="dlightrag",
        requires=root_requires,
        source=root_source,
        include_frontend=root_include_frontend,
        additional_sources=root_additional_sources,
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


def test_workspace_wheel_verifier_accepts_four_lockstep_artifacts(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path)

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 0, completed.stderr


def test_workspace_wheel_verifier_rejects_forbidden_source_import(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path, rag_source="import asyncpg\n")

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "forbidden import asyncpg" in completed.stderr


def test_workspace_wheel_verifier_scans_sdist_sources(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path, rag_sdist_source="import asyncpg\n")

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "forbidden import asyncpg" in completed.stderr


@pytest.mark.parametrize(
    "rag_source",
    [
        "import lightrag.kg.postgres_impl\n",
        "from lightrag.kg import postgres_impl\n",
        "import importlib\nimportlib.import_module('lightrag.kg.postgres_impl')\n",
        "from importlib import import_module\nimport_module('lightrag.kg.postgres_impl')\n",
        "import importlib\nMODULES = {'pg': 'lightrag.kg.postgres_impl'}\n"
        "importlib.import_module(MODULES['pg'])\n",
        "MODULE = 'lightrag.kg.postgres_impl'\n__import__(MODULE)\n",
        "import builtins\nbuiltins.__import__('lightrag.kg.postgres_impl')\n",
        "import importlib\nimportlib.import_module('.postgres_impl', 'lightrag.kg')\n",
    ],
)
def test_workspace_wheel_verifier_rejects_concrete_rag_backend_imports(
    tmp_path: Path,
    rag_source: str,
) -> None:
    _write_workspace_artifacts(tmp_path, rag_source=rag_source)

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "forbidden import lightrag.kg.postgres_impl" in completed.stderr


def test_workspace_wheel_verifier_fails_closed_without_core_rules(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path)
    config_path = tmp_path / "pyproject.toml"
    config_path.write_text("[tool.importlinter]\ncontracts = []\n", encoding="utf-8")
    for legal_name in ("LICENSE", "NOTICE"):
        (tmp_path / legal_name).write_bytes((_REPO / legal_name).read_bytes())

    completed = _verify_wheels(tmp_path, config_path=config_path)

    assert completed.returncode == 1
    assert "import policy is missing prohibitions" in completed.stderr


def test_workspace_wheel_verifier_fails_closed_without_external_rule(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path)
    config_path = tmp_path / "pyproject.toml"
    config_text = (_REPO / "pyproject.toml").read_text(encoding="utf-8")
    config_path.write_text(config_text.replace('    "asyncpg",\n', "", 1), encoding="utf-8")
    for legal_name in ("LICENSE", "NOTICE"):
        (tmp_path / legal_name).write_bytes((_REPO / legal_name).read_bytes())

    completed = _verify_wheels(tmp_path, config_path=config_path)

    assert completed.returncode == 1
    assert "import policy is missing prohibitions ['asyncpg']" in completed.stderr


def test_workspace_wheel_verifier_scopes_root_import_contracts(tmp_path: Path) -> None:
    _write_workspace_artifacts(
        tmp_path,
        root_source="from dlightrag.answer.errors import AnswerImageError\n",
    )

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 0, completed.stderr


def test_workspace_wheel_verifier_rejects_answer_import_of_postgres_adapter(
    tmp_path: Path,
) -> None:
    _write_workspace_artifacts(
        tmp_path,
        root_additional_sources={
            "answer/example.py": "from dlightrag.adapters.postgres import answer_runs\n"
        },
    )

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "forbidden import dlightrag.adapters.postgres" in completed.stderr


@pytest.mark.parametrize(
    "config_source",
    [
        "import dlightrag.answer\n",
        "from dlightrag import answer\n",
    ],
)
def test_workspace_wheel_verifier_enforces_root_source_contract(
    tmp_path: Path,
    config_source: str,
) -> None:
    _write_workspace_artifacts(
        tmp_path,
        root_additional_sources={"config.py": config_source},
    )

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "forbidden import dlightrag.answer in dlightrag/config.py" in completed.stderr


def test_workspace_wheel_verifier_rejects_mismatched_sdist_set(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path)
    (tmp_path / "dlightrag_rag_core-1.9.0.tar.gz").rename(tmp_path / "unexpected-1.9.0.tar.gz")

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "sdist set does not match" in completed.stderr


def test_workspace_wheel_verifier_rejects_version_drift(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path, rag_version="1.9.1")

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "versions are not lockstep" in completed.stderr


def test_workspace_wheel_verifier_rejects_wrong_top_level_package(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path, rag_package="wrong_package")

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "expected top-level package" in completed.stderr


def test_workspace_wheel_verifier_rejects_invalid_core_dependency(tmp_path: Path) -> None:
    _write_workspace_artifacts(
        tmp_path,
        rag_requires=("pydantic>=2.11.0", "dlightrag-agent-core==1.9.0"),
    )

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "expected DlightRAG dependencies" in completed.stderr


def test_workspace_wheel_verifier_requires_root_ai_extra(tmp_path: Path) -> None:
    _write_workspace_artifacts(
        tmp_path,
        root_requires=(
            "dlightrag-ai==1.9.0",
            "dlightrag-agent-core==1.9.0",
            "dlightrag-memory==1.9.0",
            "dlightrag-rag-core==1.9.0",
        ),
    )

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "must request the all extra" in completed.stderr


def test_workspace_wheel_verifier_requires_lockstep_dependency_pins(tmp_path: Path) -> None:
    _write_workspace_artifacts(
        tmp_path,
        root_requires=(
            "dlightrag-ai[all]>=1.9.0",
            "dlightrag-agent-core==1.9.0",
            "dlightrag-memory==1.9.0",
            "dlightrag-rag-core==1.9.0",
        ),
    )

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "dependency must be pinned as dlightrag-ai[all]==1.9.0" in completed.stderr


def test_workspace_wheel_verifier_rejects_corrupt_sdist(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path)
    (tmp_path / "dlightrag_rag_core-1.9.0.tar.gz").write_bytes(b"not a tar archive")

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "workspace wheel verification failed" in completed.stderr


def test_workspace_wheel_verifier_requires_legal_files(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path, ai_include_legal=False)

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "LICENSE and NOTICE" in completed.stderr


def test_workspace_wheel_verifier_requires_ai_model_catalog(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path, ai_include_model_catalog=False)

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "model_catalog.json" in completed.stderr


def test_workspace_wheel_verifier_requires_ai_extras(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path, ai_extras=("openai",))

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "expected extras" in completed.stderr


def test_workspace_wheel_verifier_requires_root_frontend(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path, root_include_frontend=False)

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "generated frontend" in completed.stderr
