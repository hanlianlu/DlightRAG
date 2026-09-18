# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The policy an Agent's processes run under, and the helper that applies it."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from dlightrag.engine.agent.environment.child import build_child_environment
from dlightrag.engine.agent.environment.confinement import (
    ConfinementPolicy,
    DeclaredLayer,
    WorkspaceConfinement,
    _parse,
    _supported_rights,
    landlock_abi,
)
from dlightrag.engine.agent.environment.local import LocalExecutionEnvironment

#: Rights that change bytes: write, remove, truncate, move, and every create right.
_BYTE_CHANGING = (
    0x2 | 0x10 | 0x20 | 0x2000 | 0x4000 | 0x40 | 0x80 | 0x100 | 0x200 | 0x400 | 0x800 | 0x1000
)
_CREATE_RIGHTS = 0x40 | 0x80 | 0x100 | 0x200 | 0x400 | 0x800 | 0x1000


class _Output:
    """Collect one command's merged output, the way the file tools do."""

    def __init__(self) -> None:
        self.chunks: list[str] = []

    async def feed(self, chunk: Any) -> None:
        self.chunks.append(chunk.data.decode())

    @property
    def text(self) -> str:
        return "".join(self.chunks)


def test_a_declared_layer_that_overlaps_a_forbidden_tree_is_refused(tmp_path: Path) -> None:
    """The corpus and the project tree are refused where a capability declares itself.

    Composition is the only place a layer can be added, so refusing there is what
    keeps "retrieval is the path to knowledge" a property of the build instead of a
    configuration choice (ADR 0024).
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    for path in (corpus, corpus / "inputs", tmp_path):
        with pytest.raises(ValueError, match="overlaps"):
            ConfinementPolicy(
                forbidden=(corpus,),
                declared=(DeclaredLayer(path=path, capability="sneaky"),),
            )

    allowed = ConfinementPolicy(
        forbidden=(corpus,),
        declared=(DeclaredLayer(path=tmp_path / "skills", capability="skills"),),
    )
    assert allowed.declared[0].capability == "skills"


def test_the_workspace_is_the_only_writable_grant(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "skills").mkdir()
    (tmp_path / "app").mkdir()

    confinement = ConfinementPolicy(
        declared=(DeclaredLayer(path=tmp_path / "skills", capability="skills"),),
        runtime=(tmp_path / "app",),
    ).for_workspace(workspace)
    grants = dict(confinement.rules)

    assert grants[workspace] == 0xFFFF
    assert grants[tmp_path / "skills"] & _BYTE_CHANGING == 0
    assert grants[tmp_path / "app"] & _BYTE_CHANGING == 0
    if Path("/dev") in grants:
        # Writing an existing device is allowed; creating one is not.
        assert grants[Path("/dev")] & 0x2
        assert grants[Path("/dev")] & _CREATE_RIGHTS == 0


def test_a_grant_never_names_a_right_the_kernel_cannot_restrict() -> None:
    """A grant must stay inside the rights the running ABI knows, or the kernel refuses."""
    assert _supported_rights(1) == 0x1FFF
    assert _supported_rights(2) & 0x2000
    assert _supported_rights(3) & 0x4000
    assert _supported_rights(5) & 0x8000
    assert _supported_rights(8) == _supported_rights(5)


def test_the_helper_prefix_round_trips_the_grants(tmp_path: Path) -> None:
    """The child rebuilds exactly the grants the parent validated."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    confinement = ConfinementPolicy().for_workspace(workspace)
    command = ["/bin/echo", "hello"]

    rules, parsed_command = _parse([*confinement.launch_prefix(), *command])

    assert rules == confinement.rules
    assert parsed_command == command


class _MarkerConfinement(WorkspaceConfinement):
    """A confinement whose prefix announces itself, for hosts without a kernel seam."""

    def launch_prefix(self) -> list[str]:
        return [
            sys.executable,
            "-c",
            (
                "import os, sys\n"
                "print('CONFINED', flush=True)\n"
                "os.execvp(sys.argv[1], sys.argv[1:])\n"
            ),
        ]


@pytest.mark.asyncio
async def test_every_command_goes_through_the_confinement_prefix(tmp_path: Path) -> None:
    environment = LocalExecutionEnvironment(
        tmp_path, confinement=_MarkerConfinement(workspace=tmp_path, rules=())
    )
    home, tmp = environment.prepare_process_directories()
    output = _Output()

    completed = await environment.run(
        [sys.executable, "-c", "print('payload')"],
        env=build_child_environment(home=home, tmp=tmp),
        on_output=output.feed,
    )

    assert completed.returncode == 0
    assert "CONFINED" in output.text
    assert "payload" in output.text


@pytest.mark.asyncio
async def test_an_unconfined_environment_is_still_available_to_tests(tmp_path: Path) -> None:
    environment = LocalExecutionEnvironment(tmp_path)
    home, tmp = environment.prepare_process_directories()

    completed = await environment.run(
        [sys.executable, "-c", "print('payload')"],
        env=build_child_environment(home=home, tmp=tmp),
    )

    assert completed.returncode == 0


def _probe_script(outside: Path) -> str:
    return (
        "import pathlib\n"
        "pathlib.Path('written.txt').write_text('ok')\n"
        "try:\n"
        f"    pathlib.Path({str(outside)!r}).read_text()\n"
        "except OSError:\n"
        "    print('DENIED')\n"
        "else:\n"
        "    print('READABLE')\n"
    )


@pytest.mark.skipif(landlock_abi() < 1, reason="host kernel offers no Landlock")
@pytest.mark.asyncio
async def test_a_confined_command_reaches_its_workspace_and_nothing_else(
    tmp_path: Path,
) -> None:
    """The property itself: the workspace is reachable, a sibling tree is not."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("corpus-like bytes")
    environment = LocalExecutionEnvironment(
        workspace, confinement=ConfinementPolicy().for_workspace(workspace)
    )
    home, tmp = environment.prepare_process_directories()
    output = _Output()

    completed = await environment.run(
        [sys.executable, "-c", _probe_script(secret)],
        env=build_child_environment(home=home, tmp=tmp),
        on_output=output.feed,
    )

    assert completed.returncode == 0
    assert (workspace / "written.txt").read_text() == "ok"
    assert "DENIED" in output.text
    assert "READABLE" not in output.text


@pytest.mark.skipif(landlock_abi() >= 1, reason="a Landlock host exercises the confined path")
def test_without_landlock_the_property_is_absent_and_the_command_still_runs(
    tmp_path: Path,
) -> None:
    """Graceful degradation: the Run proceeds unconfined, and `/health` says so."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("corpus-like bytes")
    prefix = ConfinementPolicy().for_workspace(workspace).launch_prefix()

    completed = subprocess.run(
        [*prefix, sys.executable, "-c", _probe_script(secret)],
        capture_output=True,
        text=True,
        check=False,
        cwd=workspace,
    )

    assert completed.returncode == 0
    assert "READABLE" in completed.stdout


def test_the_reported_state_names_the_abi_or_says_unavailable() -> None:
    state = ConfinementPolicy().state()

    assert state == "unavailable" or state.startswith("landlock:abi")


def test_the_composition_root_refuses_the_corpus_and_the_project_tree() -> None:
    """A deployment cannot hand the Agent the corpus by configuring it away.

    The deny set is built where the application is composed, so it cannot be a value
    an operator changes: this is the "retrieval is the path to knowledge" guard
    (ADR 0024), and the test pins it where composition decides it.
    """
    from types import SimpleNamespace

    from dlightrag._compose import agent_confinement_policy

    policy = agent_confinement_policy(
        SimpleNamespace(working_dir_path=Path("/app/dlightrag_storage"))  # type: ignore[arg-type]
    )

    assert Path("/app/dlightrag_storage") in policy.forbidden
    assert Path.cwd() in policy.forbidden
    for tree in policy.forbidden:
        with pytest.raises(ValueError, match="overlaps"):
            ConfinementPolicy(
                forbidden=(tree,),
                declared=(DeclaredLayer(path=tree / "anything", capability="sneaky"),),
            )


def test_a_policy_whose_paths_do_not_exist_still_runs_the_command(tmp_path: Path) -> None:
    """A declared layer that is absent is skipped, never a reason to fail a Run."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = ConfinementPolicy(
        declared=(DeclaredLayer(path=tmp_path / "not-installed", capability="skills"),)
    )
    prefix = policy.for_workspace(workspace).launch_prefix()

    completed = subprocess.run(
        [*prefix, sys.executable, "-c", "print('ran')"],
        capture_output=True,
        text=True,
        check=False,
        cwd=workspace,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "ran"


def test_the_helper_never_writes_plumbing_to_the_command_output(tmp_path: Path) -> None:
    """The tool result stays clean: confinement plumbing is not the model's business.

    A diagnostic written here would be streamed into the Tool result (ADR 0024), so
    the helper is silent whether it applied the policy or could not.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    prefix = ConfinementPolicy().for_workspace(workspace).launch_prefix()

    completed = subprocess.run(
        [*prefix, sys.executable, "-c", "print('payload')"],
        capture_output=True,
        text=True,
        check=False,
        cwd=workspace,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "payload"
    assert completed.stderr == ""
