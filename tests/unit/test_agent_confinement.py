# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The policy an Agent's processes run under, and the helper that applies it."""

from __future__ import annotations

import os
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


def test_the_reported_state_says_disabled_or_what_this_host_enforces() -> None:
    """An operator reads the effective state, so an unconfined host is stated.

    A deployment with no Agent environment is a fact about the deployment; every
    other answer is the host's, which is what keeps the absence of a kernel seam
    visible instead of implied (ADR 0024).
    """
    from dlightrag.engine.agent.environment import confinement_state

    assert confinement_state("disabled") == "disabled"
    trust = confinement_state("trust")
    assert trust == "unavailable" or trust.startswith("landlock:abi")


def test_the_composition_root_refuses_the_corpus_and_the_project_tree(
    test_config: Any,
) -> None:
    """A deployment cannot hand the Agent the corpus by configuring it away.

    The deny set is built where the application is composed, so it cannot be a value
    an operator changes: this is the "retrieval is the path to knowledge" guard
    (ADR 0024), and the test pins it where composition decides it.
    """
    from dlightrag._compose import agent_confinement_policy

    policy = agent_confinement_policy(test_config)

    assert test_config.working_dir_path in policy.forbidden
    assert Path.cwd() in policy.forbidden
    for tree in policy.forbidden:
        with pytest.raises(ValueError, match="overlaps"):
            ConfinementPolicy(
                forbidden=(tree,),
                declared=(DeclaredLayer(path=tree / "anything", capability="sneaky"),),
            )


def test_the_policy_declares_the_roots_skills_are_loaded_from(test_config: Any) -> None:
    """The capability declares what it reads, beside the capability (ADR 0024).

    The packaged built-ins are deliberately not declared: the serving process loads
    them through its own ``load_skill`` tool, and their assets are reachable from
    inside the confinement wherever the install puts them under the runtime prefix. A
    source checkout keeps them in the project tree, which the Agent may not see at all.
    """
    from dlightrag._compose import agent_confinement_policy
    from dlightrag.application.settings import agent_skills_root, owner_skills_root
    from dlightrag.engine.agent.skills import builtin_skills_root, owner_skill_root

    policy = agent_confinement_policy(test_config)
    alice = {layer.resolved() for layer in policy.for_owner("alice")}

    assert alice == {
        agent_skills_root(test_config),
        owner_skill_root(owner_skills_root(test_config), "alice"),
    }
    assert Path(str(builtin_skills_root())).resolve() not in alice


def test_one_owners_skills_are_not_another_owners(test_config: Any) -> None:
    """Owner publishing is per owner, so the grant is the shard and never the parent.

    Declaring the shared owner root would hand every Run's shell every owner's
    published skills; the shard is the unit an Agent may see (ADR 0024, owner
    isolation).
    """
    from dlightrag._compose import agent_confinement_policy
    from dlightrag.application.settings import agent_skills_root, owner_skills_root
    from dlightrag.engine.agent.skills import owner_skill_root

    policy = agent_confinement_policy(test_config)
    alice = {layer.resolved() for layer in policy.for_owner("alice")}
    bob = {layer.resolved() for layer in policy.for_owner("bob")}

    assert alice & bob == {agent_skills_root(test_config)}
    assert owner_skills_root(test_config) not in alice | bob
    assert owner_skill_root(owner_skills_root(test_config), "bob") not in alice


@pytest.mark.asyncio
async def test_binding_a_run_grants_only_its_own_owners_skills(
    test_config: Any, tmp_path: Path
) -> None:
    """The binder is where an owner becomes known, so the shard arrives with the Run."""
    from dlightrag._compose import agent_confinement_policy
    from dlightrag.application.settings import owner_skills_root
    from dlightrag.engine.agent.environment.execution import TrustExecutionAdapter
    from dlightrag.engine.agent.skills import owner_skill_root
    from dlightrag.engine.answer.workspace import bind_run_workspace
    from dlightrag.engine.runtime.workspace import InMemoryWorkspaceStore

    policy = agent_confinement_policy(test_config)
    bound = await bind_run_workspace(
        workspace_root=tmp_path,
        owner_id="alice",
        run_id="run-alice",
        fencing_epoch=1,
        recorded_epoch=None,
        store=InMemoryWorkspaceStore(),
        execution_adapter=TrustExecutionAdapter(policy),
    )
    environment = bound.environment
    assert isinstance(environment, LocalExecutionEnvironment)
    confinement = environment._confinement  # pyright: ignore[reportPrivateUsage]

    assert confinement is not None
    granted = {path for path, _access in confinement.rules}
    assert owner_skill_root(owner_skills_root(test_config), "alice") in granted
    assert owner_skill_root(owner_skills_root(test_config), "bob") not in granted


def test_a_skills_root_inside_the_project_tree_is_refused(test_config: Any, tmp_path: Path) -> None:
    """A Skill root inside the project tree would expose what the Agent may not see."""
    from dlightrag._compose import agent_confinement_policy
    from tests.config_helpers import replace_config

    inside = Path.cwd() / "skills-in-the-project"
    config = replace_config(test_config, "answer.agent.skills_root", str(inside))

    with pytest.raises(ValueError, match="overlaps"):
        agent_confinement_policy(config)


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


@pytest.mark.skipif(landlock_abi() < 1, reason="host kernel offers no Landlock")
@pytest.mark.asyncio
async def test_a_declared_skill_root_serves_its_script_while_a_sibling_stays_out(
    tmp_path: Path,
) -> None:
    """A declared layer is the point of the declaration: a skill's asset runs.

    The roots a Skill is loaded from are the reason this capability declares anything
    at all, so the property is that the Agent's own process can execute what a skill
    points at — and still cannot read a directory nobody declared.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skills = tmp_path / "skills"
    skills.mkdir()
    script = skills / "run.sh"
    script.write_text("#!/bin/sh\necho skill-ran\n")
    script.chmod(0o755)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "secret.txt").write_text("corpus-like bytes")
    policy = ConfinementPolicy(declared=(DeclaredLayer(path=skills, capability="skills"),))
    environment = LocalExecutionEnvironment(workspace, confinement=policy.for_workspace(workspace))
    home, tmp = environment.prepare_process_directories()
    output = _Output()
    script_body = (
        "import pathlib, subprocess\n"
        f"print(subprocess.run(['sh', {str(script)!r}], capture_output=True, text=True).stdout.strip())\n"
        "try:\n"
        f"    pathlib.Path({str(elsewhere / 'secret.txt')!r}).read_text()\n"
        "except OSError:\n"
        "    print('DENIED')\n"
        "else:\n"
        "    print('READABLE')\n"
    )

    completed = await environment.run(
        [sys.executable, "-c", script_body],
        env=build_child_environment(home=home, tmp=tmp),
        on_output=output.feed,
    )

    assert completed.returncode == 0
    assert "skill-ran" in output.text
    assert "DENIED" in output.text
    assert "READABLE" not in output.text


@pytest.mark.skipif(landlock_abi() < 1, reason="host kernel offers no Landlock")
@pytest.mark.asyncio
async def test_the_corpus_is_readable_by_the_process_and_denied_to_its_children(
    test_config: Any, tmp_path: Path
) -> None:
    """The property the whole decision exists for, on one path, from both sides.

    A deployment is trusted: the serving process opens the corpus it serves. The Agent
    it hosts is not: the same path is outside every grant its children receive, which is
    what makes retrieval the Agent's path to knowledge instead of a suggestion
    (ADR 0024).
    """
    corpus = test_config.working_dir_path
    corpus.mkdir(parents=True, exist_ok=True)
    source = corpus / "source.txt"
    source.write_text("corpus bytes")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = ConfinementPolicy(forbidden=(corpus,))
    environment = LocalExecutionEnvironment(workspace, confinement=policy.for_workspace(workspace))
    home, tmp = environment.prepare_process_directories()
    output = _Output()

    completed = await environment.run(
        [sys.executable, "-c", _probe_script(source)],
        env=build_child_environment(home=home, tmp=tmp),
        on_output=output.feed,
    )

    # The application reads what it serves…
    assert source.read_text() == "corpus bytes"
    # …and its Agent's own process cannot.
    assert completed.returncode == 0
    assert "DENIED" in output.text
    assert "READABLE" not in output.text


def test_the_runtime_grants_cover_the_interpreter_and_its_standard_library() -> None:
    """A venv's interpreter lives in the base prefix, not in ``sys.prefix``.

    Landlock checks the binary's *resolved* path and gates reading the standard
    library too, so granting only the virtual environment's prefix refuses the exec
    with ``PermissionError`` and leaves ``import json`` unreadable. macOS did not
    show it — this host reports no Landlock ABI and degrades to an unconfined exec —
    while Linux CI, whose base interpreter sits outside ``/usr``, refused every
    confined command. The invariant is checked here so it fails wherever Python is
    installed, not only where the kernel enforces it.
    """
    roots = ConfinementPolicy().runtime_roots()
    resolved_roots = tuple(root.resolve() for root in roots)

    interpreter = Path(sys.executable).resolve()
    stdlib = Path(os.__file__).resolve()
    for needed, why in ((interpreter, "executed"), (stdlib, "imported")):
        assert any(needed.is_relative_to(root) for root in resolved_roots), (
            f"{needed} is {why} by every confined command but no runtime root grants it: {roots}"
        )
