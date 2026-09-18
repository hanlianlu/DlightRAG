# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Kernel-enforced confinement for the processes one Agent Workspace may spawn.

An Agent Workspace is already the only place an Agent writes; this makes it the
only place its processes read. A policy is composed where the execution adapter is
resolved, validated there, and carried by every command the environment spawns: the
helper in this module applies the policy to itself and *then* execs the real
command, so the boundary is in force before any Agent code runs and survives the
exec ([ADR 0024](../../../../docs/adr/0024-the-agent-sees-only-its-workspace.md)).

Landlock is the mechanism because it is unprivileged, cannot be bypassed by how a
command is spelled, and needs nothing a restricted container withholds. Content is
reachable through the workspace, through the runtime the toolchain needs, and
through the layers capabilities declare; the corpus working directory and the
project tree are paths to refuse rather than paths to grant. A host without
Landlock runs the command unconfined, which `/health` and the Run's trace state
rather than hide.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

#: Landlock filesystem rights, in the order the kernel defines them.
_FS_EXECUTE = 1 << 0
_FS_WRITE_FILE = 1 << 1
_FS_READ_FILE = 1 << 2
_FS_READ_DIR = 1 << 3
_FS_REFER = 1 << 13
_FS_TRUNCATE = 1 << 14
_FS_IOCTL_DEV = 1 << 15

#: Every right the first Landlock ABI can restrict: execute, read, write, remove,
#: and every create right.
_FS_V1 = (1 << 13) - 1
#: The right each later ABI added, so a grant never names one the kernel rejects.
_FS_BY_ABI = (
    (5, _FS_V1 | _FS_REFER | _FS_TRUNCATE | _FS_IOCTL_DEV),
    (3, _FS_V1 | _FS_REFER | _FS_TRUNCATE),
    (2, _FS_V1 | _FS_REFER),
    (1, _FS_V1),
)

#: The Agent Workspace: the Run's whole mutable world, including moving files
#: inside it and truncating them.
_ACCESS_WORKSPACE = _FS_BY_ABI[0][1]
#: The runtime: readable and executable, never writable.
_ACCESS_RUNTIME = _FS_EXECUTE | _FS_READ_FILE | _FS_READ_DIR
#: Devices: an existing device may be written, none may be created.
_ACCESS_DEVICES = _FS_WRITE_FILE | _FS_READ_FILE | _FS_READ_DIR | _FS_IOCTL_DEV

_RUNTIME_DIRECTORIES = ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/proc")
_DEVICES = Path("/dev")

_SYSCALL_CREATE_RULESET = 444
_SYSCALL_ADD_RULE = 445
_SYSCALL_RESTRICT_SELF = 446
_CREATE_RULESET_VERSION = 1
_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38

_RULES_OPTION = "--rules"
_USAGE = f"usage: <python> confinement.py {_RULES_OPTION} <json-rules> -- <command>"

__all__ = [
    "ConfinementPolicy",
    "DeclaredLayer",
    "WorkspaceConfinement",
    "landlock_abi",
    "main",
]


@dataclass(frozen=True, slots=True)
class DeclaredLayer:
    """One path a capability needs beyond the Agent Workspace and the runtime.

    A capability declares what it reads where it is composed, in code, so the set
    of visible paths is reviewable beside the capability that asked for it rather
    than configurable per deployment (ADR 0024). A layer is read-only.
    """

    path: Path
    capability: str

    def resolved(self) -> Path:
        return self.path.expanduser().resolve()


@dataclass(frozen=True, slots=True)
class ConfinementPolicy:
    """What a deployment declares once; one policy serves every Run it answers.

    ``forbidden`` is a validation set rather than a mechanism: the corpus working
    directory and the project tree are absent from every grant by construction,
    and this is what refuses a declared layer that would put one of them back.
    """

    forbidden: tuple[Path, ...] = ()
    declared: tuple[DeclaredLayer, ...] = ()
    runtime: tuple[Path, ...] = ()
    #: Layers that depend on who the Run belongs to, resolved when a Run binds. Owner
    #: publishing is the case: one owner's skills must be readable by that owner's Runs
    #: and by no other, so the shard is declared per Run rather than once per process.
    per_owner: Callable[[str], Sequence[DeclaredLayer]] | None = None

    def __post_init__(self) -> None:
        for layer in self.declared:
            self.refuse(layer.path, layer.capability)

    def refuse(self, path: Path, capability: str) -> None:
        """Refuse one path a capability intends to serve, and say which tree it hit.

        Composition calls this for every root a capability *may* serve, not only the
        ones this Run grants: a root nobody declares today is still a root an operator
        can misconfigure, and this is the line where that fails the process instead of
        leaking it into one Run.
        """
        resolved = path.expanduser().resolve()
        for forbidden in self.forbidden:
            refused = forbidden.expanduser().resolve()
            if (
                resolved == refused
                or resolved.is_relative_to(refused)
                or refused.is_relative_to(resolved)
            ):
                raise ValueError(
                    f"declared capability layer {path} ({capability}) "
                    f"overlaps {forbidden}, which the Agent may never see"
                )

    def runtime_roots(self) -> tuple[Path, ...]:
        """Return the runtime directories every Agent command needs.

        The interpreter's own prefix is among them because the toolchain lives in
        it: a Run that cannot execute Python cannot do the work it was given. That
        the installed package also sits under that prefix is the price of a working
        toolchain, and ADR 0024 records it.
        """
        roots = self.runtime or (Path(sys.prefix), *map(Path, _RUNTIME_DIRECTORIES))
        return tuple(root for root in roots if root.is_dir())

    def for_workspace(
        self, workspace: Path, *, owner_id: str | None = None
    ) -> WorkspaceConfinement:
        """Return the grants one Run's processes receive."""
        for layer in self.for_owner(owner_id):
            self.refuse(layer.path, layer.capability)
        grants: list[tuple[Path, int]] = [(workspace, _ACCESS_WORKSPACE)]
        grants.extend((root, _ACCESS_RUNTIME) for root in self.runtime_roots())
        if _DEVICES.is_dir():
            grants.append((_DEVICES, _ACCESS_DEVICES))
        grants.extend((layer.resolved(), _ACCESS_RUNTIME) for layer in self.for_owner(owner_id))
        merged: dict[Path, int] = {}
        for path, access in grants:
            merged[path] = merged.get(path, 0) | access
        return WorkspaceConfinement(workspace=workspace, rules=tuple(merged.items()))

    def for_owner(self, owner_id: str | None) -> tuple[DeclaredLayer, ...]:
        """Return the layers this Run's owner adds to the process-wide ones."""
        if owner_id is None or self.per_owner is None:
            return self.declared
        return (*self.declared, *self.per_owner(owner_id))

    def state(self) -> str:
        """Return what this host can enforce, for `/health` and a Run's trace."""
        abi = landlock_abi()
        return f"landlock:abi{abi}" if abi >= 1 else "unavailable"


@dataclass(frozen=True, slots=True)
class WorkspaceConfinement:
    """The grants one Run's processes receive, and the way to apply them."""

    workspace: Path
    rules: tuple[tuple[Path, int], ...]

    def launch_prefix(self) -> list[str]:
        """Return the argv prefix that applies these grants and then execs."""
        payload = json.dumps(
            [[str(path), access] for path, access in self.rules], separators=(",", ":")
        )
        # Run this module as a file rather than with ``-m``: the package import chain
        # reaches this module through ``local``, and runpy warns on stderr when a
        # ``-m`` target is already imported — a warning that would land in the model's
        # tool output (ADR 0024).
        return [sys.executable, os.fspath(Path(__file__).resolve()), _RULES_OPTION, payload, "--"]


@lru_cache(maxsize=1)
def landlock_abi() -> int:
    """Return the Landlock ABI this kernel offers, or 0 when it offers none.

    A kernel without Landlock answers with an errno rather than a crash, so the
    probe is the whole portability story: the helper degrades to an unconfined
    exec, and the state is reported where an operator looks rather than where the
    model looks.
    """
    if sys.platform != "linux":
        return 0
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        version = libc.syscall(_SYSCALL_CREATE_RULESET, None, 0, _CREATE_RULESET_VERSION)
    except AttributeError, OSError:
        return 0
    return int(version) if version >= 1 else 0


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def _supported_rights(abi: int) -> int:
    for version, rights in _FS_BY_ABI:
        if abi >= version:
            return rights
    return _FS_V1


def _apply(rules: Sequence[tuple[Path, int]], abi: int) -> None:
    """Restrict this process to ``rules``; raise OSError when the kernel refuses."""
    supported = _supported_rights(abi)
    libc = ctypes.CDLL(None, use_errno=True)
    attr = _RulesetAttr(supported)
    ruleset = libc.syscall(_SYSCALL_CREATE_RULESET, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if ruleset < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")
    try:
        for path, access in rules:
            if not path.is_dir():
                continue
            # O_PATH is Linux-only, like the syscalls this runs beside.
            descriptor = os.open(path, getattr(os, "O_PATH", os.O_RDONLY) | os.O_CLOEXEC)
            try:
                rule = _PathBeneathAttr(access & supported, descriptor)
                if (
                    libc.syscall(
                        _SYSCALL_ADD_RULE, ruleset, _RULE_PATH_BENEATH, ctypes.byref(rule), 0
                    )
                    != 0
                ):
                    raise OSError(ctypes.get_errno(), f"landlock_add_rule failed for {path}")
            finally:
                os.close(descriptor)
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")
        if libc.syscall(_SYSCALL_RESTRICT_SELF, ruleset, 0) != 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
    finally:
        os.close(ruleset)


def _parse(argv: Sequence[str]) -> tuple[tuple[tuple[Path, int], ...], list[str]]:
    """Return the grants and the command a helper invocation carries."""
    if _RULES_OPTION not in argv:
        raise SystemExit(_USAGE)
    index = argv.index(_RULES_OPTION)
    payload: Any = json.loads(argv[index + 1])
    remainder = list(argv[index + 2 :])
    if not remainder or remainder[0] != "--" or len(remainder) == 1:
        raise SystemExit(_USAGE)
    return tuple((Path(entry[0]), int(entry[1])) for entry in payload), remainder[1:]


def main(argv: Sequence[str] | None = None) -> int:
    """Apply the policy to this process and exec the command it carries.

    This runs in the child, never in the answering process: a policy the kernel
    rejects costs one Tool call and cannot fail a Run. Silence is the contract in both
    cases — a host with no Landlock is recorded once per Run, and a kernel that offers
    Landlock and refuses it is the one degradation only the deployment's own seccomp
    and capability setup can see, because a word written here would be read by the
    model. ``execvp`` keeps the environment the caller already scrubbed, and the
    restriction survives it.
    """
    rules, command = _parse(list(sys.argv[1:] if argv is None else argv))
    abi = landlock_abi()
    if abi >= 1:
        try:
            _apply(rules, abi)
        except OSError:
            # Silence is deliberate. This process *is* the Agent's command, so a
            # diagnostic written here is handed to the model as tool output, and ADR
            # 0024 keeps the plumbing out of the tool result. A host's enforceable
            # state is reported once per Run instead; a kernel that offers Landlock
            # and refuses it for this process is the one degradation only the
            # deployment's own seccomp and capability setup can see.
            pass
    # Replacing this process IS the mechanism: argv is the command the caller
    # already resolved, and a shell here would only add one more program to confine.
    os.execvp(command[0], command)  # noqa: S606
    raise SystemExit("exec failed")  # unreachable: execvp replaces this process


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
