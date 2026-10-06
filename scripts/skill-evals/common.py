"""Helpers shared by the evaluation modules: paths, case loading, stack metadata, label directories, the pool, the Run lock."""

from __future__ import annotations

import fcntl
import hashlib
import itertools
import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import driver
from driver import Api

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]  # scripts/skill-evals -> the repository root
DEFAULT_CASES = HERE / "cases.json"
DEFAULT_SKILLS = REPO / "src" / "dlightrag" / "engine" / "agent" / "builtin_skills"
RESULTS = HERE / "results"
STATE = HERE / "state"
STACK_INFO = STATE / "stack.json"
MAX_CONCURRENCY = 3  # the provider's concurrency limit
API_CONTAINER = "dlightrag-eval-dlightrag-api-1"
IMAGE = "dlightrag:local"


# ---------------------------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------------------------


def load_cases(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(*parts: Any) -> str:
    """A short stable hash of JSON-able parts (a case definition, a Skill directory manifest)."""
    return hashlib.sha256(
        json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


# ---------------------------------------------------------------------------------------------
# What a directory holds: the same manifest on the host and (through docker exec) in the container
# ---------------------------------------------------------------------------------------------

_SKIP_DIRS = {"__pycache__", "node_modules", ".git"}
_SKIP_FILES = {".DS_Store"}

# Executed with the container's own Python: the same walk as `manifest()` below, so the two can be compared byte for byte.
CONTAINER_MANIFEST_PY = r"""
import hashlib, json, os, sys
root = sys.argv[1]
skip_dirs = {"__pycache__", "node_modules", ".git"}
out = {}
for base, dirs, files in os.walk(root):
    dirs[:] = sorted(d for d in dirs if d not in skip_dirs)
    for name in sorted(files):
        if name == ".DS_Store":
            continue
        path = os.path.join(base, name)
        try:
            data = open(path, "rb").read()
        except OSError as exc:
            out[os.path.relpath(path, root)] = "unreadable: " + str(exc)
            continue
        out[os.path.relpath(path, root)] = hashlib.sha256(data).hexdigest()
print(json.dumps(out, sort_keys=True))
"""


def manifest(root: Path) -> dict[str, str]:
    """{relative path: sha256} of every file under `root`, bytecode caches and node_modules excluded."""
    out: dict[str, str] = {}
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        for name in sorted(files):
            if name in _SKIP_FILES:
                continue
            path = Path(base) / name
            try:
                out[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                out[str(path.relative_to(root))] = f"unreadable: {exc}"
    return dict(sorted(out.items()))


def manifest_hash(root: Path | str | None) -> str | None:
    if not root or not Path(root).is_dir():
        return None
    return fingerprint(manifest(Path(root)))


def container_manifest(target: str) -> dict[str, str] | None:
    """The manifest of `target` as the eval API container sees it, or None when it cannot be read."""
    argv = [
        "docker",
        "exec",
        API_CONTAINER,
        "/app/.venv/bin/python",
        "-c",
        CONTAINER_MANIFEST_PY,
        target,
    ]
    try:
        done = subprocess.run(  # noqa: S603 - fixed docker argv against the eval project's own container
            argv, capture_output=True, text=True, timeout=120, check=True
        )
        return json.loads(done.stdout)
    except subprocess.SubprocessError, ValueError, OSError:
        return None


def image_id() -> str | None:
    """The id of the image the eval API runs (a baseline is only comparable on the same image)."""
    argv = ["docker", "inspect", API_CONTAINER, "--format", "{{.Image}}"]
    try:
        done = subprocess.run(  # noqa: S603 - fixed docker argv against the eval project's own container
            argv, capture_output=True, text=True, timeout=30, check=True
        )
        return done.stdout.strip().removeprefix("sha256:")[:12] or None
    except subprocess.SubprocessError, OSError:
        return None


# ---------------------------------------------------------------------------------------------
# The stack under test
# ---------------------------------------------------------------------------------------------


def stack_meta(api: Api) -> dict[str, Any]:
    """What the stack under test was started with, and what the API says it runs."""
    meta: dict[str, Any] = {}
    if STACK_INFO.exists():
        meta.update(json.loads(STACK_INFO.read_text()))
    health = api.health()
    meta["health_status"] = health.get("status")
    meta["answering_model"] = (health.get("answer_image_capability") or {}).get("model")
    skills = api.skills()
    meta["catalog"] = [skill["name"] for skill in skills]
    meta["catalog_detail"] = skills
    meta["image_id"] = image_id()
    meta["skills_manifest"] = manifest_hash(meta.get("skills_dir"))
    meta["toolkit_manifest"] = manifest_hash(meta.get("toolkit_dir"))
    return meta


def prepare_label_dir(label: str, resume: bool, kind: str = "routing") -> Path:
    """results/<label>/ may hold both a routing and a tasks evaluation; refuse to mix two of one kind."""
    out = RESULTS / label
    marker = {"routing": out / "routing.json", "tasks": out / "tasks"}[kind]
    runs = out / ("runs" if kind == "routing" else "tasks")
    if (marker.exists() and (kind == "routing" or any(marker.iterdir()))) or (
        runs.exists() and any(runs.iterdir())
    ):
        if not resume:
            sys.exit(
                f"evaluate.py: results/{label} already holds {kind} results; pass --resume to continue it "
                "or choose another --label"
            )
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------------------------
# Running things
# ---------------------------------------------------------------------------------------------


def run_in_pool(
    jobs: list[tuple[str, Any]], fn: Callable[[Any], Any], workers: int
) -> dict[str, Any]:
    """Run `fn(arg)` for every (key, arg) with at most `workers` at once; results keyed by `key`."""
    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(workers, MAX_CONCURRENCY))) as pool:
        futures = {pool.submit(fn, arg): key for key, arg in jobs}
        try:
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as exc:  # noqa: BLE001 - one failed job must not lose the others
                    print(
                        f"[evaluate] job {key} raised {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    results[key] = {"error": f"{type(exc).__name__}: {exc}"}
        except BaseException:
            for future in futures:
                future.cancel()
            driver.cancel_active_runs()
            raise
    return results


_LOCK_PATH = STATE / "runs.lock"
_lock_depth = 0
_lock_guard = threading.Lock()
_lock_fd: int | None = None


@contextmanager
def runs_lock(what: str) -> Iterator[None]:
    """At most one process submits Runs to the stack at a time: the provider allows 3 at once, and two evaluations would make 6.

    Re-entrant inside one process (variant runs routing and tasks under one hold). Judging saved reports needs no lock.
    """
    global _lock_depth, _lock_fd
    with _lock_guard:
        if _lock_depth == 0:
            STATE.mkdir(parents=True, exist_ok=True)
            fd = os.open(_LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                holder = ""
                try:
                    holder = os.pread(fd, 200, 0).decode(errors="replace").strip()
                except OSError:
                    pass
                os.close(fd)
                sys.exit(
                    f"evaluate.py: another evaluation is submitting Runs to the stack ({holder or 'unknown'}); "
                    "wait for it, or stop it (the 3-Run provider limit would be exceeded)"
                )
            os.ftruncate(fd, 0)
            os.pwrite(fd, f"pid {os.getpid()}: {what}".encode(), 0)
            _lock_fd = fd
        _lock_depth += 1
    try:
        yield
    finally:
        with _lock_guard:
            _lock_depth -= 1
            if _lock_depth == 0 and _lock_fd is not None:
                try:
                    fcntl.flock(_lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(_lock_fd)
                    _lock_fd = None


# ---------------------------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------------------------


def compress(names: list[str]) -> str:
    """['bash', 'search_web', 'search_web'] -> 'bash, search_web×2'."""
    out: list[str] = []
    for name, group in itertools.groupby(names):
        count = len(list(group))
        out.append(f"{name}×{count}" if count > 1 else name)
    return ", ".join(out) if out else "(none)"


def pct(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "n/a (0 samples)"


def rate(k: int, n: int) -> float | None:
    return k / n if n else None
