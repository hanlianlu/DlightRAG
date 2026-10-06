"""`evaluate.py variant`: evaluate a Skill variant with one command and read one table.

    evaluate.py variant --label v3 --skills-dir DIR [--toolkit-dir DIR] [--model deepseek|glm] [routing] [tasks]

What it does, in order, and what it refuses to do silently:

  1. Puts the mounts in place with `stack.sh up`, which is idempotent: Compose recreates the API container only when its
     configuration changed (another skills or toolkit directory, another model), and does nothing otherwise. Editing the files of the
     SAME directory needs no restart at all (the app lists the directory on every Run). The stack is brought up if it was down.
  2. Proves the container sees what the host holds: the sha256 manifest of the skills directory (and the toolkit directory) is
     computed on the host and inside the container and must be identical. A bind mount follows the directory it was made for, so
     `rm -rf DIR && cp ...` leaves the container looking at a deleted directory: that is caught here, the API container is recreated
     once, and a mismatch that survives aborts the run instead of measuring a Skill nobody wrote.
  3. Proves the API lists every Skill of the directory (a SKILL.md with a frontmatter the loader rejects is skipped silently by the app).
  4. Snapshots the Skill text and a manifest of the toolkit into results/<label>/, so a number can always be traced to its text.
  5. Runs routing, then tasks (never at the same time: the provider allows 3 Runs at once), with everything but the table going to
     results/<label>/logs/variant.log; writes SUMMARY.md; prints the comparison with the baseline.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import common
import compare
import routing
import summary
import tasks
from driver import Api

HERE = Path(__file__).resolve().parent
SKILLS_TARGET = "/app/.venv/lib/python3.14/site-packages/dlightrag/engine/agent/builtin_skills"
TOOLKIT_TARGET = "/usr/local/lib/echarts-render"
MODEL_NAMES = {"deepseek": "deepseek-flash", "glm": "z-ai/glm-5.3-flash"}
SNAPSHOT_MAX_BYTES = 2 * 1024 * 1024


class VariantError(SystemExit):
    """A refusal with a message for the person at the terminal."""


# ---------------------------------------------------------------------------------------------
# The stack
# ---------------------------------------------------------------------------------------------


def stack_sh(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    import os

    full = {**os.environ, **env}
    argv = [str(HERE / "stack.sh"), *args]
    return subprocess.run(argv, capture_output=True, text=True, env=full, timeout=900)  # noqa: S603 - this directory's stack.sh


def container_identity() -> str | None:
    """Container id and start time, or None when the eval API container does not exist."""
    argv = [
        "docker",
        "inspect",
        common.API_CONTAINER,
        "--format",
        "{{.Id}} {{.State.StartedAt}} {{.State.Running}}",
    ]
    done = subprocess.run(argv, capture_output=True, text=True)  # noqa: S603 - fixed docker argv; a missing container is reported by the return code
    return done.stdout.strip() if done.returncode == 0 and done.stdout.strip() else None


def container_mounts() -> dict[str, str]:
    argv = [
        "docker",
        "inspect",
        common.API_CONTAINER,
        "--format",
        "{{range .Mounts}}{{.Destination}}={{.Source}}\n{{end}}",
    ]
    done = subprocess.run(argv, capture_output=True, text=True)  # noqa: S603 - fixed docker argv; a missing container is reported by the return code
    out: dict[str, str] = {}
    for line in done.stdout.splitlines():
        dest, _, src = line.partition("=")
        if dest:
            out[dest] = src
    return out


def ensure_stack(skills_dir: Path, toolkit_dir: Path | None, model: str) -> dict[str, Any]:
    """`stack.sh up` with the wanted mounts and model; reports whether the API container was (re)created and why."""
    env = {
        "SKILLS_DIR": str(skills_dir),
        "TOOLKIT_DIR": str(toolkit_dir) if toolkit_dir else "",
        "QUERY_MODEL": model,
    }
    before = container_identity()
    done = stack_sh("up", env=env)
    if done.returncode != 0:
        tail = "\n".join((done.stdout + done.stderr).strip().splitlines()[-15:])
        raise VariantError(
            f"evaluate.py: the stack did not come up (stack.sh up exit {done.returncode}):\n{tail}"
        )
    after = container_identity()
    return {
        "was_down": before is None,
        "recreated": before != after,
        "before": before,
        "after": after,
    }


def content_problems(skills_dir: Path, toolkit_dir: Path | None) -> list[str]:
    """What differs between the host directories and what the container sees (empty: identical)."""
    problems: list[str] = []
    mounts = container_mounts()
    pairs = [("skills", skills_dir, SKILLS_TARGET)]
    if toolkit_dir:
        pairs.append(("toolkit", toolkit_dir, TOOLKIT_TARGET))
    elif TOOLKIT_TARGET in mounts:
        problems.append(f"the container still has a toolkit mounted from {mounts[TOOLKIT_TARGET]}")
    for name, host_dir, target in pairs:
        host = common.manifest(host_dir)
        seen = common.container_manifest(target)
        if seen is None:
            problems.append(f"{name}: cannot read {target} inside the container")
            continue
        if host == seen:
            continue
        only_host = sorted(set(host) - set(seen))
        only_box = sorted(set(seen) - set(host))
        changed = sorted(p for p in set(host) & set(seen) if host[p] != seen[p])
        detail = []
        if only_host:
            detail.append(f"only on the host: {', '.join(only_host[:4])}")
        if only_box:
            detail.append(f"only in the container: {', '.join(only_box[:4])}")
        if changed:
            detail.append(f"different content: {', '.join(changed[:4])}")
        problems.append(
            f"{name}: the container does not see what {host_dir} holds ({'; '.join(detail)})"
        )
    return problems


def skill_names(skills_dir: Path) -> dict[str, str]:
    """{directory name: the name its SKILL.md frontmatter declares (or the directory's own name)}."""
    out: dict[str, str] = {}
    for child in sorted(skills_dir.iterdir()):
        skill = child / "SKILL.md"
        if child.name.startswith(".") or not skill.is_file():
            continue
        head = skill.read_text(encoding="utf-8", errors="replace")[:8192]
        declared = child.name
        if head.startswith("---\n"):
            header = head[4:].partition("\n---")[0]
            match = re.search(r"^name:\s*['\"]?([^'\"\n#]+?)['\"]?\s*$", header, flags=re.MULTILINE)
            if match:
                declared = match.group(1).strip()
        out[child.name] = declared
    return out


def catalog_problems(skills_dir: Path) -> list[str]:
    listed = {s["name"] for s in Api().skills()}
    problems = []
    for directory, declared in skill_names(skills_dir).items():
        if declared not in listed and directory not in listed:
            problems.append(
                f"the API does not list '{declared}' (directory {directory}/): its SKILL.md is skipped by the loader. A frontmatter has to be "
                "YAML between '---' lines with a non-empty `description`, and values containing ': ' or ' #' need quotes"
            )
    return problems


def make_ready(skills_dir: Path, toolkit_dir: Path | None, model: str) -> dict[str, Any]:
    info = ensure_stack(skills_dir, toolkit_dir, model)
    problems = content_problems(skills_dir, toolkit_dir)
    if problems:  # a stale bind mount (the directory was replaced, not edited): recreate the API container once
        env = {
            "SKILLS_DIR": str(skills_dir),
            "TOOLKIT_DIR": str(toolkit_dir) if toolkit_dir else "",
            "QUERY_MODEL": model,
        }
        done = stack_sh("restart-api", env=env)
        if done.returncode != 0:
            raise VariantError(
                "evaluate.py: stack.sh restart-api failed:\n"
                + "\n".join((done.stdout + done.stderr).strip().splitlines()[-12:])
            )
        info["recreated"] = True
        info["after"] = container_identity()
        info["stale_mount_fixed"] = problems
        problems = content_problems(skills_dir, toolkit_dir)
    if problems:
        raise VariantError(
            "evaluate.py: the stack does not see the variant, so nothing was run:\n  "
            + "\n  ".join(problems)
        )
    health = Api().health()
    seen_model = (health.get("answer_image_capability") or {}).get("model")
    if seen_model != MODEL_NAMES[model]:
        raise VariantError(
            f"evaluate.py: the stack answers with '{seen_model}', not '{MODEL_NAMES[model]}' ({model}); nothing was run"
        )
    problems = catalog_problems(skills_dir)
    if problems:
        raise VariantError("evaluate.py: " + "\n  ".join(problems))
    return info


# ---------------------------------------------------------------------------------------------
# The result directory
# ---------------------------------------------------------------------------------------------


LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def check_label(label: str, resume: bool, replace: bool) -> None:
    """The refusals that cost nothing: they come before the stack is touched and before anything is written."""
    if not LABEL_RE.fullmatch(label):
        raise VariantError(
            f"evaluate.py: --label '{label}' must be a plain name (letters, digits, '.', '_', '-'; at most 64 characters)"
        )
    if resume and replace:
        raise VariantError("evaluate.py: --resume and --replace exclude each other")
    out = common.RESULTS / label
    if out.is_dir() and any(out.iterdir()) and not (resume or replace):
        raise VariantError(
            f"evaluate.py: results/{label} already exists; choose another --label, or pass --resume (keep its finished samples) "
            "or --replace (move it aside and start again)"
        )


def check_resume(out: Path, skills_hash: str | None, toolkit_hash: str | None, model: str) -> None:
    """A resumed label keeps its finished samples, so they must have been measured with the same Skill text, toolkit and model."""
    path = out / "variant.json"
    if not path.exists():
        return
    try:
        before = json.loads(path.read_text())
    except ValueError:
        return
    what = [
        name
        for name, key, now in (
            ("Skill text", "skills_manifest", skills_hash),
            ("toolkit", "toolkit_manifest", toolkit_hash),
            ("answering model", "model", model),
        )
        if key in before and before[key] != now
    ]
    if what:
        raise VariantError(
            f"evaluate.py: results/{out.name} was started with another {' and '.join(what)}; resuming would mix samples of two "
            "variants into one number. Use --replace (start again) or another --label"
        )


def prepare_label(label: str, replace: bool) -> Path:
    out = common.RESULTS / label
    if replace and out.is_dir() and any(out.iterdir()):
        moved = common.RESULTS / f"{label}.old-{time.strftime('%Y%m%d-%H%M%S')}"
        out.rename(moved)
        print(f"[variant] the earlier results of '{label}' are now {moved.name}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    return out


def snapshot(out: Path, skills_dir: Path, toolkit_dir: Path | None) -> dict[str, Any]:
    """Keep the Skill text next to its numbers; the toolkit as a manifest plus its small files (the ECharts library by hash only)."""
    snap: dict[str, Any] = {
        "skills_manifest": common.manifest_hash(skills_dir),
        "toolkit_manifest": common.manifest_hash(toolkit_dir),
    }
    size = sum(
        f.stat().st_size
        for f in skills_dir.rglob("*")
        if f.is_file() and "__pycache__" not in f.parts
    )
    if size <= SNAPSHOT_MAX_BYTES:
        dest = out / "skills-snapshot"
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(skills_dir, dest, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
        snap["skills_snapshot"] = str(dest)
    if toolkit_dir:
        files = common.manifest(toolkit_dir)
        (out / "toolkit-manifest.json").write_text(json.dumps(files, indent=1))
        dest = out / "toolkit-snapshot"
        shutil.rmtree(dest, ignore_errors=True)
        for rel in files:
            src = toolkit_dir / rel
            if (
                src.stat().st_size <= 300_000
                and not rel.startswith(("licenses/", "node_modules/"))
                and rel != "echarts.min.js"
            ):
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest / rel)
        snap["toolkit_snapshot"] = str(dest)
    return snap


# ---------------------------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------------------------


def _routing_opts(opts: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        cases_file=opts.cases_file,
        samples=opts.samples_routing,
        cases=opts.cases,
        label=opts.label,
        resume=opts.resume,
        max_tool_calls=None,
        max_seconds=None,
        skill_batches=None,
        concurrency=common.MAX_CONCURRENCY,
    )


def _tasks_opts(opts: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        cases_file=opts.cases_file,
        samples=opts.samples_tasks,
        tasks=opts.tasks,
        label=opts.label,
        resume=opts.resume,
        recheck=False,
        rerender=False,
        adopt=[],
        timeout=900.0,
        concurrency=common.MAX_CONCURRENCY,
        heights="800,1000",
    )


def fmt_seconds(seconds: float) -> str:
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


def selected_ids(spec: dict[str, Any], key: str, subset: str | None) -> list[str]:
    ids = [item["id"] for item in spec[key]]
    if not subset:
        return ids
    wanted = {part.strip() for part in subset.split(",") if part.strip()}
    unknown = sorted(wanted - set(ids))
    if unknown:
        raise VariantError(f"evaluate.py: unknown {key} ids {unknown}")
    return [i for i in ids if i in wanted]


def run_variant(opts: argparse.Namespace) -> int:
    skills_dir = opts.skills_dir.expanduser().resolve()
    if not skills_dir.is_dir() or not any(
        (c / "SKILL.md").is_file() for c in skills_dir.iterdir() if c.is_dir()
    ):
        raise VariantError(f"evaluate.py: --skills-dir {skills_dir} holds no <name>/SKILL.md")
    toolkit_dir = opts.toolkit_dir.expanduser().resolve() if opts.toolkit_dir else None
    if toolkit_dir and not (toolkit_dir / "echarts_render.py").is_file():
        raise VariantError(
            f"evaluate.py: --toolkit-dir {toolkit_dir} holds no echarts_render.py (it overlays /usr/local/lib/echarts-render)"
        )
    what = list(dict.fromkeys(opts.what)) or ["routing", "tasks"]
    spec = common.load_cases(opts.cases_file)
    n_cases = len(selected_ids(spec, "routing", opts.cases))
    n_tasks = len(selected_ids(spec, "tasks", opts.tasks))
    check_label(opts.label, opts.resume, opts.replace)
    if opts.resume:
        check_resume(
            common.RESULTS / opts.label,
            common.manifest_hash(skills_dir),
            common.manifest_hash(toolkit_dir),
            opts.model,
        )

    def say(line: str) -> None:
        print(line, flush=True)

    started = time.time()
    failures: list[str] = []
    durations: dict[str, float] = {}
    # The lock comes first and everything that changes state comes after readiness: a refusal (another evaluation is running, the stack does not
    # see the variant, a SKILL.md the loader rejects) leaves results/ exactly as it was, so the same command can simply be run again.
    with common.runs_lock(f"variant {opts.label}"):
        names = skill_names(skills_dir)
        say(
            f"[variant] {opts.label}: skills {', '.join(names.values())} from {skills_dir}"
            f" | toolkit {toolkit_dir or 'image default'} | model {opts.model} | against {opts.against}"
        )
        info = make_ready(skills_dir, toolkit_dir, opts.model)
        why = (
            "the stack was down: brought up"
            if info["was_down"]
            else (
                "the API container was recreated (mounts or model changed, or a stale mount was fixed)"
                if info["recreated"]
                else "no restart needed"
            )
        )
        say(
            f"[variant] stack: {why}; the container sees the skills"
            + (" and the toolkit" if toolkit_dir else "")
            + " exactly as the host holds them"
        )
        out = prepare_label(opts.label, opts.replace)
        (out / "logs").mkdir(exist_ok=True)
        logfile = (out / "logs" / "variant.log").open("a", buffering=1)

        terminal = sys.stderr

        def log(line: str) -> None:
            logfile.write(f"{time.strftime('%H:%M:%S')} {line}\n")
            if opts.verbose:
                print(line, file=terminal, flush=True)

        try:
            snap = snapshot(out, skills_dir, toolkit_dir)
            (out / "variant.json").write_text(
                json.dumps(
                    {
                        "label": opts.label,
                        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
                        "skills_dir": str(skills_dir),
                        "toolkit_dir": str(toolkit_dir) if toolkit_dir else None,
                        "model": opts.model,
                        "against": opts.against,
                        "what": what,
                        "stack": info,
                        **snap,
                    },
                    ensure_ascii=False,
                    indent=1,
                )
            )
            # quiet by default: what the evaluations and the driver write to stderr (progress, BACKOFF notices) goes to the log; --verbose shows it
            with contextlib.nullcontext() if opts.verbose else contextlib.redirect_stderr(logfile):
                if "routing" in what:
                    t0 = time.time()
                    say(
                        f"[variant] routing: {n_cases} case(s) x {opts.samples_routing} sample(s) ..."
                    )
                    try:
                        routing.run_routing(_routing_opts(opts), log=log, echo_report=False)
                    except SystemExit as exc:
                        failures.append(f"routing: {exc}")
                    except Exception as exc:  # noqa: BLE001 - tasks should still run, and the table should still print
                        failures.append(f"routing: {type(exc).__name__}: {exc}")
                        log(f"routing failed: {type(exc).__name__}: {exc}")
                    durations["routing"] = time.time() - t0
                    say(f"[variant] routing done in {fmt_seconds(durations['routing'])}")
                if "tasks" in what:
                    t0 = time.time()
                    say(
                        f"[variant] tasks: {n_tasks} task(s) x {opts.samples_tasks} sample(s), each Run up to 15 minutes ..."
                    )
                    try:
                        tasks.run_tasks(_tasks_opts(opts), log=log, echo_report=False)
                    except SystemExit as exc:
                        failures.append(f"tasks: {exc}")
                    except Exception as exc:  # noqa: BLE001
                        failures.append(f"tasks: {type(exc).__name__}: {exc}")
                        log(f"tasks failed: {type(exc).__name__}: {exc}")
                    durations["tasks"] = time.time() - t0
                    say(f"[variant] tasks done in {fmt_seconds(durations['tasks'])}")
        finally:
            logfile.close()
        summary.write_summary(opts.label, opts.cases_file)
        meta = json.loads((out / "variant.json").read_text())
        meta.update(
            {
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "durations_s": {k: round(v) for k, v in durations.items()},
                "failures": failures,
            }
        )
        (out / "variant.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        if (common.RESULTS / opts.against).is_dir():
            text = compare.compare_labels(
                opts.against,
                opts.label,
                opts.cases_file,
                color=sys.stdout.isatty() and not opts.no_color,
                sections=tuple(what),
            )
            plain = compare.compare_labels(
                opts.against, opts.label, opts.cases_file, color=False, sections=tuple(what)
            )
            (out / f"compare-vs-{opts.against}.md").write_text("```text\n" + plain + "\n```\n")
            say("")
            say(text.rstrip())
        else:
            say(
                f"[variant] no results for '{opts.against}' to compare with (run the baseline first); see {out / 'SUMMARY.md'}"
            )
        say(
            f"[variant] {opts.label} done in {fmt_seconds(time.time() - started)}: {out / 'SUMMARY.md'}  (log: {out / 'logs' / 'variant.log'})"
        )
        if opts.down_after:
            stack_sh("down", env={})
            say("[variant] stack down")
    for failure in failures:
        print(f"[variant] FAILED {failure}", file=sys.stderr)
    return 1 if failures else 0
