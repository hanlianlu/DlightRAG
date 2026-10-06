#!/usr/bin/env python3
"""The evaluations of the interactive-html Skill (Perplexity's rule: evals first).

  evaluate.py routing --samples 3 [--cases r01,n03] [--label NAME]
      Skill LOADING: every routing case, N Runs, stopped when the routing decision is visible; per-skill recall,
      precision (with `allow`), forbidden-load rate.
  evaluate.py tasks --samples 2 [--tasks t1-sales,t5-brief] [--label NAME] [--adopt LABEL:task,task]
      END-TO-END (and PROGRESSIVE loading): the tasks run to completion, the HTML artifact fetched and judged by the
      checker, the Skill reference files read recorded. `--adopt` re-judges saved Runs of another label instead of re-running.
  evaluate.py check PATH.html --viewports 360,390,820,1280 [--out DIR]
      The checker alone, on one HTML file, in the product's own sandbox frame.
  evaluate.py summary LABEL
      results/LABEL/SUMMARY.md: routing, tasks, boilerplate hits, who loaded what.
  evaluate.py compare LABEL_A LABEL_B
      Side by side: routing metrics per skill, then every task check; changes marked.
  evaluate.py variant --label NAME --skills-dir DIR [--toolkit-dir DIR] [--model deepseek|glm] [routing] [tasks]
      One command for iterating on the Skill: puts the mounts in place (restarting the stack only if they changed),
      runs the evaluations, writes the results, prints the comparison against another label (--against).

Results go to results/<label>/. The stack comes from stack.sh.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import common
import driver

HERE = Path(__file__).resolve().parent


def cmd_routing(opts: argparse.Namespace) -> int:
    import routing

    routing.run_routing(opts)
    return 0


def cmd_tasks(opts: argparse.Namespace) -> int:
    import tasks

    tasks.run_tasks(opts)
    return 0


def cmd_check(opts: argparse.Namespace) -> int:
    import tasks

    return tasks.run_check(opts)


def cmd_summary(opts: argparse.Namespace) -> int:
    import summary

    path = summary.write_summary(opts.label, opts.cases_file)
    print(path)
    return 0


def cmd_compare(opts: argparse.Namespace) -> int:
    import compare

    text = compare.compare_labels(
        opts.label_a, opts.label_b, opts.cases_file, color=sys.stdout.isatty() and not opts.no_color
    )
    print(text)
    return 0


def cmd_variant(opts: argparse.Namespace) -> int:
    import variant

    return variant.run_variant(opts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cases-file", type=Path, default=common.DEFAULT_CASES)
    sub = parser.add_subparsers(dest="cmd", required=True)

    routing = sub.add_parser("routing", help="Skill loading: precision, recall, forbidden-load")
    routing.add_argument("--samples", type=int, default=3)
    routing.add_argument("--cases", help="comma-separated case ids (default: all)")
    routing.add_argument("--label")
    routing.add_argument(
        "--resume",
        action="store_true",
        help="continue a label directory, skipping finished samples",
    )
    routing.add_argument(
        "--max-tool-calls",
        type=int,
        default=None,
        help="override every case's tool-call budget (default: the case's own, else 8)",
    )
    routing.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="override every case's time budget (default: the case's own, else 180)",
    )
    routing.add_argument(
        "--skill-batches",
        type=int,
        default=None,
        help="override the stop rule: stop after this many model turns that load a skill (default: when all required skills are loaded)",
    )
    routing.add_argument("--concurrency", type=int, default=common.MAX_CONCURRENCY)

    tasks_p = sub.add_parser("tasks", help="end-to-end tasks, judged by the checker")
    tasks_p.add_argument("--samples", type=int, default=2)
    tasks_p.add_argument("--tasks", help="comma-separated task ids to RUN (default: all)")
    tasks_p.add_argument("--label")
    tasks_p.add_argument(
        "--resume",
        action="store_true",
        help="continue a label directory: finished Runs are not repeated; reports judged by an older checker are judged again",
    )
    tasks_p.add_argument(
        "--recheck",
        action="store_true",
        help="judge the saved reports of an existing label again (no Run); implies --resume",
    )
    tasks_p.add_argument(
        "--rerender",
        action="store_true",
        help="rebuild tasks.md from the saved checks.json of an existing label: no Run, no checker",
    )
    tasks_p.add_argument(
        "--adopt",
        action="append",
        default=[],
        metavar="LABEL:TASK[,TASK]",
        help="copy the saved Runs of those tasks from another label and judge them with the current checker, instead of running them",
    )
    tasks_p.add_argument(
        "--timeout", type=float, default=900.0, help="seconds per Run (default 15 minutes)"
    )
    tasks_p.add_argument("--concurrency", type=int, default=common.MAX_CONCURRENCY)
    tasks_p.add_argument("--heights", default="800,1000")

    check = sub.add_parser("check", help="judge one HTML file in the product's sandbox frame")
    check.add_argument("path", type=Path)
    check.add_argument("--viewports", default="360,390,820,1280")
    check.add_argument("--heights", default="800,1000")
    check.add_argument("--out", type=Path)
    check.add_argument(
        "--kind",
        help="a task id or kind (e.g. t5-brief) to print that task's must verdict; default: generic",
    )

    summary = sub.add_parser("summary", help="write results/LABEL/SUMMARY.md")
    summary.add_argument("label")

    compare = sub.add_parser(
        "compare", help="side by side of two labels: routing metrics and every task check"
    )
    compare.add_argument("label_a", metavar="LABEL_A")
    compare.add_argument("label_b", metavar="LABEL_B")
    compare.add_argument("--no-color", action="store_true")

    variant = sub.add_parser(
        "variant", help="evaluate a Skill variant and compare it with the baseline"
    )
    variant.add_argument(
        "what",
        nargs="*",
        choices=["routing", "tasks"],
        help="which evaluations to run (default: both)",
    )
    variant.add_argument("--label", required=True)
    variant.add_argument(
        "--skills-dir",
        type=Path,
        default=common.DEFAULT_SKILLS,
        help="a directory in the built-in Skills layout (<name>/SKILL.md ...); default: this repository's built-in Skills",
    )
    variant.add_argument(
        "--toolkit-dir",
        type=Path,
        help="a directory overlaying /usr/local/lib/echarts-render (default: the image's own)",
    )
    variant.add_argument("--model", choices=["deepseek", "glm"], default="deepseek")
    variant.add_argument(
        "--against",
        default="baseline",
        help="the label to compare with (default baseline; none is needed)",
    )
    variant.add_argument("--samples-routing", type=int, default=3)
    variant.add_argument("--samples-tasks", type=int, default=2)
    variant.add_argument("--cases", help="only these routing cases")
    variant.add_argument("--tasks", help="only these tasks")
    variant.add_argument(
        "--resume", action="store_true", help="continue this label (finished samples are kept)"
    )
    variant.add_argument(
        "--replace",
        action="store_true",
        help="move an existing label aside (NAME.old-<time>) and start again",
    )
    variant.add_argument(
        "--down-after",
        action="store_true",
        help="take the stack down when finished (default: leave it up for the next call)",
    )
    variant.add_argument("--verbose", action="store_true", help="print progress lines")
    variant.add_argument("--no-color", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    opts, extra = parser.parse_known_args()
    if extra:
        # `variant routing --label X tasks`: argparse does not merge positionals that options separate, so the stragglers are taken here.
        stray = [a for a in extra if a in ("routing", "tasks")]
        if opts.cmd != "variant" or len(stray) != len(extra):
            parser.error(f"unrecognized arguments: {' '.join(extra)}")
        opts.what = [*opts.what, *stray]
    driver.install_exit_hooks()
    return {
        "routing": cmd_routing,
        "tasks": cmd_tasks,
        "check": cmd_check,
        "summary": cmd_summary,
        "compare": cmd_compare,
        "variant": cmd_variant,
    }[opts.cmd](opts)


if __name__ == "__main__":
    sys.exit(main())
