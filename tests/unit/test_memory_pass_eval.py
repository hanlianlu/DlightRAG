# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Session Memory Pass eval harness: its scoring rules, and one end-to-end run.

`scripts/run_memory_pass_eval.py` is the measurement of whether the pass is too conservative, and it
is only worth reading if the scoring is right: a hit counted for a violating write, or a
`min_matches` rule applied to the wrong writes, would make the summary say something the models did
not do. So each scoring rule is pinned here against fabricated records and judgements, and one run
through a scripted model and judge shows the harness fills a directory once and resumes without a
call. Nothing here calls a model.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "run_memory_pass_eval.py"
_spec = importlib.util.spec_from_file_location("run_memory_pass_eval", _SCRIPT)
assert _spec is not None and _spec.loader is not None
harness = importlib.util.module_from_spec(_spec)
sys.modules["run_memory_pass_eval"] = harness
_spec.loader.exec_module(harness)


async def _no_sleep(_seconds: float) -> None:
    return None


# ------------------------------------------------------------------------------- builders


def case_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "T1",
        "source": "synthetic",
        "note": "",
        "turns": [
            {"role": "user", "text": "以后回答都用中文。"},
            {"role": "assistant", "text": "好的。"},
        ],
        "candidates": [],
        "session_records": [],
        "record_limit": 5,
        "must": [{"id": "T1.m0", "kind": "preference", "description": "answers in Chinese"}],
        "may": [],
        "must_not": ["the task"],
        "expect_actions": [],
    }
    return {**base, **overrides}


def make_case(**overrides: Any) -> Any:
    return harness.Case.model_validate(case_dict(**overrides))


def w(
    body: str = "The owner prefers answers in Chinese.",
    *,
    action: str = "add",
    kind: str = "preference",
    targets: tuple[str, ...] = (),
) -> Any:
    return harness.WriteRecord(action=action, kind=kind, body=body, target_ids=list(targets))


def record(
    *,
    planned: list[Any] | None = None,
    case_id: str = "T1",
    config: str = "ds-flash-off",
    label: str = "prod",
    rep: int = 1,
    status: str = "ok",
    parse_error: str | None = None,
    latency: float = 1.0,
    tokens: int = 100,
    prompt_sha: str = "sha",
    error: str | None = None,
) -> Any:
    planned = planned or []
    return harness.RunRecord(
        config=config,
        label=label,
        case_id=case_id,
        rep=rep,
        status=status,
        prompt_sha=prompt_sha,
        started_at="2026-10-17T00:00:00+00:00",
        latency_s=None if status == "error" else latency,
        verdict=[
            harness.ProposedWrite(action=write.action, kind=write.kind, body=write.body)
            for write in planned
        ],
        planned=planned,
        usage=None if status == "error" else {"total_tokens": tokens},
        parse_error=parse_error,
        error=error,
    )


def entry(
    index: int,
    status: str,
    match: str | None = None,
    also: tuple[str, ...] = (),
    reason: str = "because",
) -> Any:
    return harness.JudgeEntry(
        index=index, status=status, match=match, also_matches=list(also), reason=reason
    )


# ---------------------------------------------------------------------------- case loading


# ----------------------------------------------------------------------- request building


# ----------------------------------------------------------------------------- the matrix


# ------------------------------------------------------------------------------- one run


class _Scripted:
    """A `Verdicter` that plays back replies (text or exception), repeating the last one."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.calls = 0
        self.closed = False

    async def __call__(self, messages: list[dict[str, str]]) -> Any:
        self.calls += 1
        item = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(item, BaseException):
            raise item
        return harness.ModelReply(
            text=item,
            usage={"prompt_tokens": 4, "completion_tokens": 3},
            wire=[{"response_format": "json_object", "reasoning": {}}],
        )

    async def aclose(self) -> None:
        self.closed = True


async def _run(model: Any, case: Any = None, **kwargs: Any) -> Any:
    case = case or make_case()
    return await harness.run_one(
        model,
        config="ds-flash-off",
        label="prod",
        case=case,
        rep=1,
        request=harness.build_request(case),
        sleep=_no_sleep,
        **kwargs,
    )


async def test_a_secret_never_reaches_a_record() -> None:
    model = _Scripted(RuntimeError("bad key sk-secret-123"))

    result = await _run(model, secrets=["sk-secret-123"])

    assert result.error is not None and "sk-secret-123" not in result.error
    assert "***" in result.error


# ------------------------------------------------------------------------- judge contract


def test_a_judgement_must_cover_each_write_once() -> None:
    case = make_case(may=[{"id": "T1.y0", "kind": "fact", "description": "d"}])
    planned = [w(), w("The owner is in Oslo.", kind="fact")]
    good = harness.JudgeVerdict(
        judgements=[entry(1, "acceptable"), entry(0, "match", "T1.m0", ("T1.y0",))]
    )
    harness.validate_judgement(case, planned, good)

    for bad in (
        [entry(0, "acceptable")],
        [entry(0, "acceptable"), entry(0, "acceptable")],
        [entry(0, "acceptable"), entry(2, "acceptable")],
        [entry(0, "match", "T9"), entry(1, "acceptable")],
        [entry(0, "match", "T1.m0", ("nope",)), entry(1, "acceptable")],
        [entry(0, "match"), entry(1, "acceptable")],
        [entry(0, "acceptable", "T1.m0"), entry(1, "acceptable")],
        [entry(0, "violation", "T1.m0", ("T1.y0",)), entry(1, "acceptable")],
    ):
        with pytest.raises(ValueError):
            harness.validate_judgement(case, planned, harness.JudgeVerdict(judgements=bad))
    # a violation may still name the item it phrased badly
    harness.validate_judgement(
        case,
        planned,
        harness.JudgeVerdict(judgements=[entry(0, "violation", "T1.m0"), entry(1, "acceptable")]),
    )


def _verdict_text(*entries: dict[str, Any]) -> str:
    return json.dumps({"judgements": list(entries)})


# ------------------------------------------------------------------------------- scoring


def test_a_must_item_is_hit_by_a_matching_write_and_missed_by_nothing() -> None:
    case = make_case()

    hit = harness.score_run(case, record(planned=[w()]), [entry(0, "match", "T1.m0")])
    miss = harness.score_run(case, record(planned=[]), None)
    other = harness.score_run(case, record(planned=[w("x")]), [entry(0, "acceptable")])

    assert hit.must_hits == {"T1.m0": True}
    assert miss.must_hits == {"T1.m0": False} and miss.judged and miss.scored
    assert other.must_hits == {"T1.m0": False} and other.acceptable == 1


def test_min_matches_needs_that_many_distinct_writes() -> None:
    case = make_case(
        must=[{"id": "T1.m0", "kind": "preference", "description": "d", "min_matches": 3}]
    )
    writes = [w(f"The owner prefers rule {i}.") for i in range(4)]

    two = harness.score_run(
        case,
        record(planned=writes[:2]),
        [entry(0, "match", "T1.m0"), entry(1, "match", "T1.m0")],
    )
    three = harness.score_run(
        case,
        record(planned=writes[:4]),
        [
            entry(0, "match", "T1.m0"),
            entry(1, "match", "T1.m0"),
            entry(2, "acceptable"),
            entry(3, "match", "T1.m0"),
        ],
    )

    assert two.must_hits == {"T1.m0": False}
    assert three.must_hits == {"T1.m0": True}


def test_one_write_may_carry_several_items_and_a_violation_carries_none() -> None:
    case = make_case(
        must=[
            {"id": "T1.m0", "kind": "preference", "description": "Chinese"},
            {"id": "T1.m1", "kind": "preference", "description": "English comments"},
        ]
    )

    merged = harness.score_run(
        case, record(planned=[w()]), [entry(0, "match", "T1.m0", ("T1.m1",))]
    )
    phrased_as_order = harness.score_run(
        case, record(planned=[w("Always answer in Chinese.")]), [entry(0, "violation", "T1.m0")]
    )

    assert merged.must_hits == {"T1.m0": True, "T1.m1": True}
    assert merged.matches == 1
    assert phrased_as_order.must_hits == {"T1.m0": False, "T1.m1": False}
    assert phrased_as_order.violations == 1


def test_an_expected_action_needs_the_right_action_on_the_same_targets() -> None:
    case = make_case(
        candidates=[
            {"memory_id": "a", "kind": "preference", "body": "x"},
            {"memory_id": "b", "kind": "preference", "body": "y"},
        ],
        expect_actions=[{"gold": "T1.m0", "action": "fuse", "target_ids": ["a", "b"]}],
    )

    def score(planned: list[Any]) -> Any:
        return harness.score_run(
            case, record(planned=planned), [entry(0, "match", "T1.m0")][: len(planned)]
        )

    right = score([w(action="fuse", targets=("b", "a"))])
    wrong_action = score([w(action="add")])
    wrong_targets = score([w(action="fuse", targets=("a",))])
    nothing = score([])

    assert (right.expect_matched, right.expect_correct) == (1, 1)
    assert (wrong_action.expect_matched, wrong_action.expect_correct) == (1, 0)
    assert (wrong_targets.expect_matched, wrong_targets.expect_correct) == (1, 0)
    assert (nothing.expect_total, nothing.expect_matched, nothing.expect_correct) == (1, 0, 0)


def test_a_run_with_writes_but_no_judgement_earns_nothing() -> None:
    score = harness.score_run(make_case(), record(planned=[w()]), None)

    assert not score.judged and not score.scored
    assert score.must_hits == {"T1.m0": False}


def test_an_errored_run_is_not_scored_and_a_parse_failure_is() -> None:
    case = make_case()

    errored = harness.score_run(case, record(status="error", error="x"), None)
    parse_failed = harness.score_run(case, record(parse_error="bad"), None)

    assert errored.errored and not errored.scored and errored.tokens == 0
    assert parse_failed.parse_failed and parse_failed.scored


def test_percentile_is_nearest_rank() -> None:
    assert harness.percentile([], 0.95) is None
    assert harness.percentile([3.0], 0.95) == 3.0
    assert harness.percentile([float(n) for n in range(1, 21)], 0.95) == 19.0
    assert harness.percentile([float(n) for n in range(1, 101)], 0.95) == 95.0


def _scored_fixture() -> tuple[dict[str, Any], dict[Any, Any], dict[str, Any]]:
    """Two cases (one real with a must, one synthetic without), two configs, two reps."""
    real = make_case(id="R1", source="real")
    synthetic = make_case(id="S1", must=[], may=[])
    cases = {"R1": real, "S1": synthetic}
    results: dict[Any, Any] = {}
    judgements: dict[str, Any] = {}

    def put(rec: Any, entries: list[Any] | None = None) -> None:
        results[rec.key] = rec
        if entries is not None:
            case = cases[rec.case_id]
            judgements[harness.judge_key(case, rec.planned)] = harness.JudgementRecord(
                key=harness.judge_key(case, rec.planned),
                case_id=rec.case_id,
                writes=rec.planned,
                judgements=entries,
                judge_model="judge",
                judged_at="2026-10-17T00:00:00+00:00",
            )

    # config A: finds the real must once of two reps; writes a violation on the synthetic no-must
    put(
        record(config="A", case_id="R1", rep=1, planned=[w()], latency=1.0, tokens=10),
        [entry(0, "match", "T1.m0")],
    )
    put(record(config="A", case_id="R1", rep=2, planned=[], latency=3.0, tokens=10))
    put(
        record(
            config="A", case_id="S1", rep=1, planned=[w("The task was X.")], latency=2.0, tokens=10
        ),
        [entry(0, "violation", None, reason="a task detail")],
    )
    put(
        record(
            config="A", case_id="S1", rep=2, planned=[], parse_error="bad", latency=4.0, tokens=10
        )
    )
    # config B: empty everywhere, plus one transport error
    put(record(config="B", case_id="R1", rep=1, planned=[], latency=1.0, tokens=5))
    put(record(config="B", case_id="R1", rep=2, status="error", error="429"))
    put(record(config="B", case_id="S1", rep=1, planned=[], latency=1.0, tokens=5))
    put(
        record(
            config="B", case_id="S1", rep=2, planned=[w("The owner lives in Oslo.", kind="fact")]
        )
    )
    return cases, results, judgements


def test_the_summary_rates_follow_the_scored_runs() -> None:
    cases, results, judgements = _scored_fixture()

    evaluation = harness.evaluate(results, cases, judgements)
    by_config = {s.config: s for s in evaluation.summaries}
    a, b = by_config["A"], by_config["B"]

    assert (a.runs, a.errored, a.scored) == (4, 0, 4)
    assert (a.must_hit, a.must_total) == (1, 2)
    assert (a.must_hit_real, a.must_total_real) == (1, 2)
    assert (a.must_hit_synthetic, a.must_total_synthetic) == (0, 0)
    assert a.must_recall == 0.5
    assert a.violations == 1 and a.violation_runs == 1
    assert (a.violation_runs_no_must, a.scored_no_must) == (1, 2)
    assert a.parse_failures == 1 and a.parse_failure_rate == 0.25
    assert a.empty_runs == 1  # R1 rep 2; the parse failure is not an empty answer
    assert a.latency_mean == 2.5 and a.latency_p95 == 4.0
    assert a.tokens == 40

    assert (b.runs, b.errored, b.scored) == (4, 1, 2)
    assert b.unjudged == 1  # the S1 write nobody judged
    assert (b.must_hit, b.must_total) == (0, 1)
    assert b.empty_runs == 2
    assert b.parse_failure_rate == 0
    assert evaluation.case_counts == {"total": 2, "real": 1, "synthetic": 1}
    assert [row.body for row in evaluation.violations] == ["The task was X."]
    assert [r.config for r in evaluation.errors] == ["B"]
    assert len(evaluation.unjudged) == 1


def test_per_must_item_hit_rates_count_each_scored_rep() -> None:
    cases, results, judgements = _scored_fixture()

    table = harness.must_item_table(
        [
            harness.score_run(
                cases[r.case_id],
                r,
                (
                    judgements[harness.judge_key(cases[r.case_id], r.planned)].judgements
                    if r.planned and harness.judge_key(cases[r.case_id], r.planned) in judgements
                    else None
                ),
            )
            for r in results.values()
        ]
    )

    assert table == {
        ("R1", "T1.m0"): {("A", "prod"): (1, 2), ("B", "prod"): (0, 1)},
    }


# --------------------------------------------------------------------------- resumability


class _JudgeByIndex:
    """A fake judge: matches the first write to T1.m0 and finds the rest acceptable."""

    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    async def __call__(self, messages: list[dict[str, str]]) -> Any:
        self.calls += 1
        lines = [ln for ln in messages[1]["content"].splitlines() if ln.startswith("{")]
        entries = [
            {
                "index": i,
                "match": "T1.m0" if i == 0 else None,
                "also_matches": [],
                "status": "match" if i == 0 else "acceptable",
                "reason": "r",
            }
            for i in range(len(lines))
        ]
        return harness.ModelReply(text=_verdict_text(*entries))

    async def aclose(self) -> None:
        self.closed = True


async def test_a_directory_is_filled_once_and_a_rerun_makes_no_calls(tmp_path: Path) -> None:
    case = make_case()
    said = '{"writes": [{"action": "add", "kind": "preference", "body": "The owner prefers Chinese."}]}'
    models: dict[str, _Scripted] = {}

    def factory(config: Any, structured: Any) -> _Scripted:
        models[config.name] = _Scripted(said)
        return models[config.name]

    judge = _JudgeByIndex()
    quiet = lambda _line: None  # noqa: E731

    async def fill() -> Any:
        done = harness.load_results(tmp_path / harness.RESULTS_FILE)
        pending = harness.pending_runs(
            harness.select_configs("ds-flash-off,sonnet"),
            [case],
            reps=2,
            label="prod",
            system_prompt=None,
            done=done,
        )
        made = await harness.run_matrix(
            pending, label="prod", out_dir=tmp_path, factory=factory, concurrency=3, log=quiet
        )
        evaluation, failures = await harness.finalize(
            tmp_path,
            {case.id: case},
            judge_factory=lambda: judge,
            concurrency=2,
            title="t",
            log=quiet,
        )
        return made, evaluation, failures

    made, evaluation, failures = await fill()

    assert len(made) == 4 and failures == []
    assert sum(model.calls for model in models.values()) == 4
    assert all(model.closed for model in models.values())
    assert judge.calls == 1  # four identical outputs, one verdict
    assert len((tmp_path / harness.JUDGEMENTS_FILE).read_text().splitlines()) == 1
    assert len((tmp_path / harness.RESULTS_FILE).read_text().splitlines()) == 4
    assert {s.config: s.must_hit for s in evaluation.summaries} == {"ds-flash-off": 2, "sonnet": 2}
    summary = (tmp_path / harness.SUMMARY_MD).read_text()
    assert "| ds-flash-off | 2 | 2/2 (100%) |" in summary
    assert json.loads((tmp_path / harness.SUMMARY_JSON).read_text())["cases"]["total"] == 1

    models.clear()
    made, evaluation, failures = await fill()

    assert made == [] and models == {} and judge.calls == 1
    assert len((tmp_path / harness.RESULTS_FILE).read_text().splitlines()) == 4
