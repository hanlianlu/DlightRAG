"""Offline self-test of driver.py's follower against a fake API: what the evaluations rely on and cannot afford to get wrong.

  * the transcript window (the API returns only the last 100 messages) never loses a tool call, and a repeated snapshot
    never counts one twice;
  * an early stop fires at the first load_skill batch, at the tool-call cap, at the time cap, and cancels the Run;
  * a Run is cancelled on an exception inside the follower (the `finally`), and never left tracked;
  * a rate-limit-class failure backs off, is announced, and the Run is retried; any other failure is not retried;
  * a cancel that is not accepted at once is repeated until the Run is terminal.

Run: .venv/bin/python selftest_driver.py
"""

from __future__ import annotations

import sys
import time
from collections.abc import Set as AbstractSet
from typing import Any, cast

import driver
from driver import Api, EarlyStop, Transcript, follow_run


def check(condition: object, message: object = "") -> None:
    """An assertion that is not an `assert` statement: it survives `python -O` and the repo's security gate (S101) for scripts/."""
    if not condition:
        raise AssertionError(message)


driver.time.sleep = lambda s: None  # the follower's polling and back-off sleeps are not under test


def call(i: int, tool: str = "bash", **args: Any) -> dict[str, Any]:
    return {
        "id": f"c{i}",
        "name": tool,
        "arguments": args or {"command": f"echo {i}"},
        "argument_error": None,
    }


def assistant(*calls: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


def tool_result(c: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": c["id"],
        "name": c["name"],
        "content": "ok",
        "is_error": False,
    }


class FakeApi:
    """The slice of driver.Api that follow_run uses, driven by a script of (status, transcript) steps."""

    base_url = "http://fake"

    def __init__(
        self,
        steps: list[tuple[str, list[dict[str, Any]]]],
        *,
        final_result: dict[str, Any] | None = None,
        error: tuple[str, str] | None = None,
        cancel_after: int = 0,
    ) -> None:
        self.steps, self.i = steps, 0
        self.cancelled = 0
        self.cancel_after = cancel_after  # cancel requests ignored before the Run turns cancelled
        self.final_result = final_result
        self.error = error
        self.submitted = 0

    def submit(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.submitted += 1
        return {"run_id": f"run-{self.submitted}"}

    def _state(self) -> tuple[str, list[dict[str, Any]]]:
        return self.steps[min(self.i, len(self.steps) - 1)]

    def run(self, run_id: str) -> dict[str, Any]:
        status, _ = self._state()
        if self.cancelled and self.cancelled >= max(1, self.cancel_after):
            status = "cancelled"
        self.i += 1
        out = {
            "status": status,
            "error_kind": None,
            "error_message": None,
            "result": None,
            "started_at": None,
            "finished_at": None,
        }
        if status == "failed" and self.error:
            out["error_kind"], out["error_message"] = self.error
        if status == "succeeded":
            out["result"] = self.final_result or {
                "answer": "done",
                "artifacts": [],
                "trace": {"tool_observations": []},
                "usage": {},
            }
        return out

    def transcript(self, run_id: str, limit: int = 100) -> list[dict[str, Any]]:
        _, messages = self._state()
        return messages[-limit:]

    def cancel(self, run_id: str) -> dict[str, Any]:
        self.cancelled += 1
        return {"status": "running"}


def test_window_never_loses_calls() -> None:
    """150 calls arrive over time; every poll sees only the last 100 messages."""
    t = Transcript()
    history: list[dict[str, Any]] = [{"role": "user", "content": "q"}]
    seen = 0
    for _batch in range(1, 51):
        calls = [call(seen + 1), call(seen + 2), call(seen + 3)]
        seen += 3
        history.append(assistant(*calls))
        t.merge(history[-100:])  # the window right after the model turn commits
        history.extend(tool_result(c) for c in calls)
        t.merge(history[-100:])
        t.merge(history[-100:])  # a repeated snapshot adds nothing
    check(len(t.order) == 150, len(t.order))
    check([c.seq for c in t.tool_calls] == list(range(1, 151)))
    check(t.batches == 50)
    check(t.pending() == 0)


def test_skill_loads_and_paths() -> None:
    t = Transcript()
    a = call(1, "load_skill", name="charts")
    b = call(2, "load_skill", name="charts", path="references/x.md")
    t.merge([assistant(a, b), tool_result(a), tool_result(b)])
    loads = t.skill_loads()
    check(
        [(item["name"], item["path"]) for item in loads]
        == [("charts", "SKILL.md"), ("charts", "references/x.md")],
        loads,
    )


def test_early_stop_at_first_load_skill() -> None:
    first = assistant(call(1, "bash"), call(2, "load_skill", name="charts"))
    api = FakeApi(
        [
            ("running", [{"role": "user", "content": "q"}]),
            ("running", [{"role": "user", "content": "q"}, first]),
        ]
        * 1
        + [
            (
                "running",
                [first, tool_result(first["tool_calls"][0]), tool_result(first["tool_calls"][1])],
            )
        ]
        * 50
    )
    record = follow_run(api, "q", early_stop=EarlyStop(), label="t")  # type: ignore[arg-type]
    check(record.stopped_by == "first_load_skill", record.stopped_by)
    check([s["name"] for s in record.skills_loaded] == ["charts"])
    check(api.cancelled >= 1 and record.status == "cancelled", (api.cancelled, record.status))
    check(not driver._ACTIVE, driver._ACTIVE)  # noqa: SLF001


def test_early_stop_at_tool_call_cap_by_batch() -> None:
    b1 = assistant(*[call(i) for i in range(1, 4)])
    b2 = assistant(*[call(i) for i in range(4, 10)])  # 3 + 6 = 9 >= 8
    msgs1 = [{"role": "user", "content": "q"}, b1]
    msgs2 = msgs1 + [tool_result(c) for c in b1["tool_calls"]] + [b2]
    msgs3 = msgs2 + [tool_result(c) for c in b2["tool_calls"]]
    api = FakeApi([("running", msgs1), ("running", msgs2)] + [("running", msgs3)] * 50)
    record = follow_run(api, "q", early_stop=EarlyStop(max_tool_calls=8), label="t")  # type: ignore[arg-type]
    check(
        record.stopped_by == "max_tool_calls" and record.tool_count == 9,
        (record.stopped_by, record.tool_count),
    )
    check(api.cancelled >= 1)


def test_time_cap_cancels() -> None:
    api = FakeApi([("running", [{"role": "user", "content": "q"}])] * 1000)
    real = time.monotonic
    ticks = iter(range(0, 10_000, 40))  # every call to monotonic is 40 s later
    driver.time.monotonic = lambda: float(next(ticks))
    try:
        record = follow_run(api, "q", early_stop=EarlyStop(max_seconds=180), label="t")  # type: ignore[arg-type]
    finally:
        driver.time.monotonic = real
    check(record.stopped_by == "timeout", record.stopped_by)
    check(api.cancelled >= 1)


def test_exception_cancels_the_run() -> None:
    class Boom(FakeApi):
        def transcript(self, run_id: str, limit: int = 100) -> list[dict[str, Any]]:
            if self.i >= 2:
                raise ValueError("simulated")
            return super().transcript(run_id, limit)

    api = Boom([("running", [{"role": "user", "content": "q"}])] * 20)
    try:
        follow_run(api, "q", early_stop=None, timeout_s=300, label="t")  # type: ignore[arg-type]
    except ValueError:
        pass
    else:
        raise AssertionError("the exception must propagate")
    check(api.cancelled >= 1, "the Run must be cancelled on the way out")
    check(not driver._ACTIVE, driver._ACTIVE)  # noqa: SLF001


def test_cancel_is_repeated_until_terminal() -> None:
    first = assistant(call(1, "load_skill", name="charts"))
    msgs = [{"role": "user", "content": "q"}, first, tool_result(first["tool_calls"][0])]
    api = FakeApi(
        [("running", msgs)] * 200, cancel_after=3
    )  # the Run turns cancelled only on the 3rd cancel request
    real = time.monotonic
    clock = {"t": 0.0}

    def fake_monotonic() -> float:
        clock["t"] += (
            1.5  # every look at the clock is 1.5 s later: the 5 s repeat interval is reached
        )
        return clock["t"]

    driver.time.monotonic = fake_monotonic
    try:
        record = follow_run(api, "q", early_stop=EarlyStop(), label="t")  # type: ignore[arg-type]
    finally:
        driver.time.monotonic = real
    check(record.status == "cancelled" and api.cancelled >= 3, (record.status, api.cancelled))


def test_rate_limit_backs_off_and_retries() -> None:
    announced: list[tuple[float, str]] = []
    real_hit = driver.Throttle.hit
    driver.Throttle.hit = lambda self, delay, reason, label="": announced.append((delay, reason))  # type: ignore[assignment]
    attempts = {"n": 0}

    class Flaky(FakeApi):
        def run(self, run_id: str) -> dict[str, Any]:
            attempts["n"] = self.submitted
            if self.submitted == 1:
                return {
                    "status": "failed",
                    "error_kind": "model_error",
                    "error_message": "provider returned 429: concurrency limit",
                    "result": None,
                    "started_at": None,
                    "finished_at": None,
                }
            return super().run(run_id)

    api = Flaky([("succeeded", [{"role": "user", "content": "q"}])] * 20)
    try:
        record = follow_run(api, "q", early_stop=None, timeout_s=60, label="t")  # type: ignore[arg-type]
    finally:
        driver.Throttle.hit = real_hit  # type: ignore[assignment]
    check(record.attempts == 2 and record.status == "succeeded", (record.attempts, record.status))
    check(len(record.backoffs) == 1 and "429" in record.backoffs[0]["reason"], record.backoffs)
    check(announced and announced[0][0] > 0, announced)


def test_other_failures_are_not_retried() -> None:
    class Broken(FakeApi):
        def run(self, run_id: str) -> dict[str, Any]:
            return {
                "status": "failed",
                "error_kind": "tool_error",
                "error_message": "Research Agent operation ended as failed.",
                "result": None,
                "started_at": None,
                "finished_at": None,
            }

    api = Broken([("failed", [])] * 5)
    record = follow_run(api, "q", early_stop=None, timeout_s=60, label="t")  # type: ignore[arg-type]
    check(
        record.attempts == 1 and record.status == "failed" and not record.backoffs,
        (record.attempts, record.backoffs),
    )


def _calls(
    *batches: list[dict[str, Any]], failed: AbstractSet[str] = frozenset()
) -> list[driver.ToolCall]:
    """ToolCalls as the follower holds them: one batch per model turn, in order."""
    out: list[driver.ToolCall] = []
    for number, batch in enumerate(batches, start=1):
        for raw in batch:
            out.append(
                driver.ToolCall(
                    seq=len(out) + 1,
                    batch=number,
                    call_id=raw["id"],
                    name=raw["name"],
                    args=raw["arguments"],
                    is_error=True if raw["arguments"].get("name") in failed else False,
                )
            )
    return out


def load(i: int, name: str) -> dict[str, Any]:
    return call(i, "load_skill", name=name)


def test_stop_rule_all_required_skills_must_be_loaded() -> None:
    stop = EarlyStop(required=("interactive-html", "charts"))
    first = _calls([load(1, "interactive-html")])
    check(
        driver.routing_stop(first, stop) == "",
        "two required skills: the first batch alone is not the decision",
    )
    second = _calls([load(1, "interactive-html")], [load(2, "charts")])
    check(driver.routing_stop(second, stop) == "all_loaded")
    same_turn = _calls([load(1, "charts"), load(2, "interactive-html")])
    check(driver.routing_stop(same_turn, stop) == "all_loaded", "both in one model turn is enough")


def test_stop_rule_an_allowed_first_load_does_not_end_a_one_skill_case() -> None:
    stop = EarlyStop(
        required=("interactive-html",)
    )  # charts is `allow`ed: it is not required, and not a reason to stop
    check(driver.routing_stop(_calls([load(1, "charts")]), stop) == "")
    check(
        driver.routing_stop(_calls([load(1, "charts")], [load(2, "interactive-html")]), stop)
        == "all_loaded"
    )


def test_stop_rule_forbidden_wins_and_attempts_count() -> None:
    stop = EarlyStop(required=("charts",), forbidden=("office-documents",))
    both = _calls([load(1, "charts"), load(2, "office-documents")])
    check(
        driver.routing_stop(both, stop) == "forbidden_load",
        "a violation is reported even when the required skill is loaded too",
    )
    attempted = _calls([load(1, "office-documents")], failed={"office-documents"})
    check(
        driver.routing_stop(attempted, stop) == "forbidden_load",
        "a failed attempt to load a forbidden skill is still the model's decision",
    )


def test_stop_rule_a_failed_load_does_not_satisfy_a_required_skill() -> None:
    stop = EarlyStop(required=("charts",))
    check(driver.routing_stop(_calls([load(1, "charts")], failed={"charts"}), stop) == "")
    check(
        driver.routing_stop(_calls([load(1, "charts")], [load(2, "charts")]), stop) == "all_loaded"
    )


def test_stop_rule_without_required_skills_stops_at_the_first_load_turn() -> None:
    stop = EarlyStop(
        required=()
    )  # every skill the case expects is absent from the catalog (or it expects none)
    check(driver.routing_stop(_calls([call(1, "bash")]), stop) == "")
    check(
        driver.routing_stop(_calls([call(1, "bash"), load(2, "charts")]), stop)
        == "first_load_skill"
    )


def test_stop_rule_batches_override() -> None:
    required = EarlyStop(required=("a", "b"), batches=1)
    check(
        driver.routing_stop(_calls([load(1, "a")]), required) == "first_load_skill",
        "--skill-batches 1 is the old first-batch rule",
    )
    two = EarlyStop(required=("a", "b"), batches=2)
    check(driver.routing_stop(_calls([load(1, "a")]), two) == "")
    check(driver.routing_stop(_calls([load(1, "a")], [load(2, "x")]), two) == "skill_batches")


def test_stop_rule_cap_counts_requests_and_yields_to_a_decision() -> None:
    stop = EarlyStop(required=("charts",), max_tool_calls=3)
    check(driver.routing_stop(_calls([call(1), call(2), call(3)]), stop) == "max_tool_calls")
    check(
        driver.routing_stop(_calls([call(1), call(2), load(3, "charts")]), stop) == "all_loaded",
        "the decision wins over the cap in the same batch",
    )
    check(driver.routing_stop(_calls([call(1), call(2)]), stop) == "")
    unlimited = EarlyStop(required=("charts",), max_tool_calls=None)
    check(driver.routing_stop(_calls([call(i) for i in range(1, 60)]), unlimited) == "")


def test_follower_keeps_a_two_skill_case_running_until_the_second_load() -> None:
    first = assistant(call(1, "bash"), load(2, "interactive-html"))
    second = assistant(load(3, "charts"))
    msgs1 = [
        {"role": "user", "content": "q"},
        first,
        tool_result(first["tool_calls"][0]),
        tool_result(first["tool_calls"][1]),
    ]
    msgs2 = msgs1 + [second, tool_result(second["tool_calls"][0])]
    api = FakeApi([("running", msgs1)] * 3 + [("running", msgs2)] * 50)
    record = follow_run(
        cast(Api, api),
        "q",
        early_stop=EarlyStop(
            required=("interactive-html", "charts"), max_tool_calls=40, max_seconds=420
        ),
        label="t",
    )  # type: ignore[arg-type]
    check(record.stopped_by == "all_loaded", record.stopped_by)
    check(
        [s["name"] for s in record.skills_loaded] == ["interactive-html", "charts"],
        record.skills_loaded,
    )
    check(api.cancelled >= 1 and record.status == "cancelled")


def test_follower_stops_on_the_forbidden_load() -> None:
    first = assistant(load(1, "office-documents"))
    msgs = [{"role": "user", "content": "q"}, first, tool_result(first["tool_calls"][0])]
    api = FakeApi([("running", msgs)] * 50)
    record = follow_run(
        cast(Api, api),
        "q",
        early_stop=EarlyStop(required=("charts",), forbidden=("office-documents",)),
        label="t",
    )
    check(record.stopped_by == "forbidden_load", record.stopped_by)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"ok   {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} driver tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
