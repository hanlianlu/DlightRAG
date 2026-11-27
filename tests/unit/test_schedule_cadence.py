# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The cadence of a Schedule: five-field cron on a zone's wall clock, and one instant (ADR 0103)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from dlightrag.application.schedules.cadence import (
    Cadence,
    CadenceError,
    CronExpression,
    exists,
    utc_of,
)

STOCKHOLM = ZoneInfo("Europe/Stockholm")
NEW_YORK = ZoneInfo("America/New_York")
HOUR = timedelta(hours=1)


def _cron(text: str) -> CronExpression:
    return CronExpression.parse(text)


def _local_after(text: str, after: datetime) -> datetime:
    return _cron(text).next_local(after)


@pytest.mark.parametrize(
    ("text", "after", "expected"),
    [
        # A step over a field, and the minute that follows the last one of the hour.
        ("*/15 * * * *", datetime(2026, 5, 1, 10, 7), datetime(2026, 5, 1, 10, 15)),
        ("*/15 * * * *", datetime(2026, 5, 1, 10, 45), datetime(2026, 5, 1, 11, 0)),
        # A list, a range, and a range with a step.
        ("0 9,17 * * *", datetime(2026, 5, 1, 9, 0), datetime(2026, 5, 1, 17, 0)),
        ("30 8-10 * * *", datetime(2026, 5, 1, 10, 30), datetime(2026, 5, 2, 8, 30)),
        ("0 8-18/5 * * *", datetime(2026, 5, 1, 8, 0), datetime(2026, 5, 1, 13, 0)),
        # A start with a step runs to the end of the field.
        ("10/20 * * * *", datetime(2026, 5, 1, 10, 10), datetime(2026, 5, 1, 10, 30)),
        # The day of week: Monday 08:00, and 0 and 7 both mean Sunday.
        ("0 8 * * 1", datetime(2026, 5, 1, 0, 0), datetime(2026, 5, 4, 8, 0)),
        ("0 8 * * 0", datetime(2026, 5, 1, 0, 0), datetime(2026, 5, 3, 8, 0)),
        ("0 8 * * 7", datetime(2026, 5, 1, 0, 0), datetime(2026, 5, 3, 8, 0)),
        ("0 8 * * 5-7", datetime(2026, 5, 2, 9, 0), datetime(2026, 5, 3, 8, 0)),
        # The month and the day of month cross a year end.
        ("0 0 1 1 *", datetime(2026, 5, 1, 0, 0), datetime(2027, 1, 1, 0, 0)),
        ("0 0 31 * *", datetime(2026, 4, 1, 0, 0), datetime(2026, 5, 31, 0, 0)),
        # February 29 waits for a leap year.
        ("0 0 29 2 *", datetime(2026, 3, 1, 0, 0), datetime(2028, 2, 29, 0, 0)),
    ],
)
def test_the_next_wall_clock_minute_is_found_field_by_field(
    text: str, after: datetime, expected: datetime
) -> None:
    assert _local_after(text, after) == expected


def test_a_slot_is_strictly_after_the_instant_asked() -> None:
    """The minute the search starts from is never its own answer, seconds and all."""
    assert _local_after("0 * * * *", datetime(2026, 5, 1, 10, 0, 30)) == datetime(2026, 5, 1, 11, 0)


def test_when_both_day_fields_are_restricted_either_one_matches() -> None:
    """The standard rule: the 1st of the month, or any Monday."""
    expression = _cron("0 8 1 * 1")
    days = []
    cursor = datetime(2026, 5, 1, 0, 0)
    for _ in range(4):
        cursor = expression.next_local(cursor)
        days.append(cursor.date().isoformat())
    assert days == ["2026-05-01", "2026-05-04", "2026-05-11", "2026-05-18"]


def test_when_one_day_field_is_unrestricted_the_other_must_match() -> None:
    """A field that starts with a star matches everything, so a day needs the restricted one."""
    assert _local_after("0 8 */2 * 1", datetime(2026, 5, 1, 0, 0)) == datetime(2026, 5, 11, 8, 0)


@pytest.mark.parametrize(
    "text",
    [
        "@hourly",
        "* * * *",
        "* * * * * *",
        "60 * * * *",
        "* 24 * * *",
        "* * 0 * *",
        "* * 32 * *",
        "* * * 13 *",
        "* * * * 8",
        "5-1 * * * *",
        "*/0 * * * *",
        "*/x * * * *",
        "1,,2 * * * *",
        "0 8 * * MON",
        "0 8 * JAN *",
    ],
)
def test_anything_but_five_numeric_fields_is_refused(text: str) -> None:
    with pytest.raises(CadenceError):
        _cron(text)


def test_a_macro_and_a_name_are_refused_with_the_reason() -> None:
    with pytest.raises(CadenceError, match="macros such as @hourly"):
        _cron("@hourly")
    with pytest.raises(CadenceError, match="numbers, not names"):
        _cron("0 8 * * MON")


def test_an_expression_that_never_matches_a_date_is_refused() -> None:
    with pytest.raises(CadenceError, match="never matches"):
        _cron("0 0 31 2 *").next_local(datetime(2026, 1, 1))


# -- the wall clock and the zone ------------------------------------------------------------


def test_a_nonexistent_local_time_fires_at_the_instant_the_clock_jumps() -> None:
    """02:30 on 29 March 2026 does not exist in Stockholm: the clock goes from 02:00 to 03:00."""
    jumped = utc_of(datetime(2026, 3, 29, 2, 30), STOCKHOLM)
    assert jumped == datetime(2026, 3, 29, 1, 0, tzinfo=UTC)
    assert jumped.astimezone(STOCKHOLM).hour == 3

    # New York: 02:30 on 8 March 2026 does not exist; the clock goes from 02:00 to 03:00 EDT.
    assert utc_of(datetime(2026, 3, 8, 2, 30), NEW_YORK) == datetime(2026, 3, 8, 7, 0, tzinfo=UTC)


def test_an_ambiguous_local_time_fires_once_at_its_first_occurrence() -> None:
    """02:30 on 25 October 2026 occurs twice in Stockholm; the first is still on summer time."""
    first = utc_of(datetime(2026, 10, 25, 2, 30), STOCKHOLM)
    assert first == datetime(2026, 10, 25, 0, 30, tzinfo=UTC)

    # New York: 01:30 on 1 November 2026 occurs twice; the first is EDT.
    assert utc_of(datetime(2026, 11, 1, 1, 30), NEW_YORK) == datetime(
        2026, 11, 1, 5, 30, tzinfo=UTC
    )

    cadence = Cadence("Europe/Stockholm", _cron("30 2 * * *"), None)
    slots = []
    cursor = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)
    for _ in range(3):
        cursor = cadence.next_slot(cursor) or cursor
        slots.append(cursor)
    assert slots == [
        datetime(2026, 10, 25, 0, 30, tzinfo=UTC),
        # The second 02:30 of that night is not a slot: the next one is the next day's.
        datetime(2026, 10, 26, 1, 30, tzinfo=UTC),
        datetime(2026, 10, 27, 1, 30, tzinfo=UTC),
    ]


def test_a_recurrence_keeps_its_wall_clock_across_a_daylight_saving_change() -> None:
    """Monday 08:00 in Stockholm is 07:00 UTC in winter and 06:00 UTC after the clocks go forward."""
    cadence = Cadence("Europe/Stockholm", _cron("0 8 * * 1"), None)
    # The clocks go forward in the night of 28 to 29 March 2026.
    assert cadence.next_slot(datetime(2026, 3, 16, 12, 0, tzinfo=UTC)) == datetime(
        2026, 3, 23, 7, 0, tzinfo=UTC
    )
    assert cadence.next_slot(datetime(2026, 3, 23, 12, 0, tzinfo=UTC)) == datetime(
        2026, 3, 30, 6, 0, tzinfo=UTC
    )


def test_a_gap_that_several_minutes_fall_into_makes_one_slot() -> None:
    """Every minute of 02:00-02:50 is skipped on the spring night; none of them fires twice."""
    cadence = Cadence("Europe/Stockholm", _cron("*/10 2,3 * * *"), None)
    cursor = datetime(2026, 3, 28, 12, 0, tzinfo=UTC)
    seen: list[datetime] = []
    while True:
        cursor = cadence.next_slot(cursor) or cursor
        if cursor.date() > datetime(2026, 3, 29).date():
            break
        seen.append(cursor)
    assert seen[0] == datetime(2026, 3, 29, 1, 0, tzinfo=UTC)
    assert seen == sorted(set(seen))
    assert len(seen) == 6  # 03:00, 03:10 ... 03:50 local, the 02:xx minutes folded into 03:00


# -- validation -----------------------------------------------------------------------------


def _create(cron: str | None = None, at: datetime | None = None, zone: str = "UTC") -> Cadence:
    return Cadence.create(
        zone_name=zone,
        cron=cron,
        at=at,
        now=datetime(2026, 11, 27, 12, 0, tzinfo=UTC),
        min_interval=timedelta(minutes=60),
    )


def test_a_recurrence_closer_than_the_minimum_is_refused() -> None:
    with pytest.raises(CadenceError, match="at least 60 minutes apart"):
        _create("*/30 * * * *")
    with pytest.raises(CadenceError, match="at least 60 minutes apart"):
        _create("0,20 9 * * *")
    assert _create("0 * * * *").cron == "0 * * * *"
    assert _create("0 9 * * 1-5").cron == "0 9 * * 1-5"


def test_the_minimum_is_measured_on_the_real_clock_not_the_wall_clock() -> None:
    """01:30 and 03:30 are two wall-clock hours apart, but one real hour on the night clocks go forward."""
    kwargs = {
        "zone_name": "Europe/Stockholm",
        "cron": "30 1,3 * * *",
        "at": None,
        "now": datetime(2026, 11, 27, 12, 0, tzinfo=UTC),
    }
    assert Cadence.create(**kwargs, min_interval=HOUR).cron == "30 1,3 * * *"
    with pytest.raises(CadenceError, match="2027-03-28"):
        Cadence.create(**kwargs, min_interval=timedelta(minutes=90))


def test_a_cadence_names_a_real_zone_and_exactly_one_of_a_recurrence_or_a_time() -> None:
    with pytest.raises(CadenceError, match="IANA time zone"):
        _create("0 * * * *", zone="Mars/Olympus")
    with pytest.raises(CadenceError, match="IANA time zone"):
        _create("0 * * * *", zone="../etc/passwd")
    with pytest.raises(CadenceError, match="not both"):
        _create("0 * * * *", at=datetime(2027, 1, 1, tzinfo=UTC))
    with pytest.raises(CadenceError, match="needs a recurrence or a single time"):
        _create()


def test_a_single_time_must_be_aware_and_in_the_future() -> None:
    now = datetime(2026, 11, 27, 12, 0, tzinfo=UTC)
    with pytest.raises(CadenceError, match="UTC offset"):
        _create(at=datetime(2027, 1, 1))
    with pytest.raises(CadenceError, match="in the future"):
        _create(at=now)
    once = _create(at=datetime(2027, 1, 1, 9, 0, tzinfo=STOCKHOLM), zone="Europe/Stockholm")
    assert once.at == datetime(2027, 1, 1, 8, 0, tzinfo=UTC)
    assert once.first_slot(now) == once.at
    assert once.next_slot(now) == once.at
    assert once.next_slot(once.at or now) is None


def test_the_interval_scan_costs_little_for_the_densest_allowed_recurrence() -> None:
    """Hourly is the worst case a default deployment allows; its scan must stay interactive."""
    started = datetime.now(UTC)
    _create("0 * * * *", zone="America/New_York")
    assert datetime.now(UTC) - started < timedelta(seconds=2)


@pytest.mark.parametrize(
    ("zone", "cron"),
    [
        ("America/New_York", "15 * * * *"),
        ("America/New_York", "30 * * * *"),
        ("America/New_York", "45 * * * *"),
        ("Europe/Stockholm", "15 * * * *"),
        ("Europe/Stockholm", "30 * * * *"),
        ("Europe/Stockholm", "45 * * * *"),
        ("Australia/Lord_Howe", "0 * * * *"),
    ],
)
def test_an_hourly_recurrence_is_not_refused_for_the_slot_the_spring_jump_stands_in_for(
    zone: str, cron: str
) -> None:
    """02:30 does not exist on the spring night; the stand-in at the jump is not a slot to measure."""
    assert _create(cron, zone=zone).cron == cron


def test_the_stand_in_for_a_skipped_time_still_fires_at_the_jump() -> None:
    """Accepting the recurrence changes what is measured, not what happens on the night."""
    cadence = Cadence("America/New_York", _cron("30 * * * *"), None)
    slots = []
    cursor = datetime(2027, 3, 14, 6, 30, tzinfo=UTC)  # 01:30 EST, before the clocks go forward
    for _ in range(2):
        cursor = cadence.next_slot(cursor) or cursor
        slots.append(cursor)
    assert slots == [
        datetime(2027, 3, 14, 7, 0, tzinfo=UTC),  # 02:30 does not exist: 03:00 EDT, the jump
        datetime(2027, 3, 14, 7, 30, tzinfo=UTC),
    ]


@pytest.mark.parametrize(
    ("zone", "day"),
    [
        ("Europe/Stockholm", datetime(2026, 3, 29)),
        ("America/New_York", datetime(2026, 3, 8)),
        ("Australia/Lord_Howe", datetime(2026, 10, 4)),
        ("Pacific/Chatham", datetime(2026, 9, 27)),
        ("America/Havana", datetime(2026, 3, 8)),
        ("Asia/Beirut", datetime(2026, 3, 29)),
    ],
)
def test_a_slot_the_jump_stands_in_for_is_the_jump_to_the_whole_second(
    zone: str, day: datetime
) -> None:
    tz = ZoneInfo(zone)
    jumps = 0
    for hour in range(5):
        for minute in range(60):
            local = day.replace(hour=hour, minute=minute)
            instant = utc_of(local, tz)
            assert instant.microsecond == 0
            if not exists(local, tz):
                jumps += 1
                before = (instant - timedelta(seconds=1)).astimezone(tz).utcoffset()
                assert instant.astimezone(tz).utcoffset() != before
    assert jumps > 0


@pytest.mark.parametrize(
    "text",
    [
        "0 " + ",".join(["1"] * 150) + " * * *",
        "0 008 * * *",
        "0 8 * * " + "1" * 400,
        "000 8 * * *",
    ],
)
def test_a_recurrence_that_is_too_long_or_has_a_long_number_is_refused(text: str) -> None:
    with pytest.raises(CadenceError, match="at most 200 characters|more than two digits"):
        _cron(text)


def test_two_digit_numbers_with_a_leading_zero_are_still_numbers() -> None:
    assert _cron("00 08 * * *").hours == (8,)


def test_a_single_time_that_overflows_the_calendar_is_refused() -> None:
    far = datetime(9999, 12, 31, 23, 30, tzinfo=timezone(timedelta(hours=-5)))
    with pytest.raises(CadenceError, match="out of range"):
        _create(at=far)


def test_a_single_time_keeps_its_zone_when_only_the_zone_changes() -> None:
    once = _create(at=datetime(2027, 1, 1, 9, 0, tzinfo=UTC), zone="UTC")
    moved = once.in_zone("Asia/Tokyo")
    assert (moved.zone_name, moved.at) == ("Asia/Tokyo", once.at)
    with pytest.raises(CadenceError, match="IANA time zone"):
        once.in_zone("Mars/Olympus")
