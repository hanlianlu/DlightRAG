# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""When a Schedule fires: five-field cron on a zone's wall clock, or one instant (ADR 0103).

The recurrence is read on the *naive wall clock* of the Schedule's IANA zone and only then
converted to UTC, so "Monday 08:00 Europe/Stockholm" stays 08:00 across daylight-saving
changes. Two wall-clock accidents are decided here once:

- a local time that does not exist (the spring gap) fires at the first valid instant after it,
  the moment the clock jumps;
- a local time that occurs twice (the autumn overlap) fires once, at its first occurrence.

The next slot is found field by field (month, day, hour, minute), never by stepping a minute at
a time. There is no dependency on a cron library: the grammar is the five fields with ``*``,
lists, ranges and steps, and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dlightrag.application.schedules.errors import ScheduleInputError

#: How far ahead a recurrence is searched for a date it matches before it is called empty. Eight
#: years bridges the longest gap the Gregorian calendar leaves a ``29 2`` expression.
_SEARCH_YEARS = 8

#: The window in which two consecutive slots may not be closer than the configured minimum.
MIN_INTERVAL_SCAN_DAYS = 400

#: Slots one interval scan reads at most, so a pathological expression cannot hold a request.
_MIN_INTERVAL_SCAN_CAP = 20_000

#: A five-field expression with every list item spelled out stays far inside this.
MAX_CRON_CHARS = 200

_BOUNDS = (
    ("minute", 0, 59),
    ("hour", 0, 23),
    ("day of month", 1, 31),
    ("month", 1, 12),
    ("day of week", 0, 7),
)


class CadenceError(ScheduleInputError):
    """A cadence the owner must change before the Schedule can be saved."""


@dataclass(frozen=True, slots=True)
class CronExpression:
    """One parsed five-field expression: minute, hour, day of month, month, day of week.

    Day of week counts from Sunday as 0, and 7 is Sunday too. When both day fields are
    restricted a day matches if *either* does (the standard rule); a field that starts with
    ``*`` is unrestricted, so a day must then match the other field.
    """

    text: str
    minutes: tuple[int, ...]
    hours: tuple[int, ...]
    days: frozenset[int]
    months: tuple[int, ...]
    weekdays: frozenset[int]
    days_unrestricted: bool
    weekdays_unrestricted: bool

    @classmethod
    def parse(cls, text: str) -> CronExpression:
        stripped = text.strip()
        if len(stripped) > MAX_CRON_CHARS:
            raise CadenceError(f"a recurrence is at most {MAX_CRON_CHARS} characters")
        if stripped.startswith("@"):
            raise CadenceError(
                "macros such as @hourly are not supported; write the five fields instead"
            )
        parts = stripped.split()
        if len(parts) != 5:
            raise CadenceError(
                "a recurrence is five fields: minute hour day-of-month month day-of-week"
            )
        minutes, hours, days, months, weekdays = (
            _field(part, name, low, high)
            for part, (name, low, high) in zip(parts, _BOUNDS, strict=True)
        )
        return cls(
            text=" ".join(parts),
            minutes=tuple(sorted(minutes)),
            hours=tuple(sorted(hours)),
            days=frozenset(days),
            months=tuple(sorted(months)),
            weekdays=frozenset(value % 7 for value in weekdays),
            days_unrestricted=parts[2].startswith("*"),
            weekdays_unrestricted=parts[4].startswith("*"),
        )

    def next_local(self, after: datetime) -> datetime:
        """The first matching wall-clock minute strictly after the naive ``after``."""
        current = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
        last_year = after.year + _SEARCH_YEARS
        while current.year <= last_year:
            if current.month not in self.months:
                later = next((month for month in self.months if month > current.month), None)
                current = (
                    datetime(current.year + 1, self.months[0], 1)
                    if later is None
                    else datetime(current.year, later, 1)
                )
            elif not self._day_matches(current.date()):
                current = datetime.combine(current.date() + timedelta(days=1), time.min)
            elif current.hour not in self.hours:
                later = next((hour for hour in self.hours if hour > current.hour), None)
                current = (
                    datetime.combine(current.date() + timedelta(days=1), time.min)
                    if later is None
                    else current.replace(hour=later, minute=0)
                )
            elif current.minute not in self.minutes:
                later = next((minute for minute in self.minutes if minute > current.minute), None)
                current = (
                    current.replace(minute=0) + timedelta(hours=1)
                    if later is None
                    else current.replace(minute=later)
                )
            else:
                return current
        raise CadenceError("this expression never matches a calendar date")

    def next_instant(self, after: datetime, zone: ZoneInfo) -> datetime:
        """The first slot strictly after the aware ``after``, as a UTC instant."""
        local = after.astimezone(zone).replace(tzinfo=None)
        while True:
            local = self.next_local(local)
            slot = utc_of(local, zone)
            # The autumn overlap and the spring gap can map a later wall-clock minute onto an
            # instant that is not after ``after``: that slot already fired, so read on.
            if slot > after:
                return slot

    def _day_matches(self, day: date) -> bool:
        in_month = day.day in self.days
        in_week = (day.weekday() + 1) % 7 in self.weekdays
        if self.days_unrestricted or self.weekdays_unrestricted:
            return in_month and in_week
        return in_month or in_week


def _field(text: str, name: str, low: int, high: int) -> set[int]:
    values: set[int] = set()
    for item in text.split(","):
        base, slash, step_text = item.partition("/")
        step = _number(step_text, name) if slash else 1
        if step < 1:
            raise CadenceError(f"the {name} step must be at least 1")
        if base == "*":
            start, end = low, high
        elif "-" in base:
            first, _, last = base.partition("-")
            start, end = _number(first, name), _number(last, name)
        else:
            start = _number(base, name)
            end = high if slash else start
        if not low <= start <= end <= high:
            raise CadenceError(f"the {name} field takes values from {low} to {high}, in order")
        values.update(range(start, end + 1, step))
    return values


def _number(text: str, name: str) -> int:
    if text.isascii() and text.isdigit():
        if len(text) > 2:
            raise CadenceError(f"the {name} field has a number of more than two digits")
        return int(text)
    if text.isalpha():
        raise CadenceError(f"the {name} field takes numbers, not names")
    raise CadenceError(f"the {name} field has an item that is not a number, range or step")


def exists(local: datetime, zone: ZoneInfo) -> bool:
    """Whether the wall clock of ``zone`` ever reads the naive ``local`` (the spring jump skips some)."""
    instant = local.replace(tzinfo=zone, fold=0).astimezone(UTC)
    return instant.astimezone(zone).replace(tzinfo=None) == local


def utc_of(local: datetime, zone: ZoneInfo) -> datetime:
    """The UTC instant of a naive wall-clock time in ``zone``, by the rule in the module note.

    Every instant it returns is a whole second.
    """
    instant = local.replace(tzinfo=zone, fold=0).astimezone(UTC)
    if instant.astimezone(zone).replace(tzinfo=None) == local:
        return instant
    # The wall clock skipped this time. Fold 0 reads it with the offset before the jump and fold 1
    # with the one after, so the jump itself lies between the two readings: find it to the second.
    after = local.replace(tzinfo=zone, fold=1).utcoffset() or timedelta(0)
    low = int((local - after).replace(tzinfo=UTC).timestamp())
    high = int(instant.timestamp())
    while high - low > 1:
        middle = (low + high) // 2
        if datetime.fromtimestamp(middle, UTC).astimezone(zone).utcoffset() == after:
            high = middle
        else:
            low = middle
    return datetime.fromtimestamp(high, UTC)


@dataclass(frozen=True, slots=True)
class Cadence:
    """A recurrence or one instant, and the zone the owner thinks of it in."""

    zone_name: str
    expression: CronExpression | None = None
    at: datetime | None = None

    @classmethod
    def load(cls, *, zone_name: str, cron: str | None, at: datetime | None) -> Cadence:
        """Rebuild a stored cadence, which was validated when it was written."""
        return cls(zone_name, None if cron is None else CronExpression.parse(cron), at)

    @classmethod
    def create(
        cls,
        *,
        zone_name: str,
        cron: str | None,
        at: datetime | None,
        now: datetime,
        min_interval: timedelta,
    ) -> Cadence:
        """Validate one owner-supplied cadence: a zone, and exactly one of a recurrence or an instant.

        A recurrence is refused when two consecutive slots in the next ``MIN_INTERVAL_SCAN_DAYS``
        days are closer than ``min_interval``. The scan reads slots in UTC order, so the gaps a
        daylight-saving change makes in the real clock are measured, not assumed; it stops at the
        first slot pair that is too close and after a bounded number of slots.
        """
        zone = zone_of(zone_name)
        if at is not None:
            if cron is not None:
                raise CadenceError("a Schedule has either a recurrence or a single time, not both")
            if at.tzinfo is None:
                raise CadenceError("the single time must carry a UTC offset")
            try:
                instant = at.astimezone(UTC)
            except OverflowError:
                raise CadenceError("the single time is out of range") from None
            if instant <= now:
                raise CadenceError("the single time must be in the future")
            return cls(zone_name, None, instant)
        if cron is None:
            raise CadenceError("a Schedule needs a recurrence or a single time")
        expression = CronExpression.parse(cron)
        _refuse_close_slots(expression, zone, now, min_interval)
        return cls(zone_name, expression, None)

    def same_slots(self, other: Cadence) -> bool:
        """Whether both name the same slots, so that moving from one to the other waits for nothing new.

        A recurrence is read on the wall clock of its zone, so another zone is other slots; a single
        time is one instant, and the zone only says how its owner reads it.
        """
        if self.expression is None:
            return other.expression is None and self.at == other.at
        return self.expression == other.expression and self.zone_name == other.zone_name

    def in_zone(self, zone_name: str) -> Cadence:
        """A single time read in another zone: the same instant, so nothing to measure again."""
        zone_of(zone_name)
        return replace(self, zone_name=zone_name)

    @property
    def zone(self) -> ZoneInfo:
        return zone_of(self.zone_name)

    @property
    def cron(self) -> str | None:
        return None if self.expression is None else self.expression.text

    def first_slot(self, now: datetime) -> datetime | None:
        """The slot a Schedule starts waiting for: the next one after ``now``, or the instant."""
        return self.at if self.expression is None else self.next_slot(now)

    def next_slot(self, after: datetime) -> datetime | None:
        """The first slot strictly after the aware ``after``; ``None`` once a single time is past."""
        if self.expression is not None:
            return self.expression.next_instant(after, self.zone)
        return self.at if self.at is not None and self.at > after else None


def _refuse_close_slots(
    expression: CronExpression, zone: ZoneInfo, now: datetime, min_interval: timedelta
) -> None:
    """Refuse a recurrence with two consecutive slots in the next 400 days closer than ``min_interval``.

    Slots are read in UTC order, so the gaps a daylight-saving change makes in the real clock are
    measured, not assumed. A wall-clock time the spring jump skips is not a slot of its own: it
    stands in for itself at the jump (``utc_of``), and counting that stand-in would refuse every
    ``:30`` or ``:15`` hourly recurrence for a gap that only the calendar makes.
    """
    local = now.astimezone(zone).replace(tzinfo=None)
    horizon = now + timedelta(days=MIN_INTERVAL_SCAN_DAYS)
    previous: datetime | None = None
    for _ in range(_MIN_INTERVAL_SCAN_CAP):
        local = expression.next_local(local)
        if not exists(local, zone):
            continue
        slot = utc_of(local, zone)
        if previous is not None and slot - previous < min_interval:
            raise CadenceError(
                f"slots may be at least {_minutes(min_interval)} minutes apart; "
                f"this recurrence fires at {previous.astimezone(zone):%Y-%m-%d %H:%M} and again "
                f"at {slot.astimezone(zone):%Y-%m-%d %H:%M}"
            )
        if slot > horizon:
            return
        previous = slot


def zone_of(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError, ValueError, OSError:
        raise CadenceError(f"{name!r} is not an IANA time zone, such as Europe/Stockholm") from None


def _minutes(span: timedelta) -> int:
    return int(span.total_seconds() // 60)


__all__ = [
    "MIN_INTERVAL_SCAN_DAYS",
    "Cadence",
    "CadenceError",
    "CronExpression",
    "exists",
    "utc_of",
    "zone_of",
]
