// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** Wall-clock times in an IANA zone that is not the reader's own: a Schedule's one-time slot is typed
 * as the clock on its zone's wall reads, and shown back the same way.
 *
 * The form value is what `<input type="datetime-local">` holds, `YYYY-MM-DDTHH:mm`, which names no zone;
 * the zone beside it is what gives it one.
 */

/** The reader's own IANA zone, which a new Schedule starts in. */
export function browserZone(): string {
  return new Intl.DateTimeFormat().resolvedOptions().timeZone;
}

/** How far ahead of UTC the zone's clock reads at an instant, in minutes. */
function offsetMinutes(instant: number, zone: string): number {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: zone,
    hourCycle: 'h23',
    year: 'numeric',
    month: 'numeric',
    day: 'numeric',
    hour: 'numeric',
    minute: 'numeric',
    second: 'numeric',
  }).formatToParts(new Date(instant));
  const field = (type: string): number => Number(parts.find((part) => part.type === type)?.value);
  const onWall = Date.UTC(field('year'), field('month') - 1, field('day'), field('hour'), field('minute'), field('second'));
  return (onWall - Math.floor(instant / 1000) * 1000) / 60_000;
}

const DAY_MILLISECONDS = 86_400_000;
const MINUTE_MILLISECONDS = 60_000;

/** The instant a zone's wall clock reads `local` at, or null for a zone this browser does not know or a
 * value that is not `YYYY-MM-DDTHH:mm`. A time the zone's clock skips reads as the first valid instant
 * after it, and a time it repeats as its first occurrence, as the server reads a recurrence. */
export function instantInZone(local: string, zone: string): Date | null {
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})$/.exec(local);
  if (match === null) return null;
  const [year, month, day, hour, minute] = match.slice(1).map(Number);
  const onWall = Date.UTC(year, month - 1, day, hour, minute);
  try {
    // The zone's offset a day either side of the wall time bounds the offsets it can read at: the one
    // before a clock change and the one after. An instant is the wall time's when the zone really shows
    // that offset there; with two such instants the time repeats, and with none the clock skips it.
    const offsets = [offsetMinutes(onWall - DAY_MILLISECONDS, zone), offsetMinutes(onWall + DAY_MILLISECONDS, zone)];
    const instants = offsets.map((offset) => onWall - offset * MINUTE_MILLISECONDS);
    const valid = instants.filter((instant, index) => offsetMinutes(instant, zone) === offsets[index]);
    if (valid.length > 0) return new Date(Math.min(...valid));
    // Skipped: the change lies between the two readings, and the clock reads the wall time from it on.
    let [before, after] = [Math.min(...instants), Math.max(...instants)];
    while (after - before > MINUTE_MILLISECONDS) {
      const middle = before + Math.floor((after - before) / (2 * MINUTE_MILLISECONDS)) * MINUTE_MILLISECONDS;
      if (offsetMinutes(middle, zone) === offsetMinutes(after, zone)) after = middle;
      else before = middle;
    }
    return new Date(after);
  } catch {
    return null;
  }
}

/** The zone's wall clock at an instant, as the value of a `datetime-local` input; empty for an unknown zone. */
export function localInZone(instant: Date, zone: string): string {
  try {
    const parts = new Intl.DateTimeFormat('en-CA', {
      timeZone: zone,
      hourCycle: 'h23',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    }).formatToParts(instant);
    const field = (type: string): string => parts.find((part) => part.type === type)?.value ?? '';
    return `${field('year')}-${field('month')}-${field('day')}T${field('hour')}:${field('minute')}`;
  } catch {
    return '';
  }
}
