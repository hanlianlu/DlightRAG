// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** Dates as the reader's calendar says them, and spans of time in the reader's own words: a short
 * date and a recent day, and how long something took, in one locale.
 *
 * The dates work in the reader's own time zone, because "today" and "yesterday" are the reader's days.
 * `now` is a parameter, so a render fixes one clock for every row it draws.
 */

/** A relative phrase stands in for the date only this many days back. */
const RECENT_DAYS = 7;
const DAY_MILLISECONDS = 86_400_000;

/** The calendar day a date falls on, counted from a fixed epoch in the reader's time zone. */
function calendarDay(date: Date): number {
  return Math.floor(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()) / DAY_MILLISECONDS);
}

/** Month and day, plus the year when the date is not in the current year. */
export function shortDate(date: Date, now: Date, locale: string): string {
  const sameYear = date.getFullYear() === now.getFullYear();
  return new Intl.DateTimeFormat(locale, {
    month: 'short',
    day: 'numeric',
    ...(sameYear ? {} : {year: 'numeric'}),
  }).format(date);
}

/** "today", "yesterday", or "3 days ago" for the last week; null once the date is older.
 *
 * The phrase is lower case, as it reads inside a sentence; `capitalized` raises it for a column.
 * A date ahead of the clock (a skewed server) reads as today rather than as the future.
 */
export function recentDay(date: Date, now: Date, locale: string): string | null {
  const daysAgo = Math.max(0, calendarDay(now) - calendarDay(date));
  if (daysAgo > RECENT_DAYS) return null;
  return new Intl.RelativeTimeFormat(locale, {numeric: 'auto'}).format(-daysAgo, 'day');
}

/** How long something took, in the reader's own unit words and the two largest units that matter:
 * "41s", "2m 14s", "1h 3m". A span that is not positive reads as no time at all.
 */
export function elapsed(milliseconds: number, locale: string): string {
  const total = Number.isFinite(milliseconds) ? Math.max(0, Math.floor(milliseconds / 1000)) : 0;
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor(total / 60) % 60;
  const seconds = total % 60;
  const parts: [Intl.NumberFormatOptions['unit'], number][] = hours > 0
    ? [['hour', hours], ['minute', minutes]]
    : minutes > 0 ? [['minute', minutes], ['second', seconds]] : [['second', seconds]];
  return new Intl.ListFormat(locale, {style: 'narrow', type: 'unit'}).format(parts.map(
    ([unit, count]) => new Intl.NumberFormat(locale, {style: 'unit', unit, unitDisplay: 'narrow'}).format(count),
  ));
}

/** Raise the first letter of a phrase for use on its own. */
export function capitalized(text: string, locale: string): string {
  return text.charAt(0).toLocaleUpperCase(locale) + text.slice(1);
}
