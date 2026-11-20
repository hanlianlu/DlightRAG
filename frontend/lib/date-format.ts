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

/** The instant a wire value states, or null when it states none or does not parse. */
function instantOf(value: string | number | null): Date | null {
  if (value === null || value === '') return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

/** The time of day a moment fell on, in the reader's time zone: "14:05", or "2:05:09 PM" with `seconds`
 * for a row that tells apart steps within a minute. Nothing when the wire states no moment.
 */
export function timeOfDay(value: string | number | null, locale: string, seconds = false): string {
  const date = instantOf(value);
  if (date === null) return '';
  return new Intl.DateTimeFormat(locale, seconds
    ? {hour: 'numeric', minute: 'numeric', second: 'numeric'}
    : {hour: '2-digit', minute: '2-digit'}).format(date);
}

/** A moment as its date and time, "Nov 20, 2026, 2:05 PM"; nothing when the wire states no moment. */
export function dateTime(value: string | null, locale: string): string {
  const date = instantOf(value);
  return date === null ? '' : new Intl.DateTimeFormat(locale, {dateStyle: 'medium', timeStyle: 'short'}).format(date);
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

/** A running call states nothing until it has run this long: the clock that redraws one runs once a
 * second, so a smaller number would be replaced before a reader could read it. */
const RUNNING_TIME_MIN_MS = 1000;

/** How long a Tool call reads as taking, as every surface that times one shows it: whole
 * milliseconds under a second, else seconds to one decimal ("850ms", "1.2s"). Nothing while a
 * running call is still inside its first second, and nothing for a time nobody measured, so a number
 * a reader sees is never a guess. */
export function toolTime(milliseconds: number | null, running = false): string {
  if (milliseconds === null || (running && milliseconds < RUNNING_TIME_MIN_MS)) return '';
  return milliseconds < RUNNING_TIME_MIN_MS
    ? `${Math.round(milliseconds)}ms`
    : `${(milliseconds / 1000).toFixed(1)}s`;
}
