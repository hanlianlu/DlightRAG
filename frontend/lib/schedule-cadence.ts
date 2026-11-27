// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** What a Schedule's cadence is in the reader's words, and the presets the form offers for it.
 *
 * The server holds a cadence as five cron fields or one instant. The form offers a few shapes that
 * compile to cron on the client (hourly, daily, weekdays, weekly, monthly), and reads a stored cron back
 * into one of them only when it matches the shape exactly; anything else stays Custom, which is the
 * cron text itself. The words come from here for every surface that names a cadence, so a row, the
 * form's summary and the detail dialog never say the same thing two ways.
 *
 * Days of the week count from Sunday as cron does (0 = Sunday), and are listed Monday first as the
 * reader's week reads. Every time shown here is the Schedule's own clock, in its zone.
 */

import {msg, str} from '@lit/localize';
import {browserZone, localInZone} from './zoned-time.ts';

export type Frequency = 'once' | 'hourly' | 'daily' | 'weekdays' | 'weekly' | 'monthly' | 'custom';

export type Recurrence =
  | {kind: 'hourly'; minute: number}
  | {kind: 'daily'; hour: number; minute: number}
  | {kind: 'weekdays'; hour: number; minute: number}
  /** `days` ascend, as cron lists them; each is 0 (Sunday) to 6 (Saturday). */
  | {kind: 'weekly'; days: readonly number[]; hour: number; minute: number}
  /** The day of the month is 1 to 28, so every month has it. */
  | {kind: 'monthly'; day: number; hour: number; minute: number}
  | {kind: 'custom'; cron: string};

/** The week as the reader lists it, Monday first. */
export const WEEK: readonly number[] = [1, 2, 3, 4, 5, 6, 0];

/** The last day of the month a preset offers, which every month has. */
export const LAST_PRESET_DAY = 28;

const pad = (value: number): string => String(value).padStart(2, '0');

/** `HH:MM`, the form of a time input and of every time of day this module writes. */
function clock(hour: number, minute: number): string {
  return `${pad(hour)}:${pad(minute)}`;
}

/** The hour and minute of an `HH:MM` value, or null when it is anything else. */
function parseClock(text: string): {hour: number; minute: number} | null {
  const match = /^([01]\d|2[0-3]):([0-5]\d)$/.exec(text);
  return match === null ? null : {hour: Number(match[1]), minute: Number(match[2])};
}

/** A whole number in a field with no sign, no leading zero and nothing else, inside the range. */
function whole(text: string | undefined, min: number, max: number): number | null {
  if (text === undefined || !/^(0|[1-9]\d?)$/.test(text)) return null;
  const value = Number(text);
  return value >= min && value <= max ? value : null;
}

export function cronOf(recurrence: Recurrence): string {
  switch (recurrence.kind) {
    case 'hourly': return `${recurrence.minute} * * * *`;
    case 'daily': return `${recurrence.minute} ${recurrence.hour} * * *`;
    case 'weekdays': return `${recurrence.minute} ${recurrence.hour} * * 1-5`;
    case 'weekly': return `${recurrence.minute} ${recurrence.hour} * * ${[...recurrence.days].sort((a, b) => a - b).join(',')}`;
    case 'monthly': return `${recurrence.minute} ${recurrence.hour} ${recurrence.day} * *`;
    case 'custom': return recurrence.cron;
  }
}

/** The preset a stored cron is exactly, else Custom with the cron as it was written. */
export function parseRecurrence(cron: string): Recurrence {
  const custom: Recurrence = {kind: 'custom', cron};
  const [minuteField, hourField, dayField, monthField, weekField, ...extra] = cron.trim().split(/\s+/);
  const minute = whole(minuteField, 0, 59);
  if (minute === null || monthField !== '*' || weekField === undefined || extra.length > 0) return custom;
  if (hourField === '*') return dayField === '*' && weekField === '*' ? {kind: 'hourly', minute} : custom;
  const hour = whole(hourField, 0, 23);
  if (hour === null) return custom;
  if (dayField === '*') {
    if (weekField === '*') return {kind: 'daily', hour, minute};
    if (weekField === '1-5') return {kind: 'weekdays', hour, minute};
    if (!/^[0-6](,[0-6])*$/.test(weekField)) return custom;
    const days = weekField.split(',').map(Number);
    return days.every((day, index) => day > (days[index - 1] ?? -1)) ? {kind: 'weekly', days, hour, minute} : custom;
  }
  const day = whole(dayField, 1, LAST_PRESET_DAY);
  return weekField === '*' && day !== null ? {kind: 'monthly', day, hour, minute} : custom;
}

function plural(day: number): string {
  switch (day) {
    case 1: return msg('Mondays', {id: 'schedules.cadence.day1'});
    case 2: return msg('Tuesdays', {id: 'schedules.cadence.day2'});
    case 3: return msg('Wednesdays', {id: 'schedules.cadence.day3'});
    case 4: return msg('Thursdays', {id: 'schedules.cadence.day4'});
    case 5: return msg('Fridays', {id: 'schedules.cadence.day5'});
    case 6: return msg('Saturdays', {id: 'schedules.cadence.day6'});
    default: return msg('Sundays', {id: 'schedules.cadence.day0'});
  }
}

/** "1st", "22nd" in English; the plain number where the language has no such suffix. */
function ordinal(day: number, locale: string): string {
  if (!locale.startsWith('en')) return String(day);
  const suffixes: Record<string, string> = {one: 'st', two: 'nd', few: 'rd'};
  return `${day}${suffixes[new Intl.PluralRules('en', {type: 'ordinal'}).select(day)] ?? 'th'}`;
}

export function describeRecurrence(recurrence: Recurrence, locale: string): string {
  switch (recurrence.kind) {
    case 'hourly': return msg(str`Every hour at :${pad(recurrence.minute)}`, {id: 'schedules.cadence.hourly'});
    case 'daily': return msg(str`Every day at ${clock(recurrence.hour, recurrence.minute)}`, {id: 'schedules.cadence.daily'});
    case 'weekdays': return msg(str`Weekdays at ${clock(recurrence.hour, recurrence.minute)}`, {id: 'schedules.cadence.weekdays'});
    case 'weekly': {
      const days = WEEK.filter((day) => recurrence.days.includes(day)).map(plural);
      return msg(str`${new Intl.ListFormat(locale, {type: 'conjunction'}).format(days)} at ${clock(recurrence.hour, recurrence.minute)}`, {
        id: 'schedules.cadence.weekly',
      });
    }
    case 'monthly':
      return msg(str`Monthly on the ${ordinal(recurrence.day, locale)} at ${clock(recurrence.hour, recurrence.minute)}`, {
        id: 'schedules.cadence.monthly',
      });
    case 'custom': return recurrence.cron;
  }
}

/** A weekday's short name in the reader's language: "Mon", "周一". */
export function weekdayShort(day: number, locale: string): string {
  // 1 January 2024 was a Monday, so day 0 (Sunday) is the 7th.
  return new Intl.DateTimeFormat(locale, {weekday: 'short', timeZone: 'UTC'}).format(new Date(Date.UTC(2024, 0, day === 0 ? 7 : day)));
}

/** The wall clock of a zone at an instant, as whole parts: the day counted from a fixed epoch, and the time. A
 *  zone this browser does not know reads as the browser's own, as the form does until the server refuses it. */
function wall(instant: Date, zone: string): {day: number; time: string; stamp: string} {
  const stamp = localInZone(instant, zone) || localInZone(instant, browserZone());
  const [date = '', time = ''] = stamp.split('T');
  const [year = 0, month = 1, day = 1] = date.split('-').map(Number);
  return {day: Math.floor(Date.UTC(year, month - 1, day) / 86_400_000), time, stamp};
}

function intl(zone: string, locale: string, options: Intl.DateTimeFormatOptions): Intl.DateTimeFormat {
  try {
    return new Intl.DateTimeFormat(locale, {...options, timeZone: zone});
  } catch {
    return new Intl.DateTimeFormat(locale, options);
  }
}

/** A one-time Schedule: "Once on Dec 1, 2026, 08:00". */
export function describeOnce(at: Date, zone: string, locale: string): string {
  const date = intl(zone, locale, {day: 'numeric', month: 'short', year: 'numeric'}).format(at);
  return msg(str`Once on ${date}, ${wall(at, zone).time}`, {id: 'schedules.cadence.once'});
}

/** The words for a Schedule's cadence, whichever form it holds. */
export function describeCadence(cadence: {cron: string | null; at: Date | null; zone: string}, locale: string): string {
  if (cadence.cron !== null) return describeRecurrence(parseRecurrence(cadence.cron), locale);
  return cadence.at === null ? '' : describeOnce(cadence.at, cadence.zone, locale);
}

/** When a slot falls, in full and on the Schedule's clock: "Mon, Dec 1, 08:00", with the year once it is not this one's. */
export function slotText(slot: Date, now: Date, zone: string, locale: string): string {
  const sameYear = wall(slot, zone).stamp.slice(0, 4) === wall(now, zone).stamp.slice(0, 4);
  return intl(zone, locale, {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    ...(sameYear ? {} : {year: 'numeric'}),
    hour: '2-digit',
    minute: '2-digit',
    hourCycle: 'h23',
  }).format(slot);
}

/** When a Schedule next fires, as short as a list row can hold it: "today 14:30", "tomorrow 08:00", "Mon 08:00" within the
 *  week, else the date. */
export function nextRunText(slot: Date, now: Date, zone: string, locale: string): string {
  const ahead = wall(slot, zone).day - wall(now, zone).day;
  const time = wall(slot, zone).time;
  if (ahead >= 0 && ahead <= 1) return `${new Intl.RelativeTimeFormat(locale, {numeric: 'auto'}).format(ahead, 'day')} ${time}`;
  if (ahead > 1 && ahead < 7) return `${intl(zone, locale, {weekday: 'short'}).format(slot)} ${time}`;
  return slotText(slot, now, zone, locale);
}

/** The form's cadence fields: what the reader has typed and chosen, before it is a cadence. */
export interface CadenceDraft {
  frequency: Frequency;
  /** Hourly: the minute of each hour. */
  minute: string;
  /** Daily, weekdays, weekly, monthly and once: the time of day, `HH:MM`. */
  time: string;
  /** Weekly: the days chosen, 0 (Sunday) to 6 (Saturday). */
  days: readonly number[];
  /** Monthly: the day of the month. */
  dayOfMonth: string;
  /** Once: the date, `YYYY-MM-DD`. */
  date: string;
  /** Custom: the five cron fields. */
  cron: string;
}

/** The draft a new Schedule starts from: every day at nine, the commonest thing to ask of a clock. */
export function newCadenceDraft(): CadenceDraft {
  return {frequency: 'daily', minute: '0', time: '09:00', days: [1], dayOfMonth: '1', date: '', cron: ''};
}

/** The draft a stored cadence reads back into: a preset when it is exactly one, else Custom. */
export function cadenceDraftOf(cadence: {cron: string | null; at: Date | null; zone: string}): CadenceDraft {
  const draft = newCadenceDraft();
  if (cadence.cron === null) {
    const [date = '', time = draft.time] = (cadence.at === null ? '' : localInZone(cadence.at, cadence.zone)).split('T');
    return {...draft, frequency: 'once', date, time};
  }
  const recurrence = parseRecurrence(cadence.cron);
  const time = 'hour' in recurrence ? clock(recurrence.hour, recurrence.minute) : draft.time;
  switch (recurrence.kind) {
    case 'hourly': return {...draft, frequency: 'hourly', minute: String(recurrence.minute)};
    case 'weekly': return {...draft, frequency: 'weekly', time, days: recurrence.days};
    case 'monthly': return {...draft, frequency: 'monthly', time, dayOfMonth: String(recurrence.day)};
    case 'custom': return {...draft, frequency: 'custom', cron: recurrence.cron};
    default: return {...draft, frequency: recurrence.kind, time};
  }
}

/** What a draft asks the server: the cron it compiles to, or the wall time of a one-time run; null while it is
 *  not yet a cadence (a field empty, or out of its range). The server still judges the cron it is sent. */
export function cadenceOfDraft(draft: CadenceDraft): {cron: string} | {local: string} | null {
  const at = parseClock(draft.time);
  switch (draft.frequency) {
    case 'once': return at !== null && /^\d{4}-\d{2}-\d{2}$/.test(draft.date) ? {local: `${draft.date}T${draft.time}`} : null;
    case 'custom': return draft.cron.trim() === '' ? null : {cron: draft.cron.trim()};
    case 'hourly': {
      const minute = whole(draft.minute, 0, 59);
      return minute === null ? null : {cron: cronOf({kind: 'hourly', minute})};
    }
    case 'monthly': {
      const day = whole(draft.dayOfMonth, 1, LAST_PRESET_DAY);
      return at === null || day === null ? null : {cron: cronOf({kind: 'monthly', day, ...at})};
    }
    case 'weekly':
      return at === null || draft.days.length === 0 ? null : {cron: cronOf({kind: 'weekly', days: draft.days, ...at})};
    case 'daily': return at === null ? null : {cron: cronOf({kind: 'daily', ...at})};
    case 'weekdays': return at === null ? null : {cron: cronOf({kind: 'weekdays', ...at})};
  }
}
