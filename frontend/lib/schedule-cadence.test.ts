// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import test from 'node:test';
import {
  type CadenceDraft,
  cadenceDraftOf,
  cadenceOfDraft,
  cronOf,
  describeCadence,
  describeRecurrence,
  newCadenceDraft,
  nextRunText,
  parseRecurrence,
  type Recurrence,
  slotText,
} from './schedule-cadence.ts';

const PRESETS: ReadonlyArray<[Recurrence, string, string]> = [
  [{kind: 'hourly', minute: 15}, '15 * * * *', 'Every hour at :15'],
  [{kind: 'daily', hour: 8, minute: 0}, '0 8 * * *', 'Every day at 08:00'],
  [{kind: 'weekdays', hour: 8, minute: 30}, '30 8 * * 1-5', 'Weekdays at 08:30'],
  [{kind: 'weekly', days: [1, 4], hour: 8, minute: 0}, '0 8 * * 1,4', 'Mondays and Thursdays at 08:00'],
  [{kind: 'weekly', days: [0, 6], hour: 18, minute: 5}, '5 18 * * 0,6', 'Saturdays and Sundays at 18:05'],
  [{kind: 'monthly', day: 1, hour: 8, minute: 0}, '0 8 1 * *', 'Monthly on the 1st at 08:00'],
  [{kind: 'monthly', day: 22, hour: 8, minute: 0}, '0 8 22 * *', 'Monthly on the 22nd at 08:00'],
  [{kind: 'monthly', day: 13, hour: 8, minute: 0}, '0 8 13 * *', 'Monthly on the 13th at 08:00'],
];

test('each preset compiles to its cron, reads back from it, and is named in words', () => {
  for (const [recurrence, cron, words] of PRESETS) {
    assert.equal(cronOf(recurrence), cron);
    assert.deepEqual(parseRecurrence(cron), recurrence);
    assert.equal(describeRecurrence(recurrence, 'en'), words);
  }
});

test('days chosen in any order compile in the order cron lists them', () => {
  assert.equal(cronOf({kind: 'weekly', days: [4, 0, 1], hour: 9, minute: 0}), '0 9 * * 0,1,4');
});

test('a cron that is not exactly a preset stays Custom, and is its own words', () => {
  for (const cron of [
    '*/15 * * * *',
    '0 8 * * 1-4',
    '0 8 * * 7',
    '0 8 * * 4,1',
    '0 8 * * 1,1',
    '0 8 29 * *',
    '0 8 1 6 *',
    '0 8 1 * 1',
    '0 08 * * *',
    '0 8,20 * * *',
    '0 8 * *',
    '0 8 * * * *',
    '60 8 * * *',
    '0 24 * * *',
    '0 * * * 1',
  ]) {
    assert.deepEqual(parseRecurrence(cron), {kind: 'custom', cron}, cron);
    assert.equal(describeRecurrence({kind: 'custom', cron}, 'en'), cron);
  }
});

test('a one-time Schedule is named by its date and time on its own clock', () => {
  const at = new Date('2026-12-01T07:00:00Z');
  assert.equal(describeCadence({cron: null, at, zone: 'Europe/Stockholm'}, 'en'), 'Once on Dec 1, 2026, 08:00');
  assert.equal(describeCadence({cron: null, at, zone: 'Asia/Tokyo'}, 'en'), 'Once on Dec 1, 2026, 16:00');
  assert.equal(describeCadence({cron: '0 8 * * 1-5', at: null, zone: 'UTC'}, 'en'), 'Weekdays at 08:00');
});

test('a day of the month is ordinal in English only', () => {
  assert.equal(describeRecurrence({kind: 'monthly', day: 2, hour: 8, minute: 0}, 'en'), 'Monthly on the 2nd at 08:00');
  assert.equal(describeRecurrence({kind: 'monthly', day: 2, hour: 8, minute: 0}, 'zh'), 'Monthly on the 2 at 08:00');
});

test('a draft compiles to the cadence it states, and to none while a field is missing or out of range', () => {
  const draft = (changes: Partial<CadenceDraft>): CadenceDraft => ({...newCadenceDraft(), ...changes});
  assert.deepEqual(cadenceOfDraft(draft({})), {cron: '0 9 * * *'});
  assert.deepEqual(cadenceOfDraft(draft({frequency: 'hourly', minute: '45'})), {cron: '45 * * * *'});
  assert.deepEqual(cadenceOfDraft(draft({frequency: 'weekdays', time: '07:30'})), {cron: '30 7 * * 1-5'});
  assert.deepEqual(cadenceOfDraft(draft({frequency: 'weekly', days: [1, 3]})), {cron: '0 9 * * 1,3'});
  assert.deepEqual(cadenceOfDraft(draft({frequency: 'monthly', dayOfMonth: '15'})), {cron: '0 9 15 * *'});
  assert.deepEqual(cadenceOfDraft(draft({frequency: 'custom', cron: ' 0 6 * * 1 '})), {cron: '0 6 * * 1'});
  assert.deepEqual(cadenceOfDraft(draft({frequency: 'once', date: '2026-12-01', time: '08:00'})), {local: '2026-12-01T08:00'});
  assert.equal(cadenceOfDraft(draft({frequency: 'hourly', minute: '60'})), null);
  assert.equal(cadenceOfDraft(draft({frequency: 'hourly', minute: ''})), null);
  assert.equal(cadenceOfDraft(draft({frequency: 'monthly', dayOfMonth: '29'})), null);
  assert.equal(cadenceOfDraft(draft({frequency: 'weekly', days: []})), null);
  assert.equal(cadenceOfDraft(draft({frequency: 'daily', time: ''})), null);
  assert.equal(cadenceOfDraft(draft({frequency: 'custom', cron: '  '})), null);
  assert.equal(cadenceOfDraft(draft({frequency: 'once', date: ''})), null);
});

test('a stored cadence reads back into the draft that compiles to it', () => {
  for (const [, cron] of PRESETS) {
    const draft = cadenceDraftOf({cron, at: null, zone: 'UTC'});
    assert.notEqual(draft.frequency, 'custom', cron);
    assert.deepEqual(cadenceOfDraft(draft), {cron});
  }
  const custom = cadenceDraftOf({cron: '*/20 * * * *', at: null, zone: 'UTC'});
  assert.deepEqual([custom.frequency, custom.cron], ['custom', '*/20 * * * *']);
  const once = cadenceDraftOf({cron: null, at: new Date('2026-12-01T07:00:00Z'), zone: 'Europe/Stockholm'});
  assert.deepEqual([once.frequency, once.date, once.time], ['once', '2026-12-01', '08:00']);
});

test('the next run is as short as the day allows, on the Schedule\'s own clock', () => {
  const now = new Date('2026-11-30T12:00:00Z');
  const next = (iso: string, zone = 'Europe/Stockholm') => nextRunText(new Date(iso), now, zone, 'en');
  assert.equal(next('2026-11-30T15:30:00Z'), 'today 16:30');
  assert.equal(next('2026-12-01T07:00:00Z'), 'tomorrow 08:00');
  assert.equal(next('2026-12-03T07:00:00Z'), 'Thu 08:00');
  assert.match(next('2026-12-14T07:00:00Z'), /^Mon, Dec 14, 08:00$/);
  // Two clocks, two days: 23:30 UTC is already tomorrow in Tokyo.
  assert.equal(next('2026-11-30T23:30:00Z', 'Asia/Tokyo'), 'tomorrow 08:30');
  assert.equal(next('2026-11-30T23:30:00Z', 'UTC'), 'today 23:30');
});

test('a slot names its weekday, and its year once it is not this one', () => {
  const now = new Date('2026-11-30T12:00:00Z');
  assert.equal(slotText(new Date('2026-12-01T07:00:00Z'), now, 'Europe/Stockholm', 'en'), 'Tue, Dec 1, 08:00');
  assert.equal(slotText(new Date('2027-01-04T07:00:00Z'), now, 'Europe/Stockholm', 'en'), 'Mon, Jan 4, 2027, 08:00');
});

test('a zone this browser does not know is read on the browser\'s own clock rather than failing', () => {
  const now = new Date('2026-11-30T12:00:00Z');
  assert.equal(typeof nextRunText(new Date('2026-12-01T07:00:00Z'), now, 'Mars/Olympus', 'en'), 'string');
  assert.ok(describeCadence({cron: null, at: now, zone: 'Mars/Olympus'}, 'en').startsWith('Once on '));
});
