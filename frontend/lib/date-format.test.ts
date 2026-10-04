// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import test from 'node:test';
import {capitalized, recentDay, shortDate} from './date-format.ts';

/** The reader's noon on a local calendar day: the tests never depend on the machine's time zone. */
function day(year: number, month: number, date: number, hour = 12): Date {
  return new Date(year, month - 1, date, hour, 30);
}

const NOW = day(2026, 10, 4);

test('a short date is the month and day, and carries the year only outside the current one', () => {
  assert.equal(shortDate(day(2026, 10, 2), NOW, 'en'), 'Oct 2');
  assert.equal(shortDate(day(2025, 12, 31), NOW, 'en'), 'Dec 31, 2025');
  assert.equal(shortDate(day(2026, 10, 2), NOW, 'zh'), '10月2日');
  assert.equal(shortDate(day(2025, 12, 31), NOW, 'zh'), '2025年12月31日');
});

test('the last week reads as a relative day in the reader\'s words, and older dates read as none', () => {
  assert.equal(recentDay(day(2026, 10, 4, 0), NOW, 'en'), 'today');
  assert.equal(recentDay(day(2026, 10, 3, 23), NOW, 'en'), 'yesterday');
  assert.equal(recentDay(day(2026, 10, 1), NOW, 'en'), '3 days ago');
  assert.equal(recentDay(day(2026, 9, 27), NOW, 'en'), '7 days ago');
  assert.equal(recentDay(day(2026, 9, 26), NOW, 'en'), null);
  assert.equal(recentDay(day(2026, 10, 4), NOW, 'zh'), '今天');
  assert.equal(recentDay(day(2026, 10, 3), NOW, 'zh'), '昨天');
});

test('a day is the calendar day, not the last twenty-four hours, and a future date is today', () => {
  const lateEvening = day(2026, 10, 4, 23);
  assert.equal(recentDay(day(2026, 10, 3, 23), lateEvening, 'en'), 'yesterday');
  const justAfterMidnight = new Date(2026, 9, 4, 0, 5);
  assert.equal(recentDay(new Date(2026, 9, 3, 23, 55), justAfterMidnight, 'en'), 'yesterday');
  assert.equal(recentDay(day(2026, 10, 6), NOW, 'en'), 'today');
});

test('a month boundary and a year boundary count their own days', () => {
  assert.equal(recentDay(day(2026, 9, 30), day(2026, 10, 2), 'en'), '2 days ago');
  assert.equal(recentDay(day(2025, 12, 31), day(2026, 1, 1), 'en'), 'yesterday');
});

test('a phrase stands on its own with its first letter raised', () => {
  assert.equal(capitalized('today', 'en'), 'Today');
  assert.equal(capitalized('3 days ago', 'en'), '3 days ago');
  assert.equal(capitalized('今天', 'zh'), '今天');
  assert.equal(capitalized('', 'en'), '');
});
