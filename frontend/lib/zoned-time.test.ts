// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import test from 'node:test';
import {instantInZone, localInZone} from './zoned-time.ts';

test('a wall time in a zone is the instant that zone\'s clock shows it at, on either side of a clock change', () => {
  // Stockholm is UTC+1 in winter and UTC+2 in summer.
  assert.equal(instantInZone('2026-12-14T08:00', 'Europe/Stockholm')?.toISOString(), '2026-12-14T07:00:00.000Z');
  assert.equal(instantInZone('2026-07-14T08:00', 'Europe/Stockholm')?.toISOString(), '2026-07-14T06:00:00.000Z');
  assert.equal(instantInZone('2026-12-14T08:00', 'Asia/Tokyo')?.toISOString(), '2026-12-13T23:00:00.000Z');
  assert.equal(instantInZone('2026-12-14T08:00', 'UTC')?.toISOString(), '2026-12-14T08:00:00.000Z');
});

test('a time the clock skips reads as the first valid instant, and a time it repeats as its first occurrence', () => {
  // Spring forward in Stockholm, 2027-03-28: 02:30 does not exist, the clock jumps from 02:00 to 03:00.
  assert.equal(instantInZone('2027-03-28T02:30', 'Europe/Stockholm')?.toISOString(), '2027-03-28T01:00:00.000Z');
  // Fall back, 2026-10-25: 02:30 happens twice, first at UTC+2.
  assert.equal(instantInZone('2026-10-25T02:30', 'Europe/Stockholm')?.toISOString(), '2026-10-25T00:30:00.000Z');
});

test('what the zone\'s wall clock reads is the value a datetime-local input holds, and it round-trips', () => {
  const instant = new Date('2026-12-13T23:00:00Z');
  assert.equal(localInZone(instant, 'Asia/Tokyo'), '2026-12-14T08:00');
  assert.equal(instantInZone(localInZone(instant, 'Asia/Tokyo'), 'Asia/Tokyo')?.getTime(), instant.getTime());
});

test('an unknown zone or a value that is not a wall time states no instant', () => {
  assert.equal(instantInZone('2026-12-14T08:00', 'Mars/Olympus'), null);
  assert.equal(instantInZone('next Monday', 'UTC'), null);
  assert.equal(localInZone(new Date(0), 'Mars/Olympus'), '');
});
