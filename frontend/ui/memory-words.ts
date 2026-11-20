// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The words Profile Memory is told in wherever the reader meets it: Settings, and the notice under an answer.
 *  A change as a receipt says it, and a record's kind as a label. */

import {msg, str} from '@lit/localize';
import type {MemoryOperationEvent, MemoryRecord} from '../api/memory.ts';

export function memorySummary(event: MemoryOperationEvent): string {
  const body = event.body.replace(/\s+/g, ' ').trim();
  const concise = body.length > 120 ? `${body.slice(0, 117)}…` : body;
  if (event.outcome === 'unchanged') {
    return event.operation === 'forget'
      ? msg('Already forgotten.', {id: 'settings.memory.alreadyForgotten'})
      : msg('Already remembered.', {id: 'settings.memory.alreadyRemembered'});
  }
  if (event.outcome === 'conflict') {
    return msg('Profile Memory changed; recall it before retrying.', {
      id: 'settings.memory.conflict',
    });
  }
  if (event.outcome === 'rejected') {
    return msg('Profile Memory operation was rejected.', {id: 'settings.memory.rejected'});
  }
  if (event.operation === 'forget') {
    return concise
      ? msg(str`Forgot: ${concise}`, {id: 'settings.memory.forgot'})
      : msg('Profile Memory forgotten.', {id: 'settings.memory.forgotten'});
  }
  if (event.operation === 'undo') {
    return concise
      ? msg(str`Restored: ${concise}`, {id: 'settings.memory.restored'})
      : msg('Profile Memory restored.', {id: 'settings.memory.restoredEmpty'});
  }
  return concise
    ? msg(str`Remembered: ${concise}`, {id: 'settings.memory.remembered'})
    : msg('Saved to Profile Memory.', {id: 'settings.memory.saved'});
}

/** What kind of statement a record is, as a label: what the owner wants, what they are like, what was learned. */
export function memoryKindLabel(kind: MemoryRecord['kind']): string {
  switch (kind) {
    case 'fact':
      return msg('Fact', {id: 'settings.memory.kindFact'});
    case 'experience':
      return msg('Experience', {id: 'settings.memory.kindExperience'});
    default:
      return msg('Preference', {id: 'settings.memory.kindPreference'});
  }
}
