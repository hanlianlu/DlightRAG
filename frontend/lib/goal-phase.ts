// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The words one durable goal phase shows, shared by its bar and its commands. */

import {msg} from '@lit/localize';
import type {ConversationGoal} from '../api/goal.ts';

/** The one label a phase carries. `active` means the platform continues it, so the words say
 *  ongoing rather than naming a process-local activation the wire no longer has. */
export function goalPhaseLabel(goal: Pick<ConversationGoal, 'phase'>): string {
  switch (goal.phase) {
    case 'active':
      return msg('Ongoing goal', {id: 'goalBar.ongoing'});
    case 'paused':
      return msg('Paused goal', {id: 'goalBar.paused'});
    case 'blocked':
      return msg('Blocked goal', {id: 'goalBar.blocked'});
    default:
      return msg('Complete goal', {id: 'goalBar.complete'});
  }
}
