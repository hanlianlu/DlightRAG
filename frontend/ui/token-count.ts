// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The one copy a token total gets: a turn's Run actions and the roster's agent
 *  rows both read it, so the words and the compacting live here once. */

import {msg, str} from '@lit/localize';
import {getLocale} from '../i18n/locale.ts';
import {formatCompactCount} from '../lib/number-format.ts';

export function tokenCountLabel(total: number): string {
  return msg(str`${formatCompactCount(total, getLocale())} tokens`, {id: 'usage.tokens'});
}
