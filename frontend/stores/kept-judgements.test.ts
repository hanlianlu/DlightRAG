// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import type {AnswerJudgement} from '../api/feedback.ts';
import {judgementKey, KeptJudgements} from './kept-judgements.ts';

test('what the page kept outranks an older read, and signing out leaves none of it for the next reader', () => {
  const judgements = new KeptJudgements();
  const read: AnswerJudgement = {value: 'down', reasons: ['off_topic'], note: null};
  const given: AnswerJudgement = {value: 'up', reasons: [], note: null};
  const rated = judgementKey('run-1', 'main', 7);
  const takenBack = judgementKey('run-1', 'child-a', 3);
  judgements.keep(rated, given);
  judgements.keep(takenBack, null);

  assert.deepEqual(judgements.current(rated, read), given);
  // Taking a judgement back is remembered too: the read that still holds it is older.
  assert.equal(judgements.current(takenBack, read), null);

  // What the bag's reset calls on sign-out: whoever signs in next on this tab is shown the reads alone.
  judgements.forget();

  assert.deepEqual(judgements.current(rated, read), read);
  assert.deepEqual(judgements.current(takenBack, read), read);
});
