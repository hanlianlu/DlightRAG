// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {test} from 'node:test';

import {
  isDefinitiveSaveOutcome,
  PendingSubmissionStore,
} from './pendingSubmissionStore.ts';

test('unchanged retry reuses its key while changed payload invalidates it', () => {
  const ids = ['submission-1', 'submission-2'];
  const store = new PendingSubmissionStore(() => ids.shift()!);

  assert.equal(store.getOrCreate('conversation-1', 'fingerprint-a'), 'submission-1');
  assert.equal(store.getOrCreate('conversation-1', 'fingerprint-a'), 'submission-1');
  assert.equal(store.getOrCreate('conversation-1', 'fingerprint-b'), 'submission-2');
});

test('pending keys are scoped by conversation and clear explicitly', () => {
  let next = 0;
  const store = new PendingSubmissionStore(() => `submission-${++next}`);

  assert.equal(store.getOrCreate('conversation-1', 'same-hash'), 'submission-1');
  assert.equal(store.getOrCreate('conversation-2', 'same-hash'), 'submission-2');
  store.clear('conversation-1');
  assert.equal(store.getOrCreate('conversation-1', 'same-hash'), 'submission-3');
});

test('only authoritative or definitive outcomes clear pending submissions', () => {
  assert.equal(isDefinitiveSaveOutcome({conversation_saved: true}), true);
  assert.equal(
    isDefinitiveSaveOutcome({
      conversation_saved: false,
      conversation_save_reason: 'conversation_changed',
    }),
    true,
  );
  assert.equal(
    isDefinitiveSaveOutcome({
      conversation_saved: false,
      conversation_save_reason: 'commit_outcome_unknown',
    }),
    false,
  );
  assert.equal(isDefinitiveSaveOutcome(null), false);
});
