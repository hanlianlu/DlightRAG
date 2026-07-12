// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {test} from 'node:test';

import * as saveState from './pendingSubmissionStore.ts';

const {
  describeConversationSaveOutcome,
  isDefinitiveSaveOutcome,
  PendingSubmissionStore,
} = saveState;

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

test('save outcome presentation distinguishes saved, definitive, and unknown states', () => {
  assert.deepEqual(describeConversationSaveOutcome({conversation_saved: true}), {
    state: 'saved',
    message: null,
    actionLabel: null,
  });
  assert.deepEqual(describeConversationSaveOutcome({
    conversation_saved: false,
    conversation_save_reason: 'conversation_changed',
  }), {
    state: 'not_saved',
    message: 'This answer was not saved because the conversation changed.',
    actionLabel: null,
  });
  assert.deepEqual(describeConversationSaveOutcome({
    conversation_saved: false,
    conversation_save_reason: 'persistence_failed',
  }), {
    state: 'not_saved',
    message: 'This answer was not saved because conversation storage failed.',
    actionLabel: null,
  });
  assert.deepEqual(describeConversationSaveOutcome({
    conversation_saved: false,
    conversation_save_reason: 'commit_not_found',
  }), {
    state: 'not_saved',
    message: 'This answer was not saved because no committed turn was found.',
    actionLabel: null,
  });
  assert.deepEqual(describeConversationSaveOutcome({
    conversation_saved: false,
    conversation_save_reason: 'commit_outcome_unknown',
  }), {
    state: 'unknown',
    message: 'Save status is unknown. The answer may already be saved; checking history may reconcile it.',
    actionLabel: 'Check save status',
  });
  assert.deepEqual(describeConversationSaveOutcome({
    conversation_saved: false,
    conversation_save_reason: 'storage_unavailable',
  }), {
    state: 'unknown',
    message: 'Save status could not be confirmed because conversation storage is unavailable.',
    actionLabel: 'Check save status',
  });
});

test('unknown recovery keeps the same submission key available for replay', () => {
  const store = new PendingSubmissionStore(() => 'submission-1');
  const key = store.getOrCreate('conversation-1', 'fingerprint-a');

  assert.equal(isDefinitiveSaveOutcome({
    conversation_saved: false,
    conversation_save_reason: 'commit_outcome_unknown',
  }), false);
  assert.equal(store.getOrCreate('conversation-1', 'fingerprint-a'), key);
});
