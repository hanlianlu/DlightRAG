// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {test} from 'node:test';
import {waitFor} from 'xstate';
import {AnswerSubmissionError, type AnswerSubmissionAdapter} from '../api/answer-submission.ts';
import type {AcceptedAnswer} from '../api/conversations.ts';
import type {AttachmentLease} from './attachment-store.ts';
import {
  answerSubmissionSnapshot,
  createAnswerSubmissionActor,
} from './answer-submission-machine.ts';

const accepted = {
  conversation: {conversation_id: 'c1'},
  turn: {answer_run_id: 'run-1', submission_id: 'submission-1'},
} as AcceptedAnswer;
const intent = {
  submissionId: 'submission-1',
  conversationId: 'c1',
  query: 'Question',
  workspaces: ['default'],
  mode: null,
};

function lease(): AttachmentLease & {accepted: number; restored: number; discarded: number} {
  const value = {
    items: [], accepted: 0, restored: 0, discarded: 0,
    get settled() { return value.accepted + value.restored + value.discarded > 0; },
    accept() { value.accepted += 1; },
    restore() { value.restored += 1; },
    discard() { value.discarded += 1; },
  };
  return value;
}

async function reaches(
  actor: ReturnType<typeof createAnswerSubmissionActor>,
  state: Parameters<ReturnType<typeof actor.getSnapshot>['matches']>[0],
) {
  actor.start();
  await waitFor(actor, (snapshot) => snapshot.matches(state));
  return answerSubmissionSnapshot(actor);
}

test('successful POST accepts the attachment lease', async () => {
  const held = lease();
  const adapter: AnswerSubmissionAdapter = {
    submit: async () => accepted,
    lookup: async () => { throw new Error('unused'); },
  };
  const actor = createAnswerSubmissionActor({intent, lease: held, adapter});
  const snapshot = await reaches(actor, 'accepted');
  assert.equal(snapshot.accepted, accepted);
  assert.equal(held.accepted, 0);
  actor.send({type: 'HANDOFF'});
  await waitFor(actor, (next) => next.matches('handedOff'));
  assert.equal(held.accepted, 1);
});

test('ambiguous POST reconciles by lookup and never posts automatically again', async () => {
  let posts = 0;
  let lookups = 0;
  const adapter: AnswerSubmissionAdapter = {
    submit: async () => {
      posts += 1;
      throw new AnswerSubmissionError(0, 'ambiguous', 'network');
    },
    lookup: async () => {
      lookups += 1;
      return accepted;
    },
  };
  const snapshot = await reaches(createAnswerSubmissionActor({intent, lease: lease(), adapter}), 'accepted');
  assert.equal(snapshot.accepted, accepted);
  assert.equal(posts, 1);
  assert.equal(lookups, 1);
});

test('absent lookup becomes user-retryable and Retry explicitly posts once more', async () => {
  let posts = 0;
  const adapter: AnswerSubmissionAdapter = {
    submit: async () => {
      posts += 1;
      if (posts === 1) throw new AnswerSubmissionError(503, 'service_unavailable', 'down');
      return accepted;
    },
    lookup: async () => null,
  };
  const actor = createAnswerSubmissionActor({intent, lease: lease(), adapter});
  await reaches(actor, 'retryable');
  assert.equal(posts, 1);
  actor.send({type: 'RETRY'});
  await waitFor(actor, (snapshot) => snapshot.matches('accepted'));
  assert.equal(posts, 2);
});

test('every typed request correction remains editable without reconciliation', async () => {
  const failures: Array<[number, ConstructorParameters<typeof AnswerSubmissionError>[1]]> = [
    [400, 'invalid_request'],
    [403, 'scope_forbidden'],
    [404, 'conversation_missing'],
    [413, 'attachment_rejected'],
    [422, 'invalid_request'],
  ];
  for (const [status, kind] of failures) {
    let lookups = 0;
    const adapter: AnswerSubmissionAdapter = {
      submit: async () => { throw new AnswerSubmissionError(status, kind, 'edit'); },
      lookup: async () => {
        lookups += 1;
        return null;
      },
    };
    const actor = createAnswerSubmissionActor({intent, lease: lease(), adapter});
    await reaches(actor, 'editable');
    assert.equal(lookups, 0);
    actor.send({type: 'DISCARD'});
  }
});

test('editable failures restore the same lease only after Edit', async () => {
  const held = lease();
  const adapter: AnswerSubmissionAdapter = {
    submit: async () => { throw new AnswerSubmissionError(413, 'attachment_rejected', 'large'); },
    lookup: async () => accepted,
  };
  const actor = createAnswerSubmissionActor({intent, lease: held, adapter});
  await reaches(actor, 'editable');
  assert.equal(held.restored, 0);
  actor.send({type: 'EDIT'});
  await waitFor(actor, (snapshot) => snapshot.matches('edited'));
  assert.equal(held.restored, 1);
});
