// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {
  assign,
  createActor,
  fromPromise,
  setup,
  type ActorRefFrom,
} from 'xstate';
import {
  AnswerSubmissionError,
  type AnswerSubmissionAdapter,
  type AnswerSubmissionIntent,
} from '../api/answer-submission.ts';
import type {AcceptedAnswer} from '../api/conversations.ts';
import type {AttachmentLease} from './attachment-store.ts';

export type AnswerSubmissionEvent =
  | {type: 'RETRY'}
  | {type: 'EDIT'}
  | {type: 'DISCARD'}
  | {type: 'HANDOFF'};

export type AnswerSubmissionStatus =
  | 'submitting'
  | 'reconciling'
  | 'editable'
  | 'retryable'
  | 'conflict'
  | 'login'
  | 'accepted'
  | 'handedOff'
  | 'edited'
  | 'discarded';

interface MachineInput {
  intent: AnswerSubmissionIntent;
  lease: AttachmentLease;
  adapter: AnswerSubmissionAdapter;
}

interface MachineContext extends MachineInput {
  accepted: AcceptedAnswer | null;
  error: AnswerSubmissionError | null;
}

function submissionError(value: unknown): AnswerSubmissionError {
  return value instanceof AnswerSubmissionError
    ? value
    : new AnswerSubmissionError(0, 'ambiguous', 'Answer submission failed');
}

function actorError(event: unknown): AnswerSubmissionError {
  return submissionError((event as {error?: unknown}).error);
}

function actorOutput(event: unknown): AcceptedAnswer | null {
  return (event as {output?: AcceptedAnswer}).output ?? null;
}

export const answerSubmissionMachine = setup({
  types: {
    context: {} as MachineContext,
    input: {} as MachineInput,
    events: {} as AnswerSubmissionEvent,
  },
  actors: {
    post: fromPromise(({input, signal}: {
      input: MachineContext;
      signal: AbortSignal;
    }) => input.adapter.submit(input.intent, input.lease.items.map((item) => item.file), signal)),
    lookup: fromPromise(({input, signal}: {
      input: MachineContext;
      signal: AbortSignal;
    }) => input.adapter.lookup(input.intent.submissionId, signal)),
  },
  guards: {
    editableFailure: ({event}) => {
      const error = actorError(event);
      return [
        'invalid_request',
        'attachment_rejected',
        'scope_forbidden',
        'conversation_missing',
      ].includes(error.kind)
        || error.status === 400 || error.status === 403
        || error.status === 413 || error.status === 422;
    },
    conflictFailure: ({event}) => actorError(event).status === 409,
    loginFailure: ({event}) => actorError(event).status === 401,
    lookupMissing: ({event}) => actorError(event).status === 404,
    lookupFound: ({event}) => actorOutput(event) !== null,
  },
  actions: {
    rememberError: assign({
      error: ({event}) => actorError(event),
    }),
    rememberAccepted: assign({
      accepted: ({event}) => actorOutput(event),
      error: null,
    }),
    acceptLease: ({context}) => context.lease.accept(),
    restoreLease: ({context}) => context.lease.restore(),
    discardLease: ({context}) => context.lease.discard(),
  },
}).createMachine({
  id: 'answerSubmission',
  context: ({input}) => ({...input, accepted: null, error: null}),
  initial: 'submitting',
  states: {
    submitting: {
      invoke: {
        id: 'post',
        src: 'post',
        input: ({context}) => context,
        onDone: {target: 'accepted', actions: 'rememberAccepted'},
        onError: [
          {guard: 'editableFailure', target: 'editable', actions: 'rememberError'},
          {guard: 'conflictFailure', target: 'conflict', actions: 'rememberError'},
          {guard: 'loginFailure', target: 'login', actions: 'rememberError'},
          {target: 'reconciling', actions: 'rememberError'},
        ],
      },
    },
    reconciling: {
      invoke: {
        id: 'lookup',
        src: 'lookup',
        input: ({context}) => context,
        onDone: [
          {guard: 'lookupFound', target: 'accepted', actions: 'rememberAccepted'},
          {target: 'retryable'},
        ],
        onError: [
          {guard: 'loginFailure', target: 'login', actions: 'rememberError'},
          {guard: 'lookupMissing', target: 'retryable', actions: 'rememberError'},
          {target: 'retryable', actions: 'rememberError'},
        ],
      },
    },
    editable: {on: {EDIT: 'edited', DISCARD: 'discarded'}},
    retryable: {on: {RETRY: 'submitting', EDIT: 'edited', DISCARD: 'discarded'}},
    conflict: {on: {EDIT: 'edited', DISCARD: 'discarded'}},
    login: {on: {EDIT: 'edited', DISCARD: 'discarded'}},
    accepted: {on: {HANDOFF: 'handedOff'}},
    handedOff: {type: 'final', entry: 'acceptLease'},
    edited: {type: 'final', entry: 'restoreLease'},
    discarded: {type: 'final', entry: 'discardLease'},
  },
});

export type AnswerSubmissionActor = ActorRefFrom<typeof answerSubmissionMachine>;

export interface AnswerSubmissionSnapshot {
  readonly submissionId: string;
  readonly conversationId: string | null;
  readonly status: AnswerSubmissionStatus;
  readonly accepted: AcceptedAnswer | null;
  readonly error: AnswerSubmissionError | null;
}

export function answerSubmissionSnapshot(actor: AnswerSubmissionActor): AnswerSubmissionSnapshot {
  const snapshot = actor.getSnapshot();
  return {
    submissionId: snapshot.context.intent.submissionId,
    conversationId: snapshot.context.intent.conversationId,
    status: String(snapshot.value) as AnswerSubmissionStatus,
    accepted: snapshot.context.accepted,
    error: snapshot.context.error,
  };
}

export function createAnswerSubmissionActor(input: MachineInput): AnswerSubmissionActor {
  return createActor(answerSubmissionMachine, {input});
}
