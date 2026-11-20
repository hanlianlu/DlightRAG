// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {
  assign,
  createActor,
  fromPromise,
  setup,
  type ActorRefFrom,
} from 'xstate';
import type {AnswerSubmissionAdapter, AnswerSubmissionIntent} from '../api/answer-submission.ts';
import {AnswerSubmissionError} from '../api/web-command-error.ts';
import type {AnswerSubmissionResponse} from '../api/conversations.ts';
import type {AttachmentLease} from './attachment-store.ts';

type AnswerSubmissionEvent =
  | {type: 'RETRY'}
  | {type: 'EDIT'}
  | {type: 'DISCARD'}
  | {type: 'HANDOFF'};

type AnswerSubmissionStatus =
  | 'submitting'
  | 'reconciling'
  | 'editable'
  | 'retryable'
  | 'conflict'
  | 'accepted'
  | 'delivered'
  | 'handedOff'
  | 'edited'
  | 'discarded';

interface MachineInput {
  intent: AnswerSubmissionIntent;
  lease: AttachmentLease;
  adapter: AnswerSubmissionAdapter;
}

interface MachineContext extends MachineInput {
  outcome: AnswerSubmissionResponse | null;
  error: AnswerSubmissionError | null;
}

function submissionError(value: unknown): AnswerSubmissionError {
  return value instanceof AnswerSubmissionError
    ? value
    : new AnswerSubmissionError(0, 'ambiguous', '');
}

// What an invoked request answered, handed to the guards and actions of the transition that has it. A guard
// or an action declared by name sees only the machine's own events, so the transition says what to pass.
const errorOf = ({event}: {event: {error: unknown}}) => ({error: event.error});
const outputOf = ({event}: {event: {output: unknown}}) => ({output: event.output});

const answerSubmissionMachine = setup({
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
    // Refused before acceptance. A 401 among them signs the whole page out (api/wire.ts),
    // so the shell, not this submission, offers the way back in.
    editableFailure: (_, params: {error: unknown}) => {
      const error = submissionError(params.error);
      return [
        'invalid_request',
        'attachment_rejected',
        'scope_forbidden',
        'conversation_missing',
      ].includes(error.kind)
        || error.status === 400 || error.status === 401 || error.status === 403
        || error.status === 413 || error.status === 422;
    },
    conflictFailure: (_, params: {error: unknown}) => submissionError(params.error).status === 409,
    acceptedOutcome: (_, params: {output: unknown}) =>
      (params.output as AnswerSubmissionResponse | null)?.kind === 'accepted',
    deliveredOutcome: (_, params: {output: unknown}) => {
      const outcome = params.output as AnswerSubmissionResponse | null;
      return outcome !== null && outcome.kind !== 'accepted';
    },
    lookupFound: (_, params: {output: unknown}) => params.output !== null,
  },
  actions: {
    rememberError: assign({
      error: (_, params: {error: unknown}) => submissionError(params.error),
    }),
    rememberOutcome: assign({
      outcome: (_, params: {output: unknown}) =>
        (params.output ?? null) as AnswerSubmissionResponse | null,
      error: null,
    }),
    acceptLease: ({context}) => context.lease.accept(),
    restoreLease: ({context}) => context.lease.restore(),
    discardLease: ({context}) => context.lease.discard(),
  },
}).createMachine({
  id: 'answerSubmission',
  context: ({input}) => ({...input, outcome: null, error: null}),
  initial: 'submitting',
  states: {
    submitting: {
      invoke: {
        id: 'post',
        src: 'post',
        input: ({context}) => context,
        onDone: [
          {
            guard: {type: 'acceptedOutcome', params: outputOf},
            target: 'accepted',
            actions: {type: 'rememberOutcome', params: outputOf},
          },
          {
            guard: {type: 'deliveredOutcome', params: outputOf},
            target: 'delivered',
            actions: {type: 'rememberOutcome', params: outputOf},
          },
          {target: 'accepted', actions: {type: 'rememberOutcome', params: outputOf}},
        ],
        onError: [
          {
            guard: {type: 'editableFailure', params: errorOf},
            target: 'editable',
            actions: {type: 'rememberError', params: errorOf},
          },
          {
            guard: {type: 'conflictFailure', params: errorOf},
            target: 'conflict',
            actions: {type: 'rememberError', params: errorOf},
          },
          {target: 'reconciling', actions: {type: 'rememberError', params: errorOf}},
        ],
      },
    },
    reconciling: {
      invoke: {
        id: 'lookup',
        src: 'lookup',
        input: ({context}) => context,
        onDone: [
          {
            guard: {type: 'lookupFound', params: outputOf},
            target: 'accepted',
            actions: {type: 'rememberOutcome', params: outputOf},
          },
          {target: 'retryable'},
        ],
        // A lookup that found nothing and one that failed leave the reader the same choice.
        onError: {target: 'retryable', actions: {type: 'rememberError', params: errorOf}},
      },
    },
    editable: {on: {EDIT: 'edited', DISCARD: 'discarded'}},
    retryable: {on: {RETRY: 'submitting', EDIT: 'edited', DISCARD: 'discarded'}},
    conflict: {on: {EDIT: 'edited', DISCARD: 'discarded'}},
    accepted: {on: {HANDOFF: 'handedOff'}},
    delivered: {type: 'final', entry: 'acceptLease'},
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
  /** What the send produced: an accepted turn, or delivery to the Run already live. */
  readonly outcome: AnswerSubmissionResponse | null;
  readonly error: AnswerSubmissionError | null;
}

export function answerSubmissionSnapshot(actor: AnswerSubmissionActor): AnswerSubmissionSnapshot {
  const snapshot = actor.getSnapshot();
  return {
    submissionId: snapshot.context.intent.submissionId,
    conversationId: snapshot.context.intent.conversationId,
    status: String(snapshot.value) as AnswerSubmissionStatus,
    outcome: snapshot.context.outcome,
    error: snapshot.context.error,
  };
}

export function createAnswerSubmissionActor(input: MachineInput): AnswerSubmissionActor {
  return createActor(answerSubmissionMachine, {input});
}
