// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {test} from 'node:test';
import assert from 'node:assert/strict';

import {isTerminalTurnState, type ChatTurnView} from './chat-views.ts';

// Keyed by every turn state, so the type checker makes a new state take a side here.
const TERMINAL: Record<ChatTurnView['state'], boolean> = {
  pending: false,
  streaming: false,
  retryable: false,
  succeeded: true,
  failed: true,
  cancelled: true,
};

test('a turn is terminal once its Run succeeded, failed, or was cancelled', () => {
  for (const [state, terminal] of Object.entries(TERMINAL)) {
    assert.equal(isTerminalTurnState(state as ChatTurnView['state']), terminal, state);
  }
});
