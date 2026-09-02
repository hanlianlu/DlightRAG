// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** Pure projection of one answer-run event stream onto one turn view. The
 *  feature component reduces a batch through this module; nothing here
 *  touches the DOM or dispatches events. */

import {msg} from '@lit/localize';

import type {AnswerPresentation} from '../api/conversations.ts';
import type {AnswerRunEvent} from './run-controller.ts';
import {localizedRunErrorPayload} from './run-errors.ts';
import type {ChatTurnView} from './chat-views.ts';
import {
  applyToolEvent,
  toolStatusText,
  type ToolEventPayload,
} from './tool-events.ts';

type AnswerPhase = 'routing' | 'planning' | 'searching' | 'researching' | 'generating';

export const ANSWER_PHASE_LABELS = {
  routing: 'Routing answer...',
  planning: 'Planning answer...',
  searching: 'Searching knowledge base...',
  researching: 'Researching sources...',
  generating: 'Generating answer...',
} as const satisfies Record<AnswerPhase, string>;

export function answerPhaseLabel(phase: string): string | null {
  if (!Object.hasOwn(ANSWER_PHASE_LABELS, phase)) return null;
  return ANSWER_PHASE_LABELS[phase as AnswerPhase];
}

interface DonePayload {
  status: 'succeeded' | 'cancelled';
  presentation: AnswerPresentation | null;
  usage?: Record<string, unknown>;
  evidence?: Record<string, number>;
}

function isDonePayload(payload: unknown): payload is DonePayload {
  if (payload === null || typeof payload !== 'object' || Array.isArray(payload)) return false;
  const candidate = payload as Record<string, unknown>;
  return candidate.status === 'succeeded' || candidate.status === 'cancelled';
}

function phaseText(phase: string): string | null {
  const label = answerPhaseLabel(phase);
  return label === null ? null : msg(label, {id: `chatFeature.phase.${phase}`});
}

/**
 * Fold one event onto the current turn view. Returns the same reference when
 * the event does not change the view (memory events, unknown phases), so the
 * caller can cheaply detect a no-op batch.
 */
export function applyAnswerEvent(turn: ChatTurnView, event: AnswerRunEvent): ChatTurnView {
  switch (event.kind) {
    case 'memory':
      return turn;
    case 'token':
      return {
        ...turn,
        state: 'streaming',
        streamText: turn.streamText + event.text,
        progress: turn.streamText ? turn.progress : '',
        error: '',
      };
    case 'reset':
      return {...turn, state: 'pending', streamText: '', progress: '', error: ''};
    case 'progress': {
      const payload = event.payload as {phase?: string};
      const text = phaseText(String(payload?.phase || ''));
      if (text === null) return turn;
      return {...turn, progress: text, liveStatus: text, error: ''};
    }
    case 'tool': {
      const info = event.payload as ToolEventPayload;
      if (!info || typeof info.tool_name !== 'string') return turn;
      const toolRows = applyToolEvent(turn.toolRows, event.eventType, info);
      const toolTotal = event.eventType === 'tool_start' ? turn.toolTotal + 1 : turn.toolTotal;
      const text = toolStatusText(toolRows);
      return {
        ...turn,
        toolRows,
        toolTotal,
        progress: text,
        liveStatus: text,
        sawChildren: turn.sawChildren || info.tool_name === 'spawn_agent',
        error: '',
      };
    }
    case 'error': {
      const message = localizedRunErrorPayload(event.payload);
      return {...turn, state: 'failed', error: message, progress: '', liveStatus: message};
    }
    case 'done': {
      const payload = event.payload;
      if (!isDonePayload(payload)) {
        const message = msg('Service error. Please try again.', {id: 'chatFeature.serviceError'});
        return {...turn, state: 'failed', error: message, progress: '', liveStatus: message};
      }
      if (payload.status === 'cancelled') {
        return {
          ...turn,
          state: 'cancelled',
          progress: '',
          liveStatus: msg('Answer stopped', {id: 'chatFeature.answerStopped'}),
        };
      }
      if (!payload.presentation) {
        const message = msg('Service error. Please try again.', {id: 'chatFeature.serviceError'});
        return {...turn, state: 'failed', error: message, progress: '', liveStatus: message};
      }
      return {
        ...turn,
        state: 'succeeded',
        presentation: payload.presentation,
        streamText: payload.presentation.answer_text,
        usage: payload.usage ?? {},
        evidence: payload.evidence ?? {},
        progress: '',
        liveStatus: msg('Answer ready', {id: 'chatFeature.answerReady'}),
      };
    }
  }
}
