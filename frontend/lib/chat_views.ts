// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The turn view contract shared by projection, components, and tests. It is a
 *  data shape, not a component concern, so it lives in lib where the pure
 *  projection logic can depend on it without reaching into ui/. */

import type {
  AnswerPresentation,
  ConversationAttachmentReference,
} from '../api/conversations.ts';
import type {ToolRow} from './tool_events.ts';

export interface ChatTurnView {
  id: string;
  userText: string;
  userAttachments: readonly ConversationAttachmentReference[];
  runId: string;
  state: 'pending' | 'streaming' | 'succeeded' | 'failed' | 'cancelled' | 'retryable';
  streamText: string;
  presentation: AnswerPresentation | null;
  usage: Record<string, unknown>;
  evidence: Record<string, number>;
  error: string;
  progress: string;
  liveStatus: string;
  sawChildren: boolean;
  cancelRequested: boolean;
  steeringMessages: readonly string[];
  toolRows: readonly ToolRow[];
  toolTotal: number;
  toolExpanded: boolean;
}
