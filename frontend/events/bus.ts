// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import { createNanoEvents, type Emitter } from 'nanoevents';
import type {ConversationHistory, ConversationSummary} from '../api/conversations.ts';

export interface WorkspaceRecord {
  workspace: string;
  displayName: string;
  embeddingModel: string;
}

export interface DlightragEvents {
  workspaceCreated: { workspace: string; displayName: string };
  workspaceDeleted: { workspace: string; nextWorkspace: string };
  workspaceToggled: { workspaces: readonly string[] };
  ingestWorkspaceChanged: { workspace: string };
  conversationListChanged: { conversations: readonly ConversationSummary[] };
  conversationSelected: { conversationId: string | null };
  conversationHistoryChanged: { history: ConversationHistory };
  conversationAnswerSaved: { conversationId: string };
  conversationSaveCheckRequested: { conversationId: string };
  conversationDeferredSelectionReady: { conversationId: string };
  conversationStreamChanged: { active: boolean };
}

export const bus: Emitter<DlightragEvents> = createNanoEvents<DlightragEvents>();
