// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import { createNanoEvents, type Emitter } from 'nanoevents';
import type {ConversationHistory, ConversationSummary} from '../api/conversations.ts';

export interface WorkspaceRecord {
  workspace: string;
  displayName: string;
  embeddingModel: string;
}

export interface DlightragEvents {
  workspaceCreated: (payload: { workspace: string; displayName: string }) => void;
  workspaceDeleted: (payload: { workspace: string; nextWorkspace: string }) => void;
  workspaceToggled: (payload: { workspaces: readonly string[] }) => void;
  ingestWorkspaceChanged: (payload: { workspace: string }) => void;
  conversationListChanged: (payload: { conversations: readonly ConversationSummary[] }) => void;
  conversationSelected: (payload: { conversationId: string | null }) => void;
  conversationHistoryChanged: (payload: { history: ConversationHistory }) => void;
  conversationAnswerSaved: (payload: { conversationId: string }) => void;
  conversationSaveCheckRequested: (payload: { conversationId: string }) => void;
  conversationDeferredSelectionReady: (payload: { conversationId: string }) => void;
  conversationStreamChanged: (payload: { active: boolean }) => void;
}

export const bus: Emitter<DlightragEvents> = createNanoEvents<DlightragEvents>();
