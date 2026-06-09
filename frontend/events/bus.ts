// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import { createNanoEvents, type Emitter } from 'nanoevents';

export interface WorkspaceRecord {
  workspace: string;
  displayName: string;
  embeddingModel: string;
}

export interface Exchange {
  query: string;
  answer: string;
  imageIds: string[];
}

export interface DlightragEvents {
  workspaceCreated: { workspace: string; displayName: string };
  workspaceDeleted: { workspace: string; nextWorkspace: string };
  workspaceToggled: { workspaces: readonly string[] };
  ingestWorkspaceChanged: { workspace: string };
  chatExchangeComplete: void;
}

export const bus: Emitter<DlightragEvents> = createNanoEvents<DlightragEvents>();
