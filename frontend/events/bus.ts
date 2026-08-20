// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import { createNanoEvents, type Emitter } from 'nanoevents';

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
}

export const bus: Emitter<DlightragEvents> = createNanoEvents<DlightragEvents>();
