// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import { Store } from './base';
import { workspaceStore } from './workspaceStore';

class IngestStore extends Store {
  #workspace: string | null = null;

  get workspace(): string {
    return this.#workspace ?? workspaceStore.primary;
  }

  set(workspace: string): void {
    this.#workspace = workspace;
    this.emit('ingestWorkspaceChanged', { workspace });
  }

  resetToPrimary(): void {
    this.#workspace = null;
    this.emit('ingestWorkspaceChanged', { workspace: this.workspace });
  }
}

export const ingestStore = new IngestStore();
