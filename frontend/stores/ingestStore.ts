// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import { workspaceStore } from './workspaceStore';

class IngestStore {
  #workspace: string | null = null;

  get workspace(): string {
    return this.#workspace ?? workspaceStore.primary;
  }

  set(workspace: string): void {
    this.#workspace = workspace;
  }

  resetToPrimary(): void {
    this.#workspace = null;
  }
}

export const ingestStore = new IngestStore();
