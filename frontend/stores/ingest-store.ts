// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import { Store } from './base.ts';
import { workspaceStore, type WorkspaceStore } from './workspace-store.ts';

/** The Files target: an explicit workspace, or else the search scope's primary. */
export class IngestStore extends Store {
  readonly #workspaces: WorkspaceStore;
  #workspace: string | null = null;
  #published: string;

  constructor(workspaces: WorkspaceStore) {
    super();
    this.#workspaces = workspaces;
    this.#published = this.workspace;
    // Following the primary means following its changes, not reading it once:
    // otherwise Files would list one workspace while acting on another.
    workspaces.subscribe(() => {
      if (this.workspace !== this.#published) this.#publish();
    });
  }

  get workspace(): string {
    return this.#workspace ?? this.#workspaces.primary;
  }

  set(workspace: string): void {
    this.#workspace = workspace;
    this.#publish();
  }

  resetToPrimary(): void {
    this.#workspace = null;
    this.#publish();
  }

  #publish(): void {
    this.#published = this.workspace;
    this.changed();
  }
}

export const ingestStore = new IngestStore(workspaceStore);
