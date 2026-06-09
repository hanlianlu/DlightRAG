// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import { Store } from './base';
import { type WorkspaceRecord } from '../events/bus';

const PRIMARY_COOKIE = 'dlightrag_workspace';
const ACTIVE_COOKIE = 'dlightrag_workspace_ids';

function setCookie(name: string, value: string): void {
  document.cookie = `${name}=${encodeURIComponent(value)};path=/;SameSite=Lax`;
}

function clearCookie(name: string): void {
  document.cookie = `${name}=;path=/;SameSite=Lax;Max-Age=0`;
}

class WorkspaceStore extends Store {
  #records: WorkspaceRecord[] = [];
  #active: string[] = [];

  get records(): readonly WorkspaceRecord[] {
    return this.#records;
  }

  get active(): readonly string[] {
    return this.#active;
  }

  get primary(): string {
    return this.#active[0] || '';
  }

  init(records: WorkspaceRecord[], active: string[]): void {
    this.#records = records;
    this.#active = active;
    this.#syncCookies();
  }

  toggle(workspace: string): void {
    const idx = this.#active.indexOf(workspace);
    if (idx >= 0) {
      if (this.#active.length <= 1) return;
      this.#active.splice(idx, 1);
    } else {
      this.#active.push(workspace);
    }
    this.#syncCookies();
    this.emit('workspaceToggled', { workspaces: [...this.#active] });
  }

  select(workspace: string): void {
    if (!workspace) return;
    this.#active = [workspace];
    this.#syncCookies();
    this.emit('workspaceToggled', { workspaces: [...this.#active] });
  }

  selectAll(): void {
    if (this.#records.length === 0) return;
    const allIds = this.#records.map((r) => r.workspace);
    const allSelected = allIds.every((w) => this.#active.includes(w));
    this.#active = allSelected ? [allIds[0]!] : [...allIds];
    this.#syncCookies();
    this.emit('workspaceToggled', { workspaces: [...this.#active] });
  }

  add(record: WorkspaceRecord): void {
    if (!this.#records.some((r) => r.workspace === record.workspace)) {
      this.#records.push(record);
    }
    this.#active = [record.workspace];
    this.#syncCookies();
    this.emit('workspaceCreated', { workspace: record.workspace, displayName: record.displayName });
  }

  remove(workspace: string, nextWorkspace: string): void {
    this.#records = this.#records.filter((r) => r.workspace !== workspace);
    const remaining = this.#active.filter((a) => a !== workspace);
    this.#active = remaining.length > 0 ? remaining : nextWorkspace ? [nextWorkspace] : [];
    this.#syncCookies();
    this.emit('workspaceDeleted', { workspace, nextWorkspace });
  }

  #syncCookies(): void {
    if (this.#active.length === 0) {
      clearCookie(PRIMARY_COOKIE);
      clearCookie(ACTIVE_COOKIE);
      return;
    }
    setCookie(PRIMARY_COOKIE, this.#active[0]!);
    setCookie(ACTIVE_COOKIE, this.#active.join(','));
  }
}

export const workspaceStore = new WorkspaceStore();
