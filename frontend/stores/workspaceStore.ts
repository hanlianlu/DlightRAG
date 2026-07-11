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
  #primary = '';

  get records(): readonly WorkspaceRecord[] {
    return this.#records;
  }

  get active(): readonly string[] {
    return this.#active;
  }

  get primary(): string {
    return this.#primary || this.#fallbackPrimary();
  }

  get defaultWorkspace(): string {
    return this.#defaultWorkspace();
  }

  init(records: WorkspaceRecord[], active: string[], primary = ''): void {
    this.#records = records;
    this.#active = this.#validActive(active);
    this.#primary = this.#validPrimary(primary);
    this.#syncCookies();
  }

  toggle(workspace: string): void {
    const idx = this.#active.indexOf(workspace);
    if (idx >= 0) {
      if (this.#active.length <= 1) return;
      this.#active.splice(idx, 1);
      if (this.#primary === workspace) this.#primary = this.#fallbackPrimary();
    } else {
      this.#active.push(workspace);
      this.#primary = workspace;
    }
    this.#syncCookies();
    this.emit('workspaceToggled', { workspaces: [...this.#active] });
  }

  select(workspace: string): void {
    if (!workspace) return;
    this.#active = [workspace];
    this.#primary = workspace;
    this.#syncCookies();
    this.emit('workspaceToggled', { workspaces: [...this.#active] });
  }

  selectAll(): void {
    if (this.#records.length === 0) return;
    const allIds = this.#records.map((r) => r.workspace);
    this.#active = [...allIds];
    this.#primary = this.#defaultWorkspace();
    this.#syncCookies();
    this.emit('workspaceToggled', { workspaces: [...this.#active] });
  }

  add(record: WorkspaceRecord): void {
    if (!this.#records.some((r) => r.workspace === record.workspace)) {
      this.#records.push(record);
    }
    this.#active = [record.workspace];
    this.#primary = record.workspace;
    this.#syncCookies();
    this.emit('workspaceCreated', { workspace: record.workspace, displayName: record.displayName });
  }

  remove(workspace: string, nextWorkspace: string): void {
    this.#records = this.#records.filter((r) => r.workspace !== workspace);
    const remaining = this.#active.filter((a) => a !== workspace);
    this.#active = remaining.length > 0 ? remaining : nextWorkspace ? [nextWorkspace] : [];
    if (this.#primary === workspace || !this.#active.includes(this.#primary)) {
      this.#primary = this.#fallbackPrimary(nextWorkspace);
    }
    this.#syncCookies();
    this.emit('workspaceDeleted', { workspace, nextWorkspace });
  }

  #validActive(active: string[]): string[] {
    const known = new Set(this.#records.map((record) => record.workspace));
    const result: string[] = [];
    active.forEach((workspace) => {
      if (known.has(workspace) && !result.includes(workspace)) result.push(workspace);
    });
    if (result.length > 0) return result;
    const fallback = this.#defaultWorkspace();
    return fallback ? [fallback] : [];
  }

  #validPrimary(primary: string): string {
    if (primary && this.#active.includes(primary)) return primary;
    return this.#fallbackPrimary(primary);
  }

  #defaultWorkspace(): string {
    return this.#records.find((record) => record.workspace === 'default')?.workspace
      || this.#records[0]?.workspace
      || '';
  }

  #fallbackPrimary(preferred = ''): string {
    if (preferred && this.#active.includes(preferred)) return preferred;
    const defaultWorkspace = this.#defaultWorkspace();
    if (defaultWorkspace && this.#active.includes(defaultWorkspace)) return defaultWorkspace;
    return this.#active[0] || '';
  }

  #syncCookies(): void {
    if (this.#active.length === 0) {
      clearCookie(PRIMARY_COOKIE);
      clearCookie(ACTIVE_COOKIE);
      return;
    }
    setCookie(PRIMARY_COOKIE, this.primary);
    setCookie(ACTIVE_COOKIE, this.#active.join(','));
  }
}

export const workspaceStore = new WorkspaceStore();
