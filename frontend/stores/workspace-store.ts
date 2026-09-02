// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import { Store } from './base';
import type { WorkspaceRecord } from '../events/bus';
import type { WorkspacePage } from '../api/workspaces.ts';
import {isAbortError} from '../lib/errors.ts';

const PRIMARY_COOKIE = 'dlightrag_workspace';
const ACTIVE_COOKIE = 'dlightrag_workspace_ids';

export type WorkspacePageLoader = (
  cursor: string | null,
  signal?: AbortSignal,
) => Promise<WorkspacePage>;

export type WorkspaceLoadMoreState = 'idle' | 'loading' | 'error';

function setCookie(name: string, value: string): void {
  // biome-ignore lint/suspicious/noDocumentCookie: cookies are the workspace preference channel
  document.cookie = `${name}=${encodeURIComponent(value)};path=/;SameSite=Lax`;
}

function clearCookie(name: string): void {
  // biome-ignore lint/suspicious/noDocumentCookie: cookies are the workspace preference channel
  document.cookie = `${name}=;path=/;SameSite=Lax;Max-Age=0`;
}

class WorkspaceStore extends Store {
  #records: WorkspaceRecord[] = [];
  #known: string[] = [];
  #active: string[] = [];
  #primary = '';
  #loader: WorkspacePageLoader | null = null;
  #nextCursor: string | null = null;
  #loadMoreState: WorkspaceLoadMoreState = 'idle';
  #loadMoreFlight: Promise<void> | null = null;
  #loadMoreController: AbortController | null = null;
  #loadMoreGeneration = 0;

  get records(): readonly WorkspaceRecord[] {
    return this.#records;
  }

  get knownWorkspaces(): readonly string[] {
    return this.#known;
  }

  get active(): readonly string[] {
    return this.#active;
  }

  get primary(): string {
    return this.#primary || this.#fallbackPrimary();
  }

  get hasMoreWorkspaces(): boolean {
    return this.#loader !== null && this.#nextCursor !== null;
  }

  get workspaceLoadMoreState(): WorkspaceLoadMoreState {
    return this.#loadMoreState;
  }

  init(
    records: WorkspaceRecord[],
    active: string[],
    primary = '',
    loader: WorkspacePageLoader | null = null,
    nextCursor: string | null = null,
    knownWorkspaces: string[] | null = null,
  ): void {
    this.#invalidateLoadMore();
    this.#records = records;
    // The full authorized id set stays separate from the bounded display
    // page: active/primary are server-validated against the full catalog and
    // must never be re-validated (and silently narrowed, then persisted to
    // cookies) against the first display page alone.
    this.#known = knownWorkspaces ?? records.map((record) => record.workspace);
    this.#loader = loader;
    this.#nextCursor = nextCursor;
    this.#active = this.#validActive(active);
    this.#primary = this.#validPrimary(primary);
    this.#syncCookies();
    this.emit('workspaceToggled', { workspaces: [...this.#active] });
  }

  loadMoreWorkspaces(): Promise<void> {
    if (this.#loadMoreFlight !== null) return this.#loadMoreFlight;
    if (this.#loader === null || this.#nextCursor === null) return Promise.resolve();
    const flight = this.#loadMorePage(this.#nextCursor);
    this.#loadMoreFlight = flight;
    void flight.finally(() => {
      if (this.#loadMoreFlight === flight) this.#loadMoreFlight = null;
    });
    return flight;
  }

  async #loadMorePage(cursor: string): Promise<void> {
    this.#loadMoreController?.abort();
    const controller = new AbortController();
    this.#loadMoreController = controller;
    const generation = this.#loadMoreGeneration;
    this.#loadMoreState = 'loading';
    this.changed();
    try {
      const page = await this.#loader!(cursor, controller.signal);
      if (
        controller !== this.#loadMoreController
        || generation !== this.#loadMoreGeneration
        || this.#nextCursor !== cursor
      ) {
        if (controller === this.#loadMoreController) this.#loadMoreState = 'idle';
        return;
      }
      const known = new Set(this.#records.map((record) => record.workspace));
      const appended: WorkspaceRecord[] = [];
      for (const item of page.workspaces) {
        if (!item.workspace || known.has(item.workspace)) continue;
        known.add(item.workspace);
        appended.push({
          workspace: item.workspace,
          displayName: item.display_name || item.workspace,
          embeddingModel: item.embedding_model || '',
        });
      }
      this.#records = [...this.#records, ...appended];
      this.#nextCursor = page.next_cursor ?? null;
      this.#loadMoreState = 'idle';
      this.changed();
    } catch (error) {
      if (
        controller !== this.#loadMoreController
        || generation !== this.#loadMoreGeneration
      ) return;
      if (isAbortError(error)) return;
      this.#loadMoreState = 'error';
      this.changed();
    } finally {
      if (this.#loadMoreController === controller) this.#loadMoreController = null;
    }
  }

  #invalidateLoadMore(): void {
    this.#loadMoreController?.abort();
    this.#loadMoreController = null;
    this.#loadMoreGeneration += 1;
    this.#loadMoreFlight = null;
    this.#loadMoreState = 'idle';
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
    const knownIds = this.#known.length > 0
      ? [...this.#known]
      : this.#records.map((record) => record.workspace);
    if (knownIds.length === 0) return;
    this.#active = knownIds;
    this.#primary = this.#defaultWorkspace();
    this.#syncCookies();
    this.emit('workspaceToggled', { workspaces: [...this.#active] });
  }

  /** Restore an earlier composer scope after an editable submission failure. */
  restoreActive(workspaces: readonly string[]): void {
    this.#active = this.#validActive([...workspaces]);
    this.#primary = this.#validPrimary(this.#primary);
    this.#syncCookies();
    this.emit('workspaceToggled', {workspaces: [...this.#active]});
  }

  add(record: WorkspaceRecord): void {
    if (!this.#records.some((r) => r.workspace === record.workspace)) {
      this.#records.push(record);
    }
    if (!this.#known.includes(record.workspace)) this.#known.push(record.workspace);
    this.#active = [record.workspace];
    this.#primary = record.workspace;
    this.#syncCookies();
    this.emit('workspaceCreated', { workspace: record.workspace, displayName: record.displayName });
  }

  remove(workspace: string, nextWorkspace: string): void {
    this.#records = this.#records.filter((r) => r.workspace !== workspace);
    this.#known = this.#known.filter((known) => known !== workspace);
    if (nextWorkspace && !this.#known.includes(nextWorkspace)) {
      this.#known.push(nextWorkspace);
    }
    const remaining = this.#active.filter((a) => a !== workspace);
    this.#active = remaining.length > 0 ? remaining : nextWorkspace ? [nextWorkspace] : [];
    if (this.#primary === workspace || !this.#active.includes(this.#primary)) {
      this.#primary = this.#fallbackPrimary(nextWorkspace);
    }
    this.#syncCookies();
    this.emit('workspaceDeleted', { workspace, nextWorkspace });
  }

  #validActive(active: string[]): string[] {
    const known = new Set(this.#known);
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
    if (this.#known.includes('default')) return 'default';
    return this.#records[0]?.workspace || '';
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
