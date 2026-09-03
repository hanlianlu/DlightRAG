// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import { Store } from './base.ts';
import type {WorkspacePage, WorkspacePageItem} from '../api/workspaces.ts';
import {KeysetPager, type PageLoadState} from '../lib/paged.ts';

export type WorkspaceRecord = WorkspacePageItem;

const PRIMARY_COOKIE = 'dlightrag_workspace';
const ACTIVE_COOKIE = 'dlightrag_workspace_ids';

export type WorkspacePageLoader = (
  cursor: string | null,
  signal?: AbortSignal,
) => Promise<WorkspacePage>;

export type WorkspaceLoadMoreState = PageLoadState;

function setCookie(name: string, value: string): void {
  // biome-ignore lint/suspicious/noDocumentCookie: cookies are the workspace preference channel
  document.cookie = `${name}=${encodeURIComponent(value)};path=/;SameSite=Lax`;
}

function clearCookie(name: string): void {
  // biome-ignore lint/suspicious/noDocumentCookie: cookies are the workspace preference channel
  document.cookie = `${name}=;path=/;SameSite=Lax;Max-Age=0`;
}

export class WorkspaceStore extends Store {
  #records: WorkspaceRecord[] = [];
  #known: string[] = [];
  #active: string[] = [];
  #primary = '';
  #loader: WorkspacePageLoader | null = null;
  readonly #pager: KeysetPager<WorkspacePageItem>;
  #loadMoreState: WorkspaceLoadMoreState = 'idle';

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
    return this.#loader !== null && this.#pager.hasOlder;
  }

  get workspaceLoadMoreState(): WorkspaceLoadMoreState {
    return this.#pager.state;
  }

  init(
    records: WorkspaceRecord[],
    active: string[],
    primary = '',
    loader: WorkspacePageLoader | null = null,
    nextCursor: string | null = null,
    knownWorkspaces: string[] | null = null,
  ): void {
    this.#loader = loader;
    this.#pager.reset(nextCursor);
    this.#records = records;
    // The full authorized id set stays separate from the bounded display
    // page: active/primary are server-validated against the full catalog and
    // must never be re-validated (and silently narrowed, then persisted to
    // cookies) against the first display page alone.
    this.#known = knownWorkspaces ?? records.map((record) => record.workspace);
    this.#active = this.#validActive(active);
    this.#primary = this.#validPrimary(primary);
    this.#syncCookies();
    this.changed();
  }

  loadMoreWorkspaces(): Promise<void> {
    if (this.#loader === null) return Promise.resolve();
    return this.#pager.loadNext((page) => {
      const known = new Set(this.#records.map((record) => record.workspace));
      const appended: WorkspaceRecord[] = [];
      for (const item of page.items) {
        if (!item.workspace || known.has(item.workspace)) continue;
        known.add(item.workspace);
        appended.push({
          workspace: item.workspace,
          displayName: item.displayName || item.workspace,
          embeddingModel: item.embeddingModel || '',
        });
      }
      this.#records = [...this.#records, ...appended];
    });
  }

  constructor() {
    super();
    this.#pager = new KeysetPager<WorkspacePageItem>(
      (cursor, signal) => this.#loader!(cursor, signal).then((page) => ({
        items: page.workspaces,
        nextCursor: page.nextCursor,
      })),
      () => this.changed(),
    );
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
    this.changed();
  }

  select(workspace: string): void {
    if (!workspace) return;
    this.#active = [workspace];
    this.#primary = workspace;
    this.#syncCookies();
    this.changed();
  }

  selectAll(): void {
    const knownIds = this.#known.length > 0
      ? [...this.#known]
      : this.#records.map((record) => record.workspace);
    if (knownIds.length === 0) return;
    this.#active = knownIds;
    this.#primary = this.#defaultWorkspace();
    this.#syncCookies();
    this.changed();
  }

  /** Restore an earlier composer scope after an editable submission failure. */
  restoreActive(workspaces: readonly string[]): void {
    this.#active = this.#validActive([...workspaces]);
    this.#primary = this.#validPrimary(this.#primary);
    this.#syncCookies();
    this.changed();
  }

  add(record: WorkspaceRecord): void {
    if (!this.#records.some((r) => r.workspace === record.workspace)) {
      this.#records.push(record);
    }
    if (!this.#known.includes(record.workspace)) this.#known.push(record.workspace);
    this.#active = [record.workspace];
    this.#primary = record.workspace;
    this.#syncCookies();
    this.changed();
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
    this.changed();
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
