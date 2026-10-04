// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** What the Settings browser tests share: the dialog mounted as the shell composes it, and a fetch
 *  that answers each Settings route by its method and path. */

import type {DlSettingsDialog} from '../ui/settings.ts';
import '../ui/settings.ts';
import type {DlToastRegion, ToastRequestDetail} from '../ui/toast.ts';
import '../ui/toast.ts';
import {defineDesignSystemElements} from '../design-system/index.ts';
import {waitFor} from './dom.ts';

defineDesignSystemElements();

export interface Mounted {
  shell: HTMLElement;
  settings: DlSettingsDialog;
  /** The shell's own toast region: where a notice goes while Settings is closed. */
  toast: DlToastRegion;
}

/** Mount Settings beside a toast region that answers `dl-toast-request`, as the app shell does. */
export function mountSettings(): Mounted {
  const shell = document.createElement('div');
  const toast = document.createElement('dl-toast-region') as DlToastRegion;
  toast.className = 'toast';
  shell.addEventListener('dl-toast-request', (event: CustomEvent<ToastRequestDetail>) => {
    if (event.detail.action) toast.showAction(event.detail.message, event.detail.action);
    else toast.show(event.detail.message, event.detail.duration);
  });
  const settings = document.createElement('dl-settings-dialog') as DlSettingsDialog;
  shell.append(toast, settings);
  document.body.appendChild(shell);
  return {shell, settings, toast};
}

/** Open Settings and wait until the dialog is showing. */
export async function openSettings(
  settings: DlSettingsDialog,
  page?: Parameters<DlSettingsDialog['open']>[1],
): Promise<HTMLDialogElement> {
  await settings.open(null, page);
  const dialog = settings.querySelector<HTMLDialogElement>('#settings-dialog')!;
  await waitFor(() => dialog.open);
  return dialog;
}

export interface Wired {
  method: string;
  path: string;
  search: string;
  body: unknown;
  headers: Headers;
}

export type Handler = (request: Wired) => Response | Promise<Response>;

export interface Wire {
  fetch: typeof window.fetch;
  /** Every request a handler answered, in order. */
  requests: Wired[];
  /** Every request no handler was written for: a test that expects none asserts this is empty. */
  unexpected: Wired[];
}

/** The reads every Settings page makes when it opens, so a test only writes the ones it is about. */
const DEFAULT_ROUTES: Record<string, Handler> = {
  'GET /web/api/connections/mcp': () => Response.json({revision: '0', connections: [], presets: []}),
  'GET /web/api/agent-accounts': () => Response.json(agentAccountsView()),
};

/** Answer fetches by `METHOD /path`; a route here replaces the default for the same key. */
export function wire(routes: Record<string, Handler> = {}): Wire {
  const handlers = {...DEFAULT_ROUTES, ...routes};
  const requests: Wired[] = [];
  const unexpected: Wired[] = [];
  const answer: typeof window.fetch = async (input, init) => {
    const url = new URL(String(input), window.location.origin);
    const wired: Wired = {
      method: init?.method ?? 'GET',
      path: url.pathname,
      search: url.search,
      body: typeof init?.body === 'string' ? JSON.parse(init.body) : init?.body,
      headers: new Headers(init?.headers),
    };
    const handler = handlers[`${wired.method} ${wired.path}`];
    if (!handler) {
      unexpected.push(wired);
      return new Response('unexpected request', {status: 500});
    }
    requests.push(wired);
    return await handler(wired);
  };
  return {fetch: answer, requests, unexpected};
}

/** A memory settings reply, as the wire sends it. */
export function memorySettings(enabled: boolean, activeCount: number | null = enabled ? 0 : null): Response {
  return Response.json({enabled, active_count: activeCount});
}

/** One page of stored memories, as the wire sends it. */
export function memoryPage(
  records: Array<{id: string; kind?: 'preference' | 'fact'; body: string}>,
  nextCursor: string | null = null,
): Response {
  return Response.json({
    memories: records.map(({id, kind, body}) => ({memory_id: id, kind: kind ?? 'preference', body})),
    next_cursor: nextCursor,
  });
}

export interface WireAccount {
  site: string;
  email: string | null;
  username: string | null;
  created_at: string;
  last_used_at: string | null;
}

/** The Agent Accounts view, as the wire sends it; every part can be replaced. */
export function agentAccountsView(
  accounts: WireAccount[] = [],
  registration: {allowed: boolean; enabled: boolean} = {allowed: true, enabled: true},
  available = true,
): {available: boolean; registration: {allowed: boolean; enabled: boolean}; accounts: WireAccount[]} {
  return {available, registration, accounts};
}

/** An account, with the sign-in the agent minted for it and when it was registered and last used. */
export function wireAccount(site: string, extra: Partial<WireAccount> = {}): WireAccount {
  return {
    site,
    email: `agent@${site}`,
    username: `agent-${site.split('.')[0]}`,
    created_at: '2026-01-02T03:04:05Z',
    last_used_at: null,
    ...extra,
  };
}

/** The ISO time a number of whole days before now, at the same hour, as the wire writes it. */
export function daysAgo(days: number): string {
  return new Date(Date.now() - days * 86_400_000).toISOString().replace(/\.\d{3}Z$/, 'Z');
}
