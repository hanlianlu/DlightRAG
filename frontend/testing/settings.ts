// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** What the Settings browser tests share: the dialog mounted as the shell composes it, and a fetch
 *  that answers each Settings route by its method and path. */

import type {DlSettingsDialog} from '../ui/settings.ts';
import '../ui/settings.ts';
import {type DlToastRegion, type ToastRequestDetail, toastRegionOf} from '../ui/toast.ts';
import '../ui/toast.ts';
import {defineDesignSystemElements} from '../design-system/index.ts';
import {waitFor} from './dom.ts';

defineDesignSystemElements();

interface Mounted {
  shell: HTMLElement;
  settings: DlSettingsDialog;
  /** The shell's own toast regions, one for each tone: where a notice goes while Settings is closed. */
  toast: DlToastRegion;
  errorToast: DlToastRegion;
}

/** Mount Settings beside the shell's two toast regions, which answer `dl-toast-request` by its tone as the
 *  app shell does: a receipt to the polite region, an error to the alert region. */
export function mountSettings(): Mounted {
  const shell = document.createElement('div');
  const toast = document.createElement('dl-toast-region') as DlToastRegion;
  toast.className = 'toast';
  const errorToast = document.createElement('dl-toast-region') as DlToastRegion;
  errorToast.className = 'toast';
  errorToast.tone = 'error';
  shell.addEventListener('dl-toast-request', (event: CustomEvent<ToastRequestDetail>) => {
    toastRegionOf(shell, event.detail.tone)?.show(event.detail);
  });
  const settings = document.createElement('dl-settings-dialog') as DlSettingsDialog;
  shell.append(toast, errorToast, settings);
  document.body.appendChild(shell);
  return {shell, settings, toast, errorToast};
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

/** Settings has closed and said so: the dialog is shut and the page behind it is its own again.
 *  A dialog reports a close in a task of its own, so a test that ends on `!dialog.open` hands the
 *  next test a report that is still on its way. */
export function settingsClosed(): boolean {
  return !document.querySelector('#settings-dialog[open]') && !document.body.classList.contains('settings-open');
}

interface Wired {
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
  'GET /web/api/skills/mine': () => Response.json({skills: [], limit: 20}),
  'GET /web/api/privacy/settings': () => privacySettings(false),
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

/** Noon on a local calendar day, written as the server writes a moment: UTC, with microseconds. It is
 *  built from the local calendar, so the day it names reads the same wherever the tests run. */
export function wireMoment(year: number, month: number, day: number): string {
  return new Date(year, month - 1, day, 12, 0).toISOString().replace(/\.\d{3}Z$/, '.123456Z');
}

/** The owner's Privacy setting, as the wire sends it: off, or on since a moment, by default one in an
 *  earlier year, so it always reads with its year. A test that needs a day of its own names it. */
export function privacySettings(
  contribute: boolean,
  since: string | null = contribute ? wireMoment(2025, 3, 4) : null,
): Response {
  return Response.json({contribute, since});
}

/** One page of stored memories, as the wire sends it. */
export function memoryPage(
  records: Array<{
    id: string;
    kind?: 'preference' | 'fact' | 'experience';
    body: string;
  }>,
  nextCursor: string | null = null,
): Response {
  return Response.json({
    memories: records.map(({id, kind, body}) => ({
      memory_id: id,
      kind: kind ?? 'preference',
      body,
    })),
    next_cursor: nextCursor,
  });
}

interface WireSkill {
  name: string;
  description: string;
  enabled: boolean;
}

/** One of the owner's own Skills, as the wire sends it: on, with a description that names it. */
export function wireSkill(name: string, extra: Partial<WireSkill> = {}): WireSkill {
  return {name, description: `Instructions for ${name}.`, enabled: true, ...extra};
}

/** The owner's own Skills, as the wire sends them. */
export function ownerSkills(skills: WireSkill[] = [], limit = 20): Response {
  return Response.json({skills, limit});
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

/** The reads a fully populated Settings makes: two Connections, three accounts, five memories, two Skills, Privacy on. */
export function populated(): ReturnType<typeof wire> {
  return wire({
    'GET /web/api/connections/mcp': () => Response.json({
      revision: '1',
      presets: [],
      connections: [
        {connection_id: 'a', label: 'Notion', endpoint: 'https://a.example/mcp', enabled: true,
          activation_epoch: 1, generation: 1, authentication: 'oauth', endpoint_headers: [],
          credential_header: null, credential_prefix: null,
          authorization_status: null, status: 'ready',
          tool_count: 42, declaration_tokens: 54_000},
        {connection_id: 'b', label: 'Hugging Face', endpoint: 'https://b.example/mcp', enabled: false,
          activation_epoch: 1, generation: 1, authentication: 'none', endpoint_headers: [],
          credential_header: null, credential_prefix: null,
          authorization_status: null, status: 'disabled',
          // A disabled Connection's declaration cost is hypothetical, and still projected.
          tool_count: 8, declaration_tokens: 9_400},
      ],
    }),
    'GET /web/api/agent-accounts': () => Response.json(agentAccountsView([
      wireAccount('discourse.org'), wireAccount('huggingface.co'), wireAccount('ycombinator.com'),
    ])),
    'GET /web/api/memory/settings': () => memorySettings(true, 5),
    'GET /web/api/memory': () => memoryPage([{id: 'one', body: 'Use concise answers'}]),
    'GET /web/api/skills/mine': () => ownerSkills([wireSkill('csv-report'), wireSkill('old-style-guide', {enabled: false})]),
    'GET /web/api/privacy/settings': () => privacySettings(true),
  });
}

/** The navigation rows, in order, as a reader meets them. */
export function rows(settings: DlSettingsDialog): HTMLButtonElement[] {
  return [...settings.querySelectorAll<HTMLButtonElement>('nav .dl-nav-item')];
}

export function row(settings: DlSettingsDialog, section: string): HTMLButtonElement {
  return settings.querySelector<HTMLButtonElement>(`nav .dl-nav-item[data-section="${section}"]`)!;
}

/** What a row is called: the text its aria-labelledby points at. */
export function nameOf(element: Element): string {
  const id = element.getAttribute('aria-labelledby')!;
  return element.ownerDocument.getElementById(id)!.textContent!.trim();
}

export function pageElement(settings: DlSettingsDialog, tag: string): HTMLElement {
  return settings.querySelector<HTMLElement>(tag)!;
}
