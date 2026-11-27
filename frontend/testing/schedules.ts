// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** What the Schedule browser tests share: the wire's Schedules and Fires, the sidebar mounted as the shell
 *  composes it, beside the shell's own toast regions, and the way into the Schedules dialog from its Scheduled row. */

import {defineDesignSystemElements} from '../design-system/index.ts';
import {type AppHandles, createAppHandles} from '../stores/app-handles.ts';
import type {DlConversationSidebar} from '../ui/conversation-sidebar.ts';
import '../ui/conversation-sidebar.ts';
import {type DlToastRegion, type ToastRequestDetail, type ToastTone, toastRegionOf} from '../ui/toast.ts';
import '../ui/toast.ts';
import {waitFor} from './dom.ts';
import {type Handler, type Wire, wire} from './settings.ts';

defineDesignSystemElements();

export interface WireSchedule {
  schedule_id: string;
  name: string;
  prompt: string;
  mode: 'research' | 'fast';
  workspaces: string[];
  target: 'new_conversation' | 'same_conversation';
  conversation_id: string | null;
  zone: string;
  cron: string | null;
  at: string | null;
  max_tokens_per_day: number;
  goal_max_rounds: number | null;
  state: 'active' | 'paused' | 'completed';
  pause_reason: string | null;
  changed_by: 'owner' | 'agent';
  next_fire_at: string | null;
}

/** One of the owner's Schedules, as the wire sends it: active on a weekly recurrence in Stockholm. */
export function wireSchedule(name: string, extra: Partial<WireSchedule> = {}): WireSchedule {
  return {
    schedule_id: `schedule-${name.toLowerCase().replaceAll(' ', '-')}`,
    name,
    prompt: `Run ${name}.`,
    mode: 'research',
    workspaces: ['default'],
    target: 'new_conversation',
    conversation_id: null,
    zone: 'Europe/Stockholm',
    cron: '0 8 * * 1',
    at: null,
    max_tokens_per_day: 2_000_000,
    goal_max_rounds: null,
    state: 'active',
    pause_reason: null,
    changed_by: 'owner',
    next_fire_at: '2099-01-05T07:00:00.000000Z',
    ...extra,
  };
}

/** The owner's Schedules and the ceiling a new one starts with, as the wire sends them. */
export function scheduleList(schedules: WireSchedule[] = [], ceiling = 2_000_000): Response {
  return Response.json({schedules, default_max_tokens_per_day: ceiling});
}

/** One Fire of a Schedule, as the wire sends it. */
export function wireFire(id: string, slot: string, state: string, extra: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    fire_id: id, schedule_id: 'schedule-weekly-review', slot, state, reason: null, run_id: null,
    conversation_id: null, missed_count: null, last_slot: null, manual: false, last_error: null, ...extra,
  };
}

export interface MountedSidebar {
  shell: HTMLElement;
  sidebar: DlConversationSidebar;
  handles: AppHandles;
}

/** Mount the sidebar, ready and open, in a shell whose toast regions answer `dl-toast-request` as the app's do. The
 *  sidebar carries a bag of its own, so nothing a test does reaches the page's. */
export async function mountSidebar(parent: HTMLElement = document.body): Promise<MountedSidebar> {
  const shell = document.createElement('div');
  const toast = document.createElement('dl-toast-region') as DlToastRegion;
  toast.className = 'toast';
  const errorToast = document.createElement('dl-toast-region') as DlToastRegion;
  errorToast.className = 'toast';
  errorToast.tone = 'error';
  shell.addEventListener('dl-toast-request', (event: CustomEvent<ToastRequestDetail>) => {
    toastRegionOf(shell, event.detail.tone)?.show(event.detail);
  });
  const handles = createAppHandles();
  const sidebar = document.createElement('dl-conversation-sidebar') as DlConversationSidebar;
  sidebar.handles = handles;
  sidebar.enabled = true;
  shell.append(toast, errorToast, sidebar);
  parent.append(shell);
  await sidebar.open();
  return {shell, sidebar, handles};
}

/** A fetch that answers the Schedule routes, and the conversation list as empty, so a test names only what it is about. */
export function serving(routes: Record<string, Handler> = {}, schedules: WireSchedule[] = [], ceiling = 1_500_000): Wire {
  const api = wire({
    'GET /web/api/conversations': () => Response.json({items: [], next_cursor: null}),
    'GET /web/api/schedules': () => scheduleList(schedules, ceiling),
    ...routes,
  });
  window.fetch = api.fetch;
  return api;
}

/** Open the Schedules dialog from the sidebar's Scheduled row, and return it. */
export async function listOpen(root: ParentNode): Promise<HTMLDialogElement> {
  const dialog = root.querySelector<HTMLDialogElement>('dl-schedule-list dialog')!;
  if (!dialog.open) root.querySelector<HTMLButtonElement>('#scheduled-btn')!.click();
  await waitFor(() => dialog.open);
  return dialog;
}

/** Wait until the open Schedules dialog has drawn `count` rows. */
export async function rowsShown(root: ParentNode, count: number): Promise<HTMLElement[]> {
  await waitFor(() => root.querySelectorAll('dl-schedule-list [data-schedule-id]').length === count);
  return [...root.querySelectorAll<HTMLElement>('dl-schedule-list [data-schedule-id]')];
}

/** The button of a Schedule's row in the open Schedules dialog; `id` is the Schedule's, as `schedule-weekly-review`. */
export function rowButton(root: ParentNode, id: string): HTMLButtonElement {
  return root.querySelector<HTMLButtonElement>(`dl-schedule-list [data-schedule-id="${id}"] button`)!;
}

/** Open the form for a new Schedule through the dialog's New schedule, once the list has been read. */
export async function newFormOpen(root: ParentNode): Promise<HTMLDialogElement> {
  await listOpen(root);
  const add = root.querySelector<HTMLButtonElement>('dl-schedule-list [data-new]')!;
  await waitFor(() => !add.disabled);
  add.click();
  return await formOpen(root);
}

/** Open the form for a Schedule through its row and its detail's Edit. */
export async function editFormOpen(root: ParentNode, id: string): Promise<HTMLDialogElement> {
  await listOpen(root);
  await waitFor(() => root.querySelector(`dl-schedule-list [data-schedule-id="${id}"]`) !== null);
  rowButton(root, id).click();
  const detail = await detailOpen(root);
  detail.querySelector<HTMLButtonElement>('[data-action="edit"]')!.click();
  return await formOpen(root);
}

/** What a Schedule dialog or the section has said through the shell's toast region of a tone. */
export function noticeOf(shell: HTMLElement, tone: ToastTone = 'receipt'): string {
  return toastRegionOf(shell, tone)?.textContent?.trim() ?? '';
}

/** The requests the fake wire answered for one route, in order. */
export function asked(api: Wire, method: string, path: string): Wire['requests'] {
  return api.requests.filter((request) => request.method === method && request.path === path);
}

/** Type into a field the way a reader does: its value, then the event the page listens to. */
export function type(element: HTMLInputElement | HTMLTextAreaElement, value: string): void {
  element.value = value;
  element.dispatchEvent(new Event('input', {bubbles: true}));
}

/** Choose one radio of a group by its value, as a click does; the page listens to the change it makes. */
export function choose(group: ParentNode, value: string): void {
  group.querySelector<HTMLInputElement>(`input[type="radio"][value="${value}"]`)!.click();
}

/** The value of the radio a group has chosen. */
export function chosen(group: ParentNode): string | undefined {
  return group.querySelector<HTMLInputElement>('input[type="radio"]:checked')?.value;
}

/** A control of the open Schedule form, by its `data-field`. */
export function field<T extends HTMLElement = HTMLInputElement>(root: ParentNode, name: string): T {
  return root.querySelector<T>(`dl-schedule-form [data-field="${name}"]`)!;
}

/** Wait for the form dialog to be showing, and return it. */
export async function formOpen(root: ParentNode): Promise<HTMLDialogElement> {
  const dialog = root.querySelector<HTMLDialogElement>('dl-schedule-form dialog')!;
  await waitFor(() => dialog.open && field(root, "name") !== null);
  return dialog;
}

/** Wait for the detail dialog to be showing a Schedule, and return it. */
export async function detailOpen(root: ParentNode): Promise<HTMLDialogElement> {
  const dialog = root.querySelector<HTMLDialogElement>('dl-schedule-detail dialog')!;
  await waitFor(() => dialog.open && dialog.querySelector('[data-detail]') !== null);
  return dialog;
}

/** The workspaces the reader has, of which they have chosen `active`. */
export function useWorkspaces(handles: AppHandles, names: readonly string[], active: readonly string[]): void {
  handles.workspaces.init(
    names.map((workspace) => ({workspace, displayName: workspace[0]!.toUpperCase() + workspace.slice(1), embeddingModel: 'e', changes: []})),
    [...active],
    active[0],
  );
}
