// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {sendKeys, sendMouse} from '@web/test-runner-commands';
import {createAppHandles} from '../stores/app-handles.ts';
import {buttonNamed, waitFor} from '../testing/dom.ts';
import {
  detailOpen,
  formOpen,
  listOpen,
  mountSidebar,
  rowButton,
  rowsShown,
  scheduleList,
  serving,
  wireSchedule,
} from '../testing/schedules.ts';
import './schedule-list.ts';
import type {DlScheduleList} from './schedule-list.ts';
import type {ScheduleIntentDetail} from './schedule-section.ts';

const originalFetch = window.fetch;

afterEach(() => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
  document.body.className = '';
});

const WEEKLY = wireSchedule('Weekly review');
const PAUSED = wireSchedule('Filing watch', {
  state: 'paused',
  pause_reason: 'its owner can no longer read any of its workspaces',
  next_fire_at: null,
  cron: '30 7 * * 1-5',
});
const DONE = wireSchedule('Launch', {state: 'completed', cron: null, at: '2026-12-01T07:00:00Z', next_fire_at: null});

/** A press the browser takes for a person's: the page's dialogs then each close on their own Escape. A script's
 *  `click()` carries no user activation, and the browser closes dialogs opened without one together. */
async function press(element: Element): Promise<void> {
  const box = element.getBoundingClientRect();
  await sendMouse({type: 'click', position: [Math.round(box.left + box.width / 2), Math.round(box.top + box.height / 2)]});
}

const text = (element: Element | null): string => element?.textContent?.replace(/\s+/g, ' ').trim() ?? '';

it('titles itself Scheduled and draws each Schedule as one button row: its name, then its cadence in words and when it runs next or that it is paused or done', async () => {
  serving({}, [WEEKLY, PAUSED, DONE]);
  const {sidebar} = await mountSidebar();
  const dialog = await listOpen(sidebar);
  const rows = await rowsShown(sidebar, 3);

  expect(dialog.querySelector('h2')!.textContent).to.equal('Scheduled');
  expect(rows.map((item) => item.dataset.scheduleId)).to.deep.equal(['schedule-weekly-review', 'schedule-filing-watch', 'schedule-launch']);
  expect(rows.map((item) => item.dataset.state)).to.deep.equal(['active', 'paused', 'completed']);
  // One real button a row, in the design system's navigation-list form, in its list card.
  const card = dialog.querySelector('ul')!;
  expect(card.classList.contains('dl-list-card')).to.equal(true);
  expect(card.getAttribute('role'), 'its list semantics survive a list-style of none').to.equal('list');
  for (const item of rows) {
    const button = item.querySelector('button')!;
    expect(item.tagName).to.equal('LI');
    expect(button.classList.contains('dl-nav-item') && button.classList.contains('dl-nav-item--list')).to.equal(true);
    expect(item.querySelectorAll('button')).to.have.length(1);
    // Named by both lines it shows (axe holds a name to the words on the row), so it carries no name of its own.
    expect(button.hasAttribute('aria-label') || button.hasAttribute('aria-labelledby')).to.equal(false);
  }
  expect(rows.map((item) => text(item.querySelector('[data-name]')))).to.deep.equal(['Weekly review', 'Filing watch', 'Launch']);
  expect(text(rows[0]!.querySelector('[data-meta]'))).to.match(/^Mondays at 08:00 · Next: Mon, Jan 5, 2099, 08:00$/);
  expect(text(rows[1]!.querySelector('[data-meta]'))).to.equal('Weekdays at 07:30 · Paused');
  expect(text(rows[2]!.querySelector('[data-meta]'))).to.match(/^Once on .* · Done$/);
  // The row opens the detail, which carries the actions: it has none of its own.
  expect(dialog.querySelector('[data-action], dl-menu')).to.equal(null);
  expect(text(dialog.querySelector('.dl-dialog-actions'))).to.equal('Close New schedule');
  expect(dialog.querySelector('[data-new]')!.classList.contains('dl-dialog-primary')).to.equal(true);
});

it('names the cadence alone for an active Schedule the server has no next slot for', async () => {
  serving({}, [wireSchedule('Weekly review', {next_fire_at: null})]);
  const {sidebar} = await mountSidebar();
  await listOpen(sidebar);
  const [row] = await rowsShown(sidebar, 1);

  expect(text(row!.querySelector('[data-meta]'))).to.equal('Mondays at 08:00');
});

it('says in one quiet sentence that nothing is scheduled, and still offers New schedule', async () => {
  serving({}, []);
  const {sidebar} = await mountSidebar();
  const dialog = await listOpen(sidebar);
  await waitFor(() => dialog.querySelector('[data-empty]') !== null);

  const note = dialog.querySelector('[data-empty]')!;
  expect([note.tagName, note.classList.contains('dl-note')]).to.deep.equal(['P', true]);
  expect(text(note)).to.equal('Nothing is scheduled yet. Ask the Agent in a chat, or add one here.');
  expect(dialog.querySelector('ul')).to.equal(null);
  expect(dialog.querySelector<HTMLButtonElement>('[data-new]')!.disabled).to.equal(false);
});

it('says it is loading while the list is being read, with no New schedule until it has been', async () => {
  let release: (response: Response) => void = () => {};
  serving({'GET /web/api/schedules': () => new Promise<Response>((resolve) => { release = resolve; })});
  const {sidebar} = await mountSidebar();
  const dialog = await listOpen(sidebar);

  expect(text(dialog.querySelector('[data-body]'))).to.equal('Loading schedules…');
  expect(dialog.querySelector('[data-empty]'), 'not "nothing is scheduled" before it is known').to.equal(null);
  expect(dialog.querySelector<HTMLButtonElement>('[data-new]')!.disabled).to.equal(true);
  release(scheduleList([WEEKLY]));

  await rowsShown(sidebar, 1);
  expect(dialog.querySelector<HTMLButtonElement>('[data-new]')!.disabled).to.equal(false);
});

it('says so when the list cannot be read and offers a retry, which reads it again', async () => {
  // The page reads it once, the dialog reads it again as it opens, and Retry is the third.
  let answers = 0;
  const api = serving({
    'GET /web/api/schedules': () => {
      answers += 1;
      return answers < 3 ? new Response('unavailable', {status: 503}) : scheduleList([WEEKLY]);
    },
  });
  const {sidebar, handles} = await mountSidebar();
  await waitFor(() => handles.schedules.listState === 'error');
  const dialog = await listOpen(sidebar);
  await waitFor(() => answers === 2 && dialog.querySelector('[data-failed]') !== null);

  const failed = dialog.querySelector('[data-failed]')!;
  expect([failed.tagName, failed.classList.contains('dl-note')]).to.deep.equal(['P', true]);
  expect(text(failed)).to.equal('Could not load schedules. Retry');
  expect(dialog.querySelector<HTMLButtonElement>('[data-new]')!.disabled).to.equal(true);
  const retry = buttonNamed(failed as HTMLElement, 'Retry')!;
  expect(retry.classList.contains('dl-text-action')).to.equal(true);
  retry.click();

  await rowsShown(sidebar, 1);
  expect(answers).to.equal(3);
  expect(api.unexpected).to.deep.equal([]);
});

it('reads the list itself when it opens before anything has', async () => {
  serving({}, [WEEKLY, PAUSED]);
  const list = document.createElement('dl-schedule-list') as DlScheduleList;
  list.handles = createAppHandles();
  document.body.append(list);
  expect(list.handles.schedules.listState).to.equal('idle');

  void list.open(null);

  await waitFor(() => list.querySelectorAll('[data-schedule-id]').length === 2);
  expect(list.handles.schedules.listState).to.equal('ready');
});

it('follows the store while it is open: a Schedule the Agent adds appears at once, one that is deleted goes, and the rows that stay are the same buttons', async () => {
  let current = [WEEKLY];
  serving({'GET /web/api/schedules': () => scheduleList(current)});
  const {sidebar, handles} = await mountSidebar();
  const dialog = await listOpen(sidebar);
  const [first] = await rowsShown(sidebar, 1);
  const button = first!.querySelector('button')!;
  button.focus();
  const count = () => text(sidebar.querySelector('#scheduled-btn'));
  await waitFor(() => count() === 'Scheduled 1');

  current = [WEEKLY, PAUSED];
  await handles.schedules.refresh();
  const rows = await rowsShown(sidebar, 2);
  expect(rows.map((item) => item.dataset.scheduleId)).to.deep.equal(['schedule-weekly-review', 'schedule-filing-watch']);
  expect(rows[0]!.querySelector('button'), 'the row that was there is still its own button').to.equal(button);
  expect(document.activeElement, 'and keeps focus').to.equal(button);
  await waitFor(() => count() === 'Scheduled 2');

  current = [PAUSED];
  await handles.schedules.refresh();
  const left = await rowsShown(sidebar, 1);
  expect(left[0]!.dataset.scheduleId).to.equal('schedule-filing-watch');

  current = [];
  await handles.schedules.refresh();
  await waitFor(() => dialog.querySelector('[data-empty]') !== null);
  expect(dialog.open).to.equal(true);
});

it('tells a reader who is not looking how many there are, and tells them again when the count changes under them', async () => {
  let current = [WEEKLY, PAUSED];
  serving({'GET /web/api/schedules': () => scheduleList(current)});
  const {sidebar, handles} = await mountSidebar();
  const dialog = await listOpen(sidebar);
  await rowsShown(sidebar, 2);
  const line = dialog.querySelector<HTMLElement>('[data-count-status]')!;

  expect([line.getAttribute('role'), line.classList.contains('dl-sr-only')]).to.deep.equal(['status', true]);
  expect(text(line)).to.equal('2 schedules');
  current = [WEEKLY, PAUSED, DONE];
  await handles.schedules.refresh();
  await waitFor(() => text(line) === '3 schedules');
  current = [DONE];
  await handles.schedules.refresh();
  await waitFor(() => text(line) === '1 schedule');
  current = [];
  await handles.schedules.refresh();
  await waitFor(() => text(line) === 'No schedules');
  expect(line, 'one line, updated in place, which is what a screen reader listens for').to.equal(dialog.querySelector('[data-count-status]'));
});

it('keeps focus in the list when the row that had it goes: on the first row left, or on New schedule when none is', async () => {
  serving({}, [WEEKLY, PAUSED]);
  const {sidebar, handles} = await mountSidebar();
  const dialog = await listOpen(sidebar);
  await rowsShown(sidebar, 2);

  rowButton(sidebar, 'schedule-filing-watch').focus();
  handles.schedules.drop('schedule-filing-watch');
  await waitFor(() => document.activeElement === rowButton(sidebar, 'schedule-weekly-review'));

  handles.schedules.drop('schedule-weekly-review');
  await waitFor(() => document.activeElement === dialog.querySelector('[data-new]'));
});

it('leaves focus alone when a row that is not the one it is on goes', async () => {
  serving({}, [WEEKLY, PAUSED]);
  const {sidebar, handles} = await mountSidebar();
  await listOpen(sidebar);
  await rowsShown(sidebar, 2);

  rowButton(sidebar, 'schedule-weekly-review').focus();
  handles.schedules.drop('schedule-filing-watch');
  await rowsShown(sidebar, 1);

  expect(document.activeElement).to.equal(rowButton(sidebar, 'schedule-weekly-review'));
});

it('reads the list again each time it opens, so a tab left open shows what the Agent has made since', async () => {
  let current = [WEEKLY];
  serving({'GET /web/api/schedules': () => scheduleList(current)});
  const {sidebar} = await mountSidebar();
  const dialog = await listOpen(sidebar);
  await rowsShown(sidebar, 1);
  dialog.querySelector<HTMLButtonElement>('[data-close]')!.click();
  await waitFor(() => !dialog.open && document.activeElement === sidebar.querySelector('#scheduled-btn'));

  current = [WEEKLY, PAUSED];
  await listOpen(sidebar);

  await rowsShown(sidebar, 2);
});

it('asks to open a Schedule when its row is pressed, and its detail opens on top with the list still open under it', async () => {
  serving({'GET /web/api/schedules/schedule-weekly-review/fires': () => Response.json({fires: [], next_cursor: null})}, [WEEKLY, PAUSED]);
  const {sidebar} = await mountSidebar();
  const list = sidebar.querySelector<HTMLDialogElement>('dl-schedule-list dialog')!;
  await press(sidebar.querySelector('#scheduled-btn')!);
  await waitFor(() => list.open);
  await rowsShown(sidebar, 2);
  const asks: ScheduleIntentDetail[] = [];
  sidebar.querySelector('dl-schedule-list')!.addEventListener('dl-schedule-intent', (event) => { asks.push(event.detail); });

  await press(rowButton(sidebar, 'schedule-weekly-review'));
  const detail = await detailOpen(sidebar);

  expect(asks).to.deep.equal([{intent: 'open', scheduleId: 'schedule-weekly-review', trigger: rowButton(sidebar, 'schedule-weekly-review')}]);
  expect(detail.querySelector('h2')!.textContent).to.equal('Weekly review');
  expect(list.open).to.equal(true);
  // Escape closes the dialog on top, not the one under it, and focus is back on the row.
  await sendKeys({press: 'Escape'});
  await waitFor(() => !detail.open && document.activeElement === rowButton(sidebar, 'schedule-weekly-review'));
  expect(list.open).to.equal(true);
  await sendKeys({press: 'Escape'});
  await waitFor(() => !list.open && document.activeElement === sidebar.querySelector('#scheduled-btn'));
});

it('opens the form for a new Schedule from New schedule, and focus comes back to it', async () => {
  serving({}, [WEEKLY]);
  const {sidebar} = await mountSidebar();
  const list = await listOpen(sidebar);
  await rowsShown(sidebar, 1);
  const add = list.querySelector<HTMLButtonElement>('[data-new]')!;
  await waitFor(() => !add.disabled);

  add.click();
  const form = await formOpen(sidebar);

  expect(form.querySelector('h2')!.textContent).to.equal('New schedule');
  expect(list.open).to.equal(true);
  form.querySelector<HTMLButtonElement>('button[type=button]:not([data-advanced]):not([data-zone-toggle])')!.click();
  await waitFor(() => !form.open && document.activeElement === add);
});
