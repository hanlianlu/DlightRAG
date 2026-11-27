// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {waitFor} from '../testing/dom.ts';
import {listOpen, mountSidebar, scheduleList, serving, wireSchedule} from '../testing/schedules.ts';
import type {ScheduleIntentDetail} from './schedule-section.ts';

const originalFetch = window.fetch;

afterEach(() => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
  document.body.className = '';
});

const WEEKLY = wireSchedule('Weekly review');
const PAUSED = wireSchedule('Filing watch', {state: 'paused', next_fire_at: null, cron: '30 7 * * 1-5'});

function row(root: ParentNode): HTMLButtonElement {
  return root.querySelector<HTMLButtonElement>('dl-schedule-section #scheduled-btn')!;
}

it('is one nav row: a clock, its name, and how many Schedules there are, with none of the list in the navigation', async () => {
  serving({}, [WEEKLY, PAUSED]);
  const {sidebar} = await mountSidebar();
  await waitFor(() => row(sidebar).querySelector('[data-count]') !== null);

  const scheduled = row(sidebar);
  expect(scheduled.classList.contains('dl-nav-item')).to.equal(true);
  expect(scheduled.textContent!.replace(/\s+/g, ' ').trim()).to.equal('Scheduled 2');
  expect(scheduled.querySelector('[data-count]')!.classList.contains('dl-nav-item-status')).to.equal(true);
  expect(scheduled.querySelector('.dl-icon'), 'a drawn clock').not.to.equal(null);
  // Nothing else: no toggle, no plus, no rows, no actions.
  const section = sidebar.querySelector('dl-schedule-section')!;
  expect(section.querySelectorAll('button')).to.have.length(1);
  expect(scheduled.hasAttribute('aria-expanded')).to.equal(false);
  expect(section.querySelector('[data-schedule-id], [data-new], dl-menu')).to.equal(null);
});

it('shows no count while there is none to count: no Schedule, a list still being read, or one that could not be read', async () => {
  serving({}, []);
  const none = await mountSidebar();
  await waitFor(() => none.handles.schedules.listState === 'ready');
  await none.sidebar.querySelector('dl-schedule-section')!.updateComplete;
  expect(row(none.sidebar).querySelector('[data-count]')).to.equal(null);
  expect(row(none.sidebar).textContent!.trim()).to.equal('Scheduled');
  document.body.replaceChildren();

  serving({'GET /web/api/schedules': () => new Promise<Response>(() => {})});
  const reading = await mountSidebar();
  await waitFor(() => reading.handles.schedules.listState === 'loading');
  await reading.sidebar.querySelector('dl-schedule-section')!.updateComplete;
  expect(row(reading.sidebar).querySelector('[data-count]')).to.equal(null);
  document.body.replaceChildren();

  serving({'GET /web/api/schedules': () => new Response('unavailable', {status: 503})});
  const failed = await mountSidebar();
  await waitFor(() => failed.handles.schedules.listState === 'error');
  await failed.sidebar.querySelector('dl-schedule-section')!.updateComplete;
  expect(row(failed.sidebar).querySelector('[data-count]')).to.equal(null);
});

it('follows the count as Schedules come and go', async () => {
  serving({}, [WEEKLY]);
  const {sidebar, handles} = await mountSidebar();
  await waitFor(() => row(sidebar).querySelector('[data-count]')?.textContent === '1');

  handles.schedules.drop('schedule-weekly-review');
  await waitFor(() => row(sidebar).querySelector('[data-count]') === null);
});

it('asks for the list when it is pressed, which opens the Schedules dialog, and focus comes back to the row when it closes', async () => {
  serving({}, [WEEKLY]);
  const {sidebar} = await mountSidebar();
  await waitFor(() => row(sidebar).querySelector('[data-count]') !== null);
  const asks: ScheduleIntentDetail[] = [];
  sidebar.querySelector('dl-schedule-section')!.addEventListener('dl-schedule-intent', (event) => { asks.push(event.detail); });

  const dialog = await listOpen(sidebar);

  expect(asks).to.deep.equal([{intent: 'list', scheduleId: null, trigger: row(sidebar)}]);
  expect(dialog.querySelector('h2')!.textContent).to.equal('Scheduled');
  dialog.querySelector<HTMLButtonElement>('[data-close]')!.click();
  await waitFor(() => !dialog.open && document.activeElement === row(sidebar));
});

it('reads the list once for the page, and again each time the dialog opens, so a tab left open shows what is true now', async () => {
  let reads = 0;
  serving({'GET /web/api/schedules': () => { reads += 1; return scheduleList([WEEKLY]); }});
  const {sidebar, handles} = await mountSidebar();
  await waitFor(() => handles.schedules.listState === 'ready');
  expect(reads).to.equal(1);

  const dialog = await listOpen(sidebar);
  await waitFor(() => reads === 2);
  dialog.querySelector<HTMLButtonElement>('[data-close]')!.click();
  await waitFor(() => !dialog.open && document.activeElement === row(sidebar));
  await listOpen(sidebar);
  await waitFor(() => reads === 3);
});
