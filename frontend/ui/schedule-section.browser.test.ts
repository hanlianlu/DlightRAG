// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {buttonNamed, waitFor} from '../testing/dom.ts';
import {
  asked,
  detailOpen,
  formOpen,
  mountSidebar,
  noticeOf,
  rowsShown,
  scheduleList,
  serving,
  wireFire,
  wireSchedule,
} from '../testing/schedules.ts';

const originalFetch = window.fetch;
const OPEN_KEY = 'dlightrag.schedule_section_open';

afterEach(() => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
  document.body.className = '';
  window.localStorage.removeItem(OPEN_KEY);
});

const WEEKLY = wireSchedule('Weekly review');
const PAUSED = wireSchedule('Filing watch', {
  state: 'paused',
  pause_reason: 'its owner can no longer read any of its workspaces',
  next_fire_at: null,
  cron: '30 7 * * 1-5',
});
const DONE = wireSchedule('Launch', {state: 'completed', cron: null, at: '2026-12-01T07:00:00Z', next_fire_at: null});

function toggle(root: ParentNode): HTMLButtonElement {
  return root.querySelector<HTMLButtonElement>('dl-schedule-section [data-toggle]')!;
}

function row(root: ParentNode, name: string): HTMLElement {
  return root.querySelector<HTMLElement>(`[data-schedule-id="schedule-${name}"]`)!;
}

function quick(root: ParentNode, name: string, action: string): HTMLButtonElement {
  return row(root, name).querySelector<HTMLButtonElement>(`[data-action="${action}"]`)!;
}

it('draws each Schedule as one compact row: its name, its cadence in words, and when it runs next or that it is paused', async () => {
  serving({}, [WEEKLY, PAUSED, DONE]);
  const {sidebar} = await mountSidebar();

  const rows = await rowsShown(sidebar, 3);

  expect(rows.map((item) => item.dataset.scheduleId)).to.deep.equal(['schedule-weekly-review', 'schedule-filing-watch', 'schedule-launch']);
  const weekly = rows[0]!.querySelector('button')!.textContent!.replace(/\s+/g, ' ').trim();
  expect(weekly).to.equal('Weekly review Mondays at 08:00');
  expect(rows[1]!.querySelector('button')!.textContent!.replace(/\s+/g, ' ').trim()).to.equal('Filing watch Weekdays at 07:30 · Paused');
  expect(rows[2]!.querySelector('button')!.textContent!.replace(/\s+/g, ' ').trim()).to.match(/^Launch Once on .* · Done$/);
  // A paused or finished Schedule offers no Run now or Resume-less pause, and a finished one only the menu.
  expect(quick(sidebar, 'weekly-review', 'run')).not.to.equal(null);
  expect(quick(sidebar, 'weekly-review', 'pause')).not.to.equal(null);
  expect(quick(sidebar, 'filing-watch', 'resume')).not.to.equal(null);
  expect(row(sidebar, 'launch').querySelector('[data-action]')).to.equal(null);
});

it('heads the section with a clock, its name and how many there are, and opens by itself once there is one to show', async () => {
  serving({}, [WEEKLY, PAUSED]);
  const {sidebar} = await mountSidebar();
  await rowsShown(sidebar, 2);

  const header = toggle(sidebar);
  expect(header.textContent!.replace(/\s+/g, ' ').trim()).to.equal('Scheduled 2');
  expect(header.getAttribute('aria-expanded')).to.equal('true');
  expect(header.querySelector('.dl-icon')).not.to.equal(null);
  expect(sidebar.querySelector<HTMLButtonElement>('[data-new]')!.getAttribute('aria-label')).to.equal('New schedule');
});

it('with none the header stands alone, closed, with its plus', async () => {
  serving({}, []);
  const {sidebar} = await mountSidebar();
  const add = sidebar.querySelector<HTMLButtonElement>('[data-new]')!;
  await waitFor(() => !add.disabled);

  expect(toggle(sidebar).getAttribute('aria-expanded')).to.equal('false');
  expect(sidebar.querySelector('dl-schedule-section [data-count]')).to.equal(null);
  expect(sidebar.querySelector('dl-schedule-section [data-schedule-id]')).to.equal(null);
  expect(sidebar.querySelector<HTMLElement>('#scheduled-rows')!.hidden).to.equal(true);
});

it('keeps whether it is open for this browser, whatever there is to show', async () => {
  serving({}, [WEEKLY]);
  const first = await mountSidebar();
  await rowsShown(first.sidebar, 1);

  toggle(first.sidebar).click();
  await waitFor(() => toggle(first.sidebar).getAttribute('aria-expanded') === 'false');
  expect(window.localStorage.getItem(OPEN_KEY)).to.equal('false');
  document.body.replaceChildren();

  const again = await mountSidebar();
  await waitFor(() => toggle(again.sidebar).textContent!.includes('1'));
  expect(toggle(again.sidebar).getAttribute('aria-expanded')).to.equal('false');
  expect(again.sidebar.querySelector('dl-schedule-section [data-schedule-id]')).to.equal(null);
});

it('says so when the list cannot be read, offers a retry, and has no plus until it has been read', async () => {
  let answers = 0;
  serving({
    'GET /web/api/schedules': () => {
      answers += 1;
      return answers === 1 ? new Response('unavailable', {status: 503}) : scheduleList([WEEKLY]);
    },
  });
  const {sidebar} = await mountSidebar();
  await waitFor(() => answers === 1);
  await sidebar.querySelector('dl-schedule-section')!.updateComplete;

  expect(sidebar.querySelector<HTMLButtonElement>('[data-new]')!.disabled).to.equal(true);
  toggle(sidebar).click();
  await waitFor(() => sidebar.querySelector('dl-schedule-section [role=status]') !== null);
  expect(sidebar.querySelector('dl-schedule-section [role=status]')!.textContent).to.contain('Could not load schedules.');
  buttonNamed(sidebar.querySelector('dl-schedule-section')!, 'Retry')!.click();

  await rowsShown(sidebar, 1);
  expect(sidebar.querySelector<HTMLButtonElement>('[data-new]')!.disabled).to.equal(false);
});

it('runs a Schedule now from its row, and says why when the Fire started nothing', async () => {
  const fire = (state: string, reason: string | null) => Response.json(wireFire('f1', '2099-01-01T00:00:00.000000Z', state, {reason, manual: true}));
  let replies = [fire('started', null), fire('skipped_busy', 'its previous run is still working')];
  const api = serving({'POST /web/api/schedules/schedule-weekly-review/run': () => replies.shift()!}, [WEEKLY]);
  const {sidebar, shell} = await mountSidebar();
  await rowsShown(sidebar, 1);

  quick(sidebar, 'weekly-review', 'run').click();
  await waitFor(() => noticeOf(shell) === 'Started Weekly review.');
  quick(sidebar, 'weekly-review', 'run').click();
  await waitFor(() => noticeOf(shell, 'error').includes('its previous run is still working'));

  expect(asked(api, 'POST', '/web/api/schedules/schedule-weekly-review/run')).to.have.length(2);
  // A run can pause the Schedule, so the list is read again after it.
  expect(asked(api, 'GET', '/web/api/schedules')).to.have.length(3);
  replies = [];
});

it('pauses and resumes a Schedule from its row and shows the row the server answered', async () => {
  const api = serving({
    'POST /web/api/schedules/schedule-weekly-review/pause': () => Response.json({
      ...WEEKLY, state: 'paused', pause_reason: null, next_fire_at: null,
    }),
    'POST /web/api/schedules/schedule-weekly-review/resume': () => Response.json(WEEKLY),
  }, [WEEKLY]);
  const {sidebar} = await mountSidebar();
  await rowsShown(sidebar, 1);

  quick(sidebar, 'weekly-review', 'pause').click();
  await waitFor(() => row(sidebar, 'weekly-review').dataset.state === 'paused');
  expect(row(sidebar, 'weekly-review').textContent).to.contain('Paused');
  expect(row(sidebar, 'weekly-review').textContent).not.to.contain('Next:');
  quick(sidebar, 'weekly-review', 'resume').click();
  await waitFor(() => row(sidebar, 'weekly-review').dataset.state === 'active');

  expect(api.requests.filter((request) => request.method === 'POST').map((request) => request.path)).to.deep.equal([
    '/web/api/schedules/schedule-weekly-review/pause',
    '/web/api/schedules/schedule-weekly-review/resume',
  ]);
});

it('drops a Schedule another tab deleted and says so', async () => {
  serving({
    'POST /web/api/schedules/schedule-weekly-review/pause': () => Response.json(
      {detail: 'Schedule not found', error_type: 'not_found'},
      {status: 404},
    ),
  }, [WEEKLY, PAUSED]);
  const {sidebar, shell} = await mountSidebar();
  await rowsShown(sidebar, 2);

  quick(sidebar, 'weekly-review', 'pause').click();

  await rowsShown(sidebar, 1);
  expect(noticeOf(shell, 'error')).to.contain('Weekly review no longer exists, so it was removed from the list.');
});

it('opens a row\'s menu to edit or delete it, and asks before deleting: nothing is sent on Cancel, one DELETE on confirm', async () => {
  const api = serving({'DELETE /web/api/schedules/schedule-weekly-review': () => new Response(null, {status: 204})}, [WEEKLY, PAUSED]);
  const {sidebar, shell} = await mountSidebar();
  await rowsShown(sidebar, 2);
  const menuButton = row(sidebar, 'weekly-review').querySelector<HTMLButtonElement>('[data-menu-button]')!;
  const dialog = sidebar.querySelector<HTMLDialogElement>('#schedule-delete-dialog')!;

  menuButton.click();
  await waitFor(() => row(sidebar, 'weekly-review').querySelector('dl-menu') !== null);
  expect([...row(sidebar, 'weekly-review').querySelectorAll('[role=menuitem]')].map((item) => item.textContent!.trim())).to.deep.equal(['Edit', 'Delete']);
  row(sidebar, 'weekly-review').querySelector<HTMLButtonElement>('[data-menu="delete"]')!.click();
  await waitFor(() => dialog.open);
  expect(dialog.textContent).to.contain('Delete Weekly review?');
  expect(dialog.textContent).to.contain('The conversations its runs wrote stay.');
  dialog.querySelector<HTMLButtonElement>('button[value=cancel]')!.click();
  await waitFor(() => !dialog.open && document.activeElement === menuButton);
  expect(asked(api, 'DELETE', '/web/api/schedules/schedule-weekly-review')).to.have.length(0);

  menuButton.click();
  await waitFor(() => row(sidebar, 'weekly-review').querySelector('dl-menu') !== null);
  row(sidebar, 'weekly-review').querySelector<HTMLButtonElement>('[data-menu="delete"]')!.click();
  await waitFor(() => dialog.open);
  dialog.querySelector<HTMLButtonElement>('button[value=delete]')!.click();
  await rowsShown(sidebar, 1);

  expect(asked(api, 'DELETE', '/web/api/schedules/schedule-weekly-review')).to.have.length(1);
  expect(noticeOf(shell)).to.contain('Deleted Weekly review.');
  expect(toggle(sidebar).textContent).to.contain('1');
  // The row that asked is gone, so focus goes to the section's own header.
  await waitFor(() => document.activeElement === toggle(sidebar));

  row(sidebar, 'filing-watch').querySelector<HTMLButtonElement>('[data-menu-button]')!.click();
  await waitFor(() => row(sidebar, 'filing-watch').querySelector('dl-menu') !== null);
  row(sidebar, 'filing-watch').querySelector<HTMLButtonElement>('[data-menu="edit"]')!.click();
  const form = await formOpen(sidebar);
  expect(form.querySelector<HTMLInputElement>('[data-field="name"]')!.value).to.equal('Filing watch');
});

it('opens the detail of a Schedule from its row, and the plus opens the form for a new one', async () => {
  serving({'GET /web/api/schedules/schedule-weekly-review/fires': () => Response.json({fires: [], next_cursor: null})}, [WEEKLY]);
  const {sidebar} = await mountSidebar();
  await rowsShown(sidebar, 1);

  row(sidebar, 'weekly-review').querySelector<HTMLButtonElement>('button.conversation-select')!.click();
  const detail = await detailOpen(sidebar);
  expect(detail.querySelector('h2')!.textContent).to.equal('Weekly review');
  detail.querySelector<HTMLButtonElement>('button:not([data-action])')!.click();
  await waitFor(() => !detail.open && document.activeElement === row(sidebar, 'weekly-review').querySelector('button.conversation-select'));

  sidebar.querySelector<HTMLButtonElement>('[data-new]')!.click();
  const form = await formOpen(sidebar);
  expect(form.querySelector('h2')!.textContent).to.equal('New schedule');
});

it('leaves the drawer open when an Escape is for the menu that is open', async () => {
  serving({}, [WEEKLY]);
  const {sidebar} = await mountSidebar();
  await rowsShown(sidebar, 1);
  const menuButton = row(sidebar, 'weekly-review').querySelector<HTMLButtonElement>('[data-menu-button]')!;

  menuButton.click();
  await waitFor(() => row(sidebar, 'weekly-review').querySelector('dl-menu') !== null);
  document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));

  expect(sidebar.querySelector('dl-schedule-section')!.menuOpen).to.equal(true);
  expect(sidebar.drawerOpen).to.equal(true);
});
