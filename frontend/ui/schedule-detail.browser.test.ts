// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {getLocale} from '../i18n/locale.ts';
import {dateTime} from '../lib/date-format.ts';
import {conversationRoute, newChatRoute} from '../lib/router.ts';
import {waitFor} from '../testing/dom.ts';
import {
  asked,
  detailOpen,
  formOpen,
  mountSidebar,
  noticeOf,
  rowsShown,
  serving,
  wireFire,
  wireSchedule,
} from '../testing/schedules.ts';
import {webRouter} from './router.ts';

const originalFetch = window.fetch;

afterEach(async () => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
  document.body.className = '';
  window.localStorage.removeItem('dlightrag.schedule_section_open');
  await webRouter.navigate(newChatRoute(), {replace: true, notify: false, bypassGuard: true});
});

const WEEKLY = wireSchedule('Weekly review', {prompt: 'Summarize the week.'});
const FIRES = '/web/api/schedules/schedule-weekly-review/fires';

function firesWire(): Record<string, (request: {search: string}) => Response> {
  return {
    [`GET ${FIRES}`]: (request) => Response.json(
      new URLSearchParams(request.search).get('cursor') === null
        ? {
          fires: [
            wireFire('f3', '2099-01-12T07:00:00.000000Z', 'succeeded', {conversation_id: 'conversation-3'}),
            wireFire('f2', '2099-01-11T07:00:00.000000Z', 'skipped_busy', {reason: 'its previous run is still working'}),
            wireFire('f4', '2099-01-10T07:00:00.000000Z', 'failed', {reason: 'the model refused', conversation_id: 'conversation-4'}),
          ],
          next_cursor: '2099-01-10T07:00:00.000000Z',
        }
        : {fires: [wireFire('f1', '2099-01-05T07:00:00.000000Z', 'missed', {missed_count: 3})], next_cursor: null},
    ),
  };
}

async function openDetail(name = 'weekly-review') {
  const mounted = await mountSidebar();
  await rowsShown(mounted.sidebar, 1);
  mounted.sidebar.querySelector<HTMLButtonElement>(`[data-schedule-id="schedule-${name}"] button.conversation-select`)!.click();
  const dialog = await detailOpen(mounted.sidebar);
  return {...mounted, dialog};
}

function action(dialog: HTMLElement, name: string): HTMLButtonElement | null {
  return dialog.querySelector<HTMLButtonElement>(`[data-action="${name}"]`);
}

it('says what the Schedule is, when it runs next, what it asks, and what it can be told to do', async () => {
  serving(firesWire(), [WEEKLY]);
  const {dialog} = await openDetail();

  expect(dialog.querySelector('h2')!.textContent).to.equal('Weekly review');
  expect(dialog.querySelector('[data-cadence]')!.textContent).to.equal('Mondays at 08:00 · Europe/Stockholm');
  expect(dialog.querySelector('[data-when]')!.textContent!.trim()).to.match(/^Next run: Mon, Jan 5, 2099, 08:00/);
  expect(dialog.querySelector('[data-prompt]')!.textContent).to.equal('Summarize the week.');
  expect(dialog.querySelector('[data-expand]')).to.equal(null);
  expect(['run', 'pause', 'edit', 'delete'].map((name) => action(dialog, name)?.textContent?.trim())).to.deep.equal(['Run now', 'Pause', 'Edit', 'Delete']);
});

it('folds a long prompt to a few lines until it is asked for in full', async () => {
  const long = Array.from({length: 8}, (_, index) => `Line ${index + 1} of what to ask.`).join('\n');
  serving(firesWire(), [wireSchedule('Weekly review', {prompt: long})]);
  const {dialog} = await openDetail();
  const prompt = dialog.querySelector<HTMLElement>('[data-prompt]')!;
  const expand = dialog.querySelector<HTMLButtonElement>('[data-expand]')!;

  expect(prompt.classList.contains('folded')).to.equal(true);
  expect([expand.textContent!.trim(), expand.getAttribute('aria-expanded')]).to.deep.equal(['Show more', 'false']);
  expand.click();
  await waitFor(() => !prompt.classList.contains('folded'));
  expect([expand.textContent!.trim(), expand.getAttribute('aria-expanded')]).to.deep.equal(['Show less', 'true']);
  expect(prompt.textContent).to.equal(long);
});

it('lists the recent runs with their local time, state and reason, an Open for the ones that wrote a conversation, and pages to older ones', async () => {
  const api = serving(firesWire(), [WEEKLY]);
  const {dialog} = await openDetail();
  await waitFor(() => dialog.querySelectorAll('[data-fire]').length === 3);

  const fires = [...dialog.querySelectorAll<HTMLElement>('[data-fire]')];
  expect(fires.map((item) => item.dataset.fire)).to.deep.equal(['succeeded', 'skipped_busy', 'failed']);
  expect(fires[0]!.textContent).to.contain(dateTime('2099-01-12T07:00:00.000000Z', getLocale()));
  expect(fires.map((item) => item.querySelector('[data-tone]')!.getAttribute('data-tone'))).to.deep.equal(['good', 'quiet', 'bad']);
  expect(fires[1]!.textContent).to.contain('Skipped, still busy');
  expect(fires[1]!.textContent).to.contain('its previous run is still working');
  expect(fires.map((item) => item.querySelector('[data-open]')?.getAttribute('data-open') ?? null)).to.deep.equal(['conversation-3', null, 'conversation-4']);

  dialog.querySelector<HTMLButtonElement>('[data-older]')!.click();
  await waitFor(() => dialog.querySelectorAll('[data-fire]').length === 4);
  expect(dialog.querySelector('[data-fire=missed]')!.textContent).to.contain('Missed 3 times');
  expect(dialog.querySelector('[data-older]')).to.equal(null);
  expect(asked(api, 'GET', FIRES).map((request) => new URLSearchParams(request.search).get('cursor'))).to.deep.equal([null, '2099-01-10T07:00:00.000000Z']);
});

it('says so when a Schedule has not run yet, and offers a retry when its history cannot be read', async () => {
  let answers = 0;
  serving({
    [`GET ${FIRES}`]: () => {
      answers += 1;
      return answers === 1 ? new Response('unavailable', {status: 503}) : Response.json({fires: [], next_cursor: null});
    },
  }, [WEEKLY]);
  const {dialog} = await openDetail();

  await waitFor(() => dialog.querySelector('[role=alert]') !== null);
  expect(dialog.querySelector('[role=alert]')!.textContent).to.contain('Could not load the history.');
  dialog.querySelector<HTMLButtonElement>('[role=alert] button')!.click();

  await waitFor(() => dialog.textContent!.includes('It has not run yet.'));
});

it('opens the conversation a run wrote through the sidebar\'s own path, and closes', async () => {
  serving(firesWire(), [WEEKLY]);
  const {dialog} = await openDetail();
  await waitFor(() => dialog.querySelector('[data-open]') !== null);

  dialog.querySelector<HTMLButtonElement>('[data-open="conversation-3"]')!.click();

  await waitFor(() => !dialog.open);
  await waitFor(() => webRouter.current.kind === 'conversation');
  expect(webRouter.current).to.deep.equal(conversationRoute('conversation-3'));
});

it('runs the Schedule now from its detail, says how it went, and reads the history again', async () => {
  const api = serving({
    ...firesWire(),
    'POST /web/api/schedules/schedule-weekly-review/run': () => Response.json(wireFire('f9', '2099-01-13T07:00:00.000000Z', 'started', {manual: true})),
  }, [WEEKLY]);
  const {dialog, shell} = await openDetail();
  await waitFor(() => dialog.querySelector('[data-fire]') !== null);

  action(dialog, 'run')!.click();

  await waitFor(() => noticeOf(shell) === 'Started Weekly review.');
  await waitFor(() => asked(api, 'GET', FIRES).length === 2);
});

it('pauses and resumes from its detail, and shows why a paused Schedule is paused', async () => {
  serving({
    ...firesWire(),
    'POST /web/api/schedules/schedule-weekly-review/pause': () => Response.json({
      ...WEEKLY, state: 'paused', pause_reason: null, next_fire_at: null,
    }),
    'POST /web/api/schedules/schedule-weekly-review/resume': () => Response.json(WEEKLY),
  }, [WEEKLY]);
  const {dialog} = await openDetail();

  action(dialog, 'pause')!.click();
  await waitFor(() => action(dialog, 'resume') !== null);
  expect(dialog.querySelector('[data-when]')!.textContent!.replace(/\s+/g, ' ').trim()).to.equal('Paused');
  expect(action(dialog, 'resume')!.textContent!.trim()).to.equal('Resume');
  action(dialog, 'resume')!.click();
  await waitFor(() => action(dialog, 'pause') !== null);
  expect(dialog.querySelector('[data-when]')!.textContent).to.contain('Next run');
});

it('offers a finished Schedule only its deletion', async () => {
  const done = wireSchedule('Weekly review', {state: 'completed', cron: null, at: '2099-01-05T07:00:00Z', next_fire_at: null});
  serving(firesWire(), [done]);
  const {dialog} = await openDetail();

  expect(dialog.querySelector('[data-cadence]')!.textContent).to.match(/^Once on Jan 5, 2099, 08:00 · Europe\/Stockholm$/);
  expect(dialog.querySelector('[data-when]')!.textContent).to.equal('Done');
  expect([...dialog.querySelectorAll('[data-action]')].map((button) => button.getAttribute('data-action'))).to.deep.equal(['delete']);
});

it('hands over to the edit form, and focus comes back to the row it was opened from', async () => {
  serving(firesWire(), [WEEKLY]);
  const {sidebar, dialog} = await openDetail();

  action(dialog, 'edit')!.click();
  const form = await formOpen(sidebar);
  expect(dialog.open).to.equal(false);
  expect(form.querySelector<HTMLInputElement>('[data-field="name"]')!.value).to.equal('Weekly review');
  form.querySelector<HTMLButtonElement>('button[type=button]:not([data-advanced]):not([data-zone-toggle]):not([data-day])')!.click();

  await waitFor(() => !form.open && document.activeElement === sidebar.querySelector('[data-schedule-id] button.conversation-select'));
});

it('deletes from its detail after a confirmation, and the detail goes with the Schedule', async () => {
  const api = serving({
    ...firesWire(),
    'DELETE /web/api/schedules/schedule-weekly-review': () => new Response(null, {status: 204}),
  }, [WEEKLY]);
  const {sidebar, shell, dialog} = await openDetail();
  const confirm = sidebar.querySelector<HTMLDialogElement>('#schedule-delete-dialog')!;

  action(dialog, 'delete')!.click();
  await waitFor(() => confirm.open);
  confirm.querySelector<HTMLButtonElement>('button[value=delete]')!.click();

  await waitFor(() => !dialog.open && !confirm.open);
  expect(asked(api, 'DELETE', '/web/api/schedules/schedule-weekly-review')).to.have.length(1);
  expect(noticeOf(shell)).to.contain('Deleted Weekly review.');
  expect(sidebar.querySelector('[data-schedule-id]')).to.equal(null);
});

it('closes when its Schedule is deleted somewhere else', async () => {
  serving(firesWire(), [WEEKLY]);
  const {dialog, handles} = await openDetail();

  handles.schedules.drop('schedule-weekly-review');

  await waitFor(() => !dialog.open);
});
