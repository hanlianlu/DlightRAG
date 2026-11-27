// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {browserZone} from '../lib/zoned-time.ts';
import {waitFor} from '../testing/dom.ts';
import {
  asked,
  choose,
  chosen,
  field,
  formOpen,
  mountSidebar,
  noticeOf,
  rowsShown,
  serving,
  type,
  useWorkspaces,
  wireSchedule,
} from '../testing/schedules.ts';
import type {Wire} from '../testing/settings.ts';

const originalFetch = window.fetch;

afterEach(() => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
  document.body.className = '';
  window.localStorage.removeItem('dlightrag.schedule_section_open');
});

const SLOT = {at: '2099-01-05T07:00:00.000000Z', local: '2099-01-05T08:00:00+01:00'};

/** The wire for a form: the cadence preview answers one slot, and a save answers the Schedule it was sent as. */
function forming(extra: Parameters<typeof serving>[0] = {}, schedules: Parameters<typeof serving>[1] = []): Wire {
  return serving({
    'POST /web/api/schedules/preview': () => Response.json({zone: 'Europe/Stockholm', slots: [SLOT]}),
    'POST /web/api/schedules': (request) => Response.json(wireSchedule('Digest', {
      ...(request.body as object),
      schedule_id: 'schedule-digest',
    }), {status: 201}),
    ...extra,
  }, schedules);
}

/** Open the form for a new Schedule on a page with one workspace. */
async function newForm() {
  const mounted = await mountSidebar();
  useWorkspaces(mounted.handles, ['alpha'], ['alpha']);
  const add = mounted.sidebar.querySelector<HTMLButtonElement>('[data-new]')!;
  await waitFor(() => !add.disabled);
  add.click();
  const dialog = await formOpen(mounted.sidebar);
  return {...mounted, dialog, add};
}

function previewBody(api: Wire): unknown {
  return asked(api, 'POST', '/web/api/schedules/preview').at(-1)?.body;
}

/** The cadence the form has last asked the server about, once the reader has paused. */
async function previewed(api: Wire, body: unknown): Promise<void> {
  await waitFor(() => JSON.stringify(previewBody(api)) === JSON.stringify(body), 1000);
}

/** The form has closed and given focus back to the menu button of the row it was opened from. */
async function closedAround(sidebar: HTMLElement, schedule: string, dialog: HTMLDialogElement): Promise<void> {
  const button = sidebar.querySelector(`[data-schedule-id="${schedule}"] [data-menu-button]`);
  await waitFor(() => !dialog.open && document.activeElement === button);
  await sidebar.querySelector('dl-schedule-form')!.updateComplete;
}

function submit(dialog: HTMLDialogElement): void {
  dialog.querySelector<HTMLFormElement>('form')!.requestSubmit();
}

function problem(dialog: HTMLDialogElement): string | undefined {
  return dialog.querySelector('[data-problem]')?.textContent?.trim();
}

it('asks a name, a prompt and when, with everything else defaulted, and creates the Schedule', async () => {
  const api = forming();
  const {sidebar, shell, dialog, add} = await newForm();

  expect(dialog.querySelector('h2')!.textContent).to.equal('New schedule');
  expect(document.activeElement).to.equal(field(dialog, 'name'));
  expect(chosen(field<HTMLElement>(dialog, 'frequency'))).to.equal('daily');
  expect(field(dialog, 'time').value).to.equal('09:00');
  expect(field<HTMLTextAreaElement>(dialog, 'prompt').getAttribute('placeholder')).to.equal('What should the agent do each time?');
  expect(field<HTMLTextAreaElement>(dialog, 'prompt').rows).to.equal(5);
  expect([...field<HTMLElement>(dialog, 'frequency').querySelectorAll('label')].map((chip) => chip.textContent!.trim())).to.deep.equal([
    'Once', 'Hourly', 'Daily', 'Weekdays', 'Weekly', 'Monthly', 'Custom',
  ]);
  // The design system's chips and radio rows carry every choice: the form has no native select.
  expect(dialog.querySelector('select')).to.equal(null);
  // Advanced is closed, and the zone is a word that opens its input.
  expect(dialog.querySelector('[data-advanced]')!.getAttribute('aria-expanded')).to.equal('false');
  expect(dialog.querySelector('[data-zone-toggle]')!.textContent).to.equal(browserZone());
  expect(field(dialog, 'zone')).to.equal(null);

  type(field(dialog, 'name'), 'Digest');
  type(field<HTMLTextAreaElement>(dialog, 'prompt'), 'Summarize the week.');
  await previewed(api, {zone: browserZone(), cron: '0 9 * * *'});
  await waitFor(() => dialog.querySelector('[data-next]') !== null);
  const summary = dialog.querySelector('[data-summary]')!.textContent!.replace(/\s+/g, ' ').trim();
  expect(summary.startsWith(`Every day at 09:00 · ${browserZone()} ·`)).to.equal(true);
  expect(summary).to.contain('Next: ');

  submit(dialog);
  await rowsShown(sidebar, 1);

  expect(asked(api, 'POST', '/web/api/schedules')[0]!.body).to.deep.equal({
    name: 'Digest',
    prompt: 'Summarize the week.',
    mode: 'research',
    workspaces: ['alpha'],
    target: 'new_conversation',
    zone: browserZone(),
    cron: '0 9 * * *',
    max_tokens_per_day: 1_500_000,
  });
  expect(noticeOf(shell)).to.equal('Saved Digest.');
  await waitFor(() => !dialog.open && document.activeElement === add);
});

it('compiles each frequency to the cron it means, and asks the server what that reads as', async () => {
  const api = forming();
  const {dialog} = await newForm();
  const frequency = field<HTMLElement>(dialog, 'frequency');
  const zone = browserZone();

  choose(frequency, 'hourly');
  await waitFor(() => field(dialog, 'minute') !== null);
  type(field(dialog, 'minute'), '45');
  await previewed(api, {zone, cron: '45 * * * *'});

  choose(frequency, 'weekdays');
  await waitFor(() => field(dialog, 'time') !== null);
  type(field(dialog, 'time'), '07:30');
  await previewed(api, {zone, cron: '30 7 * * 1-5'});

  choose(frequency, 'weekly');
  await waitFor(() => dialog.querySelector('[data-day]') !== null);
  const chips = [...dialog.querySelectorAll<HTMLInputElement>('[data-day]')];
  expect(chips.map((chip) => chip.parentElement!.textContent!.trim())).to.deep.equal(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']);
  expect(chips.map((chip) => chip.checked)).to.deep.equal([true, false, false, false, false, false, false]);
  chips[3]!.click();
  await previewed(api, {zone, cron: '30 7 * * 1,4'});
  chips[6]!.click();
  await previewed(api, {zone, cron: '30 7 * * 0,1,4'});
  chips[0]!.click();
  chips[3]!.click();
  await previewed(api, {zone, cron: '30 7 * * 0'});
  // Sunday is the last day left, so it stays.
  chips[6]!.click();
  await dialog.closest('dl-schedule-form')!.updateComplete;
  expect(chips[6]!.checked).to.equal(true);

  choose(frequency, 'monthly');
  await waitFor(() => field(dialog, 'day') !== null);
  type(field(dialog, 'day'), '15');
  await previewed(api, {zone, cron: '30 7 15 * *'});

  choose(frequency, 'custom');
  await waitFor(() => field(dialog, 'cron') !== null);
  expect(field(dialog, 'time')).to.equal(null);
  type(field(dialog, 'cron'), '0 6 * * 1');
  await previewed(api, {zone, cron: '0 6 * * 1'});
  expect(dialog.textContent).to.contain('Five fields: minute, hour, day of month, month, day of week.');

  choose(frequency, 'once');
  await waitFor(() => field(dialog, 'date') !== null);
  dialog.querySelector<HTMLButtonElement>('[data-zone-toggle]')!.click();
  await waitFor(() => field(dialog, 'zone') !== null);
  type(field(dialog, 'zone'), 'Asia/Tokyo');
  type(field(dialog, 'date'), '2099-06-01');
  type(field(dialog, 'time'), '08:00');
  await previewed(api, {zone: 'Asia/Tokyo', at: '2099-05-31T23:00:00.000Z'});
});

it('keeps mode, workspaces, where each run goes, goal and ceiling under Advanced, with the defaults it starts with', async () => {
  const api = forming();
  const {dialog, handles} = await newForm();
  useWorkspaces(handles, ['alpha', 'beta'], ['alpha']);
  const advanced = dialog.querySelector<HTMLButtonElement>('[data-advanced]')!;

  expect(dialog.querySelector('#schedule-advanced')!.hasAttribute('hidden')).to.equal(true);
  advanced.click();
  await waitFor(() => field(dialog, 'ceiling') !== null);
  expect(advanced.getAttribute('aria-expanded')).to.equal('true');
  expect(chosen(field<HTMLElement>(dialog, 'mode'))).to.equal('research');
  expect(field(dialog, 'ceiling').value).to.equal('1.5');
  expect(chosen(field<HTMLElement>(dialog, 'target'))).to.equal('new_conversation');
  expect(dialog.querySelector('select')).to.equal(null);
  const boxes = [...dialog.querySelectorAll<HTMLInputElement>('[data-workspace]')];
  expect(boxes.map((box) => [box.dataset.workspace, box.checked])).to.deep.equal([['alpha', true], ['beta', false]]);

  boxes[1]!.click();
  choose(field<HTMLElement>(dialog, 'mode'), 'fast');
  choose(field<HTMLElement>(dialog, 'target'), 'same_conversation');
  field(dialog, 'goal').click();
  await waitFor(() => field(dialog, 'rounds') !== null);
  const mode = field<HTMLElement>(dialog, 'mode');
  expect([chosen(mode), [...mode.querySelectorAll('input')].every((radio) => radio.disabled)]).to.deep.equal(['research', true]);
  expect(mode.textContent).to.contain('A goal runs in Research.');
  expect(field(dialog, 'rounds').value).to.equal('16');
  expect(dialog.textContent).to.contain('Goal objective');
  type(field(dialog, 'rounds'), '5');
  type(field(dialog, 'ceiling'), '0.9');
  type(field(dialog, 'name'), 'Deep dive');
  type(field<HTMLTextAreaElement>(dialog, 'prompt'), 'Map the market.');
  submit(dialog);
  await waitFor(() => asked(api, 'POST', '/web/api/schedules').length === 1);

  expect(asked(api, 'POST', '/web/api/schedules')[0]!.body).to.deep.equal({
    name: 'Deep dive',
    prompt: 'Map the market.',
    mode: 'research',
    workspaces: ['alpha', 'beta'],
    target: 'same_conversation',
    zone: browserZone(),
    cron: '0 9 * * *',
    max_tokens_per_day: 900_000,
    goal_max_rounds: 5,
  });
});

it('shows no workspace choice while there is one workspace, and runs the Schedule in it', async () => {
  const api = forming();
  const {dialog} = await newForm();

  dialog.querySelector<HTMLButtonElement>('[data-advanced]')!.click();
  await waitFor(() => field(dialog, 'ceiling') !== null);
  expect(dialog.querySelector('[data-workspace]')).to.equal(null);
  type(field(dialog, 'name'), 'Solo');
  type(field<HTMLTextAreaElement>(dialog, 'prompt'), 'Look.');
  submit(dialog);
  await waitFor(() => asked(api, 'POST', '/web/api/schedules').length === 1);

  expect((asked(api, 'POST', '/web/api/schedules')[0]!.body as {workspaces: string[]}).workspaces).to.deep.equal(['alpha']);
});

it('says what is missing without asking the server, and opens Advanced when what is wrong is in it', async () => {
  const api = forming();
  const {dialog} = await newForm();

  submit(dialog);
  await waitFor(() => problem(dialog) !== undefined);
  expect(problem(dialog)).to.equal('Give the schedule a name.');
  type(field(dialog, 'name'), 'Digest');
  submit(dialog);
  await waitFor(() => problem(dialog) === 'Write what the schedule should ask.');
  type(field<HTMLTextAreaElement>(dialog, 'prompt'), 'Look.');
  choose(field<HTMLElement>(dialog, 'frequency'), 'custom');
  await waitFor(() => field(dialog, 'cron') !== null);
  submit(dialog);
  await waitFor(() => problem(dialog) === 'Say when it should run.');
  type(field(dialog, 'cron'), '0 6 * * 1');
  dialog.querySelector<HTMLButtonElement>('[data-advanced]')!.click();
  await waitFor(() => field(dialog, 'ceiling') !== null);
  type(field(dialog, 'ceiling'), '0');
  dialog.querySelector<HTMLButtonElement>('[data-advanced]')!.click();
  await waitFor(() => dialog.querySelector('#schedule-advanced')!.hasAttribute('hidden'));
  submit(dialog);

  await waitFor(() => problem(dialog) === 'The daily limit is a number of million tokens above 0.');
  await waitFor(() => !dialog.querySelector('#schedule-advanced')!.hasAttribute('hidden'));
  expect(asked(api, 'POST', '/web/api/schedules')).to.have.length(0);
});

it('shows the reason a cadence would be refused in the line that reads it back, and keeps the form with the server\'s reason when saving is refused', async () => {
  const refusal = {detail: 'slots may be at least 60 minutes apart', error_type: 'validation'};
  const api = forming({
    'POST /web/api/schedules/preview': () => Response.json(refusal, {status: 422}),
    'POST /web/api/schedules': () => Response.json(refusal, {status: 422}),
  });
  const {dialog} = await newForm();

  choose(field<HTMLElement>(dialog, 'frequency'), 'custom');
  await waitFor(() => field(dialog, 'cron') !== null);
  type(field(dialog, 'cron'), '*/5 * * * *');
  await waitFor(() => dialog.querySelector('[data-refused]')?.textContent === refusal.detail, 1000);
  expect(dialog.querySelector('[data-refused]')!.getAttribute('role')).to.equal('alert');
  expect(dialog.querySelector('[data-next]')).to.equal(null);

  type(field(dialog, 'name'), 'Too often');
  type(field<HTMLTextAreaElement>(dialog, 'prompt'), 'Look.');
  submit(dialog);
  await waitFor(() => problem(dialog) === refusal.detail);

  expect(dialog.open).to.equal(true);
  expect(dialog.querySelector<HTMLButtonElement>('button[type=submit]')!.disabled).to.equal(false);
  expect(asked(api, 'POST', '/web/api/schedules')).to.have.length(1);
});

it('reads a stored cadence back into the frequency it is, shows the Schedule being edited, and sends only what changed', async () => {
  const stored = wireSchedule('Filing watch', {
    cron: '0 8 * * 1,4',
    workspaces: ['alpha'],
    target: 'same_conversation',
    goal_max_rounds: 4,
    max_tokens_per_day: 500_000,
    state: 'paused',
    pause_reason: null,
    next_fire_at: null,
  });
  const api = forming({
    'PATCH /web/api/schedules/schedule-filing-watch': (request) => Response.json({...stored, ...(request.body as object)}),
  }, [stored]);
  const {sidebar, handles} = await mountSidebar();
  useWorkspaces(handles, ['alpha'], ['alpha']);
  await rowsShown(sidebar, 1);

  const open = async () => {
    sidebar.querySelector<HTMLButtonElement>('[data-schedule-id] [data-menu-button]')!.click();
    await waitFor(() => sidebar.querySelector('[data-schedule-id] dl-menu') !== null);
    sidebar.querySelector<HTMLButtonElement>('[data-menu="edit"]')!.click();
    return await formOpen(sidebar);
  };
  let dialog = await open();
  expect(dialog.querySelector('h2')!.textContent).to.equal('Edit schedule');
  expect(field(dialog, 'name').value).to.equal('Filing watch');
  expect(chosen(field<HTMLElement>(dialog, 'frequency'))).to.equal('weekly');
  expect(field(dialog, 'time').value).to.equal('08:00');
  expect([...dialog.querySelectorAll<HTMLInputElement>('[data-day]')].map((chip) => chip.checked)).to.deep.equal(
    [true, false, false, true, false, false, false],
  );
  expect(dialog.querySelector('[data-zone-toggle]')!.textContent).to.equal('Europe/Stockholm');
  expect(dialog.querySelector('button[type=submit]')!.textContent).to.equal('Save');
  // What it was made as stays: its mode is the goal's, and where its runs go and whether it is a goal are not offered.
  dialog.querySelector<HTMLButtonElement>('[data-advanced]')!.click();
  await waitFor(() => field(dialog, 'ceiling') !== null);
  expect(field(dialog, 'ceiling').value).to.equal('0.5');
  expect([...field<HTMLElement>(dialog, 'mode').querySelectorAll('input')].every((radio) => radio.disabled)).to.equal(true);
  expect(field(dialog, 'target')).to.equal(null);
  expect(field(dialog, 'goal')).to.equal(null);

  // Nothing changed: saving closes it and asks nothing of the server.
  submit(dialog);
  await closedAround(sidebar, 'schedule-filing-watch', dialog);
  expect(asked(api, 'PATCH', '/web/api/schedules/schedule-filing-watch')).to.have.length(0);

  dialog = await open();
  type(field(dialog, 'name'), 'Filing digest');
  dialog.querySelector<HTMLInputElement>('[data-day="3"]')!.click();
  submit(dialog);
  await waitFor(() => sidebar.querySelector('[data-schedule-id]')!.textContent!.includes('Filing digest'));

  expect(asked(api, 'PATCH', '/web/api/schedules/schedule-filing-watch')[0]!.body).to.deep.equal({name: 'Filing digest', cron: '0 8 * * 1,3,4'});
});

it('reads a cron that is no preset as Custom, and a one-time Schedule as the date and time of its own zone', async () => {
  const custom = wireSchedule('Often', {cron: '0 */4 * * *'});
  const once = wireSchedule('Launch', {cron: null, at: '2099-05-31T23:00:00Z', zone: 'Asia/Tokyo', next_fire_at: null});
  forming({}, [custom, once]);
  const {sidebar} = await mountSidebar();
  await rowsShown(sidebar, 2);
  const edit = async (id: string) => {
    sidebar.querySelector<HTMLButtonElement>(`[data-schedule-id="${id}"] [data-menu-button]`)!.click();
    await waitFor(() => sidebar.querySelector(`[data-schedule-id="${id}"] dl-menu`) !== null);
    sidebar.querySelector<HTMLButtonElement>(`[data-schedule-id="${id}"] [data-menu="edit"]`)!.click();
    return await formOpen(sidebar);
  };

  let dialog = await edit('schedule-often');
  expect(chosen(field<HTMLElement>(dialog, 'frequency'))).to.equal('custom');
  expect(field(dialog, 'cron').value).to.equal('0 */4 * * *');
  dialog.querySelector<HTMLButtonElement>('button[type=button]:not([data-advanced]):not([data-zone-toggle])')!.click();
  await closedAround(sidebar, 'schedule-often', dialog);

  dialog = await edit('schedule-launch');
  expect(chosen(field<HTMLElement>(dialog, 'frequency'))).to.equal('once');
  expect(field(dialog, 'date').value).to.equal('2099-06-01');
  expect(field(dialog, 'time').value).to.equal('08:00');
  expect(dialog.querySelector('[data-zone-toggle]')!.textContent).to.equal('Asia/Tokyo');
});
