// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {buttonNamed, waitFor} from '../testing/dom.ts';
import {type Handler, mountSettings, openSettings, ownerSkills, type Wire, wire, wireSkill} from '../testing/settings.ts';
import type {DlSettingsDialog} from './settings.ts';
import type {DlSettingsSkills} from './settings-skills.ts';
import type {DlToastRegion} from './toast.ts';

const originalFetch = window.fetch;

afterEach(() => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
  document.body.className = '';
  // biome-ignore lint/suspicious/noDocumentCookie: the CSRF double-submit cookie is the channel under test
  document.cookie = 'dlightrag_web_csrf=; max-age=0';
});

/** Three Skills: the agent's PDF reader, one the owner turned off, and a short one. */
const SKILLS = [
  wireSkill('csv-report', {description: 'Summarise a CSV file into a short report.'}),
  wireSkill('pdf-extract', {description: 'Pull the text and tables out of a PDF.'}),
  wireSkill('old-style-guide', {description: 'House style from last year.', enabled: false}),
];

/** The answer to a route whose Skill no longer exists, in the general error envelope. */
function missing(): Response {
  return Response.json({detail: 'No such skill', error_type: 'not_found', error_kind: null}, {status: 404});
}

function page(settings: DlSettingsDialog): DlSettingsSkills {
  return settings.querySelector('dl-settings-skills')!;
}

function names(settings: DlSettingsDialog): string[] {
  return [...settings.querySelectorAll<HTMLElement>('[data-skill]')].map((row) => row.dataset.skill!);
}

function rowOf(settings: DlSettingsDialog, name: string): HTMLElement {
  return settings.querySelector<HTMLElement>(`[data-skill="${name}"]`)!;
}

function switchOf(settings: DlSettingsDialog, name: string): HTMLButtonElement {
  return settings.querySelector<HTMLButtonElement>(`[data-switch="${name}"]`)!;
}

function viewOf(settings: DlSettingsDialog, name: string): HTMLButtonElement {
  return settings.querySelector<HTMLButtonElement>(`[data-view="${name}"]`)!;
}

function deleteOf(settings: DlSettingsDialog, name: string): HTMLButtonElement {
  return settings.querySelector<HTMLButtonElement>(`[data-delete="${name}"]`)!;
}

/** What the navigation row says of Skills: its short status, and the full line a phone's list shows. */
function status(settings: DlSettingsDialog): {short: string; detail: string} {
  const row = settings.querySelector('nav .dl-nav-item[data-section="skills"]')!;
  return {
    short: row.querySelector('.dl-nav-item-status')!.textContent!.trim(),
    detail: document.getElementById(row.getAttribute('aria-describedby') ?? '')?.textContent?.trim() ?? '',
  };
}

function notice(settings: DlSettingsDialog): DlToastRegion {
  return settings.querySelector('dl-toast-region')!;
}

/** Open Settings on Skills and wait until the page has an answer to show, whichever it is. */
async function openSkills(settings: DlSettingsDialog): Promise<DlSettingsSkills> {
  await openSettings(settings, 'skills');
  const skills = page(settings);
  await waitFor(() => skills.querySelector('[data-skill], #skills-empty-title, [role=alert]') !== null);
  return skills;
}

/** What the page asked of the Skills routes, in order; the other pages' reads are not its business. */
function asked(api: Wire): string[] {
  return api.requests.filter((request) => request.path.startsWith('/web/api/skills'))
    .map((request) => `${request.method} ${request.path}`);
}

function serving(extra: Parameters<typeof wire>[0] = {}): Wire {
  const api = wire({'GET /web/api/skills/mine': () => ownerSkills(SKILLS), ...extra});
  window.fetch = api.fetch;
  return api;
}

it('lists the owner\'s Skills with their switches, marks the one that is off in words, and says how many of the quota', async () => {
  const api = serving();
  const {settings} = mountSettings();
  await openSkills(settings);

  expect(names(settings)).to.deep.equal(['csv-report', 'pdf-extract', 'old-style-guide']);
  expect(page(settings).textContent).to.contain('3 of 20 skills');
  expect(rowOf(settings, 'pdf-extract').textContent).to.contain('Pull the text and tables out of a PDF.');
  // Each switch is named for its Skill, and says its state by aria-checked, not by its name.
  const nameOf = (button: HTMLElement): string => document
    .getElementById(button.getAttribute('aria-labelledby')!)!.textContent!.trim();
  expect(names(settings).map((name) => [nameOf(switchOf(settings, name)), switchOf(settings, name).getAttribute('aria-checked')]))
    .to.deep.equal([['csv-report', 'true'], ['pdf-extract', 'true'], ['old-style-guide', 'false']]);
  // Off is a word on the row, so it never rests on colour; the others carry no such word.
  expect(rowOf(settings, 'old-style-guide').textContent).to.contain('Disabled');
  expect(rowOf(settings, 'csv-report').textContent).not.to.contain('Disabled');
  expect(status(settings)).to.deep.equal({short: '2/3', detail: '2 of 3 on'});
  expect(asked(api)).to.deep.equal(['GET /web/api/skills/mine']);
});

it('turns a Skill off with one PUT, shows the row the server answered, and updates the navigation', async () => {
  // biome-ignore lint/suspicious/noDocumentCookie: the CSRF double-submit cookie is the channel under test
  document.cookie = 'dlightrag_web_csrf=csrf-token';
  const api = serving({
    'PUT /web/api/skills/mine/pdf-extract/enabled': (request) => Response.json({
      ...wireSkill('pdf-extract'),
      description: 'Pull the text and tables out of a PDF (edited).',
      enabled: (request.body as {enabled: boolean}).enabled,
    }),
  });
  const {settings} = mountSettings();
  await openSkills(settings);

  switchOf(settings, 'pdf-extract').click();
  await waitFor(() => switchOf(settings, 'pdf-extract').getAttribute('aria-checked') === 'false'
    && !switchOf(settings, 'pdf-extract').disabled);

  const puts = api.requests.filter((request) => request.method === 'PUT');
  expect(puts.map((request) => request.body)).to.deep.equal([{enabled: false}]);
  expect(puts[0]!.headers.get('X-CSRF-Token')).to.equal('csrf-token');
  expect(puts[0]!.headers.get('Content-Type')).to.equal('application/json');
  // The row is the server's, not the one the page held, and it is marked now.
  expect(rowOf(settings, 'pdf-extract').textContent).to.contain('(edited)');
  expect(rowOf(settings, 'pdf-extract').textContent).to.contain('Disabled');
  expect(status(settings)).to.deep.equal({short: '1/3', detail: '1 of 3 on'});
  // Nothing else was read or written: the reply is the whole answer.
  expect(asked(api)).to.deep.equal(['GET /web/api/skills/mine', 'PUT /web/api/skills/mine/pdf-extract/enabled']);
});

it('holds a Skill while its command is in flight, and lets another one go ahead meanwhile', async () => {
  let releaseFirst!: (response: Response) => void;
  const api = serving({
    'PUT /web/api/skills/mine/csv-report/enabled': () => new Promise<Response>((resolve) => { releaseFirst = resolve; }),
    'PUT /web/api/skills/mine/pdf-extract/enabled': () => Response.json(wireSkill('pdf-extract', {enabled: false})),
  });
  const {settings} = mountSettings();
  await openSkills(settings);

  switchOf(settings, 'csv-report').click();
  await waitFor(() => switchOf(settings, 'csv-report').disabled);
  expect(deleteOf(settings, 'csv-report').disabled).to.equal(true);
  // A second press on the held switch sends nothing; the row beside it is free.
  switchOf(settings, 'csv-report').click();
  switchOf(settings, 'pdf-extract').click();
  await waitFor(() => switchOf(settings, 'pdf-extract').getAttribute('aria-checked') === 'false');
  expect(switchOf(settings, 'csv-report').getAttribute('aria-checked')).to.equal('true');

  releaseFirst(Response.json(wireSkill('csv-report', {enabled: false})));
  await waitFor(() => switchOf(settings, 'csv-report').getAttribute('aria-checked') === 'false'
    && !switchOf(settings, 'csv-report').disabled);
  expect(api.requests.filter((request) => request.method === 'PUT').map((request) => request.path))
    .to.deep.equal(['/web/api/skills/mine/csv-report/enabled', '/web/api/skills/mine/pdf-extract/enabled']);
  expect(names(settings)).to.deep.equal(['csv-report', 'pdf-extract', 'old-style-guide']);
  expect(status(settings)).to.deep.equal({short: '0/3', detail: '0 of 3 on'});
});

it('keeps the authoritative state after a failed toggle, and says so', async () => {
  serving({'PUT /web/api/skills/mine/pdf-extract/enabled': () => new Response('unavailable', {status: 503})});
  const {settings} = mountSettings();
  await openSkills(settings);

  const toggle = switchOf(settings, 'pdf-extract');
  toggle.focus();
  toggle.click();
  await waitFor(() => !toggle.disabled && (notice(settings).textContent?.includes('Could not update pdf-extract.') ?? false));

  expect(toggle.getAttribute('aria-checked')).to.equal('true');
  expect(rowOf(settings, 'pdf-extract').textContent).not.to.contain('Disabled');
  expect(status(settings).short).to.equal('2/3');
  // The switch that was held for the request gets focus back.
  expect(document.activeElement).to.equal(toggle);
});

it('asks before deleting: nothing is sent on Cancel, and one DELETE is sent on confirm', async () => {
  const api = serving({'DELETE /web/api/skills/mine/pdf-extract': () => new Response(null, {status: 204})});
  const {settings} = mountSettings();
  await openSkills(settings);
  const dialog = settings.querySelector<HTMLDialogElement>('#skills-delete')!;
  const trigger = deleteOf(settings, 'pdf-extract');

  trigger.click();
  await waitFor(() => dialog.open);
  expect(dialog.textContent).to.contain('Delete pdf-extract?');
  expect(dialog.textContent).to.contain('cannot be undone');
  dialog.querySelector<HTMLButtonElement>('button[value=cancel]')!.click();
  await waitFor(() => !dialog.open && document.activeElement === trigger);
  expect(api.requests.some((request) => request.method === 'DELETE')).to.equal(false);
  expect(names(settings)).to.have.length(3);

  trigger.click();
  await waitFor(() => dialog.open);
  dialog.querySelector<HTMLButtonElement>('button[value=delete]')!.click();
  await waitFor(() => !names(settings).includes('pdf-extract'));

  expect(api.requests.filter((request) => request.method === 'DELETE').map((request) => request.path))
    .to.deep.equal(['/web/api/skills/mine/pdf-extract']);
  expect(notice(settings).textContent).to.contain('Deleted pdf-extract.');
  expect(page(settings).textContent).to.contain('2 of 20 skills');
  expect(status(settings)).to.deep.equal({short: '1/2', detail: '1 of 2 on'});
  // The row the reader was on is gone, so focus lands on the one that took its place.
  await waitFor(() => document.activeElement === viewOf(settings, 'old-style-guide'));
});

it('keeps the Skill and says so when the delete fails', async () => {
  serving({'DELETE /web/api/skills/mine/pdf-extract': () => new Response('unavailable', {status: 503})});
  const {settings} = mountSettings();
  await openSkills(settings);
  const dialog = settings.querySelector<HTMLDialogElement>('#skills-delete')!;
  const trigger = deleteOf(settings, 'pdf-extract');

  trigger.click();
  await waitFor(() => dialog.open);
  dialog.querySelector<HTMLButtonElement>('button[value=delete]')!.click();
  await waitFor(() => notice(settings).textContent?.includes('Could not delete pdf-extract.') ?? false);

  expect(names(settings)).to.have.length(3);
  await waitFor(() => !trigger.disabled && document.activeElement === trigger);
});

describe('a Skill that no longer exists', () => {
  const gone: Array<{what: string; act: (settings: DlSettingsDialog) => Promise<void>; route: Record<string, Handler>}> = [
    {
      what: 'turning it off',
      act: async (settings: DlSettingsDialog) => { switchOf(settings, 'pdf-extract').click(); },
      route: {'PUT /web/api/skills/mine/pdf-extract/enabled': missing},
    },
    {
      what: 'deleting it',
      act: async (settings: DlSettingsDialog) => {
        const dialog = settings.querySelector<HTMLDialogElement>('#skills-delete')!;
        deleteOf(settings, 'pdf-extract').click();
        await waitFor(() => dialog.open);
        dialog.querySelector<HTMLButtonElement>('button[value=delete]')!.click();
      },
      route: {'DELETE /web/api/skills/mine/pdf-extract': missing},
    },
    {
      what: 'reading its SKILL.md',
      act: async (settings: DlSettingsDialog) => { viewOf(settings, 'pdf-extract').click(); },
      route: {'GET /web/api/skills/mine/pdf-extract/document': missing},
    },
  ];
  for (const {what, act, route} of gone) {
    it(`leaves the list with a notice when ${what} finds it gone`, async () => {
      serving(route);
      const {settings} = mountSettings();
      await openSkills(settings);

      await act(settings);
      await waitFor(() => !names(settings).includes('pdf-extract'));

      expect(names(settings)).to.deep.equal(['csv-report', 'old-style-guide']);
      expect(notice(settings).textContent).to.contain('pdf-extract no longer exists, so it was removed from the list.');
      expect(notice(settings).textContent).not.to.contain('Could not');
      expect(status(settings)).to.deep.equal({short: '1/2', detail: '1 of 2 on'});
    });
  }

  it('is not brought back by a toggle that was still on its way when it went', async () => {
    let releaseToggle!: (response: Response) => void;
    serving({
      'PUT /web/api/skills/mine/pdf-extract/enabled': () => new Promise<Response>((resolve) => { releaseToggle = resolve; }),
      'GET /web/api/skills/mine/pdf-extract/document': missing,
    });
    const {settings} = mountSettings();
    await openSkills(settings);

    switchOf(settings, 'pdf-extract').click();
    await waitFor(() => switchOf(settings, 'pdf-extract').disabled);
    viewOf(settings, 'pdf-extract').click();
    await waitFor(() => !names(settings).includes('pdf-extract'));
    releaseToggle(Response.json(wireSkill('pdf-extract', {enabled: false})));
    await new Promise((resolve) => setTimeout(resolve, 0));
    await page(settings).updateComplete;

    expect(names(settings)).to.deep.equal(['csv-report', 'old-style-guide']);
    expect(status(settings).short).to.equal('1/2');
  });
});

it('reads a Skill\'s SKILL.md once, shows it as plain text, and does not read it again', async () => {
  const text = '---\nname: pdf-extract\n---\n# Pull text\n<img src=x onerror="window.skillMarkup = true"> **bold**\n';
  const api = serving({
    'GET /web/api/skills/mine/pdf-extract/document': () => new Response(text, {
      headers: {'Content-Type': 'text/plain; charset=utf-8'},
    }),
  });
  const {settings} = mountSettings();
  await openSkills(settings);
  const view = viewOf(settings, 'pdf-extract');
  expect(view.getAttribute('aria-label')).to.equal('View pdf-extract');
  expect(view.getAttribute('aria-expanded')).to.equal('false');
  expect(rowOf(settings, 'pdf-extract').querySelector('pre')).to.equal(null);

  view.click();
  await waitFor(() => rowOf(settings, 'pdf-extract').querySelector('pre') !== null);
  const pre = rowOf(settings, 'pdf-extract').querySelector('pre')!;
  expect(view.getAttribute('aria-expanded')).to.equal('true');
  // The file's own words, markup and all, as text: nothing in it is a tag, a heading, or bold.
  expect(pre.textContent).to.equal(text);
  expect(pre.children).to.have.length(0);
  expect(pre.getAttribute('role')).to.equal('region');
  expect(pre.getAttribute('aria-label')).to.equal('SKILL.md of pdf-extract');
  expect(pre.tabIndex).to.equal(0);

  view.click();
  await waitFor(() => rowOf(settings, 'pdf-extract').querySelector('pre') === null);
  expect(view.getAttribute('aria-expanded')).to.equal('false');
  view.click();
  await waitFor(() => rowOf(settings, 'pdf-extract').querySelector('pre') !== null);

  expect(rowOf(settings, 'pdf-extract').querySelector('pre')!.textContent).to.equal(text);
  expect(api.requests.filter((request) => request.path.endsWith('/document'))).to.have.length(1);
});

it('says when a SKILL.md could not be read, and reads it again from Retry', async () => {
  let reads = 0;
  serving({
    'GET /web/api/skills/mine/pdf-extract/document': () => {
      reads += 1;
      return reads === 1 ? new Response('unavailable', {status: 503}) : new Response('# Pull text\n');
    },
  });
  const {settings} = mountSettings();
  await openSkills(settings);

  viewOf(settings, 'pdf-extract').click();
  const row = rowOf(settings, 'pdf-extract');
  await waitFor(() => buttonNamed(row, 'Retry') !== null);
  expect(row.textContent).to.contain('Could not load SKILL.md.');
  expect(row.querySelector('pre')).to.equal(null);

  const retry = buttonNamed(row, 'Retry')!;
  retry.focus();
  retry.click();
  await waitFor(() => row.querySelector('pre') !== null);
  expect(row.querySelector('pre')!.textContent).to.equal('# Pull text\n');
  expect(reads).to.equal(2);
  // The button that asked is gone, so focus follows to what answered.
  await waitFor(() => document.activeElement === row.querySelector('pre'));
});

it('says there is nothing yet, in the page and in the navigation, and says how the agent can make one', async () => {
  serving({'GET /web/api/skills/mine': () => ownerSkills([])});
  const {settings} = mountSettings();
  await openSkills(settings);

  expect(names(settings)).to.deep.equal([]);
  expect(page(settings).textContent).to.contain('0 of 20 skills');
  expect(page(settings).querySelector('#skills-empty-title')!.textContent).to.equal('No skills of your own yet');
  expect(page(settings).textContent).to.contain('Ask the agent to create one');
  expect(status(settings)).to.deep.equal({short: '', detail: 'None yet'});
});

it('shows loading, then a failed first read in its own note with Retry, and reads again from it', async () => {
  let reads = 0;
  let release!: (response: Response) => void;
  serving({
    'GET /web/api/skills/mine': () => {
      reads += 1;
      return reads === 1 ? new Promise<Response>((resolve) => { release = resolve; }) : ownerSkills(SKILLS);
    },
  });
  const {settings} = mountSettings();
  await openSettings(settings, 'skills');
  await waitFor(() => reads === 1);
  expect(page(settings).textContent).to.contain('Loading skills…');
  expect(names(settings)).to.deep.equal([]);

  release(new Response('unavailable', {status: 503}));
  await waitFor(() => buttonNamed(page(settings), 'Retry') !== null);
  expect(page(settings).textContent).to.contain('Could not load skills.');
  expect(page(settings).querySelector('[role=alert]')).not.to.equal(null);
  expect(status(settings)).to.deep.equal({short: '', detail: ''});

  buttonNamed(page(settings), 'Retry')!.click();
  await waitFor(() => names(settings).length === 3);
  expect(reads).to.equal(2);
  expect(status(settings).short).to.equal('2/3');
});
