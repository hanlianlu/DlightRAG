// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {sendKeys} from '@web/test-runner-commands';
import {buttonNamed, waitFor} from '../testing/dom.ts';
import {
  memorySettings,
  mountSettings,
  nameOf,
  openSettings,
  pageElement,
  populated,
  row,
  rows,
  settingsClosed,
  wire,
} from '../testing/settings.ts';
import type {DlSettingsDialog} from './settings.ts';

const originalFetch = window.fetch;

afterEach(() => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
  document.body.className = '';
});

/** The pages in the order the navigation lists them, and what a populated Settings says of each. */
const PAGES = [
  {
    section: 'connections', name: 'Connections', status: '1/2', detail: 'MCP · 1 of 2 enabled',
    description: 'External MCP servers that Research runs can call. Turning the first one on asks you to confirm once.',
  },
  {
    section: 'agent-accounts', name: 'Agent Accounts', status: '3', detail: '3 websites',
    description: 'Accounts the agent registered on websites. DlightRAG generates and seals each password; nobody can view it.',
  },
  {
    section: 'memory', name: 'Profile Memory', status: '5', detail: 'On · 5 stored',
    description: 'Preferences and facts the agent remembers about you across conversations.',
  },
  {
    section: 'skills', name: 'Skills', status: '1/2', detail: '1 of 2 on',
    description: 'Instructions the agent can load when they fit the task. Turn one off to keep it without the agent using it; delete removes it for good.',
  },
  {
    section: 'conversations', name: 'Conversation Sessions', status: '0', detail: '0 conversations · kept 365 days',
    description: 'Conversations retain 365 days',
  },
  {
    section: 'privacy', name: 'Privacy', status: '', detail: 'On',
    description: 'Choose whether your conversations may be used to improve DlightRAG.',
  },
  {
    section: 'language', name: 'Language', status: '', detail: 'Automatic',
    description: 'The language of the interface.',
  },
] as const;

/** The names of the rows that are current, as a reader would hear them. */
function currentRows(settings: DlSettingsDialog): string[] {
  return [...settings.querySelectorAll('nav [aria-current="page"]')].map(nameOf);
}

it('does not expose runtime model catalogue administration in Settings', async () => {
  window.fetch = populated().fetch;
  const {settings} = mountSettings();
  await openSettings(settings);

  // Every page is mounted while Settings is open, so this is all of what the dialog can show a reader.
  expect(/model catalogue/i.test(settings.textContent ?? ''), 'no page names it').to.equal(false);
  expect(Boolean(settings.querySelector('dl-model-catalogue'))).to.equal(false);
  expect(customElements.get('dl-model-catalogue')).to.equal(undefined);
});

it('opens a modal named Settings on Connections, with three labelled groups of pages', async () => {
  window.fetch = populated().fetch;
  const {settings} = mountSettings();
  const dialog = await openSettings(settings);

  expect(dialog.matches(':modal')).to.equal(true);
  expect(document.getElementById(dialog.getAttribute('aria-labelledby')!)!.textContent).to.equal('Settings');
  const nav = settings.querySelector('nav')!;
  expect(nav.getAttribute('aria-label')).to.equal('Settings');
  const groups = [...nav.querySelectorAll('[role="group"]')].map((group) => ({
    name: nameOf(group),
    pages: [...group.querySelectorAll('.dl-nav-item')].map(nameOf),
  }));
  expect(groups).to.deep.equal([
    {name: 'Agent', pages: ['Connections', 'Agent Accounts', 'Profile Memory', 'Skills']},
    {name: 'Data', pages: ['Conversation Sessions', 'Privacy']},
    {name: 'General', pages: ['Language']},
  ]);

  // Exactly the open page is current, the page pane is a region named for its heading, and
  // keyboard focus starts on the current row.
  const current = nav.querySelectorAll('[aria-current="page"]');
  expect([...current].map(nameOf)).to.deep.equal(['Connections']);
  const region = settings.querySelector('[role="region"]')!;
  expect(document.getElementById(region.getAttribute('aria-labelledby')!)!.textContent).to.equal('Connections');
  expect(region.textContent).to.contain('External MCP servers that Research runs can call.');
  expect(document.activeElement).to.equal(current[0]);
  expect(document.body.classList.contains('settings-open')).to.equal(true);
});

it('shows one page at a time while every page stays mounted, and keeps focus on the row it was given', async () => {
  window.fetch = populated().fetch;
  const {settings} = mountSettings();
  await openSettings(settings);
  const pages = PAGES.map(({section}) => pageElement(settings, `dl-settings-${section}`));
  expect(pages.map((page) => page.hidden)).to.deep.equal(PAGES.map((_, index) => index !== 0));

  for (const [index, {section, name, description}] of PAGES.entries()) {
    // A pointer press focuses the row it clicks; a script's click does not, so this one does.
    const target = row(settings, section);
    target.focus();
    target.click();
    await settings.updateComplete;

    expect(pages.map((page) => page.hidden), name).to.deep.equal(PAGES.map((_, other) => other !== index));
    expect(currentRows(settings)).to.deep.equal([name]);
    const region = settings.querySelector('[role="region"]')!;
    expect(document.getElementById(region.getAttribute('aria-labelledby')!)!.textContent).to.equal(name);
    expect(region.querySelector('p')!.textContent).to.equal(description);
    expect(document.activeElement).to.equal(target);
  }
});

it('moves focus among the rows with the arrow keys, Home, and End, and opens a page with Enter or Space', async () => {
  window.fetch = populated().fetch;
  const {settings} = mountSettings();
  await openSettings(settings);
  const names = PAGES.map(({name}) => name);
  const focused = (): string => nameOf(document.activeElement!);
  expect(focused()).to.equal(names[0]);

  await sendKeys({press: 'ArrowDown'});
  expect(focused()).to.equal(names[1]);
  await sendKeys({press: 'End'});
  expect(focused()).to.equal(names.at(-1));
  await sendKeys({press: 'ArrowDown'});
  expect(focused()).to.equal(names[0]);
  await sendKeys({press: 'ArrowUp'});
  expect(focused()).to.equal(names.at(-1));
  await sendKeys({press: 'Home'});
  expect(focused()).to.equal(names[0]);

  // Moving focus opens nothing; Enter and Space activate the row that has it.
  expect(currentRows(settings)).to.deep.equal([names[0]]);
  await sendKeys({press: 'ArrowDown'});
  await sendKeys({press: 'Enter'});
  await settings.updateComplete;
  expect(currentRows(settings)).to.deep.equal([names[1]]);
  await sendKeys({press: 'ArrowDown'});
  await sendKeys({press: 'Space'});
  await settings.updateComplete;
  expect(currentRows(settings)).to.deep.equal([names[2]]);
});

it('opens on the page it is asked for, and on Connections otherwise', async () => {
  window.fetch = populated().fetch;
  const {settings} = mountSettings();
  await openSettings(settings, 'memory');
  expect([...settings.querySelectorAll('nav [aria-current="page"]')].map(nameOf)).to.deep.equal(['Profile Memory']);
  expect(document.activeElement).to.equal(row(settings, 'memory'));
  settings.querySelector<HTMLDialogElement>('#settings-dialog')!.close();
  await waitFor(() => !document.body.classList.contains('settings-open'));

  await openSettings(settings);
  expect([...settings.querySelectorAll('nav [aria-current="page"]')].map(nameOf)).to.deep.equal(['Connections']);
});

it('tells each row what its page holds, in a short status and in a full line', async () => {
  window.fetch = populated().fetch;
  const {settings} = mountSettings();
  await openSettings(settings);
  const status = (section: string): string => row(settings, section)
    .querySelector('.dl-nav-item-status')!.textContent!.trim();
  const detail = (section: string): string => document
    .getElementById(row(settings, section).getAttribute('aria-describedby') ?? '')?.textContent?.trim() ?? '';

  // A row has a full line as soon as its page has reported, and every page has one to report.
  await waitFor(() => PAGES.every(({section, detail: expected}) => detail(section) === expected));
  expect(PAGES.map(({section}) => status(section))).to.deep.equal(PAGES.map(({status: expected}) => expected));
  expect(PAGES.map(({section}) => detail(section))).to.deep.equal(PAGES.map(({detail: expected}) => expected));
});

it('says nothing short of a Memory that is off, and says Off in full', async () => {
  window.fetch = wire({'GET /web/api/memory/settings': () => memorySettings(false)}).fetch;
  const {settings} = mountSettings();
  await openSettings(settings);
  const memory = row(settings, 'memory');
  await waitFor(() => Boolean(memory.getAttribute('aria-describedby')));

  expect(memory.querySelector('.dl-nav-item-status')!.textContent!.trim()).to.equal('');
  expect(document.getElementById(memory.getAttribute('aria-describedby')!)!.textContent!.trim()).to.equal('Off');
});

it('opens at once, whatever any page is still waiting for', async () => {
  const never = new Promise<Response>(() => {});
  window.fetch = wire({
    'GET /web/api/connections/mcp': () => never,
    'GET /web/api/agent-accounts': () => never,
    'GET /web/api/memory/settings': () => never,
    'GET /web/api/skills/mine': () => never,
    'GET /web/api/privacy/settings': () => never,
  }).fetch;
  const {settings} = mountSettings();
  const dialog = await openSettings(settings);

  expect(dialog.open).to.equal(true);
  expect(rows(settings)).to.have.length(PAGES.length);
  expect(row(settings, 'connections').getAttribute('aria-current')).to.equal('page');
  // No page paints a state nobody has read.
  expect(settings.textContent).to.contain('Loading Connections');
  expect(settings.textContent).to.contain('Loading agent accounts');
  expect(settings.textContent).to.contain('Loading memory settings');
  expect(settings.textContent).to.contain('Loading skills');
  expect(settings.textContent).to.contain('Loading privacy settings');
  expect(settings.querySelector('#memory-enabled-toggle')).to.equal(null);
  expect(settings.querySelector('#privacy-contribute')).to.equal(null);
});

it('tears every page but Memory down on close, drops what they reported, and gives focus back', async () => {
  window.fetch = populated().fetch;
  const {settings} = mountSettings();
  const trigger = document.createElement('button');
  document.body.append(trigger);
  trigger.focus();
  await settings.open(trigger);
  const dialog = settings.querySelector<HTMLDialogElement>('#settings-dialog')!;
  await waitFor(() => dialog.open);
  await waitFor(() => row(settings, 'connections').querySelector('.dl-nav-item-status')!.textContent!.trim() === '1/2');
  const connections = pageElement(settings, 'dl-settings-connections');
  expect(connections.isConnected).to.equal(true);

  dialog.close();
  await waitFor(() => !document.body.classList.contains('settings-open'));
  await settings.updateComplete;

  expect(connections.isConnected).to.equal(false);
  for (const {section} of PAGES.filter(({section}) => section !== 'memory')) {
    expect(settings.querySelector(`dl-settings-${section}`), section).to.equal(null);
  }
  // Memory stays, because a live Memory change arrives whenever Chat says so; it holds nothing read.
  expect(settings.querySelector('dl-settings-memory')).not.to.equal(null);
  expect(rows(settings).map((item) => item.querySelector('.dl-nav-item-status')!.textContent!.trim()))
    .to.deep.equal(PAGES.map(() => ''));
  expect(rows(settings).some((item) => item.hasAttribute('aria-current'))).to.equal(false);
  expect(document.activeElement).to.equal(trigger);
});

it('closes from the Close button, from Escape, and from the scrim, but not from a click inside', async () => {
  window.fetch = populated().fetch;
  const {settings} = mountSettings();
  const dialog = await openSettings(settings);
  const named = (name: string): HTMLElement => [...settings.querySelectorAll<HTMLElement>('dl-icon-button')]
    .find((button) => button.getAttribute('aria-label') === name)!;

  dialog.querySelector<HTMLElement>('nav')!.click();
  expect(dialog.open).to.equal(true);

  named('Close settings').click();
  await waitFor(settingsClosed);

  await openSettings(settings);
  await sendKeys({press: 'Escape'});
  await waitFor(settingsClosed);

  await openSettings(settings);
  dialog.click();
  await waitFor(settingsClosed);
});

it('starts a session of its own when it opens before the last close has been reported', async () => {
  const api = populated();
  window.fetch = api.fetch;
  const {settings} = mountSettings();
  const dialog = await openSettings(settings);
  const first = pageElement(settings, 'dl-settings-connections');
  await waitFor(() => first.isConnected && (first as unknown as {view: unknown}).view !== null);

  // A dialog closes at once but reports it in a task of its own, which a quick reopen can beat.
  dialog.close();
  await settings.open();
  await new Promise((resolve) => setTimeout(resolve, 0));

  const second = pageElement(settings, 'dl-settings-connections');
  expect(second).not.to.equal(first);
  expect(first.isConnected).to.equal(false);
  // The late report of the first close did not tear the second session down.
  expect(dialog.open, 'dialog open').to.equal(true);
  expect(document.body.classList.contains('settings-open'), 'settings-open class').to.equal(true);
  await waitFor(() => api.requests.filter((request) => request.path === '/web/api/connections/mcp').length === 2);
  await waitFor(() => (second as unknown as {view: unknown}).view !== null);
});

it('deletes every conversation through the sidebar\'s command, and closes only once it ran', async () => {
  window.fetch = populated().fetch;
  const {settings} = mountSettings();
  const asked: Array<HTMLElement | null | undefined> = [];
  let outcome = false;
  settings.deleteAllConversations = async (returnFocus) => {
    asked.push(returnFocus);
    return outcome;
  };
  const dialog = await openSettings(settings, 'conversations');
  const button = buttonNamed<HTMLButtonElement>(settings, 'Delete all conversations')!;

  button.click();
  await waitFor(() => asked.length === 1);
  await settings.updateComplete;
  expect(asked[0]).to.equal(button);
  expect(dialog.open).to.equal(true);

  outcome = true;
  button.click();
  await waitFor(settingsClosed);
  expect(asked).to.have.length(2);
});

it('shows a page\'s notice in its own region while open, and hands it to the shell while closed', async () => {
  window.fetch = populated().fetch;
  const {settings, toast} = mountSettings();
  const dialog = await openSettings(settings);
  const language = pageElement(settings, 'dl-settings-language');
  const notify = (from: Element, message: string): void => {
    from.dispatchEvent(new CustomEvent('dl-toast-request', {detail: {message}, bubbles: true, composed: true}));
  };

  notify(language, 'Inside the dialog');
  const own = settings.querySelector('dl-toast-region')!;
  await own.updateComplete;
  expect(own.textContent).to.contain('Inside the dialog');
  expect(own.closest('dialog')).to.equal(dialog);
  expect(toast.textContent?.trim() ?? '').to.equal('');

  dialog.close();
  await waitFor(() => !document.body.classList.contains('settings-open'));
  notify(pageElement(settings, 'dl-settings-memory'), 'After it closed');
  await toast.updateComplete;
  expect(toast.textContent).to.contain('After it closed');
});

it('lets a notice that still offers Undo outlive the dialog, and gives focus back to where it was opened', async () => {
  window.fetch = populated().fetch;
  const {settings, toast} = mountSettings();
  const trigger = document.createElement('button');
  document.body.append(trigger);
  trigger.focus();
  await settings.open(trigger);
  const dialog = settings.querySelector<HTMLDialogElement>('#settings-dialog')!;
  await waitFor(() => dialog.open);
  settings.querySelector('dl-settings-language')!.dispatchEvent(new CustomEvent('dl-toast-request', {
    detail: {message: 'Forgot: one', action: {actionLabel: 'Undo', onAction: async () => 'Undone', focus: true}},
    bubbles: true,
    composed: true,
  }));
  const own = settings.querySelector('dl-toast-region')!;
  await own.updateComplete;
  expect(own.querySelector('button')?.textContent?.trim()).to.equal('Undo');
  await waitFor(() => document.activeElement === own.querySelector('button'));

  dialog.close();
  await waitFor(() => !document.body.classList.contains('settings-open'));
  await toast.updateComplete;

  expect(toast.textContent).to.contain('Forgot: one');
  expect(toast.querySelector('button')?.textContent?.trim()).to.equal('Undo');
  // The reader never reached that Undo: focus is back on what opened Settings.
  expect(document.activeElement).to.equal(trigger);
});
