// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {sendKeys, setViewport} from '@web/test-runner-commands';
import {linkStyles, waitFor} from '../testing/dom.ts';
import {
  agentAccountsView,
  memoryPage,
  memorySettings,
  mountSettings,
  openSettings,
  wire,
  wireAccount,
} from '../testing/settings.ts';
import type {DlSettingsDialog} from './settings.ts';

const originalFetch = window.fetch;
const originalViewport = {width: window.innerWidth, height: window.innerHeight};

afterEach(() => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
  document.body.className = '';
});

/** The reads a fully populated Settings makes: two Connections, three accounts, five memories. */
function populated(): ReturnType<typeof wire> {
  return wire({
    'GET /web/api/connections/mcp': () => Response.json({
      revision: '1',
      presets: [],
      connections: [
        {connection_id: 'a', label: 'Notion', endpoint: 'https://a.example/mcp', enabled: true,
          activation_epoch: 1, generation: 1, authentication: 'oauth', authorization_status: null, status: 'ready'},
        {connection_id: 'b', label: 'Hugging Face', endpoint: 'https://b.example/mcp', enabled: false,
          activation_epoch: 1, generation: 1, authentication: 'none', authorization_status: null, status: 'disabled'},
      ],
    }),
    'GET /web/api/agent-accounts': () => Response.json(agentAccountsView([
      wireAccount('discourse.org'), wireAccount('huggingface.co'), wireAccount('ycombinator.com'),
    ])),
    'GET /web/api/memory/settings': () => memorySettings(true, 5),
    'GET /web/api/memory': () => memoryPage([{id: 'one', body: 'Use concise answers'}]),
  });
}

/** The navigation rows, in order, as a reader meets them. */
function rows(settings: DlSettingsDialog): HTMLButtonElement[] {
  return [...settings.querySelectorAll<HTMLButtonElement>('nav .dl-nav-item')];
}

function row(settings: DlSettingsDialog, section: string): HTMLButtonElement {
  return settings.querySelector<HTMLButtonElement>(`nav .dl-nav-item[data-section="${section}"]`)!;
}

/** What a row is called: the text its aria-labelledby points at. */
function nameOf(element: Element): string {
  const id = element.getAttribute('aria-labelledby')!;
  return element.ownerDocument.getElementById(id)!.textContent!.trim();
}

function pageElement(settings: DlSettingsDialog, tag: string): HTMLElement {
  return settings.querySelector<HTMLElement>(tag)!;
}

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
    section: 'conversations', name: 'Conversation Sessions', status: '0', detail: '0 conversations · kept 365 days',
    description: 'Conversations retain 365 days',
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
  const {settings} = mountSettings();
  await settings.updateComplete;

  expect(settings.textContent).not.to.contain('Runtime Model Catalogue');
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
    {name: 'Agent', pages: ['Connections', 'Agent Accounts', 'Profile Memory']},
    {name: 'Data', pages: ['Conversation Sessions']},
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
    .getElementById(row(settings, section).getAttribute('aria-describedby')!)!.textContent!.trim();

  await waitFor(() => PAGES.every(({section, status: expected}) => expected === '' || status(section) !== ''));
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
  expect(settings.querySelector('#memory-enabled-toggle')).to.equal(null);
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
  await waitFor(() => !dialog.open);

  await openSettings(settings);
  await sendKeys({press: 'Escape'});
  await waitFor(() => !dialog.open);

  await openSettings(settings);
  dialog.click();
  await waitFor(() => !dialog.open);
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
  expect(dialog.open).to.equal(true);
  expect(document.body.classList.contains('settings-open')).to.equal(true);
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
  const button = settings.querySelector<HTMLButtonElement>('#delete-all-btn')!;

  button.click();
  await waitFor(() => asked.length === 1);
  await settings.updateComplete;
  expect(asked[0]).to.equal(button);
  expect(dialog.open).to.equal(true);

  outcome = true;
  button.click();
  await waitFor(() => !dialog.open);
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

/** A page's notice with an Undo, and the Undo as a finger would find it: showing, and under the point it covers. */
async function noticeWithUndo(settings: DlSettingsDialog): Promise<{region: Element; undo: Element | null; hit: Element | null}> {
  settings.querySelector('dl-settings-language')!.dispatchEvent(new CustomEvent('dl-toast-request', {
    detail: {message: 'Forgot: one', action: {actionLabel: 'Undo', onAction: async () => 'Undone'}},
    bubbles: true,
    composed: true,
  }));
  const region = settings.querySelector('dl-toast-region')!;
  await region.updateComplete;
  await waitFor(() => getComputedStyle(region).opacity === '1');
  const undo = region.querySelector('button');
  const box = undo?.getBoundingClientRect();
  return {
    region,
    undo,
    hit: box ? document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2) : null,
  };
}

describe('on a phone', () => {
  let unlink: () => void;
  before(async () => {
    await setViewport({width: 390, height: 844});
    unlink = await linkStyles([
      '../design-system/index.css',
      '../styles/global.css',
      '../styles/layout.css',
      '../styles/settings.css',
      '../styles/settings-dialog.module.css',
      '../styles/settings-page.module.css',
      '../styles/settings-memory.module.css',
      '../styles/settings-agent-accounts.module.css',
      '../styles/settings-connections.module.css',
    ].map((href) => new URL(href, import.meta.url).href));
  });
  after(async () => {
    unlink();
    await setViewport(originalViewport);
  });

  const visible = (element: Element): boolean => element.getClientRects().length > 0;

  it('opens on the section list with the page out of sight, and the first row focused', async () => {
    window.fetch = populated().fetch;
    const {settings} = mountSettings();
    const dialog = await openSettings(settings);

    expect(visible(settings.querySelector('nav')!)).to.equal(true);
    expect(visible(settings.querySelector('[role="region"]')!)).to.equal(false);
    expect(document.activeElement).to.equal(rows(settings)[0]);
    // The list shows no page, so no row of it is "current".
    expect(rows(settings).some((item) => item.hasAttribute('aria-current'))).to.equal(false);
    expect(dialog.open).to.equal(true);
  });

  it('opens a page from the list, takes the reader to its title, and Back returns to the row it left', async () => {
    window.fetch = populated().fetch;
    const {settings} = mountSettings();
    await openSettings(settings);
    const memory = row(settings, 'memory');

    memory.click();
    await settings.updateComplete;
    const title = settings.querySelector<HTMLElement>('#settings-page-title')!;
    expect(visible(settings.querySelector('nav')!)).to.equal(false);
    expect(visible(settings.querySelector('[role="region"]')!)).to.equal(true);
    expect(title.textContent).to.equal('Profile Memory');
    expect(document.activeElement).to.equal(title);
    expect(memory.getAttribute('aria-current')).to.equal('page');

    const back = [...settings.querySelectorAll<HTMLElement>('dl-icon-button')]
      .find((button) => button.getAttribute('aria-label') === 'Back')!;
    expect(visible(back)).to.equal(true);
    back.click();
    await settings.updateComplete;

    expect(visible(settings.querySelector('nav')!)).to.equal(true);
    expect(visible(settings.querySelector('[role="region"]')!)).to.equal(false);
    expect(document.activeElement).to.equal(memory);
    expect(memory.hasAttribute('aria-current')).to.equal(false);
  });

  it('shows a notice on the section list as well as on a page, with its Undo within reach', async () => {
    window.fetch = populated().fetch;
    const {settings} = mountSettings();
    await openSettings(settings);

    // The list hides the page pane, and a notice from a page that is not showing still lands.
    let shown = await noticeWithUndo(settings);
    expect(visible(settings.querySelector('[role="region"]')!)).to.equal(false);
    expect(shown.undo).not.to.equal(null);
    expect(shown.hit).to.equal(shown.undo);

    row(settings, 'language').click();
    await settings.updateComplete;
    shown = await noticeWithUndo(settings);
    expect(shown.hit).to.equal(shown.undo);
  });

  it('opens straight on a page it is asked for, with the list one Back away', async () => {
    window.fetch = populated().fetch;
    const {settings} = mountSettings();
    await openSettings(settings, 'connections');

    expect(visible(settings.querySelector('nav')!)).to.equal(false);
    expect(visible(settings.querySelector('[role="region"]')!)).to.equal(true);
    expect(document.activeElement).to.equal(settings.querySelector('#settings-page-title'));
  });
});

describe('on a desktop', () => {
  let unlink: () => void;
  before(async () => {
    await setViewport({width: 1280, height: 800});
    unlink = await linkStyles([
      '../design-system/index.css',
      '../styles/global.css',
      '../styles/layout.css',
      '../styles/settings.css',
      '../styles/settings-dialog.module.css',
      '../styles/settings-page.module.css',
      '../styles/settings-memory.module.css',
    ].map((href) => new URL(href, import.meta.url).href));
  });
  after(async () => {
    unlink();
    await setViewport(originalViewport);
  });

  it('is a centered dialog with a fixed size, a hairline border, and a navigation column beside its page', async () => {
    window.fetch = populated().fetch;
    const {settings} = mountSettings();
    const dialog = await openSettings(settings);

    const box = dialog.getBoundingClientRect();
    expect([box.width, box.height]).to.deep.equal([880, 640]);
    expect(box.x + box.width / 2).to.be.closeTo(640, 1);
    expect(box.y + box.height / 2).to.be.closeTo(400, 1);
    const style = getComputedStyle(dialog);
    expect(style.borderRadius).to.equal('22px');
    expect(style.borderTopWidth).to.equal('1px');
    const nav = settings.querySelector('nav')!.getBoundingClientRect();
    const pane = settings.querySelector('[role="region"]')!.getBoundingClientRect();
    expect(nav.right).to.be.at.most(pane.left + 1);
    expect(nav.top).to.be.closeTo(pane.top, 1);

    // Another page does not resize it.
    row(settings, 'language').click();
    await settings.updateComplete;
    const after = dialog.getBoundingClientRect();
    expect([after.width, after.height]).to.deep.equal([880, 640]);
  });

  it('shows a notice with its Undo within reach of a pointer', async () => {
    window.fetch = populated().fetch;
    const {settings} = mountSettings();
    await openSettings(settings);

    const shown = await noticeWithUndo(settings);
    expect(shown.undo).not.to.equal(null);
    expect(shown.hit).to.equal(shown.undo);
  });

  it('opens every page at its top, and scrolls its own pane rather than the dialog', async () => {
    const many = Array.from({length: 30}, (_, index) => ({id: `m${index}`, body: `Memory number ${index}`}));
    window.fetch = wire({
      'GET /web/api/memory/settings': () => memorySettings(true, 30),
      'GET /web/api/memory': () => memoryPage(many),
    }).fetch;
    const {settings} = mountSettings();
    const dialog = await openSettings(settings, 'memory');
    await waitFor(() => settings.querySelectorAll('dl-settings-memory li').length === 30);
    const body = settings.querySelector<HTMLElement>('[data-page-body]')!;

    expect(body.scrollHeight).to.be.greaterThan(body.clientHeight);
    expect(dialog.scrollHeight).to.equal(dialog.clientHeight);
    body.scrollTop = 200;
    expect(body.scrollTop).to.be.greaterThan(0);
    row(settings, 'language').click();
    await settings.updateComplete;
    row(settings, 'memory').click();
    await settings.updateComplete;

    expect(body.scrollTop).to.equal(0);
  });
});
