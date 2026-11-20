// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {setLanguagePreference} from '../i18n/locale.ts';
import {buttonNamed, waitFor} from '../testing/dom.ts';
import {
  type Handler,
  mountSettings,
  openSettings,
  privacySettings,
  settingsClosed,
  wire,
  wireMoment,
} from '../testing/settings.ts';
import type {DlSettingsDialog} from './settings.ts';
import type {DlSettingsPrivacy} from './settings-privacy.ts';
import type {DlToastRegion} from './toast.ts';

const originalFetch = window.fetch;

afterEach(() => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
  document.body.className = '';
  // biome-ignore lint/suspicious/noDocumentCookie: the CSRF double-submit cookie is the channel under test
  document.cookie = 'dlightrag_web_csrf=; max-age=0';
});

/** The Privacy switch, once the page has read the owner's setting. */
function contribution(settings: DlSettingsDialog): HTMLButtonElement {
  return settings.querySelector<HTMLButtonElement>('#privacy-contribute')!;
}

function page(settings: DlSettingsDialog): DlSettingsPrivacy {
  return settings.querySelector('dl-settings-privacy')!;
}

/** The sentence under the switch that says what it does now. */
function caption(settings: DlSettingsDialog): string {
  return document.getElementById(contribution(settings).getAttribute('aria-describedby')!)!.textContent!.trim();
}

/** What the navigation says of Privacy in full: the line a phone's list shows and a screen reader hears. */
function status(settings: DlSettingsDialog): string {
  const row = settings.querySelector('nav .dl-nav-item[data-section="privacy"]')!;
  return document.getElementById(row.getAttribute('aria-describedby') ?? '')?.textContent?.trim() ?? '';
}

function notice(settings: DlSettingsDialog): DlToastRegion {
  return settings.querySelector('dl-toast-region[tone="error"]')!;
}

/** The switch has settled: it shows `checked` and is free to press again. */
function settled(settings: DlSettingsDialog, checked: boolean): boolean {
  return contribution(settings).getAttribute('aria-checked') === String(checked) && !contribution(settings).disabled;
}

/** Open Settings on Privacy and wait until the page has an answer to show, whichever it is. */
async function openPrivacy(settings: DlSettingsDialog): Promise<HTMLDialogElement> {
  const dialog = await openSettings(settings, 'privacy');
  await waitFor(() => page(settings).querySelector('#privacy-contribute, [role=alert]') !== null);
  return dialog;
}

it('paints the switch only once the owner\'s setting has been read, never an unread "off"', async () => {
  let release!: (response: Response) => void;
  const pending = new Promise<Response>((resolve) => { release = resolve; });
  const api = wire({'GET /web/api/privacy/settings': () => pending});
  window.fetch = api.fetch;
  const {settings} = mountSettings();
  await openSettings(settings, 'privacy');

  expect(contribution(settings)).to.equal(null);
  expect(page(settings).textContent).to.contain('Loading privacy settings');
  expect(status(settings)).to.equal('');

  release(privacySettings(false));
  await waitFor(() => Boolean(contribution(settings)));
  expect(contribution(settings).getAttribute('aria-checked')).to.equal('false');
  expect(contribution(settings).disabled).to.equal(false);
  expect(caption(settings)).to.equal('Off: DlightRAG does not use your conversations to evaluate or train models');
  expect(status(settings)).to.equal('Off');
  // Opening reads and nothing else: a permission is never written by looking at it.
  expect(api.requests.filter((request) => request.path.startsWith('/web/api/privacy'))
    .map((request) => `${request.method} ${request.path}`)).to.deep.equal(['GET /web/api/privacy/settings']);
});

it('opens fail-closed when the read fails, and Retry reads again and hands focus to what answers', async () => {
  let reads = 0;
  window.fetch = wire({
    'GET /web/api/privacy/settings': () => {
      reads += 1;
      return reads === 1 ? new Response('unavailable', {status: 503}) : privacySettings(true);
    },
  }).fetch;
  const {settings} = mountSettings();
  await openPrivacy(settings);

  // A refused read leaves no control at all: an unchecked switch would claim "off" for a state nobody read.
  expect(contribution(settings)).to.equal(null);
  expect(page(settings).querySelector('[role=alert]')!.textContent).to.equal('Could not load privacy settings.');
  expect(status(settings)).to.equal('');

  const retry = buttonNamed(page(settings), 'Retry')!;
  retry.focus();
  retry.click();
  await waitFor(() => Boolean(contribution(settings)));
  expect(reads).to.equal(2);
  expect(contribution(settings).getAttribute('aria-checked')).to.equal('true');
  expect(status(settings)).to.equal('On');
  // The Retry that asked is gone, so focus follows to the switch it revealed.
  await waitFor(() => document.activeElement === contribution(settings));
});

it('shows a permission that was given as on, and says what it covers and what it never does', async () => {
  window.fetch = wire({'GET /web/api/privacy/settings': () => privacySettings(true)}).fetch;
  const {settings} = mountSettings();
  await openPrivacy(settings);

  expect(contribution(settings).getAttribute('aria-checked')).to.equal('true');
  expect(document.getElementById('privacy-contribute-label')!.textContent!.trim())
    .to.equal('Use my conversations to improve DlightRAG');
  // The wire names the moment with microseconds, as the server does; the page shows the day.
  expect(caption(settings)).to.equal('Conversations started since Mar 4, 2025 may be used to evaluate and train models');
  expect(status(settings)).to.equal('On');
  const terms = [...page(settings).querySelectorAll('dt')].map((term) => term.textContent!.trim());
  const statements = [...page(settings).querySelectorAll('dd')].map((statement) => statement.textContent!.trim());
  expect(terms).to.deep.equal(['May include', 'Never includes', 'How it works']);
  expect(statements).to.deep.equal([
    'Your messages and the agent\'s answers, its reasoning and tool calls, the passages it read, and your feedback on answers with its reasons and notes.',
    'The passwords of your agent accounts and the secrets of your connections.',
    'It applies to conversations you start after you turn it on, and stops as soon as you turn it off.',
  ]);
});

it('names the day conversations count from in the app\'s own short date, with the year only when it is not this year\'s', async () => {
  window.fetch = wire({
    'GET /web/api/privacy/settings': () => privacySettings(true, wireMoment(new Date().getFullYear(), 1, 15)),
  }).fetch;
  const {settings} = mountSettings();
  await openPrivacy(settings);

  expect(caption(settings)).to.equal('Conversations started since Jan 15 may be used to evaluate and train models');
});

it('writes that date, what the permission applies to, and what off promises, in the reader\'s language', async () => {
  try {
    await setLanguagePreference('zh');
    window.fetch = wire({
      'GET /web/api/privacy/settings': () => privacySettings(true),
      'PUT /web/api/privacy/settings': () => privacySettings(false),
    }).fetch;
    const {settings} = mountSettings();
    await openPrivacy(settings);

    expect(document.getElementById('privacy-contribute-label')!.textContent!.trim()).to.equal('用我的会话来改进 DlightRAG');
    expect(caption(settings)).to.equal('2025年3月4日 及之后发起的会话可能会被用于评估和训练模型');
    expect([...page(settings).querySelectorAll('dd')].map((statement) => statement.textContent!.trim())).to.deep.equal([
      '你的消息和代理的回答、它的推理过程和工具调用、它读取的内容片段，以及你对回答的反馈（含原因和备注）。',
      '你的代理账号的密码和你的连接的密钥。',
      '适用于你开启之后发起的会话，关闭后立即停止。',
    ]);
    expect(status(settings)).to.equal('已启用');

    settings.querySelector<HTMLElement>('#privacy-contribute-label')!.click();
    await waitFor(() => settled(settings, false));
    expect(caption(settings)).to.equal('已关闭：DlightRAG 不会用你的会话来评估或训练模型');
    expect(status(settings)).to.equal('已关闭');
  } finally {
    await setLanguagePreference('en');
  }
});

it('turns with one PUT carrying the CSRF token, holds the switch meanwhile, shows what the server answers, and gives focus back', async () => {
  // biome-ignore lint/suspicious/noDocumentCookie: the CSRF double-submit cookie is the channel under test
  document.cookie = 'dlightrag_web_csrf=csrf-token';
  let release!: (response: Response) => void;
  const answer = new Promise<Response>((resolve) => { release = resolve; });
  const api = wire({
    'GET /web/api/privacy/settings': () => privacySettings(false),
    'PUT /web/api/privacy/settings': () => answer,
  });
  window.fetch = api.fetch;
  const {settings} = mountSettings();
  await openPrivacy(settings);

  contribution(settings).focus();
  contribution(settings).click();
  await waitFor(() => contribution(settings).disabled);
  // Nothing is shown as given until the server has said so, and a second press sends nothing.
  expect(contribution(settings).getAttribute('aria-checked')).to.equal('false');
  expect(status(settings)).to.equal('Off');
  contribution(settings).click();

  release(privacySettings(true));
  await waitFor(() => settled(settings, true));
  const puts = api.requests.filter((request) => request.method === 'PUT');
  expect(puts.map((request) => request.body)).to.deep.equal([{contribute: true}]);
  expect(puts[0]!.headers.get('X-CSRF-Token')).to.equal('csrf-token');
  expect(puts[0]!.headers.get('Content-Type')).to.equal('application/json');
  // The caption names the day conversations count from, as the server's reply does.
  expect(caption(settings)).to.equal('Conversations started since Mar 4, 2025 may be used to evaluate and train models');
  expect(status(settings)).to.equal('On');
  // The switch was disabled for the request, which drops focus in some engines: the reader keeps their place.
  await waitFor(() => document.activeElement === contribution(settings));
});

it('turns off from a tap anywhere on its card', async () => {
  const api = wire({
    'GET /web/api/privacy/settings': () => privacySettings(true),
    'PUT /web/api/privacy/settings': (request) => privacySettings((request.body as {contribute: boolean}).contribute),
  });
  window.fetch = api.fetch;
  const {settings} = mountSettings();
  await openPrivacy(settings);

  // The card is the switch's label, so a tap on its words turns it.
  settings.querySelector<HTMLElement>('#privacy-contribute-label')!.click();
  await waitFor(() => settled(settings, false));

  expect(api.requests.filter((request) => request.method === 'PUT').map((request) => request.body))
    .to.deep.equal([{contribute: false}]);
  expect(caption(settings)).to.equal('Off: DlightRAG does not use your conversations to evaluate or train models');
  expect(status(settings)).to.equal('Off');
});

it('keeps the stored state after a refused PUT, says so, and gives focus back', async () => {
  window.fetch = wire({
    'GET /web/api/privacy/settings': () => privacySettings(false),
    'PUT /web/api/privacy/settings': () => new Response('unavailable', {status: 503}),
  }).fetch;
  const {settings} = mountSettings();
  await openPrivacy(settings);

  contribution(settings).focus();
  contribution(settings).click();
  await waitFor(() => !contribution(settings).disabled
    && (notice(settings).textContent?.includes('Could not save privacy settings.') ?? false));

  expect(contribution(settings).getAttribute('aria-checked')).to.equal('false');
  expect(status(settings)).to.equal('Off');
  await waitFor(() => document.activeElement === contribution(settings));
});

/** A GET the test keeps in its hand until it lets it go, and whether the page has taken the answer's body by
 *  now. A reply is late for a page only once the page has read it, and a test that looks before then proves
 *  nothing, so the test waits for this signal and not for a turn of the clock. */
function heldRead(): {handler: Handler; started: () => boolean; release: (response: Response) => void; bodyRead: () => boolean} {
  let answer: ((response: Response) => void) | null = null;
  let read = false;
  return {
    handler: () => new Promise<Response>((resolve) => { answer = resolve; }),
    started: () => answer !== null,
    release: (response) => {
      const json = response.json.bind(response);
      response.json = async () => {
        const body = await json();
        read = true;
        return body;
      };
      answer?.(response);
    },
    bodyRead: () => read,
  };
}

it('takes a held read once it is released while Settings is still open', async () => {
  const held = heldRead();
  window.fetch = wire({'GET /web/api/privacy/settings': held.handler}).fetch;
  const {settings} = mountSettings();
  await openSettings(settings, 'privacy');
  await waitFor(held.started);

  held.release(privacySettings(true));
  await waitFor(held.bodyRead);
  // The control for the test below: the same read, answered in the same way, does land on a page that is open.
  await waitFor(() => Boolean(contribution(settings)));
  expect(page(settings).settings).not.to.equal(null);
  expect(status(settings)).to.equal('On');
});

it('changes nothing for a read that answers after Settings has closed', async () => {
  const held = heldRead();
  window.fetch = wire({'GET /web/api/privacy/settings': held.handler}).fetch;
  const {settings} = mountSettings();
  const dialog = await openSettings(settings, 'privacy');
  await waitFor(held.started);
  const closing = page(settings);

  dialog.close();
  await waitFor(settingsClosed);
  held.release(privacySettings(true));
  // The page has the answer in hand once its body is read; what it does with it comes in the turns after.
  await waitFor(held.bodyRead);
  await closing.updateComplete;
  await new Promise((resolve) => setTimeout(resolve, 0));

  // The page that asked is gone with its dialog, and what it was told afterwards is not kept.
  expect(closing.isConnected).to.equal(false);
  expect(closing.settings).to.equal(null);
  expect(closing.querySelector('#privacy-contribute')).to.equal(null);
  expect(status(settings)).to.equal('');
});
