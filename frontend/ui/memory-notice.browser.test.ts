// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** What the Run left in Profile Memory, said under its answer: when it appears, what it lists, how the reader
 *  takes a change back, and how it is reached from the keyboard. How it is drawn is
 *  `memory-notice.layout.browser.test.ts`'s. */

import {expect} from '@esm-bundle/chai';
import {sendKeys} from '@web/test-runner-commands';
import type {RunMemoryChange} from '../api/memory.ts';
import {defineDesignSystemElements} from '../design-system/index.ts';
import {setLanguagePreference} from '../i18n/locale.ts';
import {type AppHandles, createAppHandles} from '../stores/app-handles.ts';
import {MemoryNotices, type PageVisibility} from '../stores/memory-notices.ts';
import {waitFor} from '../testing/dom.ts';
import type {DlMemoryNotice} from './memory-notice.ts';
import './memory-notice.ts';
import type {ToastRequestDetail} from './toast.ts';

defineDesignSystemElements();

const originalFetch = window.fetch;
const RUN = 'run-1';

const change = (
  id: string,
  action: RunMemoryChange['action'] = 'add',
  writer: RunMemoryChange['writer'] = 'conversation',
  kind: RunMemoryChange['kind'] = 'preference',
): RunMemoryChange => ({
  memoryId: `memory-${id}`, kind, body: `Body ${id}`, action, changeId: `change-${id}`, writer,
});

let toasts: ToastRequestDetail[] = [];
let settingsRequests: unknown[] = [];
const noteToast = (event: Event): void => { toasts.push((event as CustomEvent<ToastRequestDetail>).detail); };
const noteSettings = (event: Event): void => { settingsRequests.push((event as CustomEvent).detail); };

beforeEach(() => {
  toasts = [];
  settingsRequests = [];
  document.addEventListener('dl-toast-request', noteToast);
  document.addEventListener('dl-settings-request', noteSettings);
});

afterEach(async () => {
  document.removeEventListener('dl-toast-request', noteToast);
  document.removeEventListener('dl-settings-request', noteSettings);
  window.fetch = originalFetch;
  document.body.replaceChildren();
  await setLanguagePreference('en');
});

/** A server whose answer for the Run the test sets, and that counts what the store asked. */
function server(initial: readonly RunMemoryChange[] = []) {
  const state = {listed: [...initial], asked: 0};
  return Object.assign(state, {
    read: async (): Promise<readonly RunMemoryChange[]> => {
      state.asked += 1;
      return state.listed;
    },
  });
}

function bagOf(remote: ReturnType<typeof server>, options: {delays?: number[]; visibility?: PageVisibility} = {}): AppHandles {
  return {
    ...createAppHandles(),
    memoryNotices: new MemoryNotices({read: remote.read, delays: options.delays ?? [20, 40, 60], visibility: options.visibility}),
  };
}

/** The element in the wrapping flex row that holds it in the product, after the row's last word. */
async function mount(handles: AppHandles, runId = RUN): Promise<DlMemoryNotice> {
  const row = document.createElement('div');
  row.className = 'dl-action-row';
  const usage = document.createElement('span');
  usage.textContent = '1,234 tokens';
  row.append(usage);
  const element = document.createElement('dl-memory-notice');
  element.handles = handles;
  element.runId = runId;
  row.append(element);
  document.body.append(row);
  await element.updateComplete;
  return element;
}

const trigger = (element: DlMemoryNotice): HTMLButtonElement | null => (
  element.querySelector<HTMLButtonElement>('button[aria-haspopup="dialog"]')
);
const popover = (element: DlMemoryNotice): HTMLElement => element.querySelector<HTMLElement>('[role="dialog"]')!;
const undoButtons = (element: DlMemoryNotice): HTMLButtonElement[] => (
  [...element.querySelectorAll<HTMLButtonElement>('[data-undo]')]
);
const rows = (element: DlMemoryNotice): HTMLElement[] => [...element.querySelectorAll<HTMLElement>('li')];
const text = (node: Element | null): string => node?.textContent?.replace(/\s+/g, ' ').trim() ?? '';

/** A server that keeps what the memory routes were asked, and answers them as the real ones do. */
function memoryRoutes(reply?: (request: {method: string; path: string}) => Response | undefined) {
  const requests: {method: string; path: string}[] = [];
  window.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), window.location.origin);
    const request = {method: init?.method ?? 'GET', path: url.pathname};
    requests.push(request);
    const custom = reply?.(request);
    if (custom) return custom;
    const forgotten = /^\/web\/api\/memory\/([^/]+)$/.exec(request.path);
    if (request.method === 'DELETE' && forgotten) {
      return Response.json({
        action: 'forget', outcome: 'changed', change_id: `forget-of-${forgotten[1]}`,
        memory_ids: [forgotten[1]], kind: 'preference', body: `Body ${forgotten[1]!.replace('memory-', '')}`,
      });
    }
    const undone = /^\/web\/api\/memory\/changes\/([^/]+)\/undo$/.exec(request.path);
    if (request.method === 'POST' && undone) {
      return Response.json({
        action: 'undo', outcome: 'changed', change_id: `undo-of-${undone[1]}`, memory_ids: ['restored'],
        kind: 'fact', body: 'Lives in Berlin.', target_change_id: undone[1],
      });
    }
    return new Response('not found', {status: 404});
  }) as typeof fetch;
  return requests;
}

describe('when it appears', () => {
  it('draws nothing while the Run has left nothing', async () => {
    const remote = server();
    const element = await mount(bagOf(remote));
    await waitFor(() => remote.asked > 0);
    await element.updateComplete;

    expect(trigger(element)).to.equal(null);
    expect(element.querySelector('[data-action-part]')).to.equal(null);
  });

  it('is one more part of the row, naming how many the Run left, once the server has said so', async () => {
    const element = await mount(bagOf(server([change('1'), change('2', 'update')])));
    await waitFor(() => trigger(element) !== null);

    const button = trigger(element)!;
    expect(text(button)).to.equal('Remembered 2');
    expect(button.getAttribute('aria-expanded')).to.equal('false');
    expect(button.getAttribute('aria-controls')).to.equal(popover(element).id);
    expect(popover(element).hidden).to.equal(true);
    // The row's own parts are its direct children; this one declares itself so the row draws its line.
    expect(button.closest('[data-action-part]')).to.not.equal(null);
    expect(button.closest('.dl-action-row')).to.not.equal(null);
    expect(button.querySelector('svg')).to.not.equal(null);
    expect(button.querySelector('svg')!.getAttribute('aria-hidden')).to.equal('true');
  });

  it('says it in Chinese when the page speaks Chinese', async () => {
    await setLanguagePreference('zh');
    const element = await mount(bagOf(server([change('1'), change('2')])));
    await waitFor(() => trigger(element) !== null);

    expect(text(trigger(element))).to.equal('记住了 2 条');
    trigger(element)!.click();
    await element.updateComplete;
    expect(text(popover(element).querySelector('p'))).to.equal('从这段对话记住的');
    expect(text(undoButtons(element)[0]!)).to.equal('撤销');
    expect(text(popover(element).querySelector('[data-manage]'))).to.equal('在设置中管理记忆');
    expect(text(rows(element)[0]!.firstElementChild)).to.equal('偏好');
  });

  it('fades in when it arrives while the reader is here, and is not animated when it is drawn with the page', async () => {
    const remote = server();
    const handles = bagOf(remote);
    const element = await mount(handles);
    await element.updateComplete;
    expect(trigger(element)).to.equal(null);

    remote.listed = [change('1')];
    handles.memoryNotices.watch(RUN);
    await waitFor(() => trigger(element) !== null);
    expect(trigger(element)!.closest('[data-action-part]')!.hasAttribute('data-fresh'), 'it arrived').to.equal(true);

    // Drawn again from what the page already read, as a turn scrolled back into view is.
    const again = await mount(handles);
    await waitFor(() => trigger(again) !== null);
    expect(trigger(again)!.closest('[data-action-part]')!.hasAttribute('data-fresh'), 'it was already here').to.equal(false);
  });

  it('is read once for a turn that was already answered, however many times it is drawn', async () => {
    const remote = server([change('1')]);
    const handles = bagOf(remote);
    const first = await mount(handles);
    await waitFor(() => trigger(first) !== null);
    first.remove();
    const second = await mount(handles);
    await waitFor(() => trigger(second) !== null);

    expect(remote.asked).to.equal(1);
  });

  it('looks again after a Run it watched settles, and shows what the pass learned', async () => {
    const remote = server();
    const handles = bagOf(remote);
    const element = await mount(handles);
    handles.memoryNotices.watch(RUN);
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(trigger(element), 'the pass has not reported').to.equal(null);

    remote.listed = [change('1', 'add', 'conversation')];
    await waitFor(() => trigger(element) !== null);

    expect(text(trigger(element))).to.equal('Remembered 1');
  });
});

describe('announcing', () => {
  it('says it once, politely, to a reader who was here for the answer, and leaves their focus alone', async () => {
    const remote = server();
    const handles = bagOf(remote);
    const element = await mount(handles);
    const field = document.createElement('input');
    document.body.append(field);
    field.focus();
    handles.memoryNotices.watch(RUN);
    await element.updateComplete;
    expect(element.querySelector('[role="status"]')?.getAttribute('aria-live'), 'the region exists before the news')
      .to.equal('polite');
    expect(text(element.querySelector('[role="status"]'))).to.equal('');

    remote.listed = [change('1'), change('2')];
    await waitFor(() => trigger(element) !== null);
    await element.updateComplete;

    expect(text(element.querySelector('[role="status"]'))).to.equal('Remembered 2');
    expect(document.activeElement).to.equal(field);

    // It is made once: a later change to the count is not a second announcement.
    remote.listed = [change('1'), change('2'), change('3')];
    handles.memoryNotices.watch(RUN);
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(handles.memoryNotices.announces(RUN)).to.equal(false);
  });

  it('says nothing for a turn that was already answered when the page opened', async () => {
    const element = await mount(bagOf(server([change('1')])));
    await waitFor(() => trigger(element) !== null);
    await element.updateComplete;

    expect(element.querySelector('[role="status"]')).to.equal(null);
  });
});

describe('the popover', () => {
  it('lists each change with its kind, its words and its Undo, and a way into Settings', async () => {
    const element = await mount(bagOf(server([
      change('1', 'add', 'agent', 'preference'),
      change('2', 'update', 'conversation', 'fact'),
      change('3', 'fuse', 'conversation', 'experience'),
    ])));
    await waitFor(() => trigger(element) !== null);

    trigger(element)!.click();
    await element.updateComplete;

    expect(trigger(element)!.getAttribute('aria-expanded')).to.equal('true');
    expect(popover(element).hidden).to.equal(false);
    expect(popover(element).getAttribute('aria-labelledby')).to.equal(popover(element).querySelector('p')!.id);
    expect(text(popover(element).querySelector('p'))).to.equal('Remembered from this conversation');
    expect(rows(element).map((row) => [
      text(row.querySelector('span')), text(row.querySelector('p')), text(row.querySelector('button')),
    ])).to.deep.equal([
      ['Preference', 'Body 1', 'Undo'],
      ['Fact', 'Body 2', 'Undo'],
      ['Experience', 'Body 3', 'Undo'],
    ]);
    expect(undoButtons(element)[1]!.getAttribute('aria-label')).to.equal('Undo: Body 2');
    expect(text(popover(element).querySelector('[data-manage]'))).to.equal('Manage memory in Settings');
  });

  it('opens from the keyboard onto its first control, and Escape closes it and gives focus back', async () => {
    const element = await mount(bagOf(server([change('1'), change('2')])));
    await waitFor(() => trigger(element) !== null);
    trigger(element)!.focus();

    await sendKeys({press: 'Enter'});
    await waitFor(() => trigger(element)!.getAttribute('aria-expanded') === 'true');
    await waitFor(() => document.activeElement === undoButtons(element)[0]);

    await sendKeys({press: 'Tab'});
    expect(document.activeElement).to.equal(undoButtons(element)[1]);

    await sendKeys({press: 'Escape'});
    await waitFor(() => trigger(element)!.getAttribute('aria-expanded') === 'false');
    await waitFor(() => document.activeElement === trigger(element));
    expect(popover(element).hidden).to.equal(true);
  });

  it('closes when focus leaves it for somewhere else', async () => {
    const element = await mount(bagOf(server([change('1')])));
    const elsewhere = document.createElement('button');
    document.body.append(elsewhere);
    await waitFor(() => trigger(element) !== null);
    trigger(element)!.focus();
    trigger(element)!.click();
    await element.updateComplete;

    elsewhere.focus();
    await element.updateComplete;

    expect(trigger(element)!.getAttribute('aria-expanded')).to.equal('false');
  });

  it('closes on a press outside it', async () => {
    const element = await mount(bagOf(server([change('1')])));
    await waitFor(() => trigger(element) !== null);
    trigger(element)!.click();
    await element.updateComplete;
    await new Promise((resolve) => setTimeout(resolve, 5));

    document.body.click();
    await element.updateComplete;

    expect(trigger(element)!.getAttribute('aria-expanded')).to.equal('false');
  });

  it('opens Settings on the Profile Memory page, from a popover it has put away, with focus on its part', async () => {
    const element = await mount(bagOf(server([change('1')])));
    await waitFor(() => trigger(element) !== null);
    trigger(element)!.click();
    await element.updateComplete;

    popover(element).querySelector<HTMLButtonElement>('[data-manage]')!.click();
    await element.updateComplete;

    expect(settingsRequests).to.deep.equal([{page: 'memory'}]);
    expect(trigger(element)!.getAttribute('aria-expanded')).to.equal('false');
    expect(document.activeElement).to.equal(trigger(element));
  });
});

describe('taking a change back', () => {
  async function opened(changes: RunMemoryChange[]): Promise<DlMemoryNotice> {
    const element = await mount(bagOf(server(changes)));
    await waitFor(() => trigger(element) !== null);
    trigger(element)!.click();
    await element.updateComplete;
    return element;
  }

  it('forgets an addition, and says so in a receipt that offers to bring it back', async () => {
    const requests = memoryRoutes();
    const element = await opened([change('1'), change('2')]);

    undoButtons(element)[0]!.click();
    await waitFor(() => toasts.length > 0);
    await element.updateComplete;

    expect(requests).to.deep.equal([{method: 'DELETE', path: '/web/api/memory/memory-1'}]);
    expect(text(rows(element)[0]!)).to.contain('Undone');
    expect(undoButtons(element).map((button) => text(button.closest('li')))).to.have.length(1);
    expect(text(trigger(element)), 'it counts what still stands').to.equal('Remembered 1');
    expect(toasts[0]!.message).to.equal('Forgot: Body 1');
    expect(toasts[0]!.tone).to.equal(undefined);
    expect(toasts[0]!.action?.actionLabel).to.equal('Undo');
  });

  it('lets go of the row when the receipt\'s Undo brings the record back, for it is no longer this Run\'s', async () => {
    const requests = memoryRoutes();
    const element = await opened([change('1'), change('2')]);
    undoButtons(element)[0]!.click();
    await waitFor(() => toasts.length > 0);

    const said = await toasts[0]!.action!.onAction();
    await element.updateComplete;

    expect(requests.at(-1)).to.deep.equal({method: 'POST', path: '/web/api/memory/changes/forget-of-memory-1/undo'});
    expect(said).to.equal('Profile Memory change undone.');
    expect(rows(element)).to.have.length(1);
    expect(text(trigger(element))).to.equal('Remembered 1');
  });

  it('undoes a correction by its change, and names the record it brings back', async () => {
    const requests = memoryRoutes();
    const element = await opened([change('1', 'update', 'conversation', 'fact')]);

    undoButtons(element)[0]!.click();
    await waitFor(() => toasts.length > 0);
    await element.updateComplete;

    expect(requests).to.deep.equal([{method: 'POST', path: '/web/api/memory/changes/change-1/undo'}]);
    expect(toasts[0]!.message).to.equal('Restored: Lives in Berlin.');
    // An undo cannot itself be undone, so the receipt offers no way to.
    expect(toasts[0]!.action).to.equal(undefined);
    expect(text(rows(element)[0]!)).to.contain('Undone');
  });

  it('undoes a merge by its change, without naming one of the records it brings back', async () => {
    memoryRoutes();
    const element = await opened([change('1', 'fuse')]);

    undoButtons(element)[0]!.click();
    await waitFor(() => toasts.length > 0);

    expect(toasts[0]!.message).to.equal('Profile Memory restored.');
  });

  it('reads "Undone" in place of the count once every change has been taken back, and stays to say so', async () => {
    memoryRoutes();
    const element = await opened([change('1'), change('2')]);

    undoButtons(element)[0]!.click();
    await waitFor(() => undoButtons(element).length === 1);
    undoButtons(element)[0]!.click();
    await waitFor(() => undoButtons(element).length === 0);
    await element.updateComplete;

    expect(text(trigger(element))).to.equal('Undone');
    expect(rows(element).every((row) => text(row).includes('Undone'))).to.equal(true);
  });

  it('moves focus to the next thing the reader can do when the button they pressed is gone', async () => {
    memoryRoutes();
    const element = await opened([change('1'), change('2')]);
    undoButtons(element)[0]!.focus();

    undoButtons(element)[0]!.click();
    await waitFor(() => rows(element).length === 2 && text(rows(element)[0]!).includes('Undone'));
    await waitFor(() => document.activeElement === undoButtons(element)[0]);
    expect(undoButtons(element)).to.have.length(1);

    undoButtons(element)[0]!.click();
    await waitFor(() => document.activeElement === popover(element).querySelector('[data-manage]'));
  });

  it('holds the control still while its request is out, and takes the change back once', async () => {
    let release!: () => void;
    const requests = memoryRoutes(() => undefined);
    const delegate = window.fetch;
    window.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      await new Promise<void>((resolve) => { release = resolve; });
      return await delegate(input, init);
    }) as typeof fetch;
    const element = await opened([change('1')]);

    const button = undoButtons(element)[0]!;
    button.click();
    await element.updateComplete;
    expect(button.getAttribute('aria-disabled')).to.equal('true');
    button.click();
    release();
    await waitFor(() => toasts.length > 0);

    expect(requests.filter((request) => request.method === 'DELETE')).to.have.length(1);
  });

  it('tells a refusal in an error and leaves the change standing, ready to be tried again', async () => {
    memoryRoutes(() => new Response('{}', {status: 500}));
    const element = await opened([change('1', 'update')]);

    undoButtons(element)[0]!.click();
    await waitFor(() => toasts.length > 0);
    await element.updateComplete;

    expect(toasts[0]).to.include({message: 'Could not undo this memory.', tone: 'error'});
    expect(text(trigger(element))).to.equal('Remembered 1');
    expect(undoButtons(element)).to.have.length(1);
    expect(undoButtons(element)[0]!.getAttribute('aria-disabled')).to.equal(null);
  });

  it('tells a conflict as the profile having moved on, and leaves the change standing', async () => {
    memoryRoutes(() => Response.json({
      action: 'undo', outcome: 'conflict', change_id: 'undo-1', memory_ids: [], body: '',
    }));
    const element = await opened([change('1', 'update')]);

    undoButtons(element)[0]!.click();
    await waitFor(() => toasts.length > 0);

    expect(toasts[0]).to.include({
      message: 'Profile Memory changed; recall it before retrying.', tone: 'error',
    });
    expect(text(trigger(element))).to.equal('Remembered 1');
  });
});

describe('a change the server no longer lists', () => {
  it('goes from the notice the next time the server is asked', async () => {
    const remote = server([change('1'), change('2')]);
    const handles = bagOf(remote);
    const element = await mount(handles);
    await waitFor(() => trigger(element) !== null);
    expect(text(trigger(element))).to.equal('Remembered 2');

    remote.listed = [change('2')];
    handles.memoryNotices.watch(RUN);
    await waitFor(() => text(trigger(element)) === 'Remembered 1');
  });
});

describe('the owner changing their profile elsewhere', () => {
  it('reads again what an answer shows, and draws nothing once the server lists nothing, as when Memory is turned off', async () => {
    const remote = server([change('1'), change('2')]);
    const handles = bagOf(remote);
    const element = await mount(handles);
    await waitFor(() => trigger(element) !== null);
    trigger(element)!.click();
    await element.updateComplete;

    remote.listed = [];
    handles.memoryNotices.refresh();
    await waitFor(() => trigger(element) === null);

    expect(remote.asked).to.equal(2);
    expect(element.querySelector('[role="dialog"]')).to.equal(null);
  });

  it('keeps a change the reader took back in the popover while the rest is read again', async () => {
    memoryRoutes();
    const remote = server([change('1'), change('2')]);
    const handles = bagOf(remote);
    const element = await mount(handles);
    await waitFor(() => trigger(element) !== null);
    trigger(element)!.click();
    await element.updateComplete;
    undoButtons(element)[0]!.click();
    await waitFor(() => text(rows(element)[0]!).includes('Undone'));

    remote.listed = [change('2')];
    handles.memoryNotices.refresh();
    await waitFor(() => remote.asked === 2);
    await element.updateComplete;

    expect(rows(element).map((row) => text(row).includes('Undone'))).to.deep.equal([true, false]);
    expect(text(trigger(element))).to.equal('Remembered 1');
  });
});
