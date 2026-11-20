// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** Settings as the shipped stylesheets lay it out (the product-styles page of the runner): on a phone the
 *  dialog is a list that opens pages, on a desktop it is both at once. */

import {expect} from '@esm-bundle/chai';
import {setViewport} from '@web/test-runner-commands';
import {buttonNamed, waitFor} from '../testing/dom.ts';
import {
  memoryPage,
  memorySettings,
  mountSettings,
  openSettings,
  ownerSkills,
  pageElement,
  populated,
  row,
  rows,
  wire,
  wireSkill,
} from '../testing/settings.ts';
import type {DlSettingsDialog} from './settings.ts';

const originalFetch = window.fetch;
const originalViewport = {width: window.innerWidth, height: window.innerHeight};

afterEach(() => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
  document.body.className = '';
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
  before(async () => {
    await setViewport({width: 390, height: 844});
  });
  after(async () => {
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

  it('lists the accounts as rows when Settings opens straight on them, though the page was out of sight when it mounted', async () => {
    window.fetch = populated().fetch;
    const {settings} = mountSettings();
    await openSettings(settings, 'agent-accounts');

    const accounts = pageElement(settings, 'dl-settings-agent-accounts');
    await waitFor(() => accounts.querySelectorAll('li').length === 3);
    expect(accounts.querySelector('table')).to.equal(null);
  });

  it('opens straight on a page it is asked for, with the list one Back away', async () => {
    window.fetch = populated().fetch;
    const {settings} = mountSettings();
    await openSettings(settings, 'connections');

    expect(visible(settings.querySelector('nav')!)).to.equal(false);
    expect(visible(settings.querySelector('[role="region"]')!)).to.equal(true);
    expect(document.activeElement).to.equal(settings.querySelector('#settings-page-title'));
  });

  it('draws a memory as a short block, with the kind under the words and Forget at the row\'s end', async () => {
    window.fetch = wire({
      'GET /web/api/memory/settings': () => memorySettings(true, 2),
      'GET /web/api/memory': () => memoryPage([
        {id: 'one', body: 'Prefers concise answers.'},
        {id: 'two', body: 'The owner expects a clickable link whenever the Agent says a deliverable has been published.'},
      ]),
    }).fetch;
    const {settings} = mountSettings();
    const dialog = await openSettings(settings, 'memory');
    await waitFor(() => settings.querySelectorAll('dl-settings-memory li').length === 2);
    const [short, long] = [...settings.querySelectorAll<HTMLElement>('dl-settings-memory li')];
    const part = (item: HTMLElement, name: string): HTMLElement => item.querySelector(`[class*=${name}]`)!;
    const box = (element: Element): DOMRect => element.getBoundingClientRect();
    // The words take the row's width, the kind stands under them, and Forget keeps the row's end.
    expect(box(part(short!, 'text')).left - box(short!).left).to.be.at.most(13);
    expect(box(part(short!, 'kind')).top).to.be.at.least(box(part(short!, 'text')).bottom);
    expect(box(part(short!, 'kind')).right).to.be.lessThan(box(part(short!, 'forget')).left);
    // One line of words is a short row, and the longer memory is not a card either.
    expect(box(short!).height).to.be.at.most(90);
    expect(box(long!).height).to.be.at.most(150);
    // A phone's pointer still gets the control ladder's target for Forget.
    expect(box(part(short!, 'forget')).height).to.be.at.least(44);
    expect(dialog.open).to.equal(true);
  });
});

describe('in a window narrower than the dialog and its margins', () => {
  before(async () => {
    await setViewport({width: 800, height: 700});
  });
  after(async () => {
    await setViewport(originalViewport);
  });

  it('keeps the kind at the tail beside the words rather than taking a line of its own', async () => {
    window.fetch = wire({
      'GET /web/api/memory/settings': () => memorySettings(true, 1),
      'GET /web/api/memory': () => memoryPage([{id: 'one',
        body: 'The owner expects a clickable link (and the workspace path) whenever the Agent says a deliverable has been published or attached, rather than a bare claim of publication.'}]),
    }).fetch;
    const {settings} = mountSettings();
    await openSettings(settings, 'memory');
    await waitFor(() => settings.querySelectorAll('dl-settings-memory li').length === 1);
    const item = settings.querySelector<HTMLElement>('dl-settings-memory li')!;
    const part = (name: string): HTMLElement => item.querySelector(`[class*=${name}]`)!;
    const box = (element: Element): DOMRect => element.getBoundingClientRect();

    // The words begin at the row's own edge, and the kind still stands beside them, not under them.
    expect(box(part('text')).left - box(item).left).to.be.at.most(13);
    expect(box(part('kind')).left).to.be.at.least(box(part('text')).right - 1);
    expect(box(part('kind')).top).to.be.lessThan(box(part('text')).bottom);
    // The row is the words, the padding around them, and at most the control beside a short line.
    expect(box(item).height - box(part('text')).height).to.be.at.most(34);
  });
});

describe('on a desktop', () => {
  before(async () => {
    await setViewport({width: 1280, height: 800});
  });
  after(async () => {
    await setViewport(originalViewport);
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
    window.fetch = wire({
      'GET /web/api/skills/mine': () => ownerSkills(
        Array.from({length: 20}, (_, index) => wireSkill(`skill-${index}`)),
      ),
    }).fetch;
    const {settings} = mountSettings();
    const dialog = await openSettings(settings, 'skills');
    await waitFor(() => settings.querySelectorAll('dl-settings-skills [data-skill]').length === 20);
    const body = settings.querySelector<HTMLElement>('[data-page-body]')!;

    expect(body.scrollHeight).to.be.greaterThan(body.clientHeight);
    expect(dialog.scrollHeight).to.equal(dialog.clientHeight);
    body.scrollTop = 200;
    expect(body.scrollTop).to.be.greaterThan(0);
    row(settings, 'language').click();
    await settings.updateComplete;
    row(settings, 'skills').click();
    await settings.updateComplete;

    expect(body.scrollTop).to.equal(0);
  });

  it('keeps a long memory list in the card\'s own window, with its title and the Clear action still', async () => {
    const many = Array.from({length: 30}, (_, index) => ({id: `m${index}`, body: `Memory number ${index}`}));
    window.fetch = wire({
      'GET /web/api/memory/settings': () => memorySettings(true, 30),
      'GET /web/api/memory': () => memoryPage(many, 'older'),
    }).fetch;
    const {settings} = mountSettings();
    const dialog = await openSettings(settings, 'memory');
    await waitFor(() => settings.querySelectorAll('dl-settings-memory li').length === 30);
    const body = settings.querySelector<HTMLElement>('[data-page-body]')!;
    const scroller = settings.querySelector<HTMLElement>('dl-settings-memory [data-memory-window]')!;
    const title = settings.querySelector<HTMLElement>('#memory-list-title')!;
    const first = settings.querySelector<HTMLElement>('dl-settings-memory li')!;
    const clear = buttonNamed<HTMLButtonElement>(settings, 'Clear all memories')!;
    const loadMore = settings.querySelector<HTMLButtonElement>('[data-load-older="memories"]')!;
    const box = (element: Element): DOMRect => element.getBoundingClientRect();

    // The pane and the dialog are the window's frame, never its scroller; the card's own box is,
    // and a keyboard reaches it as a named region.
    expect(dialog.scrollHeight).to.equal(dialog.clientHeight);
    expect(body.scrollHeight).to.equal(body.clientHeight);
    expect(scroller.scrollHeight).to.be.greaterThan(scroller.clientHeight);
    expect(scroller.getAttribute('role')).to.equal('region');
    expect(scroller.getAttribute('aria-labelledby')).to.equal('memory-list-title');
    expect(scroller.tabIndex).to.equal(0);
    const still = {title: box(title).top, clear: box(clear).top, loadMore: box(loadMore).top};

    scroller.scrollTop = 200;
    expect(scroller.scrollTop).to.be.greaterThan(0);
    // The rows pass under the title while it, the Clear action and the next-page button hold still,
    // and the Clear action stays inside the dialog rather than below its edge.
    expect(box(first).top).to.be.lessThan(box(title).top);
    expect(box(title).top).to.equal(still.title);
    expect(box(clear).top).to.equal(still.clear);
    expect(box(loadMore).top).to.equal(still.loadMore);
    expect(box(clear).bottom).to.be.at.most(box(dialog).bottom + 0.5);
    expect(box(loadMore).bottom).to.be.at.most(box(dialog).bottom + 0.5);

    // This page's top is its window's, so it opens on the newest memories again.
    row(settings, 'language').click();
    await settings.updateComplete;
    row(settings, 'memory').click();
    await waitFor(() => scroller.scrollTop === 0);
  });

  it('lays each memory out as its own words, with the kind at the row\'s tail', async () => {
    window.fetch = wire({
      'GET /web/api/memory/settings': () => memorySettings(true, 3),
      'GET /web/api/memory': () => memoryPage([
        {id: 'one',
          body: 'The owner expects a clickable link (and the workspace path) whenever the Agent says a deliverable has been published or attached, rather than a bare claim of publication.'},
        {id: 'two', kind: 'fact', body: 'Prefers concise answers.'},
        {id: 'three', kind: 'experience', body: 'Two lines of a memory.\nAnd a second one.'},
      ]),
    }).fetch;
    const {settings} = mountSettings();
    await openSettings(settings, 'memory');
    await waitFor(() => settings.querySelectorAll('dl-settings-memory li').length === 3);
    const items = [...settings.querySelectorAll<HTMLElement>('dl-settings-memory li')];
    const part = (item: HTMLElement, name: string): HTMLElement => item.querySelector(`[class*=${name}]`)!;
    const box = (element: Element): DOMRect => element.getBoundingClientRect();
    const line = Number.parseFloat(getComputedStyle(part(items[0]!, 'text')).lineHeight);

    for (const item of items) {
      const text = box(part(item, 'text'));
      const kind = box(part(item, 'kind'));
      // The memory's own words come first, at the row's left edge, and the kind stands after them
      // at the row's tail rather than before them.
      expect(text.left - box(item).left).to.be.at.most(13);
      expect(kind.left).to.be.at.least(text.right - 1);
      expect(kind.right).to.be.at.most(box(item).right - 13);
      // The kind is no caption above the words: the row is the words, the padding around them, and
      // at most the control that sits beside a short line.
      expect(box(item).height - text.height).to.be.at.most(34);
    }
    // A short memory is a short row, and a second line of words costs a line rather than a card.
    expect(box(items[1]!).height).to.be.at.most(56);
    expect(box(items[0]!).height - box(items[2]!).height).to.be.closeTo(line, 2);
  });
});
