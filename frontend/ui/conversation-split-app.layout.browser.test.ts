// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {setViewport} from '@web/test-runner-commands';
import {type DlSplitLayout, defineDesignSystemElements } from '../design-system/index.ts';
import './app.ts';
import type {DlApp} from './app.ts';
import {initializeBrowserAdapters} from './browser-adapters.ts';
import type {DlConversationSidebar} from './conversation-sidebar.ts';
import type {DlInspector} from './inspector.ts';
import {setLanguagePreference} from '../i18n/locale.ts';
import {waitFor} from '../testing/dom.ts';
import {bootstrapResponse} from '../testing/app.ts';

defineDesignSystemElements();

after(() => {
  document.body.replaceChildren();
  document.body.className = '';
});

let mounted: Promise<DlApp> | null = null;

/** The one shell this file mounts: the split adapter binds to the first one for the life of the page. */
function mountDesktopShell(): Promise<DlApp> {
  mounted ??= mountShell();
  return mounted;
}

async function mountShell(): Promise<DlApp> {
  await setViewport({width: 1440, height: 900});
  window.fetch = async (input) => bootstrapResponse(input);

  const app = document.createElement('dl-app') as DlApp;
  app.style.display = 'block';
  app.style.width = '1440px';
  app.style.height = '900px';
  document.body.append(app);
  await app.ready;
  await initializeBrowserAdapters(app);
  await app.querySelector('dl-conversation-sidebar')?.updateComplete;
  await new Promise((resolve) => requestAnimationFrame(resolve));
  return app;
}

/** A pointer drag of a divider, as the pointer reports it: down at the first x, on to each of the rest, up at the last. */
function drag(split: DlSplitLayout, [from, ...path]: readonly number[], {release = true} = {}): void {
  const report = (type: string, clientX: number): void => {
    split.divider.dispatchEvent(new PointerEvent(type, {bubbles: true, button: 0, clientX, pointerId: 21}));
  };
  report('pointerdown', from);
  for (const x of path) report('pointermove', x);
  if (release) report('pointerup', path.at(-1) ?? from);
}

it('composed desktop shell exposes a hittable conversation divider', async () => {
  const app = await mountDesktopShell();
  const split = app.querySelector<DlSplitLayout>('#conversation-split')!;
  const sidebar = app.querySelector('#conversation-sidebar')!;
  const dump = {
    disabled: split.disabled,
    collapsed: split.hasAttribute('data-collapsed'),
    size: split.size,
    openClass: sidebar.classList.contains('open'),
    pointerEvents: getComputedStyle(split.divider).pointerEvents,
    label: split.divider.getAttribute('aria-label'),
    bodyOpen: document.body.classList.contains('conversation-sidebar-open'),
  };
  expect(dump, JSON.stringify(dump)).to.include({
    disabled: false,
    collapsed: false,
    openClass: true,
  });
  expect(split.size).to.be.at.least(240);
  expect(getComputedStyle(split.divider).pointerEvents).to.not.equal('none');

  const before = split.size;
  split.divider.focus();
  split.divider.dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowRight', bubbles: true}));
  expect(split.size).to.be.greaterThan(before);
});

it('opens the Inspector at its share of the panel split beside the conversations', async () => {
  const app = await mountDesktopShell();
  const split = app.querySelector<DlSplitLayout>('#panel-split')!;
  const inspector = app.querySelector<DlInspector>('dl-inspector')!;
  const expected = Math.min(560, Math.round(0.38 * split.clientWidth));

  await inspector.openFiles();
  await waitFor(() => document.body.classList.contains('files-panel-open'));
  expect(split.size, 'Files opens at 38% of the panel split, not the window').to.equal(expected);

  inspector.close(false);
  await waitFor(() => !document.body.classList.contains('files-panel-open'));
  expect(split.size, 'and the width it opened at does not outlive it').to.equal(0);
});

it('names each divider in the language the shell is showing', async () => {
  const app = await mountDesktopShell();
  const labels = (): (string | null)[] => ['#conversation-split', '#panel-split', '#artifact-canvas-split']
    .map((id) => app.querySelector<DlSplitLayout>(id)!.divider.getAttribute('aria-label'));

  expect(labels()).to.deep.equal(['Resize conversations', 'Resize side panel', 'Resize file preview']);
  try {
    await setLanguagePreference('zh');
    await app.updateComplete;
    expect(labels()).to.deep.equal(['调整会话列表的宽度', '调整侧边面板的宽度', '调整文件预览的宽度']);
  } finally {
    await setLanguagePreference('en');
  }
});

// The two ways a reader shuts the conversations from the divider end in the same place, the panel put
// away. What differs is whether they were holding the divider when it went, which decides where focus
// belongs: a divider that leaves cannot keep it, and a pointer drag has no business taking it. Each test
// reads what happened, then puts the shell it shares back as it found it, so a failure stays its own.
it('puts the conversations away on Enter and hands focus to the control that brings them back', async () => {
  const app = await mountDesktopShell();
  const split = app.querySelector<DlSplitLayout>('#conversation-split')!;
  const sidebar = app.querySelector<DlConversationSidebar>('dl-conversation-sidebar')!;
  const opener = app.querySelector<HTMLButtonElement>('#conversation-sidebar-open')!;
  const width = split.size;

  split.divider.focus();
  split.divider.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', bubbles: true}));
  await waitFor(() => split.hasAttribute('data-collapsed'));
  const open = document.body.classList.contains('conversation-sidebar-open');
  const focused = document.activeElement?.id;

  await sidebar.open(opener);
  expect(open, 'the panel is put away').to.equal(false);
  expect(focused, 'and focus does not stay on a divider that has left').to.equal('conversation-sidebar-open');
  expect(split.size, 'the width it had is the width it comes back at').to.equal(width);
});

it('leaves focus where it was when a pointer drag puts the conversations away', async () => {
  const app = await mountDesktopShell();
  const split = app.querySelector<DlSplitLayout>('#conversation-split')!;
  const sidebar = app.querySelector<DlConversationSidebar>('dl-conversation-sidebar')!;
  const opener = app.querySelector<HTMLButtonElement>('#conversation-sidebar-open')!;
  app.querySelector<HTMLButtonElement>('#files-btn')!.focus();

  // The divider is the sidebar's right edge; taking the pointer to the window's left edge is past any threshold.
  drag(split, [split.size, 0]);
  await waitFor(() => split.hasAttribute('data-collapsed'));
  const open = document.body.classList.contains('conversation-sidebar-open');
  const focused = document.activeElement?.id;

  await sidebar.open(opener);
  expect(open, 'the panel is put away').to.equal(false);
  expect(focused, 'the reader was not on the divider, so nothing is taken from them').to.equal('files-btn');
});

it('asks for the default width on a double-click, wherever the reader has dragged it', async () => {
  const app = await mountDesktopShell();
  const split = app.querySelector<DlSplitLayout>('#conversation-split')!;
  const sidebar = app.querySelector<DlConversationSidebar>('dl-conversation-sidebar')!;
  const opener = app.querySelector<HTMLButtonElement>('#conversation-sidebar-open')!;
  const stored = (): string | null => window.localStorage.getItem('dlightrag-conversation-sidebar-width');
  const doubleClick = (): boolean => split.divider.dispatchEvent(new MouseEvent('dblclick', {bubbles: true}));
  // A default is a share of the viewport the conversation split spans (ADR 0067), not a token.
  const listDefault = Math.round(0.18 * split.clientWidth);

  drag(split, [split.size, 340]);
  const dragged = split.size;
  doubleClick();
  const reset = split.size;
  const kept = stored();

  // The same from a panel that was put away and brought back.
  split.divider.focus();
  split.divider.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', bubbles: true}));
  await waitFor(() => split.hasAttribute('data-collapsed'));
  await sidebar.open(opener);
  drag(split, [split.size, 340]);
  const draggedAgain = split.size;
  doubleClick();

  expect(dragged, 'the drag took it there').to.equal(340);
  expect(reset, 'a double-click asks for the share of the viewport it opens at, not the one it has').to.equal(listDefault);
  expect(kept, 'and that is stored as a drag stores its width').to.equal(String(listDefault));
  expect(draggedAgain, 'a reopened panel is dragged as any other').to.equal(340);
  expect(split.size, 'and reset as any other').to.equal(listDefault);
  expect(stored()).to.equal(String(listDefault));
});

it('cancels a pointer drag on Escape before the panel that closes itself on it can answer', async () => {
  const app = await mountDesktopShell();
  const split = app.querySelector<DlSplitLayout>('#panel-split')!;
  const sidebar = app.querySelector<DlConversationSidebar>('dl-conversation-sidebar')!;
  const opener = app.querySelector<HTMLButtonElement>('#conversation-sidebar-open')!;
  const inspector = app.querySelector('dl-inspector')!;
  const stored = (): string | null => window.localStorage.getItem('dlightrag-panel-width');

  // A width the window cannot always show: the Inspector is dragged wide while the conversations are
  // put away, and then they come back, so what is drawn is less than what the reader chose.
  sidebar.collapse(false);
  await waitFor(() => !document.body.classList.contains('conversation-sidebar-open'));
  await inspector.openSources({
    answerText: '', parts: [], sources: [], evidenceImages: [], linkCards: [], artifacts: [],
    artifactOutcome: {status: 'complete', issues: []},
  });
  drag(split, [1000, 1000 - (800 - split.size)]);
  const chosen = split.size;
  await sidebar.open(opener);
  const start = split.size;
  const clamped = {drawn: start, stored: stored()};

  // What a panel that listens for Escape on the page, as the Inspector does, would be told.
  let heard = 0;
  const listener = (): void => { heard += 1; };
  document.addEventListener('keydown', listener);
  const press = (init: KeyboardEventInit = {}): void => {
    document.body.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true, cancelable: true, ...init}));
  };
  const handled = (event: Event): void => event.preventDefault();

  // The Inspector is the end pane: the pointer going right narrows it. The button stays down.
  drag(split, [1000, 1080], {release: false});
  const narrowed = split.size;

  // The Escape that cancels an IME composition is the composition's, and one already answered is not asked twice.
  press({isComposing: true});
  press({keyCode: 229});
  window.addEventListener('keydown', handled, {capture: true});
  press();
  window.removeEventListener('keydown', handled, {capture: true});
  const left = {width: split.size, heard};

  press();
  const cancelled = {width: split.size, open: inspector.open, heard, stored: stored()};
  split.divider.dispatchEvent(new PointerEvent('pointermove', {clientX: 1120, pointerId: 21}));
  const followed = split.size;
  // Let go of the button, as the page sees it once the divider no longer holds the pointer.
  document.body.dispatchEvent(new PointerEvent('pointerup', {bubbles: true, composed: true, clientX: 1120, pointerId: 21}));
  const released = {width: split.size, stored: stored(), resizing: document.body.hasAttribute('data-resizing')};

  // With no drag going on, Escape is the panel's again.
  press();
  const plain = {open: inspector.open, heard};
  document.removeEventListener('keydown', listener);
  inspector.close(false);

  expect(start, 'the window shows less than the reader chose').to.be.below(chosen);
  expect(clamped.stored, 'and that drawing it so did not change what they chose').to.equal(String(chosen));
  expect(narrowed, 'the drag narrowed it').to.be.below(start);
  expect(left, 'those Escapes leave the drag alone').to.deep.equal({width: narrowed, heard: 3});
  expect(cancelled, 'Escape restores the width the drag began at, and the panel stays, unheard, with the choice kept')
    .to.deep.equal({width: start, open: true, heard: 3, stored: String(chosen)});
  expect(followed, 'a cancelled drag does not follow the pointer on').to.equal(start);
  expect(released, 'nor does letting go, which ends the resizing the page was told of')
    .to.deep.equal({width: start, stored: String(chosen), resizing: false});
  expect(plain, 'and a later Escape is heard again, and closes the Inspector').to.deep.equal({open: false, heard: 4});
});
