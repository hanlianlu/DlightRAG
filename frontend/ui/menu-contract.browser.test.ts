// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Every menu button in the product answers the same keys the same way. */

import {expect} from '@esm-bundle/chai';
import {defineDesignSystemElements} from '../design-system/index.ts';
import {productionHandles} from '../stores/app-handles.ts';
import type {DlChatComposer} from './chat-composer.ts';
import './chat-composer.ts';
import type {DlConversationList} from './conversation-list.ts';
import './conversation-list.ts';
import type {DlThemeControl} from './theme.ts';
import './theme.ts';

defineDesignSystemElements();

const {conversations} = productionHandles();
const originalMatchMedia = window.matchMedia;

interface MenuButton {
  trigger: HTMLElement;
  /** The open menu, or null once it is closed. */
  menu: () => HTMLElement | null;
  labels: readonly string[];
  /** A letter that only the last item's label starts with. */
  lastInitial: string;
}

async function frame(): Promise<void> {
  await new Promise<void>((resolve) => { requestAnimationFrame(() => { resolve(); }); });
  await new Promise<void>((resolve) => { setTimeout(resolve, 0); });
}

async function press(key: string, target = document.activeElement as HTMLElement): Promise<void> {
  target.dispatchEvent(new KeyboardEvent('keydown', {key, bubbles: true, composed: true, cancelable: true}));
  await frame();
}

function focusedLabel(): string {
  const active = document.activeElement as HTMLElement | null;
  return (active?.getAttribute('aria-label') ?? active?.textContent ?? '').replace(/\s+/g, ' ').trim();
}

function open(menu: HTMLElement | null): boolean {
  return Boolean(menu?.isConnected) && !menu?.hidden;
}

async function themeMenu(): Promise<MenuButton> {
  window.matchMedia = (() => ({
    matches: false, media: '', onchange: null, addListener() {}, removeListener() {},
    addEventListener() {}, removeEventListener() {}, dispatchEvent: () => true,
  })) as typeof window.matchMedia;
  // A checked item away from the top proves opening does not jump to it.
  localStorage.setItem('dlightrag-theme', 'dark');
  const control = document.createElement('dl-theme-control') as DlThemeControl;
  document.body.appendChild(control);
  await control.updateComplete;
  return {
    trigger: control.querySelector<HTMLElement>('#theme-trigger')!,
    menu: () => control.querySelector<HTMLElement>('#theme-menu'),
    labels: ['System', 'Light', 'Dark'],
    lastInitial: 'd',
  };
}

async function composerMenu(picker: 'mode' | 'effort'): Promise<MenuButton> {
  localStorage.setItem('dlightrag.answerMode', 'research');
  const composer = document.createElement('dl-chat-composer') as DlChatComposer;
  composer.agentEffortOffer = {levels: ['low', 'high', 'max'], default: 'high'};
  document.body.appendChild(composer);
  await composer.updateComplete;
  return {
    trigger: composer.querySelector<HTMLElement>(`.composer-${picker}-trigger`)!,
    menu: () => composer.querySelector<HTMLElement>(`#composer-${picker}-menu`),
    labels: picker === 'mode' ? ['Auto', 'Fast', 'Research'] : ['Low', 'High Default', 'Max'],
    lastInitial: picker === 'mode' ? 'r' : 'm',
  };
}

async function conversationActions(): Promise<MenuButton> {
  conversations.upsertSummary({
    conversationId: 'menu-contract',
    title: 'Menu contract',
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z',
    forkedFromConversationId: null,
    forkedFromTitle: null,
  });
  const list = document.createElement('dl-conversation-list') as DlConversationList;
  document.body.appendChild(list);
  await list.updateComplete;
  return {
    trigger: list.querySelector<HTMLElement>('.conversation-actions-button')!,
    menu: () => list.querySelector<HTMLElement>('dl-menu'),
    labels: ['Rename', 'Delete'],
    lastInitial: 'd',
  };
}

afterEach(() => {
  window.matchMedia = originalMatchMedia;
  localStorage.removeItem('dlightrag-theme');
  localStorage.removeItem('dlightrag.answerMode');
  conversations.dispose();
  document.body.replaceChildren();
});

const MENU_BUTTONS: ReadonlyArray<[string, () => Promise<MenuButton>]> = [
  ['theme', themeMenu],
  ['answer mode', () => composerMenu('mode')],
  ['agent effort', () => composerMenu('effort')],
  ['conversation actions', conversationActions],
];

for (const [name, mount] of MENU_BUTTONS) {
  it(`${name} menu follows the shared keyboard contract`, async () => {
    const {trigger, menu, labels, lastInitial} = await mount();
    const first = labels[0]!;
    const second = labels[1]!;
    const last = labels.at(-1)!;
    expect(open(menu())).to.equal(false);

    await press('ArrowDown', trigger);
    expect(open(menu()), 'ArrowDown opens the menu').to.equal(true);
    expect(focusedLabel(), 'ArrowDown opens on the first item').to.equal(first);
    await press('ArrowDown');
    expect(focusedLabel()).to.equal(second);
    await press('End');
    expect(focusedLabel()).to.equal(last);
    await press('ArrowDown');
    expect(focusedLabel(), 'ArrowDown wraps').to.equal(first);
    await press('ArrowUp');
    expect(focusedLabel(), 'ArrowUp wraps').to.equal(last);
    await press('Home');
    expect(focusedLabel()).to.equal(first);
    await press(lastInitial);
    expect(focusedLabel(), 'a typed letter moves to the item it starts').to.equal(last);
    await press('Escape');
    expect(open(menu()), 'Escape closes the menu').to.equal(false);
    expect(document.activeElement, 'Escape returns focus to the button').to.equal(trigger);

    await press('ArrowUp', trigger);
    expect(focusedLabel(), 'ArrowUp opens on the last item').to.equal(last);
    await press('Escape');
    await press('Enter', trigger);
    expect(focusedLabel(), 'Enter opens on the first item').to.equal(first);
    await press('Escape');
    expect(document.activeElement).to.equal(trigger);

    trigger.click();
    await frame();
    expect(focusedLabel(), 'a click opens on the first item too').to.equal(first);
    await press('Escape');
    expect(document.activeElement).to.equal(trigger);
  });
}

for (const [name, mount] of MENU_BUTTONS) {
  it(`${name} menu closes when focus leaves it, and only Escape sends focus back`, async () => {
    const {trigger, menu} = await mount();
    const elsewhere = document.createElement('button');
    elsewhere.textContent = 'Elsewhere';
    document.body.append(elsewhere);

    await press('ArrowDown', trigger);
    expect(open(menu())).to.equal(true);
    const items = [...menu()!.querySelectorAll<HTMLElement>('[role^="menuitem"]')];
    expect(items.map((item) => item.tabIndex), 'Tab never lands on an item').to.deep.equal(items.map(() => -1));
    await press('Tab');
    expect(open(menu()), 'Tab closes the menu').to.equal(false);
    expect(document.activeElement === trigger, 'Tab does not send focus back to the button')
      .to.equal(false);

    await press('ArrowDown', trigger);
    expect(open(menu())).to.equal(true);
    elsewhere.focus();
    await frame();
    expect(open(menu()), 'focus leaving closes the menu').to.equal(false);
    expect(document.activeElement === elsewhere, 'and focus stays where it went').to.equal(true);

    await press('ArrowDown', trigger);
    expect(open(menu())).to.equal(true);
    // A press on the button focuses it before its click: the click, not the focus, closes the menu.
    trigger.focus();
    await frame();
    expect(open(menu()), 'focus on the menu\'s own button keeps it open').to.equal(true);
    trigger.click();
    await frame();
    expect(open(menu()), 'the button\'s click closes it').to.equal(false);
  });
}

it('keeps a busy conversation\'s Delete in reach but inert', async () => {
  const {trigger, menu} = await conversationActions();
  const list = trigger.closest<DlConversationList>('dl-conversation-list')!;
  const deletes: string[] = [];
  list.addEventListener('dl-conversation-delete', (event) => { deletes.push(event.detail.conversationId); });
  list.busy = true;
  await list.updateComplete;

  await press('ArrowUp', trigger);
  expect(focusedLabel(), 'the busy item still takes focus').to.equal('Delete');
  const remove = document.activeElement as HTMLButtonElement;
  expect(remove.getAttribute('aria-disabled')).to.equal('true');
  expect(remove.disabled).to.equal(false);
  remove.click();
  await frame();
  expect(deletes).to.deep.equal([]);
  expect(open(menu()), 'an inert item leaves the menu open').to.equal(true);

  list.busy = false;
  await list.updateComplete;
  expect(remove.hasAttribute('aria-disabled')).to.equal(false);
  remove.click();
  await frame();
  expect(deletes).to.deep.equal(['menu-contract']);
});
