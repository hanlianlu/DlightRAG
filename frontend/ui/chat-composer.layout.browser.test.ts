// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The composer's pickers as the shipped stylesheets draw them (the product-styles page of the runner): how
 *  their triggers and rows answer the pointer and the keyboard, and how an effort row is laid out. */

import {expect} from '@esm-bundle/chai';
import {resetMouse, sendMouse} from '@web/test-runner-commands';
import {defineDesignSystemElements} from '../design-system/index.ts';
import type {DlChatComposer} from './chat-composer.ts';
import './chat-composer.ts';

defineDesignSystemElements();

const CLEAR = 'rgba(0, 0, 0, 0)';

async function mount(): Promise<DlChatComposer> {
  const composer = document.createElement('dl-chat-composer') as DlChatComposer;
  composer.agentEffortOffer = {levels: ['low', 'high', 'max'], default: 'high'};
  // The pickers open upwards from a composer at the foot of the page, and need the room above it.
  const foot = document.createElement('div');
  foot.style.marginTop = '320px';
  foot.append(composer);
  document.body.append(foot);
  await composer.updateComplete;
  return composer;
}

/** What a token paints, read from a probe: the shipped sheet is minified, so no token text is compared. */
function token(property: 'color' | 'backgroundColor', name: string): string {
  const probe = document.createElement('div');
  probe.style[property] = `var(${name})`;
  document.body.append(probe);
  const value = getComputedStyle(probe)[property];
  probe.remove();
  return value;
}

async function hover(node: Element): Promise<void> {
  const box = node.getBoundingClientRect();
  await sendMouse({type: 'move', position: [Math.round(box.left + box.width / 2), Math.round(box.top + box.height / 2)]});
  await Promise.all(node.getAnimations({subtree: true}).map((animation) => animation.finished));
}

/** Programmatic focus after no pointer press is keyboard focus to :focus-visible, as in a real Tab. */
function focusByKeyboard(node: HTMLElement): CSSStyleDeclaration {
  node.focus();
  expect(node.matches(':focus-visible'), 'focus-visible').to.equal(true);
  return getComputedStyle(node);
}

function expectRing(style: CSSStyleDeclaration): void {
  expect(style.outlineStyle).to.equal('solid');
  expect(style.outlineWidth).to.equal('2px');
  expect(style.outlineColor).to.equal(token('color', '--color-control-ring'));
}

afterEach(async () => {
  document.body.replaceChildren();
  await resetMouse();
});

for (const picker of [
  {name: 'answer mode', trigger: '#composer-mode', menu: '#composer-mode-menu', row: '[data-mode]'},
  {name: 'agent effort', trigger: '#composer-effort', menu: '#composer-effort-menu', row: '[data-effort]'},
]) {
  describe(`the ${picker.name} picker`, () => {
    it('lights its trigger under the pointer and rings it on keyboard focus', async () => {
      const composer = await mount();
      const trigger = composer.querySelector<HTMLButtonElement>(picker.trigger)!;
      expect(getComputedStyle(trigger).backgroundColor).to.equal(CLEAR);

      await hover(trigger);
      expect(getComputedStyle(trigger).backgroundColor).to.equal(token('backgroundColor', '--color-bg-hover'));
      await resetMouse();
      expectRing(focusByKeyboard(trigger));
    });

    it('lights a row under the pointer and rings it on focus, and the chosen row keeps its own surface', async () => {
      const composer = await mount();
      composer.querySelector<HTMLButtonElement>(picker.trigger)!.click();
      await composer.updateComplete;
      const rows = [...composer.querySelectorAll<HTMLButtonElement>(`${picker.menu} ${picker.row}`)];
      const chosen = rows.find((row) => row.getAttribute('aria-checked') === 'true')!;
      // The menu opens on its first row, which keeps focus; the pointer goes to a row that has neither.
      const other = rows.filter((row) => row !== chosen).at(-1)!;
      const elevated = token('backgroundColor', '--color-bg-elevated');

      expect(getComputedStyle(other).backgroundColor).to.equal(CLEAR);
      expect(getComputedStyle(chosen).backgroundColor, 'the chosen row is set apart at rest').to.equal(elevated);
      await hover(other);
      expect(getComputedStyle(other).backgroundColor).to.equal(token('backgroundColor', '--color-bg-hover'));
      await hover(chosen);
      expect(getComputedStyle(chosen).backgroundColor, 'and under the pointer').to.equal(elevated);
      await resetMouse();
      expectRing(focusByKeyboard(other));
    });
  });
}

it('lights the attach control under the pointer', async () => {
  const composer = await mount();
  const attach = composer.querySelector<HTMLButtonElement>('.composer-plus')!;
  expect(getComputedStyle(attach).backgroundColor).to.equal(CLEAR);
  await hover(attach);
  expect(getComputedStyle(attach).backgroundColor).to.equal(token('backgroundColor', '--color-bg-hover'));
});

describe('an effort row', () => {
  async function openEffortMenu() {
    const composer = await mount();
    composer.querySelector<HTMLButtonElement>('#composer-effort')!.click();
    await composer.updateComplete;
    const menu = composer.querySelector<HTMLElement>('#composer-effort-menu')!;
    return {composer, menu, row: menu.querySelector<HTMLElement>('[data-effort="high"]')!};
  }

  it('keeps the Default marker beside the label it qualifies, however wide the menu is made', async () => {
    const {menu, row} = await openEffortMenu();
    const label = row.querySelector(':scope > span:not(.composer-effort-default)')!;
    const marker = row.querySelector('.composer-effort-default')!;
    // The words, not the boxes they sit in: a track that grew would carry the box and leave the words behind.
    const words = (node: Element): DOMRect => {
      const range = document.createRange();
      range.selectNodeContents(node);
      return range.getBoundingClientRect();
    };
    const gap = (): number => words(marker).left - words(label).right;

    // Content-sized tracks only: free space never separates a qualifier from what it describes.
    const natural = gap();
    expect(natural, 'the marker follows its label').to.be.within(0, 24);
    menu.style.minWidth = '420px';
    expect(gap(), 'a wider menu leaves them where they were').to.be.closeTo(natural, 0.5);
    expect(row.getBoundingClientRect().width, 'the row fills the wider menu').to.be.greaterThan(380);
  });

  it('shares the mode picker\'s menu width instead of widening its own', async () => {
    const {composer, menu} = await openEffortMenu();
    const effort = {width: menu.getBoundingClientRect().width, minWidth: getComputedStyle(menu).minWidth};
    composer.querySelector<HTMLButtonElement>('#composer-mode')!.click();
    await composer.updateComplete;
    const mode = composer.querySelector<HTMLElement>('#composer-mode-menu')!;

    expect(mode.hidden).to.equal(false);
    expect(effort.width).to.be.closeTo(mode.getBoundingClientRect().width, 24);
    expect(effort.minWidth).to.equal(getComputedStyle(mode).minWidth);
  });
});
