// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Typographic and geometric invariants of the shared control primitives. */

import {expect} from '@esm-bundle/chai';

const stylesheets = [
  '../tokens/utopia.css',
  '../styles/global.css',
  '../styles/primitives.css',
];

before(async () => {
  await Promise.all(stylesheets.map(async (href) => new Promise<void>((resolve, reject) => {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = new URL(href, import.meta.url).href;
    link.addEventListener('load', () => { resolve(); }, {once: true});
    link.addEventListener('error', () => { reject(new Error(`could not load ${href}`)); }, {once: true});
    document.head.appendChild(link);
  })));
});

function fixture(className: string, content: string): HTMLElement {
  const node = document.createElement('div');
  node.className = className;
  node.innerHTML = content;
  document.body.appendChild(node);
  return node;
}

function centeredDelta(el: Element): {x: number; y: number} {
  const rect = el.getBoundingClientRect();
  return {
    x: Math.abs(window.innerWidth / 2 - (rect.left + rect.width / 2)),
    y: Math.abs(window.innerHeight / 2 - (rect.top + rect.height / 2)),
  };
}

afterEach(() => {
  document.body.replaceChildren();
});

it('a modal confirm dialog centers on both axes', () => {
  const dialog = document.createElement('dialog');
  dialog.className = 'confirm-dialog';
  dialog.innerHTML = '<form method="dialog"><h2>Delete?</h2><p>body</p></form>';
  document.body.appendChild(dialog);
  dialog.showModal();

  const delta = centeredDelta(dialog);
  expect(delta.x).to.be.lessThanOrEqual(1);
  expect(delta.y).to.be.lessThanOrEqual(1);
  dialog.close();
});

it('confirm-dialog checkbox labels match the body typography exactly', () => {
  const dialog = fixture('confirm-dialog', `
    <p class="confirm-body">Remembered preferences and facts will be forgotten.</p>
    <label class="ui-dialog-checkbox"><input type="checkbox" /> Also clear Profile memory</label>
  `);
  const body = dialog.querySelector<HTMLElement>('.confirm-body')!;
  const label = dialog.querySelector<HTMLElement>('.ui-dialog-checkbox')!;
  const bodyStyle = getComputedStyle(body);
  const labelStyle = getComputedStyle(label);

  expect(labelStyle.fontSize).to.equal(bodyStyle.fontSize);
  expect(labelStyle.color).to.equal(bodyStyle.color);
  expect(labelStyle.fontFamily).to.equal(bodyStyle.fontFamily);
});

it('confirm-dialog checkbox input centers against its label line', () => {
  const dialog = fixture('confirm-dialog', `
    <label class="ui-dialog-checkbox"><input type="checkbox" /> Also clear Profile memory</label>
  `);
  const input = dialog.querySelector<HTMLInputElement>('input')!;
  const label = dialog.querySelector<HTMLElement>('.ui-dialog-checkbox')!;
  const inputRect = input.getBoundingClientRect();
  const labelRect = label.getBoundingClientRect();
  const inputCenter = inputRect.top + inputRect.height / 2;
  const labelCenter = labelRect.top + labelRect.height / 2;

  expect(Math.abs(inputCenter - labelCenter)).to.be.lessThanOrEqual(2);
});

it('dialog radio inputs use the active theme accent', () => {
  const dialog = fixture('confirm-dialog', `
    <label class="ui-dialog-checkbox"><input type="radio" checked /> Automatic</label>
  `);
  const input = dialog.querySelector<HTMLInputElement>('input')!;
  const accentProbe = document.createElement('span');
  accentProbe.style.color = 'var(--color-accent-action)';
  dialog.appendChild(accentProbe);

  expect(getComputedStyle(input).accentColor).to.equal(getComputedStyle(accentProbe).color);
});
