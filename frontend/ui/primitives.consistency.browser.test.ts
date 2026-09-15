// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Typographic and geometric invariants of the shared control primitives. */

import {expect} from '@esm-bundle/chai';
import {render} from 'lit';
import {icon} from '../design-system/index.ts';

const stylesheets = [
  '../design-system/index.css',
  '../styles/global.css',
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
    <label class="dl-dialog-checkbox"><input type="checkbox" /> Also clear Profile memory</label>
  `);
  const body = dialog.querySelector<HTMLElement>('.confirm-body')!;
  const label = dialog.querySelector<HTMLElement>('.dl-dialog-checkbox')!;
  const bodyStyle = getComputedStyle(body);
  const labelStyle = getComputedStyle(label);

  expect(labelStyle.fontSize).to.equal(bodyStyle.fontSize);
  expect(labelStyle.color).to.equal(bodyStyle.color);
  expect(labelStyle.fontFamily).to.equal(bodyStyle.fontFamily);
});

it('confirm-dialog checkbox input centers against its label line', () => {
  const dialog = fixture('confirm-dialog', `
    <label class="dl-dialog-checkbox"><input type="checkbox" /> Also clear Profile memory</label>
  `);
  const input = dialog.querySelector<HTMLInputElement>('input')!;
  const label = dialog.querySelector<HTMLElement>('.dl-dialog-checkbox')!;
  const inputRect = input.getBoundingClientRect();
  const labelRect = label.getBoundingClientRect();
  const inputCenter = inputRect.top + inputRect.height / 2;
  const labelCenter = labelRect.top + labelRect.height / 2;

  expect(Math.abs(inputCenter - labelCenter)).to.be.lessThanOrEqual(2);
});

it('switch foundations satisfy both symmetry invariants', () => {
  const token = (name: string): number => Number.parseFloat(
    getComputedStyle(document.documentElement).getPropertyValue(name),
  );

  // The compact variant is the same control one icon step down, so both sizes prove out.
  // The variant composes with the base class, exactly as the product markup uses it.
  for (const [className, suffix] of [
    ['dl-switch', ''],
    ['dl-switch dl-switch--sm', '-sm'],
  ] as const) {
    const width = token(`--size-switch${suffix}-width`);
    const height = token(`--size-switch${suffix}-height`);
    const thumb = token(`--size-switch${suffix}-thumb`);
    const inset = token(`--size-switch${suffix}-inset`);

    expect(height).to.equal(thumb + 2 * inset);

    // The travel token is a calc() and stays uncomputed in getComputedStyle, so prove the rendered
    // displacement against the same invariant instead of reading the token text.
    const button = document.createElement('button');
    button.className = className;    button.setAttribute('role', 'switch');
    button.setAttribute('aria-checked', 'true');
    document.body.appendChild(button);
    const rendered = getComputedStyle(button, '::after').transform;
    const travel = Number.parseFloat(/matrix\(1, 0, 0, 1, ([\d.]+),/.exec(rendered)?.[1] ?? 'NaN');

    expect(travel).to.equal(width - thumb - 2 * inset);
  }
});

it('a switch thumb sits at the same inset inside its track in both states', () => {
  const control = (checked: boolean): HTMLButtonElement => {
    const button = document.createElement('button');
    button.className = 'dl-switch';
    button.setAttribute('role', 'switch');
    button.setAttribute('aria-checked', String(checked));
    document.body.appendChild(button);
    return button;
  };
  const off = control(false);
  const on = control(true);
  const track = off.getBoundingClientRect();
  const settled = getComputedStyle(off);
  const thumb = getComputedStyle(off, '::after');
  const border = Number.parseFloat(settled.borderTopWidth);
  const inset = border + Number.parseFloat(thumb.marginInlineStart);
  const size = Number.parseFloat(thumb.width);

  // `align-items: center` must place the thumb at the same inset the track reserves on its ends.
  expect(Math.abs((track.height - 2 * border - size) / 2 - Number.parseFloat(thumb.marginInlineStart)))
    .to.be.lessThanOrEqual(0.5);

  const travel = (button: HTMLButtonElement): number => {
    const transform = getComputedStyle(button, '::after').transform;
    return Number.parseFloat(/matrix\(1, 0, 0, 1, ([\d.]+),/.exec(transform)?.[1] ?? '0');
  };
  const leftGap = (button: HTMLButtonElement): number => border
    + Number.parseFloat(getComputedStyle(button, '::after').marginInlineStart) + travel(button);

  expect(travel(off)).to.equal(0);
  expect(Math.abs(leftGap(off) - inset)).to.be.lessThanOrEqual(0.5);
  expect(travel(on)).to.be.greaterThan(0);
  expect(Math.abs((track.width - leftGap(on) - size) - inset)).to.be.lessThanOrEqual(0.5);
});

it('dialog radio inputs use the active theme accent', () => {
  const dialog = fixture('confirm-dialog', `
    <label class="dl-dialog-checkbox"><input type="radio" checked /> Automatic</label>
  `);
  const input = dialog.querySelector<HTMLInputElement>('input')!;
  const accentProbe = document.createElement('span');
  accentProbe.style.color = 'var(--color-accent-action)';
  dialog.appendChild(accentProbe);

  expect(getComputedStyle(input).accentColor).to.equal(getComputedStyle(accentProbe).color);
});

it('popover create icon is geometrically centered inside its square button', () => {
  const row = fixture('dl-popover-create', `
    <button class="dl-popover-create-btn" aria-label="Create workspace"></button>
  `);
  const button = row.querySelector<HTMLButtonElement>('button')!;
  render(icon('add', {size: 'sm', className: 'dl-popover-create-icon'}), button);
  const iconElement = row.querySelector<SVGElement>('svg')!;
  const buttonRect = button.getBoundingClientRect();
  const iconRect = iconElement.getBoundingClientRect();

  expect(buttonRect.width).to.equal(buttonRect.height);
  expect(Math.abs(buttonRect.left + buttonRect.width / 2 - iconRect.left - iconRect.width / 2))
    .to.be.lessThanOrEqual(0.5);
  expect(Math.abs(buttonRect.top + buttonRect.height / 2 - iconRect.top - iconRect.height / 2))
    .to.be.lessThanOrEqual(0.5);
});
