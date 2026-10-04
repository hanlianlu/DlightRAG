// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Typographic and geometric invariants of the shared control primitives. */

import {expect} from '@esm-bundle/chai';
import {setViewport} from '@web/test-runner-commands';
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

const DESKTOP = {width: 1280, height: 800};
const PHONE = {width: 390, height: 844};
const PHONE_LANDSCAPE = {width: 844, height: 390};

/** Run a check at one viewport size, and leave the next test the one it found. */
async function atViewport(size: {width: number; height: number}, check: () => void): Promise<void> {
  const original = {width: window.innerWidth, height: window.innerHeight};
  await setViewport(size);
  try {
    check();
  } finally {
    await setViewport(original);
  }
}

/** A switch with room around it, so what reaches past its track has somewhere to land. */
function switchControl(className: string, checked = false): HTMLButtonElement {
  const stage = fixture('', '');
  stage.style.padding = '40px';
  const button = document.createElement('button');
  button.className = className;
  button.setAttribute('role', 'switch');
  button.setAttribute('aria-checked', String(checked));
  stage.append(button);
  return button;
}

it('switch foundations satisfy both symmetry invariants', async () => {
  const token = (name: string): number => Number.parseFloat(
    getComputedStyle(document.documentElement).getPropertyValue(name),
  );

  // The dense variant is the same control one icon step down, so both sizes prove out beside a
  // pointer. The variant composes with the base class, exactly as the product markup uses it.
  await atViewport(DESKTOP, () => {
    for (const [className, suffix] of [
      ['dl-switch', ''],
      ['dl-switch dl-switch--dense', '-sm'],
    ] as const) {
      const width = token(`--size-switch${suffix}-width`);
      const height = token(`--size-switch${suffix}-height`);
      const thumb = token(`--size-switch${suffix}-thumb`);
      const inset = token(`--size-switch${suffix}-inset`);

      expect(height).to.equal(thumb + 2 * inset);

      // The travel token is a calc() and stays uncomputed in getComputedStyle, so prove the rendered
      // displacement against the same invariant instead of reading the token text.
      const button = switchControl(className, true);
      const rendered = getComputedStyle(button, '::after').transform;
      const travel = Number.parseFloat(/matrix\(1, 0, 0, 1, ([\d.]+),/.exec(rendered)?.[1] ?? 'NaN');

      expect(travel).to.equal(width - thumb - 2 * inset);
      expect(button.getBoundingClientRect().width).to.equal(width);
      expect(button.getBoundingClientRect().height).to.equal(height);
    }
  });
});

it('a dense switch is compact beside a pointer and the regular size where a finger is the pointer', async () => {
  const size = (className: string): number[] => {
    const box = switchControl(className).getBoundingClientRect();
    return [box.width, box.height];
  };

  await atViewport(DESKTOP, () => {
    expect(size('dl-switch')).to.deep.equal([40, 24]);
    expect(size('dl-switch dl-switch--dense')).to.deep.equal([28, 16]);
  });
  // The phone layout is a narrow viewport or a short one; both put a finger on the control.
  await atViewport(PHONE, () => {
    expect(size('dl-switch dl-switch--dense')).to.deep.equal([40, 24]);
  });
  await atViewport(PHONE_LANDSCAPE, () => {
    expect(size('dl-switch dl-switch--dense')).to.deep.equal([40, 24]);
  });
});

it('a switch is as easy to hit as the control ladder says, however small its track is drawn', async () => {
  const ladder = document.createElement('div');
  ladder.style.width = 'var(--control-hit-target)';
  document.body.append(ladder);
  const reach = ladder.getBoundingClientRect().width / 2;
  const edges = (distance: number): Array<[number, number]> => [
    [distance, 0], [-distance, 0], [0, distance], [0, -distance],
  ];

  const check = (): void => {
    for (const className of ['dl-switch', 'dl-switch dl-switch--dense']) {
      const button = switchControl(className);
      const track = button.getBoundingClientRect();
      const hit = ([dx, dy]: [number, number]): boolean => document.elementFromPoint(
        track.left + track.width / 2 + dx, track.top + track.height / 2 + dy,
      ) === button;

      // Every edge of the hit area belongs to the switch, and nothing past it does.
      expect(edges(reach - 1).map(hit), className).to.deep.equal([true, true, true, true]);
      expect(edges(reach + 2).map(hit), className).to.deep.equal([false, false, false, false]);
      button.parentElement!.remove();
    }
  };
  await atViewport(DESKTOP, check);
  await atViewport(PHONE, check);
});

it('a navigation row shows its short status, and in the list form its full line and a disclosure instead', () => {
  const row = (form: string): string => `
    <button class="dl-nav-item ${form}" type="button">
      <span class="dl-nav-item-text">
        <span class="dl-nav-item-label">Connections</span>
        <span class="dl-nav-item-detail">MCP · 1 of 2 enabled</span>
      </span>
      <span class="dl-nav-item-status">1/2</span>
      <span class="dl-nav-item-disclosure">&gt;</span>
    </button>`;
  const nav = fixture('', `<nav>${row('')}${row('dl-nav-item--list')}</nav>`);
  const [plain, list] = [...nav.querySelectorAll<HTMLElement>('.dl-nav-item')];
  const shown = (item: HTMLElement): string[] => ['status', 'detail', 'disclosure']
    .filter((part) => item.querySelector(`.dl-nav-item-${part}`)!.getClientRects().length > 0);

  expect(shown(plain!)).to.deep.equal(['status']);
  expect(shown(list!)).to.deep.equal(['detail', 'disclosure']);
  // The list form is an entry of a list that clips it, so its corners are the list's.
  expect(getComputedStyle(list!).borderTopLeftRadius).to.equal('0px');
  expect(getComputedStyle(plain!).borderTopLeftRadius).not.to.equal('0px');
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
