// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {render} from 'lit';
import {DlIconButton, DlMenu, DlSplitLayout, defineDesignSystemElements, icon} from '../index.ts';

defineDesignSystemElements();

before(async () => new Promise<void>((resolve, reject) => {
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = new URL('../index.css', import.meta.url).href;
  link.addEventListener('load', () => { resolve(); }, {once: true});
  link.addEventListener('error', () => { reject(new Error('could not load design-system CSS')); }, {once: true});
  document.head.append(link);
}));

afterEach(() => {
  document.body.replaceChildren();
});

it('renders semantic icons as decorative fixed-size SVG', () => {
  const sizes = {xs: '12px', sm: '16px', md: '20px', lg: '24px'} as const;
  for (const [size, expected] of Object.entries(sizes)) {
    const host = document.createElement('div');
    document.body.append(host);
    render(icon('add', {size: size as keyof typeof sizes}), host);
    const svg = host.querySelector('svg')!;
    expect(svg.getAttribute('aria-hidden')).to.equal('true');
    expect(svg.getAttribute('focusable')).to.equal('false');
    expect(svg.getAttribute('stroke-width')).to.equal('1.75');
    expect(svg.classList.contains('dl-icon--stroke')).to.equal(true);
    expect(getComputedStyle(svg).width).to.equal(expected);
    expect(getComputedStyle(svg.querySelector('path')!).vectorEffect).to.equal('non-scaling-stroke');
    expect(svg.querySelectorAll('path')).to.have.length(2);
  }
  const fillHost = document.createElement('div');
  document.body.append(fillHost);
  render(icon('stop'), fillHost);
  expect(fillHost.querySelector('svg')?.classList.contains('dl-icon--fill')).to.equal(true);
});

it('registers design-system elements idempotently', () => {
  expect(() => defineDesignSystemElements()).not.to.throw();
  expect(customElements.get('dl-split-layout')).to.equal(DlSplitLayout);
  expect(customElements.get('dl-icon-button')).to.equal(DlIconButton);
  expect(customElements.get('dl-menu')).to.equal(DlMenu);
});

it('renders a decorative icon inside an accessible host button', () => {
  const button = document.createElement('dl-icon-button') as DlIconButton;
  button.name = 'close';
  button.size = 'sm';
  button.setAttribute('aria-label', 'Close panel');
  document.body.append(button);
  const inner = button.shadowRoot!.querySelector('button')!;
  expect(inner.getAttribute('aria-label')).to.equal('Close panel');
  expect(inner.querySelector('svg')?.getAttribute('aria-hidden')).to.equal('true');
  const glyph = button.shadowRoot!.querySelector('svg')!;
  expect(glyph.classList.contains('dl-icon--sm')).to.equal(true);
  expect(glyph.getBoundingClientRect().width).to.be.at.least(12);
  expect(glyph.getBoundingClientRect().height).to.be.at.least(12);
});

it('moves focus among slotted menuitems and dismisses on Escape', () => {
  const menu = document.createElement('dl-menu') as DlMenu;
  menu.innerHTML = '<button type="button" role="menuitem">Rename</button>' +
    '<button type="button" role="menuitem">Delete</button>';
  document.body.append(menu);
  const items = [...menu.querySelectorAll<HTMLButtonElement>('[role="menuitem"]')];
  items[0].focus();
  const dismissed: string[] = [];
  menu.addEventListener('dl-menu-dismiss', () => { dismissed.push('yes'); });
  menu.dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowDown', bubbles: true}));
  expect(document.activeElement).to.equal(items[1]);
  menu.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
  expect(dismissed).to.deep.equal(['yes']);
});

it('emits normalized input and commit events for keyboard resizing', () => {
  const split = document.createElement('dl-split-layout');
  split.setAttribute('orientation', 'horizontal');
  split.setAttribute('primary', 'start');
  split.setAttribute('size', '150');
  split.setAttribute('min', '100');
  split.setAttribute('max', '200');
  split.style.cssText = 'display:block;width:500px;height:160px';
  split.innerHTML = '<div slot="start">Start</div><div slot="end">End</div>';
  document.body.append(split);

  const input: number[] = [];
  const change: number[] = [];
  split.addEventListener('dl-split-input', (event) => input.push(event.detail.position));
  split.addEventListener('dl-split-change', (event) => change.push(event.detail.position));
  split.divider.dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowRight', bubbles: true}));
  split.divider.dispatchEvent(new KeyboardEvent('keydown', {key: 'End', bubbles: true}));

  expect(input).to.deep.equal([160, 200]);
  expect(change).to.deep.equal([160, 200]);
  expect(split.size).to.equal(200);
  expect(split.divider.getAttribute('aria-valuenow')).to.equal('200');
  expect(split.divider.getAttribute('aria-orientation')).to.equal('vertical');
});

it('reflects normalized bounds through size and separator ARIA', () => {
  const split = document.createElement('dl-split-layout');
  split.size = 500;
  split.min = 100;
  split.max = 200;
  split.style.cssText = 'display:block;width:500px;height:160px';
  document.body.append(split);

  expect(split.size).to.equal(200);
  expect(split.divider.getAttribute('aria-valuenow')).to.equal('200');
  split.max = 50;
  expect(split.size).to.equal(100);
  expect(split.divider.getAttribute('aria-valuemax')).to.equal('100');
  expect(split.divider.getAttribute('aria-valuenow')).to.equal('100');

  split.max = 1_000;
  split.size = 900;
  expect(split.size).to.equal(499);
  expect(split.divider.getAttribute('aria-valuemax')).to.equal('499');
  split.removeAttribute('max');
  expect(split.divider.getAttribute('aria-valuemax')).to.equal('499');

  split.size = 0;
  expect(split.divider.hasAttribute('role')).to.equal(false);
  expect(split.divider.getAttribute('aria-hidden')).to.equal('true');
  expect(split.divider.hasAttribute('aria-valuenow')).to.equal(false);
});

it('emits live and commit pixels for pointer resizing', () => {
  const split = document.createElement('dl-split-layout');
  split.primary = 'end';
  split.size = 150;
  split.min = 100;
  split.max = 220;
  split.style.cssText = 'display:block;width:500px;height:160px';
  split.innerHTML = '<div slot="start">Start</div><div slot="end">End</div>';
  document.body.append(split);
  const input: number[] = [];
  const change: number[] = [];
  split.addEventListener('dl-split-input', (event) => input.push(event.detail.position));
  split.addEventListener('dl-split-change', (event) => change.push(event.detail.position));

  split.divider.dispatchEvent(new PointerEvent('pointerdown', {
    bubbles: true,
    button: 0,
    clientX: 400,
    pointerId: 7,
  }));
  split.divider.dispatchEvent(new PointerEvent('pointermove', {clientX: 380, pointerId: 7}));
  split.divider.dispatchEvent(new PointerEvent('pointerup', {clientX: 380, pointerId: 7}));

  expect(input).to.deep.equal([170]);
  expect(change).to.deep.equal([170]);
  expect(split.size).to.equal(170);
});

it('keeps the divider above a high z-index slotted pane', () => {
  const split = document.createElement('dl-split-layout');
  split.primary = 'start';
  split.size = 200;
  split.style.cssText = 'display:block;width:500px;height:160px';
  split.innerHTML = '<div slot="start" style="z-index:120;position:absolute;inset:0"></div><div slot="end"></div>';
  document.body.append(split);
  const box = split.divider.getBoundingClientRect();
  const y = box.top + 40;
  const center = split.shadowRoot?.elementFromPoint(box.left + Math.max(box.width / 2, 0.5), y);
  const overlap = split.shadowRoot?.elementFromPoint(box.left - 4, y);
  expect(center, `center=${center?.nodeName}`).to.equal(split.divider);
  expect(overlap, `overlap=${overlap?.nodeName}.${(overlap as HTMLElement | null)?.id}`).to.equal(
    split.divider,
  );
});

it('mirrors horizontal pointer and keyboard direction under RTL', () => {
  const split = document.createElement('dl-split-layout');
  split.dir = 'rtl';
  split.primary = 'start';
  split.size = 200;
  split.min = 100;
  split.max = 400;
  split.style.cssText = 'display:block;width:500px;height:160px';
  document.body.append(split);

  split.divider.dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowLeft'}));
  expect(split.size).to.equal(210);
  split.divider.dispatchEvent(new PointerEvent('pointerdown', {
    button: 0,
    clientX: 200,
    pointerId: 11,
  }));
  split.divider.dispatchEvent(new PointerEvent('pointermove', {clientX: 180, pointerId: 11}));
  split.divider.dispatchEvent(new PointerEvent('pointerup', {clientX: 180, pointerId: 11}));
  expect(split.size).to.equal(230);

  split.primary = 'end';
  split.size = 200;
  split.divider.dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowRight'}));
  expect(split.size).to.equal(210);
});

it('owns one active pointer and cancels dragging when disabled', () => {
  const split = document.createElement('dl-split-layout');
  split.size = 200;
  split.min = 100;
  split.max = 400;
  split.style.cssText = 'display:block;width:500px;height:160px';
  document.body.append(split);

  split.divider.dispatchEvent(new PointerEvent('pointerdown', {
    button: 0,
    clientX: 200,
    pointerId: 21,
  }));
  split.divider.dispatchEvent(new PointerEvent('pointermove', {clientX: 250, pointerId: 22}));
  expect(split.size).to.equal(200);
  split.disabled = true;
  split.divider.dispatchEvent(new PointerEvent('pointermove', {clientX: 250, pointerId: 21}));
  expect(split.size).to.equal(200);
});

it('supports end-primary and nested layouts without leaking product state', () => {
  const outer = document.createElement('dl-split-layout');
  outer.primary = 'end';
  outer.size = 180;
  outer.min = 100;
  outer.max = 300;
  outer.style.cssText = 'display:block;width:600px;height:200px';
  const inner = document.createElement('dl-split-layout');
  inner.slot = 'start';
  inner.size = 120;
  inner.min = 80;
  inner.max = 240;
  inner.innerHTML = '<div slot="start">A</div><div slot="end">B</div>';
  const end = document.createElement('div');
  end.slot = 'end';
  outer.append(inner, end);
  document.body.append(outer);

  inner.divider.dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowRight'}));
  expect(inner.size).to.equal(130);
  expect(outer.size).to.equal(180);
});
