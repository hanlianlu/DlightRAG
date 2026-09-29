// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {resetMouse, sendMouse} from '@web/test-runner-commands';
import {render} from 'lit';
import {
  DlIconButton,
  DlMenu,
  DlSplitLayout,
  defineDesignSystemElements,
  icon,
  type MenuDismissDetail,
  menuButtonFocus,
  rovingFocusKeydown,
} from '../index.ts';

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
  expect(inner.getBoundingClientRect().width).to.be.at.least(44);
  expect(inner.getBoundingClientRect().height).to.be.at.least(44);
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

/** A menu with its controlling button and a control elsewhere, recording each dismissal. */
function dismissibleMenu() {
  const button = document.createElement('button');
  button.setAttribute('aria-controls', 'actions');
  button.textContent = 'Actions';
  const menu = document.createElement('dl-menu') as DlMenu;
  menu.id = 'actions';
  menu.innerHTML = '<button type="button" role="menuitem" tabindex="-1">Rename</button>'
    + '<button type="button" role="menuitem" tabindex="-1">Delete</button>';
  const elsewhere = document.createElement('button');
  elsewhere.textContent = 'Elsewhere';
  document.body.append(button, menu, elsewhere);
  const dismissals: boolean[] = [];
  menu.addEventListener('dl-menu-dismiss', (event: CustomEvent<MenuDismissDetail>) => {
    dismissals.push(event.detail.restoreFocus);
  });
  const [rename, remove] = [...menu.querySelectorAll<HTMLButtonElement>('[role="menuitem"]')];
  return {button, menu, elsewhere, rename: rename!, remove: remove!, dismissals};
}

it('closes on Tab and on focus leaving without asking for focus back; only Escape asks', () => {
  const {button, menu, elsewhere, rename, remove, dismissals} = dismissibleMenu();
  rename.focus();
  const tab = new KeyboardEvent('keydown', {key: 'Tab', bubbles: true, cancelable: true});
  rename.dispatchEvent(tab);
  expect(dismissals).to.deep.equal([false]);
  expect(tab.defaultPrevented, 'the browser still moves focus on').to.equal(false);

  remove.focus();
  menu.focus();
  expect(document.activeElement === menu, 'a press on the menu chrome keeps focus in the menu')
    .to.equal(true);
  rename.focus();
  expect(dismissals, 'focus moving within the menu, its chrome included, keeps it open')
    .to.deep.equal([false]);

  button.focus();
  expect(dismissals, 'its own button closes it on click instead').to.deep.equal([false]);

  rename.focus();
  elsewhere.focus();
  expect(dismissals).to.deep.equal([false, false]);

  menu.hidden = true;
  rename.focus();
  elsewhere.focus();
  expect(dismissals, 'a closed menu asks nothing').to.deep.equal([false, false]);

  menu.hidden = false;
  rename.focus();
  rename.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
  expect(dismissals).to.deep.equal([false, false, true]);
});

it('closes on a real click on its own button, whether or not the engine focuses the button', async () => {
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = 'Actions';
  button.setAttribute('aria-controls', 'owned');
  const menu = document.createElement('dl-menu') as DlMenu;
  menu.id = 'owned';
  menu.innerHTML = '<button type="button" role="menuitem" tabindex="-1">Rename</button>';
  document.body.append(button, menu);
  // A minimal owner: its button toggles the menu and a dismissal closes it,
  // with the surface's own [hidden] rule.
  let open = false;
  let opened = 0;
  const show = (next: boolean): void => {
    open = next;
    menu.hidden = !next;
    menu.style.display = next ? '' : 'none';
    if (!next) return;
    opened += 1;
    menu.focusItem('first');
  };
  show(false);
  button.addEventListener('click', () => { show(!open); });
  menu.addEventListener('dl-menu-dismiss', () => { show(false); });
  const box = button.getBoundingClientRect();
  const center: [number, number] = [
    Math.round(box.left + box.width / 2),
    Math.round(box.top + box.height / 2),
  ];

  try {
    await sendMouse({type: 'click', position: center});
    expect(open, 'a press on the button opens the menu').to.equal(true);
    expect(document.activeElement === menu.querySelector('[role="menuitem"]')).to.equal(true);
    // WebKit leaves a pressed button unfocused: the item's focus lands on no element.
    await sendMouse({type: 'click', position: center});
    expect(open, 'a second press on its button closes the menu').to.equal(false);
    expect(opened, 'and does not reopen it').to.equal(1);
  } finally {
    await resetMouse();
  }
});

it('keeps an aria-disabled item in reach but never activates it', () => {
  const {rename, remove} = dismissibleMenu();
  const activated: string[] = [];
  rename.addEventListener('click', () => { activated.push('rename'); });
  remove.addEventListener('click', () => { activated.push('delete'); });
  remove.setAttribute('aria-disabled', 'true');

  rename.focus();
  rename.dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowDown', bubbles: true}));
  expect(document.activeElement === remove, 'the disabled item takes focus').to.equal(true);
  remove.click();
  rename.click();
  expect(activated).to.deep.equal(['rename']);

  remove.removeAttribute('aria-disabled');
  remove.click();
  expect(activated).to.deep.equal(['rename', 'delete']);
});

it('keeps one menu contract across item roles, disabled items, and typeahead', () => {
  const menu = document.createElement('dl-menu') as DlMenu;
  menu.innerHTML = '<button type="button" role="menuitemradio" aria-checked="true">Auto</button>'
    + '<button type="button" role="menuitemradio" disabled>Archive</button>'
    + '<button type="button" role="menuitemcheckbox" aria-label="Fast answers">Fast</button>'
    + '<button type="button" role="menuitem" aria-disabled="true">Finance</button>'
    + '<button type="button" role="menuitem">Research</button>';
  document.body.append(menu);
  const [auto, , fast, , research] = [...menu.querySelectorAll<HTMLButtonElement>('button')];
  const key = (name: string, init: KeyboardEventInit = {}): boolean => {
    const event = new KeyboardEvent('keydown', {key: name, bubbles: true, cancelable: true, ...init});
    (document.activeElement as HTMLElement).dispatchEvent(event);
    return event.defaultPrevented;
  };

  const [, , , finance] = [...menu.querySelectorAll<HTMLButtonElement>('button')];
  const focused = (item: HTMLButtonElement | undefined): boolean => document.activeElement === item;

  menu.focusItem('last');
  expect(focused(research)).to.equal(true);
  menu.focusItem('first');
  expect(focused(auto)).to.equal(true);
  key('ArrowDown');
  expect(focused(fast), 'a natively disabled item cannot take focus').to.equal(true);
  key('ArrowDown');
  expect(focused(finance), 'an aria-disabled item still does').to.equal(true);
  key('ArrowUp');
  key('ArrowUp');
  key('ArrowUp');
  expect(focused(research), 'ArrowUp wraps').to.equal(true);
  key('a');
  expect(focused(auto), 'typeahead searches on from the current item').to.equal(true);
  key('f');
  expect(focused(fast), 'typeahead reads the accessible label').to.equal(true);
  key('f');
  expect(focused(finance), 'typeahead moves on to the next match').to.equal(true);
  key('f');
  expect(focused(fast), 'and wraps to the first').to.equal(true);
  expect(key('r', {ctrlKey: true}), 'a shortcut is not typeahead').to.equal(false);
  expect(focused(fast)).to.equal(true);
  expect(key(' '), 'Space stays with the item').to.equal(false);
  expect(key('Enter'), 'Enter stays with the item').to.equal(false);
});

it('anchors a surface to its trigger wrapper along the writing direction', () => {
  const wrapper = document.createElement('div');
  wrapper.style.cssText = 'position: relative; width: 200px; height: 40px; margin: 120px 80px;';
  const surface = document.createElement('div');
  surface.className = 'dl-anchored';
  surface.style.cssText = 'width: 60px; height: 30px; --anchored-gap: 7px;';
  wrapper.append(surface);
  document.body.append(wrapper);
  const place = (classes: string, dir: 'ltr' | 'rtl' = 'ltr') => {
    surface.className = classes;
    wrapper.dir = dir;
    const outer = wrapper.getBoundingClientRect();
    const inner = surface.getBoundingClientRect();
    return {
      start: dir === 'ltr' ? inner.left - outer.left : outer.right - inner.right,
      end: dir === 'ltr' ? outer.right - inner.right : inner.left - outer.left,
      below: inner.top - outer.bottom,
      above: outer.top - inner.bottom,
    };
  };

  for (const dir of ['ltr', 'rtl'] as const) {
    const below = place('dl-anchored', dir);
    expect([below.start, below.below], `${dir}: below, from the start edge`).to.deep.equal([0, 7]);
    const end = place('dl-anchored dl-anchored--end', dir);
    expect([end.end, end.below], `${dir}: aligned with the end edge`).to.deep.equal([0, 7]);
    const above = place('dl-anchored dl-anchored--above dl-anchored--end', dir);
    expect([above.end, above.above], `${dir}: opening upward`).to.deep.equal([0, 7]);
  }
});

it('opens menus from their button with one key map', () => {
  const focus = (key: string) => menuButtonFocus(new KeyboardEvent('keydown', {key}));
  expect(['ArrowDown', 'Enter', ' ', 'ArrowUp', 'Escape', 'a'].map(focus))
    .to.deep.equal(['first', 'first', 'first', 'last', null, null]);
});

it('lets a picker field keep its caret keys and typing', () => {
  const picker = document.createElement('div');
  picker.innerHTML = '<button type="button">Alpha</button><button type="button">Beta</button>'
    + '<input type="text" aria-label="New workspace name">';
  document.body.append(picker);
  const [alpha, beta] = [...picker.querySelectorAll<HTMLButtonElement>('button')];
  const input = picker.querySelector('input')!;
  const roam = (name: string): boolean => {
    const event = new KeyboardEvent('keydown', {key: name, bubbles: true, cancelable: true});
    input.focus();
    input.dispatchEvent(event);
    return rovingFocusKeydown(event, [alpha!, beta!]);
  };

  expect(roam('Home')).to.equal(false);
  expect(document.activeElement).to.equal(input);
  expect(roam('b')).to.equal(false);
  expect(document.activeElement).to.equal(input);
  expect(roam('ArrowDown')).to.equal(true);
  expect(document.activeElement, 'the arrows leave the field for the choices').to.equal(alpha);
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

it('lets a product owner raise one isolated pane for an overlay', () => {
  const split = document.createElement('dl-split-layout');
  split.style.cssText = [
    'display:block',
    'width:500px',
    'height:160px',
    '--split-start-layer:2',
  ].join(';');
  split.innerHTML = '<div slot="start"></div><div slot="end"></div>';
  document.body.append(split);

  const start = split.shadowRoot?.querySelector<HTMLElement>('#start');
  const end = split.shadowRoot?.querySelector<HTMLElement>('#end');
  expect(getComputedStyle(start!).zIndex).to.equal('2');
  expect(getComputedStyle(end!).zIndex).to.equal('0');
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
