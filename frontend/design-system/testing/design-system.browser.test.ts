// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {resetMouse, sendMouse} from '@web/test-runner-commands';
import {render} from 'lit';
import {linkStyles} from '../../testing/dom.ts';
import {
  DlIconButton,
  DlMenu,
  DlSplitLayout,
  defineDesignSystemElements,
  icon,
  type IconName,
  type MenuDismissDetail,
  menuButtonFocus,
  rovingFocusKeydown,
} from '../index.ts';

defineDesignSystemElements();

// The primitives are judged on their own stylesheet, without the product's.
before(async () => { await linkStyles([new URL('../index.css', import.meta.url).href]); });

afterEach(() => {
  document.body.replaceChildren();
});

it('renders semantic icons as decorative fixed-size SVG', () => {
  const sizes = {xs: '12px', sm: '16px', md: '20px', lg: '24px'} as const;
  // The weight is 1.5 of the glyph's own 24 units; the 12px step keeps Lucide's 2-unit floor so its ink
  // stays a CSS pixel there.
  const weights = {xs: '2px', sm: '1.5px', md: '1.5px', lg: '1.5px'} as const;
  for (const [size, expected] of Object.entries(sizes)) {
    const host = document.createElement('div');
    document.body.append(host);
    render(icon('add', {size: size as keyof typeof sizes}), host);
    const svg = host.querySelector('svg')!;
    expect(svg.getAttribute('aria-hidden')).to.equal('true');
    expect(svg.getAttribute('focusable')).to.equal('false');
    expect(svg.getAttribute('stroke-width')).to.equal('1.5');
    expect(svg.classList.contains('dl-icon--stroke')).to.equal(true);
    expect(getComputedStyle(svg).width).to.equal(expected);
    // The weight is stated in the glyph's own 24-unit viewBox, so the engine scales it with the drawing:
    // nothing pins the line to CSS pixels as the icon shrinks, and no shape opts out of that scale.
    expect(getComputedStyle(svg).strokeWidth).to.equal(weights[size as keyof typeof weights]);
    expect(getComputedStyle(svg.querySelector('path')!).vectorEffect).to.equal('none');
    expect(svg.querySelectorAll('path')).to.have.length(2);
  }
  const fillHost = document.createElement('div');
  document.body.append(fillHost);
  render(icon('stop'), fillHost);
  expect(fillHost.querySelector('svg')?.classList.contains('dl-icon--fill')).to.equal(true);
});

it('carries the paint a shape states for itself, beside the whole-glyph paint', () => {
  // lucide draws key-round's dot as a filled circle inside an unfilled glyph, and an authored solid rules
  // its hole even-odd so the hole does not depend on how its subpaths are wound.
  const host = document.createElement('div');
  document.body.append(host);
  render(icon('agent-accounts'), host);
  expect(host.querySelector('circle')?.getAttribute('fill')).to.equal('currentColor');
  expect(host.querySelector('path')?.getAttribute('fill')).to.equal(null);

  const solid = document.createElement('div');
  document.body.append(solid);
  render(icon('thumbs-up-filled'), solid);
  expect(solid.querySelector('path')?.getAttribute('fill-rule')).to.equal('evenodd');
});

/** What one icon paints, read back from a 96 px raster of the glyph as the renderer drew it. */
async function painted(name: IconName): Promise<{
  box: {left: number; top: number; right: number; bottom: number};
  alphaAt: (x: number, y: number) => number;
}> {
  const host = document.createElement('div');
  document.body.append(host);
  render(icon(name, {size: 'lg'}), host);
  const svg = host.querySelector('svg')!;
  // A serialized SVG is rastered as a standalone image, not as a styled element: Chromium takes its size
  // from the inline style, but Firefox and WebKit take the intrinsic size from the width and height
  // attributes alone and draw the viewBox into a default 300 x 150 box without them.
  svg.setAttribute('width', '96');
  svg.setAttribute('height', '96');
  svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  // A serialized glyph carries no stylesheet, but its weight is stated in its own viewBox, so it rasters
  // at the proportion the product draws: the 96px board is four times the 24px glyph, and the stroke is
  // four times the 1.5px of ink a 24px mark carries.
  const image = new Image();
  image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(new XMLSerializer().serializeToString(svg))}`;
  await image.decode();
  const canvas = document.createElement('canvas');
  canvas.width = 96;
  canvas.height = 96;
  const context = canvas.getContext('2d')!;
  context.drawImage(image, 0, 0);
  const {data} = context.getImageData(0, 0, 96, 96);
  const alphaAt = (x: number, y: number): number => data[(Math.round(y) * 96 + Math.round(x)) * 4 + 3]!;
  let left = 96;
  let top = 96;
  let right = -1;
  let bottom = -1;
  for (let y = 0; y < 96; y += 1) {
    for (let x = 0; x < 96; x += 1) {
      if (alphaAt(x, y) < 24) continue;
      left = Math.min(left, x);
      top = Math.min(top, y);
      right = Math.max(right, x);
      bottom = Math.max(bottom, y);
    }
  }
  return {box: {left, top, right, bottom}, alphaAt};
}

it('draws the weight with the drawing, so the stroke scales with the icon', async () => {
  const {alphaAt} = await painted('add');
  // The 96px raster is four times the 24px glyph, so the 1.5-unit weight inks six device pixels across the
  // plus's upright; a weight pinned to CSS pixels would ink two there whatever the drawing did.
  let ink = 0;
  for (let x = 0; x < 96; x += 1) ink += alphaAt(x, 24) / 255;
  expect(ink).to.be.above(5.5);
  expect(ink).to.be.below(6.5);
});

it('draws a solid thumb as its outline with a hole, never as ink added outside it', async () => {
  const viewBox = 96 / 24;
  // Where the cuff sits in each glyph: the point at its centre, one point inside each wall around it, and
  // points well inside the hand, where a filled silhouette is inked and an outline is not.
  const cuffs = {
    'thumbs-up': {
      centre: [4.85, 16],
      walls: [[4.85, 10.65], [4.85, 21.35], [2.65, 16], [6.9, 16]],
      interior: [[12, 13], [14, 15]],
    },
    'thumbs-down': {
      centre: [19.15, 8],
      walls: [[19.15, 2.65], [19.15, 13.35], [17.1, 8], [21.35, 8]],
      interior: [[12, 11], [10, 9]],
    },
  } as const;

  for (const [outlineName, {centre, walls, interior}] of Object.entries(cuffs)) {
    const solid = await painted(`${outlineName}-filled` as IconName);
    const outline = await painted(outlineName as IconName);
    // The cuff is a hole: empty where it is, inked on every side of it.
    expect(solid.alphaAt(centre[0] * viewBox, centre[1] * viewBox), `${outlineName}: the cuff is empty`)
      .to.be.below(24);
    for (const [x, y] of walls) {
      expect(solid.alphaAt(x * viewBox, y * viewBox), `${outlineName}: the cuff is inked at ${x},${y}`)
        .to.be.above(150);
    }
    // The hand is filled in the solid and empty in the outline: the solid is the full silhouette, not an
    // outline of it. A proportional weight inks much of what the fill covers, so counting ink cannot tell
    // the two apart; a point well inside the hand can.
    for (const [x, y] of interior) {
      expect(solid.alphaAt(x * viewBox, y * viewBox), `${outlineName}: the hand is filled at ${x},${y}`)
        .to.be.above(150);
      expect(outline.alphaAt(x * viewBox, y * viewBox), `${outlineName}: the outline leaves ${x},${y} empty`)
        .to.be.below(24);
    }
    // A hole subtracts ink; it does not add any, so the solid keeps the outline's silhouette. The solid is
    // the stroke's own geometry filled, so its ink box may only reach the outline's ink box, which the
    // outline already inflates by half a stroke; the one pixel is antialiasing, not that overhang.
    const overhang = 1;
    expect(solid.box.left, `${outlineName}: nothing is drawn beyond the outline`).to.be.at.least(outline.box.left - overhang);
    expect(solid.box.top, `${outlineName}: nothing is drawn beyond the outline`).to.be.at.least(outline.box.top - overhang);
    expect(solid.box.right, `${outlineName}: nothing is drawn beyond the outline`).to.be.at.most(outline.box.right + overhang);
    expect(solid.box.bottom, `${outlineName}: nothing is drawn beyond the outline`).to.be.at.most(outline.box.bottom + overhang);
  }
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
  // The element's own tree states the same viewBox-unit weight the stylesheet does, so a mark in the
  // shadow tree and a mark in the page draw one weight.
  expect(getComputedStyle(glyph).strokeWidth).to.equal('1.5px');
  expect(getComputedStyle(glyph.querySelector('path')!).vectorEffect).to.equal('none');
  // The 12px step keeps the 2-unit floor inside the shadow tree too.
  const tiny = document.createElement('dl-icon-button') as DlIconButton;
  tiny.name = 'close';
  tiny.size = 'xs';
  document.body.append(tiny);
  expect(getComputedStyle(tiny.shadowRoot!.querySelector('svg')!).strokeWidth).to.equal('2px');
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

// Script focus rings only until the page has seen a pointer press, so this test stays ahead of the ones that press.
it('rings a dialog checkbox row on keyboard focus', () => {
  const host = document.createElement('div');
  host.innerHTML = '<label class="dl-dialog-checkbox"><input type="checkbox"> Remember this</label>';
  const probe = document.createElement('div');
  probe.style.color = 'var(--color-control-ring)';
  document.body.append(host, probe);
  const input = host.querySelector('input')!;
  input.focus();

  expect(input.matches(':focus-visible')).to.equal(true);
  const ring = getComputedStyle(host.querySelector('label')!);
  expect(ring.outlineStyle).to.equal('solid');
  expect(ring.outlineWidth).to.equal('2px');
  expect(ring.outlineColor).to.equal(getComputedStyle(probe).color);
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

it('gives a link the button face, never the link underline', () => {
  const row = document.createElement('div');
  row.innerHTML = '<button class="dl-btn" type="button">Retry</button>'
    + '<a class="dl-btn" href="/download">Download</a>';
  document.body.append(row);
  const face = (element: Element): string[] => {
    const style = getComputedStyle(element);
    return ['text-decoration-line', 'color', 'border-top-width', 'border-top-style', 'border-top-color',
      'height', 'font-size', 'cursor'].map((property) => style.getPropertyValue(property));
  };
  const link = row.querySelector('a')!;

  expect(face(link)).to.deep.equal(face(row.querySelector('button')!));
  expect(getComputedStyle(link).textDecorationLine).to.equal('none');
});

it('lights a dialog checkbox row under the pointer, with its tint inset past the text', async () => {
  const host = document.createElement('div');
  host.style.cssText = 'width:300px;padding:40px';
  host.innerHTML = '<label class="dl-dialog-checkbox"><input type="checkbox"> Remember this</label>';
  const probe = document.createElement('div');
  probe.style.backgroundColor = 'var(--color-bg-hover)';
  document.body.append(host, probe);
  const row = host.querySelector('label')!;

  expect(getComputedStyle(row).backgroundColor, 'plain at rest').to.equal('rgba(0, 0, 0, 0)');
  // The tint reaches past the text by the row's padding, so the words stay where the form's other rows keep them.
  const padding = Number.parseFloat(getComputedStyle(row).paddingInlineStart);
  expect(padding).to.be.greaterThan(0);
  expect(row.getBoundingClientRect().left + padding).to.be.closeTo(host.getBoundingClientRect().left + 40, 0.5);

  const box = row.getBoundingClientRect();
  try {
    await sendMouse({type: 'move', position: [Math.round(box.left + box.width - 4), Math.round(box.top + box.height / 2)]});
    await Promise.all(row.getAnimations().map((animation) => animation.finished));
    expect(getComputedStyle(row).backgroundColor).to.equal(getComputedStyle(probe).backgroundColor);
  } finally {
    await resetMouse();
  }
});

/** A colour, length or size a token states, as the engine computes it. */
function tokenOf(property: string, token: string): string {
  const probe = document.createElement('div');
  probe.style.setProperty(property, `var(${token})`);
  document.body.append(probe);
  const value = getComputedStyle(probe).getPropertyValue(property);
  probe.remove();
  return value;
}

it('keeps a note\'s face inside a confirm dialog, on a paragraph, and leaves the dialog its own sentences', () => {
  const dialog = document.createElement('dialog');
  dialog.className = 'confirm-dialog';
  dialog.innerHTML = '<p>The dialog\'s own sentence.</p><p class="dl-note">A quiet line.</p>';
  document.body.append(dialog);
  const [sentence, note] = [...dialog.querySelectorAll('p')].map((paragraph) => getComputedStyle(paragraph));

  expect(note!.fontSize, 'a note is caption size').to.equal(tokenOf('font-size', '--font-size-caption'));
  expect(note!.marginBottom, 'and has no margin of the dialog\'s').to.equal('0px');
  expect(sentence!.fontSize).to.equal(tokenOf('font-size', '--font-size-detail'));
  expect(sentence!.marginBottom, 'the dialog\'s sentence keeps its margin').not.to.equal('0px');
});

it('draws a list card as one bordered, clipped list with a hairline before every entry but the first', () => {
  const host = document.createElement('div');
  host.innerHTML = '<ul class="dl-list-card"><li>One</li><li>Two</li><li>Three</li></ul>';
  document.body.append(host);
  const card = getComputedStyle(host.querySelector('ul')!);
  const entries = [...host.querySelectorAll('li')].map((entry) => getComputedStyle(entry).borderTopWidth);

  expect([card.borderTopWidth, card.borderTopStyle, card.borderTopColor]).to.deep.equal(['1px', 'solid', tokenOf('color', '--color-border-subtle')]);
  expect(card.borderTopLeftRadius).to.equal(tokenOf('border-radius', '--radius-card'));
  expect([card.overflow, card.listStyleType, card.paddingLeft, card.marginTop]).to.deep.equal(['clip', 'none', '0px', '0px']);
  expect(entries).to.deep.equal(['0px', '1px', '1px']);
});

it('draws the dialog\'s primary action on its plain actions\' geometry, dims it when disabled, and tints only an enabled one under the pointer', async () => {
  const host = document.createElement('div');
  host.innerHTML = '<div class="dl-dialog-actions"><button type="button">Close</button>'
    + '<button type="button" class="dl-dialog-primary">Save</button>'
    + '<button type="button" class="dl-dialog-primary" disabled>Saving</button></div>';
  document.body.append(host);
  const [plain, primary, disabled] = [...host.querySelectorAll('button')];
  const face = (button: Element): string[] => {
    const style = getComputedStyle(button);
    return ['height', 'padding-left', 'padding-right', 'font-size', 'font-weight', 'border-top-width', 'border-top-left-radius']
      .map((property) => style.getPropertyValue(property));
  };

  expect(face(primary!), 'the same geometry and weight as the button beside it').to.deep.equal(face(plain!));
  expect(getComputedStyle(primary!).backgroundColor).to.equal(tokenOf('background-color', '--color-action-bg'));
  expect(getComputedStyle(primary!).color).to.equal(tokenOf('color', '--color-on-action'));
  expect([getComputedStyle(disabled!).opacity, getComputedStyle(disabled!).cursor]).to.deep.equal(['0.5', 'not-allowed']);

  const under = async (button: Element): Promise<string> => {
    const box = button.getBoundingClientRect();
    await sendMouse({type: 'move', position: [Math.round(box.left + box.width / 2), Math.round(box.top + box.height / 2)]});
    await Promise.all(button.getAnimations().map((animation) => animation.finished));
    return getComputedStyle(button).backgroundColor;
  };
  try {
    expect(await under(primary!), 'tinted under the pointer').to.equal(tokenOf('background-color', '--color-action-bg-hover'));
    expect(await under(disabled!), 'a disabled one is not').to.equal(tokenOf('background-color', '--color-action-bg'));
  } finally {
    await resetMouse();
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

it('names its separator with the label its owner gives it, and follows a change', () => {
  const split = document.createElement('dl-split-layout');
  split.setAttribute('label', 'Resize the list');
  document.body.append(split);
  expect(split.divider.getAttribute('aria-label')).to.equal('Resize the list');

  split.setAttribute('label', 'Resize the panel');
  expect(split.divider.getAttribute('aria-label')).to.equal('Resize the panel');
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

/** One synthetic drag on the divider, the way a pointer reports it. */
function drag(split: HTMLElement & {divider: HTMLDivElement}, from: number, to: number): void {
  split.divider.dispatchEvent(new PointerEvent('pointerdown', {
    bubbles: true, button: 0, clientX: from, pointerId: 11,
  }));
  split.divider.dispatchEvent(new PointerEvent('pointermove', {clientX: to, pointerId: 11}));
  split.divider.dispatchEvent(new PointerEvent('pointerup', {clientX: to, pointerId: 11}));
}

function collapsible(primary: 'start' | 'end' = 'start'): HTMLElement & {
  divider: HTMLDivElement; size: number; min: number;
} {
  const split = document.createElement('dl-split-layout') as HTMLElement & {
    divider: HTMLDivElement; size: number; min: number;
  };
  split.setAttribute('primary', primary);
  split.setAttribute('size', '200');
  split.setAttribute('min', '120');
  split.style.cssText = 'display:block;width:500px;height:160px';
  split.innerHTML = '<div slot="start">Start</div><div slot="end">End</div>';
  document.body.append(split);
  return split;
}

it('snaps a panel the reader drags past the threshold all the way shut', () => {
  const split = collapsible();
  const change: number[] = [];
  split.addEventListener('dl-split-change', (event) => change.push((event as CustomEvent).detail.position));

  // Half of the smallest width is the threshold by default, and dragging the start pane narrower
  // takes it below that: 200 -> -50 is past it, so the panel goes.
  drag(split, 350, 100);
  expect(split.size, 'the panel is shut, not left a sliver wide').to.equal(0);
  expect(split.hasAttribute('data-collapsed'), 'and says so').to.equal(true);
  expect(split.divider.hasAttribute('role'), 'while its divider leaves the tab order').to.equal(false);
  expect(change).to.deep.equal([0]);
});

it('holds a panel near the threshold still, and only brings it back past it', () => {
  const split = collapsible();
  drag(split, 350, 100);
  expect(split.size).to.equal(0);

  // A drag that stays inside the threshold leaves it shut: that is the hysteresis.
  drag(split, 0, 40);
  expect(split.size, 'a nudge inside the threshold is not a decision').to.equal(0);

  drag(split, 0, 80);
  expect(split.size, 'past the threshold it comes back at its smallest width').to.equal(120);
});

it('collapses with Enter and restores the width it had, as the window splitter pattern asks', () => {
  const split = collapsible();
  const change: number[] = [];
  split.addEventListener('dl-split-change', (event) => change.push((event as CustomEvent).detail.position));

  split.divider.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', bubbles: true}));
  expect(split.size, 'Enter collapses it').to.equal(0);
  split.divider.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', bubbles: true}));
  expect(split.size, 'and Enter again restores the width it had, not its smallest').to.equal(200);
  expect(change).to.deep.equal([0, 200]);
});

it('asks its owner for the default width when the divider is double-clicked', () => {
  const split = collapsible();
  const resets: Event[] = [];
  split.addEventListener('dl-split-reset', (event) => resets.push(event));
  split.divider.dispatchEvent(new MouseEvent('dblclick', {bubbles: true}));
  expect(resets.length, 'a reset is a request: the default width belongs to whoever owns the panel').to.equal(1);
});
