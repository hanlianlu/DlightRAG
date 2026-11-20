// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The Connections page as the shipped stylesheets lay it out (the product-styles page of the runner). */

import {expect} from '@esm-bundle/chai';
import {setViewport} from '@web/test-runner-commands';
import {draft, mount, openCard, replies} from '../testing/connections.ts';
import {fieldNamed} from '../testing/dom.ts';
import connectionStyles from '../styles/settings-connections.module.css';

const originalFetch = window.fetch;
const originalViewport = {width: window.innerWidth, height: window.innerHeight};

afterEach(async () => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
  await setViewport(originalViewport);
});

it('puts Label and Endpoint side by side in a wide pane, and the delete hint beside Delete…', async () => {
  await setViewport({width: 1280, height: 800});
  window.fetch = replies([draft]);
  const feature = mount();
  await openCard(feature, 'fixture');

  const label = fieldNamed(feature, 'Label')!.getBoundingClientRect();
  const endpoint = feature.querySelector(`.${connectionStyles.endpointRow}`)!.getBoundingClientRect();
  expect(label.right).to.be.at.most(endpoint.left);
  expect(label.top).to.be.closeTo(endpoint.top, 12);

  const deleteButton = feature.querySelector('[data-delete="fixture"]')!;
  const button = deleteButton.getBoundingClientRect();
  const hint = deleteButton.previousElementSibling!.getBoundingClientRect();
  expect(hint.right).to.be.at.most(button.left);
  expect(Math.abs(hint.top + hint.height / 2 - (button.top + button.height / 2))).to.be.below(button.height);
});

it('stacks them in one column on a phone', async () => {
  await setViewport({width: 390, height: 844});
  window.fetch = replies([draft]);
  const feature = mount();
  await openCard(feature, 'fixture');

  const label = fieldNamed(feature, 'Label')!.getBoundingClientRect();
  const endpoint = feature.querySelector(`.${connectionStyles.endpointRow}`)!.getBoundingClientRect();
  expect(endpoint.top).to.be.at.least(label.bottom);
});

describe('the card row', () => {
  async function cardRow() {
    window.fetch = replies([draft]);
    const feature = mount();
    await openCard(feature, 'fixture');
    const toggle = feature.querySelector<HTMLButtonElement>('[data-card="fixture"]')!;
    const chevron = toggle.querySelector<HTMLElement>(`.${connectionStyles.chevron}`)!;
    return {feature, toggle, chevron, glyph: chevron.querySelector('svg')!};
  }

  const centre = (node: Element): [number, number] => {
    const box = node.getBoundingClientRect();
    return [box.left + box.width / 2, box.top + box.height / 2];
  };

  it('pins one glyph-sized disclosure box to the row end and turns it on its own centre', async () => {
    const {feature, toggle, chevron, glyph} = await cardRow();
    // The card opened for the helper; close it to see the glyph at rest, then open it again by the row.
    toggle.click();
    await feature.updateComplete;
    chevron.getAnimations().forEach((animation) => { animation.finish(); });

    const row = toggle.getBoundingClientRect();
    const padding = Number.parseFloat(getComputedStyle(toggle).paddingRight);
    // A track that grew with the panel would draw the glyph at the box's left edge and pivot on the box.
    expect(chevron.getBoundingClientRect().right, 'at the row end').to.be.closeTo(row.right - padding, 1);
    expect(chevron.getBoundingClientRect().width, 'glyph-sized').to.be.closeTo(glyph.getBoundingClientRect().width, 1);
    expect(centre(glyph), 'the glyph stands in the middle of its box').to.deep.equal(centre(chevron));
    const rest = centre(glyph);

    toggle.click();
    await feature.updateComplete;
    chevron.getAnimations().forEach((animation) => { animation.finish(); });
    expect(toggle.getAttribute('aria-expanded')).to.equal('true');
    expect(getComputedStyle(chevron).transform, 'turned for an open card').to.not.equal('none');
    const [x, y] = centre(glyph);
    expect(x, 'it pivots on the glyph').to.be.closeTo(rest[0], 0.5);
    expect(y, 'it pivots on the glyph').to.be.closeTo(rest[1], 0.5);
  });

  it('rings inside its own box on keyboard focus, because the card clips what lies outside it', async () => {
    const {toggle} = await cardRow();
    toggle.focus();
    expect(toggle.matches(':focus-visible')).to.equal(true);

    const probe = document.createElement('div');
    probe.style.color = 'var(--color-control-ring)';
    document.body.append(probe);
    const ring = getComputedStyle(toggle);
    expect(ring.outlineStyle).to.equal('solid');
    expect(ring.outlineWidth).to.equal('2px');
    expect(ring.outlineColor).to.equal(getComputedStyle(probe).color);

    const reach = Number.parseFloat(ring.outlineOffset) + Number.parseFloat(ring.outlineWidth);
    const box = toggle.getBoundingClientRect();
    const card = toggle.closest('article')!.getBoundingClientRect();
    expect(box.left - reach, 'left').to.be.at.least(card.left);
    expect(box.right + reach, 'right').to.be.at.most(card.right);
    expect(box.top - reach, 'top').to.be.at.least(card.top);
    // The row is round enough for the ring to follow it.
    expect(Number.parseFloat(ring.borderTopLeftRadius)).to.be.greaterThan(0);
  });
});
