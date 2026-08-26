// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import type {DlImageLightbox} from './image_lightbox.ts';
import './image_lightbox.ts';

afterEach(() => { document.body.replaceChildren(); });

it('owns safe gallery navigation, Escape, ARIA, and focus restoration', async () => {
  const trigger = document.createElement('button');
  trigger.textContent = 'Open chart';
  const lightbox = document.createElement('dl-image-lightbox') as DlImageLightbox;
  const states: boolean[] = [];
  lightbox.addEventListener('dl-image-lightbox-state-change', (event) => {
    states.push(event.detail.open);
  });
  document.body.append(trigger, lightbox);
  trigger.focus();

  await lightbox.open('/first.png', trigger, ['/first.png', '/second.png']);

  expect(lightbox.getAttribute('role')).to.equal('dialog');
  expect(lightbox.getAttribute('aria-modal')).to.equal('true');
  expect(lightbox.getAttribute('aria-hidden')).to.equal('false');
  expect(document.activeElement).to.equal(lightbox);
  expect(states).to.deep.equal([true]);

  const backward = new KeyboardEvent('keydown', {
    key: 'Tab', shiftKey: true, bubbles: true, cancelable: true,
  });
  document.dispatchEvent(backward);
  expect(backward.defaultPrevented).to.equal(true);
  expect(document.activeElement).to.equal(
    lightbox.querySelector<HTMLButtonElement>('[aria-label="Next"]'),
  );

  document.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'ArrowLeft', bubbles: true, cancelable: true,
  }));
  await lightbox.updateComplete;
  expect(lightbox.querySelector<HTMLImageElement>('img')?.src)
    .to.equal(new URL('/second.png', location.origin).href);

  document.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Escape', bubbles: true, cancelable: true,
  }));
  await new Promise((resolve) => requestAnimationFrame(resolve));
  expect(lightbox.getAttribute('aria-hidden')).to.equal('true');
  expect(lightbox.inert).to.equal(true);
  expect(states).to.deep.equal([true, false]);
  expect(document.activeElement).to.equal(trigger);
});

it('does not restore stale focus when close is immediately followed by reopen', async () => {
  const firstTrigger = document.createElement('button');
  const secondTrigger = document.createElement('button');
  const lightbox = document.createElement('dl-image-lightbox') as DlImageLightbox;
  document.body.append(firstTrigger, secondTrigger, lightbox);

  await lightbox.open('/first.png', firstTrigger);
  lightbox.close();
  const reopened = lightbox.open('/second.png', secondTrigger);
  await reopened;
  await new Promise((resolve) => requestAnimationFrame(resolve));

  expect(lightbox.getAttribute('aria-hidden')).to.equal('false');
  expect(lightbox.inert).to.equal(false);
  expect(document.activeElement).not.to.equal(firstTrigger);
  expect(document.activeElement).to.equal(lightbox);
});
