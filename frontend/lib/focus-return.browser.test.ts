// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {FocusReturn} from './focus-return.ts';

const nextFrame = (): Promise<void> => new Promise((resolve) => requestAnimationFrame(() => resolve()));

function buttons(): [HTMLButtonElement, HTMLButtonElement] {
  const opener = document.createElement('button');
  const other = document.createElement('button');
  document.body.append(opener, other);
  return [opener, other];
}

afterEach(() => { document.body.replaceChildren(); });

it('gives focus back to the opener a frame after the surface closes', async () => {
  const [opener, other] = buttons();
  const focus = new FocusReturn();
  focus.remember(opener);
  other.focus();

  focus.restore(() => false);
  expect(document.activeElement).to.equal(other);
  await nextFrame();

  expect(document.activeElement).to.equal(opener);
});

it('remembers what holds focus when no opener is named', async () => {
  const [opener] = buttons();
  const focus = new FocusReturn();
  opener.focus();
  focus.remember();
  document.body.focus();

  focus.restore(() => false);
  await nextFrame();

  expect(document.activeElement).to.equal(opener);
});

it('does not restore stale focus when close is immediately followed by reopen', async () => {
  const [first, second] = buttons();
  const focus = new FocusReturn();
  focus.remember(first);

  focus.restore(() => false);
  focus.invalidate();
  focus.remember(second);
  second.focus();
  await nextFrame();

  expect(document.activeElement).to.equal(second);
});

it('does not move focus for a surface that is open again by the frame, or one that was discarded', async () => {
  const [opener, other] = buttons();
  const focus = new FocusReturn();
  other.focus();

  focus.remember(opener);
  focus.restore(() => true);
  await nextFrame();
  expect(document.activeElement, 'open again by the frame').to.equal(other);

  focus.remember(opener);
  focus.restore(() => false);
  focus.discard();
  await nextFrame();
  expect(document.activeElement, 'discarded before the frame').to.equal(other);
  expect(focus.target).to.equal(null);
});
