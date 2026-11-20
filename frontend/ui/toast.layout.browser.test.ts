// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The toast as the shipped stylesheets draw it (the product-styles page of the runner). */

import {expect} from '@esm-bundle/chai';
import {setViewport} from '@web/test-runner-commands';
import {seriousViolations} from '../testing/axe.ts';
import {mountToast, withUndo} from '../testing/toast.ts';
import type {DlToastRegion} from './toast.ts';

const originalViewport = {width: window.innerWidth, height: window.innerHeight};

afterEach(() => {
  document.body.replaceChildren();
});

describe('as the product styles it', () => {
  const LONG = `Could not save the change: ${'the server refused the connection and gave no reason '.repeat(6)}`
    + `https://example.com/${'a'.repeat(120)}`;
  before(async () => {
    await setViewport({width: 390, height: 844});
  });
  after(async () => {
    await setViewport(originalViewport);
  });

  const message = (toast: DlToastRegion): HTMLElement => toast.querySelector<HTMLElement>('.toast-message')!;

  it('wraps a long error inside the screen, where a long receipt keeps one line', async () => {
    const [error, receipt, oneLine] = [mountToast('error'), mountToast(), mountToast()];
    error.show({message: LONG});
    receipt.show({message: LONG});
    oneLine.show({message: 'Saved.'});
    await Promise.all([error.updateComplete, receipt.updateComplete, oneLine.updateComplete]);

    const box = error.getBoundingClientRect();
    expect(box.left).to.be.at.least(0);
    expect(box.right).to.be.at.most(window.innerWidth);
    expect(message(error).scrollWidth, 'nothing of it is clipped').to.be.at.most(message(error).clientWidth);
    expect(message(error).clientHeight, 'it takes many lines').to.be.at.least(3 * message(oneLine).clientHeight);
    expect(message(receipt).clientHeight, 'a receipt takes one').to.equal(message(oneLine).clientHeight);
    expect(receipt.getBoundingClientRect().width, 'as wide as the screen allows')
      .to.be.greaterThan(window.innerWidth / 2);
  });

  it('shows an error above a receipt without one covering the other', async () => {
    const error = mountToast('error');
    const receipt = mountToast();
    error.show({message: 'Could not save.'});
    receipt.show(withUndo('Saved elsewhere.'));
    await Promise.all([error.updateComplete, receipt.updateComplete]);

    expect(error.getBoundingClientRect().bottom).to.be.lessThan(receipt.getBoundingClientRect().top);
  });

  for (const colorMode of ['dark', 'light']) {
    it(`has no serious axe violations when it is open on the ${colorMode} theme`, async () => {
      // The page clips what lies outside the body, and axe skips what that clips: as in the app, the toast is
      // fixed inside a container as tall as the screen.
      const screen = document.body.appendChild(document.createElement('div'));
      screen.style.height = '100dvh';
      const error = mountToast('error', screen);
      error.show(withUndo(LONG));
      await error.updateComplete;

      // Axe would fetch the sheets again to read them, and cannot find the ones they import.
      expect(await seriousViolations(error, {colorModes: [colorMode], preload: false})).to.deep.equal([]);
    });
  }
});
