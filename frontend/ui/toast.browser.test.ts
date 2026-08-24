// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {showActionToast, showToast} from './toast.ts';

function mountToast(): HTMLElement {
  const toast = document.createElement('div');
  toast.id = 'toast';
  toast.className = 'toast';
  document.body.appendChild(toast);
  return toast;
}

afterEach(() => {
  document.body.replaceChildren();
});

it('renders escaped text and settles an asynchronous Undo in place', async () => {
  const toast = mountToast();
  let calls = 0;
  showActionToast('<img src=x> Remembered', {
    actionLabel: 'Undo',
    onAction: async () => {
      calls += 1;
      return 'Profile Memory change undone.';
    },
  });

  expect(toast.querySelector('img')).to.equal(null);
  expect(toast.textContent).to.contain('<img src=x> Remembered');
  (toast.querySelector('button') as HTMLButtonElement).click();
  await new Promise((resolve) => setTimeout(resolve, 0));

  expect(calls).to.equal(1);
  expect(toast.textContent).to.equal('Profile Memory change undone.');
  expect(toast.querySelector('button')).to.equal(null);
});

it('replaces an actionable receipt with a plain toast without stale controls', () => {
  const toast = mountToast();
  showActionToast('Remembered', {actionLabel: 'Undo', onAction: async () => {}});

  showToast('Already remembered.');

  expect(toast.textContent).to.equal('Already remembered.');
  expect(toast.querySelector('button')).to.equal(null);
});
