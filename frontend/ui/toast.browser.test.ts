// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import type {DlToastRegion} from './toast.ts';
import './toast.ts';

function mountToast(): DlToastRegion {
  const toast = document.createElement('dl-toast-region');
  toast.className = 'toast';
  document.body.appendChild(toast);
  return toast;
}

afterEach(() => {
  document.body.replaceChildren();
});

it('renders escaped text and settles an asynchronous public Undo command in place', async () => {
  const toast = mountToast();
  let calls = 0;
  toast.showAction('<img src=x> Remembered', {
    actionLabel: 'Undo',
    onAction: async () => {
      calls += 1;
      return 'Profile Memory change undone.';
    },
  });
  await toast.updateComplete;

  expect(toast.querySelector('img')).to.equal(null);
  expect(toast.textContent).to.contain('<img src=x> Remembered');
  toast.querySelector<HTMLButtonElement>('button')?.click();
  await new Promise((resolve) => setTimeout(resolve, 0));
  await toast.updateComplete;

  expect(calls).to.equal(1);
  expect(toast.textContent?.trim()).to.equal('Profile Memory change undone.');
  expect(toast.querySelector('button')).to.equal(null);
});

it('uses domain-neutral fallbacks when an action supplies no receipt', async () => {
  const toast = mountToast();
  toast.showAction('First change', {actionLabel: 'Undo', onAction: async () => {}});
  await toast.updateComplete;
  toast.querySelector<HTMLButtonElement>('button')?.click();
  await new Promise((resolve) => setTimeout(resolve, 0));
  await toast.updateComplete;
  expect(toast.textContent?.trim()).to.equal('Change undone.');

  toast.showAction('Second change', {
    actionLabel: 'Undo',
    onAction: async () => { throw new Error('conflict'); },
  });
  await toast.updateComplete;
  toast.querySelector<HTMLButtonElement>('button')?.click();
  await new Promise((resolve) => setTimeout(resolve, 0));
  await toast.updateComplete;
  expect(toast.textContent?.trim()).to.equal('Could not undo the change.');
});

it('replaces an actionable receipt with a plain command without stale controls', async () => {
  const toast = mountToast();
  toast.showAction('Remembered', {actionLabel: 'Undo', onAction: async () => {}});
  toast.show('Already remembered.');
  await toast.updateComplete;

  expect(toast.textContent?.trim()).to.equal('Already remembered.');
  expect(toast.querySelector('button')).to.equal(null);
});

it('does not resume while hover ends but keyboard focus remains inside', async () => {
  const toast = mountToast();
  toast.showAction('Remembered', {
    actionLabel: 'Undo',
    onAction: async () => {},
    duration: 40,
  });
  await toast.updateComplete;
  const action = toast.querySelector<HTMLButtonElement>('button')!;

  toast.dispatchEvent(new MouseEvent('mouseenter'));
  action.focus();
  toast.dispatchEvent(new MouseEvent('mouseleave'));
  await new Promise((resolve) => setTimeout(resolve, 60));
  await toast.updateComplete;

  expect(document.activeElement).to.equal(action);
  expect(toast.textContent).to.contain('Remembered');
  action.blur();
  await new Promise((resolve) => setTimeout(resolve, 60));
  await toast.updateComplete;
  expect(toast.textContent?.trim()).to.equal('');
});

it('pauses an actionable receipt while Shell modality makes it unreachable', async () => {
  const toast = mountToast();
  toast.showAction('Remembered', {
    actionLabel: 'Undo',
    onAction: async () => {},
    duration: 40,
  });
  await toast.updateComplete;
  expect(toast.inert).to.equal(false);

  toast.shellInert = true;
  await toast.updateComplete;
  expect(toast.inert).to.equal(true);
  await new Promise((resolve) => setTimeout(resolve, 60));
  await toast.updateComplete;
  expect(toast.textContent).to.contain('Remembered');
  expect(toast.querySelector('button')?.textContent).to.equal('Undo');

  toast.shellInert = false;
  await toast.updateComplete;
  expect(toast.inert).to.equal(false);
  await new Promise((resolve) => setTimeout(resolve, 60));
  await toast.updateComplete;

  expect(toast.inert).to.equal(true);
  expect(toast.querySelector('button')).to.equal(null);
  expect(toast.textContent?.trim()).to.equal('');
});
