// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import type {DlToastRegion} from './toast.ts';
import './toast.ts';

const originalSetTimeout = window.setTimeout;

function mountToast(): DlToastRegion {
  const toast = document.createElement('dl-toast-region');
  toast.className = 'toast';
  document.body.appendChild(toast);
  return toast;
}

/** A receipt's three seconds pass in the next task; the ids stay real, so pausing still clears them. */
function receiptsExpireAtOnce(): void {
  window.setTimeout = ((handler: TimerHandler) => originalSetTimeout(handler, 0)) as typeof window.setTimeout;
}

afterEach(() => {
  window.setTimeout = originalSetTimeout;
  document.body.replaceChildren();
});

it('keeps a receipt for three seconds, and the answer to its Undo as long', async () => {
  const toast = mountToast();
  const delays: (number | undefined)[] = [];
  window.setTimeout = ((handler: TimerHandler, delay?: number) => {
    delays.push(delay);
    return originalSetTimeout(handler, 0);
  }) as typeof window.setTimeout;

  toast.show({message: 'Plain receipt'});
  toast.show({message: 'Undo available', action: {actionLabel: 'Undo', onAction: async () => {}}});
  await toast.updateComplete;
  toast.querySelector<HTMLButtonElement>('button')!.click();
  await new Promise((resolve) => originalSetTimeout(resolve, 0));

  expect(delays).to.deep.equal([3000, 3000, 3000]);
});

it('puts focus on the action when the receipt asks for it, and only then', async () => {
  const toast = mountToast();

  toast.show({message: 'Plain change', action: {actionLabel: 'Undo', onAction: async () => {}}});
  await toast.updateComplete;
  expect(document.activeElement).to.equal(document.body);

  toast.show({message: 'Forgot: one', action: {actionLabel: 'Undo', onAction: async () => {}, focus: true}});
  await toast.updateComplete;
  expect(document.activeElement).to.equal(toast.querySelector('.toast-action'));
  expect(toast.inert).to.equal(false);

  // A receipt that replaced it first has nothing of the earlier one's to focus.
  (document.activeElement as HTMLElement).blur();
  toast.show({message: 'Asked to focus', action: {actionLabel: 'Undo', onAction: async () => {}, focus: true}});
  toast.show({message: 'Replaced at once'});
  await toast.updateComplete;
  expect(document.activeElement).to.equal(document.body);
});

it('renders escaped text and settles an asynchronous public Undo command in place', async () => {
  const toast = mountToast();
  let calls = 0;
  toast.show({
    message: '<img src=x> Remembered',
    action: {
      actionLabel: 'Undo',
      onAction: async () => {
        calls += 1;
        return 'Profile Memory change undone.';
      },
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
  toast.show({message: 'First change', action: {actionLabel: 'Undo', onAction: async () => {}}});
  await toast.updateComplete;
  toast.querySelector<HTMLButtonElement>('button')?.click();
  await new Promise((resolve) => setTimeout(resolve, 0));
  await toast.updateComplete;
  expect(toast.textContent?.trim()).to.equal('Change undone.');

  toast.show({
    message: 'Second change',
    action: {
      actionLabel: 'Undo',
      onAction: async () => { throw new Error('conflict'); },
    },
  });
  await toast.updateComplete;
  toast.querySelector<HTMLButtonElement>('button')?.click();
  await new Promise((resolve) => setTimeout(resolve, 0));
  await toast.updateComplete;
  expect(toast.textContent?.trim()).to.equal('Could not undo the change.');
});

it('replaces an actionable receipt with a plain command without stale controls', async () => {
  const toast = mountToast();
  toast.show({message: 'Remembered', action: {actionLabel: 'Undo', onAction: async () => {}}});
  toast.show({message: 'Already remembered.'});
  await toast.updateComplete;

  expect(toast.textContent?.trim()).to.equal('Already remembered.');
  expect(toast.querySelector('button')).to.equal(null);
});

it('does not resume while hover ends but keyboard focus remains inside', async () => {
  const toast = mountToast();
  receiptsExpireAtOnce();
  toast.show({message: 'Remembered', action: {actionLabel: 'Undo', onAction: async () => {}}});
  await toast.updateComplete;
  const action = toast.querySelector<HTMLButtonElement>('button')!;

  toast.dispatchEvent(new MouseEvent('mouseenter'));
  action.focus();
  toast.dispatchEvent(new MouseEvent('mouseleave'));
  await new Promise((resolve) => setTimeout(resolve, 0));
  await toast.updateComplete;

  expect(toast.textContent).to.contain('Remembered');
  expect(document.activeElement).to.equal(action);
  action.blur();
  await new Promise((resolve) => setTimeout(resolve, 0));
  await toast.updateComplete;
  expect(toast.textContent?.trim()).to.equal('');
});

it('pauses an actionable receipt while Shell modality makes it unreachable', async () => {
  const toast = mountToast();
  receiptsExpireAtOnce();
  toast.show({message: 'Remembered', action: {actionLabel: 'Undo', onAction: async () => {}}});
  await toast.updateComplete;
  expect(toast.inert).to.equal(false);

  toast.shellInert = true;
  await toast.updateComplete;
  expect(toast.inert).to.equal(true);
  await new Promise((resolve) => setTimeout(resolve, 0));
  await toast.updateComplete;
  expect(toast.textContent).to.contain('Remembered');
  expect(toast.querySelector('button')?.textContent).to.equal('Undo');

  toast.shellInert = false;
  await toast.updateComplete;
  expect(toast.inert).to.equal(false);
  await new Promise((resolve) => setTimeout(resolve, 0));
  await toast.updateComplete;

  expect(toast.inert).to.equal(true);
  expect(toast.querySelector('button')).to.equal(null);
  expect(toast.textContent?.trim()).to.equal('');
});
