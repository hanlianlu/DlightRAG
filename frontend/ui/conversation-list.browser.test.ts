// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** A conversation's actions menu keeps Delete in reach while the conversation is busy, and inert. */

import {expect} from '@esm-bundle/chai';
import {defineDesignSystemElements} from '../design-system/index.ts';
import {productionHandles} from '../stores/app-handles.ts';
import type {DlConversationList} from './conversation-list.ts';
import './conversation-list.ts';

defineDesignSystemElements();

const {conversations} = productionHandles();

async function frame(): Promise<void> {
  await new Promise<void>((resolve) => { requestAnimationFrame(() => { resolve(); }); });
  await new Promise<void>((resolve) => { setTimeout(resolve, 0); });
}

async function press(key: string, target = document.activeElement as HTMLElement): Promise<void> {
  target.dispatchEvent(new KeyboardEvent('keydown', {key, bubbles: true, composed: true, cancelable: true}));
  await frame();
}

afterEach(() => {
  conversations.dispose();
  document.body.replaceChildren();
});

it('keeps a busy conversation\'s Delete in reach but inert', async () => {
  conversations.upsertSummary({
    conversationId: 'menu-contract',
    title: 'Menu contract',
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z',
    forkedFromConversationId: null,
    forkedFromTitle: null,
  });
  const list = document.createElement('dl-conversation-list') as DlConversationList;
  document.body.appendChild(list);
  await list.updateComplete;
  const trigger = list.querySelector<HTMLElement>('.conversation-actions-button')!;
  const menu = (): HTMLElement | null => list.querySelector<HTMLElement>('dl-menu');
  const open = (): boolean => Boolean(menu()?.isConnected) && !menu()?.hidden;
  const deletes: string[] = [];
  list.addEventListener('dl-conversation-delete', (event) => { deletes.push(event.detail.conversationId); });
  list.busy = true;
  await list.updateComplete;

  await press('ArrowUp', trigger);
  const remove = document.activeElement as HTMLButtonElement;
  expect(remove.textContent?.trim(), 'the busy item still takes focus').to.equal('Delete');
  expect(remove.getAttribute('aria-disabled')).to.equal('true');
  expect(remove.disabled).to.equal(false);
  remove.click();
  await frame();
  expect(deletes).to.deep.equal([]);
  expect(open(), 'an inert item leaves the menu open').to.equal(true);

  list.busy = false;
  await list.updateComplete;
  expect(remove.hasAttribute('aria-disabled')).to.equal(false);
  remove.click();
  await frame();
  expect(deletes).to.deep.equal(['menu-contract']);
});
