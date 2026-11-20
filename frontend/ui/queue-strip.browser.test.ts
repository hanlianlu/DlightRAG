// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The queue strip: what it reads, and the reader's own corrections to held messages. */

import {expect} from '@esm-bundle/chai';
import {waitFor} from '../testing/dom.ts';
import type {DlQueueStrip} from './queue-strip.ts';
import './queue-strip.ts';

const originalFetch = window.fetch;

const wire = (id: string, position: number, content: string) => ({
  message_id: id,
  content,
  position,
  created_at: '2026-01-01T00:00:00Z',
});

function serve(initial: ReturnType<typeof wire>[]) {
  let items = [...initial];
  const requests: string[] = [];
  window.fetch = async (input, init) => {
    const request = new Request(new URL(String(input), 'http://localhost'), init);
    requests.push(`${request.method} ${new URL(request.url).pathname}`);
    const messageId = new URL(request.url).pathname.split('/').at(-1) ?? '';
    if (request.method === 'DELETE') {
      items = items.filter((item) => item.message_id !== messageId);
      return new Response(null, {status: 204});
    }
    if (request.method === 'PATCH') {
      const body = await request.clone().json() as {content: string};
      const changed = items.find((item) => item.message_id === messageId)!;
      changed.content = body.content;
      return Response.json({message: changed});
    }
    if (request.method === 'PUT') {
      const body = await request.clone().json() as {message_ids: string[]};
      items = body.message_ids.map((id, index) => {
        const stored = items.find((item) => item.message_id === id)!;
        return {...stored, position: index + 1};
      });
      return Response.json({items});
    }
    return Response.json({items});
  };
  return requests;
}

async function settle(strip: DlQueueStrip): Promise<void> {
  await strip.updateComplete;
  await new Promise((resolve) => setTimeout(resolve, 0));
  await strip.updateComplete;
}

async function mount(items: ReturnType<typeof wire>[], conversationId = 'c-1') {
  const requests = serve(items);
  const strip = document.createElement('dl-queue-strip') as DlQueueStrip;
  strip.conversationId = conversationId;
  document.body.appendChild(strip);
  await settle(strip);
  return {strip, requests};
}

afterEach(() => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
});

it('draws nothing while no message waits', async () => {
  const {strip} = await mount([]);
  expect(strip.querySelector('.region')).to.equal(null);
});

it('collapses to one line and opens the held words the reader can edit', async () => {
  const {strip} = await mount([
    wire('message-a', 1, 'first correction'),
    wire('message-b', 2, 'second correction'),
  ]);

  const summary = strip.querySelector<HTMLButtonElement>('.summary')!;
  expect(summary.textContent).to.contain('2 queued after this run');
  expect(summary.getAttribute('aria-expanded')).to.equal('false');
  expect(strip.querySelector('.list')).to.equal(null);

  summary.click();
  await settle(strip);

  expect(summary.getAttribute('aria-expanded')).to.equal('true');
  expect([...strip.querySelectorAll('.content')].map((row) => row.textContent?.trim())).to.deep.equal([
    'first correction',
    'second correction',
  ]);
});

it('rewrites one held message and keeps the reader on the row', async () => {
  const {strip, requests} = await mount([wire('message-a', 1, 'first correction')]);
  strip.querySelector<HTMLButtonElement>('.summary')!.click();
  await settle(strip);

  strip.querySelector<HTMLButtonElement>('[aria-label="Edit"]')!.click();
  await strip.updateComplete;
  const input = strip.querySelector<HTMLInputElement>('.input')!;
  input.value = 'corrected words';
  input.dispatchEvent(new InputEvent('input', {bubbles: true}));
  input.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', bubbles: true}));
  // The correction is a request and then a redraw, so it is waited for, not guessed at in one tick.
  await waitFor(() => strip.querySelector('.content')?.textContent?.trim() === 'corrected words');

  expect(requests).to.contain('PATCH /web/api/conversations/c-1/queue/message-a');
});

it('reorders held messages and drops one', async () => {
  const {strip, requests} = await mount([
    wire('message-a', 1, 'first'),
    wire('message-b', 2, 'second'),
  ]);
  strip.querySelector<HTMLButtonElement>('.summary')!.click();
  await settle(strip);

  const held = (): Array<string | undefined> =>
    [...strip.querySelectorAll('.content')].map((row) => row.textContent?.trim());
  strip.querySelectorAll<HTMLButtonElement>('[aria-label="Move later"]')[0]!.click();
  await waitFor(() => held()[0] === 'second');
  expect(requests).to.contain('PUT /web/api/conversations/c-1/queue/order');
  expect(held()).to.deep.equal(['second', 'first']);

  strip.querySelector<HTMLButtonElement>('[aria-label="Remove"]')!.click();
  await waitFor(() => strip.querySelectorAll('.row').length === 1);
  expect(requests).to.contain('DELETE /web/api/conversations/c-1/queue/message-b');
});
