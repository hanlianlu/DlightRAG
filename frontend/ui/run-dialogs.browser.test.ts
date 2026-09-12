// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import './run-dialogs.ts';
import type {ChildControlReceipt, ChildObservation} from '../api/conversations.ts';
import type {ChildRosterEntry, DlChildrenRoster} from './run-dialogs.ts';

function entry(id: string, status = 'succeeded'): ChildRosterEntry {
  return {childSessionId: id, status, objective: `objective ${id}`};
}

function deferredPage() {
  let resolve!: (page: {children: ChildRosterEntry[]; nextCursor: string | null}) => void;
  const promise = new Promise<{children: ChildRosterEntry[]; nextCursor: string | null}>(
    (done) => { resolve = done; },
  );
  return {promise, resolve};
}

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error('condition did not become true');
}

function roster(): DlChildrenRoster {
  const element = document.createElement('dl-children-roster') as DlChildrenRoster;
  document.body.appendChild(element);
  return element;
}

afterEach(() => {
  document.body.replaceChildren();
});

it('legacy fetcher renders every child without a paging control', async () => {
  const panel = roster();
  panel.open(async () => [entry('a'), entry('b')]);
  await waitFor(() => panel.querySelectorAll('li[role="listitem"]').length === 2);

  expect([...panel.querySelectorAll('li[role="listitem"]')].map((li) => li.textContent?.trim()))
    .to.deep.equal(['succeeded: objective a', 'succeeded: objective b']);
  expect(panel.querySelector('[data-load-older-children]')).to.equal(null);
});

it('paged roster renders the newest page and appends older pages with dedup', async () => {
  const older = deferredPage();
  let olderRequests = 0;
  const panel = roster();
  panel.open(
    async () => [entry('newest')],
    async (cursor) => {
      if (cursor === null) {
        return {children: [entry('newest')], nextCursor: 'older-1'};
      }
      expect(cursor).to.equal('older-1');
      olderRequests += 1;
      return older.promise;
    },
  );
  await waitFor(() => panel.querySelectorAll('li[role="listitem"]').length === 1);

  const button = panel.querySelector<HTMLButtonElement>('[data-load-older-children]')!;
  expect(button.textContent?.trim()).to.equal('Load older children');
  button.click();
  const flight = panel.loadOlderChildren();
  expect(panel.loadOlderChildren()).to.equal(flight);
  await panel.updateComplete;
  expect(button.disabled).to.equal(true);
  expect(button.getAttribute('aria-busy')).to.equal('true');
  expect(olderRequests).to.equal(1);

  older.resolve({
    children: [entry('newest'), entry('older')],
    nextCursor: null,
  });
  await flight;
  await panel.updateComplete;

  expect([...panel.querySelectorAll('li[role="listitem"]')].map((li) => li.textContent?.trim()))
    .to.deep.equal(['succeeded: objective newest', 'succeeded: objective older']);
  expect(panel.querySelector('[data-load-older-children]')).to.equal(null);
  expect(panel.querySelector('[data-roster-status]')?.textContent).to.contain(
    'Loaded 1 older child.',
  );
});

it('older-page failure keeps loaded rows and stays retryable', async () => {
  let attempts = 0;
  const panel = roster();
  panel.open(
    async () => [entry('newest')],
    async (cursor) => {
      if (cursor === null) {
        return {children: [entry('newest')], nextCursor: 'older-1'};
      }
      expect(cursor).to.equal('older-1');
      attempts += 1;
      if (attempts === 1) throw new Error('unavailable');
      return {children: [entry('older')], nextCursor: null};
    },
  );
  await waitFor(() => panel.querySelectorAll('li[role="listitem"]').length === 1);

  await panel.loadOlderChildren();
  await panel.updateComplete;
  expect(panel.querySelector('[data-load-older-children]')?.textContent).to.contain(
    'Retry loading older children',
  );

  await panel.loadOlderChildren();
  await panel.updateComplete;
  expect(panel.querySelectorAll('li[role="listitem"]')).to.have.length(2);
  expect(panel.querySelector('[data-load-older-children]')).to.equal(null);
});

it('refresh resets the traversal and rejects a late older response', async () => {
  const older = deferredPage();
  let firstPage = 0;
  const panel = roster();
  panel.open(
    async () => [entry('fresh')],
    async (cursor) => {
      if (cursor === null) {
        firstPage += 1;
        return {children: [entry('fresh')], nextCursor: 'older-1'};
      }
      return older.promise;
    },
  );
  await waitFor(() => firstPage === 1);

  const flight = panel.loadOlderChildren();
  await panel.refresh();
  await waitFor(() => firstPage === 2);
  older.resolve({children: [entry('stale')], nextCursor: null});
  await flight;
  await panel.updateComplete;

  expect([...panel.querySelectorAll('li[role="listitem"]')].map((li) => li.textContent?.trim()))
    .to.deep.equal(['succeeded: objective fresh']);
  expect(panel.querySelector('[data-load-older-children]')?.textContent).to.contain(
    'Load older children',
  );
});

it('closing the dialog aborts in-flight pages and resets paging state', async () => {
  const older = deferredPage();
  const panel = roster();
  panel.open(
    async () => [entry('newest')],
    async (cursor) => (cursor === null
      ? {children: [entry('newest')], nextCursor: 'older-1'}
      : older.promise),
  );
  await waitFor(() => panel.querySelectorAll('li[role="listitem"]').length === 1);

  const flight = panel.loadOlderChildren();
  const dialog = panel.querySelector<HTMLDialogElement>('dialog')!;
  dialog.close();
  // The native close event is a queued task; wait until the reset is visible.
  await waitFor(() => panel.querySelectorAll('li[role="listitem"]').length === 0);
  older.resolve({children: [entry('stale')], nextCursor: null});
  await flight;
  await panel.updateComplete;

  expect(panel.querySelectorAll('li[role="listitem"]')).to.have.length(0);
  expect(panel.querySelector('[data-load-older-children]')).to.equal(null);
  expect(panel.querySelector('.roster-list')?.textContent).to.contain(
    'No child agents were started.',
  );
});

function observation(status = 'running'): ChildObservation {
  return {
    runId: 'run-1',
    child: {
      childSessionId: 'newest',
      status,
      objective: 'objective newest',
      modelRole: 'query',
      usage: null,
      operationId: 'op-1',
      operationSequence: 1,
      operationStatus: status,
      cancellationOrigin: status === 'cancelled' ? 'user' : null,
      summary: 'working',
      resultHandles: ['ev-1'],
    },
    transcript: [{role: 'user', content: 'inspect', toolCalls: [], toolCallId: '', name: '', isError: false}],
    controls: [{
      controlSequence: 3, kind: 'steer', content: 'focus', origin: 'user',
      consumed: false, consumedAt: null, createdAt: null, operationId: 'op-1',
    }],
    questions: [{
      requestId: 'req-1', question: 'Which source?', status: 'pending',
      reply: null, replyOrigin: null, expiresAt: null, createdAt: null,
    }],
    result: {status, summary: 'working', handles: ['ev-1'], operationId: 'op-1'},
  };
}

it('selecting a child shows lineage and posts a queued steer', async () => {
  const steered: string[] = [];
  const panel = roster();
  panel.open(
    async () => [entry('newest', 'running')],
    async () => ({children: [entry('newest', 'running')], nextCursor: null}),
    {
      runId: 'run-1',
      observe: async () => observation('running'),
      control: async (_child, action, content) => {
        steered.push(`${action}:${content}`);
        return {
          runId: 'run-1', childSessionId: 'newest', action, outcome: 'queued',
          operationId: 'op-1', operationSequence: 1, controlSequence: 4, consumedAt: null,
          requestId: null,
        };
      },
    },
  );
  await waitFor(() => Boolean(panel.querySelector('[data-child-session="newest"]')));
  panel.querySelector<HTMLButtonElement>('[data-child-session="newest"]')!.click();
  await waitFor(() => Boolean(panel.querySelector('[name="instruction"]')));

  expect(panel.textContent).to.contain('Queued');
  expect(panel.textContent).to.contain('Which source?');
  expect(panel.textContent).to.contain('ev-1');

  const instruction = panel.querySelector<HTMLTextAreaElement>('[name="instruction"]')!;
  instruction.value = 'focus on dates';
  instruction.closest('form')!.dispatchEvent(new Event('submit', {bubbles: true, cancelable: true}));
  await waitFor(() => steered.length === 1);
  expect(steered[0]).to.equal('steer:focus on dates');
  await waitFor(() => (panel.textContent || '').includes('has not necessarily followed'));
});

it('rejected terminal steer stays explicit and does not look like success', async () => {
  const panel = roster();
  panel.open(
    async () => [entry('newest', 'running')],
    async () => ({children: [entry('newest', 'running')], nextCursor: null}),
    {
      runId: 'run-1',
      observe: async () => observation('running'),
      control: async () => {
        const error = new Error('terminal_child') as Error & {status: number; outcome: string};
        error.status = 409;
        error.outcome = 'terminal_child';
        throw error;
      },
    },
  );
  await waitFor(() => Boolean(panel.querySelector('[data-child-session="newest"]')));
  panel.querySelector<HTMLButtonElement>('[data-child-session="newest"]')!.click();
  await waitFor(() => Boolean(panel.querySelector('[name="instruction"]')));
  const instruction = panel.querySelector<HTMLTextAreaElement>('[name="instruction"]')!;
  instruction.value = 'too late';
  instruction.closest('form')!.dispatchEvent(new Event('submit', {bubbles: true, cancelable: true}));
  await waitFor(() => (panel.textContent || '').includes('already terminal'));
});

async function until(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  throw new Error('condition did not become true');
}

function observationFor(id: string, status = 'running'): ChildObservation {
  return {
    runId: 'run-1',
    child: {
      childSessionId: id,
      status,
      objective: `objective ${id}`,
      modelRole: 'query',
      usage: null,
      operationId: `op-${id}`,
      operationSequence: 1,
      operationStatus: status,
      cancellationOrigin: null,
      summary: `${id} working`,
      resultHandles: [],
    },
    transcript: [],
    controls: [],
    questions: [{
      requestId: `req-${id}`,
      question: `Question from ${id}?`,
      status: 'pending',
      reply: null,
      replyOrigin: null,
      expiresAt: null,
      createdAt: null,
    }],
    result: {status, summary: `${id} working`, handles: [], operationId: `op-${id}`},
  };
}

async function openInteractive(
  control: NonNullable<Parameters<DlChildrenRoster['open']>[2]>['control'],
  reply?: NonNullable<Parameters<DlChildrenRoster['open']>[2]>['reply'],
): Promise<DlChildrenRoster> {
  const panel = roster();
  panel.open(
    async () => [entry('a', 'running'), entry('b', 'running')],
    async () => ({children: [entry('a', 'running'), entry('b', 'running')], nextCursor: null}),
    {
      runId: 'run-1',
      observe: async (id) => observationFor(id),
      control,
      reply,
    },
  );
  await until(() => Boolean(panel.querySelector('[data-child-session="a"]')));
  panel.querySelector<HTMLButtonElement>('[data-child-session="a"]')!.click();
  await until(() => Boolean(panel.querySelector('[name="instruction"]')));
  return panel;
}

it('preserves steer draft and focus across an SSE observation refresh', async () => {
  let observes = 0;
  const panel = roster();
  panel.open(
    async () => [entry('a', 'running'), entry('b', 'running')],
    async () => ({children: [entry('a', 'running'), entry('b', 'running')], nextCursor: null}),
    {
      runId: 'run-1',
      observe: async (id) => {
        observes += 1;
        return observationFor(id);
      },
      control: async () => ({outcome: 'queued'} as never),
    },
  );
  await until(() => Boolean(panel.querySelector('[data-child-session="a"]')));
  panel.querySelector<HTMLButtonElement>('[data-child-session="a"]')!.click();
  await until(() => Boolean(panel.querySelector('[name="instruction"]')));
  const afterSelect = observes;

  const before = panel.querySelector<HTMLTextAreaElement>('[name="instruction"]')!;
  before.value = 'do not lose this user decision';
  before.focus();
  expect(document.activeElement).to.equal(before);

  panel.refreshIfFollowing('run-1');
  await until(() => observes > afterSelect);
  await panel.updateComplete;

  const after = panel.querySelector<HTMLTextAreaElement>('[name="instruction"]');
  expect(after).to.not.equal(null);
  expect(after?.value).to.equal('do not lose this user decision');
  expect(document.activeElement).to.equal(after);
});

it('does not restore a stale capture over text typed during an in-flight refresh', async () => {
  let release!: () => void;
  const gate = new Promise<void>((resolve) => { release = resolve; });
  let observes = 0;
  let block = false;
  const panel = roster();
  panel.open(
    async () => [entry('a', 'running')],
    async () => ({children: [entry('a', 'running')], nextCursor: null}),
    {
      runId: 'run-1',
      observe: async (id) => {
        observes += 1;
        if (block) await gate;
        return observationFor(id);
      },
      control: async () => ({outcome: 'queued'} as never),
    },
  );
  await until(() => Boolean(panel.querySelector('[data-child-session="a"]')));
  panel.querySelector<HTMLButtonElement>('[data-child-session="a"]')!.click();
  await until(() => Boolean(panel.querySelector('[name="instruction"]')));

  const input = panel.querySelector<HTMLTextAreaElement>('[name="instruction"]')!;
  input.value = 'hello';
  input.focus();
  const afterSelect = observes;
  block = true;
  panel.refreshIfFollowing('run-1');
  panel.refreshIfFollowing('run-1');
  panel.refreshIfFollowing('run-1');
  await until(() => observes === afterSelect + 1);
  input.value = 'hello world';
  release();
  await until(() => observes >= afterSelect + 1);
  await panel.updateComplete;
  await new Promise((resolve) => setTimeout(resolve, 20));

  expect(observes).to.be.at.most(afterSelect + 2);
  const after = panel.querySelector<HTMLTextAreaElement>('[name="instruction"]');
  expect(after?.value).to.equal('hello world');
  expect(document.activeElement).to.equal(after);
});

it('keeps child B free of child A\'s late steer receipt and busy state', async () => {
  let resolve!: (receipt: {
    runId: string; childSessionId: string; action: string; outcome: string;
    operationId: string; operationSequence: number; controlSequence: number;
    consumedAt: null; requestId: null;
  }) => void;
  const pending = new Promise<Parameters<typeof resolve>[0]>((done) => { resolve = done; });
  const calls: string[] = [];
  const panel = await openInteractive(async (id) => {
    calls.push(id);
    return pending;
  });

  const input = panel.querySelector<HTMLTextAreaElement>('[name="instruction"]')!;
  input.value = 'only child A';
  input.closest('form')!.requestSubmit();
  await until(() => calls.length === 1);

  panel.querySelector<HTMLButtonElement>('[data-child-session="b"]')!.click();
  await until(() => {
    const current = panel.querySelector<HTMLTextAreaElement>('[name="instruction"]');
    return Boolean(current && current !== input);
  });
  const bInput = panel.querySelector<HTMLTextAreaElement>('[name="instruction"]')!;
  expect(bInput.disabled).to.equal(false);
  bInput.value = 'child B draft';

  resolve({
    runId: 'run-1', childSessionId: 'a', action: 'steer', outcome: 'queued',
    operationId: 'op-a', operationSequence: 1, controlSequence: 4,
    consumedAt: null, requestId: null,
  });
  await new Promise((done) => setTimeout(done, 30));
  await panel.updateComplete;

  expect(calls[0]).to.equal('a');
  expect(panel.querySelector('[data-child-session="b"]')?.getAttribute('aria-current')).to.equal('true');
  expect(panel.textContent).to.not.contain('has not necessarily followed');
  expect(panel.querySelector<HTMLTextAreaElement>('[name="instruction"]')?.value).to.equal('child B draft');
  expect(panel.querySelector<HTMLTextAreaElement>('[name="instruction"]')?.disabled).to.equal(false);
});

it('does not attach a late receipt to a closed and reopened roster dialog', async () => {
  let resolve!: (receipt: {
    runId: string; childSessionId: string; action: string; outcome: string;
    operationId: string; operationSequence: number; controlSequence: number;
    consumedAt: null; requestId: null;
  }) => void;
  const pending = new Promise<Parameters<typeof resolve>[0]>((done) => { resolve = done; });
  const panel = await openInteractive(async () => pending);

  const input = panel.querySelector<HTMLTextAreaElement>('[name="instruction"]')!;
  input.value = 'only child A';
  input.closest('form')!.requestSubmit();
  await until(() => Boolean(panel.querySelector('textarea[name="instruction"][disabled], textarea[name="instruction"][aria-disabled]')));

  panel.querySelector<HTMLDialogElement>('dialog')!.close();
  await until(() => panel.querySelectorAll('li[role="listitem"]').length === 0);

  panel.open(
    async () => [entry('a', 'running'), entry('b', 'running')],
    async () => ({children: [entry('a', 'running'), entry('b', 'running')], nextCursor: null}),
    {
      runId: 'run-1',
      observe: async (id) => observationFor(id),
      control: async () => pending,
    },
  );
  await until(() => Boolean(panel.querySelector('[data-child-session="a"]')));
  panel.querySelector<HTMLButtonElement>('[data-child-session="a"]')!.click();
  await until(() => Boolean(panel.querySelector('[name="instruction"]')));

  resolve({
    runId: 'run-1', childSessionId: 'a', action: 'steer', outcome: 'queued',
    operationId: 'op-a', operationSequence: 1, controlSequence: 4,
    consumedAt: null, requestId: null,
  });
  await new Promise((done) => setTimeout(done, 30));
  await panel.updateComplete;

  expect(panel.textContent).to.not.contain('has not necessarily followed');
  expect(panel.querySelector<HTMLTextAreaElement>('[name="instruction"]')?.disabled).to.equal(false);
});

it('keeps a late guidance reply correlated to the requesting child', async () => {
  let resolve!: (receipt: {
    runId: string; childSessionId: string; action: string; outcome: string;
    operationId: null; operationSequence: null; controlSequence: null;
    consumedAt: null; requestId: string;
  }) => void;
  const pending = new Promise<Parameters<typeof resolve>[0]>((done) => { resolve = done; });
  const replies: string[] = [];
  const panel = await openInteractive(
    async () => ({outcome: 'queued'} as never),
    async (requestId) => {
      replies.push(requestId);
      return pending;
    },
  );

  const reply = panel.querySelector<HTMLTextAreaElement>('[name="reply"]')!;
  reply.value = 'use the report';
  reply.closest('form')!.requestSubmit();
  await until(() => replies.length === 1);
  expect(replies[0]).to.equal('req-a');

  panel.querySelector<HTMLButtonElement>('[data-child-session="b"]')!.click();
  await until(() => {
    const current = panel.querySelector<HTMLTextAreaElement>('[name="reply"]');
    return Boolean(current && current !== reply);
  });

  resolve({
    runId: 'run-1', childSessionId: 'a', action: 'reply', outcome: 'replied',
    operationId: null, operationSequence: null, controlSequence: null,
    consumedAt: null, requestId: 'req-a',
  });
  await new Promise((done) => setTimeout(done, 30));
  await panel.updateComplete;

  expect(panel.querySelector('[data-child-session="b"]')?.getAttribute('aria-current')).to.equal('true');
  expect(panel.textContent).to.not.contain('Reply sent.');
  expect(panel.querySelector<HTMLTextAreaElement>('[name="reply"]')?.disabled).to.equal(false);
});

it('does not leak a steer draft from child A onto child B', async () => {
  const panel = await openInteractive(async () => ({outcome: 'queued'} as never));
  const aInput = panel.querySelector<HTMLTextAreaElement>('[name="instruction"]')!;
  aInput.value = 'only for A';

  panel.querySelector<HTMLButtonElement>('[data-child-session="b"]')!.click();
  await until(() => {
    const current = panel.querySelector<HTMLTextAreaElement>('[name="instruction"]');
    return Boolean(current && current !== aInput);
  });
  expect(panel.querySelector<HTMLTextAreaElement>('[name="instruction"]')?.value).to.equal('');

  panel.querySelector<HTMLButtonElement>('[data-child-session="a"]')!.click();
  await until(() => {
    const current = panel.querySelector<HTMLTextAreaElement>('[name="instruction"]');
    return Boolean(current && current.value === 'only for A');
  });
  expect(panel.querySelector<HTMLTextAreaElement>('[name="instruction"]')?.value).to.equal('only for A');
});

async function changingObservation(
  control: NonNullable<Parameters<DlChildrenRoster['open']>[2]>['control'],
  reply?: NonNullable<Parameters<DlChildrenRoster['open']>[2]>['reply'],
  status = 'running',
) {
  let current = observationFor('a', status);
  const panel = roster();
  panel.open(async () => [current.child], undefined, {
    runId: 'run-1', observe: async () => current, control, reply,
  });
  await until(() => Boolean(panel.querySelector('[data-child-session="a"]')));
  panel.querySelector<HTMLButtonElement>('[data-child-session="a"]')!.click();
  await until(() => Boolean(panel.querySelector('[name="instruction"]')));
  return {
    panel,
    async set(operationId: string, requestId = 'req-a', nextStatus = status, previousRequestStatus?: string) {
      current = observationFor('a', nextStatus);
      if (previousRequestStatus) {
        current.questions.push({...current.questions[0]!, status: previousRequestStatus});
      }
      current.child.operationId = operationId;
      current.child.cancellationOrigin = nextStatus === 'cancelled' ? 'user' : null;
      current.questions[0]!.requestId = requestId;
      await panel.refresh();
      await panel.updateComplete;
    },
  };
}

function commandReceipt(action: string, operationId = 'op-a', requestId: string | null = null): ChildControlReceipt {
  return {
    runId: 'run-1', childSessionId: 'a', action,
    outcome: action === 'reply' ? 'replied' : action === 'continue' ? 'accepted' : 'queued',
    operationId, operationSequence: 1, controlSequence: null, consumedAt: null, requestId,
  };
}

for (const kind of ['steer', 'continue', 'reply'] as const) {
  it(`keys native ${kind} editors across repeated Operation/request changes without draft cross-talk`, async () => {
    const state = await changingObservation(
      async () => commandReceipt(kind), async () => commandReceipt('reply'),
      kind === 'continue' ? 'cancelled' : 'running',
    );
    const {panel} = state;
    await state.set('op-a');
    const input = () => panel.querySelector<HTMLTextAreaElement>(`[data-editor="${kind}"] textarea`)!;
    const box = () => panel.querySelector<HTMLInputElement>('[name="reauthorize"]');
    const a = input();
    a.value = 'only A';
    if (box()) box()!.checked = true;
    a.focus();
    a.setSelectionRange(1, 4);
    await state.set('op-a');
    expect(input()).to.equal(a);
    expect(document.activeElement).to.equal(a);
    expect([a.selectionStart, a.selectionEnd]).to.deep.equal([1, 4]);
    expect(a.value).to.equal('only A');
    for (const id of ['b', 'c', 'd']) {
      await state.set(kind === 'reply' ? 'op-a' : `op-${id}`, `req-${id}`);
      expect(input()).to.not.equal(a);
      expect(input().value).to.equal('');
      if (box()) expect(box()!.checked).to.equal(false);
      input().value = `only ${id}`;
    }
    await state.set('op-a');
    expect(input().value).to.equal('only A');
    if (box()) expect(box()!.checked).to.equal(true);
  });
}

for (const action of ['steer', 'continue', 'cancel', 'reply'] as const) {
  it(`fences concurrent same-child ${action} commands by current Operation/request`, async () => {
    const pending: Array<{resolve: (receipt: ChildControlReceipt) => void; reject: (error: Error) => void}> = [];
    const send = () => new Promise<ChildControlReceipt>((resolve, reject) => pending.push({resolve, reject}));
    const state = await changingObservation(send, send, action === 'continue' ? 'cancelled' : 'running');
    const {panel} = state;
    const form = () => panel.querySelector<HTMLFormElement>(`[data-editor="${action}"]`)!;
    const input = () => form().querySelector<HTMLTextAreaElement>('textarea');
    const submit = (text: string) => {
      if (input()) input()!.value = text;
      form().requestSubmit();
    };
    submit('A command');
    await until(() => pending.length === 1);
    await state.set(action === 'reply' ? 'op-a' : 'op-b', 'req-b');
    expect(form().querySelector<HTMLButtonElement>('button')!.disabled).to.equal(false);
    if (input()) expect(input()!.value).to.equal('');
    submit('B command');
    await until(() => pending.length === 2);
    pending[0]!.resolve(commandReceipt(action));
    await new Promise((resolve) => setTimeout(resolve, 20));
    await panel.updateComplete;
    expect(form().querySelector<HTMLButtonElement>('button')!.disabled).to.equal(true);
    if (input()) expect(input()!.value).to.equal('B command');
    expect(panel.textContent).to.not.contain(action === 'reply' ? 'Reply sent.' : action === 'continue' ? 'Continuation accepted' : 'has not necessarily followed');
    pending[1]!.resolve(commandReceipt(action, 'op-b', 'req-b'));
    await until(() => !form().querySelector<HTMLButtonElement>('button')!.disabled);
    if (input()) expect(input()!.value).to.equal('');
    expect(panel.textContent).to.contain(action === 'reply' ? 'Reply sent.' : action === 'continue' ? 'Continuation accepted' : 'has not necessarily followed');
  });
}

it('resets the current identity editor after switching away and back during a command', async () => {
  let resolve!: (receipt: ChildControlReceipt) => void;
  const state = await changingObservation(() => new Promise((done) => { resolve = done; }));
  const a = state.panel.querySelector<HTMLTextAreaElement>('[name="instruction"]')!;
  a.value = 'A command';
  a.closest('form')!.requestSubmit();
  await state.set('op-b');
  state.panel.querySelector<HTMLTextAreaElement>('[name="instruction"]')!.value = 'B draft';
  await state.set('op-a');
  const restored = state.panel.querySelector<HTMLTextAreaElement>('[name="instruction"]')!;
  expect(restored).to.not.equal(a);
  expect(restored.value).to.equal('A command');
  expect(restored.disabled).to.equal(true);
  resolve(commandReceipt('steer'));
  await until(() => !restored.disabled);
  expect(restored.value).to.equal('');
  await state.set('op-b');
  expect(state.panel.querySelector<HTMLTextAreaElement>('[name="instruction"]')!.value).to.equal('B draft');
});

for (const action of ['continue', 'reply'] as const) {
  it(`preserves normal ${action} success feedback after refreshing the accepted state`, async () => {
    let current = observationFor('a', action === 'continue' ? 'succeeded' : 'running');
    const panel = roster();
    const send = async () => {
      if (action === 'continue') {
        current = observationFor('a');
        current.child.operationId = 'op-b';
      } else {
        current = {...current, questions: current.questions.map((question) => ({
          ...question, status: 'replied', reply: 'answer',
        }))};
      }
      return commandReceipt(action, action === 'continue' ? 'op-b' : 'op-a', action === 'reply' ? 'req-a' : null);
    };
    panel.open(async () => [current.child], undefined, {
      runId: 'run-1', observe: async () => current, control: send, reply: send,
    });
    await until(() => Boolean(panel.querySelector('[data-child-session="a"]')));
    panel.querySelector<HTMLButtonElement>('[data-child-session="a"]')!.click();
    await until(() => Boolean(panel.querySelector(`[data-editor="${action}"] textarea`)));
    const input = panel.querySelector<HTMLTextAreaElement>(`[data-editor="${action}"] textarea`)!;
    input.value = 'answer';
    input.closest('form')!.requestSubmit();
    await until(() => !panel.contains(input));
    const expected = action === 'continue' ? 'Continuation accepted as a new operation.' : 'Reply sent.';
    expect(panel.querySelector('.roster-observation')?.textContent).to.contain(expected);
    expect(panel.querySelector('[data-roster-status]')?.textContent).to.contain(expected);
    if (action === 'continue') {
      expect(panel.querySelector<HTMLTextAreaElement>('[name="instruction"]')!.value).to.equal('');
    }
  });
}

for (const status of ['replied', 'expired', 'cancelled']) {
  it(`does not attach a late reply to a historical ${status} request while a new request is open`, async () => {
    let resolve!: (receipt: ChildControlReceipt) => void;
    const state = await changingObservation(async () => commandReceipt('steer'), () => new Promise((done) => { resolve = done; }));
    const a = state.panel.querySelector<HTMLTextAreaElement>('[name="reply"]')!;
    a.value = 'A answer';
    a.closest('form')!.requestSubmit();
    await state.set('op-a', 'req-b', 'running', status);
    const b = state.panel.querySelector<HTMLTextAreaElement>('[name="reply"]')!;
    b.value = 'B answer';
    resolve(commandReceipt('reply', 'op-a', 'req-a'));
    await new Promise((done) => setTimeout(done, 20));
    expect(b.value).to.equal('B answer');
    expect(b.disabled).to.equal(false);
    expect(state.panel.textContent).to.not.contain('Reply sent.');
  });
}

it('late request rejection cannot replace the current request outcome or draft', async () => {
  let reject!: (error: Error) => void;
  const state = await changingObservation(async () => commandReceipt('steer'), () => new Promise((_resolve, fail) => { reject = fail; }));
  const a = state.panel.querySelector<HTMLTextAreaElement>('[name="reply"]')!;
  a.value = 'A';
  a.closest('form')!.requestSubmit();
  await state.set('op-a', 'req-b');
  const b = state.panel.querySelector<HTMLTextAreaElement>('[name="reply"]')!;
  b.value = 'B';
  reject(Object.assign(new Error('rejected'), {status: 409, outcome: 'run_terminal'}));
  await new Promise((resolve) => setTimeout(resolve, 20));
  expect(b.value).to.equal('B');
  expect(b.disabled).to.equal(false);
  expect(state.panel.textContent).to.not.contain('parent run is terminal');
});
