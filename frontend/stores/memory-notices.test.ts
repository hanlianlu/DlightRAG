// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {afterEach, beforeEach, mock, test} from 'node:test';
import type {RunMemoryChange} from '../api/memory.ts';
import {MemoryNotices, type PageVisibility} from './memory-notices.ts';

const change = (id: string, writer: RunMemoryChange['writer'] = 'conversation'): RunMemoryChange => ({
  memoryId: `memory-${id}`,
  kind: 'preference',
  body: `Body ${id}`,
  action: 'add',
  changeId: `change-${id}`,
  writer,
});

/** A page the test shows and hides. */
function page(shown = true) {
  const handlers = new Set<() => void>();
  let visible = shown;
  const view = {
    /** How many listeners are on the page's visibility now. */
    listening: 0,
    visible: () => visible,
    onChange: (handler: () => void) => {
      handlers.add(handler);
      view.listening += 1;
      return () => { handlers.delete(handler); view.listening -= 1; };
    },
    show() { visible = true; for (const handler of [...handlers]) handler(); },
    hide() { visible = false; for (const handler of [...handlers]) handler(); },
  } satisfies PageVisibility & Record<string, unknown>;
  return view;
}

/** A server whose answer for each Run the test sets, and that counts what was asked. */
function server() {
  const answers = new Map<string, readonly RunMemoryChange[]>();
  const asked: string[] = [];
  return {
    answers,
    asked,
    read: async (runId: string): Promise<readonly RunMemoryChange[]> => {
      asked.push(runId);
      return answers.get(runId) ?? [];
    },
  };
}

/** Let the reads and their continuations that are already due settle. */
const settle = async (): Promise<void> => { for (let turn = 0; turn < 5; turn += 1) await Promise.resolve(); };

beforeEach(() => { mock.timers.enable({apis: ['setTimeout']}); });
afterEach(() => { mock.timers.reset(); });

test('a turn answered before the page opened is read once, however often it is drawn', async () => {
  const remote = server();
  remote.answers.set('run-1', [change('1', 'agent')]);
  const notices = new MemoryNotices({read: remote.read, visibility: page()});

  notices.ensure('run-1');
  notices.ensure('run-1');
  await settle();
  notices.ensure('run-1');
  await settle();

  assert.deepEqual(remote.asked, ['run-1']);
  assert.deepEqual(notices.changes('run-1').map((each) => each.changeId), ['change-1']);
  assert.equal(notices.announces('run-1'), false, 'a page that did not watch the Run has no news');
});

test('a read that fails is tried again the next time the turn is drawn', async () => {
  let fail = true;
  const notices = new MemoryNotices({
    read: async () => { if (fail) throw new Error('offline'); return [change('1')]; },
    visibility: page(),
  });

  notices.ensure('run-1');
  await settle();
  assert.equal(notices.changes('run-1').length, 0);
  fail = false;
  notices.ensure('run-1');
  await settle();

  assert.equal(notices.changes('run-1').length, 1);
});

test('a Run that just settled is read at 3, 8 and 20 seconds, and no more once the pass has reported', async () => {
  const remote = server();
  const notices = new MemoryNotices({read: remote.read, visibility: page()});

  notices.watch('run-1');
  mock.timers.tick(2999);
  await settle();
  assert.deepEqual(remote.asked, [], 'nothing before the first try');

  mock.timers.tick(1);
  await settle();
  assert.deepEqual(remote.asked, ['run-1']);

  remote.answers.set('run-1', [change('1')]);
  mock.timers.tick(5000);
  await settle();
  assert.deepEqual(remote.asked, ['run-1', 'run-1'], 'the second try, 8 seconds after the Run settled');
  assert.deepEqual(notices.changes('run-1').map((each) => each.changeId), ['change-1']);
  assert.equal(notices.announces('run-1'), true, 'the reader was here for the answer, so this is news');

  mock.timers.tick(60000);
  await settle();
  assert.equal(remote.asked.length, 2, 'the pass reported; there is nothing left to wait for');
});

test('with nothing to find, the last try is the 20 seconds mark and then the watch ends', async () => {
  const remote = server();
  const notices = new MemoryNotices({read: remote.read, visibility: page()});

  notices.watch('run-1');
  for (const step of [3000, 5000, 12000, 60000]) {
    mock.timers.tick(step);
    await settle();
  }

  assert.equal(remote.asked.length, 3);
  assert.equal(notices.changes('run-1').length, 0);
});

test('what the Agent left is no sign that the pass has reported, so the watch goes on', async () => {
  const remote = server();
  remote.answers.set('run-1', [change('1', 'agent')]);
  const notices = new MemoryNotices({read: remote.read, visibility: page()});

  notices.watch('run-1');
  mock.timers.tick(3000);
  await settle();
  remote.answers.set('run-1', [change('1', 'agent'), change('2', 'conversation')]);
  mock.timers.tick(5000);
  await settle();

  assert.deepEqual(notices.changes('run-1').map((each) => each.changeId), ['change-1', 'change-2']);
  mock.timers.tick(60000);
  await settle();
  assert.equal(remote.asked.length, 2);
});

test('a live memory event for the turn is read as soon as it settles, before the first try', async () => {
  const remote = server();
  remote.answers.set('run-1', [change('1', 'agent')]);
  const notices = new MemoryNotices({read: remote.read, visibility: page()});

  notices.noteAgentWrote('run-1');
  assert.deepEqual(remote.asked, [], 'it is read after the Run, not during it');
  notices.watch('run-1');
  await settle();

  assert.deepEqual(remote.asked, ['run-1']);
  assert.equal(notices.changes('run-1').length, 1);
});

test('no try is made while the page is hidden, and the one that fell due is made when it is shown', async () => {
  const remote = server();
  const view = page(false);
  const notices = new MemoryNotices({read: remote.read, visibility: view});

  notices.watch('run-1');
  mock.timers.tick(3000);
  await settle();
  mock.timers.tick(100000);
  await settle();
  assert.deepEqual(remote.asked, [], 'nobody was looking');
  assert.equal(view.listening, 1);

  remote.answers.set('run-1', [change('1')]);
  view.show();
  await settle();

  assert.deepEqual(remote.asked, ['run-1']);
  assert.equal(notices.changes('run-1').length, 1);
  assert.equal(view.listening, 0, 'the page is not listened to once nothing waits on it');
});

test('an undone change stays where it was, no longer counts, and survives the next read', async () => {
  const remote = server();
  remote.answers.set('run-1', [change('1'), change('2')]);
  const notices = new MemoryNotices({read: remote.read, visibility: page()});
  let redraws = 0;
  notices.subscribe(() => { redraws += 1; });
  notices.ensure('run-1');
  await settle();

  notices.markUndone('run-1', 'change-1');
  assert.deepEqual(
    notices.changes('run-1').map((each) => [each.changeId, each.undone]),
    [['change-1', true], ['change-2', false]],
  );
  const drawn = redraws;
  notices.markUndone('run-1', 'change-1');
  assert.equal(redraws, drawn, 'taking back what is taken back changes nothing');

  // The server lists the records still active, so the one taken back is no longer among them.
  remote.answers.set('run-1', [change('2'), change('3')]);
  notices.watch('run-1');
  mock.timers.tick(3000);
  await settle();

  assert.deepEqual(
    notices.changes('run-1').map((each) => [each.changeId, each.undone]),
    [['change-1', true], ['change-2', false], ['change-3', false]],
  );
});

test('a change the server no longer lists, which the reader did not take back, goes', async () => {
  const remote = server();
  remote.answers.set('run-1', [change('1'), change('2')]);
  const notices = new MemoryNotices({read: remote.read, visibility: page()});
  notices.ensure('run-1');
  await settle();

  remote.answers.set('run-1', [change('2')]);
  notices.watch('run-1');
  mock.timers.tick(3000);
  await settle();

  assert.deepEqual(notices.changes('run-1').map((each) => each.changeId), ['change-2']);
});

test('announcing is for the page that watched the Run, and happens once', async () => {
  const remote = server();
  remote.answers.set('run-1', [change('1')]);
  const notices = new MemoryNotices({read: remote.read, visibility: page()});

  notices.watch('run-1');
  assert.equal(notices.announces('run-1'), true);
  notices.announced('run-1');

  assert.equal(notices.announces('run-1'), false);
});

test('forgetting drops what was read and every try still to come', async () => {
  const remote = server();
  remote.answers.set('run-1', [change('1')]);
  const notices = new MemoryNotices({read: remote.read, visibility: page()});
  notices.ensure('run-1');
  await settle();
  notices.watch('run-2');

  notices.forget();
  mock.timers.tick(60000);
  await settle();

  assert.equal(notices.changes('run-1').length, 0);
  assert.deepEqual(remote.asked, ['run-1'], 'no try was made for the Run that was being watched');
});

test('a read that returns after the store was forgotten is let go', async () => {
  let release: (() => void) | undefined;
  const notices = new MemoryNotices({
    // The server answers whenever it does; the signal is the store's to honour.
    read: () => new Promise((resolve) => { release = () => { resolve([change('1')]); }; }),
    visibility: page(),
  });
  notices.ensure('run-1');

  notices.forget();
  release?.();
  await settle();

  assert.equal(notices.changes('run-1').length, 0, 'the last reader\'s memory reached the next');
});

test('a change the owner made elsewhere has what was read read again, and keeps showing it until then', async () => {
  const remote = server();
  remote.answers.set('run-1', [change('1'), change('2')]);
  const notices = new MemoryNotices({read: remote.read, visibility: page()});
  let redraws = 0;
  notices.subscribe(() => { redraws += 1; });
  notices.ensure('run-1');
  await settle();
  notices.ensure('run-1');
  await settle();
  assert.equal(remote.asked.length, 1, 'read once while it is true');

  remote.answers.set('run-1', []);
  const before = redraws;
  notices.refresh();
  assert.equal(notices.changes('run-1').length, 2, 'nothing goes until the server has said so');
  assert.equal(redraws, before + 1, 'the notices under the answers are told to look');
  notices.ensure('run-1');
  await settle();

  assert.equal(remote.asked.length, 2);
  assert.equal(notices.changes('run-1').length, 0);
});
