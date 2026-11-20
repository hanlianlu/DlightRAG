// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {afterEach, beforeEach, mock, test} from 'node:test';
import type {RunActivity, RunStatus} from '../api/run-activity.ts';
import {
  RunActivityStore,
  type ActivityVisibility,
  type SeenStorage,
} from './run-activity-store.ts';

function run(
  runId: string,
  status: RunStatus,
  {conversationId = `c-${runId}`, title = `Title ${runId}`, goalRound = false}: {
    conversationId?: string;
    title?: string | null;
    goalRound?: boolean;
  } = {},
): RunActivity {
  return {
    runId,
    conversationId,
    conversationTitle: title,
    status,
    phase: status === 'running' ? 'researching' : null,
    createdAt: '2026-10-24T10:00:00Z',
    startedAt: status === 'queued' ? null : '2026-10-24T10:00:01Z',
    finishedAt: status === 'succeeded' || status === 'failed' || status === 'cancelled'
      ? '2026-10-24T10:05:00Z'
      : null,
    goalRound,
  };
}

/** An always-running snapshot: the page watched one Run for as long as the answers last. */
function runningAnswers(count: number): RunActivity[][] {
  return Array.from({length: count}, () => [run('r1', 'running')]);
}

/** The page the test shows and hides. */
function page(shown = true) {
  const handlers = new Set<() => void>();
  let visible = shown;
  const view = {
    listening: 0,
    visible: () => visible,
    onChange: (handler: () => void) => {
      handlers.add(handler);
      view.listening += 1;
      return () => { handlers.delete(handler); view.listening -= 1; };
    },
    show() { visible = true; for (const handler of [...handlers]) handler(); },
    hide() { visible = false; for (const handler of [...handlers]) handler(); },
  } satisfies ActivityVisibility & Record<string, unknown>;
  return view;
}

/** What this browser would keep, in memory. */
function storage(initial: string | null = null) {
  const held = {value: initial, writes: 0};
  const seen: SeenStorage = {
    read: () => held.value,
    write: (next) => { held.value = next; held.writes += 1; },
  };
  return {held, seen};
}

/** The server's answers, one snapshot per read, and what was asked. */
function server(answers: Array<RunActivity[] | Error> = []) {
  const asked: Array<{limit: number; signal: AbortSignal}> = [];
  const list = async (limit: number, signal: AbortSignal) => {
    asked.push({limit, signal});
    const next = answers.shift() ?? [];
    if (next instanceof Error) throw next;
    return {items: next};
  };
  return {asked, list};
}

/** Let the reads and their continuations that are already due settle. */
async function settle(turns = 10): Promise<void> {
  for (let turn = 0; turn < turns; turn += 1) await Promise.resolve();
}

/** A started store on a page whose visibility the test drives. */
function watch(remote: ReturnType<typeof server>, view = page(), kept = storage()) {
  const store = new RunActivityStore({list: remote.list, visibility: view, storage: kept.seen});
  store.start();
  return {store, view, kept};
}

beforeEach(() => { mock.timers.enable({apis: ['setTimeout']}); });
afterEach(() => { mock.timers.reset(); });

test('the first read states what is running and unread, and announces nothing', async () => {
  const remote = server([[
    run('r-running', 'running', {conversationId: 'c-running'}),
    run('r-done', 'succeeded', {conversationId: 'c-done'}),
  ]]);
  const kept = storage();
  const store = new RunActivityStore({
    list: remote.list,
    visibility: page(),
    storage: kept.seen,
  });
  const settled: RunActivity[] = [];
  store.subscribeSettled((each) => { settled.push(each); });
  let redraws = 0;
  store.subscribe(() => { redraws += 1; });

  store.start();
  await settle();

  assert.deepEqual(remote.asked.map(({limit}) => limit), [30], 'the default page is asked for');
  assert.equal(settled.length, 0, 'the read the page opened with is not news');
  assert.equal(store.loaded, true);
  assert.equal(store.activeCount, 1);
  assert.deepEqual([...store.activeConversationIds], ['c-running']);
  assert.equal(store.unreadCount, 1);
  assert.deepEqual([...store.unreadConversationIds], ['c-done']);
  assert.equal(store.runs.length, 2, 'the snapshot is kept for consumers');
  assert.ok(redraws >= 1);
  assert.equal(store.watching, true, 'a Run under way is watched');
  store.stop();
});

test('the first read on an idle page asks once and then sleeps, keeping the snapshot', async () => {
  const remote = server([[run('r-done', 'succeeded')]]);
  const {store} = watch(remote);
  await settle();

  assert.equal(remote.asked.length, 1, 'the discovery read is the only read');
  assert.equal(store.watching, false, 'an idle page holds no timer');
  assert.equal(store.loaded, true);
  assert.deepEqual([...store.unreadConversationIds], ['c-r-done'], 'the unread mark stays drawn');

  mock.timers.tick(60000);
  await settle();
  assert.equal(remote.asked.length, 1, 'and nothing is asked for while it sleeps');
  assert.equal(store.runs.length, 1, 'the snapshot stays for what is drawn from it');
  store.stop();
});

test('a run watched from running to settled is announced once, then never again', async () => {
  const remote = server([
    [run('r1', 'running')],
    [run('r1', 'succeeded')],
    [run('r1', 'succeeded')],
  ]);
  const store = new RunActivityStore({list: remote.list, visibility: page(), storage: storage().seen});
  const settled: RunActivity[] = [];
  store.subscribeSettled((each) => { settled.push(each); });

  store.start();
  await settle();
  assert.equal(settled.length, 0, 'still running on the first read');

  mock.timers.tick(5000);
  await settle();
  assert.deepEqual(settled.map((each) => each.runId), ['r1']);
  assert.deepEqual([...store.unreadConversationIds], ['c-r1']);

  mock.timers.tick(5000);
  await settle();
  assert.deepEqual(settled.map((each) => each.runId), ['r1'], 'a settled run is announced once');
  assert.equal(store.watching, false, 'the tail read left the page asleep');

  mock.timers.tick(60000);
  await settle();
  assert.deepEqual(settled.map((each) => each.runId), ['r1'], 'and a sleeping page announces nothing');
  store.stop();
});

test('a goal round still reaches settlement subscribers, and keeps its unread mark', async () => {
  const remote = server([
    [run('r-goal', 'running', {conversationId: 'c-goal', goalRound: true})],
    [run('r-goal', 'succeeded', {conversationId: 'c-goal', goalRound: true})],
  ]);
  const store = new RunActivityStore({list: remote.list, visibility: page(), storage: storage().seen});
  const settled: RunActivity[] = [];
  store.subscribeSettled((each) => { settled.push(each); });
  store.start();
  await settle();

  mock.timers.tick(5000);
  await settle();

  assert.deepEqual(settled.map((each) => each.runId), ['r-goal']);
  assert.equal(settled[0]!.goalRound, true, 'the notifier decides to stay silent, not the store');
  assert.deepEqual([...store.unreadConversationIds], ['c-goal'], 'the marker stays what it was');
  store.stop();
});

test('a run that appears settled after the first read is announced once', async () => {
  const remote = server([
    [run('r1', 'running')],
    [run('r1', 'running'), run('r2', 'failed')],
    [run('r1', 'running'), run('r2', 'failed')],
  ]);
  const store = new RunActivityStore({list: remote.list, visibility: page(), storage: storage().seen});
  const settled: RunActivity[] = [];
  store.subscribeSettled((each) => { settled.push(each); });
  store.start();
  await settle();

  mock.timers.tick(5000);
  await settle();
  assert.deepEqual(settled.map((each) => each.runId), ['r2']);

  mock.timers.tick(5000);
  await settle();
  assert.deepEqual(settled.map((each) => each.runId), ['r2'], 'a run already read as settled is not news');
  store.stop();
});

test('a run already seen never announces itself', async () => {
  const kept = storage(JSON.stringify(['r1']));
  const remote = server([
    [run('r1', 'running')],
    [run('r1', 'succeeded')],
  ]);
  const store = new RunActivityStore({list: remote.list, visibility: page(), storage: kept.seen});
  const settled: RunActivity[] = [];
  store.subscribeSettled((each) => { settled.push(each); });
  store.start();
  await settle();

  mock.timers.tick(5000);
  await settle();
  assert.equal(settled.length, 0);
  assert.equal(store.unreadCount, 0, 'what the reader has seen is not unread');
  store.stop();
});

test('what was seen survives a new store through the same storage, and an unseen run does not', async () => {
  const kept = storage();
  const first = new RunActivityStore({
    list: server([[run('r1', 'succeeded')]]).list,
    visibility: page(),
    storage: kept.seen,
  });
  first.start();
  await settle();
  assert.deepEqual([...first.unreadConversationIds], ['c-r1']);
  first.markAllSeen();
  assert.deepEqual(JSON.parse(kept.held.value ?? 'null'), ['r1']);
  first.stop();

  const second = new RunActivityStore({
    list: server([
      [run('r1', 'succeeded')],
      [run('r1', 'succeeded'), run('r2', 'succeeded')],
    ]).list,
    visibility: page(),
    storage: kept.seen,
  });
  const settled: RunActivity[] = [];
  second.subscribeSettled((each) => { settled.push(each); });
  second.start();
  await settle();
  assert.deepEqual([...second.unreadConversationIds], [], 'the run seen before this store is still seen');
  assert.equal(settled.length, 0, 'and its first read announced nothing either');

  // An idle page sleeps; the next look is a wake, as a new Run starting is.
  second.wake();
  await settle();
  assert.deepEqual([...second.unreadConversationIds], ['c-r2']);
  assert.deepEqual(settled.map((each) => each.conversationId), ['c-r2']);
  second.stop();
});

test('a malformed stored value behaves as nothing seen', async () => {
  for (const raw of ['not json', '{"r1":true}', 'null']) {
    const remote = server([[run('r1', 'succeeded')]]);
    const store = new RunActivityStore({list: remote.list, visibility: page(), storage: storage(raw).seen});
    store.start();
    await settle();
    assert.deepEqual(
      [...store.unreadConversationIds],
      ['c-r1'],
      `${raw} must not hide an answer`,
    );
    store.stop();
  }
  // A stored list without an id in it is nothing seen too, as is an id that is not a string.
  for (const raw of ['[1,2]', '[""]', '["r1", 2]']) {
    const remote = server([[run('r1', 'succeeded')]]);
    const store = new RunActivityStore({list: remote.list, visibility: page(), storage: storage(raw).seen});
    store.start();
    await settle();
    assert.deepEqual([...store.unreadConversationIds], ['c-r1'], `${raw} must not hide an answer`);
    store.stop();
  }
});

test('the polling interval is 5 seconds visible and 15 hidden, and becoming visible reads at once', async () => {
  const view = page(true);
  const remote = server(runningAnswers(8));
  const {store} = watch(remote, view);
  await settle();
  assert.equal(remote.asked.length, 1, 'starting reads at once');
  assert.equal(view.listening, 1);

  mock.timers.tick(4999);
  await settle();
  assert.equal(remote.asked.length, 1, 'nothing before the visible interval');
  mock.timers.tick(1);
  await settle();
  assert.equal(remote.asked.length, 2);

  view.hide();
  mock.timers.tick(14999);
  await settle();
  assert.equal(remote.asked.length, 2, 'nothing before the hidden interval');
  mock.timers.tick(1);
  await settle();
  assert.equal(remote.asked.length, 3);

  view.show();
  await settle();
  assert.equal(remote.asked.length, 4, 'becoming visible reads at once, without waiting');
  mock.timers.tick(5000);
  await settle();
  assert.equal(remote.asked.length, 5, 'the nearer interval is back');

  store.stop();
  assert.equal(view.listening, 0, 'stopping lets the page go');
  mock.timers.tick(60000);
  await settle();
  assert.equal(remote.asked.length, 5, 'and nothing is read after it');
});

test('a refresh while one is in flight shares it rather than reading twice', async () => {
  const asked: number[] = [];
  let release!: () => void;
  const store = new RunActivityStore({
    list: (limit) => {
      asked.push(limit);
      return new Promise((resolve) => {
        release = () => { resolve({items: []}); };
      });
    },
    visibility: page(),
    storage: storage().seen,
  });

  const first = store.refresh();
  const second = store.refresh();
  assert.equal(asked.length, 1);
  release();
  await Promise.all([first, second]);
  assert.equal(asked.length, 1, 'one read in flight, shared with the second caller');
});

test('a failed read keeps the snapshot and the next tick tries again', async () => {
  const view = page();
  const remote = server([
    [run('r1', 'running')],
    new Error('offline'),
    [run('r1', 'succeeded')],
  ]);
  const store = new RunActivityStore({list: remote.list, visibility: view, storage: storage().seen});
  const settled: RunActivity[] = [];
  store.subscribeSettled((each) => { settled.push(each); });
  store.start();
  await settle();

  mock.timers.tick(5000);
  await settle();
  assert.equal(store.runs.length, 1, 'the failed read did not empty what the page knows');
  assert.equal(store.watching, true, 'a page with work keeps watching after a failure');

  mock.timers.tick(5000);
  await settle();
  assert.deepEqual(settled.map((each) => each.runId), ['r1'], 'reading again after a failure still announces');
  store.stop();
});

test('a failed read on an idle page sleeps until the page has a reason to look again', async () => {
  const remote = server([
    new Error('offline'),
    [run('r1', 'running')],
    [run('r1', 'running')],
  ]);
  const {store} = watch(remote);
  await settle();
  assert.equal(remote.asked.length, 1);
  assert.equal(store.watching, false, 'nothing is retried by a page with no work');

  mock.timers.tick(60000);
  await settle();
  assert.equal(remote.asked.length, 1);

  store.wake();
  await settle();
  assert.equal(remote.asked.length, 2, 'a wake is the reason to look again');
  assert.equal(store.watching, true, 'work found means watching resumes');
  mock.timers.tick(5000);
  await settle();
  assert.equal(remote.asked.length, 3);
  store.stop();
});

test('the last settle earns one tail read five seconds later, and the page sleeps after it', async () => {
  const remote = server([
    [run('r1', 'running')],
    [run('r1', 'succeeded')],
    [run('r1', 'succeeded')],
  ]);
  const {store} = watch(remote);
  await settle();
  assert.equal(remote.asked.length, 1);

  mock.timers.tick(5000);
  await settle();
  assert.equal(remote.asked.length, 2, 'the poll saw the last Running Run settle');
  assert.equal(store.watching, true, 'the tail look is still to come');
  assert.deepEqual([...store.unreadConversationIds], ['c-r1']);

  mock.timers.tick(4999);
  await settle();
  assert.equal(remote.asked.length, 2, 'the tail look waits the five seconds');
  mock.timers.tick(1);
  await settle();
  assert.equal(remote.asked.length, 3, 'the tail look reads once');
  assert.equal(store.watching, false, 'and then the page sleeps');
  assert.deepEqual([...store.unreadConversationIds], ['c-r1'], 'the unread mark stays drawn');

  mock.timers.tick(60000);
  await settle();
  assert.equal(remote.asked.length, 3, 'a sleeping page asks for nothing');
  store.stop();
});

test('the tail read catches a goal round continuation, and polling resumes', async () => {
  const remote = server([
    [run('r1', 'running', {conversationId: 'c-goal', goalRound: true})],
    [run('r1', 'succeeded', {conversationId: 'c-goal', goalRound: true})],
    [
      run('r2', 'running', {conversationId: 'c-goal', goalRound: true}),
      run('r1', 'succeeded', {conversationId: 'c-goal', goalRound: true}),
    ],
    [
      run('r2', 'running', {conversationId: 'c-goal', goalRound: true}),
      run('r1', 'succeeded', {conversationId: 'c-goal', goalRound: true}),
    ],
  ]);
  const {store} = watch(remote);
  const settled: RunActivity[] = [];
  store.subscribeSettled((each) => { settled.push(each); });
  await settle();

  mock.timers.tick(5000);
  await settle();
  assert.deepEqual(settled.map((each) => each.runId), ['r1'], 'the round settled');

  mock.timers.tick(5000);
  await settle();
  assert.equal(remote.asked.length, 3, 'the tail read caught the continuation');
  assert.deepEqual([...store.activeConversationIds], ['c-goal'], 'the conversation runs across rounds');
  assert.equal(store.watching, true, 'and polling is back');

  mock.timers.tick(5000);
  await settle();
  assert.equal(remote.asked.length, 4, 'the continuation is polled like any other Run');
  store.stop();
});

test('a wake reads a dormant page once and resumes polling when it finds work', async () => {
  const remote = server([
    [],
    [run('r1', 'running')],
    [run('r1', 'running')],
  ]);
  const {store} = watch(remote);
  await settle();
  assert.equal(store.watching, false);

  store.wake();
  await settle();
  assert.equal(remote.asked.length, 2, 'the wake read at once');
  assert.deepEqual([...store.activeConversationIds], ['c-r1']);
  assert.equal(store.watching, true, 'work found means polling is back');

  mock.timers.tick(5000);
  await settle();
  assert.equal(remote.asked.length, 3);
  store.stop();
});

test('an attention read that finds nothing makes the one tail look, then sleeps', async () => {
  // The submission Run may be accepted a moment after the page hears it start: the tail look is
  // what catches it, and the page still sleeps after that one look.
  const remote = server([
    [],
    [],
    [],
  ]);
  const {store} = watch(remote);
  await settle();
  assert.equal(remote.asked.length, 1);

  store.wake();
  await settle();
  assert.equal(remote.asked.length, 2, 'the wake read found nothing yet');
  assert.equal(store.watching, true, 'the one tail look is to come');

  mock.timers.tick(4999);
  await settle();
  assert.equal(remote.asked.length, 2);
  mock.timers.tick(1);
  await settle();
  assert.equal(remote.asked.length, 3, 'the tail look reads once');
  assert.equal(store.watching, false, 'and an idle page sleeps');

  mock.timers.tick(60000);
  await settle();
  assert.equal(remote.asked.length, 3);
  store.stop();
});

test('a wake that interrupts the tail look is that look, not a second one', async () => {
  const remote = server([
    [run('r1', 'running')],
    [run('r1', 'succeeded')],
    [run('r1', 'succeeded')],
  ]);
  const {store} = watch(remote);
  await settle();

  mock.timers.tick(5000);
  await settle();
  assert.equal(remote.asked.length, 2, 'the poll saw the settle, and the tail look is pending');

  store.wake();
  await settle();
  assert.equal(remote.asked.length, 3, 'the wake read the tail look early');
  assert.equal(store.watching, false, 'and no second tail look follows');

  mock.timers.tick(60000);
  await settle();
  assert.equal(remote.asked.length, 3);
  store.stop();
});

test('marking one conversation seen persists its settled runs and takes away its marker', async () => {
  const kept = storage();
  const remote = server([[
    run('r1', 'succeeded', {conversationId: 'c1'}),
    run('r2', 'failed', {conversationId: 'c2'}),
    run('r3', 'running', {conversationId: 'c1'}),
  ]]);
  const store = new RunActivityStore({list: remote.list, visibility: page(), storage: kept.seen});
  store.start();
  await settle();
  let redraws = 0;
  store.subscribe(() => { redraws += 1; });

  store.markConversationSeen('c1');

  assert.deepEqual([...store.unreadConversationIds], ['c2'], 'only the other conversation is unread');
  assert.deepEqual(JSON.parse(kept.held.value ?? 'null'), ['r1'], 'the running run is not seen');
  assert.equal(redraws, 1);

  store.markConversationSeen('c1');
  assert.equal(redraws, 1, 'marking what is already seen asks for no redraw');

  store.markAllSeen();
  assert.deepEqual([...store.unreadConversationIds], []);
  assert.deepEqual(JSON.parse(kept.held.value ?? 'null'), ['r1', 'r2']);
  store.stop();
});

test('the stored seen ids keep the newest 200 terminal runs and drop the oldest', async () => {
  // The server orders newest first, so this list runs from r249 down to r0.
  const runs = Array.from({length: 250}, (_, index) => run(`r${249 - index}`, 'succeeded'));
  const kept = storage();
  const store = new RunActivityStore({
    list: server([runs]).list,
    visibility: page(),
    storage: kept.seen,
  });
  store.start();
  await settle();

  store.markAllSeen();
  const persisted = JSON.parse(kept.held.value ?? 'null') as string[];
  assert.equal(persisted.length, 200);
  assert.equal(persisted.at(-1), 'r249', 'the newest terminal run is kept last');
  assert.equal(persisted[0], 'r50', 'the oldest kept is the 200th newest');
  assert.equal(persisted.includes('r49'), false, 'anything older than the bound is gone');
  store.stop();
});

test('forgetting ends the polling, drops the snapshot, and clears what the reader had seen', async () => {
  const kept = storage();
  const view = page();
  const remote = server([
    [run('r1', 'succeeded')],
    [run('r1', 'succeeded')],
  ]);
  const store = new RunActivityStore({list: remote.list, visibility: view, storage: kept.seen});
  store.start();
  await settle();
  store.markAllSeen();
  assert.equal(kept.held.value, JSON.stringify(['r1']));

  store.forget();

  assert.equal(store.runs.length, 0);
  assert.equal(store.unreadCount, 0);
  assert.equal(store.loaded, false);
  assert.equal(store.watching, false);
  assert.equal(kept.held.value, null, 'the next reader starts new');
  assert.equal(view.listening, 0);

  const settled: RunActivity[] = [];
  store.subscribeSettled((each) => { settled.push(each); });
  store.start();
  await settle();
  assert.equal(remote.asked.length, 2, 'starting after forgetting reads again');
  assert.equal(settled.length, 0, 'and its first read still announces nothing');
  assert.deepEqual([...store.unreadConversationIds], ['c-r1']);
  store.stop();
});

test('stopping ends a read in flight, drops a pending tail look, and drops its answer', async () => {
  let release!: (items: RunActivity[]) => void;
  let signal: AbortSignal | undefined;
  const store = new RunActivityStore({
    list: (_limit, requestSignal) => {
      signal = requestSignal;
      return new Promise((resolve) => {
        release = (items) => { resolve({items}); };
      });
    },
    visibility: page(),
    storage: storage().seen,
  });
  store.start();
  assert.equal(signal?.aborted, false);

  store.stop();
  assert.equal(signal?.aborted, true, 'stopping aborts the read');
  release([run('r1', 'succeeded')]);
  await settle();
  assert.equal(store.runs.length, 0, 'the answer to an aborted read is let go');

  store.wake();
  await settle();
  assert.equal(store.watching, false, 'a stopped store reads for nobody');
});

test('a pending tail look does not survive stopping, and the next start reads a discovery page', async () => {
  const remote = server([
    [run('r1', 'running')],
    [run('r1', 'succeeded')],
    [run('r1', 'succeeded')],
  ]);
  const {store} = watch(remote);
  await settle();

  mock.timers.tick(5000);
  await settle();
  assert.equal(remote.asked.length, 2, 'the settle is known and the tail look is pending');

  store.stop();
  mock.timers.tick(60000);
  await settle();
  assert.equal(remote.asked.length, 2, 'a stopped page keeps no timer');
  assert.equal(store.runs.length, 1, 'and keeps the snapshot');

  store.start();
  await settle();
  assert.equal(remote.asked.length, 3, 'starting again reads at once');
  assert.deepEqual(
    [...store.unreadConversationIds],
    ['c-r1'],
    'the unread mark outlived stopping and starting',
  );
  store.stop();
});

test('the page is read as often as it is asked, and the limit can be set', async () => {
  const remote = server([]);
  const store = new RunActivityStore({
    list: remote.list,
    visibility: page(),
    storage: storage().seen,
    limit: 10,
  });
  store.start();
  await settle();
  assert.deepEqual(remote.asked.map(({limit}) => limit), [10]);
  store.stop();
});
