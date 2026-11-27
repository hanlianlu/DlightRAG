// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import type {ConversationSummary} from '../api/conversations.ts';
import type {RunActivity, RunStatus} from '../api/run-activity.ts';
import {defineDesignSystemElements} from '../design-system/index.ts';
import {newChatRoute} from '../lib/router.ts';
import {type AppHandles, createAppHandles} from '../stores/app-handles.ts';
import {RunActivityStore, type SeenStorage} from '../stores/run-activity-store.ts';
import {waitFor} from '../testing/dom.ts';
import type {DlConversationList} from './conversation-list.ts';
import './conversation-list.ts';
import type {DlRunNotifications} from './run-notifications.ts';
import './run-notifications.ts';
import {webRouter} from './router.ts';

defineDesignSystemElements();

const notificationDescriptor = Object.getOwnPropertyDescriptor(window, 'Notification');
const originalHasFocus = document.hasFocus;
const originalFetch = window.fetch;

const sleep = async (milliseconds: number): Promise<void> =>
  await new Promise((resolve) => setTimeout(resolve, milliseconds));

/** Wait until the predicate holds, up to a bounded wall-clock time: the goal outcome is retried. */
async function eventually(predicate: () => boolean, attempts = 60): Promise<void> {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    if (predicate()) return;
    await sleep(50);
  }
  throw new Error('condition did not become true');
}

/** Serve one goal phase to the notifier's read of the conversation goal. */
function serveGoal(
  phase: 'active' | 'complete' | 'blocked',
  overrides: Record<string, unknown> = {},
): void {
  window.fetch = async () =>
    Response.json({
      conversation_id: 'c-open',
      goal_id: 'goal-1',
      revision: 2,
      objective: 'finish the sealed model',
      phase,
      max_rounds: 8,
      rounds_started: 3,
      blocked_reason: phase === 'blocked' ? 'the goal reached its round cap' : null,
      ...overrides,
    });
}

/** The Notifications the notifier made, in the order it made them. */
let sent: FakeNotification[] = [];

class FakeNotification {
  static permission: NotificationPermission = 'granted';
  static async requestPermission(): Promise<NotificationPermission> {
    return FakeNotification.permission;
  }

  readonly title: string;
  readonly options: NotificationOptions;
  onclick: (() => void) | null = null;
  closed = false;

  constructor(title: string, options: NotificationOptions = {}) {
    this.title = title;
    this.options = options;
    sent.push(this);
  }

  close(): void {
    this.closed = true;
  }
}

function installNotification(permission: NotificationPermission): void {
  FakeNotification.permission = permission;
  Object.defineProperty(window, 'Notification', {configurable: true, value: FakeNotification});
}

function run(
  runId: string,
  status: RunStatus,
  {conversationId = `c-${runId}`, title = `Title ${runId}`, goalRound = false}: {
    conversationId?: string;
    title?: string | null;
    goalRound?: boolean;
  } = {},
): RunActivity {
  const terminal = status === 'succeeded' || status === 'failed' || status === 'cancelled';
  return {
    runId,
    conversationId,
    conversationTitle: title,
    status,
    phase: status === 'running' ? 'researching' : null,
    createdAt: '2026-10-24T10:00:00Z',
    startedAt: status === 'queued' ? null : '2026-10-24T10:00:01Z',
    finishedAt: terminal ? '2026-10-24T10:05:00Z' : null,
    goalRound,
  };
}

function summary(conversationId: string, title: string): ConversationSummary {
  return {
    conversationId,
    title,
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-02T00:00:00Z',
    forkedFromConversationId: null,
    forkedFromTitle: null,
    schedule: null,
  };
}

/** A store the test feeds one snapshot at a time, with storage that keeps nothing between tests. */
function activity() {
  const snapshots: RunActivity[][] = [];
  const storage: SeenStorage = {read: () => null, write: () => {}};
  const store = new RunActivityStore({
    list: async () => ({items: snapshots.shift() ?? []}),
    visibility: {visible: () => true, onChange: () => () => {}},
    storage,
  });
  return {
    store,
    push(items: RunActivity[]): void { snapshots.push(items); },
  };
}

/** A bag whose Run activity is the store the test drives. */
function bagWith(store: RunActivityStore): AppHandles {
  return {...createAppHandles(), runActivity: store};
}

function mountNotifier(bag: AppHandles, enabled = true): DlRunNotifications {
  const notifier = document.createElement('dl-run-notifications') as DlRunNotifications;
  notifier.handles = bag;
  notifier.enabled = enabled;
  document.body.append(notifier);
  return notifier;
}

/** Feed the store a first read and then the settled snapshot the notifier should announce. */
async function settleAfter(
  fixture: ReturnType<typeof activity>,
  running: RunActivity,
  settled: RunActivity,
): Promise<void> {
  fixture.push([running]);
  await fixture.store.refresh();
  fixture.push([settled]);
  await fixture.store.refresh();
}

beforeEach(() => {
  sent = [];
  installNotification('granted');
  Object.defineProperty(document, 'hidden', {configurable: true, get: () => false});
  document.hasFocus = () => false;
});

afterEach(async () => {
  document.body.replaceChildren();
  window.fetch = originalFetch;
  document.hasFocus = originalHasFocus;
  Reflect.deleteProperty(document, 'hidden');
  if (notificationDescriptor) Object.defineProperty(window, 'Notification', notificationDescriptor);
  else Reflect.deleteProperty(window, 'Notification');
  await webRouter.navigate(newChatRoute(), {replace: true, notify: false, bypassGuard: true});
});

it('notifies once for a settled run in a conversation other than the open one', async () => {
  const fixture = activity();
  const bag = bagWith(fixture.store);
  const notifier = mountNotifier(bag);

  await settleAfter(
    fixture,
    run('r1', 'running', {conversationId: 'c-other', title: 'Volvo outlook'}),
    run('r1', 'succeeded', {conversationId: 'c-other', title: 'Volvo outlook'}),
  );

  expect(sent).to.have.length(1);
  expect(sent[0]!.title).to.equal('Answer ready');
  expect(sent[0]!.options.body).to.equal('Your answer for Volvo outlook is ready.');
  notifier.remove();
});

it('notifies for the conversation the reader has open too, as the one owner for every conversation', async () => {
  const fixture = activity();
  const bag = bagWith(fixture.store);
  bag.conversations.adoptCreatedConversation(summary('c-open', 'Open conversation'));
  const notifier = mountNotifier(bag);

  await settleAfter(
    fixture,
    run('r1', 'running', {conversationId: 'c-open', title: 'Open conversation'}),
    run('r1', 'succeeded', {conversationId: 'c-open', title: 'Open conversation'}),
  );

  expect(sent).to.have.length(1);
  expect(sent[0]!.options.body).to.equal('Your answer for Open conversation is ready.');
  notifier.remove();
});

it('never announces a goal round while the goal is still running', async () => {
  serveGoal('active');
  const openFixture = activity();
  const openBag = bagWith(openFixture.store);
  openBag.conversations.adoptCreatedConversation(summary('c-open', 'An active goal'));
  const openNotifier = mountNotifier(openBag);
  await settleAfter(
    openFixture,
    run('r1', 'running', {conversationId: 'c-open', goalRound: true}),
    run('r1', 'succeeded', {conversationId: 'c-open', goalRound: true}),
  );
  // The outcome read is retried for the driver's write; stay past every attempt so a late
  // announcement would fail this test rather than leak into the next one.
  await sleep(1300);
  expect(sent).to.have.length(0, 'the open conversation\'s round gets no notice either');
  expect([...openFixture.store.unreadConversationIds], 'its unread marker stays for the reader').
    to.deep.equal(['c-open']);
  openNotifier.remove();

  const otherFixture = activity();
  const otherNotifier = mountNotifier(bagWith(otherFixture.store));
  await settleAfter(
    otherFixture,
    run('r2', 'running', {conversationId: 'c-other', goalRound: true}),
    run('r2', 'succeeded', {conversationId: 'c-other', goalRound: true}),
  );
  await sleep(1300);
  expect(sent).to.have.length(0, 'a per-round notice would claim the goal is over');
  expect([...otherFixture.store.unreadConversationIds]).to.deep.equal(['c-other']);
  otherNotifier.remove();
});

it('waits for the driver\u2019s terminal goal phase before announcing a finished round', async () => {
  let reads = 0;
  window.fetch = async () => {
    reads += 1;
    // The page's settlement read races the driver's own write: the first two reads see the goal
    // still active, the third sees what the driver recorded.
    const phase = reads < 3 ? 'active' : 'complete';
    return Response.json({
      conversation_id: 'c-race',
      goal_id: 'goal-1',
      revision: 2,
      objective: 'finish the sealed model',
      phase,
      max_rounds: 16,
      rounds_started: 4,
      blocked_reason: null,
    });
  };
  const fixture = activity();
  const bag = bagWith(fixture.store);
  bag.conversations.adoptCreatedConversation(summary('c-race', 'An active goal'));
  const notifier = mountNotifier(bag);

  await settleAfter(
    fixture,
    run('r1', 'running', {conversationId: 'c-race', title: 'An active goal', goalRound: true}),
    run('r1', 'succeeded', {conversationId: 'c-race', title: 'An active goal', goalRound: true}),
  );

  await eventually(() => sent.length === 1);
  expect(sent[0]!.title).to.equal('Goal complete');
  expect(reads).to.be.greaterThan(2, 'the notifier retried past the driver write');
  notifier.remove();
});

it('announces the goal a human turn ended, instead of the answer notice', async () => {
  serveGoal('complete');
  const fixture = activity();
  const notifier = mountNotifier(bagWith(fixture.store));

  await settleAfter(
    fixture,
    run('r1', 'running', {conversationId: 'c-human', title: 'Volvo outlook'}),
    run('r1', 'succeeded', {conversationId: 'c-human', title: 'Volvo outlook'}),
  );

  // The answer notice is synchronous; the goal's own notice follows the projected phase read.
  await waitFor(() => sent.length === 2);
  expect(sent.map((notification) => notification.title))
    .to.deep.equal(['Answer ready', 'Goal complete']);
  expect(sent[1]!.options.body).to.equal('The goal "Volvo outlook" is complete.');
  notifier.remove();
});

it('leaves an active goal to the answer notice when a human turn settles', async () => {
  serveGoal('active');
  const fixture = activity();
  const notifier = mountNotifier(bagWith(fixture.store));

  await settleAfter(
    fixture,
    run('r1', 'running', {conversationId: 'c-human', title: 'Volvo outlook'}),
    run('r1', 'succeeded', {conversationId: 'c-human', title: 'Volvo outlook'}),
  );

  await waitFor(() => sent.length === 1);
  await sleep(300);
  expect(sent.map((notification) => notification.title)).to.deep.equal(['Answer ready']);
  notifier.remove();
});

it('reads the goal once for a human turn, which writes its own goal outcome', async () => {
  let reads = 0;
  window.fetch = async () => {
    reads += 1;
    return Response.json({
      conversation_id: 'c-human',
      goal_id: 'goal-1',
      revision: 2,
      objective: 'finish the sealed model',
      phase: 'active',
      max_rounds: 16,
      rounds_started: 4,
      blocked_reason: null,
    });
  };
  const fixture = activity();
  const notifier = mountNotifier(bagWith(fixture.store));

  await settleAfter(
    fixture,
    run('r1', 'running', {conversationId: 'c-human'}),
    run('r1', 'succeeded', {conversationId: 'c-human'}),
  );

  await waitFor(() => sent.length === 1);
  await sleep(500);
  expect(reads, 'a human turn writes its goal outcome in its own Run: no retry').to.equal(1);
  notifier.remove();
});

it('says a blocked goal at its cap must raise the cap in the notification body', async () => {
  serveGoal('blocked', {max_rounds: 4, rounds_started: 4});
  const fixture = activity();
  const notifier = mountNotifier(bagWith(fixture.store));

  await settleAfter(
    fixture,
    run('r1', 'running', {conversationId: 'c-cap', title: 'Capped goal', goalRound: true}),
    run('r1', 'succeeded', {conversationId: 'c-cap', title: 'Capped goal', goalRound: true}),
  );

  await waitFor(() => sent.length === 1);
  expect(sent[0]!.title).to.equal('Goal blocked');
  expect(sent[0]!.options.body).to.equal(
    'The goal "Capped goal" reached its round cap; raise the cap to continue.',
  );
  notifier.remove();
});

it('announces the goal itself when a goal round settles into a complete goal', async () => {
  serveGoal('complete');
  const fixture = activity();
  const bag = bagWith(fixture.store);
  bag.conversations.adoptCreatedConversation(summary('c-open', 'An active goal'));
  const notifier = mountNotifier(bag);

  await settleAfter(
    fixture,
    run('r1', 'running', {conversationId: 'c-open', title: 'An active goal', goalRound: true}),
    run('r1', 'succeeded', {conversationId: 'c-open', title: 'An active goal', goalRound: true}),
  );

  await waitFor(() => sent.length === 1);
  expect(sent[0]!.title).to.equal('Goal complete');
  expect(sent[0]!.options.body).to.equal('The goal "An active goal" is complete.');
  notifier.remove();
});

it('announces a blocked goal once, so the reader knows it waits for them', async () => {
  serveGoal('blocked');
  const fixture = activity();
  const notifier = mountNotifier(bagWith(fixture.store));

  await settleAfter(
    fixture,
    run('r1', 'running', {conversationId: 'c-other', title: 'Volvo outlook', goalRound: true}),
    run('r1', 'succeeded', {conversationId: 'c-other', title: 'Volvo outlook', goalRound: true}),
  );

  await waitFor(() => sent.length === 1);
  expect(sent[0]!.title).to.equal('Goal blocked');
  expect(sent[0]!.options.body).to.equal(
    'The goal "Volvo outlook" is blocked; open the conversation to resume it.',
  );
  notifier.remove();
});

it('falls back to the untitled copy for a conversation without a title', async () => {
  const fixture = activity();
  const notifier = mountNotifier(bagWith(fixture.store));

  await settleAfter(
    fixture,
    run('r1', 'running', {conversationId: 'c-untitled', title: null}),
    run('r1', 'succeeded', {conversationId: 'c-untitled', title: null}),
  );

  expect(sent).to.have.length(1);
  expect(sent[0]!.options.body).to.equal('DlightRAG finished generating your answer.');
  notifier.remove();
});

it('stays silent while the page has focus', async () => {
  document.hasFocus = () => true;
  const fixture = activity();
  const notifier = mountNotifier(bagWith(fixture.store));

  await settleAfter(fixture, run('r1', 'running'), run('r1', 'succeeded'));

  expect(sent).to.have.length(0);
  notifier.remove();
});

it('stays silent while it is not enabled or the reader has not allowed notifications', async () => {
  const fixture = activity();
  const bag = bagWith(fixture.store);
  const notifier = mountNotifier(bag, false);
  await settleAfter(fixture, run('r1', 'running'), run('r1', 'succeeded'));
  expect(sent).to.have.length(0);
  notifier.remove();

  installNotification('default');
  const second = mountNotifier(bag);
  await settleAfter(fixture, run('r2', 'running'), run('r2', 'succeeded'));
  expect(sent).to.have.length(0, 'nothing is shown without permission');
  second.remove();
});

it('opens the conversation and marks it read when its notification is clicked', async () => {
  const fixture = activity();
  const bag = bagWith(fixture.store);
  const notifier = mountNotifier(bag);

  await settleAfter(
    fixture,
    run('r1', 'running', {conversationId: 'c-other', title: 'Volvo outlook'}),
    run('r1', 'succeeded', {conversationId: 'c-other', title: 'Volvo outlook'}),
  );
  const notification = sent[0]!;
  expect([...fixture.store.unreadConversationIds]).to.deep.equal(['c-other']);

  notification.onclick?.();

  await waitFor(() => window.location.pathname === '/web/conversations/c-other');
  expect(notification.closed, 'the notification is dismissed').to.equal(true);
  expect([...fixture.store.unreadConversationIds], 'the conversation is read now').to.deep.equal([]);
  notifier.remove();
});

it('puts a running dot and an unread dot on the conversation rows they name', async () => {
  const fixture = activity();
  const bag = bagWith(fixture.store);
  bag.conversations.upsertSummary(summary('c-running', 'Running conversation'));
  bag.conversations.upsertSummary(summary('c-done', 'Finished conversation'));
  fixture.push([]);
  await fixture.store.refresh();
  fixture.push([
    run('r1', 'running', {conversationId: 'c-running'}),
    run('r2', 'succeeded', {conversationId: 'c-done'}),
  ]);
  await fixture.store.refresh();

  const list = document.createElement('dl-conversation-list') as DlConversationList;
  list.handles = bag;
  document.body.append(list);
  await list.updateComplete;

  const running = list.querySelector('[aria-label="Running"]');
  const unread = list.querySelector('[aria-label="Finished, unread"]');
  expect(running?.getAttribute('role')).to.equal('img');
  expect(unread?.getAttribute('role')).to.equal('img');
  expect(running?.closest('[data-conversation-id]')?.getAttribute('data-conversation-id'))
    .to.equal('c-running');
  expect(unread?.closest('[data-conversation-id]')?.getAttribute('data-conversation-id'))
    .to.equal('c-done');
  expect(list.querySelectorAll('.conversation-run-dot')).to.have.length(2);
  list.remove();
});
