// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {
  WebRouter,
  conversationRoute,
  newChatRoute,
  parseWebRoute,
  routePath,
  type NavigationDriver,
  type NavigationSnapshot,
} from './router.ts';

class FakeDriver implements NavigationDriver {
  snapshot: NavigationSnapshot;
  readonly pushes: string[] = [];
  readonly replacements: string[] = [];
  readonly goCalls: number[] = [];
  readonly #entries: NavigationSnapshot[];
  #position = 0;
  #listener: ((snapshot: NavigationSnapshot) => void) | null = null;

  constructor(path = '/web/') {
    this.snapshot = {href: `https://app.example.com${path}`, state: null};
    this.#entries = [this.snapshot];
  }

  current(): NavigationSnapshot {
    return this.snapshot;
  }

  push(path: string, state: Record<string, unknown>): void {
    this.pushes.push(path);
    this.snapshot = {href: `https://app.example.com${path}`, state};
    this.#entries.splice(this.#position + 1, Infinity, this.snapshot);
    this.#position = this.#entries.length - 1;
  }

  replace(path: string, state: Record<string, unknown>): void {
    this.replacements.push(path);
    this.snapshot = {href: `https://app.example.com${path}`, state};
    this.#entries[this.#position] = this.snapshot;
  }

  go(delta: number): void {
    this.goCalls.push(delta);
    const next = this.#position + delta;
    if (next < 0 || next >= this.#entries.length) return;
    this.#position = next;
    this.snapshot = this.#entries[this.#position];
    queueMicrotask(() => { this.#listener?.(this.snapshot); });
  }

  subscribe(listener: (snapshot: NavigationSnapshot) => void): () => void {
    this.#listener = listener;
    return () => { this.#listener = null; };
  }

  back(): void {
    this.go(-1);
  }
}

test('route parser distinguishes new chat, conversations, and unsupported pages', () => {
  assert.deepEqual(parseWebRoute('https://app.example.com/web/'), newChatRoute());
  assert.deepEqual(
    parseWebRoute('https://app.example.com/web/conversations/a%20b'),
    conversationRoute('a b'),
  );
  assert.deepEqual(parseWebRoute('https://app.example.com/web/files'), {kind: 'not-found'});
  assert.equal(routePath(conversationRoute('a/b')), '/web/conversations/a%2Fb');
});

test('accepted navigation owns history and notifies one listener', async () => {
  const driver = new FakeDriver();
  const router = new WebRouter(driver);
  const seen: string[] = [];
  router.start((route) => { seen.push(routePath(route)); });

  assert.equal(await router.navigate(conversationRoute('one')), true);

  assert.deepEqual(driver.pushes, ['/web/conversations/one']);
  assert.deepEqual(seen, ['/web/conversations/one']);
  assert.deepEqual(router.current, conversationRoute('one'));
});

test('a rejected navigation neither changes history nor notifies', async () => {
  const driver = new FakeDriver();
  const router = new WebRouter(driver);
  let notified = false;
  router.start(() => { notified = true; });
  router.setGuard(() => false);

  assert.equal(await router.navigate(conversationRoute('blocked')), false);

  assert.deepEqual(driver.pushes, []);
  assert.equal(notified, false);
  assert.deepEqual(router.current, newChatRoute());
});

test('one async push guard rejects overlapping navigation attempts', async () => {
  const driver = new FakeDriver();
  const router = new WebRouter(driver);
  let decide!: (allowed: boolean) => void;
  router.setGuard(() => new Promise<boolean>((resolve) => { decide = resolve; }));
  const first = router.navigate(conversationRoute('first'));

  assert.equal(await router.navigate(conversationRoute('second')), false);
  decide(true);
  assert.equal(await first, true);
  assert.deepEqual(router.current, conversationRoute('first'));
});

test('a rejected back navigation restores the prior history position', async () => {
  const driver = new FakeDriver();
  const router = new WebRouter(driver);
  router.start(() => {});
  await router.navigate(conversationRoute('one'));
  router.setGuard(() => false);

  driver.back();
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.deepEqual(driver.goCalls, [-1, 1]);
  assert.deepEqual(router.current, conversationRoute('one'));
});

test('repeated Back during an async guard cannot split URL and route state', async () => {
  const driver = new FakeDriver();
  const router = new WebRouter(driver);
  const seen: string[] = [];
  let decide!: (allowed: boolean) => void;
  const decision = new Promise<boolean>((resolve) => { decide = resolve; });
  router.start((route) => { seen.push(routePath(route)); });
  await router.navigate(conversationRoute('one'));
  router.setGuard(() => decision);

  driver.back();
  await new Promise((resolve) => setTimeout(resolve, 0));
  driver.back();
  await new Promise((resolve) => setTimeout(resolve, 0));
  decide(true);
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.deepEqual(router.current, newChatRoute());
  assert.equal(driver.current().href.endsWith('/web/'), true);
  assert.deepEqual(seen, ['/web/conversations/one', '/web/']);
});

test('silent replacement adopts an accepted server conversation without rerendering', async () => {
  const driver = new FakeDriver();
  const router = new WebRouter(driver);
  let notifications = 0;
  router.start(() => { notifications += 1; });

  assert.equal(await router.navigate(
    conversationRoute('created'),
    {replace: true, notify: false, bypassGuard: true},
  ), true);

  assert.equal(notifications, 0);
  assert.deepEqual(router.current, conversationRoute('created'));
  assert.equal(driver.replacements.at(-1), '/web/conversations/created');
});
