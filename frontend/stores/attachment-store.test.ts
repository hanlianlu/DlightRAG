// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {test} from 'node:test';

import {AttachmentStore, type PendingAttachment} from './attachment-store.ts';

function file(name: string, type: string, size = 10): File {
  return {name, type, size} as File;
}

function makeStore() {
  const revoked: string[] = [];
  let seq = 0;
  let urlSeq = 0;
  const store = new AttachmentStore({
    createId: () => `id-${(seq += 1)}`,
    createObjectUrl: () => `blob:url-${(urlSeq += 1)}`,
    revokeObjectUrl: (url) => {
      revoked.push(url);
    },
  });
  return {store, revoked};
}

test('one ordered collection preserves user insertion order across mixed kinds', () => {
  const {store} = makeStore();
  store.add(file('a.png', 'image/png'), 'image');
  store.add(file('b.pdf', 'application/pdf'), 'document');
  store.add(file('c.png', 'image/png'), 'image');

  assert.deepEqual(
    store.list().map((item) => item.file.name),
    ['a.png', 'b.pdf', 'c.png'],
  );
  assert.deepEqual(
    store.list().map((item) => item.kind),
    ['image', 'document', 'image'],
  );
});

test('every pending item receives a stable id and preview object url', () => {
  const {store} = makeStore();
  const first = store.add(file('a.png', 'image/png'), 'image');
  const second = store.add(file('b.pdf', 'application/pdf'), 'document');

  assert.equal(first.id, 'id-1');
  assert.equal(second.id, 'id-2');
  assert.match(first.objectUrl, /^blob:url-/);
  assert.notEqual(first.objectUrl, second.objectUrl);
  assert.notEqual(first.id, second.id);
});

test('size and imageCount track the collection', () => {
  const {store} = makeStore();
  store.add(file('a.png', 'image/png'), 'image');
  store.add(file('b.pdf', 'application/pdf'), 'document');
  store.add(file('c.png', 'image/png'), 'image');

  assert.equal(store.size, 3);
  assert.equal(store.imageCount, 2);
});

test('remove by id drops only that item, keeps order, and revokes its url', () => {
  const {store, revoked} = makeStore();
  const a = store.add(file('a.png', 'image/png'), 'image');
  const b = store.add(file('b.pdf', 'application/pdf'), 'document');
  const c = store.add(file('c.png', 'image/png'), 'image');

  store.remove(b.id);

  assert.deepEqual(
    store.list().map((item) => item.id),
    [a.id, c.id],
  );
  assert.deepEqual(revoked, [b.objectUrl]);
});

test('clear empties the collection and revokes every url', () => {
  const {store, revoked} = makeStore();
  const a = store.add(file('a.png', 'image/png'), 'image');
  const b = store.add(file('b.pdf', 'application/pdf'), 'document');

  store.clear();

  assert.equal(store.size, 0);
  assert.deepEqual(revoked.sort(), [a.objectUrl, b.objectUrl].sort());
});

test('list returns a snapshot that cannot mutate store ownership', () => {
  const {store} = makeStore();
  store.add(file('a.png', 'image/png'), 'image');

  (store.list() as PendingAttachment[]).length = 0;

  assert.equal(store.size, 1);
});

test('a lease moves the existing File and Blob URL and can restore them', () => {
  const {store, revoked} = makeStore();
  const original = store.add(file('a.png', 'image/png'), 'image');

  const lease = store.leaseAll();

  assert.equal(store.size, 0);
  assert.equal(lease.items[0], original);
  lease.restore();
  assert.equal(store.list()[0], original);
  assert.deepEqual(revoked, []);
});

test('accepting or discarding a lease revokes its single Blob URL once', () => {
  const {store, revoked} = makeStore();
  const accepted = store.add(file('a.png', 'image/png'), 'image');
  const acceptedLease = store.leaseAll();
  acceptedLease.accept();
  acceptedLease.discard();
  const discarded = store.add(file('b.pdf', 'application/pdf'), 'document');
  store.leaseAll().discard();

  assert.deepEqual(revoked, [accepted.objectUrl, discarded.objectUrl]);
});

test('subscribers are notified on add, remove, clear, lease, and restore', () => {
  const {store} = makeStore();
  let notifications = 0;
  store.subscribe(() => {
    notifications += 1;
  });

  const a = store.add(file('a.png', 'image/png'), 'image');
  store.add(file('b.pdf', 'application/pdf'), 'document');
  store.remove(a.id);
  const lease = store.leaseAll();
  lease.restore();
  store.clear();

  assert.equal(notifications, 6);
});
