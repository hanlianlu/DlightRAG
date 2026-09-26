// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {IngestStore} from './ingest-store.ts';
import {WorkspaceStore} from './workspace-store.ts';

const originalDocument = globalThis.document;

test.beforeEach(() => {
  // WorkspaceStore persists the scope in cookies.
  Object.defineProperty(globalThis, 'document', {configurable: true, value: {cookie: ''}});
});

test.afterEach(() => {
  Object.defineProperty(globalThis, 'document', {configurable: true, value: originalDocument});
});

function stores(): {workspaces: WorkspaceStore; ingest: IngestStore; published: () => number} {
  const workspaces = new WorkspaceStore();
  workspaces.init([
    {workspace: 'default', displayName: 'Default', embeddingModel: 'embed'},
    {workspace: 'research', displayName: 'Research', embeddingModel: 'embed'},
  ], ['default'], 'default', null, null, null, 'default');
  const ingest = new IngestStore(workspaces);
  let count = 0;
  ingest.subscribe(() => { count += 1; });
  return {workspaces, ingest, published: () => count};
}

test('Files follows the search primary it reads, and says so when it moves', () => {
  const {workspaces, ingest, published} = stores();

  workspaces.select('research');

  assert.equal(ingest.workspace, 'research');
  assert.equal(published(), 1);
});

test('a scope change that keeps the primary does not re-target Files', () => {
  const {workspaces, ingest, published} = stores();

  workspaces.toggle('research'); // joins the scope and becomes its primary
  workspaces.toggle('default'); // narrows the scope; research stays primary

  assert.equal(ingest.workspace, 'research');
  assert.equal(published(), 1);
});

test('an explicit Files target stops following the primary', () => {
  const {workspaces, ingest, published} = stores();

  ingest.set('research');
  workspaces.select('default');

  assert.equal(ingest.workspace, 'research');
  assert.equal(published(), 1);
});
