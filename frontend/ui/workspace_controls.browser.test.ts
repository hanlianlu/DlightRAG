// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {workspaceStore} from '../stores/workspaceStore.ts';
import type {DlWorkspaceScope} from './workspace_scope.ts';
import './workspace_scope.ts';
import type {DlIngestTarget} from './ingest_target.ts';
import './ingest_target.ts';
import type {ToastRequestDetail} from './toast.ts';

const originalFetch = window.fetch;

function buttonNamed(root: ParentNode, name: string): HTMLButtonElement | null {
  return Array.from(root.querySelectorAll<HTMLButtonElement>('button'))
    .find((button) => (button.getAttribute('aria-label') || button.textContent?.trim()) === name)
    ?? null;
}

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error('condition did not become true');
}

function mountScope(): DlWorkspaceScope {
  const scope = document.createElement('dl-workspace-scope') as DlWorkspaceScope;
  document.body.appendChild(scope);
  return scope;
}

beforeEach(() => {
  workspaceStore.init([
    {workspace: 'default', displayName: 'Default', embeddingModel: 'embed'},
  ], ['default'], 'default');
});

afterEach(() => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
});

it('owns a native expanded trigger and closes after typed creation intent', async () => {
  window.fetch = async () => new Response(JSON.stringify({
    workspace: 'research', display_name: 'Research',
  }), {status: 200, headers: {'Content-Type': 'application/json'}});
  const scope = mountScope();
  await scope.updateComplete;

  expect(customElements.get('workspace-scope')).to.equal(undefined);
  expect(customElements.get('workspace-create')).to.equal(undefined);
  expect(customElements.get('ingest-target')).to.equal(undefined);
  expect(scope.hasAttribute('role')).to.equal(false);
  expect(scope.hasAttribute('tabindex')).to.equal(false);
  const trigger = buttonNamed(scope, 'Choose search workspaces')!;
  expect(trigger.getAttribute('aria-haspopup')).to.equal('dialog');
  expect(trigger.getAttribute('aria-expanded')).to.equal('false');
  expect(trigger.getAttribute('aria-controls')).to.equal('workspace-popover');

  trigger.click();
  await scope.updateComplete;
  const popover = scope.querySelector<HTMLElement>('[role="dialog"][aria-label="Workspaces"]')!;
  expect(popover.hidden).to.equal(false);
  expect(popover.querySelector('[role="option"]')).to.equal(null);
  expect(popover.querySelector('dl-workspace-create')).not.to.equal(null);
  expect(trigger.getAttribute('aria-expanded')).to.equal('true');

  const input = scope.querySelector<HTMLInputElement>('[aria-label="New workspace name"]')!;
  input.value = 'Research';
  scope.querySelector<HTMLButtonElement>('[aria-label="Create workspace"]')?.click();
  await waitFor(() => workspaceStore.primary === 'research'
    && trigger.getAttribute('aria-expanded') === 'false');
  await scope.updateComplete;

  expect(scope.textContent).to.contain('Research');
  expect(popover.hidden).to.equal(true);
  expect(trigger.getAttribute('aria-expanded')).to.equal('false');
  expect(document.activeElement).to.equal(trigger);
});

it('restores typed-name deletion intent after a failed workspace request', async () => {
  let calls = 0;
  window.fetch = async () => {
    calls += 1;
    return new Response(JSON.stringify({error: 'Deletion denied'}), {
      status: 500,
      headers: {'Content-Type': 'application/json'},
    });
  };
  const scope = mountScope();
  await scope.updateComplete;
  buttonNamed(scope, 'Choose search workspaces')?.click();
  await scope.updateComplete;
  scope.querySelector<HTMLButtonElement>('[aria-label="Delete workspace Default"]')?.click();
  await waitFor(() => Boolean(scope.querySelector<HTMLDialogElement>('dialog')?.open));
  const input = scope.querySelector<HTMLInputElement>('[aria-label="Type Default to confirm"]')!;
  input.value = 'Default';
  input.dispatchEvent(new Event('input'));
  await scope.updateComplete;
  const submit = buttonNamed(scope, 'Delete')!;
  expect(submit.disabled).to.equal(false);

  submit.click();
  await waitFor(() => calls === 1 && submit.disabled === false);

  expect(input.value).to.equal('Default');
  expect(scope.querySelector<HTMLDialogElement>('dialog')?.open).to.equal(true);
  expect(document.activeElement).to.equal(input);
});

it('keeps submitted creation connected while its popover is dismissed', async () => {
  let resolveCreate!: (response: Response) => void;
  let requestSignal: AbortSignal | null | undefined;
  window.fetch = async (_input, init) => {
    requestSignal = init?.signal;
    return await new Promise<Response>((resolve) => { resolveCreate = resolve; });
  };
  const scope = mountScope();
  await scope.updateComplete;
  buttonNamed(scope, 'Choose search workspaces')?.click();
  await scope.updateComplete;
  const input = scope.querySelector<HTMLInputElement>('[aria-label="New workspace name"]')!;
  input.value = 'Research';
  scope.querySelector<HTMLButtonElement>('[aria-label="Create workspace"]')?.click();

  scope.close();
  await scope.updateComplete;
  const composer = document.createElement('textarea');
  document.body.appendChild(composer);
  composer.focus();
  expect(scope.querySelector('dl-workspace-create')).not.to.equal(null);
  expect(scope.querySelector<HTMLElement>('[role="dialog"][aria-label="Workspaces"]')?.hidden)
    .to.equal(true);
  resolveCreate(new Response(JSON.stringify({
    workspace: 'research', display_name: 'Research',
  }), {status: 200, headers: {'Content-Type': 'application/json'}}));
  await waitFor(() => workspaceStore.primary === 'research');

  expect(requestSignal?.aborted).to.equal(false);
  expect(workspaceStore.records.some((record) => record.workspace === 'research')).to.equal(true);
  expect(document.activeElement).to.equal(composer);
});

it('aborts a submitted creation only when its Feature disconnects', async () => {
  let requestSignal: AbortSignal | null | undefined;
  window.fetch = async (_input, init) => await new Promise<Response>((_resolve, reject) => {
    requestSignal = init?.signal;
    requestSignal?.addEventListener('abort', () => {
      reject(new DOMException('Aborted', 'AbortError'));
    }, {once: true});
  });
  const scope = mountScope();
  await scope.updateComplete;
  buttonNamed(scope, 'Choose search workspaces')?.click();
  await scope.updateComplete;
  const input = scope.querySelector<HTMLInputElement>('[aria-label="New workspace name"]')!;
  input.value = 'Research';
  scope.querySelector<HTMLButtonElement>('[aria-label="Create workspace"]')?.click();
  await waitFor(() => requestSignal !== undefined);

  scope.remove();
  await waitFor(() => requestSignal?.aborted === true);
  expect(workspaceStore.records.some((record) => record.workspace === 'research')).to.equal(false);

  document.body.appendChild(scope);
  await scope.updateComplete;
  expect(scope.querySelector<HTMLInputElement>('[aria-label="New workspace name"]')?.disabled)
    .to.equal(false);
});

it('reports a submitted creation failure after its popover is dismissed', async () => {
  let rejectCreate!: (reason: unknown) => void;
  window.fetch = async () => await new Promise<Response>((_resolve, reject) => {
    rejectCreate = reject;
  });
  let receipt: ToastRequestDetail | null = null;
  const scope = mountScope();
  scope.addEventListener('dl-toast-request', (event) => { receipt = event.detail; });
  await scope.updateComplete;
  buttonNamed(scope, 'Choose search workspaces')?.click();
  await scope.updateComplete;
  const input = scope.querySelector<HTMLInputElement>('[aria-label="New workspace name"]')!;
  input.value = 'Research';
  scope.querySelector<HTMLButtonElement>('[aria-label="Create workspace"]')?.click();

  scope.close();
  await scope.updateComplete;
  rejectCreate(new TypeError('network unavailable'));
  await waitFor(() => receipt !== null);

  expect(receipt).to.deep.equal({message: 'Failed to create workspace', duration: 3000});
  expect(scope.querySelector<HTMLElement>('[role="dialog"][aria-label="Workspaces"]')?.hidden)
    .to.equal(true);
});

it('keeps a pending deletion modal and isolates the next deletion operation', async () => {
  workspaceStore.init([
    {workspace: 'default', displayName: 'Default', embeddingModel: 'embed'},
    {workspace: 'research', displayName: 'Research', embeddingModel: 'embed'},
  ], ['default'], 'default');
  const requests: Array<(response: Response) => void> = [];
  window.fetch = async () => await new Promise<Response>((resolve) => { requests.push(resolve); });
  const scope = mountScope();
  await scope.updateComplete;
  const trigger = buttonNamed(scope, 'Choose search workspaces')!;
  trigger.click();
  await scope.updateComplete;
  scope.querySelector<HTMLButtonElement>('[aria-label="Delete workspace Default"]')?.click();
  await waitFor(() => Boolean(scope.querySelector<HTMLDialogElement>('dialog')?.open));
  let input = scope.querySelector<HTMLInputElement>('[aria-label="Type Default to confirm"]')!;
  input.value = 'Default';
  input.dispatchEvent(new Event('input'));
  await scope.updateComplete;
  buttonNamed(scope, 'Delete')?.click();
  await waitFor(() => requests.length === 1
    && buttonNamed(scope, 'Deleting…')?.disabled === true);

  let dialog = scope.querySelector<HTMLDialogElement>('dialog')!;
  const cancel = buttonNamed(scope, 'Cancel')!;
  expect(cancel.disabled).to.equal(true);
  cancel.click();
  const cancelEvent = new Event('cancel', {cancelable: true});
  dialog.dispatchEvent(cancelEvent);
  expect(cancelEvent.defaultPrevented).to.equal(true);
  expect(dialog.open).to.equal(true);

  requests[0]!(new Response(JSON.stringify({error: 'Deletion denied'}), {
    status: 500,
    headers: {'Content-Type': 'application/json'},
  }));
  await waitFor(() => buttonNamed(scope, 'Cancel')?.disabled === false
    && input.readOnly === false);
  buttonNamed(scope, 'Cancel')?.click();
  await waitFor(() => !scope.querySelector<HTMLDialogElement>('dialog')?.open
    && document.activeElement === buttonNamed(scope, 'Choose search workspaces'));
  await scope.updateComplete;

  const nextTrigger = buttonNamed(scope, 'Choose search workspaces')!;
  nextTrigger.click();
  await scope.updateComplete;
  expect(nextTrigger.getAttribute('aria-expanded')).to.equal('true');
  const deleteResearch = scope.querySelector<HTMLButtonElement>(
    '[aria-label="Delete workspace Research"]',
  )!;
  expect(deleteResearch).not.to.equal(null);
  deleteResearch.click();
  await waitFor(() => Boolean(scope.querySelector<HTMLInputElement>(
    '[aria-label="Type Research to confirm"]',
  )) && Boolean(scope.querySelector<HTMLDialogElement>('dialog[open]')));
  dialog = scope.querySelector<HTMLDialogElement>('dialog[open]')!;
  input = scope.querySelector<HTMLInputElement>('[aria-label="Type Research to confirm"]')!;
  input.value = 'Research';
  input.dispatchEvent(new Event('input'));
  await scope.updateComplete;
  const submit = buttonNamed(scope, 'Delete')!;
  expect(submit.disabled).to.equal(false);
  submit.click();
  await waitFor(() => requests.length === 2);
  expect(scope.querySelector<HTMLInputElement>('[aria-label="Type Research to confirm"]')?.readOnly)
    .to.equal(true);
  requests[1]!(new Response(JSON.stringify({
    workspace: 'research', next_workspace: 'default',
  }), {status: 200, headers: {'Content-Type': 'application/json'}}));
  await waitFor(() => !dialog.open);

  expect(workspaceStore.records.some((record) => record.workspace === 'research')).to.equal(false);
  expect(dialog.open).to.equal(false);
});

it('uses native dialog popover controls and restores ingest-trigger focus', async () => {
  workspaceStore.init([
    {workspace: 'default', displayName: 'Default', embeddingModel: 'embed'},
    {workspace: 'research', displayName: 'Research', embeddingModel: 'embed'},
  ], ['default'], 'default');
  const ingest = document.createElement('dl-ingest-target') as DlIngestTarget;
  ingest.active = true;
  document.body.appendChild(ingest);
  await ingest.updateComplete;
  const trigger = ingest.querySelector<HTMLButtonElement>('#ingest-target-trigger')!;
  expect(trigger.tagName).to.equal('BUTTON');
  expect(trigger.getAttribute('aria-haspopup')).to.equal('dialog');
  expect(trigger.getAttribute('aria-controls')).to.equal('ingest-target-popover');

  trigger.click();
  await ingest.updateComplete;
  const popover = ingest.querySelector<HTMLElement>(
    '[role="dialog"][aria-label="Select ingest workspace"]',
  )!;
  expect(popover.hidden).to.equal(false);
  expect(popover.querySelector('[role="option"]')).to.equal(null);
  expect(document.activeElement).to.equal(
    popover.querySelector('[data-ingest-workspace-choice][aria-pressed="true"]'),
  );

  document.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Escape', bubbles: true, cancelable: true,
  }));
  await waitFor(() => popover.hidden && document.activeElement === trigger);

  trigger.click();
  await ingest.updateComplete;
  const createInput = ingest.querySelector<HTMLInputElement>('[aria-label="New workspace name"]')!;
  createInput.focus();
  ingest.querySelector('dl-workspace-create')?.dispatchEvent(new CustomEvent(
    'dl-workspace-created',
    {detail: {workspace: 'new'}, bubbles: true, composed: true},
  ));
  await waitFor(() => popover.hidden && document.activeElement === trigger);
});

it('resets workspace popover and deletion state across disconnect and reconnect', async () => {
  const scope = mountScope();
  const modalStates: boolean[] = [];
  scope.addEventListener('dl-modal-state-change', (event) => {
    modalStates.push(event.detail.open);
  });
  await scope.updateComplete;
  const trigger = buttonNamed(scope, 'Choose search workspaces')!;
  trigger.click();
  await scope.updateComplete;
  scope.querySelector<HTMLButtonElement>('[aria-label="Delete workspace Default"]')?.click();
  const dialog = scope.querySelector<HTMLDialogElement>('dialog')!;
  await waitFor(() => dialog.open);
  expect(modalStates.at(-1)).to.equal(true);

  scope.remove();
  document.body.appendChild(scope);
  await scope.updateComplete;

  const reconnectedTrigger = buttonNamed(scope, 'Choose search workspaces')!;
  const popover = scope.querySelector<HTMLElement>('[role="dialog"][aria-label="Workspaces"]')!;
  expect(reconnectedTrigger.getAttribute('aria-expanded')).to.equal('false');
  expect(popover.hidden).to.equal(true);
  expect(dialog.open).to.equal(false);
  await waitFor(() => modalStates.at(-1) === false);

  reconnectedTrigger.click();
  await scope.updateComplete;
  document.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Escape', bubbles: true, cancelable: true,
  }));
  await waitFor(() => popover.hidden && document.activeElement === reconnectedTrigger);
});

it('resets ingest popover state and document listeners across reconnect', async () => {
  const ingest = document.createElement('dl-ingest-target') as DlIngestTarget;
  ingest.active = true;
  document.body.appendChild(ingest);
  await ingest.updateComplete;
  const trigger = ingest.querySelector<HTMLButtonElement>(
    '[aria-label="Files in Default; choose file workspace"]',
  )!;
  trigger.click();
  await ingest.updateComplete;
  expect(trigger.getAttribute('aria-expanded')).to.equal('true');

  ingest.remove();
  document.body.appendChild(ingest);
  await ingest.updateComplete;
  const reconnectedTrigger = ingest.querySelector<HTMLButtonElement>(
    '[aria-label="Files in Default; choose file workspace"]',
  )!;
  const popover = ingest.querySelector<HTMLElement>(
    '[role="dialog"][aria-label="Select ingest workspace"]',
  )!;
  expect(reconnectedTrigger.getAttribute('aria-expanded')).to.equal('false');
  expect(popover.hidden).to.equal(true);

  reconnectedTrigger.click();
  await ingest.updateComplete;
  document.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Escape', bubbles: true, cancelable: true,
  }));
  await waitFor(() => popover.hidden && document.activeElement === reconnectedTrigger);
});

it('restores workspace-selector trigger focus when deletion is cancelled', async () => {
  const scope = mountScope();
  await scope.updateComplete;
  const trigger = buttonNamed(scope, 'Choose search workspaces')!;
  trigger.click();
  await scope.updateComplete;
  scope.querySelector<HTMLButtonElement>('[aria-label="Delete workspace Default"]')?.click();
  await waitFor(() => Boolean(scope.querySelector<HTMLDialogElement>('dialog')?.open));

  buttonNamed(scope, 'Cancel')?.click();
  await waitFor(() => document.activeElement === trigger);

  expect(scope.querySelector<HTMLDialogElement>('dialog')?.open).to.equal(false);
  expect(document.activeElement).to.equal(trigger);
});
