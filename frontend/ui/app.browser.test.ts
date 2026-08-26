// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import type {AnswerArtifact, AnswerPresentation} from '../api/conversations.ts';
import './app.ts';
import type {DlApp} from './app.ts';
import type {DlChatFeature} from './chat_feature.ts';
import type {ImageOpenDetail} from './image_lightbox.ts';
import type {DlContinuationDialog} from './run_dialogs.ts';
import type {DlSettingsDialog} from './settings.ts';
import type {
  AnswerPresentationElement,
  AnswerSourceOpenDetail,
} from './answer_presentation.ts';
import type {DlToastRegion, ToastRequestDetail} from './toast.ts';

const bootstrap = {
  contract_version: 1,
  workspaces: [
    {workspace: 'default', display_name: 'Default', embedding_model: 'embed-test'},
  ],
  primary_workspace: 'default',
  active_workspaces: ['default'],
  answer_attachments: {
    count_limit: 6,
    image_max_bytes: 1024,
    document_max_bytes: 2048,
    extensions: ['md', 'pdf'],
    image_capability: 'supported',
    image_limit: 3,
    accept: 'image/*,.md,.pdf',
  },
  active_html_preview_enabled: true,
} as const;

const SAFE_PNG =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=';

const originalFetch = window.fetch;
const originalMatchMedia = window.matchMedia;
const OriginalResizeObserver = window.ResizeObserver;

function desktopMedia(query: string): MediaQueryList {
  return {
    matches: query === '(min-width: 1200px)',
    media: query,
    onchange: null,
    addListener() {},
    removeListener() {},
    addEventListener() {},
    removeEventListener() {},
    dispatchEvent: () => true,
  };
}

function compactMedia(query: string): MediaQueryList {
  return {
    matches: query === '(max-width: 1199px)',
    media: query,
    onchange: null,
    addListener() {},
    removeListener() {},
    addEventListener() {},
    removeEventListener() {},
    dispatchEvent: () => true,
  };
}

function response(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {'Content-Type': 'application/json'},
  });
}

function bootstrapResponse(input: RequestInfo | URL): Response {
  return String(input) === '/web/api/conversations' ? response([]) : response(bootstrap);
}

function dialogNamed(root: ParentNode, name: string): HTMLDialogElement | null {
  return Array.from(root.querySelectorAll<HTMLDialogElement>('dialog')).find((dialog) => {
    const labelledBy = dialog.getAttribute('aria-labelledby');
    return labelledBy
      ? root.querySelector<HTMLElement>(`#${labelledBy}`)?.textContent?.trim() === name
      : dialog.getAttribute('aria-label') === name;
  }) ?? null;
}

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error('condition did not become true');
}

beforeEach(() => {
  // Bootstrap tests do not exercise WebAwesome split geometry; its observer can
  // report a benign loop when the test document intentionally has no app CSS.
  window.ResizeObserver = class {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  } as typeof ResizeObserver;
});

afterEach(() => {
  window.fetch = originalFetch;
  window.matchMedia = originalMatchMedia;
  window.ResizeObserver = OriginalResizeObserver;
  document.body.replaceChildren();
  document.body.className = '';
});

it('renders the application shell from the typed bootstrap before resolving ready', async () => {
  window.fetch = async (input) => bootstrapResponse(input);
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);

  const loaded = await app.ready;

  expect(loaded).to.deep.equal(bootstrap);
  const shell = app.querySelector<HTMLElement>('#app');
  expect(shell?.inert).to.equal(false);
  expect(app.querySelector('dl-workspace-scope')?.textContent).to.contain('All workspaces (1)');
  const chat = app.querySelector<DlChatFeature>('dl-chat-feature');
  expect(chat?.attachmentPolicy?.countLimit).to.equal(6);
  expect(chat?.attachmentPolicy?.documentMaxBytes).to.equal(2048);
  expect(chat?.attachmentAccept).to.equal('image/*,.md,.pdf');
  expect(chat?.querySelector('dl-chat-message-list')).not.to.equal(null);
  expect(chat?.querySelector('dl-chat-composer')).not.to.equal(null);
});

it('keeps the composed compact conversation modal interactive and inerts sibling Shell UI', async () => {
  window.matchMedia = compactMedia;
  window.fetch = async (input) => bootstrapResponse(input);
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);
  await app.ready;

  const offer = app.querySelector<HTMLElement>('#notify-offer')!;
  offer.hidden = false;
  const toast = app.querySelector<DlToastRegion>('dl-toast-region')!;
  const open = app.querySelector<HTMLButtonElement>('[aria-label="Open conversations"]')!;
  open.click();
  await waitFor(() => document.body.classList.contains('conversation-drawer-open'));
  await app.updateComplete;

  const navigation = app.querySelector<HTMLElement>('nav[aria-label="Conversations"]')!;
  const chat = app.querySelector<DlChatFeature>('dl-chat-feature')!;
  const topbar = app.querySelector<HTMLElement>('.topbar')!;
  expect(navigation.inert).to.equal(false);
  expect(document.activeElement).to.equal(app.querySelector('#new-conversation-btn'));
  expect(topbar.inert).to.equal(false);
  expect(chat.inert).to.equal(true);
  expect(offer.inert).to.equal(true);
  toast.showAction('Composed receipt', {actionLabel: 'Undo', onAction: async () => {}});
  await toast.updateComplete;
  expect(toast.textContent).to.contain('Composed receipt');
  expect(toast.inert).to.equal(true);
  expect(app.querySelector<HTMLElement>('dl-workspace-scope')?.inert).to.equal(true);
  expect(app.querySelector<HTMLElement>('#files-btn')?.inert).to.equal(true);

  document.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Escape', bubbles: true, cancelable: true,
  }));
  await waitFor(() => !document.body.classList.contains('conversation-drawer-open'));
  await app.updateComplete;
  await toast.updateComplete;
  expect(chat.inert).to.equal(false);
  expect(offer.inert).to.equal(false);
  expect(toast.inert).to.equal(false);
  expect(app.querySelector<HTMLElement>('dl-workspace-scope')?.inert).to.equal(false);
  expect(document.activeElement).to.equal(open);
});

it('pauses an actionable receipt while the Image Lightbox makes the app inert', async () => {
  window.fetch = async (input) => bootstrapResponse(input);
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);
  await app.ready;

  const toast = app.querySelector<DlToastRegion>('dl-toast-region')!;
  toast.showAction('Undo available', {
    actionLabel: 'Undo',
    onAction: async () => {},
    duration: 40,
  });
  const returnFocus = Array.from(app.querySelectorAll<HTMLButtonElement>('button'))
    .find((button) => button.textContent?.trim() === 'Files')!;
  returnFocus.dispatchEvent(new CustomEvent<ImageOpenDetail>('dl-image-open', {
    detail: {src: SAFE_PNG, gallery: [SAFE_PNG], returnFocus},
    bubbles: true,
    composed: true,
  }));
  const lightbox = app.querySelector<HTMLElement>('[role="dialog"][aria-label="Image viewer"]')!;
  await waitFor(() => lightbox.getAttribute('aria-hidden') === 'false');
  await app.updateComplete;
  await toast.updateComplete;

  expect(toast.inert).to.equal(true);
  await new Promise((resolve) => setTimeout(resolve, 60));
  await toast.updateComplete;
  expect(toast.textContent).to.contain('Undo available');
  expect(toast.querySelector('button')?.textContent).to.equal('Undo');

  document.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Escape', bubbles: true, cancelable: true,
  }));
  await waitFor(() => lightbox.getAttribute('aria-hidden') === 'true');
  await app.updateComplete;
  await toast.updateComplete;
  await new Promise((resolve) => requestAnimationFrame(resolve));
  expect(toast.inert).to.equal(false);
  expect(document.activeElement).to.equal(returnFocus);

  await new Promise((resolve) => setTimeout(resolve, 60));
  await toast.updateComplete;
  expect(toast.textContent?.trim()).to.equal('');
  expect(toast.querySelector('button')).to.equal(null);
});

it('keeps Undo available across Settings and a sibling native modal', async () => {
  window.fetch = async (input) => String(input) === '/web/api/memory/settings'
    ? response({enabled: true, active_count: 0})
    : bootstrapResponse(input);
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);
  await app.ready;

  const settings = app.querySelector<DlSettingsDialog>('dl-settings-dialog')!;
  const trigger = app.querySelector<HTMLButtonElement>('[aria-label="Settings"]')!;
  await settings.open(trigger);
  const settingsDialog = dialogNamed(settings, 'Settings')!;
  await waitFor(() => settingsDialog.open);

  let undone = false;
  settings.dispatchEvent(new CustomEvent<ToastRequestDetail>('dl-toast-request', {
    detail: {
      message: 'Undo available',
      action: {
        actionLabel: 'Undo',
        duration: 40,
        onAction: async () => { undone = true; },
      },
    },
    bubbles: true,
    composed: true,
  }));
  const toast = app.querySelector<DlToastRegion>('dl-toast-region')!;
  await app.updateComplete;
  await toast.updateComplete;
  expect(toast.inert).to.equal(true);

  await new Promise((resolve) => setTimeout(resolve, 60));
  expect(toast.querySelector<HTMLButtonElement>('button')?.textContent).to.equal('Undo');

  const continuation = app.querySelector<DlContinuationDialog>('dl-continuation-dialog')!;
  continuation.open('follow-up');
  const continuationDialog = dialogNamed(continuation, 'Follow up')!;
  await waitFor(() => continuationDialog.open);
  settingsDialog.close();
  await waitFor(() => !settingsDialog.open);
  await app.updateComplete;
  await toast.updateComplete;
  expect(toast.inert).to.equal(true);

  continuationDialog.close();
  await waitFor(() => !continuationDialog.open && !toast.inert);
  expect(toast.inert).to.equal(false);
  toast.querySelector<HTMLButtonElement>('button')?.click();
  await waitFor(() => undone && toast.textContent?.includes('Change undone.') === true);
  expect(toast.textContent).to.contain('Change undone.');
});

it('routes typed toast intent through Shell composition to the public Toast command', async () => {
  window.fetch = async (input) => bootstrapResponse(input);
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);
  await app.ready;

  const toast = app.querySelector<DlToastRegion>('dl-toast-region')!;
  app.querySelector('dl-settings-dialog')?.dispatchEvent(
    new CustomEvent<ToastRequestDetail>('dl-toast-request', {
      detail: {
        message: 'Composed receipt',
        action: {actionLabel: 'Undo', onAction: async () => {}},
      },
      bubbles: true,
      composed: true,
    }),
  );
  await toast.updateComplete;

  expect(toast.textContent).to.contain('Composed receipt');
  expect(toast.inert).to.equal(false);
  expect(toast.querySelector('button')?.textContent).to.equal('Undo');
});

it('closes conversation-scoped Inspector content on a typed route reset', async () => {
  window.matchMedia = desktopMedia;
  window.fetch = async (input) => bootstrapResponse(input);
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);
  await app.ready;

  const inspector = app.querySelector('dl-inspector')!;
  await inspector.openSources({
    answer_text: '', parts: [], sources: [], evidence_images: [], artifacts: [],
    artifact_outcome: {status: 'complete', issues: []},
  });
  app.querySelector('dl-conversation-sidebar')?.dispatchEvent(new CustomEvent(
    'dl-conversation-route-change',
    {
      bubbles: true,
      composed: true,
      detail: {previousConversationId: null, nextConversationId: null},
    },
  ));
  await inspector.updateComplete;

  expect(inspector.open).to.equal(false);
  expect(inspector.inert).to.equal(true);
});

it('owns Shell message layout while preserving the welcome for an empty conversation', async () => {
  window.fetch = async (input) => bootstrapResponse(input);
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);
  await app.ready;
  const chat = app.querySelector<DlChatFeature>('dl-chat-feature')!;

  chat.view = {kind: 'ready', conversationId: 'empty', history: [], lineage: null};
  await chat.updateComplete;
  await chat.querySelector('dl-chat-message-list')?.updateComplete;
  expect(app.querySelector('.app')?.classList.contains('has-messages')).to.equal(false);
  expect(chat.querySelector('.welcome')?.textContent).to.contain('Ask anything');

  const answer: AnswerPresentation = {
    answer_text: 'Stored answer.',
    parts: [{
      type: 'markdown', text: 'Stored answer.', html: '<p>Stored answer.</p>',
      artifact: null, evidence_image: null, inline: false,
    }],
    sources: [], evidence_images: [], artifacts: [],
    artifact_outcome: {status: 'complete', issues: []},
  };
  chat.view = {
    kind: 'ready', conversationId: 'filled', lineage: null,
    history: [{
      turn_id: 'turn-1', turn_number: 1, answer_run_id: 'run-1', submission_id: 'submission-1',
      status: 'succeeded', cancel_requested: false, user_text: 'Question', assistant_text: 'Stored answer.',
      user_attachments: [], presentation: answer, usage: {}, evidence: {}, error_kind: null,
      error_message: null, created_at: '2026-01-01T00:00:00Z',
    }],
  };
  await chat.updateComplete;
  expect(app.querySelector('.app')?.classList.contains('has-messages')).to.equal(true);

  chat.view = {kind: 'ready', conversationId: 'empty-again', history: [], lineage: null};
  await chat.updateComplete;
  await chat.querySelector('dl-chat-message-list')?.updateComplete;
  expect(app.querySelector('.app')?.classList.contains('has-messages')).to.equal(false);
  expect(chat.querySelector('.welcome')).not.to.equal(null);
});

it('opens Sources as the only compact modal when intent originates in Canvas', async () => {
  window.matchMedia = compactMedia;
  const presentation: AnswerPresentation = {
    answer_text: 'See [1].',
    parts: [{
      type: 'markdown', text: 'See [1].',
      html: '<p>See <cite class="citation-badge" data-ref="1">1</cite>.</p>',
      artifact: null, evidence_image: null, inline: false,
    }],
    sources: [{
      id: '1', title: 'Source', source_url: null, download_url: null, chunks: [],
    }],
    evidence_images: [], artifacts: [], artifact_outcome: {status: 'complete', issues: []},
  };
  window.fetch = async (input) => String(input).includes('/presentation')
    ? response(presentation)
    : bootstrapResponse(input);
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);
  await app.ready;
  const returnFocus = app.querySelector<HTMLButtonElement>('#files-btn')!;
  const artifact: AnswerArtifact = {
    resource_id: 'report-1', role: 'primary_report', media_type: 'text/markdown',
    label: 'Report', filename: 'report.md', byte_size: 20, digest: 'a'.repeat(64),
    presentation: 'markdown', status: 'available',
    uri: 'dlightrag://answer/run-1/artifacts/report-1', width: null, height: null,
    data_url: '/web/api/answer/run-1/artifacts/report-1',
    download_url: '/web/api/answer/run-1/artifacts/report-1?download=1',
    presentation_url: '/web/api/answer/run-1/artifacts/report-1/presentation', issue: null,
  };
  const canvas = app.querySelector('dl-artifact-canvas')!;
  await canvas.open(artifact, returnFocus);
  await canvas.updateComplete;
  await canvas.querySelector('dl-answer-presentation')?.updateComplete;

  const canvasPresentation = canvas.querySelector<AnswerPresentationElement>(
    'dl-answer-presentation',
  )!;
  canvasPresentation.dispatchEvent(new CustomEvent<AnswerSourceOpenDetail>('answer-source-open', {
    bubbles: true,
    composed: true,
    detail: {presentation, referenceId: '1', returnFocus: canvasPresentation},
  }));
  await waitFor(() => app.querySelector('#panel')?.classList.contains('open') === true);
  await new Promise((resolve) => requestAnimationFrame(resolve));

  expect(canvas.classList.contains('open')).to.equal(false);
  expect(canvas.inert).to.equal(true);
  expect(app.querySelector<HTMLElement>('#panel')?.inert).to.equal(false);
  expect(document.body.classList.contains('panel-drawer-open')).to.equal(true);
  expect(document.body.classList.contains('artifact-canvas-modal')).to.equal(false);
  expect(document.activeElement).to.equal(app.querySelector('#panel-close-btn'));

  app.querySelector<HTMLButtonElement>('#panel-close-btn')?.click();
  await new Promise((resolve) => requestAnimationFrame(resolve));
  expect(document.activeElement).to.equal(returnFocus);
});

it('restores a desktop Canvas citation when Sources closes alongside it', async () => {
  window.matchMedia = desktopMedia;
  const presentation: AnswerPresentation = {
    answer_text: 'See [1].',
    parts: [{
      type: 'markdown', text: 'See [1].',
      html: '<p>See <cite class="citation-badge" data-ref="1" tabindex="0">1</cite>.</p>',
      artifact: null, evidence_image: null, inline: false,
    }],
    sources: [{
      id: '1', title: 'Source', source_url: null, download_url: null, chunks: [],
    }],
    evidence_images: [], artifacts: [], artifact_outcome: {status: 'complete', issues: []},
  };
  window.fetch = async (input) => String(input).includes('/presentation')
    ? response(presentation)
    : bootstrapResponse(input);
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);
  await app.ready;
  const artifact: AnswerArtifact = {
    resource_id: 'report-desktop', role: 'primary_report', media_type: 'text/markdown',
    label: 'Report', filename: 'report.md', byte_size: 20, digest: 'b'.repeat(64),
    presentation: 'markdown', status: 'available',
    uri: 'dlightrag://answer/run-1/artifacts/report-desktop', width: null, height: null,
    data_url: '/web/api/answer/run-1/artifacts/report-desktop',
    download_url: '/web/api/answer/run-1/artifacts/report-desktop?download=1',
    presentation_url: '/web/api/answer/run-1/artifacts/report-desktop/presentation', issue: null,
  };
  const canvas = app.querySelector('dl-artifact-canvas')!;
  await canvas.open(artifact, app.querySelector('#files-btn'));
  await canvas.updateComplete;
  await canvas.querySelector('dl-answer-presentation')?.updateComplete;
  const canvasPresentation = canvas.querySelector<AnswerPresentationElement>(
    'dl-answer-presentation',
  )!;
  canvasPresentation.tabIndex = 0;
  canvasPresentation.focus();
  canvasPresentation.dispatchEvent(new CustomEvent<AnswerSourceOpenDetail>('answer-source-open', {
    bubbles: true,
    composed: true,
    detail: {presentation, referenceId: '1', returnFocus: canvasPresentation},
  }));
  await waitFor(() => app.querySelector('#panel')?.classList.contains('open') === true);
  expect(canvas.classList.contains('open')).to.equal(true);
  expect(canvas.layout).to.equal('side');

  app.querySelector<HTMLButtonElement>('#panel-close-btn')?.click();
  await new Promise((resolve) => requestAnimationFrame(resolve));

  expect(app.querySelector('#panel')?.classList.contains('open')).to.equal(false);
  expect(canvas.classList.contains('open')).to.equal(true);
  expect(document.activeElement).to.equal(canvasPresentation);
});

it('dismisses a lone desktop Artifact Canvas from the conversation area', async () => {
  window.matchMedia = desktopMedia;
  window.fetch = async (input) => bootstrapResponse(input);
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);
  await app.ready;
  const canvas = app.querySelector('dl-artifact-canvas')!;
  const artifact: AnswerArtifact = {
    resource_id: 'chart-1', role: 'attachment', media_type: 'image/png',
    label: 'Chart', filename: 'chart.png', byte_size: 20, digest: 'c'.repeat(64),
    presentation: 'image', status: 'available',
    uri: 'dlightrag://answer/run-1/artifacts/chart-1', width: 100, height: 100,
    data_url: '/web/api/answer/run-1/artifacts/chart-1',
    download_url: '/web/api/answer/run-1/artifacts/chart-1?download=1',
    presentation_url: null, issue: null,
  };
  await canvas.open(artifact, app.querySelector('#files-btn'));
  await canvas.updateComplete;
  expect(canvas.classList.contains('open')).to.equal(true);
  expect(app.querySelector('dl-inspector')?.open).to.equal(false);

  app.querySelector('main[aria-label="Chat"]')?.dispatchEvent(new MouseEvent('click', {
    bubbles: true,
    composed: true,
  }));

  expect(canvas.classList.contains('open')).to.equal(false);
  expect(document.body.classList.contains('artifact-canvas-open')).to.equal(false);

  await canvas.open(artifact, app.querySelector('#files-btn'));
  canvas.prepareForInspector();
  const inspectorReturn = app.querySelector<HTMLButtonElement>('#theme-trigger')!;
  const inspector = app.querySelector('dl-inspector')!;
  await inspector.openSources({
    answer_text: '', parts: [], sources: [], evidence_images: [], artifacts: [],
    artifact_outcome: {status: 'complete', issues: []},
  }, undefined, undefined, inspectorReturn);
  app.querySelector('main[aria-label="Chat"]')?.dispatchEvent(new MouseEvent('click', {
    bubbles: true,
    composed: true,
  }));
  await new Promise((resolve) => requestAnimationFrame(resolve));

  expect(inspector.open).to.equal(false);
  expect(canvas.classList.contains('open')).to.equal(false);
  expect(document.activeElement).to.equal(inspectorReturn);
});

it('fails closed and resolves the same ready promise after an explicit retry', async () => {
  let attempts = 0;
  window.fetch = async (input) => {
    if (String(input) === '/web/api/conversations') return response([]);
    attempts += 1;
    return attempts === 1 ? response({detail: 'down'}, 503) : response(bootstrap);
  };
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);
  await waitFor(() => app.bootState === 'error');

  expect(app.querySelector<HTMLElement>('#app')?.inert).to.equal(true);
  const retry = app.querySelector<HTMLButtonElement>('.bootstrap-status button');
  expect(retry).not.to.equal(null);
  retry?.click();

  await app.ready;
  expect(attempts).to.equal(2);
  expect(app.bootState).to.equal('ready');
});
