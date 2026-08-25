// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import type {AnswerArtifact, AnswerPresentation} from '../api/conversations.ts';
import './app.ts';
import type {DlApp} from './app.ts';
import type {DlChatFeature} from './chat_feature.ts';
import type {
  AnswerPresentationElement,
  AnswerSourceOpenDetail,
} from './answer_presentation.ts';
import {setupPanel} from './panel.ts';

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
  window.fetch = async () => response(bootstrap);
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);

  const loaded = await app.ready;

  expect(loaded).to.deep.equal(bootstrap);
  const shell = app.querySelector<HTMLElement>('#app');
  expect(shell?.inert).to.equal(false);
  expect(app.querySelector('workspace-scope')?.getAttribute('data-primary')).to.equal('default');
  const chat = app.querySelector<DlChatFeature>('dl-chat-feature');
  expect(chat?.attachmentPolicy?.countLimit).to.equal(6);
  expect(chat?.attachmentPolicy?.documentMaxBytes).to.equal(2048);
  expect(chat?.attachmentAccept).to.equal('image/*,.md,.pdf');
  expect(chat?.querySelector('dl-chat-message-list')).not.to.equal(null);
  expect(chat?.querySelector('dl-chat-composer')).not.to.equal(null);
});

it('owns Shell message layout while preserving the welcome for an empty conversation', async () => {
  window.fetch = async () => response(bootstrap);
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
    : response(bootstrap);
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
    detail: {referenceId: '1', returnFocus: canvasPresentation},
  }));
  await waitFor(() => app.querySelector('#panel')?.classList.contains('open') === true);
  await new Promise((resolve) => requestAnimationFrame(resolve));

  expect(canvas.classList.contains('open')).to.equal(false);
  expect(canvas.inert).to.equal(true);
  expect(app.querySelector<HTMLElement>('#panel')?.inert).to.equal(false);
  expect(document.body.classList.contains('panel-drawer-open')).to.equal(true);
  expect(document.body.classList.contains('artifact-canvas-modal')).to.equal(false);
  expect(document.activeElement).to.equal(app.querySelector('#panel-close-btn'));
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
    : response(bootstrap);
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);
  await app.ready;
  setupPanel();
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
    detail: {referenceId: '1', returnFocus: canvasPresentation},
  }));
  await waitFor(() => app.querySelector('#panel')?.classList.contains('open') === true);
  expect(canvas.classList.contains('open')).to.equal(true);
  expect(canvas.layout).to.equal('side');

  app.querySelector<HTMLButtonElement>('#panel-close-btn')?.click();

  expect(app.querySelector('#panel')?.classList.contains('open')).to.equal(false);
  expect(canvas.classList.contains('open')).to.equal(true);
  expect(document.activeElement).to.equal(canvasPresentation);
});

it('fails closed and resolves the same ready promise after an explicit retry', async () => {
  let attempts = 0;
  window.fetch = async () => {
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
