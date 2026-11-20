// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import type {AnswerArtifact, AnswerPresentation} from '../api/conversations.ts';
import {defineDesignSystemElements} from '../design-system/index.ts';
import './app.ts';

defineDesignSystemElements();

import type {ChildObservation} from '../api/conversations.ts';
import type {AnswerJudgement} from '../api/feedback.ts';
import {noteRefusal} from '../api/wire.ts';
import {conversationRoute, createBrowserRouter, newChatRoute} from '../lib/router.ts';
import {createAppHandles, productionHandles} from '../stores/app-handles.ts';
import {judgementKey} from '../stores/kept-judgements.ts';
import type {TracesSource} from '../stores/traces-source.ts';
import {buttonNamed, waitFor} from '../testing/dom.ts';
import {bootstrap, bootstrapResponse} from '../testing/app.ts';
import {DEFAULT_CHANGES} from '../testing/workspaces.ts';
import type {
  AnswerPresentationElement,
  AnswerSourceOpenDetail,
} from './answer-presentation.ts';
import type {DlApp} from './app.ts';
import type {DlChatFeature} from './chat-feature.ts';
import type {DlConversationSidebar} from './conversation-sidebar.ts';
import type {ImageOpenDetail} from './image-lightbox.ts';
import {webRouter} from './router.ts';
import type {DlContinuationDialog} from './run-dialogs.ts';
import type {DlSettingsDialog} from './settings.ts';
import {type DlToastRegion, type ToastRequestDetail, toastRegionOf} from './toast.ts';

const SAFE_PNG =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=';

const originalFetch = window.fetch;
const originalMatchMedia = window.matchMedia;
const originalSetTimeout = window.setTimeout;
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
    matches: query === '(width < 1200px)',
    media: query,
    onchange: null,
    addListener() {},
    removeListener() {},
    addEventListener() {},
    removeEventListener() {},
    dispatchEvent: () => true,
  };
}

/** Every wait the page asks for takes a seventy-fifth of its time: a receipt's three seconds pass in 40 ms. */
function pageTimersRunFast(): void {
  window.setTimeout = ((handler: TimerHandler, delay = 0) => originalSetTimeout(handler, delay / 75)) as typeof window.setTimeout;
}

function response(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {'Content-Type': 'application/json'},
  });
}

/** The reads Settings makes while it opens on Connections after a provider authorization. */
function connectionsReturnFetch(writes: string[] = []): typeof window.fetch {
  return async (input, init) => {
    if (init?.method && init.method !== 'GET') writes.push(String(input));
    if (String(input) === '/web/api/connections/mcp') {
      return response({revision: '0', connections: [], presets: []});
    }
    if (String(input) === '/web/api/agent-accounts') {
      return response({available: true, registration: {allowed: true, enabled: true}, accounts: []});
    }
    if (String(input) === '/web/api/memory/settings') return response({enabled: false, active_count: null});
    return bootstrapResponse(input);
  };
}

function dialogNamed(root: ParentNode, name: string): HTMLDialogElement | null {
  return Array.from(root.querySelectorAll<HTMLDialogElement>('dialog')).find((dialog) => {
    const labelledBy = dialog.getAttribute('aria-labelledby');
    return labelledBy
      ? root.querySelector<HTMLElement>(`#${labelledBy}`)?.textContent?.trim() === name
      : dialog.getAttribute('aria-label') === name;
  }) ?? null;
}

beforeEach(() => {
  // Bootstrap tests do not exercise split-layout geometry; its observer can
  // report a benign loop when the test document intentionally has no app CSS.
  window.ResizeObserver = class {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  } as typeof ResizeObserver;
});

afterEach(async () => {
  window.fetch = originalFetch;
  window.matchMedia = originalMatchMedia;
  window.setTimeout = originalSetTimeout;
  window.ResizeObserver = OriginalResizeObserver;
  document.body.replaceChildren();
  document.body.className = '';
  // The page keeps the route and the conversation a test opened; the next app starts on a new chat.
  productionHandles().conversations.openNew();
  await webRouter.navigate(newChatRoute(), {replace: true, notify: false, bypassGuard: true});
});

it('renders the application shell from the typed bootstrap before resolving ready', async () => {
  window.fetch = async (input) => bootstrapResponse(input);
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);

  const loaded = await app.ready;

  expect(loaded).to.deep.equal({
    contractVersion: 6,
    workspaces: [{workspace: 'default', displayName: 'Default', embeddingModel: 'embed-test', changes: DEFAULT_CHANGES}],
    workspacesNextCursor: null,
    primaryWorkspace: 'default',
    activeWorkspaces: ['default'],
    knownWorkspaces: ['default'],
    defaultWorkspace: 'default',
    answerAttachments: {
      countLimit: 6,
      imageMaxBytes: 1024,
      documentMaxBytes: 2048,
      extensions: ['md', 'pdf'],
      imageCapability: 'supported',
      imageLimit: 3,
      accept: 'image/*,.md,.pdf',
    },
    activeHtmlPreviewEnabled: true,
    agentEffort: {levels: ['low', 'high', 'max'], default: 'high'},
    answerFeedbackEnabled: true,
    // This deployment offers no browser channel, so the page starts no SDK.
    observability: null,
  });
  const shell = app.querySelector<HTMLElement>('#app');
  expect(shell?.inert).to.equal(false);
  expect(app.querySelector('dl-workspace-scope')?.textContent).to.contain('All workspaces (1)');
  const chat = app.querySelector<DlChatFeature>('dl-chat-feature');
  expect(chat?.attachmentPolicy?.countLimit).to.equal(6);
  expect(chat?.attachmentPolicy?.documentMaxBytes).to.equal(2048);
  expect(chat?.attachmentAccept).to.equal('image/*,.md,.pdf');
  expect(chat?.querySelector('dl-chat-message-list')).not.to.equal(null);
  expect(chat?.querySelector('dl-chat-composer')).not.to.equal(null);
  // Whether the reader is offered the thumbs reaches the chat and the Inspector, which hands it to the dock.
  expect(chat?.answerFeedbackEnabled).to.equal(true);
  expect(app.querySelector('dl-inspector')?.answerFeedbackEnabled).to.equal(true);
  // The offer reaches the composer, which is what draws the effort picker.
  expect(chat?.agentEffortOffer).to.deep.equal({levels: ['low', 'high', 'max'], default: 'high'});
  const composer = app.querySelector('dl-chat-composer');
  await composer?.updateComplete;
  expect(composer?.querySelector('.composer-effort-label')?.textContent).to.equal('High');
});

it('names the page for the conversation it shows, with one heading 1 that says the same', async () => {
  window.matchMedia = desktopMedia;
  let titledName = 'Quarterly revenue';
  const wire = (id: string) => ({
    conversation_id: id,
    title: id === 'titled' ? titledName : null,
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-02T00:00:00Z',
  });
  window.fetch = async (input, init) => {
    const url = String(input);
    if (url === '/web/api/conversations') {
      return response({items: [wire('titled'), wire('untitled')], next_cursor: null});
    }
    if (url === '/web/api/conversations/titled' && init?.method === 'PATCH') {
      titledName = 'Revenue, renamed';
      return response(wire('titled'));
    }
    if (url === '/web/api/conversations/gone/history') return response({detail: 'Conversation not found'}, 404);
    const history = /^\/web\/api\/conversations\/(titled|untitled)\/history$/.exec(url);
    if (history) return response({conversation: wire(history[1]!), turns: [], commands: [], next_cursor: null});
    return bootstrapResponse(input);
  };
  document.title = 'Before the app';
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);
  await app.ready;
  await waitFor(() => productionHandles().conversations.listState === 'ready');
  const headings = (): string[] => [...app.querySelectorAll('h1')].map((heading) => heading.textContent?.trim() ?? '');

  // A new chat is the product's own name.
  await waitFor(() => document.title === 'DlightRAG');
  expect(headings()).to.deep.equal(['DlightRAG']);
  expect(app.querySelector('h1')?.classList.contains('dl-sr-only'), 'and it is not drawn').to.equal(true);

  await webRouter.navigate(conversationRoute('titled'));
  await waitFor(() => document.title === 'Quarterly revenue · DlightRAG');
  expect(headings()).to.deep.equal(['Quarterly revenue']);

  // The title follows a rename, with no new route.
  await productionHandles().conversations.rename('titled', 'Revenue, renamed');
  await waitFor(() => document.title === 'Revenue, renamed · DlightRAG');
  expect(headings()).to.deep.equal(['Revenue, renamed']);

  // A conversation with no title yet, and one that is gone, name the page as a new chat does.
  await webRouter.navigate(conversationRoute('untitled'));
  await waitFor(() => productionHandles().conversations.activeConversationId === 'untitled');
  await waitFor(() => document.title === 'DlightRAG');
  expect(headings()).to.deep.equal(['DlightRAG']);
  await webRouter.navigate(conversationRoute('gone'));
  await waitFor(() => productionHandles().conversations.viewState === 'unavailable');
  expect(document.title).to.equal('DlightRAG');

  await webRouter.navigate(conversationRoute('titled'));
  await waitFor(() => document.title === 'Revenue, renamed · DlightRAG');
  await webRouter.navigate(newChatRoute());
  await waitFor(() => document.title === 'DlightRAG');

  // A page without the app is not named for a conversation it no longer shows.
  await webRouter.navigate(conversationRoute('titled'));
  await waitFor(() => document.title === 'Revenue, renamed · DlightRAG');
  app.remove();
  expect(document.title).to.equal('Before the app');
});

it('keeps one class on body for each panel that is open, and takes every one away when the Shell leaves', async () => {
  window.matchMedia = desktopMedia;
  window.fetch = async (input) => bootstrapResponse(input);
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);
  await app.ready;
  const classes = (): string[] => [...document.body.classList].sort();

  // The conversation list is open on a desktop.
  await waitFor(() => document.body.classList.contains('conversation-sidebar-open'));
  expect(classes()).to.deep.equal(['conversation-sidebar-open']);

  await app.querySelector('dl-inspector')!.openFiles();
  await waitFor(() => document.body.classList.contains('files-panel-open'));
  const canvas = app.querySelector('dl-artifact-canvas')!;
  await canvas.open({
    resourceId: 'chart-1', mediaType: 'image/png', label: 'Chart', filename: 'chart.png', byteSize: 20,
    digest: 'c'.repeat(64), presentation: 'image', status: 'available',
    uri: 'dlightrag://answer/run-1/artifacts/chart-1', width: 100, height: 100,
    dataUrl: '/web/api/answer/run-1/artifacts/chart-1',
    downloadUrl: '/web/api/answer/run-1/artifacts/chart-1?download=1',
    presentationUrl: null, issue: null, attestation: null,
  }, app.querySelector('#files-btn'));
  await waitFor(() => document.body.classList.contains('artifact-canvas-open'));
  // Only what a stylesheet reads is written: the Canvas being wide or over the screen is not a class.
  expect(canvas.layout).to.equal('wide');
  expect(classes()).to.deep.equal(['artifact-canvas-open', 'conversation-sidebar-open', 'files-panel-open']);

  app.remove();
  expect(classes(), 'what the Shell wrote stayed behind').to.deep.equal([]);
});

it('forgets what its own bag kept when the reader signs out, and leaves another bag alone', async () => {
  window.fetch = async (input) => bootstrapResponse(input);
  const bag = createAppHandles();
  const page = productionHandles();
  const key = judgementKey('run-9', 'main', 1);
  const given: AnswerJudgement = {value: 'up', reasons: [], note: null};
  bag.judgements.keep(key, given);
  page.judgements.keep(key, given);
  let built = 0;
  const source = (): TracesSource => { built += 1; return appCountingSource({child: 0}); };
  bag.traces.sourceFor('run-9', source);
  bag.submissions.start({
    submissionId: 'held', conversationId: null, query: 'Held', workspaces: [], mode: null,
  }, bag.attachments.leaseAll(), {
    submit: () => new Promise(() => undefined),
    lookup: async () => null,
  });
  const app = document.createElement('dl-app') as DlApp;
  app.handles = bag;
  document.body.appendChild(app);
  await app.ready;

  noteRefusal(401);
  await waitFor(() => app.signedOut);

  expect(bag.judgements.current(key, null), 'a rating outlived the sign-out').to.equal(null);
  expect(bag.submissions.list(), 'a submission outlived it').to.have.length(0);
  bag.traces.sourceFor('run-9', source);
  expect(built, 'the pages its last reader read were kept').to.equal(2);
  expect(page.judgements.current(key, null), 'the page\'s own bag was forgotten instead').to.deep.equal(given);
  page.judgements.forget();
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
  const sidebar = app.querySelector<DlConversationSidebar>('dl-conversation-sidebar')!;
  await sidebar.updateComplete;
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
  toast.show({message: 'Composed receipt', action: {actionLabel: 'Undo', onAction: async () => {}}});
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
  pageTimersRunFast();
  toast.show({message: 'Undo available', action: {actionLabel: 'Undo', onAction: async () => {}}});
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
  await new Promise((resolve) => originalSetTimeout(resolve, 60));
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

  await new Promise((resolve) => originalSetTimeout(resolve, 60));
  await toast.updateComplete;
  expect(toast.classList.contains('visible')).to.equal(false);
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
        onAction: async () => { undone = true; },
      },
    },
    bubbles: true,
    composed: true,
  }));
  // Raised from outside the dialog, the notice shows in the dialog's own region: the shell's lies under the scrim.
  const toast = app.querySelector<DlToastRegion>('dl-toast-region')!;
  const own = toastRegionOf(settingsDialog)!;
  await own.updateComplete;
  expect(own.querySelector<HTMLButtonElement>('button')?.textContent).to.equal('Undo');
  expect(toast.visible).to.equal(false);

  const continuation = app.querySelector<DlContinuationDialog>('dl-continuation-dialog')!;
  continuation.open();
  const continuationDialog = dialogNamed(continuation, 'Fork this answer')!;
  await waitFor(() => continuationDialog.open);
  settingsDialog.close();
  // Settings goes with its region and hands over what still awaits the reader, to a shell another modal holds.
  await waitFor(() => toast.visible);
  await app.updateComplete;
  await toast.updateComplete;
  expect(toast.inert).to.equal(true);

  pageTimersRunFast();
  await new Promise((resolve) => originalSetTimeout(resolve, 60));
  expect(toast.querySelector<HTMLButtonElement>('button')?.textContent).to.equal('Undo');

  continuationDialog.close();
  await waitFor(() => !continuationDialog.open && !toast.inert);
  expect(toast.inert).to.equal(false);
  toast.querySelector<HTMLButtonElement>('button')?.click();
  await waitFor(() => undone && toast.textContent?.includes('Change undone.') === true);
  expect(toast.textContent).to.contain('Change undone.');
});

it('shows in the open Settings dialog what the sidebar reports of a delete-all that failed, and keeps it after', async () => {
  window.fetch = async (input, init) => (
    String(input) === '/web/api/conversations' && init?.method === 'DELETE'
      ? new Response(null, {status: 500})
      : bootstrapResponse(input)
  );
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);
  await app.ready;
  const settings = app.querySelector<DlSettingsDialog>('dl-settings-dialog')!;
  await settings.open(null, 'conversations');
  const settingsDialog = dialogNamed(settings, 'Settings')!;
  await waitFor(() => settingsDialog.open);

  buttonNamed(settings, 'Delete all conversations')!.click();
  const confirm = app.querySelector<HTMLDialogElement>('#delete-all-conversations-dialog')!;
  await waitFor(() => confirm.open);
  buttonNamed(confirm, 'Delete all')!.click();

  // The sidebar raised the error from outside the dialog, which holds the top layer: it shows in the
  // dialog's error region, in reach, and not in the shell's under the scrim.
  const own = toastRegionOf(settingsDialog, 'error')!;
  const shell = toastRegionOf(app, 'error')!;
  expect(own, 'the dialog has an error region of its own').not.to.equal(shell);
  await waitFor(() => own.visible);
  expect(own.textContent).to.contain('Could not delete conversations.');
  expect(own.closest('[inert]') === null, 'in reach, in the accessibility tree').to.equal(true);
  expect(shell.visible).to.equal(false);

  // The reader had not dismissed it when the dialog closed, so the shell goes on showing it.
  settingsDialog.close();
  await waitFor(() => shell.visible);
  expect(shell.textContent).to.contain('Could not delete conversations.');
});

it('keeps an error within reach while a compact conversation drawer holds the shell, and its clock stopped', async () => {
  window.matchMedia = compactMedia;
  window.fetch = async (input) => bootstrapResponse(input);
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);
  await app.ready;
  const error = toastRegionOf(app, 'error')!;
  const receipt = toastRegionOf(app)!;

  const open = app.querySelector<HTMLButtonElement>('[aria-label="Open conversations"]')!;
  open.click();
  await waitFor(() => document.body.classList.contains('conversation-drawer-open'));
  await app.updateComplete;
  pageTimersRunFast();
  error.show({message: 'Could not save.'});
  receipt.show({message: 'Saved.', action: {actionLabel: 'Undo', onAction: async () => {}}});
  await app.updateComplete;
  await Promise.all([error.updateComplete, receipt.updateComplete]);

  expect(receipt.inert, 'a receipt is out of reach under the drawer').to.equal(true);
  expect(error.closest('[inert]') === null, 'an error is not: it stays in the accessibility tree').to.equal(true);
  const dismiss = buttonNamed(error, 'Dismiss')!;
  dismiss.focus();
  expect(document.activeElement === dismiss, 'and its Dismiss takes focus').to.equal(true);
  await new Promise((resolve) => originalSetTimeout(resolve, 150));
  expect(error.visible, 'it outlasts its eight seconds, which run no faster than the reader can act').to.equal(true);
});

it('puts focus in the Chat\'s box when an error dismissed from the keyboard has nowhere to give it back', async () => {
  window.fetch = async (input) => bootstrapResponse(input);
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);
  await app.ready;
  const error = toastRegionOf(app, 'error')!;
  error.show({message: 'Could not save.'});
  await error.updateComplete;
  const composer = app.querySelector<HTMLTextAreaElement>('[aria-label="Message"]')!;
  expect(document.activeElement === composer).to.equal(false);

  // The toast asks when the place the reader came from cannot take focus back.
  error.dispatchEvent(new CustomEvent('dl-toast-focus-lost', {bubbles: true, composed: true}));

  await waitFor(() => document.activeElement === composer);
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

it('closes conversation-scoped Inspector content when the page moves to another conversation', async () => {
  window.matchMedia = desktopMedia;
  window.fetch = async (input) => String(input) === '/web/api/conversations/moved/history'
    ? response({
      conversation: {
        conversation_id: 'moved', title: 'Moved',
        created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z',
      },
      turns: [],
      commands: [],
      next_cursor: null,
    })
    : bootstrapResponse(input);
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);
  await app.ready;

  const inspector = app.querySelector('dl-inspector')!;
  await inspector.openSources({
    answerText: '', parts: [], sources: [], evidenceImages: [], linkCards: [], artifacts: [],
    artifactOutcome: {status: 'complete', issues: []},
  });
  expect(inspector.open).to.equal(true);
  await webRouter.navigate(conversationRoute('moved'));
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
  expect(chat.querySelector('.welcome')?.textContent).to.contain('Deep work, delegated.');

  const answer: AnswerPresentation = {
    answerText: 'Stored answer.',
    parts: [{
      type: 'markdown', text: 'Stored answer.', html: '<p>Stored answer.</p>',
      artifact: null, evidenceImage: null, inline: false,
    }],
    sources: [], evidenceImages: [], linkCards: [], artifacts: [],
    artifactOutcome: {status: 'complete', issues: []},
  };
  chat.view = {
    kind: 'ready', conversationId: 'filled', lineage: null,
    history: [{
      turnId: 'turn-1', turnNumber: 1, answerRunId: 'run-1', submissionId: 'submission-1',
      status: 'succeeded', cancelRequested: false, userText: 'Question', assistantText: 'Stored answer.',
      userAttachments: [], steeringMessages: [], presentation: answer, usage: {}, errorKind: null,
      errorMessage: null, childCount: 0, answerSequence: null, feedback: null, createdAt: '2026-01-01T00:00:00Z',
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

function childRow(id: string): Record<string, unknown> {
  return {
    child_session_id: id, status: 'running', objective: `objective ${id}`, model_role: 'query',
    usage: null, operation_id: `op-${id}`, operation_sequence: 1, operation_status: 'running',
    cancellation_origin: null, summary: null, result_handles: [], started_at: null,
    finished_at: null, pending_questions: 0,
  };
}

/** A conversation whose one settled turn started two children, as the server puts it on the wire. */
const settledTurnWire = {
  turn_id: 'turn-1', turn_number: 1, answer_run_id: 'run-1', submission_id: 'submission-1',
  status: 'succeeded', cancel_requested: false, user_text: 'Question', assistant_text: 'Answer.',
  user_attachments: [], steering_messages: [], usage: {}, error_kind: null, error_message: null,
  child_count: 2,
  created_at: '2026-01-01T00:00:00Z',
  presentation: {
    answer_text: 'Answer.',
    parts: [{
      type: 'markdown', text: 'Answer.', html: '<p>Answer.</p>',
      artifact: null, evidence_image: null, inline: false,
    }],
    sources: [], evidence_images: [], artifacts: [], artifact_outcome: {status: 'complete', issues: []},
  },
};

/** The app on a conversation whose settled turn started two children, with the server answering the Run's roster. */
async function appWithChildren(roster: {requests: number}, media = desktopMedia): Promise<DlApp> {
  window.matchMedia = media;
  window.fetch = async (input) => {
    const url = String(input);
    if (url === '/web/api/conversations/with-children/history') {
      return response({
        conversation: {
          conversation_id: 'with-children', title: 'Children',
          created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z',
        },
        turns: [settledTurnWire],
        commands: [],
        next_cursor: null,
      });
    }
    if (url === '/web/api/answer/run-1/children') {
      roster.requests += 1;
      return response({run_id: 'run-1', children: [childRow('a'), childRow('b')], next_cursor: null, run_status: 'running', run_started_at: null, run_finished_at: null});
    }
    if (url.startsWith('/web/api/answer/run-1/transcript')) {
      return response({messages: [], next_before: null, running: false});
    }
    if (url.startsWith('/web/api/answer/run-1/children/')) {
      return response({run_id: 'run-1', child: childRow('a'), controls: [], questions: [], result: null});
    }
    return bootstrapResponse(input);
  };
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);
  await app.ready;
  await webRouter.navigate(conversationRoute('with-children'));
  await waitFor(() => tracesButton(app) !== undefined);
  return app;
}

function tracesButton(app: DlApp): HTMLButtonElement | undefined {
  return Array.from(app.querySelectorAll<HTMLButtonElement>('dl-chat-message-list button'))
    .find((button) => button.textContent?.trim() === 'Agent traces');
}

it('opens a settled turn\'s Child agents beside the chat, follows its Run, and leaves them open on a click in the chat', async () => {
  pageTimersRunFast();
  const roster = {requests: 0};
  const app = await appWithChildren(roster);
  const inspector = app.querySelector('dl-inspector')!;

  tracesButton(app)!.click();
  // The main agent and its two children.
  await waitFor(() => inspector.querySelectorAll('[data-agent-session]').length === 3);

  expect(inspector.kind).to.equal('traces');
  expect(inspector.querySelector('#panel-title')?.textContent).to.equal('Agent traces');
  expect(document.body.classList.contains('traces-panel-open')).to.equal(true);

  // The Run states a tool step, and the open dock fetches its roster again for it.
  app.querySelector('dl-chat-feature')!.dispatchEvent(new CustomEvent('dl-run-activity', {
    bubbles: true, composed: true,
    detail: {runId: 'run-1', events: [{kind: 'tool', eventType: 'tool_start', payload: {}}]},
  }));
  await waitFor(() => roster.requests === 2);

  // The dock is watched beside the chat, so a click in the chat does not take it away.
  app.querySelector('main[aria-label="Chat"]')?.dispatchEvent(new MouseEvent('click', {bubbles: true, composed: true}));
  await new Promise((resolve) => requestAnimationFrame(resolve));
  expect(inspector.open).to.equal(true);
  expect(inspector.kind).to.equal('traces');

  // The children belong to their conversation, which the page leaving takes them with.
  await webRouter.navigate(newChatRoute());
  await inspector.updateComplete;
  expect(inspector.open).to.equal(false);
  expect(document.body.classList.contains('traces-panel-open')).to.equal(false);
});

it('stops offering the thumbs in the chat and the Inspector once the server refuses a rating because feedback is off', async () => {
  window.matchMedia = desktopMedia;
  const settledChild = {...childRow('a'), status: 'succeeded', result_sequence: 12};
  window.fetch = async (input, init) => {
    const url = String(input);
    if (url === '/web/api/answer/run-1/feedback' && init?.method === 'PUT') {
      return response({detail: 'Answer feedback is turned off for you', error_type: 'auth'}, 403);
    }
    if (url === '/web/api/conversations/rated/history') {
      return response({
        conversation: {
          conversation_id: 'rated', title: 'Rated',
          created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z',
        },
        turns: [{...settledTurnWire, answer_sequence: 7, feedback: null}],
        commands: [],
        next_cursor: null,
      });
    }
    if (url === '/web/api/answer/run-1/children') {
      return response({run_id: 'run-1', children: [settledChild], next_cursor: null, run_status: 'running', run_started_at: null, run_finished_at: null});
    }
    if (url.startsWith('/web/api/answer/run-1/transcript')) {
      return response({messages: [], next_before: null, running: false});
    }
    if (url.startsWith('/web/api/answer/run-1/children/')) {
      return response({run_id: 'run-1', child: settledChild, controls: [], questions: [], result: null});
    }
    return bootstrapResponse(input);
  };
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);
  await app.ready;
  await webRouter.navigate(conversationRoute('rated'));
  await waitFor(() => tracesButton(app) !== undefined);
  const chat = app.querySelector<DlChatFeature>('dl-chat-feature')!;
  const inspector = app.querySelector('dl-inspector')!;
  tracesButton(app)!.click();
  await waitFor(() => inspector.querySelectorAll('[data-agent-session]').length === 2);
  inspector.querySelector<HTMLButtonElement>('[data-agent-session="a"]')!.click();
  const thumbs = (root: ParentNode) => root.querySelectorAll<HTMLButtonElement>('[data-thumb]');
  await waitFor(() => thumbs(inspector).length === 2);
  expect(thumbs(chat)).to.have.length(2);

  thumbs(inspector)[0]!.focus();
  thumbs(inspector)[0]!.click();
  await waitFor(() => thumbs(inspector).length === 0 && thumbs(chat).length === 0);
  const toast = app.querySelector<DlToastRegion>('dl-toast-region[tone="error"]')!;
  await toast.updateComplete;

  expect(toast.textContent).to.contain('Feedback is turned off for you.');
  // The thumb the reader was on went with the rest, and they are on the child's title and not on the page.
  expect(document.activeElement).to.equal(inspector.querySelector('dl-agent-session [data-title]'));
  expect(chat.answerFeedbackEnabled).to.equal(false);
  expect(inspector.answerFeedbackEnabled).to.equal(false);
});

it('closes Child agents on a click in the chat when the Inspector is a modal drawer over it', async () => {
  const app = await appWithChildren({requests: 0}, compactMedia);
  const inspector = app.querySelector('dl-inspector')!;
  tracesButton(app)!.click();
  await waitFor(() => inspector.kind === 'traces');
  expect(inspector.querySelector('aside')!.getAttribute('aria-modal')).to.equal('true');

  // There is no chat beside it to protect: the click is on the scrim, which closes it as it closes Files.
  app.querySelector('main[aria-label="Chat"]')?.dispatchEvent(new MouseEvent('click', {bubbles: true, composed: true}));
  await inspector.updateComplete;
  expect(inspector.open).to.equal(false);
  expect(document.body.classList.contains('traces-panel-open')).to.equal(false);
});

it('closes the Artifact Canvas when Child agents open, and a click in the chat closes the Canvas but not them', async () => {
  const app = await appWithChildren({requests: 0});
  const inspector = app.querySelector('dl-inspector')!;
  const canvas = app.querySelector('dl-artifact-canvas')!;
  const artifact: AnswerArtifact = {
    resourceId: 'chart-1', mediaType: 'image/png',
    label: 'Chart', filename: 'chart.png', byteSize: 20, digest: 'c'.repeat(64),
    presentation: 'image', status: 'available',
    uri: 'dlightrag://answer/run-1/artifacts/chart-1', width: 100, height: 100,
    dataUrl: '/web/api/answer/run-1/artifacts/chart-1',
    downloadUrl: '/web/api/answer/run-1/artifacts/chart-1?download=1',
    presentationUrl: null, issue: null,
    attestation: null,
  };
  await canvas.open(artifact, app.querySelector('#files-btn'));
  // A Canvas beside the chat leaves the chat's own controls reachable.
  buttonNamed(canvas, 'Side')!.click();
  await canvas.updateComplete;
  expect(canvas.classList.contains('open')).to.equal(true);

  tracesButton(app)!.click();
  await waitFor(() => inspector.kind === 'traces');
  expect(canvas.classList.contains('open')).to.equal(false);

  await canvas.open(artifact, app.querySelector('#files-btn'));
  app.querySelector('main[aria-label="Chat"]')?.dispatchEvent(new MouseEvent('click', {bubbles: true, composed: true}));
  expect(canvas.classList.contains('open')).to.equal(false);
  expect(inspector.kind).to.equal('traces');
});

it('opens Sources as the only compact modal when intent originates in Canvas', async () => {
  window.matchMedia = compactMedia;
  // The canvas parses this response through the api schema: serve wire shape.
  // The canvas parses this response through the api schema: serve wire shape.
  const presentationWire = {
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
  };;
  // What the element would hold after parsing; the dispatched event detail
  // carries the domain object, like the real component does.
  const presentation: AnswerPresentation = {
    answerText: 'See [1].',
    parts: [{
      type: 'markdown', text: 'See [1].',
      html: '<p>See <cite class="citation-badge" data-ref="1">1</cite>.</p>',
      artifact: null, evidenceImage: null, inline: false,
    }],
    sources: [{
      id: '1', title: 'Source', sourceUrl: null, downloadUrl: null, chunks: [],
    }],
    evidenceImages: [], linkCards: [], artifacts: [], artifactOutcome: {status: 'complete', issues: []},
  };;
  window.fetch = async (input) => String(input).includes('/presentation')
    ? response(presentationWire)
    : bootstrapResponse(input);
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);
  await app.ready;
  const returnFocus = app.querySelector<HTMLButtonElement>('#files-btn')!;
  const artifact: AnswerArtifact = {
    resourceId: 'report-1', mediaType: 'text/markdown',
    label: 'Report', filename: 'report.md', byteSize: 20, digest: 'a'.repeat(64),
    presentation: 'markdown', status: 'available',
    uri: 'dlightrag://answer/run-1/artifacts/report-1', width: null, height: null,
    dataUrl: '/web/api/answer/run-1/artifacts/report-1',
    downloadUrl: '/web/api/answer/run-1/artifacts/report-1?download=1',
    presentationUrl: '/web/api/answer/run-1/artifacts/report-1/presentation', issue: null,
    attestation: null,
  };
  const canvas = app.querySelector('dl-artifact-canvas')!;
  await canvas.open(artifact, returnFocus);
  await canvas.updateComplete;
  await canvas.querySelector('dl-answer-presentation')?.updateComplete;

  const canvasPresentation = canvas.querySelector<AnswerPresentationElement>(
    'dl-answer-presentation',
  )!;
  canvasPresentation.dispatchEvent(new CustomEvent<AnswerSourceOpenDetail>('dl-answer-source-open', {
    bubbles: true,
    composed: true,
    detail: {presentation, referenceId: '1', returnFocus: canvasPresentation},
  }));
  await waitFor(() => app.querySelector('#panel')?.classList.contains('open') === true);
  await new Promise((resolve) => requestAnimationFrame(resolve));

  expect(canvas.classList.contains('open')).to.equal(false);
  expect(canvas.inert).to.equal(true);
  expect(app.querySelector<HTMLElement>('#panel')?.inert).to.equal(false);
  expect(app.querySelector('#panel')?.getAttribute('aria-modal')).to.equal('true');
  expect(app.querySelector('.topbar')?.hasAttribute('inert')).to.equal(true);
  expect(document.activeElement).to.equal(app.querySelector('#panel-close-btn'));

  app.querySelector<HTMLButtonElement>('#panel-close-btn')?.click();
  await new Promise((resolve) => requestAnimationFrame(resolve));
  expect(document.activeElement).to.equal(returnFocus);
});

it('preserves a wide desktop Canvas when Sources opens and restores its citation', async () => {
  window.matchMedia = desktopMedia;
  // The canvas parses this response through the api schema: serve wire shape.
  const presentationWire = {
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
  };;
  // What the element would hold after parsing; the dispatched event detail
  // carries the domain object, like the real component does.
  const presentation: AnswerPresentation = {
    answerText: 'See [1].',
    parts: [{
      type: 'markdown', text: 'See [1].',
      html: '<p>See <cite class="citation-badge" data-ref="1" tabindex="0">1</cite>.</p>',
      artifact: null, evidenceImage: null, inline: false,
    }],
    sources: [{
      id: '1', title: 'Source', sourceUrl: null, downloadUrl: null, chunks: [],
    }],
    evidenceImages: [], linkCards: [], artifacts: [], artifactOutcome: {status: 'complete', issues: []},
  };;
  window.fetch = async (input) => String(input).includes('/presentation')
    ? response(presentationWire)
    : bootstrapResponse(input);
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);
  await app.ready;
  const artifact: AnswerArtifact = {
    resourceId: 'report-desktop', mediaType: 'text/markdown',
    label: 'Report', filename: 'report.md', byteSize: 20, digest: 'b'.repeat(64),
    presentation: 'markdown', status: 'available',
    uri: 'dlightrag://answer/run-1/artifacts/report-desktop', width: null, height: null,
    dataUrl: '/web/api/answer/run-1/artifacts/report-desktop',
    downloadUrl: '/web/api/answer/run-1/artifacts/report-desktop?download=1',
    presentationUrl: '/web/api/answer/run-1/artifacts/report-desktop/presentation', issue: null,
    attestation: null,
  };
  const canvas = app.querySelector('dl-artifact-canvas')!;
  await canvas.open(artifact, app.querySelector('#files-btn'));
  await canvas.updateComplete;
  await canvas.querySelector('dl-answer-presentation')?.updateComplete;
  const canvasPresentation = canvas.querySelector<AnswerPresentationElement>(
    'dl-answer-presentation',
  )!;
  const wideButton = Array.from(canvas.querySelectorAll<HTMLButtonElement>(
    '.artifact-canvas-layout-actions button',
  )).find((button) => button.textContent?.trim() === 'Wide');
  wideButton?.click();
  await canvas.updateComplete;
  canvasPresentation.tabIndex = 0;
  canvasPresentation.focus();
  canvasPresentation.dispatchEvent(new CustomEvent<AnswerSourceOpenDetail>('dl-answer-source-open', {
    bubbles: true,
    composed: true,
    detail: {presentation, referenceId: '1', returnFocus: canvasPresentation},
  }));
  await waitFor(() => app.querySelector('#panel')?.classList.contains('open') === true);
  expect(canvas.classList.contains('open')).to.equal(true);
  expect(canvas.layout).to.equal('wide');

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
    resourceId: 'chart-1', mediaType: 'image/png',
    label: 'Chart', filename: 'chart.png', byteSize: 20, digest: 'c'.repeat(64),
    presentation: 'image', status: 'available',
    uri: 'dlightrag://answer/run-1/artifacts/chart-1', width: 100, height: 100,
    dataUrl: '/web/api/answer/run-1/artifacts/chart-1',
    downloadUrl: '/web/api/answer/run-1/artifacts/chart-1?download=1',
    presentationUrl: null, issue: null,
    attestation: null,
  };
  await canvas.open(artifact, app.querySelector('#files-btn'));
  await canvas.updateComplete;
  expect(canvas.classList.contains('open')).to.equal(true);
  expect(canvas.layout).to.equal('wide');
  expect(app.querySelector('dl-inspector')?.open).to.equal(false);
  const chat = app.querySelector<DlChatFeature>('dl-chat-feature')!;
  expect(chat.inert).to.equal(true);
  expect(chat.querySelector<HTMLElement>('#chat-messages')?.inert).to.equal(false);
  expect(chat.querySelector<HTMLElement>('dl-chat-composer')?.inert).to.equal(false);

  app.querySelector('main[aria-label="Chat"]')?.dispatchEvent(new MouseEvent('click', {
    bubbles: true,
    composed: true,
  }));

  expect(canvas.classList.contains('open')).to.equal(false);
  expect(document.body.classList.contains('artifact-canvas-open')).to.equal(false);
  await app.updateComplete;
  await chat.updateComplete;
  await chat.querySelector('dl-chat-message-list')?.updateComplete;
  await chat.querySelector('dl-chat-composer')?.updateComplete;
  expect(chat.querySelector<HTMLElement>('#chat-messages')?.inert).to.equal(false);
  expect(chat.querySelector<HTMLElement>('dl-chat-composer')?.inert).to.equal(false);

  await canvas.open(artifact, app.querySelector('#files-btn'));
  canvas.prepareForInspector();
  const inspectorReturn = app.querySelector<HTMLButtonElement>('#theme-trigger')!;
  const inspector = app.querySelector('dl-inspector')!;
  await inspector.openSources({
    answerText: '', parts: [], sources: [], evidenceImages: [], linkCards: [], artifacts: [],
    artifactOutcome: {status: 'complete', issues: []},
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

it('wakes the Run activity store when a submission starts running, and only then', async () => {
  let activityReads = 0;
  window.fetch = async (input) => {
    const url = String(input);
    if (url.startsWith('/web/api/run-activity')) {
      activityReads += 1;
      return response({items: []});
    }
    return bootstrapResponse(input);
  };
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);
  await app.ready;
  // The shell reads the activity once on opening, then sleeps on an idle page. A wake while that
  // read is still in flight would share it instead of making another, so the test waits for the
  // read to finish, not for a frame to pass: how long a read takes is the machine's.
  await waitFor(() => activityReads === 1);
  await waitFor(() => app.handles.runActivity.loaded && !app.handles.runActivity.watching);
  const settle = (): Promise<void> => new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
  await settle();
  expect(activityReads, 'an idle page asks for nothing more').to.equal(1);

  const chat = app.querySelector('dl-chat-feature')!;
  chat.dispatchEvent(new CustomEvent('dl-chat-running-change', {
    bubbles: true, composed: true, detail: {active: true},
  }));
  await waitFor(() => activityReads === 2);
  expect(app.chatRunning, 'a running submission is the page\'s reason to look').to.equal(true);

  chat.dispatchEvent(new CustomEvent('dl-chat-running-change', {
    bubbles: true, composed: true, detail: {active: false},
  }));
  await settle();
  expect(activityReads, 'the Run settling gets no wake of its own').to.equal(2);
});

it('fails closed and resolves the same ready promise after an explicit retry', async () => {
  let attempts = 0;
  window.fetch = async (input) => {
    const url = String(input);
    if (url === '/web/api/conversations') {
      return response({items: [], next_cursor: null});
    }
    // The shell also reads the reader's Run activity once it is ready; that read is not a bootstrap.
    if (url.startsWith('/web/api/run-activity')) return response({items: []});
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

it('brings an unsent message back when the shell is loaded again', async () => {
  // The page a provider returns to is a fresh load, and the reader's half-typed question is
  // still theirs: the composer keeps it outside the page's memory.
  localStorage.setItem('dlightrag.composerDraft', 'Typed before the provider sent me back');
  window.fetch = async (input) => bootstrapResponse(input);
  try {
    const app = document.createElement('dl-app') as DlApp;
    document.body.append(app);
    await app.ready;
    await waitFor(() => Boolean(app.querySelector('.composer-input')));

    expect(app.querySelector<HTMLTextAreaElement>('.composer-input')!.value)
      .to.equal('Typed before the provider sent me back');
  } finally {
    localStorage.removeItem('dlightrag.composerDraft');
  }
});

it('answers an expired sign-in with the way back to this page, not a load failure', async () => {
  const previous = window.location.href;
  window.history.replaceState(null, '', '?view=files#latest');
  window.fetch = async () => response({detail: 'Token expired', error_type: 'auth'}, 401);
  try {
    const app = document.createElement('dl-app') as DlApp;
    document.body.appendChild(app);
    await waitFor(() => app.signedOut);
    await app.updateComplete;

    const status = app.querySelector<HTMLElement>('.bootstrap-status');
    expect(status?.getAttribute('role')).to.equal('alert');
    expect(status?.textContent).to.contain('Your session has ended. Sign in again to continue.');
    expect(status?.textContent).not.to.contain('could not load');
    const signIn = status?.querySelector<HTMLAnchorElement>('a');
    expect(signIn?.textContent?.trim()).to.equal('Sign in');
    const target = new URL(signIn!.href);
    expect(target.pathname).to.equal('/web/login');
    expect(target.searchParams.get('next')).to.equal(`${window.location.pathname}?view=files#latest`);
    expect(app.querySelector('#app')).to.equal(null);
  } finally {
    window.history.replaceState(null, '', previous);
  }
});

it('signs out the whole page when a Feature request meets a 401, with no failure of its own', async () => {
  let refused = false;
  window.fetch = async (input, init) => {
    if (String(input) === '/web/api/answer' && init?.method === 'POST') {
      refused = true;
      return response({detail: 'Token expired', error_type: 'auth'}, 401);
    }
    return bootstrapResponse(input);
  };
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);
  await app.ready;
  try {
    const input = app.querySelector<HTMLTextAreaElement>('[aria-label="Message"]')!;
    input.value = 'Question after the sign-in expired';
    input.dispatchEvent(new Event('input', {bubbles: true}));
    await app.querySelector('dl-chat-composer')?.updateComplete;
    app.querySelector<HTMLButtonElement>('[aria-label="Send"]')?.click();
    await waitFor(() => app.signedOut);
    await app.updateComplete;

    expect(refused).to.equal(true);
    expect(app.querySelector('.bootstrap-status')?.textContent)
      .to.contain('Your session has ended. Sign in again to continue.');
    // The Features are gone, so the refused submission reports nothing of its own.
    expect(app.querySelector('dl-chat-feature')).to.equal(null);
    expect(app.querySelector('.submission-failure')).to.equal(null);
    expect(app.textContent).not.to.contain('could not be submitted');
    // The sign-out took the submission the refusal belonged to, so none waits for the next reader.
    expect(productionHandles().submissions.list()).to.have.length(0);
  } finally {
    productionHandles().submissions.dispose();
  }
});

it('keeps the OAuth return marker through the router a page load builds before the shell reads it', async () => {
  // A real return is a fresh page load: ui/router.ts builds the one browser router while
  // ./app.ts is still being imported, so its canonicalization runs before the shell reads the
  // address it landed on. Reproduce that order, not the address alone.
  const previous = window.location.href;
  window.history.replaceState(null, '', '/web/?settings=connections&authorization=restart');
  createBrowserRouter(window);
  window.fetch = connectionsReturnFetch();
  try {
    const app = document.createElement('dl-app') as DlApp;
    document.body.append(app);
    await app.ready;
    await waitFor(() => Boolean(app.querySelector('dl-settings-connections')));
    const settings = app.querySelector<DlSettingsDialog>('dl-settings-dialog')!;
    await waitFor(() => settings.querySelector<HTMLDialogElement>('#settings-dialog')?.open === true);

    const current = [...settings.querySelectorAll<HTMLElement>('nav [aria-current="page"]')];
    expect(current.map((item) => item.dataset.section)).to.deep.equal(['connections']);
    expect(settings.querySelector<HTMLElement>('dl-settings-connections')!.hidden).to.equal(false);
  } finally {window.history.replaceState(null, '', previous);}
});

it('opens Settings on its Connections page after the fixed OAuth return, without starting authorization', async () => {
  const previous = window.location.href;
  const writes: string[] = [];
  window.history.replaceState(null, '', '?settings=connections&authorization=restart');
  window.fetch = connectionsReturnFetch(writes);
  try {
    const app = document.createElement('dl-app') as DlApp;
    document.body.append(app);
    await app.ready;
    await waitFor(() => Boolean(app.querySelector('dl-settings-connections')));
    const settings = app.querySelector<DlSettingsDialog>('dl-settings-dialog')!;
    await waitFor(() => settings.querySelector<HTMLDialogElement>('#settings-dialog')?.open === true);

    // The return path lands on the Connections page itself; a failed grant reports itself inside
    // the affected card, not as a dialog-level banner, and nothing is written.
    const current = [...settings.querySelectorAll<HTMLElement>('nav [aria-current="page"]')];
    expect(current.map((item) => item.dataset.section)).to.deep.equal(['connections']);
    expect(settings.querySelector<HTMLElement>('dl-settings-connections')!.hidden).to.equal(false);
    expect(settings.querySelector('[role="region"]')?.textContent).to.contain('External MCP servers');
    expect(writes).to.deep.equal([]);
    expect(window.location.search).not.to.contain('settings=');
    expect(window.location.search).not.to.contain('authorization=');
  } finally {window.history.replaceState(null, '', previous);}
});

async function appCountingChildReads(
  reads: {children: number; transcript: number; child: number},
  runId = 'run-1',
): Promise<DlApp> {
  window.matchMedia = desktopMedia;
  window.fetch = async (input) => {
    const url = String(input);
    if (url === '/web/api/conversations/with-children/history') {
      return response({
        conversation: {conversation_id: 'with-children', title: 'Children',
          created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z'},
        turns: [{...settledTurnWire, answer_run_id: runId}], commands: [], next_cursor: null,
      });
    }
    if (url === `/web/api/answer/${runId}/children`) {
      reads.children += 1;
      return response({run_id: runId, children: [childRow('a'), childRow('b')], next_cursor: null, run_status: 'running', run_started_at: null, run_finished_at: null});
    }
    if (url.startsWith(`/web/api/answer/${runId}/transcript`)) {
      reads.transcript += 1;
      return response({messages: [], next_before: null, running: false});
    }
    if (url.startsWith(`/web/api/answer/${runId}/children/`)) {
      reads.child += 1;
      // A child that has settled: its page cannot change, which is what makes keeping it safe.
      return response({
        run_id: runId,
        child: {...childRow('a'), status: 'succeeded', operation_status: 'succeeded'},
        controls: [], questions: [], result: null,
      });
    }
    return bootstrapResponse(input);
  };
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);
  await app.ready;
  await webRouter.navigate(conversationRoute('with-children'));
  await waitFor(() => tracesButton(app) !== undefined);
  return app;
}

/** A TracesSource shaped like the one the shell hands its dock, counting the child reads. */
function appCountingSource(reads: {child: number}): TracesSource {
  return {
    runId: 'run-3',
    mainAgent: () => ({childSessionId: 'main'}),
    presentation: () => null,
    page: async () => ({children: [], next_cursor: null}),
    observe: async () => {
      reads.child += 1;
      return {
        runId: 'run-3',
        child: {...childRow('a'), status: 'succeeded', operation_status: 'succeeded'},
        controls: [], questions: [], result: null,
      } as unknown as ChildObservation;
    },
    activity: async () => ({messages: [], next_before: null, running: false}),
    control: async () => ({}) as never,
    reply: async () => ({}) as never,
  } as unknown as TracesSource;
}

function childRows(app: DlApp): HTMLButtonElement[] {
  return [...app.querySelectorAll<HTMLButtonElement>(
    'dl-inspector-traces [data-agent-session]:not([data-agent-session="main"])',
  )];
}

it('reading a child once: walking to the next and back does not read the first again', async () => {
  pageTimersRunFast();
  const reads = {children: 0, transcript: 0, child: 0};
  const app = await appCountingChildReads(reads);
  tracesButton(app)!.click();
  await waitFor(() => childRows(app).length === 2);

  childRows(app)[0]!.click();
  await waitFor(() => reads.child === 1);
  childRows(app)[1]!.click();
  await waitFor(() => reads.child === 2);
  const withBothRead = reads.child;

  childRows(app)[0]!.click();
  await new Promise((resolve) => setTimeout(resolve, 40));

  expect(reads.child, 'the child already read was read again').to.equal(withBothRead);
});

it('warming a child: the pointer on its row reads it before any click', async () => {
  pageTimersRunFast();
  const reads = {children: 0, transcript: 0, child: 0};
  const app = await appCountingChildReads(reads, 'run-2');
  tracesButton(app)!.click();
  await waitFor(() => childRows(app).length === 2);

  childRows(app)[1]!.dispatchEvent(new MouseEvent('mouseenter'));
  await waitFor(() => reads.child === 1);

  childRows(app)[1]!.click();
  await new Promise((resolve) => setTimeout(resolve, 40));

  expect(reads.child, 'the click read a child the warm already read').to.equal(1);
});

it('signing out leaves no kept page behind for the next reader of this page', async () => {
  pageTimersRunFast();
  const reads = {children: 0, transcript: 0, child: 0};
  const app = await appCountingChildReads(reads, 'run-3');
  tracesButton(app)!.click();
  await waitFor(() => childRows(app).length === 2);

  childRows(app)[0]!.click();
  await waitFor(() => reads.child === 1);

  // Asking for that Run's source must hand back the one the dock already read through.
  const built = {forKept: false, afterSignOut: false};
  const runSource = (probe: keyof typeof built) => {
    const source = appCountingSource(reads);
    built[probe] = true;
    return source;
  };

  const { judgements, traces } = productionHandles();
  await traces.sourceFor('run-3', () => runSource('forKept')).observe('a', new AbortController().signal);
  expect(built.forKept, 'the Run had to be built although its dock had already read').to.equal(false);
  expect(reads.child, 'the dock read the child more than once').to.equal(1);

  // A judgement the reader gave is the page's to remember, and not the next reader's.
  const rated = judgementKey('run-3', 'main', 7);
  judgements.keep(rated, {value: 'up', reasons: [], note: null});

  noteRefusal(401); // the one refusal every route funnels through
  await waitFor(() => app.signedOut === true);

  await traces.sourceFor('run-3', () => runSource('afterSignOut')).observe('a', new AbortController().signal);
  expect(built.afterSignOut, 'the Run still held the page its last reader read').to.equal(true);
  expect(judgements.current(rated, null), 'the judgement outlived the reader who gave it').to.equal(null);
});
