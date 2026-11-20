// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {defineDesignSystemElements} from '../design-system/index.ts';
import {conversationRoute, newChatRoute, type WebRoute} from '../lib/router.ts';
import {createAppHandles, productionHandles} from '../stores/app-handles.ts';
import {bootstrapResponse} from '../testing/app.ts';
import {
  followConversationView,
  type FollowedChat,
  stopFollowingConversationViews,
} from '../testing/conversation-view.ts';
import {waitFor} from '../testing/dom.ts';
import type {DlApp} from './app.ts';
import './app.ts';
import type {ChatView} from './chat-message-list.ts';
import type {DlChatFeature} from './chat-feature.ts';
import type {DlConversationSidebar} from './conversation-sidebar.ts';
import {webRouter} from './router.ts';

defineDesignSystemElements();

const originalFetch = window.fetch;

function wire(id: string, title: string | null = id): Record<string, unknown> {
  return {conversation_id: id, title, created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-02T00:00:00Z'};
}

/** The server holding conversations `a` and `b`; a history it is asked for twice is read twice. */
function serveConversations(reads: string[] = []): void {
  window.fetch = async (input) => {
    const url = String(input);
    reads.push(url);
    if (url === '/web/api/conversations') {
      return Response.json({items: [wire('a'), wire('b')], next_cursor: null});
    }
    const history = /^\/web\/api\/conversations\/([^/]+)\/history$/.exec(url);
    if (history && ['a', 'b'].includes(history[1]!)) {
      return Response.json({conversation: wire(history[1]!), turns: [], commands: [], next_cursor: null});
    }
    return Response.json({detail: 'not found'}, {status: 404});
  };
}

type StubChat = {-readonly [K in keyof FollowedChat]: FollowedChat[K]} & {detached: number; cleared: number};

/** A Chat as the route meets it: it keeps the view it is told, and counts what the route asks of it. */
function chatStub(init: Partial<StubChat> = {}): StubChat {
  const chat: StubChat = {
    view: {kind: 'new'} as ChatView,
    hasDraft: false,
    hasStagedAttachments: false,
    hasUnresolvedSubmission: false,
    submissionPending: false,
    detached: 0,
    cleared: 0,
    detachRun: () => { chat.detached += 1; },
    clearDraft: () => { chat.cleared += 1; },
    ...init,
  };
  return chat;
}

afterEach(async () => {
  stopFollowingConversationViews();
  document.body.replaceChildren();
  productionHandles().conversations.openNew();
  window.fetch = originalFetch;
  await webRouter.navigate(newChatRoute(), {replace: true, notify: false, bypassGuard: true});
});

describe('following the address', () => {
  it('reads the list, opens what the address names, and answers each move with the view of it', async () => {
    serveConversations();
    await webRouter.navigate(conversationRoute('a'), {replace: true, notify: false, bypassGuard: true});
    const chat = chatStub();
    const routes = followConversationView(chat);

    routes.start();
    await waitFor(() => chat.view.kind === 'ready');
    expect(chat.view).to.include({kind: 'ready', conversationId: 'a'});

    expect(await webRouter.navigate(conversationRoute('b'))).to.equal(true);
    await waitFor(() => chat.view.kind === 'ready' && chat.view.conversationId === 'b');

    expect(await webRouter.navigate(newChatRoute())).to.equal(true);
    await waitFor(() => chat.view.kind === 'new');
  });

  it('starts once however often it is asked, and reads the list once', async () => {
    const reads: string[] = [];
    serveConversations(reads);
    const routes = followConversationView(chatStub());

    routes.start();
    routes.start();
    await waitFor(() => reads.length > 0);
    await new Promise((resolve) => setTimeout(resolve, 20));

    expect(reads.filter((url) => url === '/web/api/conversations')).to.have.length(1);
  });

  it('says the address names no page until the store publishes another view, and then says what it holds', async () => {
    serveConversations();
    const chat = chatStub();
    const routes = followConversationView(chat);
    routes.start();
    await waitFor(() => chat.view.kind === 'new');

    expect(await webRouter.navigate({kind: 'not-found'})).to.equal(true);
    await waitFor(() => chat.view.kind === 'unavailable');
    expect(chat.view).to.deep.equal({kind: 'unavailable', hasRecent: false});

    expect(await webRouter.navigate(conversationRoute('a'))).to.equal(true);
    await waitFor(() => chat.view.kind === 'ready');
    expect(chat.view).to.include({kind: 'ready', conversationId: 'a'});
  });

  it('hands the Chat one view for each thing the store published, and the same view while it holds still', async () => {
    serveConversations();
    const chat = chatStub();
    const routes = followConversationView(chat);
    routes.start();
    await waitFor(() => chat.view.kind === 'new');
    await webRouter.navigate(conversationRoute('a'));
    await waitFor(() => chat.view.kind === 'ready');

    const ready = routes.view;
    expect(routes.view, 'nothing was published, so nothing is new').to.equal(ready);
    // A change to the list is not a change to the view: the Chat's input is not set again for it.
    productionHandles().conversations.upsertSummary({
      conversationId: 'z', title: 'Z', createdAt: '2026-01-01T00:00:00Z', updatedAt: '2026-01-03T00:00:00Z',
      forkedFromConversationId: null, forkedFromTitle: null,
    });
    expect(routes.view).to.equal(ready);

    await productionHandles().conversations.refreshActive();
    expect(routes.view, 'the view was published again').not.to.equal(ready);
    expect(routes.view).to.include({kind: 'ready', conversationId: 'a'});
  });

  it('hands the Chat a view of its own each time the store publishes one, a new chat over a new chat included', async () => {
    serveConversations();
    const chat = chatStub();
    const routes = followConversationView(chat);
    routes.start();
    await waitFor(() => chat.view.kind === 'new');
    const conversations = productionHandles().conversations;
    const first = chat.view;

    // A question sent from a new chat makes the conversation without a view of its own: its answer is
    // on show already. A new chat opened over that is a thing that happened, though it reads alike.
    conversations.adoptCreatedConversation({
      conversationId: 'made', title: 'Made', createdAt: '2026-01-01T00:00:00Z', updatedAt: '2026-01-01T00:00:00Z',
      forkedFromConversationId: null, forkedFromTitle: null,
    });
    expect(chat.view, 'adopting a conversation publishes no view').to.equal(first);
    conversations.openNew();
    expect(chat.view).to.deep.equal({kind: 'new'});
    expect(chat.view, 'the new chat is not the one the Chat was told before').not.to.equal(first);

    // An address that names no page is told the same way, however often it is asked for.
    expect(await webRouter.navigate({kind: 'not-found'})).to.equal(true);
    await waitFor(() => chat.view.kind === 'unavailable');
    const missing = chat.view;
    expect(await webRouter.navigate({kind: 'not-found'})).to.equal(true);
    expect(chat.view).to.deep.equal(missing);
    expect(chat.view, 'asked for again, it is another view').not.to.equal(missing);
  });

  it('tells the Chat what is open in the bag it was given, not in the page\'s', async () => {
    serveConversations();
    const bag = createAppHandles();
    const chat = chatStub();
    const routes = followConversationView(chat, {handles: bag});

    routes.start();
    await webRouter.navigate(conversationRoute('b'));
    await waitFor(() => chat.view.kind === 'ready');

    expect(bag.conversations.activeConversationId).to.equal('b');
    expect(productionHandles().conversations.activeConversationId, 'the page\'s own bag was opened').to.equal(null);
    expect(chat.view).to.include({conversationId: 'b'});
  });

  it('lets go of the address when it stops: no move is answered, and no draft holds one back', async () => {
    serveConversations();
    const chat = chatStub({hasDraft: true});
    let asked = 0;
    const routes = followConversationView(chat, {ui: {confirmDiscardDraft: async () => { asked += 1; return false; }}});
    routes.start();
    await waitFor(() => chat.view.kind === 'new');

    routes.stop();

    expect(await webRouter.navigate(conversationRoute('a'))).to.equal(true);
    expect(asked, 'a guard that was let go still asked').to.equal(0);
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(productionHandles().conversations.activeConversationId, 'a move was answered').to.equal(null);
  });
});

describe('leaving a conversation', () => {
  it('lets go of the Run being followed, and says what belonged to the conversation is to go, once for each move', async () => {
    serveConversations();
    const chat = chatStub();
    let left = 0;
    const routes = followConversationView(chat, {ui: {conversationLeft: () => { left += 1; }}});
    routes.start();
    await waitFor(() => chat.view.kind === 'new');
    expect({detached: chat.detached, left}, 'the first answer left nothing').to.deep.equal({detached: 0, left: 0});

    await webRouter.navigate(conversationRoute('a'));
    await waitFor(() => chat.view.kind === 'ready');
    expect({detached: chat.detached, left}).to.deep.equal({detached: 1, left: 1});

    // The same conversation again is no move.
    await productionHandles().conversations.open('a');
    expect({detached: chat.detached, left}).to.deep.equal({detached: 1, left: 1});

    await webRouter.navigate(newChatRoute());
    await waitFor(() => chat.view.kind === 'new');
    expect({detached: chat.detached, left}).to.deep.equal({detached: 2, left: 2});
  });

  it('puts the compact drawer away once a move is answered, and not for the first answer', async () => {
    serveConversations();
    const chat = chatStub();
    let closed = 0;
    const routes = followConversationView(chat, {ui: {closeDrawer: async () => { closed += 1; }}});
    routes.start();
    await waitFor(() => chat.view.kind === 'new');
    expect(closed, 'the first answer opened no drawer').to.equal(0);

    await webRouter.navigate(conversationRoute('a'));

    expect(closed).to.equal(1);
  });
});

describe('standing guard', () => {
  it('lets a move go that loses nothing', async () => {
    serveConversations();
    const chat = chatStub();
    const routes = followConversationView(chat);
    routes.start();
    await waitFor(() => chat.view.kind === 'new');

    expect(await webRouter.navigate(conversationRoute('a'))).to.equal(true);
  });

  it('holds a move while a question is still being accepted, and says why', async () => {
    serveConversations();
    const chat = chatStub({submissionPending: true});
    let said = 0;
    const routes = followConversationView(chat, {ui: {waitForAcceptedQuestion: () => { said += 1; }}});
    routes.start();
    await waitFor(() => chat.view.kind === 'new');

    expect(await webRouter.navigate(conversationRoute('a'))).to.equal(false);
    expect(said).to.equal(1);
    expect(webRouter.current.kind).to.equal('new');

    chat.submissionPending = false;
    expect(await webRouter.navigate(conversationRoute('a'))).to.equal(true);
    expect(said, 'a move that went through said nothing').to.equal(1);
  });

  it('holds a move while a deletion is under way, without a word of its own', async () => {
    serveConversations();
    const chat = chatStub();
    let said = 0;
    const routes = followConversationView(chat, {ui: {waitForAcceptedQuestion: () => { said += 1; }}});
    routes.start();
    await waitFor(() => chat.view.kind === 'new');

    routes.lifecycleInFlight = true;
    expect(await webRouter.navigate(conversationRoute('a'))).to.equal(false);
    routes.lifecycleInFlight = false;
    expect(await webRouter.navigate(conversationRoute('a'))).to.equal(true);
    expect(said).to.equal(0);
  });

  it('asks the reader before an unsent message goes, forgets it when they agree, and keeps it when they do not', async () => {
    serveConversations();
    const chat = chatStub({hasDraft: true});
    const asked: WebRoute[] = [];
    let answer = false;
    const routes = followConversationView(chat, {
      ui: {confirmDiscardDraft: async (next) => { asked.push(next); return answer; }},
    });
    routes.start();
    await waitFor(() => chat.view.kind === 'new');

    expect(await webRouter.navigate(conversationRoute('a'))).to.equal(false);
    expect(asked).to.deep.equal([conversationRoute('a')]);
    expect(chat.cleared, 'the reader did not agree, and the message stayed').to.equal(0);
    expect(webRouter.current.kind).to.equal('new');

    answer = true;
    expect(await webRouter.navigate(conversationRoute('a'))).to.equal(true);
    expect(chat.cleared, 'the reader agreed, and the message went with the move').to.equal(1);
    await waitFor(() => chat.view.kind === 'ready');
  });

  it('does not ask again for a move that stays on the same page', async () => {
    serveConversations();
    const chat = chatStub({hasDraft: true});
    let asked = 0;
    const routes = followConversationView(chat, {ui: {confirmDiscardDraft: async () => { asked += 1; return false; }}});
    routes.start();
    await waitFor(() => chat.view.kind === 'new');

    expect(await webRouter.navigate(newChatRoute())).to.equal(true);
    expect(asked).to.equal(0);
  });
});

describe('the Chat\'s own buttons', () => {
  it('opens a new chat or the most recent conversation, and reads again what could not be shown', async () => {
    const reads: string[] = [];
    serveConversations(reads);
    const chat = chatStub();
    const routes = followConversationView(chat);
    routes.start();
    await waitFor(() => productionHandles().conversations.fallbackConversationId !== null);

    routes.chatViewAction('recent');
    await waitFor(() => chat.view.kind === 'ready');
    expect(chat.view).to.include({conversationId: productionHandles().conversations.fallbackConversationId});

    routes.chatViewAction('new');
    await waitFor(() => chat.view.kind === 'new');

    // With a conversation open, retrying reads it again; with none, it reads the list.
    await webRouter.navigate(conversationRoute('a'));
    await waitFor(() => chat.view.kind === 'ready');
    const histories = (): number => reads.filter((url) => url.endsWith('/a/history')).length;
    const before = histories();
    routes.chatViewAction('retry');
    await waitFor(() => histories() === before + 1);

    await webRouter.navigate(newChatRoute());
    await waitFor(() => chat.view.kind === 'new');
    const lists = (): number => reads.filter((url) => url === '/web/api/conversations').length;
    const listed = lists();
    routes.chatViewAction('retry');
    await waitFor(() => lists() === listed + 1);
  });

  it('reads the list and opens the address again on reload', async () => {
    const reads: string[] = [];
    serveConversations(reads);
    const chat = chatStub();
    const routes = followConversationView(chat);
    routes.start();
    await waitFor(() => chat.view.kind === 'new');
    const listed = reads.filter((url) => url === '/web/api/conversations').length;

    await routes.reload();

    expect(reads.filter((url) => url === '/web/api/conversations')).to.have.length(listed + 1);
    expect(chat.view.kind).to.equal('new');
  });
});

describe('leaving the page', () => {
  it('asks the reader about unresolved failed submissions, and does not hold in-app moves back', async () => {
    serveConversations();
    const chat = chatStub({hasUnresolvedSubmission: true});
    const routes = followConversationView(chat);
    routes.start();
    await waitFor(() => chat.view.kind === 'new');

    const leaving = new Event('beforeunload', {cancelable: true}) as BeforeUnloadEvent;
    window.dispatchEvent(leaving);
    expect(leaving.defaultPrevented).to.equal(true);
    expect(await webRouter.navigate(conversationRoute('a'))).to.equal(true);
  });

  it('asks about staged files, which a reload loses, and not about a typed draft, which it keeps', () => {
    const chat = chatStub({hasDraft: true, hasStagedAttachments: false});
    followConversationView(chat);

    const typed = new Event('beforeunload', {cancelable: true}) as BeforeUnloadEvent;
    window.dispatchEvent(typed);
    // The message is stored, so leaving with it is not leaving it behind.
    expect(typed.defaultPrevented).to.equal(false);

    chat.hasStagedAttachments = true;
    const staged = new Event('beforeunload', {cancelable: true}) as BeforeUnloadEvent;
    window.dispatchEvent(staged);
    expect(staged.defaultPrevented).to.equal(true);
  });

  it('stops asking once it is disconnected', () => {
    const chat = chatStub({hasUnresolvedSubmission: true});
    const routes = followConversationView(chat);
    routes.hostDisconnected();

    const leaving = new Event('beforeunload', {cancelable: true}) as BeforeUnloadEvent;
    window.dispatchEvent(leaving);
    expect(leaving.defaultPrevented).to.equal(false);
  });
});

/** The page, with an answer started from a new chat and still streaming: the conversation it made is on
 *  show, and the store holds it without a view of its own. */
async function showStreamingAnswer(): Promise<{app: DlApp; chat: DlChatFeature}> {
  const summary = {conversation_id: 'conv-x', title: 'X', created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-02T00:00:00Z'};
  const turn = {
    turn_id: 'turn-x', turn_number: 1, answer_run_id: 'run-x', submission_id: 'sub-x', status: 'running',
    cancel_requested: false, user_text: 'First question', assistant_text: '', user_attachments: [],
    steering_messages: [], presentation: null, usage: {}, error_kind: null, error_message: null,
    created_at: '2026-01-01T00:00:00Z',
  };
  window.fetch = async (input, init) => {
    const url = String(input);
    if (url === '/web/api/answer' && init?.method === 'POST') {
      return Response.json({kind: 'accepted', conversation: summary, turn});
    }
    if (url.startsWith('/web/api/runs/run-x/events')) {
      // The answer goes on: its stream never ends.
      return new Response(new ReadableStream({start() {}}), {headers: {'Content-Type': 'text/event-stream'}});
    }
    if (init?.method === 'DELETE') return new Response(null, {status: 204});
    return bootstrapResponse(input);
  };
  await webRouter.navigate(newChatRoute(), {replace: true, notify: false, bypassGuard: true});
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);
  await app.ready;
  const chat = app.querySelector<DlChatFeature>('dl-chat-feature')!;
  await waitFor(() => productionHandles().conversations.viewRevision > 0);

  const input = app.querySelector<HTMLTextAreaElement>('[aria-label="Message"]')!;
  input.value = 'First question';
  input.dispatchEvent(new Event('input', {bubbles: true}));
  await app.querySelector('dl-chat-composer')?.updateComplete;
  app.querySelector<HTMLButtonElement>('[aria-label="Send"]')!.click();
  await waitFor(() => webRouter.current.kind === 'conversation');
  await waitFor(() => chat.turns.some((shown) => shown.runId === 'run-x'));
  return {app, chat};
}

describe('a conversation the page leaves while its answer is still streaming', () => {
  it('shows an empty chat once a new chat is opened', async () => {
    const {app, chat} = await showStreamingAnswer();

    app.querySelector<HTMLButtonElement>('#new-conversation-btn')!.click();
    await waitFor(() => webRouter.current.kind === 'new');

    await waitFor(() => chat.turns.length === 0);
    expect(chat.view.kind).to.equal('new');
  });

  it('shows an empty chat once every conversation is deleted', async () => {
    const {app, chat} = await showStreamingAnswer();
    const sidebar = app.querySelector<DlConversationSidebar>('dl-conversation-sidebar')!;

    const deleting = sidebar.deleteAll();
    const dialog = sidebar.querySelector<HTMLDialogElement>('#delete-all-conversations-dialog')!;
    await waitFor(() => dialog.open);
    dialog.querySelector<HTMLButtonElement>('button[value="delete-all"]')!.click();
    expect(await deleting).to.equal(true);

    await waitFor(() => chat.turns.length === 0);
    expect(chat.view.kind).to.equal('new');
  });
});
