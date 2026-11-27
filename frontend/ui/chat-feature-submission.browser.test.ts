// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The chat feature taking what the reader sends: where the message stays while it is refused, loading, or
 *  unavailable, and what the page says when the server turns it down. */

import {expect} from '@esm-bundle/chai';
import {sendKeys} from '@web/test-runner-commands';
import {AnswerSubmissionError} from '../api/web-command-error.ts';
import {setLanguagePreference} from '../i18n/locale.ts';
import {productionHandles} from '../stores/app-handles.ts';
import {DRAFT_KEY, attachmentStore, conversationStore, conversationWire, policy, restoreChatFeatureFixtures, settle, storedTurn, submissions, workspaceStore} from '../testing/chat-feature.ts';
import {EVERY_CHANGE} from '../testing/workspaces.ts';
import {followConversationView} from '../testing/conversation-view.ts';
import {chooseFiles, waitFor} from '../testing/dom.ts';
import './chat-composer.ts';
import './chat-feature.ts';
import type {DlChatComposer} from './chat-composer.ts';
import type {DlChatFeature} from './chat-feature.ts';
import type {ToastRequestDetail} from './toast.ts';

restoreChatFeatureFixtures();

/** The words the reader typed into the feature's composer box, as the composer's own handler hears them. */
async function typeInto(feature: DlChatFeature, text: string): Promise<HTMLTextAreaElement> {
  const input = feature.querySelector<HTMLTextAreaElement>('.composer-input')!;
  input.value = text;
  input.dispatchEvent(new Event('input', {bubbles: true}));
  await feature.querySelector('dl-chat-composer')?.updateComplete;
  return input;
}

/** Press Enter in the composer's box the way a reader does, and let the feature draw what came of it. */
async function pressEnter(feature: DlChatFeature): Promise<void> {
  feature.querySelector<HTMLTextAreaElement>('.composer-input')!.focus();
  await sendKeys({press: 'Enter'});
  await settle(feature);
}

/** Every toast the feature asks the app to show, in order. */
function toastsOf(feature: DlChatFeature): string[] {
  const toasts: string[] = [];
  feature.addEventListener('dl-toast-request', (event) => {
    toasts.push((event as CustomEvent<ToastRequestDetail>).detail.message);
  });
  return toasts;
}

/** The question an answer request carries, whether it went as JSON or, with files, as a form. */
function queryOf(init: RequestInit): string {
  return init.body instanceof FormData
    ? String(init.body.get('query'))
    : String(JSON.parse(String(init.body)).query);
}

/** A POST to /web/api/answer the server refuses before accepting anything: the reader may edit and resend. */
function refuseAnswers(): string[] {
  const posts: string[] = [];
  window.fetch = async (input, init) => {
    if (init?.method === 'POST' && String(input) === '/web/api/answer') {
      posts.push(queryOf(init));
      return jsonResponse(422, {kind: 'invalid_request', message: 'Revise this request'});
    }
    throw new Error(`unexpected fetch: ${String(input)}`);
  };
  return posts;
}

/** A request that never answers, and ends when it is aborted. */
function pendingRequest(init?: RequestInit): Promise<Response> {
  return new Promise<Response>((_resolve, reject) => {
    init?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), {once: true});
  });
}

async function submissionFailureText(respond: () => Response): Promise<string | undefined> {
  workspaceStore.init([
    {workspace: 'alpha', displayName: 'Alpha', embeddingModel: '', changes: EVERY_CHANGE},
  ], ['alpha'], 'alpha');
  window.fetch = async (input) => String(input).includes('/answer-submissions/')
    ? new Response(null, {status: 404})
    : respond();
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready',
    conversationId: 'failure-reason',
    lineage: null,
    history: [storedTurn()],
  };
  document.body.appendChild(feature);
  await settle(feature);
  const input = feature.querySelector<HTMLTextAreaElement>('[aria-label="Message"]')!;
  input.value = 'A new question';
  input.dispatchEvent(new Event('input', {bubbles: true}));
  await feature.querySelector('dl-chat-composer')?.updateComplete;
  feature.querySelector<HTMLButtonElement>('[aria-label="Send"]')?.click();
  await waitFor(() => feature.turns.length === 2 && feature.turns.at(-1)?.state === 'failed');
  await feature.updateComplete;
  return feature.querySelector('.submission-failure span')?.textContent?.trim();
}

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {'Content-Type': 'application/json'},
  });
}

/** One command the server executed and kept, as the history and command routes state it. */
function commandWire(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    command_id: 'command-1',
    sequence: 2,
    name: 'goal',
    input: '',
    result: 'No goal in this conversation.',
    tone: 'success',
    created_at: '2026-01-01T00:00:00Z',
    result_data: null,
    ...overrides,
  };
}

/** One turn as the history route states it; the default is settled, so nothing is followed. */
function historyTurnWire(
  number: number,
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    turn_id: `turn-${number}`,
    turn_number: number,
    answer_run_id: `run-${number}`,
    submission_id: `submission-${number}`,
    status: 'cancelled',
    cancel_requested: false,
    user_text: `Question ${number}`,
    assistant_text: '',
    user_attachments: [],
    steering_messages: [],
    presentation: null,
    usage: {},
    error_kind: null,
    error_message: null,
    created_at: '2026-01-01T00:00:00Z',
    ...overrides,
  };
}

/** One history page as the route states it: the turns and commands of a page, and its cursor. */
function historyWire(
  conversationId: string,
  page: {
    turns?: readonly Record<string, unknown>[];
    commands?: readonly Record<string, unknown>[];
    nextCursor?: string | null;
  } = {},
): Record<string, unknown> {
  return {
    conversation: conversationWire({
      conversationId,
      title: conversationId,
      createdAt: '2026-01-01T00:00:00Z',
      updatedAt: '2026-01-01T00:00:00Z',
      forkedFromConversationId: null,
      forkedFromTitle: null,
      schedule: null,
    }),
    turns: page.turns ?? [],
    commands: page.commands ?? [],
    next_cursor: page.nextCursor ?? null,
  };
}

/** What the chat's message list draws, in order: `turn:<id>` and `command:<id>` for the page. */
function drawnOrder(feature: DlChatFeature): string[] {
  const messages = feature.querySelector<HTMLElement>('#chat-messages');
  return [...(messages?.children ?? [])].flatMap((child) => {
    const element = child as HTMLElement;
    const commandId = element.dataset.commandId;
    if (commandId) return [`command:${commandId}`];
    if (element.dataset.turnSlot !== undefined && element.dataset.turnId) {
      return [`turn:${element.dataset.turnId}`];
    }
    return [];
  });
}

it('forces a submitted turn into view without forcing later stream updates', async () => {
  let finishRequest!: () => void;
  window.fetch = () => new Promise<Response>((resolve) => {
    finishRequest = () => resolve(new Response(JSON.stringify({
      kind: 'invalid_request', message: 'Invalid question',
    }), {
      status: 422,
      headers: {'Content-Type': 'application/json'},
    }));
  });
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready',
    conversationId: 'submit-scroll',
    lineage: null,
    history: [storedTurn()],
  };
  document.body.appendChild(feature);
  await settle(feature);

  const area = feature.querySelector<HTMLElement>('main[aria-label="Chat"]')!;
  let scrollTop = 0;
  Object.defineProperties(area, {
    scrollHeight: {configurable: true, value: 1000},
    clientHeight: {configurable: true, value: 100},
    scrollTop: {
      configurable: true,
      get: () => scrollTop,
      set: (value: number) => { scrollTop = value; },
    },
  });
  const input = feature.querySelector<HTMLTextAreaElement>('[aria-label="Message"]')!;
  input.value = 'A new question';
  input.dispatchEvent(new Event('input', {bubbles: true}));
  await feature.querySelector('dl-chat-composer')?.updateComplete;
  feature.querySelector<HTMLButtonElement>('[aria-label="Send"]')?.click();
  await waitFor(() => feature.turns.length === 2);
  await feature.updateComplete;
  const submitting = feature.querySelector<HTMLButtonElement>('[aria-label="Submitting"]');
  expect(submitting?.disabled).to.equal(true);
  await feature.querySelector('dl-chat-message-list')?.updateComplete;
  await new Promise((resolve) => requestAnimationFrame(resolve));
  expect(scrollTop).to.equal(1000);

  finishRequest();
  await waitFor(() => feature.turns.at(-1)?.state === 'failed');
  scrollTop = 0;
  feature.turns = feature.turns.map((turn, index) => index === feature.turns.length - 1
    ? {...turn, state: 'streaming', streamText: 'Later token'}
    : turn);
  await feature.querySelector('dl-chat-message-list')?.updateComplete;
  await new Promise((resolve) => requestAnimationFrame(resolve));
  expect(scrollTop).to.equal(0);
});

it('Edit restores the original query, mode, workspaces, and attachment lease', async () => {
  workspaceStore.init([
    {workspace: 'alpha', displayName: 'Alpha', embeddingModel: '', changes: EVERY_CHANGE},
    {workspace: 'beta', displayName: 'Beta', embeddingModel: '', changes: EVERY_CHANGE},
  ], ['alpha'], 'alpha');
  const submissionIds: string[] = [];
  window.fetch = async (_input, init) => {
    const body = init?.body;
    submissionIds.push(body instanceof FormData
      ? String(body.get('submission_id'))
      : String(JSON.parse(String(body)).submission_id));
    return new Response(JSON.stringify({
      kind: 'invalid_request',
      message: 'Revise this request',
    }), {
      status: 422,
      headers: {'Content-Type': 'application/json'},
    });
  };

  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.attachmentPolicy = policy;
  feature.attachmentAccept = 'image/*,.md,.pdf';
  feature.view = {
    kind: 'ready',
    conversationId: 'edit-intent',
    lineage: null,
    history: [],
  };
  document.body.appendChild(feature);
  await settle(feature);

  const composer = feature.querySelector('dl-chat-composer') as DlChatComposer;
  chooseFiles(composer, [new File(['notes'], 'notes.md', {type: 'text/markdown'})]);
  composer.querySelector<HTMLButtonElement>('.composer-mode-trigger')?.click();
  await composer.updateComplete;
  composer.querySelector<HTMLButtonElement>('[data-mode="research"]')?.click();
  const input = composer.querySelector<HTMLTextAreaElement>('[aria-label="Message"]')!;
  input.value = 'Original question';
  input.dispatchEvent(new Event('input', {bubbles: true}));
  await composer.updateComplete;
  composer.querySelector<HTMLButtonElement>('[aria-label="Send"]')?.click();
  await waitFor(() => feature.turns.at(-1)?.state === 'failed');

  workspaceStore.select('beta');
  const edit = Array.from(feature.querySelectorAll<HTMLButtonElement>('.submission-failure button'))
    .find((button) => button.textContent?.trim() === 'Edit')!;
  edit.click();
  await settle(feature);

  expect(composer.querySelector<HTMLTextAreaElement>('[aria-label="Message"]')?.value)
    .to.equal('Original question');
  expect(composer.querySelector<HTMLButtonElement>('.composer-mode-trigger')?.textContent?.trim())
    .to.equal('Research');
  expect(workspaceStore.active).to.deep.equal(['alpha']);
  expect(attachmentStore.list().map((item) => item.file.name)).to.deep.equal(['notes.md']);

  composer.querySelector<HTMLButtonElement>('[aria-label="Send"]')?.click();
  await waitFor(() => submissionIds.length === 2 && feature.turns.at(-1)?.state === 'failed');
  expect(submissionIds[1]).not.to.equal(submissionIds[0]);
});

it('puts the message being edited first and keeps what the reader typed since after a blank line, so Edit loses no words', async () => {
  const posts = refuseAnswers();
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {kind: 'new'};
  document.body.appendChild(feature);
  await settle(feature);
  const send = async (): Promise<void> => {
    feature.querySelector<HTMLButtonElement>('[aria-label="Send"]')?.click();
    await waitFor(() => feature.querySelector('.submission-failure') !== null);
  };
  const edit = async (): Promise<void> => {
    [...feature.querySelectorAll<HTMLButtonElement>('.submission-failure button')]
      .find((button) => button.textContent?.trim() === 'Edit')!.click();
    await settle(feature);
  };

  // What the reader typed since the message went is theirs too.
  await typeInto(feature, 'The first question');
  await send();
  const input = await typeInto(feature, 'A thought typed meanwhile');
  await edit();
  expect(input.value).to.equal('The first question\n\nA thought typed meanwhile');
  window.dispatchEvent(new Event('pagehide'));
  expect(localStorage.getItem(DRAFT_KEY), 'and what is kept is what is in the box')
    .to.equal('The first question\n\nA thought typed meanwhile');

  // An empty box takes the message as it was.
  await send();
  expect(input.value).to.equal('');
  await edit();
  expect(input.value).to.equal(posts[1]);

  // A box that already holds the same words does not hold them twice.
  await send();
  await typeInto(feature, `  ${posts[2]}  `);
  await edit();
  expect(input.value).to.equal(posts[2]);
});

it('keeps failed actors per route and restores their optimistic turn on return', async () => {
  window.fetch = async (input, init) => {
    if (init?.method === 'POST' && String(input) === '/web/api/answer') {
      throw new TypeError('Failed to fetch');
    }
    if (String(input).startsWith('/web/api/answer-submissions/')) {
      return new Response(null, {status: 404});
    }
    throw new Error(`unexpected fetch: ${String(input)}`);
  };
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {kind: 'new'};
  document.body.appendChild(feature);
  await settle(feature);

  const input = feature.querySelector<HTMLTextAreaElement>('[aria-label="Message"]')!;
  input.value = 'Keep this failed question';
  input.dispatchEvent(new Event('input', {bubbles: true}));
  await feature.querySelector('dl-chat-composer')?.updateComplete;
  feature.querySelector<HTMLButtonElement>('[aria-label="Send"]')?.click();
  await waitFor(() => feature.turns.at(-1)?.state === 'failed');
  await waitFor(() => feature.textContent?.includes('Retry') ?? false);
  expect(feature.textContent).to.contain('Retry');

  feature.view = {
    kind: 'ready',
    conversationId: 'another-conversation',
    lineage: null,
    history: [],
  };
  await settle(feature);
  expect(Boolean(feature.querySelector('.submission-failure'))).to.equal(false);
  expect(feature.textContent).not.to.contain('Keep this failed question');

  feature.view = {kind: 'new'};
  await settle(feature);
  expect(feature.textContent).to.contain('Keep this failed question');
  expect(feature.textContent).to.contain('Retry');
});

it('says an answer was refused at the reader\'s own limit, keeps the draft, and offers Retry', async () => {
  // The real refusal of POST /web/api/answer over the active_answer_runs policy: 429, the
  // retryable command kind, and the error kind the page localizes.
  const posts: string[] = [];
  window.fetch = async (input, init) => {
    if (init?.method === 'POST' && String(input) === '/web/api/answer') {
      posts.push(String(input));
      return new Response(JSON.stringify({
        kind: 'service_unavailable',
        message: 'You have as many answers in progress as you may run at once. '
          + 'Wait for one to finish or stop one, then try again.',
        error_kind: 'active_run_limit',
      }), {status: 429, headers: {'Content-Type': 'application/json'}});
    }
    // Nothing was accepted, so the submission lookup finds nothing.
    if (String(input).startsWith('/web/api/answer-submissions/')) {
      return new Response(null, {status: 404});
    }
    throw new Error(`unexpected fetch: ${String(input)}`);
  };
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {kind: 'new'};
  document.body.appendChild(feature);
  await settle(feature);

  const input = feature.querySelector<HTMLTextAreaElement>('[aria-label="Message"]')!;
  input.value = 'Keep this question';
  input.dispatchEvent(new Event('input', {bubbles: true}));
  await feature.querySelector('dl-chat-composer')?.updateComplete;
  feature.querySelector<HTMLButtonElement>('[aria-label="Send"]')?.click();
  await waitFor(() => feature.querySelector('.submission-failure button') !== null);

  const failure = (): HTMLElement => feature.querySelector<HTMLElement>('.submission-failure')!;
  const buttons = (): string[] => Array.from(failure().querySelectorAll('button'))
    .map((button) => button.textContent?.trim() ?? '');
  expect(failure().getAttribute('role')).to.equal('alert');
  expect(failure().textContent).to.contain('You have as many answers in progress as you may run at once.');
  expect(buttons()).to.deep.equal(['Retry', 'Edit', 'Discard']);
  try {
    await setLanguagePreference('zh');
    await feature.updateComplete;
    expect(failure().textContent).to.contain('你同时进行的回答已达到上限');
    expect(buttons()).to.deep.equal(['重试', '编辑', '放弃']);
  } finally {
    await setLanguagePreference('en');
  }
  await feature.updateComplete;

  // The reader's words are still theirs: Edit hands the draft back to the composer.
  Array.from(failure().querySelectorAll('button'))
    .find((button) => button.textContent?.trim() === 'Edit')!.click();
  await settle(feature);
  expect(feature.querySelector<HTMLTextAreaElement>('[aria-label="Message"]')?.value)
    .to.equal('Keep this question');
  expect(posts).to.have.length(1);
});

it('keeps a new message in the box, and in storage, while an earlier one that failed waits on Retry, Edit, or Discard', async () => {
  const posts = refuseAnswers();
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {kind: 'new'};
  const toasts = toastsOf(feature);
  document.body.appendChild(feature);
  await settle(feature);

  await typeInto(feature, 'The first question');
  feature.querySelector<HTMLButtonElement>('[aria-label="Send"]')?.click();
  await waitFor(() => feature.querySelector('.submission-failure button') !== null);
  const banner = feature.querySelector<HTMLElement>('.submission-failure')!;
  const buttons = [...banner.querySelectorAll('button')];
  expect(buttons.map((button) => button.textContent?.trim())).to.deep.equal(['Edit', 'Discard']);
  expect(posts).to.deep.equal(['The first question']);

  // The reader goes on without settling it: a second question, and Enter.
  const input = await typeInto(feature, 'The second question');
  await pressEnter(feature);

  expect(posts, 'nothing was sent').to.have.length(1);
  expect(feature.turns, 'and no turn was added').to.have.length(1);
  expect(input.value, 'the words are still in the box').to.equal('The second question');
  window.dispatchEvent(new Event('pagehide'));
  expect(localStorage.getItem(DRAFT_KEY), 'and in storage').to.equal('The second question');
  expect(toasts, 'the server refused it, so there is no Retry to name').to.deep.equal([
    'Edit or discard your earlier message first.',
  ]);
  // (Compared as a flag: when two page elements differ, the runner stalls printing the page.)
  expect(document.activeElement === banner, 'the reader is taken to the banner, not to a choice in it').to.equal(true);
  expect(banner.getAttribute('tabindex'), 'a banner that can hold focus without being a stop in the tab order').to.equal('-1');
  // A second Enter from a reader who has not looked up chooses nothing for them: Edit would have taken the
  // second question out of the box and put the first back.
  await sendKeys({press: 'Enter'});
  await settle(feature);
  expect(document.activeElement === banner, 'and focus stays where it was put').to.equal(true);
  expect(feature.querySelector('.submission-failure') === banner, 'the banner is still there').to.equal(true);
  expect(input.value, 'the second question is where it was').to.equal('The second question');
  expect(posts, 'and nothing went').to.have.length(1);

  // The words wait there in their own language too.
  try {
    await setLanguagePreference('zh');
    await settle(feature);
    await pressEnter(feature);
    expect(toasts.at(-1)).to.equal('请先编辑或放弃之前的消息。');
  } finally {
    await setLanguagePreference('en');
  }
  expect(input.value).to.equal('The second question');

  // Once the earlier message is settled, the same words go.
  banner.querySelectorAll('button')[1]!.click();
  await settle(feature);
  expect(feature.querySelectorAll('.submission-failure')).to.have.length(0);
  await pressEnter(feature);
  await waitFor(() => posts.length === 2);
  expect(posts[1]).to.equal('The second question');
  expect(input.value, 'taken, so the box lets go').to.equal('');
  expect(toasts, 'the two refusals above are all there were').to.have.length(2);
});

it('names Retry too when the earlier message can be retried, and takes the reader to it', async () => {
  const posts: string[] = [];
  window.fetch = async (input, init) => {
    if (init?.method === 'POST' && String(input) === '/web/api/answer') {
      posts.push(queryOf(init));
      throw new TypeError('Failed to fetch');
    }
    // The server never heard of it, so the reader may send the same message again.
    if (String(input).startsWith('/web/api/answer-submissions/')) return new Response(null, {status: 404});
    throw new Error(`unexpected fetch: ${String(input)}`);
  };
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {kind: 'new'};
  const toasts = toastsOf(feature);
  document.body.appendChild(feature);
  await settle(feature);

  await typeInto(feature, 'The first question');
  feature.querySelector<HTMLButtonElement>('[aria-label="Send"]')?.click();
  await waitFor(() => feature.querySelectorAll('.submission-failure button').length === 3);
  const buttons = [...feature.querySelectorAll<HTMLButtonElement>('.submission-failure button')];
  expect(buttons.map((button) => button.textContent?.trim())).to.deep.equal(['Retry', 'Edit', 'Discard']);

  const input = await typeInto(feature, 'The second question');
  await pressEnter(feature);

  expect(posts).to.deep.equal(['The first question']);
  expect(input.value).to.equal('The second question');
  expect(toasts).to.deep.equal(['Retry, edit or discard your earlier message first.']);
  const banner = feature.querySelector<HTMLElement>('.submission-failure')!;
  expect(document.activeElement === banner, 'the reader is taken to the banner, and not to Retry').to.equal(true);
  // A second Enter does not choose Retry for them.
  await sendKeys({press: 'Enter'});
  await settle(feature);
  expect(posts, 'the first message was sent again').to.deep.equal(['The first question']);
  expect(buttons.every((button) => button.isConnected), 'the choices are all still on offer').to.equal(true);
});

it('keeps the message in the box, and says so, while the conversation is still loading', async () => {
  const requests: string[] = [];
  let history!: (response: Response) => void;
  window.fetch = (input, init) => {
    requests.push(String(input));
    if (String(input).endsWith('/slow-conversation/history')) {
      return new Promise<Response>((resolve) => { history = resolve; });
    }
    if (init?.method === 'POST' && String(input) === '/web/api/answer') {
      return Promise.resolve(jsonResponse(422, {kind: 'invalid_request', message: 'Revise this request'}));
    }
    return pendingRequest(init);
  };
  const summary = {
    conversationId: 'slow-conversation', title: 'Slow', createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z', forkedFromConversationId: null, forkedFromTitle: null, schedule: null,
  };
  const opening = conversationStore.open('slow-conversation');
  expect(conversationStore.viewState).to.equal('loading');
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {kind: 'loading'};
  const toasts = toastsOf(feature);
  document.body.appendChild(feature);
  await settle(feature);

  const input = await typeInto(feature, 'Typed while it loads');
  await pressEnter(feature);

  expect(requests.filter((url) => url === '/web/api/answer'), 'nothing was sent').to.deep.equal([]);
  expect(feature.turns).to.have.length(0);
  expect(input.value, 'the words are still in the box').to.equal('Typed while it loads');
  window.dispatchEvent(new Event('pagehide'));
  expect(localStorage.getItem(DRAFT_KEY), 'and in storage').to.equal('Typed while it loads');
  expect(toasts).to.deep.equal(['This conversation is still loading. Your message is still in the box.']);

  try {
    await setLanguagePreference('zh');
    await settle(feature);
    await pressEnter(feature);
    expect(toasts.at(-1)).to.equal('会话仍在加载，你的消息还在输入框里。');
  } finally {
    await setLanguagePreference('en');
  }

  // When the conversation has arrived, the same words are sent.
  history(jsonResponse(200, {conversation: conversationWire(summary), turns: [], commands: [], next_cursor: null}));
  expect(await opening).to.equal('ready');
  feature.view = {kind: 'ready', conversationId: 'slow-conversation', lineage: null, history: []};
  await settle(feature);
  await pressEnter(feature);
  await waitFor(() => requests.includes('/web/api/answer'));
  expect(input.value).to.equal('');
  expect(toasts).to.have.length(2);
});

it('keeps the message in the box when the conversation is unavailable, with the message that says so', async () => {
  window.fetch = async (input) => {
    if (String(input).endsWith('/gone-conversation/history')) return jsonResponse(404, {detail: 'not found'});
    throw new Error(`unexpected fetch: ${String(input)}`);
  };
  expect(await conversationStore.open('gone-conversation')).to.equal('unavailable');
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {kind: 'unavailable', hasRecent: false};
  const toasts = toastsOf(feature);
  document.body.appendChild(feature);
  await settle(feature);

  const input = await typeInto(feature, 'Typed into a conversation that is gone');
  await pressEnter(feature);

  expect(feature.turns).to.have.length(0);
  expect(input.value).to.equal('Typed into a conversation that is gone');
  window.dispatchEvent(new Event('pagehide'));
  expect(localStorage.getItem(DRAFT_KEY)).to.equal('Typed into a conversation that is gone');
  expect(toasts).to.deep.equal(['Conversation service is unavailable. Please retry loading the conversation.']);
});

it('delivers a message offered while a Run is active to the queue instead of refusing it', async () => {
  const queued = {
    message_id: 'message-1',
    content: 'Another question',
    position: 1,
    created_at: '2026-01-01T00:00:00Z',
  };
  window.fetch = ((input, init) => {
    const url = String(input);
    if (url.endsWith('/events')) return pendingRequest(init);
    if (url === '/web/api/answer' && init?.method === 'POST') {
      return Response.json({kind: 'queue', message: queued});
    }
    if (url.endsWith('/queue')) return Response.json({items: [queued]});
    throw new Error(`unexpected fetch: ${url}`);
  }) as typeof fetch;
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready',
    conversationId: 'conversation-active-run',
    lineage: null,
    history: [{...storedTurn(), answerRunId: 'run-active', turnId: 'turn-active', status: 'running', presentation: null}],
  };
  const toasts = toastsOf(feature);
  document.body.appendChild(feature);
  await feature.updateComplete;
  const composer = feature.querySelector('dl-chat-composer') as DlChatComposer;
  await waitFor(() => composer.running === true);

  const offer = new CustomEvent('dl-composer-submit', {
    detail: {
      query: 'Another question',
      mode: null,
      requestedSkill: null,
      effort: null,
      delivery: 'queue',
    },
    bubbles: true, composed: true, cancelable: true,
  });
  composer.dispatchEvent(offer);

  expect(offer.defaultPrevented, 'the words were taken').to.equal(false);
  await waitFor(() => feature.turns.length === 1);
  await waitFor(() => toasts.length === 1);
  expect(toasts).to.deep.equal(['Queued — it will be sent after this answer.']);
  feature.detachRun();
});

it('refuses a message when a submission this page has not drawn holds the conversation, and gives the files back', async () => {
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.attachmentPolicy = policy;
  feature.view = {kind: 'new'};
  const toasts = toastsOf(feature);
  document.body.appendChild(feature);
  await settle(feature);
  const composer = feature.querySelector('dl-chat-composer') as DlChatComposer;

  // An earlier message, failed, that the registry holds for the new chat; this page's view has not changed
  // since, so it is not the one the page tracks.
  submissions.start({
    submissionId: 'held-submission', conversationId: null, query: 'Held', workspaces: [], mode: null,
  }, attachmentStore.leaseAll(), {
    submit: async () => { throw new AnswerSubmissionError(422, 'invalid_request', 'Revise this request'); },
    lookup: async () => null,
  });
  await waitFor(() => submissions.list()[0]?.status === 'editable');

  chooseFiles(composer, [new File(['notes'], 'notes.md', {type: 'text/markdown'})]);
  const input = await typeInto(feature, 'Another question');
  const offer = new CustomEvent('dl-composer-submit', {
    detail: {query: 'Another question', mode: null, requestedSkill: null, effort: null},
    bubbles: true, composed: true, cancelable: true,
  });
  composer.dispatchEvent(offer);

  expect(offer.defaultPrevented).to.equal(true);
  expect(feature.turns).to.have.length(0);
  expect(attachmentStore.list().map((item) => item.file.name), 'the file is back where the reader staged it')
    .to.deep.equal(['notes.md']);
  expect(input.value).to.equal('Another question');
  expect(toasts).to.deep.equal(['Edit or discard your earlier message first.']);
});

it('keeps the words in the box when taking the message throws', async () => {
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {kind: 'new'};
  document.body.appendChild(feature);
  await settle(feature);
  const input = await typeInto(feature, 'Words that must survive');
  (attachmentStore as {leaseAll: unknown}).leaseAll = () => { throw new Error('lease failed'); };
  // The page reports an error a listener throws; this test hears it, and the runner logs it too.
  const reported: string[] = [];
  const onerror = window.onerror;
  window.onerror = (message) => { reported.push(String(message)); return true; };
  try {
    await pressEnter(feature);
  } finally {
    window.onerror = onerror;
    delete (attachmentStore as {leaseAll?: unknown}).leaseAll;
  }

  expect(reported.join(' '), 'the failure is reported, not swallowed').to.contain('lease failed');
  expect(feature.turns).to.have.length(0);
  expect(input.value, 'and the box was not emptied for a message nobody has').to.equal('Words that must survive');
});

it('clears the box and its stored draft for a message the feature takes, and sends nothing twice', async () => {
  const posts: string[] = [];
  window.fetch = (input, init) => {
    if (init?.method === 'POST' && String(input) === '/web/api/answer') posts.push(queryOf(init));
    return pendingRequest(init);
  };
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {kind: 'new'};
  const toasts = toastsOf(feature);
  document.body.appendChild(feature);
  await settle(feature);

  const input = await typeInto(feature, 'A question that goes');
  window.dispatchEvent(new Event('pagehide'));
  expect(localStorage.getItem(DRAFT_KEY)).to.equal('A question that goes');
  await pressEnter(feature);

  await waitFor(() => posts.length === 1);
  expect(feature.turns.map((turn) => turn.userText)).to.deep.equal(['A question that goes']);
  expect(input.value, 'taken, so the box lets go').to.equal('');
  expect(localStorage.getItem(DRAFT_KEY), 'and so does storage').to.equal(null);
  expect(toasts).to.deep.equal([]);

  // The next Enter has no words to send, and the one in flight is not sent again.
  await pressEnter(feature);
  expect(posts).to.have.length(1);
});

it('runs a matched command on the server and draws the command it kept, never as an answer run', async () => {
  workspaceStore.init([
    {workspace: 'alpha', displayName: 'Alpha', embeddingModel: '', changes: EVERY_CHANGE},
  ], ['alpha'], 'alpha');
  const requests: {url: string; init?: RequestInit}[] = [];
  let goalReads = 0;
  window.fetch = async (input, init) => {
    const url = String(input);
    requests.push({url, init});
    const method = init?.method ?? 'GET';
    if (method === 'GET' && url.endsWith('/history')) {
      return jsonResponse(200, historyWire('goal-post'));
    }
    if (method === 'GET' && url.endsWith('/goal')) {
      goalReads += 1;
      return jsonResponse(200, null);
    }
    if (method === 'POST' && url === '/web/api/conversations/goal-post/commands') {
      return jsonResponse(200, {command: commandWire({
        command_id: 'command-cap',
        sequence: 1,
        input: 'cap 7',
        result: 'Goal cap raised.\nObjective: finish the sealed model\nRound: 3/7',
      })});
    }
    throw new Error(`unexpected fetch: ${method} ${url}`);
  };
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  document.body.append(feature);
  const toasts = toastsOf(feature);
  followConversationView(feature);
  await conversationStore.open('goal-post');
  await settle(feature);
  const goalReadsBefore = goalReads;

  await typeInto(feature, '/goal cap 7');
  await pressEnter(feature);
  await waitFor(() => toasts.length === 1);

  const post = requests.find(
    (request) => request.init?.method === 'POST' && request.url.endsWith('/commands'),
  );
  expect(post?.url).to.equal('/web/api/conversations/goal-post/commands');
  expect(JSON.parse(String(post?.init?.body))).to.deep.equal({
    name: 'goal',
    input: 'cap 7',
    workspaces: ['alpha'],
  });
  expect(requests.some((request) => request.url === '/web/api/answer')).to.equal(false);
  expect(feature.turns, 'a command starts no answer turn').to.have.length(0);
  expect(conversationStore.commands.map((command) => command.commandId))
    .to.deep.equal(['command-cap']);
  expect(goalReads, 'the goal bar reads again after the command').to.be.greaterThan(
    goalReadsBefore,
  );

  expect(feature.querySelector('[data-command-id="command-cap"]'),
    'a successful command is control-plane work, not a transcript row').to.equal(null);
  expect(toasts).to.deep.equal(['Goal cap raised.\nObjective: finish the sealed model\nRound: 3/7']);
});

it('creates the conversation first, then starts the /goal round without a reload', async () => {
  workspaceStore.init([
    {workspace: 'alpha', displayName: 'Alpha', embeddingModel: '', changes: EVERY_CHANGE},
  ], ['alpha'], 'alpha');
  const requests: {url: string; init?: RequestInit}[] = [];
  window.fetch = async (input, init) => {
    const url = String(input);
    const method = init?.method ?? 'GET';
    requests.push({url, init});
    if (method === 'GET' && url.startsWith('/web/api/conversations?')) {
      return jsonResponse(200, {items: [], next_cursor: null});
    }
    if (method === 'POST' && url === '/web/api/conversations') {
      return jsonResponse(200, {
        conversation_id: 'new-conv',
        title: null,
        created_at: '2026-10-25T00:00:00Z',
        updated_at: '2026-10-25T00:00:00Z',
      });
    }
    if (method === 'GET' && url.endsWith('/conversations/new-conv/history')) {
      return jsonResponse(200, historyWire('new-conv', {
        turns: [historyTurnWire(1, {status: 'running'})],
      }));
    }
    if (method === 'GET' && url.endsWith('/conversations/new-conv/goal')) {
      return jsonResponse(200, null);
    }
    if (method === 'POST' && url === '/web/api/conversations/new-conv/commands') {
      return jsonResponse(200, {command: commandWire({
        command_id: 'command-created',
        sequence: 1,
        input: 'ship it',
        result: 'Ongoing goal\nObjective: ship it\nRound: 1/16',
      })});
    }
    if (method === 'GET' && url.startsWith('/web/api/run-activity?')) {
      return jsonResponse(200, {items: []});
    }
    throw new Error(`unexpected fetch: ${method} ${url}`);
  };
  const activity = productionHandles().runActivity;
  const originalWake = activity.wake;
  let wakes = 0;
  activity.wake = () => {
    wakes += 1;
  };
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  document.body.append(feature);
  followConversationView(feature);
  const toasts = toastsOf(feature);
  await settle(feature);

  try {
    await typeInto(feature, '/goal ship it');
    await pressEnter(feature);
    await waitFor(() => toasts.length === 1);

    const created = requests.filter(
      (request) => request.init?.method === 'POST' && request.url === '/web/api/conversations',
    );
    expect(created, 'the command created the conversation first').to.have.length(1);
    const commandPost = requests.find(
      (request) => request.url === '/web/api/conversations/new-conv/commands',
    );
    expect(commandPost?.init?.method).to.equal('POST');
    expect(JSON.parse(String(commandPost?.init?.body)), 'the command carries the composer\'s scope')
      .to.deep.equal({name: 'goal', input: 'ship it', workspaces: ['alpha']});
    expect(conversationStore.activeConversationId).to.equal('new-conv');
    const historyReads = requests.filter(
      (request) => request.url.endsWith('/conversations/new-conv/history'),
    );
    expect(historyReads.length, 'the page read the new conversation after the command')
      .to.be.greaterThan(0);
    expect(wakes, 'the page woke Run activity for the round the command started').to.equal(1);
    await waitFor(() => feature.turns.length === 1);
    expect(feature.turns[0], 'the server-started round turn is attached to the view').to.not.equal(undefined);
    expect(feature.turns[0]?.state, 'the attached round turn is the live one')
      .to.be.oneOf(['pending', 'running']);
    expect(toasts).to.deep.equal(['Ongoing goal\nObjective: ship it\nRound: 1/16']);
  } finally {
    activity.wake = originalWake;
    window.history.replaceState(null, '', '/web');
  }
});

it('does not create a conversation for a bare /goal status command', async () => {
  const requests: {url: string; init?: RequestInit}[] = [];
  window.fetch = async (input, init) => {
    const url = String(input);
    const method = init?.method ?? 'GET';
    requests.push({url, init});
    if (method === 'GET' && url.startsWith('/web/api/conversations?')) {
      return jsonResponse(200, {items: [], next_cursor: null});
    }
    throw new Error(`unexpected fetch: ${method} ${url}`);
  };
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  document.body.append(feature);
  const toasts = toastsOf(feature);
  await settle(feature);

  await typeInto(feature, '/goal');
  await pressEnter(feature);
  await waitFor(() => toasts.length === 1);

  expect(requests.some(
    (request) => request.init?.method === 'POST' && request.url === '/web/api/conversations',
  ), 'a read-only command leaves no empty conversation behind').to.equal(false);
  expect(conversationStore.activeConversationId).to.equal(null);
  expect(toasts).to.deep.equal(['No goal in this conversation. /goal <objective> creates one.']);
});

it('localizes a goal command receipt through lit-localize', async () => {
  window.fetch = async (input, init) => {
    const url = String(input);
    const method = init?.method ?? 'GET';
    if (method === 'GET' && url.endsWith('/goal')) return jsonResponse(200, null);
    if (method === 'POST' && url.endsWith('/commands')) {
      return jsonResponse(200, {command: commandWire({
        command_id: 'command-localized',
        sequence: 2,
        result: 'CANONICAL RESULT THAT MUST NOT SHOW',
        result_data: {
          kind: 'goal_status',
          goal: {
            objective: 'finish the sealed model',
            phase: 'blocked',
            max_rounds: 7,
            rounds_started: 3,
            blocked_reason: 'the goal reached its round cap',
          },
        },
      })});
    }
    throw new Error(`unexpected fetch: ${method} ${url}`);
  };
  conversationStore.adoptCreatedConversation({
    conversationId: 'goal-localized',
    title: 'Localized goal',
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z',
    forkedFromConversationId: null,
    forkedFromTitle: null,
    schedule: null,
  });
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready',
    conversationId: 'goal-localized',
    lineage: null,
    history: [],
  };
  const toasts = toastsOf(feature);
  document.body.appendChild(feature);
  await settle(feature);

  try {
    await setLanguagePreference('zh');
    await settle(feature);
    await typeInto(feature, '/goal');
    await pressEnter(feature);
    await waitFor(() => toasts.length === 1);
    expect(toasts[0]).to.equal('已阻塞的目标 · 轮次：3/7');
    expect(feature.querySelector('[data-command-id="command-localized"]'),
      'the success is a receipt, not a transcript card').to.equal(null);
  } finally {
    await setLanguagePreference('en');
    await settle(feature);
  }
});

it('draws a command the server kept even before the conversation history is read', async () => {
  window.fetch = async (input, init) => {
    const url = String(input);
    const method = init?.method ?? 'GET';
    if (method === 'GET' && url.endsWith('/goal')) return jsonResponse(200, null);
    if (method === 'POST' && url.endsWith('/commands')) {
      return jsonResponse(200, {command: commandWire({
        command_id: 'command-status',
        sequence: 4,
        result: 'Blocked goal\nObjective: finish the sealed model\nRound: 3/7\nBlocked: the goal reached its round cap',
      })});
    }
    throw new Error(`unexpected fetch: ${method} ${url}`);
  };
  conversationStore.adoptCreatedConversation({
    conversationId: 'goal-status',
    title: 'Goal status',
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z',
    forkedFromConversationId: null,
    forkedFromTitle: null,
    schedule: null,
  });
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready',
    conversationId: 'goal-status',
    lineage: null,
    history: [],
  };
  const toasts = toastsOf(feature);
  document.body.appendChild(feature);
  await settle(feature);

  await typeInto(feature, '/goal');
  await pressEnter(feature);
  await waitFor(() => toasts.length === 1);
  expect(toasts[0]).to.contain('Blocked goal');
  expect(toasts[0]).to.contain('Round: 3/7');
  expect(feature.querySelector('[data-command-id="command-status"]'),
    'the goal state lives in the goal bar').to.equal(null);
  expect(conversationStore.commands.map((command) => command.commandId))
    .to.deep.equal(['command-status']);
});

it('draws a command the server refused as a failed command, not as a model answer', async () => {
  window.fetch = async (input, init) => {
    const url = String(input);
    const method = init?.method ?? 'GET';
    if (method === 'GET' && url.endsWith('/goal')) return jsonResponse(200, null);
    if (method === 'POST' && url.endsWith('/commands')) {
      return jsonResponse(200, {command: commandWire({
        command_id: 'command-invalid',
        sequence: 2,
        input: 'cap',
        result: 'Add a positive number: /goal cap 5',
        tone: 'error',
      })});
    }
    throw new Error(`unexpected fetch: ${method} ${url}`);
  };
  conversationStore.adoptCreatedConversation({
    conversationId: 'goal-invalid',
    title: 'Invalid goal command',
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z',
    forkedFromConversationId: null,
    forkedFromTitle: null,
    schedule: null,
  });
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready',
    conversationId: 'goal-invalid',
    lineage: null,
    history: [],
  };
  document.body.appendChild(feature);
  await settle(feature);

  await typeInto(feature, '/goal cap');
  await pressEnter(feature);
  await waitFor(() => feature.querySelector('[data-command-id="command-invalid"]') !== null);

  const card = feature.querySelector<HTMLElement>('[data-command-id="command-invalid"]')!;
  expect(card.getAttribute('data-tone')).to.equal('error');
  expect(card.textContent).to.contain('Add a positive number: /goal cap 5');
  expect(card.querySelector('article, button'), 'a refused command is no model answer').to.equal(null);
});

it('reloads a conversation with its commands interleaved among the turns by sequence', async () => {
  const historyReads: string[] = [];
  window.fetch = async (input, init) => {
    const url = String(input);
    const method = init?.method ?? 'GET';
    if (method === 'GET' && url.endsWith('/goal')) return jsonResponse(200, null);
    if (method === 'GET' && url.includes('/history')) {
      historyReads.push(url);
      const older = url.includes('cursor=');
      return jsonResponse(200, historyWire('history-commands', older ? {
        turns: [historyTurnWire(1)],
        commands: [commandWire({
          command_id: 'command-early',
          sequence: 2,
          input: 'pause',
          result: 'Goal paused.',
          tone: 'error',
        })],
      } : {
        turns: [historyTurnWire(5)],
        commands: [commandWire({
          command_id: 'command-late',
          sequence: 6,
          input: 'cap 7',
          result: 'Goal cap raised.',
          tone: 'error',
        })],
        nextCursor: 'older-cursor',
      }));
    }
    throw new Error(`unexpected fetch: ${method} ${url}`);
  };
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  document.body.append(feature);
  followConversationView(feature);

  await conversationStore.open('history-commands');
  await settle(feature);
  expect(drawnOrder(feature)).to.deep.equal(['turn:turn-5', 'command:command-late']);
  expect(feature.textContent).to.contain('Question 5');
  expect(feature.textContent).to.contain('/goal cap 7');
  expect(feature.textContent).to.contain('Goal cap raised.');

  await conversationStore.loadOlderMessages();
  await settle(feature);

  expect(drawnOrder(feature), 'the older page joins the range above the turns already shown')
    .to.deep.equal(['turn:turn-1', 'command:command-early', 'turn:turn-5', 'command:command-late']);
  expect(historyReads.at(-1)).to.contain('cursor=older-cursor');
  for (const card of feature.querySelectorAll('[data-command-id]')) {
    expect(card.querySelector('article, button'), 'a stored command is no model answer').to.equal(null);
  }
});

it('localizes commands loaded from history and keeps the canonical result as their fallback', async () => {
  window.fetch = async (input, init) => {
    const url = String(input);
    const method = init?.method ?? 'GET';
    if (method === 'GET' && url.endsWith('/goal')) return jsonResponse(200, null);
    if (method === 'GET' && url.includes('/history')) {
      return jsonResponse(200, historyWire('history-localized', {
        commands: [
          commandWire({
            command_id: 'command-none',
            sequence: 2,
            result_data: {kind: 'goal_status', goal: null},
            tone: 'error',
          }),
          commandWire({
            command_id: 'command-cleared',
            sequence: 4,
            result_data: {kind: 'goal_cleared'},
            tone: 'error',
          }),
          commandWire({
            command_id: 'command-fallback',
            sequence: 6,
            result: 'Canonical English result.',
            tone: 'error',
          }),
        ],
      }));
    }
    throw new Error(`unexpected fetch: ${method} ${url}`);
  };
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  document.body.append(feature);
  followConversationView(feature);
  try {
    await setLanguagePreference('zh');
    await conversationStore.open('history-localized');
    await settle(feature);

    expect(drawnOrder(feature)).to.deep.equal([
      'command:command-none',
      'command:command-cleared',
      'command:command-fallback',
    ]);
    const text = (commandId: string): string => (
      feature.querySelector(`[data-command-id="${commandId}"] .command-node-text`)
        ?.textContent ?? ''
    );
    expect(text('command-none')).to.equal('这个对话还没有目标；/goal <目标> 可以创建一个。');
    expect(text('command-cleared')).to.equal('目标已清除。');
    expect(text('command-fallback'), 'a command without structured data keeps the server words')
      .to.equal('Canonical English result.');
    expect(feature.textContent).to.not.contain('No goal in this conversation.');
  } finally {
    await setLanguagePreference('en');
    await settle(feature);
  }
});

it('a rejected submission shows the localized copy for its error kind', async () => {
  const text = await submissionFailureText(() => jsonResponse(422, {
    kind: 'invalid_request',
    message: 'Server wording',
    error_kind: 'unsupported_resource_capability',
  }));
  expect(text).to.equal('This request needs a resource capability that no answer mode can provide.');
});

it('an unmapped rejection shows the server reason', async () => {
  const text = await submissionFailureText(() => jsonResponse(422, {
    kind: 'invalid_request',
    message: 'Server wording',
    error_kind: 'unmapped',
  }));
  expect(text).to.equal('Server wording');
});

it('a failure without a reason shows the localized fallback', async () => {
  const text = await submissionFailureText(() => new Response('upstream failure', {status: 502}));
  expect(text).to.equal('The answer could not be submitted.');
});

it('sends a failed turn\'s words again from the turn itself', async () => {
  const posts: {query: string; delivery?: string}[] = [];
  window.fetch = (input, init) => {
    const url = String(input);
    if (init?.method === 'POST' && url === '/web/api/answer') {
      const body = JSON.parse(String(init.body)) as {query: string; delivery?: string};
      posts.push({query: body.query, delivery: body.delivery});
      return pendingRequest(init);
    }
    throw new Error(`unexpected fetch: ${url}`);
  };
  conversationStore.adoptCreatedConversation({
    conversationId: 'retry-conversation',
    title: null,
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z',
    forkedFromConversationId: null,
    forkedFromTitle: null,
    schedule: null,
  });
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready',
    conversationId: 'retry-conversation',
    lineage: null,
    history: [{
      ...storedTurn(),
      turnId: 'turn-failed',
      answerRunId: 'run-failed',
      status: 'failed',
      presentation: null,
      errorKind: 'session_busy',
      errorMessage: null,
      userText: 'Rebuild the sealed model',
    }],
  };
  document.body.appendChild(feature);
  await settle(feature);

  feature.querySelector<HTMLButtonElement>('[aria-label="Retry this message"]')!.click();

  await waitFor(() => posts.length === 1);
  expect(posts[0]).to.deep.equal({query: 'Rebuild the sealed model', delivery: undefined});
  await waitFor(() => feature.turns.at(-1)?.userText === 'Rebuild the sealed model');
});
