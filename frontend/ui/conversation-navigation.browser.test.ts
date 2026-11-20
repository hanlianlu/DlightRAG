// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {setViewport} from '@web/test-runner-commands';
import type {ConversationSummary} from '../api/conversations.ts';
import {defineDesignSystemElements} from '../design-system/index.ts';
import {setLanguagePreference} from '../i18n/locale.ts';
import {conversationRoute, newChatRoute} from '../lib/router.ts';
import {productionHandles} from '../stores/app-handles.ts';
import type {
  ConversationIntentDetail,
  ConversationRenameDetail,
  DlConversationList,
} from './conversation-list.ts';
import './conversation-list.ts';
import type {ChatView} from './chat-message-list.ts';
import type {
  ConversationSidebarStateDetail,
  DlConversationSidebar,
} from './conversation-sidebar.ts';
import './conversation-sidebar.ts';
import {webRouter} from './router.ts';
import {followConversationView, type FollowedChat, stopFollowingConversationViews} from '../testing/conversation-view.ts';
import {waitFor} from '../testing/dom.ts';

const {conversations: conversationStore} = productionHandles();

defineDesignSystemElements();

const originalFetch = window.fetch;
const originalMatchMedia = window.matchMedia;

const first: ConversationSummary = {
  conversationId: 'conversation-1',
  title: 'Research notes',
  createdAt: '2026-01-01T00:00:00Z',
  updatedAt: '2026-01-02T00:00:00Z',
  forkedFromConversationId: null,
  forkedFromTitle: null,
};

function media(desktop: boolean): (query: string) => MediaQueryList {
  return (query) => ({
    matches: desktop ? query === '(min-width: 1200px)' : query === '(width < 1200px)',
    media: query,
    onchange: null,
    addListener() {},
    removeListener() {},
    addEventListener() {},
    removeEventListener() {},
    dispatchEvent: () => true,
  });
}

const firstWire = {
  conversation_id: 'conversation-1',
  title: 'Research notes',
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-02T00:00:00Z',
};

function response(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {'Content-Type': 'application/json'},
  });
}

function conversationPage(items: ConversationSummary[], nextCursor: string | null = null): Response {
  return response({
    items: items.map((item) => ({
      conversation_id: item.conversationId,
      title: item.title,
      created_at: item.createdAt,
      updated_at: item.updatedAt,
      forked_from_conversation_id: item.forkedFromConversationId,
      forked_from_title: item.forkedFromTitle,
    })),
    next_cursor: nextCursor,
  });
}

function button(root: ParentNode, name: string): HTMLButtonElement {
  const match = [...root.querySelectorAll<HTMLButtonElement>('button')].find((candidate) => {
    return candidate.getAttribute('aria-label') === name || candidate.textContent?.trim() === name;
  });
  if (!match) throw new Error(`button not found: ${name}`);
  return match;
}

function conversationsOpenButton(): HTMLButtonElement {
  const match = document.querySelector<HTMLButtonElement>('#conversation-sidebar-open');
  if (!match) throw new Error('open conversations button not found');
  return match;
}

/** A Chat as the route meets it: it keeps the view it is told, and counts what the route asks of it. */
type StubChat = {-readonly [K in keyof FollowedChat]: FollowedChat[K]} & {detached: number; cleared: number};

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

beforeEach(() => {
  document.body.replaceChildren();
  document.body.className = '';
  const trigger = document.createElement('button');
  trigger.id = 'conversation-sidebar-open';
  trigger.type = 'button';
  trigger.setAttribute('aria-label', 'Open conversations');
  document.body.append(trigger);
  window.localStorage.removeItem('dlightrag.conversation_sidebar_collapsed');
  conversationStore.openNew();
});

afterEach(async () => {
  stopFollowingConversationViews();
  document.body.replaceChildren();
  document.body.className = '';
  conversationStore.dispose();
  conversationStore.openNew();
  window.fetch = originalFetch;
  window.matchMedia = originalMatchMedia;
  window.localStorage.removeItem('dlightrag.conversation_sidebar_collapsed');
  await webRouter.navigate(newChatRoute(), {
    replace: true,
    notify: false,
    bypassGuard: true,
  });
});

it('publishes list item intent and owns menu keyboard behavior through ARIA', async () => {
  window.fetch = async () => conversationPage([first]);
  await conversationStore.loadList();
  const list = document.createElement('dl-conversation-list') as DlConversationList;
  document.body.appendChild(list);
  await list.updateComplete;

  let selected: ConversationIntentDetail | null = null;
  list.addEventListener('dl-conversation-select', (event) => { selected = event.detail; });
  button(list, 'Research notes').click();
  expect(selected).to.deep.equal({conversationId: first.conversationId});

  const actions = button(list, 'Actions for Research notes');
  actions.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'ArrowDown', bubbles: true, cancelable: true,
  }));
  await list.updateComplete;
  const menu = list.querySelector<HTMLElement>('[role="menu"][aria-label="Actions for Research notes"]');
  expect(menu).not.to.equal(null);
  expect(document.activeElement?.textContent?.trim()).to.equal('Rename');

  menu?.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Escape', bubbles: true, cancelable: true,
  }));
  await list.updateComplete;
  expect(list.querySelector('[role="menu"]')).to.equal(null);
  expect(document.activeElement).to.equal(actions);
});

it('renders JavaScript-sensitive conversation titles as inert exact text', async () => {
  const title = 'line\u2028separator\u2029</script><script>window.__conversationTitleRan=true</script>';
  const sensitive = {...first, title};
  const browser = window as Window & {__conversationTitleRan?: boolean};
  delete browser.__conversationTitleRan;
  window.fetch = async () => conversationPage([sensitive]);

  await conversationStore.loadList();
  const list = document.createElement('dl-conversation-list') as DlConversationList;
  document.body.appendChild(list);
  await list.updateComplete;

  expect(list.querySelector<HTMLButtonElement>('.conversation-select')?.textContent).to.equal(title);
  expect(list.querySelector('script')).to.equal(null);
  expect(browser.__conversationTitleRan).to.equal(undefined);
});

it('keeps accessible Load older and retry controls outside list ownership', async () => {
  const older = {...first, conversation_id: 'conversation-older', title: 'Older notes'};
  const urls: string[] = [];
  let olderAttempts = 0;
  window.fetch = async (input) => {
    const url = String(input);
    urls.push(url);
    if (!url.includes('?cursor=')) return conversationPage([first], 'opaque-next');
    olderAttempts += 1;
    return olderAttempts === 1
      ? response({detail: 'temporarily unavailable'}, 503)
      : conversationPage([older]);
  };
  await conversationStore.loadList();
  const list = document.createElement('dl-conversation-list') as DlConversationList;
  document.body.appendChild(list);
  await list.updateComplete;

  const loadOlder = button(list, 'Load older conversations');
  const ownedList = list.querySelector<HTMLElement>('[role="list"]')!;
  expect(list.getAttribute('role')).to.equal(null);
  expect([...ownedList.children].every(
    (child) => child.getAttribute('role') === 'listitem',
  )).to.equal(true);
  expect(loadOlder.closest('[role="list"]')).to.equal(null);
  expect(urls).to.deep.equal(['/web/api/conversations']);
  loadOlder.click();
  await waitFor(() => list.textContent?.includes('Could not load older conversations.') ?? false);
  const retry = button(list, 'Retry loading older conversations');
  expect(retry.closest('[role="list"]')).to.equal(null);
  expect([...ownedList.children].every(
    (child) => child.getAttribute('role') === 'listitem',
  )).to.equal(true);
  expect(retry.getAttribute('aria-busy')).to.equal('false');
  retry.click();
  await waitFor(() => list.textContent?.includes('Older notes') ?? false);

  expect(urls).to.deep.equal([
    '/web/api/conversations',
    '/web/api/conversations?cursor=opaque-next',
    '/web/api/conversations?cursor=opaque-next',
  ]);
  expect(list.querySelector('[data-load-older="conversations"]')).to.equal(null);
  expect(list.querySelector('[data-load-older-status="conversations"]')?.textContent)
    .to.contain('Loaded older conversations.');
});

it('starts rename from a double-click on the conversation title', async () => {
  window.fetch = async () => conversationPage([first]);
  await conversationStore.loadList();
  const list = document.createElement('dl-conversation-list') as DlConversationList;
  document.body.appendChild(list);
  await list.updateComplete;

  list.querySelector('[data-conversation-id]')!.dispatchEvent(new MouseEvent('dblclick', {
    bubbles: true, cancelable: true,
  }));
  await list.updateComplete;
  const input = list.querySelector<HTMLInputElement>('[aria-label="Conversation title"]');
  expect(input).not.to.equal(null);
  expect(document.activeElement).to.equal(input);
});

it('restores row focus after keyboard rename completion and cancellation', async () => {
  window.fetch = async () => conversationPage([first]);
  await conversationStore.loadList();
  const list = document.createElement('dl-conversation-list') as DlConversationList;
  document.body.appendChild(list);
  await list.updateComplete;

  const renamed: ConversationRenameDetail[] = [];
  list.addEventListener('dl-conversation-rename', (event) => { renamed.push(event.detail); });

  button(list, 'Actions for Research notes').click();
  await list.updateComplete;
  button(list, 'Rename').click();
  await list.updateComplete;
  const input = list.querySelector<HTMLInputElement>('[aria-label="Conversation title"]')!;
  input.value = 'Updated title';
  input.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Enter', bubbles: true, cancelable: true,
  }));
  await list.updateComplete;

  expect(renamed).to.deep.equal([{
    conversationId: first.conversationId,
    title: 'Updated title',
  }]);
  expect(document.activeElement).to.equal(button(list, 'Actions for Research notes'));

  button(list, 'Actions for Research notes').click();
  await list.updateComplete;
  button(list, 'Rename').click();
  await list.updateComplete;
  const cancelled = list.querySelector<HTMLInputElement>('[aria-label="Conversation title"]')!;
  cancelled.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Escape', bubbles: true, cancelable: true,
  }));
  await list.updateComplete;

  expect(renamed).to.have.length(1);
  expect(document.activeElement).to.equal(button(list, 'Actions for Research notes'));
});

it('leaves the Enter and Escape of an IME composition to the composition while a title is renamed', async () => {
  window.fetch = async () => conversationPage([first]);
  await conversationStore.loadList();
  const list = document.createElement('dl-conversation-list') as DlConversationList;
  document.body.appendChild(list);
  await list.updateComplete;
  const renamed: ConversationRenameDetail[] = [];
  list.addEventListener('dl-conversation-rename', (event) => { renamed.push(event.detail); });

  button(list, 'Actions for Research notes').click();
  await list.updateComplete;
  button(list, 'Rename').click();
  await list.updateComplete;
  const input = list.querySelector<HTMLInputElement>('[aria-label="Conversation title"]')!;
  input.value = 'yanjiu bi';
  const press = (key: string, init: KeyboardEventInit): KeyboardEvent => {
    const event = new KeyboardEvent('keydown', {key, bubbles: true, cancelable: true, ...init});
    input.dispatchEvent(event);
    return event;
  };

  // Chromium marks the keys of a composition; WebKit sends them after it ended, as keyCode 229.
  for (const init of [{isComposing: true}, {keyCode: 229}]) {
    expect(press('Enter', init).defaultPrevented, 'the Enter that commits a composition is left alone').to.equal(false);
    expect(press('Escape', init).defaultPrevented, 'the Escape that cancels a composition is left alone').to.equal(false);
    await list.updateComplete;
    expect(renamed, 'no title was committed with half a word in it').to.deep.equal([]);
    expect(list.querySelector('[aria-label="Conversation title"]') === input, 'the field is still open').to.equal(true);
  }

  // The same keys once the composition is over do rename.
  press('Enter', {});
  await list.updateComplete;
  expect(renamed).to.deep.equal([{conversationId: first.conversationId, title: 'yanjiu bi'}]);
});

it('names each conversation\'s actions after it, and keeps the plain name for one without a title', async () => {
  const untitled = {...first, conversationId: 'conversation-untitled', title: null};
  const second = {...first, conversationId: 'conversation-2', title: 'Budget review'};
  window.fetch = async () => conversationPage([first, second, untitled]);
  await conversationStore.loadList();
  const list = document.createElement('dl-conversation-list') as DlConversationList;
  document.body.appendChild(list);
  await list.updateComplete;

  const names = (): (string | null)[] => [first, second, untitled].map((conversation) => list.querySelector(
    `[data-conversation-id="${conversation.conversationId}"] .conversation-actions-button`,
  )?.getAttribute('aria-label') ?? null);
  expect(names()).to.deep.equal([
    'Actions for Research notes',
    'Actions for Budget review',
    'Conversation actions',
  ]);

  // The menu a button opens is named the way the button is.
  button(list, 'Actions for Budget review').click();
  await list.updateComplete;
  expect(list.querySelector('[role="menu"]')?.getAttribute('aria-label')).to.equal('Actions for Budget review');

  try {
    await setLanguagePreference('zh');
    await list.updateComplete;
    expect(names()).to.deep.equal([
      'Research notes 的会话操作',
      'Budget review 的会话操作',
      '会话操作',
    ]);
    expect(list.querySelector('[role="menu"]')?.getAttribute('aria-label')).to.equal('Budget review 的会话操作');
  } finally {
    await setLanguagePreference('en');
  }
});

it('owns desktop collapse state, focus, and typed Shell state', async () => {
  window.matchMedia = media(true);
  const sidebar = document.createElement('dl-conversation-sidebar') as DlConversationSidebar;
  sidebar.enabled = true;
  let state: ConversationSidebarStateDetail | null = null;
  sidebar.addEventListener('dl-conversation-sidebar-state-change', (event) => {
    state = event.detail;
  });
  document.body.appendChild(sidebar);
  await sidebar.updateComplete;

  const navigation = sidebar.querySelector<HTMLElement>('nav[aria-label="Conversations"]')!;
  expect(navigation.inert).to.equal(false);
  expect(navigation.hasAttribute('aria-hidden')).to.equal(false);
  expect(state).to.deep.equal({expanded: true, compact: false});
  expect(document.body.classList.contains('conversation-sidebar-open')).to.equal(false);

  button(sidebar, 'Collapse conversations').click();
  await sidebar.updateComplete;
  expect(navigation.inert).to.equal(true);
  expect(navigation.getAttribute('aria-hidden')).to.equal('true');
  expect(state).to.deep.equal({expanded: false, compact: false});
  expect(document.activeElement).to.equal(conversationsOpenButton());
  expect(window.localStorage.getItem('dlightrag.conversation_sidebar_collapsed')).to.equal('true');

  await sidebar.open(conversationsOpenButton());
  expect(navigation.inert).to.equal(false);
  expect(state).to.deep.equal({expanded: true, compact: false});
  expect(document.activeElement).to.equal(button(sidebar, 'New chat'));
  sidebar.shellInert = true;
  await sidebar.updateComplete;
  expect(navigation.inert).to.equal(true);
});

it('owns compact modality, a cancelable open command, focus wrapping, and Escape restore', async () => {
  window.matchMedia = media(false);
  const sidebar = document.createElement('dl-conversation-sidebar') as DlConversationSidebar;
  sidebar.enabled = true;
  let state: ConversationSidebarStateDetail | null = null;
  sidebar.addEventListener('dl-conversation-sidebar-state-change', (event) => {
    state = event.detail;
  });
  const trigger = document.createElement('button');
  trigger.textContent = 'External trigger';
  document.body.append(trigger, sidebar);
  await sidebar.updateComplete;

  sidebar.addEventListener('dl-conversation-sidebar-opening', (event) => {
    event.preventDefault();
  }, {once: true});
  expect(await sidebar.open(trigger)).to.equal(false);
  expect(state).to.deep.equal({expanded: false, compact: true});

  expect(await sidebar.open(trigger)).to.equal(true);
  const navigation = sidebar.querySelector<HTMLElement>('nav[aria-label="Conversations"]')!;
  expect(navigation.getAttribute('role')).to.equal('dialog');
  expect(navigation.getAttribute('aria-modal')).to.equal('true');
  expect(state).to.deep.equal({expanded: true, compact: true});
  expect(document.activeElement).to.equal(button(sidebar, 'New chat'));

  const settings = button(sidebar, 'Settings');
  settings.focus();
  settings.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Tab', bubbles: true, cancelable: true,
  }));
  expect(document.activeElement).to.equal(button(sidebar, 'New chat'));

  document.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Escape', bubbles: true, cancelable: true,
  }));
  await sidebar.updateComplete;
  expect(navigation.getAttribute('aria-hidden')).to.equal('true');
  expect(state).to.deep.equal({expanded: false, compact: true});
  expect(document.activeElement).to.equal(trigger);
});

it('normalizes drawer state and focus across compact and desktop breakpoints', async () => {
  await setViewport({width: 800, height: 900});
  const sidebar = document.createElement('dl-conversation-sidebar') as DlConversationSidebar;
  sidebar.enabled = true;
  let state: ConversationSidebarStateDetail | null = null;
  sidebar.addEventListener('dl-conversation-sidebar-state-change', (event) => {
    state = event.detail;
  });
  document.body.appendChild(sidebar);
  await sidebar.updateComplete;

  expect(await sidebar.open(conversationsOpenButton())).to.equal(true);
  expect(document.activeElement).to.equal(button(sidebar, 'New chat'));

  await setViewport({width: 1400, height: 900});
  await waitFor(() => state?.compact === false);
  await sidebar.updateComplete;
  const navigation = sidebar.querySelector<HTMLElement>('nav[aria-label="Conversations"]')!;
  expect(state).to.deep.equal({expanded: true, compact: false});
  expect(navigation.inert).to.equal(false);
  expect(navigation.hasAttribute('aria-modal')).to.equal(false);
  expect(document.activeElement).to.equal(button(sidebar, 'New chat'));

  await setViewport({width: 800, height: 900});
  await waitFor(() => state?.compact === true);
  await sidebar.updateComplete;
  expect(state).to.deep.equal({expanded: false, compact: true});
  expect(navigation.inert).to.equal(true);
  expect(navigation.hasAttribute('aria-modal')).to.equal(false);
  expect(document.activeElement).to.equal(conversationsOpenButton());
});

it('asks the router to move when a conversation or a new chat is chosen, and nothing else', async () => {
  window.matchMedia = media(true);
  window.fetch = async () => conversationPage([first]);
  await conversationStore.loadList();
  const sidebar = document.createElement('dl-conversation-sidebar') as DlConversationSidebar;
  sidebar.enabled = true;
  document.body.appendChild(sidebar);
  await sidebar.updateComplete;

  button(sidebar, 'Research notes').click();
  await waitFor(() => webRouter.current.kind === 'conversation');
  expect(webRouter.current).to.deep.equal(conversationRoute(first.conversationId));

  button(sidebar, 'New chat').click();
  await waitFor(() => webRouter.current.kind === 'new');
});

it('puts the compact drawer away once a conversation is chosen, and hands the reader the box', async () => {
  window.matchMedia = media(false);
  window.fetch = async () => conversationPage([first]);
  await conversationStore.loadList();
  const sidebar = document.createElement('dl-conversation-sidebar') as DlConversationSidebar;
  sidebar.enabled = true;
  let chosen = 0;
  sidebar.addEventListener('dl-conversation-chosen', () => { chosen += 1; });
  document.body.appendChild(sidebar);
  await sidebar.updateComplete;
  expect(await sidebar.open(conversationsOpenButton())).to.equal(true);
  expect(sidebar.querySelector('nav')?.getAttribute('role')).to.equal('dialog');

  button(sidebar, 'Research notes').click();
  await waitFor(() => chosen === 1);

  expect(sidebar.querySelector('nav')?.hasAttribute('role'), 'the drawer was put away').to.equal(false);
  expect(webRouter.current).to.deep.equal(conversationRoute(first.conversationId));
});

it('leaves the Escape that cancels an IME composition to the composition, and closes the drawer for the next one', async () => {
  window.matchMedia = media(false);
  const sidebar = document.createElement('dl-conversation-sidebar') as DlConversationSidebar;
  sidebar.enabled = true;
  const trigger = document.createElement('button');
  document.body.append(trigger, sidebar);
  await sidebar.updateComplete;
  expect(await sidebar.open(trigger)).to.equal(true);

  // Chromium marks the key of a composition; WebKit sends it after it ended, as keyCode 229.
  for (const init of [{isComposing: true}, {keyCode: 229}]) {
    const cancelled = new KeyboardEvent('keydown', {key: 'Escape', bubbles: true, cancelable: true, ...init});
    document.dispatchEvent(cancelled);
    await sidebar.updateComplete;
    expect(cancelled.defaultPrevented, 'the Escape of a composition was taken').to.equal(false);
    expect(sidebar.querySelector('nav')?.getAttribute('role'), 'the drawer closed under it').to.equal('dialog');
  }

  document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true, cancelable: true}));
  await sidebar.updateComplete;
  expect(sidebar.querySelector('nav')?.hasAttribute('role')).to.equal(false);
});

it('keeps route navigation available while an independent rename settles', async () => {
  window.matchMedia = media(true);
  let resolveRename: ((response: Response) => void) | undefined;
  window.fetch = async (input, init) => {
    const url = String(input);
    if (url === '/web/api/conversations') return conversationPage([first]);
    if (url.endsWith(`/${first.conversationId}/history`)) {
      return response({conversation: firstWire, turns: [], commands: [], next_cursor: null});
    }
    if (url.endsWith(`/${first.conversationId}`) && init?.method === 'PATCH') {
      return await new Promise<Response>((resolve) => { resolveRename = resolve; });
    }
    return response({detail: 'not found'}, 404);
  };
  const chat = chatStub();
  const sidebar = document.createElement('dl-conversation-sidebar') as DlConversationSidebar;
  const routes = followConversationView(chat);
  sidebar.routes = routes;
  sidebar.enabled = true;
  document.body.appendChild(sidebar);
  routes.start();
  await waitFor(() => sidebar.querySelector('[role="listitem"]') !== null);

  button(sidebar, 'Research notes').click();
  await waitFor(() => chat.view.kind === 'ready');
  button(sidebar, 'Actions for Research notes').click();
  await sidebar.updateComplete;
  button(sidebar, 'Rename').click();
  await sidebar.updateComplete;
  const input = sidebar.querySelector<HTMLInputElement>('[aria-label="Conversation title"]')!;
  input.value = 'Renaming';
  input.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Enter', bubbles: true, cancelable: true,
  }));
  await waitFor(() => resolveRename !== undefined);

  expect(await webRouter.navigate(newChatRoute())).to.equal(true);
  await waitFor(() => chat.view.kind === 'new');
  const finishRename = resolveRename;
  if (!finishRename) throw new Error('rename request was not started');
  finishRename(response({...first, title: 'Renaming'}));
  await new Promise((resolve) => setTimeout(resolve, 0));
  expect(chat.view.kind).to.equal('new');
});

it('holds the page still while a deletion is under way, and lets it move when it is done', async () => {
  window.matchMedia = media(true);
  let finishDelete: (() => void) | undefined;
  window.fetch = async (input, init) => {
    if (String(input) === '/web/api/conversations' && init?.method === 'DELETE') {
      await new Promise<void>((resolve) => { finishDelete = resolve; });
      return new Response(null, {status: 204});
    }
    if (String(input) === '/web/api/conversations') return conversationPage([]);
    return response({detail: 'not found'}, 404);
  };
  const chat = chatStub();
  const sidebar = document.createElement('dl-conversation-sidebar') as DlConversationSidebar;
  const routes = followConversationView(chat);
  sidebar.routes = routes;
  sidebar.enabled = true;
  document.body.appendChild(sidebar);
  routes.start();
  await sidebar.updateComplete;

  const deletion = sidebar.deleteAll();
  await waitFor(() => Boolean(sidebar.querySelector<HTMLDialogElement>(
    '#delete-all-conversations-dialog',
  )?.open));
  button(sidebar, 'Delete all').click();
  await waitFor(() => finishDelete !== undefined);

  expect(routes.lifecycleInFlight).to.equal(true);
  expect(await webRouter.navigate(conversationRoute('elsewhere'))).to.equal(false);
  expect(button(sidebar, 'New chat').disabled, 'and the controls that start another wait').to.equal(true);

  finishDelete!();
  expect(await deletion).to.equal(true);
  expect(routes.lifecycleInFlight).to.equal(false);
  expect(await webRouter.navigate(newChatRoute())).to.equal(true);
});

it('tells the Shell the conversation is gone after delete-all succeeds on the new route', async () => {
  window.matchMedia = media(true);
  window.fetch = async (input, init) => {
    if (String(input) === '/web/api/conversations' && init?.method === 'DELETE') {
      return new Response(null, {status: 204});
    }
    if (String(input) === '/web/api/conversations') return conversationPage([]);
    return response({detail: 'not found'}, 404);
  };
  const chat = chatStub();
  const sidebar = document.createElement('dl-conversation-sidebar') as DlConversationSidebar;
  let left = 0;
  sidebar.routes = followConversationView(chat, {ui: {conversationLeft: () => { left += 1; }}});
  sidebar.enabled = true;
  document.body.appendChild(sidebar);
  await sidebar.updateComplete;

  const deletion = sidebar.deleteAll();
  await waitFor(() => Boolean(sidebar.querySelector<HTMLDialogElement>(
    '#delete-all-conversations-dialog',
  )?.open));
  button(sidebar, 'Delete all').click();

  expect(await deletion).to.equal(true);
  expect(left).to.equal(1);
});

it('tells the Shell too when delete-all closes the conversation that was open, which stops its Run and draft', async () => {
  window.matchMedia = media(true);
  window.fetch = async (input, init) => {
    const url = String(input);
    if (url === '/web/api/conversations' && init?.method === 'DELETE') {
      return new Response(null, {status: 204});
    }
    if (url === '/web/api/conversations') return conversationPage([first]);
    if (url.endsWith(`/${first.conversationId}/history`)) {
      return response({conversation: firstWire, turns: [], commands: [], next_cursor: null});
    }
    return response({detail: 'not found'}, 404);
  };
  const chat = chatStub();
  const sidebar = document.createElement('dl-conversation-sidebar') as DlConversationSidebar;
  let left = 0;
  const routes = followConversationView(chat, {ui: {conversationLeft: () => { left += 1; }}});
  sidebar.routes = routes;
  sidebar.enabled = true;
  document.body.appendChild(sidebar);
  routes.start();
  await waitFor(() => sidebar.querySelector('[role="listitem"]') !== null);
  button(sidebar, 'Research notes').click();
  await waitFor(() => chat.view.kind === 'ready');
  expect(left, 'opening it left the new chat').to.equal(1);
  const detachedBefore = chat.detached;

  const deletion = sidebar.deleteAll();
  await waitFor(() => Boolean(sidebar.querySelector<HTMLDialogElement>(
    '#delete-all-conversations-dialog',
  )?.open));
  button(sidebar, 'Delete all').click();

  expect(await deletion).to.equal(true);
  expect(chat.detached, 'the Run was let go before the conversations were').to.equal(detachedBefore + 1);
  expect(chat.cleared, 'the draft that was written for it went with it').to.equal(1);
  expect(left, 'what belonged to the conversation went with it').to.equal(2);
  expect(chat.view.kind).to.equal('new');
});

it('aborts a confirmed mutation and ignores its result after disconnect', async () => {
  window.matchMedia = media(true);
  let mutationStarted = false;
  let mutationAborted = false;
  window.fetch = async (input, init) => {
    if (String(input) === '/web/api/conversations' && init?.method === 'DELETE') {
      mutationStarted = true;
      return await new Promise<Response>((_resolve, reject) => {
        init.signal?.addEventListener('abort', () => {
          mutationAborted = true;
          reject(new DOMException('Aborted', 'AbortError'));
        }, {once: true});
      });
    }
    if (String(input) === '/web/api/conversations') return conversationPage([]);
    return response({detail: 'not found'}, 404);
  };
  const chat = chatStub();
  const sidebar = document.createElement('dl-conversation-sidebar') as DlConversationSidebar;
  let routeResets = 0;
  sidebar.routes = followConversationView(chat, {ui: {conversationLeft: () => { routeResets += 1; }}});
  sidebar.enabled = true;
  document.body.appendChild(sidebar);
  await sidebar.updateComplete;

  const deletion = sidebar.deleteAll();
  await waitFor(() => Boolean(sidebar.querySelector<HTMLDialogElement>(
    '#delete-all-conversations-dialog',
  )?.open));
  button(sidebar, 'Delete all').click();
  await waitFor(() => mutationStarted);
  sidebar.remove();

  expect(await deletion).to.equal(false);
  expect(mutationAborted).to.equal(true);
  expect(conversationStore.mutationPending).to.equal(false);
  expect(chat.cleared).to.equal(0);
  expect(routeResets).to.equal(0);
});

it('aborts optional memory clearing and ignores its result after disconnect', async () => {
  window.matchMedia = media(true);
  let memoryStarted = false;
  let memoryAborted = false;
  window.fetch = async (input, init) => {
    const url = String(input);
    if (url === '/web/api/conversations' && init?.method === 'DELETE') {
      return new Response(null, {status: 204});
    }
    if (url === '/web/api/memory/clear' && init?.method === 'POST') {
      memoryStarted = true;
      return await new Promise<Response>((_resolve, reject) => {
        init.signal?.addEventListener('abort', () => {
          memoryAborted = true;
          reject(new DOMException('Aborted', 'AbortError'));
        }, {once: true});
      });
    }
    if (url === '/web/api/conversations') return conversationPage([]);
    return response({detail: 'not found'}, 404);
  };
  const chat = chatStub();
  const toasts: unknown[] = [];
  const onToast = (event: Event): void => { toasts.push((event as CustomEvent).detail); };
  document.addEventListener('dl-toast-request', onToast);
  const focusSentinel = document.createElement('button');
  document.body.appendChild(focusSentinel);
  const sidebar = document.createElement('dl-conversation-sidebar') as DlConversationSidebar;
  let routeResets = 0;
  sidebar.routes = followConversationView(chat, {ui: {conversationLeft: () => { routeResets += 1; }}});
  sidebar.enabled = true;
  document.body.appendChild(sidebar);
  await sidebar.updateComplete;

  const deletion = sidebar.deleteAll();
  await waitFor(() => Boolean(sidebar.querySelector<HTMLDialogElement>(
    '#delete-all-conversations-dialog',
  )?.open));
  const clearProfileMemories = sidebar.querySelector<HTMLInputElement>(
    '#delete-all-also-clear-memory',
  );
  if (!clearProfileMemories) throw new Error('memory clearing checkbox was not rendered');
  clearProfileMemories.checked = true;
  button(sidebar, 'Delete all').click();
  await waitFor(() => memoryStarted);
  sidebar.remove();
  focusSentinel.focus();

  expect(await deletion).to.equal(false);
  expect(memoryAborted).to.equal(true);
  expect(conversationStore.mutationPending).to.equal(false);
  expect(chat.cleared).to.equal(1);
  expect(routeResets).to.equal(0);
  document.removeEventListener('dl-toast-request', onToast);
  expect(toasts, 'a deletion that was torn down tells nobody it failed').to.deep.equal([]);
  expect(document.activeElement).to.equal(focusSentinel);
});

it('cancels a pending draft dialog on disconnect without wedging later navigation', async () => {
  window.matchMedia = media(true);
  window.fetch = async (input) => {
    const url = String(input);
    if (url === '/web/api/conversations') return conversationPage([first]);
    if (url.endsWith(`/${first.conversationId}/history`)) {
      return response({conversation: firstWire, turns: [], commands: [], next_cursor: null});
    }
    return response({detail: 'not found'}, 404);
  };
  const chat = chatStub({hasDraft: true});
  const sidebar = document.createElement('dl-conversation-sidebar') as DlConversationSidebar;
  const routes = followConversationView(chat, {
    ui: {confirmDiscardDraft: (next) => sidebar.confirmDiscardDraft(next)},
  });
  sidebar.routes = routes;
  sidebar.enabled = true;
  document.body.appendChild(sidebar);
  routes.start();
  await waitFor(() => sidebar.querySelector('[role="listitem"]') !== null);

  const blockedNavigation = webRouter.navigate(conversationRoute(first.conversationId));
  await waitFor(() => Boolean(sidebar.querySelector<HTMLDialogElement>(
    '#discard-draft-dialog',
  )?.open));
  sidebar.remove();
  expect(await blockedNavigation).to.equal(false);
  expect(chat.cleared, 'the draft was kept: nobody agreed to lose it').to.equal(0);

  // Once the reader has nothing unsent, the same move goes through.
  chat.hasDraft = false;
  expect(await webRouter.navigate(conversationRoute(first.conversationId))).to.equal(true);
  await waitFor(() => chat.view.kind === 'ready');
});
