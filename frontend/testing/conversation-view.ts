// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** What a test needs of the Shell to give a Chat a conversation: the route controller `dl-app` owns, set up
 *  on a host that never renders, and a Chat told the view the controller holds for the conversation store
 *  each time the store publishes a new one. A test opens conversations in the store, or navigates the router
 *  once it starts the controller, and watches the Chat draw them. */

import type {ReactiveControllerHost} from 'lit';
import type {AppHandles} from '../stores/app-handles.ts';
import {productionHandles} from '../stores/app-handles.ts';
import {
  ConversationRouteController,
  type ConversationRouteChat,
  type ConversationRouteUi,
} from '../ui/conversation-route-controller.ts';
import type {ChatView} from '../ui/chat-message-list.ts';

/** A Chat as the route meets it, with the view it is told. `dl-chat-feature` is one. */
export type FollowedChat = ConversationRouteChat & {view: ChatView};

interface FollowOptions {
  handles?: AppHandles;
  /** What the route asks the rest of the Shell; anything left out is answered as a reader who agrees. */
  ui?: Partial<ConversationRouteUi>;
}

const cleanups: Array<() => void> = [];

/** A host that never renders: the controller only needs something to attach to, and something to ask for
 *  a redraw, which here is telling the Chat. */
function inertHost(redraw: () => void): ReactiveControllerHost {
  return {
    addController: () => undefined,
    removeController: () => undefined,
    requestUpdate: redraw,
    updateComplete: Promise.resolve(true),
  };
}

/** Tell `chat` the view the store holds now, and each time it publishes another. The controller is
 *  connected but not started: a test that wants the router calls `start()`. */
export function followConversationView(
  chat: FollowedChat,
  {handles = productionHandles(), ui = {}}: FollowOptions = {},
): ConversationRouteController {
  const tell = (): void => { chat.view = routes.view; };
  const routes = new ConversationRouteController(inertHost(tell), {
    handles: () => handles,
    chat: () => chat,
    ui: {
      confirmDiscardDraft: async () => true,
      waitForAcceptedQuestion: () => undefined,
      conversationLeft: () => undefined,
      closeDrawer: async () => undefined,
      ...ui,
    },
  });
  routes.hostConnected();
  tell();
  cleanups.push(handles.conversations.subscribe(tell), () => { routes.hostDisconnected(); });
  return routes;
}

/** Stop telling every Chat followed so far, and let go of the router, as a test does when it is over. */
export function stopFollowingConversationViews(): void {
  for (const cleanup of cleanups.splice(0)) cleanup();
}
