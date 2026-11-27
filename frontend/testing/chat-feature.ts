// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** What the chat-feature browser tests share: the wire fixtures, the stores the page keeps, and the cleanup
 *  a test of the feature owes the next one. */

import type {AcceptedAnswer, AnswerPresentation, ConversationSummary, ConversationTurn} from '../api/conversations.ts';
import {defineDesignSystemElements} from '../design-system/index.ts';
import {productionHandles} from '../stores/app-handles.ts';
import '../ui/chat-composer.ts';
import type {DlChatFeature} from '../ui/chat-feature.ts';
import '../ui/chat-feature.ts';
import {stopFollowingConversationViews} from './conversation-view.ts';

export const {
  attachments: attachmentStore,
  conversations: conversationStore,
  submissions,
  workspaces: workspaceStore,
} = productionHandles();

defineDesignSystemElements();

const originalFetch = window.fetch;

export const originalRevokeObjectURL = URL.revokeObjectURL;

/** Where the composer keeps the words that are not sent yet. */
export const DRAFT_KEY = 'dlightrag.composerDraft';

export const policy = {
  countLimit: 6,
  imageMaxBytes: 1024,
  documentMaxBytes: 2048,
  extensions: new Set(['md', 'pdf']),
  imageCapability: 'supported' as const,
  imageLimit: 3,
};

export const presentation: AnswerPresentation = {
  answerText: 'A stored answer.',
  parts: [{
    type: 'markdown',
    text: 'A stored answer.',
    html: '<p>A stored answer.</p>',
    artifact: null,
    evidenceImage: null,
    inline: false,
  }],
  sources: [],
  evidenceImages: [], linkCards: [],
  artifacts: [],
  artifactOutcome: {status: 'complete', issues: []},
};

export function storedTurn(): ConversationTurn {
  return {
    turnId: 'turn-1',
    turnNumber: 1,
    answerRunId: 'run-1',
    submissionId: 'submission-1',
    status: 'succeeded',
    cancelRequested: false,
    userText: 'Question',
    assistantText: presentation.answerText,
    userAttachments: [],
    steeringMessages: [],
    presentation,
    usage: {},
    errorKind: null,
    errorMessage: null,
    childCount: 0,
    answerSequence: null,
    feedback: null,
    createdAt: '2026-01-01T00:00:00Z',
  };
}

export function continuationDescriptor(conversationId: string): AcceptedAnswer {
  return {
    kind: 'accepted',
    conversation: {
      conversationId: conversationId,
      title: 'Continuation',
      createdAt: '2026-01-01T00:00:00Z',
      updatedAt: '2026-01-01T00:00:00Z',
    forkedFromConversationId: null,
    forkedFromTitle: null,
    schedule: null,
    },
    turn: {
      ...storedTurn(),
      answerRunId: `run-${conversationId}`,
      turnId: `turn-${conversationId}`,
      submissionId: `submission-${conversationId}`,
      status: 'queued',
    },
  };
}

// What the server actually puts on the wire; these mappers mirror the
// snake_case REST payloads back out of parsed camelCase domain fixtures.
export function conversationWire(conversation: ConversationSummary): Record<string, unknown> {
  return {
    conversation_id: conversation.conversationId,
    title: conversation.title,
    created_at: conversation.createdAt,
    updated_at: conversation.updatedAt,
    forked_from_conversation_id: conversation.forkedFromConversationId,
    forked_from_title: conversation.forkedFromTitle,
    schedule: conversation.schedule && {schedule_id: conversation.schedule.scheduleId, name: conversation.schedule.name},
  };
}

export async function settle(element: DlChatFeature): Promise<void> {
  await element.updateComplete;
  await element.querySelector('dl-chat-message-list')?.updateComplete;
  await element.querySelector('dl-chat-composer')?.updateComplete;
  await element.querySelector('dl-answer-presentation')?.updateComplete;
}

/** Put back what a test of the chat feature leaves behind, after each test of the file that calls it. */
export function restoreChatFeatureFixtures(): void {
  afterEach(() => {
    window.fetch = originalFetch;
    URL.revokeObjectURL = originalRevokeObjectURL;
    stopFollowingConversationViews();
    submissions.dispose();
    attachmentStore.clear();
    conversationStore.openNew();
    document.body.replaceChildren();
    localStorage.removeItem('dlightrag.answerMode');
    // Words a test left in the box would come back in the next composer.
    localStorage.removeItem(DRAFT_KEY);
  });
}
