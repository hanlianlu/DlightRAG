// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Explicit store bag the Shell constructs once and passes to Features.

 *  Primitives never receive this. Features do not import store singletons.
 */

import {answerEventCursorStore, AnswerEventCursorStore} from './answer-event-cursor-store.ts';
import {attachmentStore, AttachmentStore} from './attachment-store.ts';
import {conversationStore, ConversationStore} from './conversation-store.ts';
import {ingestStore, IngestStore} from './ingest-store.ts';
import {workspaceStore, WorkspaceStore} from './workspace-store.ts';

export interface AppHandles {
  readonly conversations: ConversationStore;
  readonly workspaces: WorkspaceStore;
  readonly ingest: IngestStore;
  readonly attachments: AttachmentStore;
  readonly answerEventCursors: AnswerEventCursorStore;
}

let produced: AppHandles | null = null;

/** The process-wide bag wrapping today's store instances.

 *  Tests may pass a different bag into a Feature. Production Shell assigns
 *  this object once so every Feature sees the same stores. */
export function productionHandles(): AppHandles {
  produced ??= {
    conversations: conversationStore,
    workspaces: workspaceStore,
    ingest: ingestStore,
    attachments: attachmentStore,
    answerEventCursors: answerEventCursorStore,
  };
  return produced;
}

export function createAppHandles(overrides: Partial<AppHandles> = {}): AppHandles {
  return {
    conversations: overrides.conversations ?? new ConversationStore(),
    workspaces: overrides.workspaces ?? new WorkspaceStore(),
    ingest: overrides.ingest ?? new IngestStore(),
    attachments: overrides.attachments ?? new AttachmentStore(),
    answerEventCursors: overrides.answerEventCursors ?? new AnswerEventCursorStore(),
  };
}
