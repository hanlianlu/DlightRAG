// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Explicit store bag the Shell constructs once and passes to Features.

 *  createAppHandles() is the only constructor of the app's shared stores;
 *  no store module holds an instance. Primitives never receive this bag.
 */

import {AnswerEventCursorStore} from './answer-event-cursor-store.ts';
import {AttachmentStore} from './attachment-store.ts';
import {ConversationStore} from './conversation-store.ts';
import {IngestStore} from './ingest-store.ts';
import {WorkspaceStore} from './workspace-store.ts';

export interface AppHandles {
  readonly conversations: ConversationStore;
  readonly workspaces: WorkspaceStore;
  readonly ingest: IngestStore;
  readonly attachments: AttachmentStore;
  readonly answerEventCursors: AnswerEventCursorStore;
}

let produced: AppHandles | null = null;

/** The process-wide bag, constructed on first use.

 *  Every Feature defaults to it, so the Shell and the Features it composes
 *  share one set of stores; tests may pass a different bag into a Feature. */
export function productionHandles(): AppHandles {
  produced ??= createAppHandles();
  return produced;
}

/** Construct a complete bag of new stores. */
export function createAppHandles(): AppHandles {
  const workspaces = new WorkspaceStore();
  return {
    conversations: new ConversationStore(),
    workspaces,
    ingest: new IngestStore(workspaces),
    attachments: new AttachmentStore(),
    answerEventCursors: new AnswerEventCursorStore(),
  };
}
