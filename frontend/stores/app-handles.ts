// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The bag of the app's shared stores.

 *  createAppHandles() is the only constructor of the shared stores, and
 *  productionHandles() holds the page's one bag; no store module holds an
 *  instance of its own. Primitives never receive this bag.
 */

import {AnswerEventCursorStore} from './answer-event-cursor-store.ts';
import {AnswerSubmissionRegistry} from './answer-submission-registry.ts';
import {AttachmentStore} from './attachment-store.ts';
import {KeptTraces} from './child-observation-cache.ts';
import {ConversationStore} from './conversation-store.ts';
import {IngestStore} from './ingest-store.ts';
import {KeptJudgements} from './kept-judgements.ts';
import {MemoryNotices} from './memory-notices.ts';
import {RunActivityStore} from './run-activity-store.ts';
import {ScheduleStore} from './schedule-store.ts';
import {WorkspaceStore} from './workspace-store.ts';

export interface AppHandles {
  readonly conversations: ConversationStore;
  readonly workspaces: WorkspaceStore;
  readonly ingest: IngestStore;
  readonly attachments: AttachmentStore;
  readonly answerEventCursors: AnswerEventCursorStore;
  /** The answers this page is submitting or that wait on the reader, one per conversation. */
  readonly submissions: AnswerSubmissionRegistry;
  /** What this page last saw the server keep of the reader's judgements. */
  readonly judgements: KeptJudgements;
  /** The child pages each Run's dock has read. */
  readonly traces: KeptTraces;
  /** What each answer's memory notice has read of the memory its Run left. */
  readonly memoryNotices: MemoryNotices;
  /** The reader's recent Run activity: which conversations are running, and which have an answer
   *  they have not opened. */
  readonly runActivity: RunActivityStore;
  /** The reader's Schedules, which the sidebar's Scheduled section and its dialogs share. */
  readonly schedules: ScheduleStore;
  /** Forget what the signed-in reader sent, rated, and read, as signing out must: whoever is shown this
   *  page next is shown none of it. Their unsent message is the composer's own to forget. */
  reset(): void;
}

let produced: AppHandles | null = null;

/** The process-wide bag, constructed on first use.

 *  Every Feature defaults to it, so the Shell and the Features it composes
 *  share one set of stores; a test passes a different bag into a Feature. */
export function productionHandles(): AppHandles {
  produced ??= createAppHandles();
  return produced;
}

/** Construct a complete bag of new stores. */
export function createAppHandles(): AppHandles {
  const workspaces = new WorkspaceStore();
  const submissions = new AnswerSubmissionRegistry();
  const judgements = new KeptJudgements();
  const traces = new KeptTraces();
  const memoryNotices = new MemoryNotices();
  const runActivity = new RunActivityStore();
  const schedules = new ScheduleStore();
  return {
    conversations: new ConversationStore(),
    workspaces,
    ingest: new IngestStore(workspaces),
    attachments: new AttachmentStore(),
    answerEventCursors: new AnswerEventCursorStore(),
    submissions,
    judgements,
    traces,
    memoryNotices,
    runActivity,
    schedules,
    reset: () => {
      submissions.dispose();
      judgements.forget();
      traces.forget();
      memoryNotices.forget();
      runActivity.forget();
      schedules.forget();
    },
  };
}
