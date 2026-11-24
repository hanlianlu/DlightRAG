// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg} from '@lit/localize';
import {html, type PropertyValues, type TemplateResult} from 'lit';
import {waitFor} from 'xstate';
import type {AgentEffort} from '../api/agent-effort.ts';
import type {AnswerDelivery, AnswerMode} from '../api/answer-request.ts';
import {BrowserAnswerSubmissionAdapter} from '../api/answer-submission.ts';
import {
  type ConversationTurn,
  createConversation,
  forkAnswerRun,
  postConversationCommand,
} from '../api/conversations.ts';
import type {MemoryOperationEvent} from '../api/memory.ts';
import {runActive} from '../api/run-activity.ts';
import {AnswerSubmissionError} from '../api/web-command-error.ts';
import type {AgentEffortOffer} from '../lib/agent-effort.ts';
import {EMPTY_AGENT_EFFORT_OFFER} from '../lib/agent-effort.ts';
import type {AttachmentPolicy} from '../lib/attachment-policy.ts';
import {raise} from '../lib/dom.ts';
import {goalWasWritten} from '../lib/goal-activity.ts';
import {findWebCommand, type CommandInvocation} from '../lib/web-commands.ts';
import {LatestRequest} from '../lib/latest-request.ts';
import {LightElement, StoreController} from '../lib/lit-host.ts';
import {catalogText, type LocalizedText, localizedText, serverText } from '../lib/localized-text.ts';
import {conversationRoute} from '../lib/router.ts';
import {
  type AnswerRunEvent,
  type FollowResult,
  RunController,
  statesSomething,
} from '../lib/run-controller.ts';
import {errorKindText} from '../lib/run-errors.ts';
import {applyAnswerEvent} from '../lib/turn-projection.ts';
import {
  type AnswerSubmissionActor,
  answerSubmissionSnapshot,
} from '../stores/answer-submission-machine.ts';
import {type AppHandles, productionHandles } from '../stores/app-handles.ts';
import type {PendingAttachment} from '../stores/attachment-store.ts';
import {runTracesSource, type TracesSource} from '../stores/traces-source.ts';
import type {
  ComposerSubmitDetail,
  DlChatComposer,
} from './chat-composer.ts';
import './chat-composer.ts';
import './goal-bar.ts';
import type {DlGoalBar} from './goal-bar.ts';
import './queue-strip.ts';
import type {DlQueueStrip} from './queue-strip.ts';
import {
  ANSWER_RECONNECT_COPY,
  answerReconnectState,
  type ChatReconnectDetail,
  type ChatRetryTurnDetail,
  type ChatView,
  MAX_STEERING_MESSAGES,
  storedTurnView,
} from './chat-message-list.ts';
import './chat-message-list.ts';
import {
  type ChatCommandView,
  chatCommandView,
  type ChatTurnView,
  isTerminalTurnState,
} from '../lib/chat-views.ts';
import {webRouter} from './router.ts';
import {requestToast} from './toast-request.ts';

/** Why a command failed: kind copy, the server's reason, or the fallback, kept as
 *  the source a view can hold so a language change redraws it. */
function commandError(error: unknown, fallback: LocalizedText): LocalizedText {
  if (!(error instanceof AnswerSubmissionError)) return fallback;
  return errorKindText(error.errorKind)
    ?? (error.message ? serverText(error.message) : fallback);
}

/** The words one submission carried, from the actor's own intent. */
function queryOf(actor: AnswerSubmissionActor): string {
  return actor.getSnapshot().context.intent.query;
}

function submissionError(error: AnswerSubmissionError | null): LocalizedText {
  return commandError(
    error,
    catalogText('The answer could not be submitted.', 'chatFeature.submissionFailed'),
  );
}

/** The submission states that wait on the reader: the banner offers Retry, Edit, and Discard. */
const SUBMISSION_BANNER_STATUSES: readonly string[] = ['editable', 'retryable', 'conflict'];

/** A message the feature took: the submission to watch, and the turn that shows it meanwhile. */
interface AcceptedSubmission {
  actor: AnswerSubmissionActor;
  turnId: string;
  conversationId: string | null;
}

export interface ChatRunningChangeDetail {
  active: boolean;
}

export interface ChatContentChangeDetail {
  hasMessages: boolean;
}

export interface ChatRunActivityDetail {
  runId: string;
  /** The Run's own events, as the batch they arrived in: an open dock reads its pages for these. */
  events: readonly AnswerRunEvent[];
}

/** The turn as the reader sees it the moment they send it, before the server has accepted anything. */
function optimisticTurn(
  submissionId: string,
  query: string,
  attachments: readonly PendingAttachment[],
): ChatTurnView {
  return {
    id: `local-${submissionId}`,
    userText: query,
    userAttachments: attachments.map((item, index) => ({
      attachmentId: item.id,
      ordinal: index + 1,
      kind: item.kind,
      filename: item.file.name,
      mimeType: item.file.type,
      byteSize: item.file.size,
      url: item.objectUrl,
      thumbnailUrl: item.objectUrl,
      label: item.file.name,
    })),
    runId: '',
    state: 'pending',
    streamText: '',
    processEntries: [],
    presentation: null,
    usage: {},
    error: null,
    activity: null,
    cancelRequested: false,
    steeringMessages: [],
    toolRows: [],
    answerSequence: null,
    feedback: null,
  };
}

/** Chat composition root: owns submission, following, replay, and run intent. */
export class DlChatFeature extends LightElement {
  static properties = {
    handles: {attribute: false},
    view: {attribute: false},
    attachmentPolicy: {attribute: false},
    attachmentAccept: {type: String},
    agentEffortOffer: {attribute: false},
    answerFeedbackEnabled: {attribute: false},
    interactionLocked: {attribute: false},
    turns: {state: true},
    commands: {state: true},
    runRevision: {state: true},
  };

  declare handles: AppHandles;
  declare view: ChatView;
  declare attachmentPolicy: AttachmentPolicy | null;
  declare attachmentAccept: string;
  declare agentEffortOffer: AgentEffortOffer;
  /** Whether the reader is offered the thumbs under a settled answer. */
  declare answerFeedbackEnabled: boolean;
  declare interactionLocked: boolean;
  declare turns: readonly ChatTurnView[];
  /** The server-executed commands of the conversation on show, which the message list interleaves
   *  with the turns by sequence. */
  declare commands: readonly ChatCommandView[];
  declare runRevision: number;

  readonly #runController: RunController;
  readonly #submissionAdapter = new BrowserAnswerSubmissionAdapter();
  #submissionActor: AnswerSubmissionActor | null = null;
  #submissionTurnId: string | null = null;
  readonly #continuation = new LatestRequest();
  #pendingResume: {conversationId: string; stored: ConversationTurn} | null = null;
  #scrollRequest = 0;
  #announcedActive = false;
  #announcedHasMessages: boolean | null = null;
  /** Whether the conversation has a live Run the server knows about, followed here or not. */
  #conversationLive = false;
  #stopRunActivity: (() => void) | null = null;
  #attachRequested = false;

  constructor() {
    super();
    this.handles = productionHandles();
    this.view = {kind: 'new'};
    this.attachmentPolicy = null;
    this.attachmentAccept = '';
    this.agentEffortOffer = EMPTY_AGENT_EFFORT_OFFER;
    this.answerFeedbackEnabled = false;
    this.interactionLocked = false;
    this.turns = [];
    this.commands = [];
    this.runRevision = 0;
    this.#runController = new RunController({
      onStateChange: () => this.#runStateChanged(),
      cursorStore: () => this.handles.answerEventCursors,
    });
    // Submissions: list(), actor(), start().
    new StoreController(this, () => [this.handles.submissions]);
  }

  get submissionPending(): boolean {
    return this.handles.submissions.list().some(
      ({status}) => status === 'submitting' || status === 'reconciling',
    );
  }

  get hasUnresolvedSubmission(): boolean {
    return this.handles.submissions.list().length > 0;
  }

  get hasDraft(): boolean {
    return this.#composer()?.hasDraft ?? this.handles.attachments.size > 0;
  }

  /** Files the reader staged, which live in memory alone; a typed draft outlives a reload. */
  get hasStagedAttachments(): boolean {
    return this.handles.attachments.size > 0;
  }

  override connectedCallback(): void {
    super.connectedCallback();
    // The server knows when a Run is live, including a goal round this page did not start: it is
    // the one fact the composer's queue/steer choice and the attach path both read.
    this.#stopRunActivity = this.handles.runActivity.subscribe(() => this.#activityChanged());
    this.#activityChanged();
  }

  override disconnectedCallback(): void {
    super.disconnectedCallback();
    this.#continuation.cancel();
    this.#runController.disconnect();
    this.#stopRunActivity?.();
    this.#stopRunActivity = null;
  }

  detachRun(): void {
    this.#continuation.cancel();
    this.#runController.detach();
  }

  clearDraft(): void {
    this.#composer()?.clearDraft();
    if (!this.#composer()) this.handles.attachments.clear();
  }

  focusComposer(): void {
    this.#composer()?.focusInput();
  }

  /** The turn on show that a Run answers. */
  #turnOfRun(runId: string): ChatTurnView | undefined {
    return this.turns.find((turn) => turn.runId === runId);
  }

  /** The commands the message list draws: the store's for the conversation on show, and none while
   *  the view names no loaded conversation. */
  #commandsForView(): readonly ChatCommandView[] {
    const conversationId = this.view.kind === 'ready' ? this.view.conversationId : null;
    if (conversationId === null
      || this.handles.conversations.activeConversationId !== conversationId) {
      return [];
    }
    return this.handles.conversations.commands.map(chatCommandView);
  }

  /** What the Inspector's Run traces dock reads for a Run this chat shows. The dock's reads go
   *  straight to the server; the chat adds the turn the Run answers. */
  tracesSource(runId: string): TracesSource {
    return runTracesSource(runId, {turn: () => this.#turnOfRun(runId)});
  }

  async forkRun(
    runId: string,
    query: string,
  ): Promise<void> {
    const signal = this.#continuation.start();
    try {
      const descriptor = await forkAnswerRun(runId, query, crypto.randomUUID(), signal);
      if (!this.#continuation.isCurrent(signal)) return;
      this.handles.conversations.upsertSummary(descriptor.conversation);
      await webRouter.navigate(conversationRoute(descriptor.conversation.conversationId));
    } catch (error) {
      if (this.#continuation.isCurrent(signal)) {
        requestToast(this, {
          tone: 'error',
          message: localizedText(commandError(
            error,
            catalogText(
              'The continuation could not be started.',
              'chatFeature.continuationFailed',
            ),
          )),
        });
      }
    } finally {
      this.#continuation.finish(signal);
    }
  }

  protected override willUpdate(changed: PropertyValues<this>): void {
    if (!changed.has('view')) return;
    this.#pendingResume = null;
    const routeConversationId = this.view.kind === 'ready' ? this.view.conversationId : null;
    this.#submissionActor = this.handles.submissions.actor(routeConversationId);
    this.#submissionTurnId = this.#submissionActor
      ? `local-${answerSubmissionSnapshot(this.#submissionActor).submissionId}`
      : null;
    if (this.view.kind === 'ready') {
      const stored = this.view.history.map(storedTurnView);
      const previousView = changed.get('view') as ChatView | undefined;
      if (
        previousView?.kind === 'ready'
        && previousView.conversationId === this.view.conversationId
      ) {
        const currentById = new Map(this.turns.map((turn) => [turn.id, turn]));
        const currentByRunId = new Map(
          this.turns.filter((turn) => turn.runId).map((turn) => [turn.runId, turn]),
        );
        const mergedStored = stored.map((turn) => {
          const current = currentById.get(turn.id) ?? currentByRunId.get(turn.runId);
          return current && !isTerminalTurnState(current.state) && !isTerminalTurnState(turn.state) ? current : turn;
        });
        const storedIds = new Set(stored.map((turn) => turn.id));
        const storedRunIds = new Set(stored.map((turn) => turn.runId).filter(Boolean));
        const live = this.turns.filter((turn) => (
          !isTerminalTurnState(turn.state)
          && !storedIds.has(turn.id)
          && (!turn.runId || !storedRunIds.has(turn.runId))
        ));
        this.turns = [...mergedStored, ...live];
      } else {
        this.turns = stored;
      }
      const pending = [...this.view.history].reverse().find(
        (turn) => turn.status === 'queued' || turn.status === 'running',
      );
      if (pending) {
        this.#pendingResume = {
          conversationId: this.view.conversationId,
          stored: pending,
        };
      }
    } else {
      this.turns = [];
    }
    // Commands belong to the conversation the store holds, not to the route alone: the store keeps
    // them beside the turns and republishes a view whenever one is appended, so the list stays in step.
    this.commands = this.#commandsForView();
    const submissionTurn = this.#submissionTurnForRoute();
    if (submissionTurn && !this.turns.some((turn) => turn.id === submissionTurn.id)) {
      this.turns = [...this.turns, submissionTurn];
    }
  }

  protected override updated(changed: PropertyValues<this>): void {
    const hasMessages = this.turns.length > 0 || this.commands.length > 0;
    if (hasMessages !== this.#announcedHasMessages) {
      this.#announcedHasMessages = hasMessages;
      raise(this, 'dl-chat-content-change', {hasMessages});
    }
    if (!changed.has('view') || !this.#pendingResume) return;
    const pending = this.#pendingResume;
    this.#pendingResume = null;
    queueMicrotask(() => {
      if (!this.isConnected || this.view.kind !== 'ready'
          || this.view.conversationId !== pending.conversationId) return;
      const conversationId = this.handles.conversations.activeConversationId || pending.conversationId;
      if (conversationId) void this.#resumeStoredTurn(conversationId, pending.stored);
    });
  }

  protected override render(): TemplateResult {
    void this.runRevision;
    return html`
      <dl-chat-message-list .handles=${this.handles} .view=${this.view} .turns=${this.turns}
        .commands=${this.commands}
        .scrollRequest=${this.#scrollRequest}
        .answerFeedbackEnabled=${this.answerFeedbackEnabled}
        .interactionLocked=${this.interactionLocked}
        @dl-chat-reconnect=${this.#reconnect}
        @dl-chat-retry-turn=${this.#retryTurn}
        @dl-chat-load-older=${this.#loadOlderMessages}></dl-chat-message-list>
      ${this.#submissionFailureControls()}
      <dl-goal-bar .conversationId=${this.handles.conversations.activeConversationId}
        @dl-goal-resumed=${this.#goalResumed}></dl-goal-bar>
      <dl-queue-strip
        .conversationId=${this.handles.conversations.activeConversationId}></dl-queue-strip>
      <dl-chat-composer
        .handles=${this.handles}
        .agentEffortOffer=${this.agentEffortOffer}
        ?inert=${this.interactionLocked}
        .running=${this.#runController.active}
        .live=${this.#conversationLive || this.#runController.active}
        .submissionPending=${this.submissionPending}
        .stopping=${this.#runController.stopping}
        .attachmentPolicy=${this.attachmentPolicy}
        .attachmentAccept=${this.attachmentAccept}
        @dl-composer-submit=${this.#submit}
        @dl-composer-cancel=${this.#cancel}
      ></dl-chat-composer>
    `;
  }

  #submissionFailureControls(): TemplateResult {
    if (!this.#submissionActor) return html``;
    const snapshot = answerSubmissionSnapshot(this.#submissionActor);
    if (!SUBMISSION_BANNER_STATUSES.includes(snapshot.status)) return html``;
    return html`
      <div role="alert" class="submission-failure" tabindex="-1">
        <span>${localizedText(submissionError(snapshot.error))}</span>
        ${snapshot.status === 'retryable' ? html`
          <button class="dl-btn" type="button" @click=${this.#retrySubmission}>
            ${msg('Retry', {id: 'chatFeature.submissionRetry'})}
          </button>
        ` : null}
        <button class="dl-btn" type="button" @click=${this.#editSubmission}>
          ${msg('Edit', {id: 'chatFeature.submissionEdit'})}
        </button>
        <button class="dl-btn" type="button" @click=${this.#discardSubmission}>
          ${msg('Discard', {id: 'chatFeature.submissionDiscard'})}
        </button>
      </div>
    `;
  }

  #loadOlderMessages = (event: Event): void => {
    event.stopPropagation();
    void this.handles.conversations.loadOlderMessages();
  };

  #submissionTurnForRoute(): ChatTurnView | null {
    const actor = this.#submissionActor;
    if (!actor) return null;
    const snapshot = answerSubmissionSnapshot(actor);
    if (['accepted', 'delivered', 'handedOff', 'edited', 'discarded'].includes(snapshot.status)) {
      return null;
    }
    const {intent, lease} = actor.getSnapshot().context;
    const turn = optimisticTurn(intent.submissionId, intent.query, lease.items);
    if (snapshot.status === 'submitting' || snapshot.status === 'reconciling') return turn;
    return {
      ...turn,
      state: 'failed',
      error: submissionError(snapshot.error),
    };
  }

  #runStateChanged(): void {
    this.runRevision += 1;
    const active = this.#runController.active || this.submissionPending;
    if (active === this.#announcedActive) return;
    this.#announcedActive = active;
    raise(this, 'dl-chat-running-change', {active});
  }

  #composer(): DlChatComposer | null {
    return this.querySelector<DlChatComposer>('dl-chat-composer');
  }


  #submit = (event: CustomEvent<ComposerSubmitDetail>): void => {
    event.stopPropagation();
    const {query, mode, requestedSkill, effort, delivery} = event.detail;
    const command = findWebCommand(query);
    if (command !== null) {
      void this.#runWebCommand(command);
      return;
    }
    let accepted: AcceptedSubmission | null = null;
    try {
      accepted = this.#acceptSubmission(query, mode, requestedSkill, effort, delivery);
    } finally {
      // The composer clears its box only for words that were taken: a refusal cancels the event, and so
      // does a throw, which would otherwise leave the box emptied for a message nobody has.
      if (accepted === null) event.preventDefault();
    }
    if (accepted) void this.#observeSubmission(accepted.actor, accepted.turnId, accepted.conversationId);
  };

  /** A failed turn's words are sent again: as a queued message when a Run is live, otherwise as
   *  an ordinary new turn. Nothing of the failed Run was kept but the words, and they are the
   *  reader's. */
  #retryTurn = (event: CustomEvent<ChatRetryTurnDetail>): void => {
    event.stopPropagation();
    const delivery = this.#conversationLive || this.#runController.active ? 'queue' : undefined;
    let accepted: AcceptedSubmission | null = null;
    try {
      accepted = this.#acceptSubmission(event.detail.query, null, null, null, delivery);
    } finally {
      if (accepted !== null) {
        void this.#observeSubmission(accepted.actor, accepted.turnId, accepted.conversationId);
      }
    }
  };

  /** A resume may start the goal's next round server-side: re-read so that turn joins the view. */
  #goalResumed = (): void => {
    void this.handles.conversations.refreshActive();
  };

  #goalBar(): DlGoalBar | null {
    return this.querySelector<DlGoalBar>('dl-goal-bar');
  }

  #queueStrip(): DlQueueStrip | null {
    return this.querySelector<DlQueueStrip>('dl-queue-strip');
  }

  /** A server-started Run (a goal round, another tab) is followed by reading the conversation
   *  again: its running turn joins the view and the normal resume path attaches the stream. */
  #activityChanged(): void {
    const conversationId = this.handles.conversations.activeConversationId;
    const live = conversationId !== null
      && this.handles.runActivity.runs.some(
        (run) => run.conversationId === conversationId && runActive(run.status),
      );
    if (live !== this.#conversationLive) {
      this.#conversationLive = live;
      this.requestUpdate();
      // A goal round this page did not start increments the bar's round counter before its own
      // turn settles here; the bar reads the Goal when such a Run becomes live.
      if (live && !this.#runController.active) this.#goalBar()?.refresh();
    }
    if (!live) {
      this.#attachRequested = false;
      return;
    }
    if (this.#runController.active || this.#attachRequested) return;
    this.#attachRequested = true;
    void this.handles.conversations.refreshActive();
  }

  /** Run one registered command where it belongs: on the server, which owns the execution and keeps
   *  the command and its result in the conversation's history. The browser only names the command and
   *  the words its catalog matched, then draws the command the server kept. */
  async #runWebCommand(invocation: CommandInvocation): Promise<void> {
    const conversations = this.handles.conversations;
    let conversationId =
      conversations.answerConversationId ?? conversations.activeConversationId;
    let created = false;
    if (conversationId === null) {
      // A bare `/goal` is the read-only status form: it must not create (or leave) a conversation
      // just to say there is no goal. Anything with arguments is a command that does work.
      if (invocation.command.name === 'goal' && invocation.args.trim() === '') {
        requestToast(this, {
          tone: 'receipt',
          message: msg('No goal in this conversation. /goal <objective> creates one.', {
            id: 'goalCommand.none',
          }),
        });
        return;
      }
      try {
        const summary = await createConversation();
        // The new conversation becomes the active one before the command runs: `/goal <objective>`
        // must start round 1 immediately, and the round's turn has to land in a conversation the
        // page is showing rather than behind a list row.
        conversations.adoptCreatedConversation(summary);
        conversationId = summary.conversationId;
        created = true;
        await webRouter.navigate(conversationRoute(conversationId));
      } catch (error) {
        requestToast(this, {
          tone: 'error',
          message: localizedText(commandError(
            error,
            catalogText('The command failed.', 'chatFeature.commandFailed'),
          )),
        });
        return;
      }
    }
    try {
      // A goal command may start round 1, which continues the reader's current workspace
      // selection; the other commands carry it too (the server ignores it where it does not apply).
      const command = await postConversationCommand(conversationId, {
        name: invocation.command.name,
        input: invocation.args,
        workspaces: [...this.handles.workspaces.active],
      });
      conversations.appendCommand(conversationId, command);
      this.commands = this.#commandsForView();
      const view = chatCommandView(command);
      // A successful command is control-plane work, not conversation content: confirm it with a
      // receipt and let the goal bar carry the state. Only a failure leaves a transcript notice.
      if (view.tone !== 'error') {
        requestToast(this, {tone: 'receipt', message: view.result()});
      }
      this.#goalBar()?.refresh();
      if (command.name === 'goal') {
        // A `/goal` command may have started a round server-side (create, or a resume that leaves
        // the Session idle). Wake Run activity first so the live round is followed at once, then
        // read the conversation so its new turn appears without a reload. Neither is awaited: the
        // command is already kept, and a read that fails costs a refresh, never the receipt.
        this.handles.runActivity.wake();
      }
      if (command.name === 'goal' || created) {
        void conversations.refreshActive().catch(() => {
          // The command is persisted; the view catches up on the next read.
        });
      }
    } catch (error) {
      requestToast(this, {
        tone: 'error',
        message: localizedText(commandError(
          error,
          catalogText('The command failed.', 'chatFeature.commandFailed'),
        )),
      });
    }
  }

  #cancel = (event: Event): void => {
    event.stopPropagation();
    void this.#runController.cancel();
  };

  #reconnect = (event: CustomEvent<ChatReconnectDetail>): void => {
    event.stopPropagation();
    const turn = this.turns.find((candidate) => candidate.runId === event.detail.runId);
    const conversationId = this.handles.conversations.activeConversationId;
    if (!turn || !conversationId || this.#runController.active) return;
    this.#setTurn(turn.id, {
      state: 'pending',
      error: null,
      activity: turn.cancelRequested
        ? {kind: 'copy', source: 'Stopping...', id: 'chatFeature.stopping'}
        : {kind: 'copy', source: 'Answer in progress...', id: 'chatFeature.answerInProgress'},
    });
    this.handles.answerEventCursors.trackRun(conversationId, turn.runId);
    if (!this.#runController.beginFollow(turn.runId, turn.cancelRequested)) return;
    void this.#followTurn(turn.id, conversationId, turn.runId);
  };

  /** Take the reader's message if it can be sent now, and say why when it cannot. This is the part of a
   *  submission that needs no await, so the composer hears the answer inside its own event and clears its
   *  box only for words that were taken. A refusal changes nothing: the words are still where the reader
   *  wrote them. */
  #acceptSubmission(
    query: string,
    mode: AnswerMode | null,
    requestedSkill: string | null,
    effort: AgentEffort | null,
    delivery?: AnswerDelivery,
  ): AcceptedSubmission | null {
    // A send while a Run is live is delivered to it, not admitted as a competing Run; the composer
    // only offers that with a delivery. A send with none while a Run is active is a command or a
    // programmatic offer, and stays refused.
    if (this.#runController.active && delivery === undefined) return null;
    if (this.#submissionActor) {
      this.#askForEarlierSubmission(this.#submissionActor);
      return null;
    }
    const {conversations} = this.handles;
    if (!conversations.canAnswer) {
      requestToast(this, {
        tone: 'error',
        message: conversations.viewState === 'loading'
          ? msg('This conversation is still loading. Your message is still in the box.', {
            id: 'chatFeature.conversationLoading',
          })
          : msg('Conversation service is unavailable. Please retry loading the conversation.', {
            id: 'chatFeature.conversationUnavailable',
          }),
      });
      return null;
    }
    const conversationId = conversations.answerConversationId;
    const lease = this.handles.attachments.leaseAll();
    const submissionId = crypto.randomUUID();
    const turn = optimisticTurn(submissionId, query, lease.items);
    const actor = this.handles.submissions.start({
      query,
      mode,
      conversationId,
      submissionId,
      workspaces: [...this.handles.workspaces.active],
      ...(requestedSkill ? {requestedSkill} : {}),
      effort,
      ...(delivery ? {delivery} : {}),
    }, lease, this.#submissionAdapter);
    if (!actor) {
      lease.restore();
      // The registry keeps one submission per conversation. One this page has not drawn yet is still the
      // earlier message to settle first.
      const earlier = this.handles.submissions.actor(conversationId);
      if (earlier) this.#askForEarlierSubmission(earlier);
      return null;
    }
    this.#submissionActor = actor;
    this.#submissionTurnId = turn.id;
    actor.subscribe(() => {
      this.requestUpdate();
      const status = answerSubmissionSnapshot(actor).status;
      if (status !== 'accepted' && status !== 'delivered') this.#runStateChanged();
    });
    this.#scrollRequest += 1;
    this.turns = [...this.turns, turn];
    this.#runStateChanged();
    return {actor, turnId: turn.id, conversationId};
  }

  /** A message that failed is still waiting on the reader, and a new one cannot go until they settle it:
   *  say so, in the choices its banner offers, and take them to the banner itself. Not to one of its buttons:
   *  the reader pressed Enter to send, and a second Enter must not choose Retry, Edit, or Discard for them.
   *  A message still on its way needs no words, since the composer says Submitting. */
  #askForEarlierSubmission(earlier: AnswerSubmissionActor): void {
    const {status} = answerSubmissionSnapshot(earlier);
    if (!SUBMISSION_BANNER_STATUSES.includes(status)) return;
    requestToast(this, {
      tone: 'error',
      message: status === 'retryable'
        ? msg('Retry, edit or discard your earlier message first.', {
          id: 'chatFeature.earlierSubmissionRetryable',
        })
        : msg('Edit or discard your earlier message first.', {
          id: 'chatFeature.earlierSubmissionEditable',
        }),
    });
    void this.updateComplete.then(() => {
      this.querySelector<HTMLElement>('.submission-failure')?.focus();
    });
  }

  async #observeSubmission(
    actor: AnswerSubmissionActor,
    turnId: string,
    expectedConversationId: string | null,
  ): Promise<void> {
    try {
      await waitFor(actor, (snapshot) => (
        !snapshot.matches('submitting') && !snapshot.matches('reconciling')
      ));
    } catch {
      return;
    }
    if (this.#submissionActor !== actor) return;
    const snapshot = answerSubmissionSnapshot(actor);
    const outcome = snapshot.outcome;
    if (snapshot.status === 'delivered' && outcome !== null && outcome.kind !== 'accepted') {
      this.#submissionActor = null;
      this.#submissionTurnId = null;
      this.turns = this.turns.filter((turn) => turn.id !== turnId);
      if (outcome.kind === 'queue') {
        this.#queueStrip()?.refresh();
        requestToast(this, {
          tone: 'receipt',
          message: msg('Queued — it will be sent after this answer.', {
            id: 'chatFeature.queued',
          }),
        });
      } else {
        const turn = this.#turnOfRun(outcome.runId);
        if (turn) {
          this.#setTurn(turn.id, {
            steeringMessages: [...turn.steeringMessages, queryOf(actor)].slice(-MAX_STEERING_MESSAGES),
          });
        }
        requestToast(this, {
          tone: 'receipt',
          message: msg('Sent to the Run that is working.', {id: 'chatFeature.steered'}),
        });
      }
      this.#runStateChanged();
      return;
    }
    if (snapshot.status !== 'accepted' || snapshot.outcome?.kind !== 'accepted') {
      this.#setTurnError(turnId, submissionError(snapshot.error));
      this.#runStateChanged();
      return;
    }
    const accepted = snapshot.outcome;
    const acceptedConversationId = accepted.conversation.conversationId;
    if (expectedConversationId && acceptedConversationId !== expectedConversationId) {
      this.#setTurnError(turnId, catalogText(
        'The answer was accepted for an unexpected conversation.',
        'chatFeature.unexpectedConversation',
      ));
      actor.send({type: 'HANDOFF'});
      this.#submissionActor = null;
      this.#runStateChanged();
      return;
    }
    this.#submissionActor = null;
    this.#submissionTurnId = null;
    if (!expectedConversationId) {
      this.handles.conversations.adoptCreatedConversation(accepted.conversation);
      await webRouter.navigate(conversationRoute(acceptedConversationId), {
        replace: true,
        notify: false,
        bypassGuard: true,
      });
    } else {
      this.handles.conversations.upsertSummary(accepted.conversation);
    }
    const stored = accepted.turn;
    this.handles.answerEventCursors.trackRun(acceptedConversationId, stored.answerRunId);
    this.#replaceStoredTurn(turnId, stored);
    const following = this.#runController.beginFollow(
      stored.answerRunId,
      stored.cancelRequested,
    );
    actor.send({type: 'HANDOFF'});
    if (!following) {
      this.#runStateChanged();
      return;
    }
    this.#runStateChanged();
    await this.#followTurn(turnId, acceptedConversationId, stored.answerRunId);
  }

  #retrySubmission = (): void => {
    const actor = this.#submissionActor;
    const turnId = this.#submissionTurnId;
    if (!actor || !turnId) return;
    actor.send({type: 'RETRY'});
    this.#setTurn(turnId, {state: 'pending', error: null});
    void this.#observeSubmission(
      actor,
      turnId,
      answerSubmissionSnapshot(actor).conversationId,
    );
  };

  #editSubmission = (): void => this.#finishFailedSubmission('EDIT');
  #discardSubmission = (): void => this.#finishFailedSubmission('DISCARD');

  #finishFailedSubmission(type: 'EDIT' | 'DISCARD'): void {
    const actor = this.#submissionActor;
    if (!actor) return;
    const intent = actor.getSnapshot().context.intent;
    actor.send({type});
    if (type === 'EDIT') {
      this.handles.workspaces.restoreActive(intent.workspaces);
      this.querySelector<DlChatComposer>('dl-chat-composer')
        ?.restoreSubmission(
          intent.query,
          intent.mode,
          intent.requestedSkill ?? null,
          intent.effort ?? null,
        );
    }
    if (this.#submissionTurnId) {
      this.turns = this.turns.filter((turn) => turn.id !== this.#submissionTurnId);
    }
    this.#submissionActor = null;
    this.#submissionTurnId = null;
    this.#runStateChanged();
    if (type === 'EDIT') this.focusComposer();
  }

  async #resumeStoredTurn(conversationId: string, stored: ConversationTurn): Promise<void> {
    if (this.#runController.active) return;
    this.handles.answerEventCursors.trackRun(conversationId, stored.answerRunId);
    if (!this.#runController.beginFollow(stored.answerRunId, stored.cancelRequested)) return;
    await this.#followTurn(
      stored.turnId || stored.answerRunId,
      conversationId,
      stored.answerRunId,
    );
  }

  async #followTurn(turnId: string, conversationId: string, runId: string): Promise<void> {
    let result: FollowResult;
    try {
      result = await this.#runController.follow(
        conversationId,
        runId,
        (events) => this.#handleRunBatch(turnId, events),
      );
    } catch {
      result = {
        kind: 'error',
        message: catalogText('Connection error. Please try again.', 'chatFeature.connectionError'),
      };
    }
    if (this.#runController.runId !== runId) return;

    let finished = false;
    if (result.kind === 'terminal') {
      if (result.stored) this.#replaceStoredTurn(turnId, result.stored);
      finished = true;
    } else if (result.kind === 'retryable') {
      const cancelRequested = result.stored.cancelRequested;
      const reconnectState = answerReconnectState(cancelRequested);
      this.#setTurn(turnId, {
        state: 'retryable',
        error: catalogText(
          ANSWER_RECONNECT_COPY[reconnectState].status,
          `chatMessageList.reconnect.${reconnectState}.status`,
        ),
        activity: null,
        cancelRequested,
      });
    } else if (result.kind === 'error') {
      this.#setTurnError(turnId, result.message);
      finished = true;
    }
    this.#runController.finish(runId);
    if (finished) this.#queueStrip()?.refresh();
    if (!finished) return;
    this.handles.answerEventCursors.clear(conversationId);
    void this.handles.conversations.refreshActive();
  }

  #handleRunBatch(turnId: string, events: readonly AnswerRunEvent[]): void {
    // A Goal the Agent created or moved mid-Run is durable when its Tool settles, so the bar
    // reads it now instead of waiting for the turn to end: a reader watching the Run sees the
    // goal appear without a reload.
    if (goalWasWritten(events)) this.#goalBar()?.refresh();
    const turnIndex = this.turns.findIndex((candidate) => candidate.id === turnId);
    const turn = this.turns[turnIndex];
    for (const event of events) {
      if (event.kind === 'memory') {
        raise(this, 'dl-chat-memory-operation', event.operation);
        // The Agent changed the profile in this Run: its answer will say what it left, once it has settled.
        if (turn?.runId && event.operation.outcome === 'changed') {
          this.handles.memoryNotices.noteAgentWrote(turn.runId);
        }
      }
    }
    if (!turn) return;
    // One observation time per batch: every row this batch starts is stamped from
    // the same instant, and the view owns the ticking that follows.
    const now = performance.now();
    const projected = events.reduce(
      (view, event) => applyAnswerEvent(view, event, now),
      turn,
    );
    if (projected === turn) return;
    const nextTurns = [...this.turns];
    nextTurns[turnIndex] = projected;
    this.turns = nextTurns;
    // Tokens stream every frame, but a tool event is something an open dock may show.
    this.#noteRunActivity(turn, projected, events);
  }

  /** Hand an open Agent traces dock what its Run just stated: the events themselves, and always the
   *  moment the Run settled however that became known. A dock reads its pages for these, so nothing
   *  it shows waits for a clock — and a stream of tokens never reaches it at all. */
  #noteRunActivity(
    previous: ChatTurnView,
    next: ChatTurnView,
    events: readonly AnswerRunEvent[],
  ): void {
    const runId = next.runId || previous.runId;
    if (!runId) return;
    const ended = !isTerminalTurnState(previous.state) && isTerminalTurnState(next.state);
    // An answer that has just been given is looked at for what the Run left in memory, which the pass
    // is still working out; the notice under it appears when there is something to say.
    if (ended && next.state === 'succeeded') this.handles.memoryNotices.watch(runId);
    if (ended) {
      this.#goalBar()?.refresh();
      this.#queueStrip()?.refresh();
    }
    // Tokens are the chat's own words and draw here; what an open dock reads is what they state.
    if (!statesSomething(events) && !ended) return;
    raise(this, 'dl-run-activity', {runId, events});
  }

  #replaceStoredTurn(turnId: string, stored: ConversationTurn): void {
    this.#updateTurn(turnId, () => ({...storedTurnView(stored), id: turnId}));
  }

  #setTurnError(turnId: string, error: LocalizedText): void {
    this.#setTurn(turnId, {state: 'failed', error, activity: null});
  }

  #setTurn(turnId: string, patch: Partial<ChatTurnView>): void {
    this.#updateTurn(turnId, (turn) => ({...turn, ...patch}));
  }

  #updateTurn(turnId: string, change: (turn: ChatTurnView) => ChatTurnView): void {
    const index = this.turns.findIndex((turn) => turn.id === turnId);
    const previous = this.turns[index];
    if (previous === undefined) return;
    const next = change(previous);
    const turns = [...this.turns];
    turns[index] = next;
    this.turns = turns;
    this.#noteRunActivity(previous, next, []);
  }
}

customElements.define('dl-chat-feature', DlChatFeature);

declare global {
  interface HTMLElementTagNameMap {
    'dl-chat-feature': DlChatFeature;
  }

  interface HTMLElementEventMap {
    'dl-chat-content-change': CustomEvent<ChatContentChangeDetail>;
    'dl-chat-running-change': CustomEvent<ChatRunningChangeDetail>;
    'dl-chat-memory-operation': CustomEvent<MemoryOperationEvent>;
    'dl-run-activity': CustomEvent<ChatRunActivityDetail>;
  }
}
