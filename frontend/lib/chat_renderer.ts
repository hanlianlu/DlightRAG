// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {
  releaseMessageAttachmentObjectUrls,
  renderMessageAttachmentImages,
} from '../ui/images.ts';
import {bindPrimaryReportControl} from '../ui/report-panel.ts';
import type {AnswerPresentationElement} from '../ui/answer_presentation.ts';
import '../ui/answer_presentation.ts';
import {createDocumentChip} from './document_chip.ts';
import {answerErrorMessage} from './errors.ts';
import {parseData} from './sse.ts';
import {safeSameOriginHref} from './urls.ts';
import chatStyles from '../styles/chat.module.css';
import type {
  AnswerPresentation,
  ConversationAttachmentReference,
  ConversationHistory,
  ConversationTurn,
} from '../api/conversations.ts';

// ── types ────────────────────────────────────────────────────────────

export interface ChatTurn {
  chatArea: HTMLElement;
  aiDiv: HTMLDivElement;
  contentDiv: HTMLDivElement;
  actionsDiv: HTMLDivElement;
}

export interface DonePayload {
  status: 'succeeded' | 'cancelled';
  presentation: AnswerPresentation | null;
  usage?: Record<string, unknown>;
  evidence?: Record<string, number>;
}

interface ProgressPayload {
  phase: string;
}

type PhaseLabel = 'routing' | 'planning' | 'searching' | 'researching' | 'generating';

const PHASE_LABELS: Record<PhaseLabel, string> = {
  routing: 'Choosing answer mode...',
  planning: 'Analyzing query...',
  searching: 'Searching knowledge base...',
  researching: 'Researching sources...',
  generating: 'Generating answer...',
};

// ── helpers ───────────────────────────────────────────────────────────

function activateChatMode(): void {
  const app = document.querySelector('.app');
  if (app && !app.classList.contains('has-messages')) app.classList.add('has-messages');
}

function scrollToBottom(turn: ChatTurn): void {
  if (turn.chatArea) turn.chatArea.scrollTop = turn.chatArea.scrollHeight;
}

// ── public API ────────────────────────────────────────────────────────

// Cap rendered chat DOM nodes (~HISTORY_CAP turns x user+ai) so long-lived
// sessions don't accumulate an unbounded number of message nodes.
const MAX_CHAT_MESSAGE_NODES = 200;

// Auto-scroll sticks to the bottom only while the reader is already near it, so
// scrolling up to re-read earlier text is not fought by incoming tokens.
const STICK_TO_BOTTOM_PX = 160;

// Marks the raw-token tail appended while the answer streams. The rendered
// answer is derived once, from the run's canonical result, when it finishes.
const STREAM_TAIL_CLASS = 'stream-tail';

// Marks the recoverable-reconnect offer so exactly one can ever be live.
const RECONNECT_CLASS = 'answer-reconnect';

/** Retire the reconnect offer: the run is being followed again, or has settled. */
function clearAnswerReconnect(turn: ChatTurn): void {
  turn.contentDiv.querySelectorAll('.' + RECONNECT_CLASS).forEach((node) => node.remove());
}

function isDonePayload(value: unknown): value is DonePayload {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
  const payload = value as Record<string, unknown>;
  return payload.status === 'succeeded' || payload.status === 'cancelled';
}


function pruneOldMessages(chatMessages: HTMLElement): void {
  while (chatMessages.childElementCount > MAX_CHAT_MESSAGE_NODES) {
    const oldest = chatMessages.firstElementChild;
    if (!oldest) return;
    releaseMessageAttachmentObjectUrls(oldest);
    oldest.remove();
  }
}

// Render document attachments as compact chips (filename + size) that download
// the original via a same-origin link, using the shared chip component.
function renderMessageDocuments(
  container: Element,
  documents: readonly ConversationAttachmentReference[],
): void {
  if (documents.length === 0) return;
  const strip = document.createElement('div');
  strip.className = chatStyles.messageDocuments;
  documents.forEach(function (reference) {
    strip.appendChild(
      createDocumentChip({
        filename: reference.filename,
        byteSize: reference.byte_size,
        href: safeSameOriginHref(reference.url) || undefined,
      }),
    );
  });
  container.appendChild(strip);
}

export function createChatTurn(
  query: string,
  attachments?: readonly ConversationAttachmentReference[],
): ChatTurn {
  const chatMessages = document.getElementById('chat-messages')!;
  const chatArea = document.getElementById('chat-area')!;

  activateChatMode();

  const userWrapper = document.createElement('div');
  userWrapper.className = chatStyles.userMessageWrapper;
  if (attachments && attachments.length > 0) {
    renderMessageAttachmentImages(
      userWrapper,
      attachments.filter((attachment) => attachment.kind === 'image'),
    );
    renderMessageDocuments(
      userWrapper,
      attachments.filter((attachment) => attachment.kind === 'document'),
    );
  }

  const userDiv = document.createElement('div');
  userDiv.className = chatStyles.userMessage;
  userDiv.textContent = query;
  userWrapper.appendChild(userDiv);
  chatMessages.appendChild(userWrapper);

  const aiDiv = document.createElement('div');
  aiDiv.className = chatStyles.aiMessage;

  const headerDiv = document.createElement('div');
  headerDiv.className = chatStyles.aiMessageHeader;
  const dot = document.createElement('span');
  dot.className = chatStyles.dot;
  dot.textContent = '●';
  headerDiv.appendChild(dot);
  headerDiv.appendChild(document.createTextNode(' DlightRAG'));
  aiDiv.appendChild(headerDiv);

  const contentDiv = document.createElement('div');
  contentDiv.className = chatStyles.aiMessageContent;
  const streamingDot = document.createElement('span');
  streamingDot.className = chatStyles.streamingDot;
  contentDiv.appendChild(streamingDot);
  aiDiv.appendChild(contentDiv);

  const actionsDiv = document.createElement('div');
  actionsDiv.className = chatStyles.runActions;
  aiDiv.appendChild(actionsDiv);

  const liveStatus = document.createElement('span');
  liveStatus.className = 'sr-only';
  liveStatus.setAttribute('role', 'status');
  liveStatus.setAttribute('aria-live', 'polite');
  aiDiv.appendChild(liveStatus);

  chatMessages.appendChild(aiDiv);
  pruneOldMessages(chatMessages);

  const turn: ChatTurn = {chatArea, aiDiv, contentDiv, actionsDiv};
  scrollToBottom(turn);
  return turn;
}

function applyFinalAnswer(
  turn: ChatTurn,
  presentation: AnswerPresentation,
  usage: Record<string, unknown> = {},
  evidence: Record<string, number> = {},
): void {
  const element = document.createElement('answer-presentation') as AnswerPresentationElement;
  element.presentation = presentation;
  turn.contentDiv.replaceChildren(element);
  bindPrimaryReportControl(
    turn.aiDiv,
    turn.aiDiv.dataset.runId || '',
    presentation.primary_report,
  );
  renderRunActions(turn, 'terminal', usage, evidence);
}

function runActionButton(turn: ChatTurn, action: string, label: string): HTMLButtonElement {
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = label;
  button.dataset.runAction = action;
  button.addEventListener('click', function() {
    turn.aiDiv.dispatchEvent(new CustomEvent('answer-run-action', {
      bubbles: true,
      detail: {action, runId: turn.aiDiv.dataset.runId || ''},
    }));
  });
  return button;
}

function renderRunActions(
  turn: ChatTurn,
  state: 'running' | 'terminal',
  usage: Record<string, unknown> = {},
  evidence: Record<string, number> = {},
): void {
  turn.actionsDiv.replaceChildren();
  if (!turn.aiDiv.dataset.runId) return;
  if (state === 'running') {
    turn.actionsDiv.append(
      runActionButton(turn, 'steer', 'Steer'),
      runActionButton(turn, 'children', 'Child agents'),
    );
    return;
  }
  turn.actionsDiv.append(
    runActionButton(turn, 'follow-up', 'Follow up'),
    runActionButton(turn, 'fork', 'Fork'),
    runActionButton(turn, 'children', 'Child agents'),
  );
  const evidenceCount = Number(evidence.chunks || 0);
  const usageDetails = usage.usage_details as Record<string, unknown> | undefined;
  const tokenCount = Number(usageDetails?.total_tokens || 0);
  if (evidenceCount || tokenCount) {
    const summary = document.createElement('span');
    summary.className = chatStyles.runSummary;
    summary.textContent = [
      evidenceCount ? `${evidenceCount} evidence chunks` : '',
      tokenCount ? `${tokenCount} tokens` : '',
    ].filter(Boolean).join(' · ');
    turn.actionsDiv.appendChild(summary);
  }
}


export function clearChatViewport(): void {
  const chatMessages = document.getElementById('chat-messages');
  if (!chatMessages) return;
  releaseMessageAttachmentObjectUrls(chatMessages);
  const welcome = document.getElementById('welcome');
  chatMessages.replaceChildren(...(welcome ? [welcome] : []));
  document.querySelector('.app')?.classList.remove('has-messages');
}

export interface PendingHistoryTurn {
  turn: ChatTurn;
  stored: ConversationTurn;
}

/**
 * Render stored history and report the turn whose run has not finished.
 *
 * A reloaded page therefore rediscovers a queued or running answer from the
 * conversation itself, without remembering the response that started it.
 */
export function renderConversationHistory(
  history: ConversationHistory,
): PendingHistoryTurn | null {
  clearChatViewport();
  let pending: PendingHistoryTurn | null = null;
  for (const stored of history.turns) {
    const turn = createChatTurn(stored.user_text, stored.user_attachments);
    renderStoredTurn(turn, stored);
    if (stored.status === 'queued' || stored.status === 'running') {
      pending = {turn, stored};
    }
  }
  return pending;
}

/** Render one stored turn from its run's state: pending, terminal, or answered. */
export function renderStoredTurn(turn: ChatTurn, stored: ConversationTurn): void {
  clearAnswerReconnect(turn);
  turn.aiDiv.dataset.runId = stored.answer_run_id;
  if (stored.status === 'succeeded') {
    if (stored.presentation) {
      applyFinalAnswer(turn, stored.presentation, stored.usage, stored.evidence);
    }
    else setAnswerError(turn, 'Stored answer presentation is unavailable.');
    return;
  }
  if (stored.status === 'failed') {
    setAnswerError(turn, answerErrorMessage({message: stored.error_message}));
    return;
  }
  if (stored.status === 'cancelled') {
    markAnswerStopped(turn);
    return;
  }
  markAnswerPending(turn, stored.cancel_requested);
}

/** Show that a queued or running answer is still being produced elsewhere. */
export function markAnswerPending(turn: ChatTurn, cancelRequested = false): void {
  const indicator = document.createElement('span');
  indicator.className = chatStyles.streamingDot + ' ' + chatStyles.progressPhase;
  indicator.setAttribute(
    'data-phase',
    cancelRequested ? 'Stopping...' : 'Generating answer...',
  );
  turn.contentDiv.replaceChildren(indicator);
  renderRunActions(turn, 'running');
}

function renderConversationState(message: string, isError: boolean): HTMLElement | null {
  clearChatViewport();
  activateChatMode();
  const chatMessages = document.getElementById('chat-messages');
  if (!chatMessages) return null;
  const state = document.createElement('div');
  state.className = isError ? chatStyles.textError : '';
  state.setAttribute('role', isError ? 'alert' : 'status');
  state.setAttribute('aria-live', isError ? 'assertive' : 'polite');
  state.textContent = message;
  chatMessages.appendChild(state);
  return state;
}

export function renderConversationHistoryLoading(): void {
  renderConversationState('Loading conversation history…', false);
}

export function renderConversationHistoryError(onRetry: () => void): void {
  const state = renderConversationState('Conversation history is unavailable.', true);
  if (!state) return;
  const retry = document.createElement('button');
  retry.type = 'button';
  retry.textContent = 'Retry conversation history';
  retry.setAttribute('aria-label', 'Retry loading conversation history');
  retry.addEventListener('click', onRetry);
  state.append(document.createTextNode(' '), retry);
}

export function renderConversationUnavailable(
  onNew: () => void,
  onRecent?: (() => void) | null,
): void {
  const state = renderConversationState('Conversation unavailable.', true);
  if (!state) return;
  const newChat = document.createElement('button');
  newChat.type = 'button';
  newChat.textContent = 'Start a new chat';
  newChat.addEventListener('click', onNew);
  state.append(document.createTextNode(' '), newChat);
  if (!onRecent) return;
  const recent = document.createElement('button');
  recent.type = 'button';
  recent.textContent = 'Open recent conversation';
  recent.addEventListener('click', onRecent);
  state.append(document.createTextNode(' '), recent);
}

export function setAnswerError(turn: ChatTurn, message: unknown): void {
  turn.contentDiv.textContent = typeof message === 'string' ? message : 'Service error. Please try again.';
  turn.contentDiv.classList.add(chatStyles.textError);
  renderRunActions(turn, 'terminal');
}

/** A recoverable connection failure: the run continues, so offer a reattach. */
export function setAnswerRetryable(turn: ChatTurn, message: string, onRetry: () => void): void {
  // The run keeps producing, so this is a status rather than the terminal error
  // style, and neither the live progress indicator nor an earlier offer may sit
  // beside the notice.
  clearAnswerReconnect(turn);
  turn.contentDiv
    .querySelectorAll('.' + chatStyles.streamingDot + ', .' + chatStyles.progressPhase)
    .forEach((node) => node.remove());
  turn.contentDiv.classList.remove(chatStyles.textError);
  const offer = document.createElement('span');
  offer.className = RECONNECT_CLASS;
  const notice = document.createElement('span');
  notice.setAttribute('role', 'status');
  notice.textContent = message;
  const retry = document.createElement('button');
  retry.type = 'button';
  retry.textContent = 'Reconnect';
  retry.setAttribute('aria-label', 'Reconnect to this answer');
  retry.addEventListener('click', onRetry);
  offer.append(notice, document.createTextNode(' '), retry);
  turn.contentDiv.appendChild(offer);
}

export function markAnswerStopped(turn: ChatTurn): void {
  // Keep the partial answer that already streamed; drop the live streaming
  // indicator and append a subtle "Stopped" marker instead of wiping content.
  turn.contentDiv
    .querySelectorAll('.' + chatStyles.streamingDot + ', .' + chatStyles.progressPhase)
    .forEach((el) => el.remove());
  const note = document.createElement('div');
  note.className = chatStyles.stoppedNote;
  note.textContent = 'Stopped';
  turn.contentDiv.appendChild(note);
  renderRunActions(turn, 'terminal');
}

export function createAnswerRenderer(turn: ChatTurn) {
  let fullAnswer = '';
  let failed = false;
  let outcome: DonePayload | null = null;
  let scrollScheduled = false;
  let streamStarted = false;

  // Coalesce autoscroll into one write per animation frame, and only follow the
  // stream while the reader is already near the bottom.
  function scheduleAutoScroll(): void {
    if (scrollScheduled) return;
    scrollScheduled = true;
    requestAnimationFrame(function () {
      scrollScheduled = false;
      const area = turn.chatArea;
      if (!area) return;
      if (area.scrollHeight - area.scrollTop - area.clientHeight <= STICK_TO_BOTTOM_PX) {
        area.scrollTop = area.scrollHeight;
      }
    });
  }

  // Clear the placeholder streaming dot / progress phase the first time real
  // answer content arrives, so the draft starts from a clean slate.
  function startStreamViewport(): void {
    if (streamStarted) return;
    turn.contentDiv.replaceChildren();
    streamStarted = true;
  }

  function appendStreamTail(token: string): void {
    const last = turn.contentDiv.lastElementChild;
    let tail: HTMLElement;
    if (last instanceof HTMLElement && last.classList.contains(STREAM_TAIL_CLASS)) {
      tail = last;
    } else {
      tail = document.createElement('span');
      tail.className = STREAM_TAIL_CLASS;
      turn.contentDiv.appendChild(tail);
    }
    tail.append(token);
  }

  function handleToken(data: string): void {
    const text = parseData(data);
    const token = typeof text === 'string' ? text : String(text);
    startStreamViewport();
    fullAnswer += token;
    appendStreamTail(token);
    scheduleAutoScroll();
  }

  // A resumed run regenerates its answer, so drop the interrupted draft before
  // the replacement tokens arrive.
  function handleReset(): void {
    fullAnswer = '';
    streamStarted = false;
    turn.contentDiv.replaceChildren();
  }

  function handleDone(data: string): void {
    const payload = parseData(data);
    if (!isDonePayload(payload)) {
      failed = true;
      setAnswerError(turn, 'Service error. Please try again.');
      return;
    }
    outcome = payload;
    if (payload.status === 'cancelled') {
      markAnswerStopped(turn);
      return;
    }
    if (!payload.presentation) {
      failed = true;
      setAnswerError(turn, 'Service error. Please try again.');
      return;
    }
    fullAnswer = payload.presentation.answer_text;
    applyFinalAnswer(turn, payload.presentation, payload.usage, payload.evidence);
    const live = turn.aiDiv.querySelector('.sr-only');
    if (live) live.textContent = 'Answer ready';
  }

  function handleProgress(data: string): void {
    const info = parseData(data) as ProgressPayload;
    const label: string = PHASE_LABELS[info.phase as PhaseLabel] || info.phase;
    // Resolve the phase indicator, creating one when tokens have already
    // replaced the initial streaming dot. The done event replaces contentDiv,
    // clearing any created node.
    const existing = turn.contentDiv.querySelector<HTMLElement>(
      '.' + chatStyles.progressPhase + ', .' + chatStyles.streamingDot,
    );
    const indicator = existing ?? turn.contentDiv.appendChild(document.createElement('span'));
    indicator.textContent = '';
    indicator.className = chatStyles.streamingDot + ' ' + chatStyles.progressPhase;
    indicator.setAttribute('data-phase', label || '');
    const live = turn.aiDiv.querySelector('.sr-only');
    if (live) live.textContent = label || '';
  }

  return {
    handle(eventType: string, data: string): void {
      // Any durable event proves the reattach worked, so retire its offer.
      clearAnswerReconnect(turn);
      if (eventType === 'token') handleToken(data);
      else if (eventType === 'reset') handleReset();
      else if (eventType === 'done') handleDone(data);
      else if (eventType === 'progress') handleProgress(data);
      else if (eventType === 'error') {
        failed = true;
        setAnswerError(turn, answerErrorMessage(parseData(data)));
      }
    },
    get answer(): string {
      return fullAnswer;
    },
    get failed(): boolean {
      return failed;
    },
    /** Set once a terminal event arrived; the run needs no further following. */
    get terminal(): boolean {
      return failed || outcome !== null;
    },
    get outcome(): DonePayload | null {
      return outcome;
    },
  };
}
