// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {renderMessageAttachmentImages} from '../ui/images.ts';
import {renderMath} from './math.ts';
import {renderDiagrams} from '../ui/mermaid.ts';
import {createDocumentChip} from './document_chip.ts';
import {answerErrorMessage} from './errors.ts';
import {llmFragmentFromSanitizedHtml} from './safe_html.ts';
import {parseData} from './sse.ts';
import chatStyles from '../styles/chat.module.css';
import type {
  ConversationAttachmentReference,
  ConversationHistory,
  ConversationTurn,
} from '../api/conversations.ts';

// SSE HTML payloads are sanitized server-side by nh3 and again here before
// browser insertion. Keep new HTML sinks behind frontend/lib/safe_html.ts.

// ── types ────────────────────────────────────────────────────────────

export interface ChatTurn {
  chatArea: HTMLElement;
  aiDiv: HTMLDivElement;
  contentDiv: HTMLDivElement;
}

export interface DonePayload {
  status: 'succeeded' | 'cancelled';
  html: string;
  answer: string;
}

interface ProgressPayload {
  phase: string;
}

type PhaseLabel = 'planning' | 'searching' | 'researching' | 'generating';

const PHASE_LABELS: Record<PhaseLabel, string> = {
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

function fixExternalLinks(container: ParentNode): void {
  container.querySelectorAll('a[href]').forEach(function (el: Element) {
    const a = el as HTMLAnchorElement;
    if (a.hasAttribute('download')) return;
    a.setAttribute('target', '_blank');
    a.setAttribute('rel', 'noopener noreferrer');
  });
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

// Presence check only: MathJax typesetting is skipped while streaming a preview
// that has no math delimiters, avoiding a full re-typeset of the whole growing
// answer on every 0.3s snapshot.
const MATH_DELIMITER = /\$|\\\(|\\\[/;

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
    chatMessages.firstElementChild?.remove();
  }
}

// Same-origin http(s) guard for stored document download links, mirroring the
// image src admission in ui/images.ts so a poisoned reference cannot inject a
// javascript: or cross-origin href into the history view.
function _safeDocumentHref(src: unknown): string {
  if (typeof src !== 'string') return '';
  const value = src.trim();
  if (!value) return '';
  let url: URL;
  try {
    url = new URL(value, window.location.origin);
  } catch {
    return '';
  }
  if (
    (url.protocol === 'http:' || url.protocol === 'https:') &&
    url.origin === window.location.origin
  ) {
    return url.href;
  }
  return '';
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
        href: _safeDocumentHref(reference.url) || undefined,
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

  const liveStatus = document.createElement('span');
  liveStatus.className = 'sr-only';
  liveStatus.setAttribute('role', 'status');
  liveStatus.setAttribute('aria-live', 'polite');
  aiDiv.appendChild(liveStatus);

  chatMessages.appendChild(aiDiv);
  pruneOldMessages(chatMessages);

  const turn: ChatTurn = {chatArea, aiDiv, contentDiv};
  scrollToBottom(turn);
  return turn;
}

function applyFinalAnswerHtml(turn: ChatTurn, html: string): void {
  const fragment = llmFragmentFromSanitizedHtml(html);
  const answerContent = fragment.querySelector('#answer-content');
  const sourceData = fragment.querySelector('#source-data');
  const imageStrip = fragment.querySelector('.answer-image-strip');
  const refList = fragment.querySelector('.answer-references');

  if (answerContent) {
    const answerNodes = Array.from(answerContent.childNodes).map((node) => node.cloneNode(true));
    turn.contentDiv.replaceChildren(...answerNodes);
  }
  if (imageStrip) turn.contentDiv.appendChild(imageStrip.cloneNode(true));
  if (refList) turn.contentDiv.appendChild(refList.cloneNode(true));
  if (sourceData) {
    (sourceData as HTMLElement).className = 'source-data hidden';
    sourceData.removeAttribute('id');
    fixExternalLinks(sourceData);
    turn.aiDiv.appendChild(sourceData);
  }

  renderMath(turn.contentDiv);
  renderDiagrams(turn.contentDiv);
  fixExternalLinks(turn.contentDiv);
}


export function clearChatViewport(): void {
  const chatMessages = document.getElementById('chat-messages');
  if (!chatMessages) return;
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
  if (stored.status === 'succeeded') {
    applyFinalAnswerHtml(turn, stored.answer_html);
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

export function setAnswerError(turn: ChatTurn, message: unknown): void {
  turn.contentDiv.textContent = typeof message === 'string' ? message : 'Service error. Please try again.';
  turn.contentDiv.classList.add(chatStyles.textError);
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
    fullAnswer = payload.answer;
    applyFinalAnswerHtml(turn, payload.html);
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
