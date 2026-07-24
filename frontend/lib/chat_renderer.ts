// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {renderMessageImages} from '../ui/images.ts';
import {renderMath} from '../ui/mathjax.ts';
import {renderDiagrams} from '../ui/mermaid.ts';
import {createDocumentChip} from './document_chip.ts';
import {answerErrorMessage, notifyAnswerWarning} from './errors.ts';
import {llmFragmentFromSanitizedHtml, setSanitizedLlmHtml} from './safe_html.ts';
import {parseData} from './sse.ts';
import chatStyles from '../styles/chat.module.css';
import type {
  ConversationDocumentReference,
  ConversationHistory,
  ConversationImageReference,
  ConversationSummary,
} from '../api/conversations.ts';
import type {ConversationSavePresentation} from '../stores/pendingSubmissionStore.ts';

// SSE HTML payloads are sanitized server-side by nh3 and again here before
// browser insertion. Keep new HTML sinks behind frontend/lib/safe_html.ts.

// ── types ────────────────────────────────────────────────────────────

interface ChatTurn {
  chatArea: HTMLElement;
  aiDiv: HTMLDivElement;
  contentDiv: HTMLDivElement;
}

type SSEData = string;

export interface DonePayload {
  html?: string;
  answer?: string;
  conversation_saved?: boolean;
  conversation_save_reason?: string | null;
  conversation?: ConversationSummary | null;
}

export interface AnswerRendererOptions {
  onWarning?: (message: string) => void;
}

interface ProgressPayload {
  phase: string;
}

type PhaseLabel = 'planning' | 'searching' | 'generating' | 'saving';

const PHASE_LABELS: Record<PhaseLabel, string> = {
  planning: 'Analyzing query...',
  searching: 'Searching knowledge base...',
  generating: 'Generating answer...',
  saving: 'Saving conversation...',
};

// ── helpers ───────────────────────────────────────────────────────────

function activateChatMode(): void {
  const app = document.querySelector('.app');
  if (app && !app.classList.contains('has-messages')) app.classList.add('has-messages');
}

function fixExternalLinks(container: ParentNode): void {
  container.querySelectorAll('a[href]').forEach(function (el: Element) {
    const a = el as HTMLAnchorElement;
    if (!a.getAttribute('target')) {
      a.setAttribute('target', '_blank');
      a.setAttribute('rel', 'noopener noreferrer');
    }
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

// Marks the raw-token tail appended between previews (see createAnswerRenderer).
const STREAM_TAIL_CLASS = 'stream-tail';

// Split a sanitized answer-HTML snapshot into its top-level markdown blocks so
// completed blocks can be frozen while only the trailing block is re-rendered.
function answerBlocksFromHtml(html: string): HTMLElement[] {
  const fragment = llmFragmentFromSanitizedHtml(html);
  return Array.from(fragment.children).filter(
    (node): node is HTMLElement => node instanceof HTMLElement,
  );
}

// Typeset a block only when it contains math delimiters, so a frozen block is
// typeset once and math-free blocks skip MathJax entirely.
function typesetBlockMath(block: HTMLElement): void {
  if (MATH_DELIMITER.test(block.textContent || '')) renderMath(block);
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

// Render stored Composer documents as compact chips (filename + size) that
// download the original via a same-origin link, using the shared chip component.
function renderMessageDocuments(
  container: Element,
  documents?: readonly ConversationDocumentReference[],
): void {
  if (!documents || documents.length === 0) return;
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
  images?: readonly ConversationImageReference[],
  documents?: readonly ConversationDocumentReference[],
): ChatTurn {
  const chatMessages = document.getElementById('chat-messages')!;
  const chatArea = document.getElementById('chat-area')!;

  activateChatMode();

  const userWrapper = document.createElement('div');
  userWrapper.className = chatStyles.userMessageWrapper;
  renderMessageImages(userWrapper, images);
  renderMessageDocuments(userWrapper, documents);

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
    turn.aiDiv.appendChild(sourceData);
  }

  renderMath(turn.contentDiv);
  renderDiagrams(turn.contentDiv);
  fixExternalLinks(turn.contentDiv);
}

export function renderStoredAnswer(turn: ChatTurn, html: string): void {
  applyFinalAnswerHtml(turn, html);
}

export function clearChatViewport(): void {
  const chatMessages = document.getElementById('chat-messages');
  if (!chatMessages) return;
  const welcome = document.getElementById('welcome');
  chatMessages.replaceChildren(...(welcome ? [welcome] : []));
  document.querySelector('.app')?.classList.remove('has-messages');
}

export function renderConversationHistory(history: ConversationHistory): void {
  clearChatViewport();
  for (const stored of history.turns) {
    const turn = createChatTurn(stored.user_text, stored.user_images, stored.user_documents);
    renderStoredAnswer(turn, stored.answer_html);
  }
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

export function renderAnswerSaveOutcome(
  turn: ChatTurn,
  presentation: ConversationSavePresentation,
  onRecovery: () => void,
): void {
  if (presentation.state === 'saved' || !presentation.message) return;
  const status = document.createElement('div');
  if (presentation.state === 'not_saved') status.className = chatStyles.textError;
  status.setAttribute('role', 'status');
  status.setAttribute('aria-live', 'polite');
  status.textContent = presentation.message;
  if (presentation.actionLabel) {
    const action = document.createElement('button');
    action.type = 'button';
    action.textContent = presentation.actionLabel;
    action.setAttribute('aria-label', presentation.actionLabel);
    action.addEventListener('click', onRecovery);
    status.append(document.createTextNode(' '), action);
  }
  turn.aiDiv.appendChild(status);
}

export function setAnswerError(turn: ChatTurn, message: unknown): void {
  turn.contentDiv.textContent = typeof message === 'string' ? message : 'Service error. Please try again.';
  turn.contentDiv.classList.add(chatStyles.textError);
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

export function createAnswerRenderer(turn: ChatTurn, options: AnswerRendererOptions = {}) {
  let fullAnswer = '';
  let failed = false;
  let saveOutcome: DonePayload | null = null;
  let scrollScheduled = false;
  let streamStarted = false;
  // Count of leading answer blocks that are complete and frozen: rendered and
  // typeset once, then never touched again so their MathJax output, text
  // selection and scroll position survive the rest of the stream.
  let frozenBlocks = 0;

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
  // answer content arrives, so the block viewport starts from a clean slate.
  function startStreamViewport(): void {
    if (streamStarted) return;
    turn.contentDiv.replaceChildren();
    streamStarted = true;
  }

  // Append raw tokens to a trailing span for sub-preview smoothness; the next
  // preview rebuilds the trailing region and absorbs this tail.
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

  function handleToken(data: SSEData): void {
    const text = parseData(data);
    const token = typeof text === 'string' ? text : String(text);
    startStreamViewport();
    fullAnswer += token;
    appendStreamTail(token);
    scheduleAutoScroll();
  }

  // Server sends the whole accumulated answer HTML each preview. Rather than
  // wiping and re-rendering it all (which re-typesets every equation and drops
  // selection), freeze completed blocks and rebuild only the trailing one.
  function handlePreview(data: SSEData): void {
    const previewHtml = parseData(data);
    const html = typeof previewHtml === 'string' ? previewHtml : '';
    startStreamViewport();
    const blocks = answerBlocksFromHtml(html);
    // Drop the volatile trailing region (previous in-progress block + raw token
    // tail); frozen leading blocks are left untouched.
    while (turn.contentDiv.children.length > frozenBlocks) {
      turn.contentDiv.lastElementChild?.remove();
    }
    for (let i = frozenBlocks; i < blocks.length; i += 1) {
      const block = blocks[i].cloneNode(true) as HTMLElement;
      turn.contentDiv.appendChild(block);
      typesetBlockMath(block);
      // Only a completed (non-trailing) block has a stable fence; the trailing
      // block may still be mid-stream, so its diagram waits for a later preview
      // or the done backstop in applyFinalAnswerHtml.
      if (i < blocks.length - 1) renderDiagrams(block);
      fixExternalLinks(block);
    }
    // Every block except the last is now complete, so freeze them.
    frozenBlocks = Math.max(frozenBlocks, blocks.length - 1);
    scheduleAutoScroll();
  }

  function handleDone(data: SSEData): void {
    const payload = parseData(data);
    const html = typeof payload === 'string' ? payload : (payload as DonePayload).html;
    if (typeof payload !== 'string') {
      fullAnswer = (payload as DonePayload).answer || fullAnswer;
      saveOutcome = payload as DonePayload;
    }

    applyFinalAnswerHtml(turn, html || '');
    const live = turn.aiDiv.querySelector('.sr-only');
    if (live) live.textContent = 'Answer ready';
  }

  function handleHighlights(data: SSEData): void {
    const highlightsHtml = parseData(data);
    const sourceData = turn.aiDiv.querySelector('.source-data');
    if (sourceData) {
      // Update this turn's stored sources so opening the panel shows highlights.
      // The server also persists them, so history and reloads stay in sync.
      setSanitizedLlmHtml(sourceData, typeof highlightsHtml === 'string' ? highlightsHtml : '');
    }
  }

  function handleProgress(data: SSEData): void {
    const info = parseData(data) as ProgressPayload;
    const label: string = PHASE_LABELS[info.phase as PhaseLabel] || info.phase;
    // Resolve the phase indicator, creating one when tokens have already
    // replaced the initial streaming dot so late phases (e.g. "Saving…") stay
    // visible. The done event replaces contentDiv, clearing any created node.
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
    handle(eventType: string, data: SSEData): void {
      if (eventType === 'token') handleToken(data);
      else if (eventType === 'preview') handlePreview(data);
      else if (eventType === 'done') handleDone(data);
      else if (eventType === 'highlights') handleHighlights(data);
      else if (eventType === 'progress') handleProgress(data);
      else if (eventType === 'warning') {
        notifyAnswerWarning(parseData(data), options.onWarning);
      }
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
    get saveOutcome(): DonePayload | null {
      return saveOutcome;
    },
  };
}
