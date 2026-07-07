// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {conversationStore} from '../stores/conversationStore.ts';
import {renderMessageImages} from '../ui/images.ts';
import {renderMath} from '../ui/mathjax.ts';
import {parseData} from './sse.ts';
import {llmFragmentFromSanitizedHtml, setSanitizedLlmHtml} from './safe_html.ts';
import chatStyles from '../styles/chat.module.css';

// SSE HTML payloads are sanitized server-side by nh3 and again here before
// browser insertion. Keep new HTML sinks behind frontend/lib/safe_html.ts.

// ── types ────────────────────────────────────────────────────────────

interface ChatTurn {
  chatArea: HTMLElement;
  aiDiv: HTMLDivElement;
  contentDiv: HTMLDivElement;
}

type SSEData = string;

interface DonePayload {
  html?: string;
  answer?: string;
  current_image_ids?: string[];
}

interface MetaPayload {
  history_kept?: number;
}

interface ProgressPayload {
  phase: string;
}

type PhaseLabel = 'planning' | 'searching' | 'generating';

const PHASE_LABELS: Record<PhaseLabel, string> = {
  planning: 'Analyzing query...',
  searching: 'Searching knowledge base...',
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
    if (!a.getAttribute('target')) {
      a.setAttribute('target', '_blank');
      a.setAttribute('rel', 'noopener noreferrer');
    }
  });
}

function markOutOfContext(historyKept: number): void {
  const chatMessages = document.getElementById('chat-messages');
  if (!chatMessages) return;
  const outOfContextCount = conversationStore.historyWindow.length - historyKept;
  if (outOfContextCount <= 0) {
    chatMessages.querySelectorAll('.' + chatStyles.outOfContext).forEach(function (el) {
      el.classList.remove(chatStyles.outOfContext);
    });
    const old = chatMessages.querySelector('.' + chatStyles.contextDivider);
    if (old) old.remove();
    return;
  }

  let msgIndex = 0;
  let dividerInsertBefore: Element | null = null;
  Array.from(chatMessages.children).forEach(function (node) {
    if (node.classList.contains(chatStyles.contextDivider)) return;
    if (msgIndex < outOfContextCount) {
      node.classList.add(chatStyles.outOfContext);
    } else {
      node.classList.remove(chatStyles.outOfContext);
      if (dividerInsertBefore === null) dividerInsertBefore = node;
    }
    msgIndex++;
  });

  const existing = chatMessages.querySelector('.' + chatStyles.contextDivider);
  if (existing) existing.remove();
  if (dividerInsertBefore) {
    const divider = document.createElement('div');
    divider.className = chatStyles.contextDivider;
    divider.textContent = 'Above messages are outside AI memory';
    chatMessages.insertBefore(divider, dividerInsertBefore);
  }
}

function scrollToBottom(turn: ChatTurn): void {
  if (turn.chatArea) turn.chatArea.scrollTop = turn.chatArea.scrollHeight;
}

// ── public API ────────────────────────────────────────────────────────

// Cap rendered chat DOM nodes (~HISTORY_CAP turns x user+ai) so long-lived
// sessions don't accumulate an unbounded number of message nodes.
const MAX_CHAT_MESSAGE_NODES = 200;

function pruneOldMessages(chatMessages: HTMLElement): void {
  while (chatMessages.childElementCount > MAX_CHAT_MESSAGE_NODES) {
    chatMessages.firstElementChild?.remove();
  }
}

export function createChatTurn(query: string): ChatTurn {
  const chatMessages = document.getElementById('chat-messages')!;
  const chatArea = document.getElementById('chat-area')!;

  activateChatMode();

  const userWrapper = document.createElement('div');
  userWrapper.className = chatStyles.userMessageWrapper;
  renderMessageImages(userWrapper);

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

  chatMessages.appendChild(aiDiv);
  pruneOldMessages(chatMessages);

  const turn: ChatTurn = {chatArea, aiDiv, contentDiv};
  scrollToBottom(turn);
  return turn;
}

export function setAnswerError(turn: ChatTurn, message: unknown): void {
  turn.contentDiv.textContent = typeof message === 'string' ? message : 'Service error. Please try again.';
  turn.contentDiv.classList.add(chatStyles.textError);
}

export function createAnswerRenderer(turn: ChatTurn) {
  let firstToken = true;
  let fullAnswer = '';
  let imageIds: string[] = [];
  let failed = false;

  function handleToken(data: SSEData): void {
    const text = parseData(data);
    const token = typeof text === 'string' ? text : String(text);
    if (firstToken) {
      turn.contentDiv.textContent = '';
      firstToken = false;
    }
    fullAnswer += token;
    const span = document.createElement('span');
    span.textContent = token;
    turn.contentDiv.appendChild(span);
    scrollToBottom(turn);
  }

  function handlePreview(data: SSEData): void {
    const previewHtml = parseData(data);
    setSanitizedLlmHtml(turn.contentDiv, typeof previewHtml === 'string' ? previewHtml : '');
    renderMath(turn.contentDiv);
    fixExternalLinks(turn.contentDiv);
    scrollToBottom(turn);
  }

  function handleDone(data: SSEData): void {
    const payload = parseData(data);
    const html = typeof payload === 'string' ? payload : (payload as DonePayload).html;
    if (typeof payload !== 'string') {
      fullAnswer = (payload as DonePayload).answer || fullAnswer;
      imageIds = (payload as DonePayload).current_image_ids || [];
    }

    const fragment = llmFragmentFromSanitizedHtml(html || '');
    const answerContent = fragment.querySelector('#answer-content');
    const sourceData = fragment.querySelector('#source-data');
    const imageStrip = fragment.querySelector('.answer-image-strip');
    const refList = fragment.querySelector('.answer-references');

    if (answerContent) {
      const answerNodes = Array.from(answerContent.childNodes).map((node) => node.cloneNode(true));
      turn.contentDiv.replaceChildren(...answerNodes);
    }
    if (imageStrip) {
      turn.contentDiv.appendChild(imageStrip.cloneNode(true));
    }
    if (refList) {
      turn.contentDiv.appendChild(refList.cloneNode(true));
    }
    if (sourceData) {
      (sourceData as HTMLElement).className = 'source-data hidden';
      sourceData.removeAttribute('id');
      turn.aiDiv.appendChild(sourceData);
    }

    renderMath(turn.contentDiv);
    fixExternalLinks(turn.contentDiv);
  }

  function handleHighlights(data: SSEData): void {
    const highlightsHtml = parseData(data);
    const sourceData = turn.aiDiv.querySelector('.source-data');
    if (sourceData) {
      setSanitizedLlmHtml(sourceData, typeof highlightsHtml === 'string' ? highlightsHtml : '');
    }
  }

  function handleMeta(data: SSEData): void {
    const meta = parseData(data) as MetaPayload;
    if (typeof meta.history_kept === 'number') markOutOfContext(meta.history_kept);
  }

  function handleProgress(data: SSEData): void {
    const info = parseData(data) as ProgressPayload;
    const label: string = PHASE_LABELS[info.phase as PhaseLabel] || info.phase;
    const dot = turn.contentDiv.querySelector('.' + chatStyles.streamingDot);
    if (dot) {
      dot.textContent = '';
      dot.className = chatStyles.streamingDot + ' ' + chatStyles.progressPhase;
      dot.setAttribute('data-phase', label || '');
    }
  }

  return {
    handle(eventType: string, data: SSEData): void {
      if (eventType === 'token') handleToken(data);
      else if (eventType === 'preview') handlePreview(data);
      else if (eventType === 'done') handleDone(data);
      else if (eventType === 'highlights') handleHighlights(data);
      else if (eventType === 'meta') handleMeta(data);
      else if (eventType === 'progress') handleProgress(data);
      else if (eventType === 'error') {
        failed = true;
        setAnswerError(turn, parseData(data));
      }
    },
    get answer(): string {
      return fullAnswer;
    },
    get imageIds(): string[] {
      return imageIds;
    },
    get failed(): boolean {
      return failed;
    },
  };
}
