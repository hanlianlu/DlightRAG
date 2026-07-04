// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import { bus } from '../events/bus';
import { conversationStore } from '../stores/conversationStore';
import chatStyles from '../styles/chat.module.css';

let confirmTimer: ReturnType<typeof setTimeout> | null = null;

function showClearButton(): void {
  const btn = document.getElementById('clear-history-btn');
  if (!btn) return;
  btn.classList.remove('hidden');
}

function hideClearButton(): void {
  const btn = document.getElementById('clear-history-btn');
  if (!btn) return;
  btn.classList.add('hidden');
  resetConfirmState(btn);
}

function resetConfirmState(btn: HTMLElement): void {
  if (confirmTimer) {
    clearTimeout(confirmTimer);
    confirmTimer = null;
  }
  btn.textContent = 'Clear';
  btn.classList.remove('clear-history-confirm');
}

function restoreWelcomeState(): void {
  const app = document.querySelector('.app');
  if (app) app.classList.remove('has-messages');

  const messages = document.getElementById('chat-messages');
  if (!messages) return;

  // Remove all user/ai message nodes, keep the welcome element
  const toRemove: Element[] = [];
  messages.querySelectorAll(
    `.${chatStyles.userMessageWrapper}, .${chatStyles.aiMessage}, .${chatStyles.contextDivider}`,
  ).forEach(function (el) {
    toRemove.push(el);
  });
  toRemove.forEach(function (el) {
    el.remove();
  });
}

export function setupClearHistory(): void {
  const btn = document.getElementById('clear-history-btn');
  if (!btn) return;

  // Show the button when a conversation starts
  bus.on('chatExchangeComplete', showClearButton);

  // If history was restored on page load, show the button
  bus.on('chatHistoryRestored', showClearButton);

  // After clearing, reset the UI
  bus.on('chatHistoryCleared', function () {
    restoreWelcomeState();
    hideClearButton();
  });

  // Two-step confirmation: first click → "Clear?", second click within 3s → confirm
  btn.addEventListener('click', async function () {
    if (btn.classList.contains('clear-history-confirm')) {
      resetConfirmState(btn);
      await conversationStore.clear();
      return;
    }

    btn.textContent = 'Clear this session?';
    btn.classList.add('clear-history-confirm');
    confirmTimer = setTimeout(function () {
      resetConfirmState(btn);
    }, 3000);
  });

  // Show button immediately if there's already history
  if (conversationStore.historyWindow.length > 0) {
    showClearButton();
  }
}
