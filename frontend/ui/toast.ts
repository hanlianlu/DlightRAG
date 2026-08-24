// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** One replace-in-place toast surface with optional asynchronous action. */

let toastTimer: ReturnType<typeof setTimeout> | null = null;
let remainingDuration = 0;

export interface ActionToastOptions {
  actionLabel: string;
  onAction: () => Promise<string | void>;
  duration?: number;
}

function toastElement(): HTMLElement | null {
  return document.getElementById('toast');
}

function stopTimer(): void {
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = null;
}

function hideToast(): void {
  stopTimer();
  toastElement()?.classList.remove('visible');
}

function scheduleHide(duration: number): void {
  stopTimer();
  remainingDuration = duration;
  toastTimer = setTimeout(hideToast, duration);
}

function setMessage(el: HTMLElement, message: string): HTMLSpanElement {
  const text = document.createElement('span');
  text.className = 'toast-message';
  text.textContent = message;
  el.replaceChildren(text);
  return text;
}

export function showToast(message: string, duration = 3000): void {
  const el = toastElement();
  if (!el) return;
  setMessage(el, message);
  el.classList.add('visible');
  scheduleHide(duration);
}

export function showActionToast(message: string, options: ActionToastOptions): void {
  const el = toastElement();
  if (!el) return;
  const text = setMessage(el, message);
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'ui-btn toast-action';
  button.textContent = options.actionLabel;
  el.appendChild(button);
  el.classList.add('visible');

  const duration = options.duration ?? 12_000;
  const pause = (): void => stopTimer();
  const resume = (): void => scheduleHide(remainingDuration || duration);
  el.onmouseenter = pause;
  el.onmouseleave = resume;
  el.addEventListener('focusin', pause, {once: true});
  el.addEventListener('focusout', resume, {once: true});

  button.addEventListener('click', async () => {
    stopTimer();
    button.disabled = true;
    try {
      const next = await options.onAction();
      text.textContent = next || 'Profile Memory change undone.';
      button.remove();
      scheduleHide(3000);
    } catch {
      text.textContent = 'Could not undo the Profile Memory change.';
      button.remove();
      scheduleHide(5000);
    }
  }, {once: true});
  scheduleHide(duration);
}
