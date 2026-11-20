// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The pieces more than one Settings page draws or does, so each is done one way. */

import {html, type LitElement, nothing, type TemplateResult} from 'lit';
import styles from '../styles/settings-page.module.css';
import {requestToast} from './toast-request.ts';

interface SwitchCard {
  /** The switch's own id; the ids of its two texts derive from it. */
  id: string;
  label: string;
  caption: string;
  checked: boolean;
  disabled?: boolean;
  /** The switch is off for a reason the owner cannot change, so its label reads quieter. */
  muted?: boolean;
  onToggle: (event: Event) => void;
}

/** A card holding one switch and the sentence that says what it does now.
 *
 * The card is the switch's label, so a tap anywhere on it turns the switch: the control itself is
 * a few pixels tall, and the row it sits in is the target a finger can find. The switch keeps its
 * own name and description for assistive technology.
 */
export function switchCard(card: SwitchCard): TemplateResult {
  const label = `${card.id}-label`;
  const caption = `${card.id}-caption`;
  return html`
    <label class="${styles.card} ${styles.row}">
      <span class=${styles.rowText}>
        <span id=${label} class="${styles.rowLabel} ${card.muted ? styles.rowLabelMuted : ''}"
          >${card.label}</span>
        <span id=${caption} class=${styles.rowCaption}>${card.caption}</span>
      </span>
      <button id=${card.id} class="dl-switch dl-switch--dense" type="button" role="switch"
        aria-checked=${String(card.checked)} aria-labelledby=${label} aria-describedby=${caption}
        ?disabled=${card.disabled} @click=${card.onToggle}></button>
    </label>`;
}

/** A page whose owner-level switch is committed one command at a time, held in its `pending`. */
type SwitchHost = LitElement & {pending: boolean};

interface SwitchCommit<Reply> {
  /** The page's lifetime: once it has ended, the commit changes nothing and says nothing. */
  signal: AbortSignal;
  /** Send the value the switch was pressed to; the server answers with what it stored. */
  send: (signal: AbortSignal) => Promise<Reply>;
  /** The reply replaces what the page held, so the switch shows the server's state and never a guess. */
  accept: (reply: Reply) => void;
  /** The notice when the command fails; the switch then stays as the page last read it. */
  failure: string;
  /** The page's own work as the command starts, such as dropping the reads it makes stale. */
  begin?: () => void;
  /** The page's own work once the command has settled, whichever way it went. */
  settled?: () => void;
}

/** Commit one press of a Settings switch whose setting the server stores for the owner.
 *
 * Pessimistic: the page holds its `pending` for the request, so the switch cannot be pressed again
 * meanwhile, and it shows the state the server answers with, never the one the press asked for. A
 * failed request leaves the switch where it was and says so. The reader keeps their place: a switch
 * that was disabled for the request drops focus in some engines, so it gets focus back.
 */
export async function commitSwitch<Reply>(host: SwitchHost, event: Event, commit: SwitchCommit<Reply>): Promise<void> {
  const {signal} = commit;
  const toggle = event.currentTarget as HTMLElement;
  if (signal.aborted || host.pending) return;
  const focused = document.activeElement === toggle;
  host.pending = true;
  commit.begin?.();
  try {
    const reply = await commit.send(signal);
    if (!signal.aborted) commit.accept(reply);
  } catch {
    if (!signal.aborted) requestToast(host, {message: commit.failure, tone: 'error'});
  } finally {
    if (!signal.aborted) {
      host.pending = false;
      commit.settled?.();
      await host.updateComplete;
      if (focused) toggle.focus();
    }
  }
}

interface LoadState {
  /** The page; it is where focus is looked for once a retry has been answered. */
  host: LitElement;
  /** The first read failed; otherwise it is under way. */
  failed: boolean;
  /** What the page says while it reads. */
  loading: string;
  /** What it says when the read failed. */
  failure: string;
  /** The label of the button that reads again. */
  retry: string;
  /** Read again; it settles once the read has been answered, whichever way it went. */
  onRetry: () => Promise<void>;
}

/** A control the reader can act on, or the empty-state heading a page made focusable for them. */
const FOCUSABLE =
  'button:not(:disabled), a[href], input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex]';

/** What a page shows until it has read what it is about: that it is reading, or that it could not,
 * with a way to try again. Until then the page draws none of its own controls, so nothing unread
 * can pass for its state.
 *
 * The Retry that was pressed is replaced by what answers: the page, or this note again. A reader who
 * pressed it from the keyboard would be left on nothing, so focus follows to the first control the
 * page now shows, which is the new Retry when the read failed again.
 */
export function loadState(state: LoadState): TemplateResult {
  const retry = async (event: Event): Promise<void> => {
    const focused = document.activeElement === event.currentTarget;
    await state.onRetry();
    if (!focused || !state.host.isConnected) return;
    await state.host.updateComplete;
    // A closed dialog of the page holds buttons too, and none of them can take focus.
    [...state.host.querySelectorAll<HTMLElement>(FOCUSABLE)].find((element) => !element.closest('dialog:not([open])'))?.focus();
  };
  return state.failed
    ? html`
      <div class=${styles.failed}>
        <p class=${styles.note} role="alert">${state.failure}</p>
        <button type="button" class="dl-btn" @click=${retry}>${state.retry}</button>
      </div>`
    : html`<p class=${styles.hint} role="status">${state.loading}</p>`;
}

interface DangerCard {
  /** What the card is for, and the name of its button; a phone's button wears it in full. */
  action: string;
  /** The button's words beside a pointer, where the card's own label is already next to it. */
  short: string;
  caption: string;
  /** The caption is a figure that changes while it shows, so a screen reader is told when it does. */
  live?: boolean;
  /** Whether the button stacks under the text at full width, and so has room for the full action. */
  phone: boolean;
  disabled?: boolean;
  onClick: (event: Event) => void;
}

/** A card with one destructive action and the sentence that says what it will do. */
export function dangerCard(card: DangerCard): TemplateResult {
  return html`
    <div class="${styles.card} ${styles.row} ${styles.dangerRow}">
      <span class=${styles.rowText}>
        <span class=${styles.rowLabel}>${card.action}</span>
        <span class=${styles.rowCaption} aria-live=${card.live ? 'polite' : nothing}>${card.caption}</span>
      </span>
      <button type="button" class="dl-btn dl-btn-danger-text" aria-label=${card.action}
        ?disabled=${card.disabled} @click=${card.onClick}>${card.phone ? card.action : card.short}</button>
    </div>`;
}
