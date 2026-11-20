// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The page's completion notification: the one constructor of a browser Notification for a Run.

 * The store observes when a Run settles; this element turns that into one browser Notification for
 * every conversation, the open one included, but only while the reader is away from the page and only
 * with permission already granted. A goal round announces nothing per round — a per-round notice would
 * claim the conversation finished while the goal is still running — except when the projected goal
 * itself has ended: a complete or blocked goal gets one notice, because the goal bar that carries the
 * state is behind the page the reader left. A human turn that ends the goal in its own Run gets the
 * same goal notice instead of the answer notice, so the reader is told the stronger fact.
 * No permission is asked here: the offer (notifications.ts) asks, and this reads the answer it got.
 * Nothing visual is drawn; the element exists as a Feature.
 */

import {msg, str} from '@lit/localize';
import {html, type TemplateResult} from 'lit';
import {type ConversationGoal, readConversationGoal} from '../api/goal.ts';
import type {RunActivity} from '../api/run-activity.ts';
import {LightElement} from '../lib/lit-host.ts';
import {conversationRoute} from '../lib/router.ts';
import {type AppHandles, productionHandles} from '../stores/app-handles.ts';
import type {RunActivityStore} from '../stores/run-activity-store.ts';
import {webRouter} from './router.ts';

/** Whether the reader is away: tab hidden, or another window has focus. */
function away(): boolean {
  // App switching can preserve visibility; focus completes the browser signal.
  return document.hidden || !document.hasFocus();
}

function supported(): boolean {
  return typeof window !== 'undefined' && 'Notification' in window;
}

/** How long the notifier waits for the settlement driver to write the terminal goal phase.
 *  The driver observes the same settlement the page does and may write `complete`/`blocked`
 *  just after it, so a single immediate read would miss the goal's own outcome. A human turn
 *  writes it inside its own Run, so that read is not retried. */
const GOAL_OUTCOME_ATTEMPTS = 3;
const GOAL_OUTCOME_DELAY_MS = 400;

/** The announced ids are a session-local memory, not a log: keep the recent window bounded. */
const ANNOUNCED_GOALS_LIMIT = 128;

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

/** Announces a settled Run, whichever conversation it belongs to. */
export class DlRunNotifications extends LightElement {
  static properties = {
    handles: {attribute: false},
    enabled: {attribute: false},
  };

  declare handles: AppHandles;
  declare enabled: boolean;

  /** The store whose settlements are followed; the bag can be handed in after construction. */
  #followed: RunActivityStore | null = null;
  #release: (() => void) | null = null;
  /** Goal outcomes already announced, so a repeated settlement is one notice; bounded. */
  readonly #announcedGoals = new Set<string>();

  constructor() {
    super();
    this.handles = productionHandles();
    this.enabled = false;
  }

  override connectedCallback(): void {
    super.connectedCallback();
    this.#follow();
  }

  override disconnectedCallback(): void {
    this.#release?.();
    this.#release = null;
    this.#followed = null;
    this.#announcedGoals.clear();
    super.disconnectedCallback();
  }

  protected override updated(): void {
    this.#follow();
  }

  protected override render(): TemplateResult {
    return html``;
  }

  #follow(): void {
    const store = this.handles.runActivity;
    if (store === this.#followed) return;
    this.#release?.();
    this.#followed = store;
    this.#release = store.subscribeSettled((run) => { this.#settled(run); });
  }

  #settled(run: RunActivity): void {
    if (!this.enabled) return;
    if (run.goalRound) {
      // The driver writes the terminal goal phase after the settlement it observed.
      void this.#announceGoalOutcome(run, {retry: true});
      return;
    }
    if (!this.#canNotify()) return;
    // The answer notice stays synchronous, exactly as before. A human turn can also be the turn
    // that ended the goal (its own tool call wrote `complete`/`blocked`), so the projected phase is
    // read beside it and a terminal goal adds the goal's own notice. That read decides only the
    // extra notice; it never delays the answer notification the reader already earned.
    this.#show(run, msg('Answer ready', {id: 'notifications.answerReadyTitle'}), {
      body: run.conversationTitle
        ? msg(str`Your answer for ${run.conversationTitle} is ready.`, {
          id: 'runNotifications.answerReadyBody',
        })
        : msg('DlightRAG finished generating your answer.', {
          id: 'notifications.answerReadyBody',
        }),
    });
    void this.#announceHumanGoalEnd(run);
  }

  /** The goal's outcome when a human turn itself completed or blocked it. */
  async #announceHumanGoalEnd(run: RunActivity): Promise<void> {
    if (!this.#canNotify()) return;
    if (this.#announcedGoals.has(run.runId)) return;
    let goal: ConversationGoal | null;
    try {
      goal = await readConversationGoal(run.conversationId);
    } catch {
      return;
    }
    if (goal === null || !this.#isTerminal(goal)) return;
    this.#showGoalOutcome(run, goal);
  }

  /** One notice for a goal that ended: complete, or blocked until a human resumes it. */
  async #announceGoalOutcome(run: RunActivity, options: {retry: boolean}): Promise<void> {
    if (!this.#canNotify()) return;
    if (this.#announcedGoals.has(run.runId)) return;
    const attempts = options.retry ? GOAL_OUTCOME_ATTEMPTS : 1;
    for (let attempt = 0; attempt < attempts; attempt += 1) {
      if (attempt > 0) await delay(GOAL_OUTCOME_DELAY_MS);
      let goal: ConversationGoal | null;
      try {
        goal = await readConversationGoal(run.conversationId);
      } catch {
        return;
      }
      if (goal === null) return;
      if (!this.#isTerminal(goal)) continue;
      this.#showGoalOutcome(run, goal);
      return;
    }
  }

  #isTerminal(goal: ConversationGoal): boolean {
    return goal.phase === 'complete' || goal.phase === 'blocked';
  }

  #showGoalOutcome(run: RunActivity, goal: ConversationGoal): void {
    const complete = goal.phase === 'complete';
    const capBlocked = !complete && goal.roundsLeft === 0;
    const titled = run.conversationTitle !== null && run.conversationTitle.trim() !== '';
    this.#rememberAnnounced(run.runId);
    this.#show(
      run,
      complete
        ? msg('Goal complete', {id: 'notifications.goalCompleteTitle'})
        : msg('Goal blocked', {id: 'notifications.goalBlockedTitle'}),
      {
        body: complete
          ? titled
            ? msg(str`The goal "${run.conversationTitle}" is complete.`, {
              id: 'runNotifications.goalCompleteBody',
            })
            : msg('The goal in this conversation is complete.', {
              id: 'notifications.goalCompleteBody',
            })
          : capBlocked
            ? titled
              ? msg(str`The goal "${run.conversationTitle}" reached its round cap; raise the cap to continue.`, {
                id: 'runNotifications.goalCapBlockedBody',
              })
              : msg('The goal in this conversation reached its round cap; raise the cap to continue.', {
                id: 'notifications.goalCapBlockedBody',
              })
            : titled
              ? msg(str`The goal "${run.conversationTitle}" is blocked; open the conversation to resume it.`, {
                id: 'runNotifications.goalBlockedBody',
              })
              : msg('The goal in this conversation is blocked; open it to resume.', {
                id: 'notifications.goalBlockedBody',
              }),
      },
    );
  }

  #rememberAnnounced(runId: string): void {
    this.#announcedGoals.add(runId);
    while (this.#announcedGoals.size > ANNOUNCED_GOALS_LIMIT) {
      const oldest = this.#announcedGoals.values().next().value;
      if (oldest === undefined) return;
      this.#announcedGoals.delete(oldest);
    }
  }

  #canNotify(): boolean {
    return away() && supported() && Notification.permission === 'granted';
  }

  #show(run: RunActivity, title: string, options: NotificationOptions): void {
    try {
      const notification = new Notification(title, options);
      notification.onclick = () => {
        window.focus();
        notification.close();
        this.handles.runActivity.markConversationSeen(run.conversationId);
        void webRouter.navigate(conversationRoute(run.conversationId));
      };
    } catch {
      // Some browsers reject construction outside a service worker.
    }
  }
}

customElements.define('dl-run-notifications', DlRunNotifications);

declare global {
  interface HTMLElementTagNameMap {
    'dl-run-notifications': DlRunNotifications;
  }
}
