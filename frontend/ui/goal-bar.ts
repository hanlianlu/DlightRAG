// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The Goal bar above the composer: the conversation's one long objective, and the reader's controls.
 *
 * It draws nothing while the conversation carries no goal, which is most conversations. A live goal
 * shows its phase and round, the objective, and a blocked condition when one stopped continuation.
 * The platform continues an active goal on its own — including after a restart — so `active` needs
 * no resume control: the reader may pause it (which stops the goal now and cancels the round in
 * progress), resume a paused or blocked goal, edit its objective, or clear it. `/goal` in the
 * composer is the same surface without the bar.
 *
 * The bar owns its read: it asks again whenever the conversation changes, after a command, when a
 * goal-writing Tool settles (the Agent creating or moving the goal mid-Run), and when a turn
 * settles, so a goal the Agent created while this tab watched appears without a reload.
 *
 * Visually it is a card in the composer's column, not a toolbar across the window: the same width and
 * left edge as the box the reader types in, the status in a pill, the objective as the card's one large
 * line, and the controls as the design system's dense icon buttons at its end.
 */

import {msg} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {
  type ConversationGoal,
  type GoalAction,
  GoalCommandError,
  readConversationGoal,
  sendGoalCommand,
} from '../api/goal.ts';
import {type IconName, icon } from '../design-system/index.ts';
import {goalPhaseLabel} from '../lib/goal-phase.ts';
import {LightElement} from '../lib/lit-host.ts';
import {localizedText} from '../lib/localized-text.ts';
import {goalCommandErrorText} from '../lib/web-commands.ts';
import styles from '../styles/goal-bar.module.css';

export class DlGoalBar extends LightElement {
  static override properties = {
    conversationId: {attribute: false},
  };

  declare conversationId: string | null;
  #goal: ConversationGoal | null = null;
  #editing = false;
  #draft = '';
  #capPrompt = false;
  #capDraft = '';
  #capError: string | null = null;
  #busy = false;
  #failure: string | null = null;

  override connectedCallback(): void {
    super.connectedCallback();
    void this.#load();
  }

  protected override updated(changed: PropertyValues<this>): void {
    super.updated(changed);
    if (changed.has('conversationId')) {
      this.#editing = false;
      this.#capPrompt = false;
      this.#capError = null;
      void this.#load();
    }
  }

  /** Read the goal again; the chat calls this after a settled turn or a `/goal` command. */
  refresh(): void {
    void this.#load();
  }

  async #load(): Promise<void> {
    const conversationId = this.conversationId ?? null;
    if (!conversationId) {
      this.#goal = null;
      this.requestUpdate();
      return;
    }
    try {
      const goal = await readConversationGoal(conversationId, this.lifetime);
      if (!this.isConnected || this.conversationId !== conversationId) return;
      this.#goal = goal;
      if (goal === null || goal.roundsLeft > 0) {
        this.#capPrompt = false;
        this.#capError = null;
      }
    } catch {
      if (!this.isConnected) return;
      // A bar that cannot read a goal draws nothing; the conversation itself is unaffected.
      this.#goal = null;
    }
    this.requestUpdate();
  }

  /** Tell the chat that a resume may have started the goal's next round server-side. */
  #announceResume(): void {
    this.dispatchEvent(new CustomEvent('dl-goal-resumed', {bubbles: true, composed: true}));
  }

  /** The words a refused command shows: the localized reason for its stable code, else the
   *  server's own sentence (a later domain reason is still readable), else a generic failure.
   *  A mapped code is deliberately generic, so the server's sentence is appended when it adds
   *  the actual condition ("pause needs an active goal; this one is complete"). */
  #commandFailure(error: unknown): string {
    if (error instanceof GoalCommandError) {
      const detail = error.message.trim();
      const localized = goalCommandErrorText(error.kind);
      if (localized !== null) {
        const head = localizedText(localized);
        return detail === '' || detail === localized.source ? head : `${head} ${detail}`;
      }
      if (detail !== '') return detail;
    }
    return msg('The goal command failed.', {id: 'goalBar.commandFailed'});
  }

  async #command(
    action: GoalAction,
    extra: {objective?: string; maxGoalRounds?: number} = {},
  ): Promise<void> {
    const conversationId = this.conversationId ?? null;
    if (!conversationId || this.#busy) return;
    this.#busy = true;
    this.requestUpdate();
    try {
      const goal = await sendGoalCommand(conversationId, {...extra, action}, this.lifetime);
      if (!this.isConnected || this.conversationId !== conversationId) return;
      this.#goal = goal;
      this.#editing = false;
      this.#capPrompt = false;
      this.#capError = null;
      this.#failure = null;
      if (action === 'resume') this.#announceResume();
    } catch (error) {
      if (!this.isConnected) return;
      this.#failure = this.#commandFailure(error);
    } finally {
      this.#busy = false;
      if (this.isConnected) this.requestUpdate();
    }
  }

  #beginEdit(): void {
    if (!this.#goal) return;
    this.#draft = this.#goal.objective;
    this.#failure = null;
    this.#capPrompt = false;
    this.#capError = null;
    this.#editing = true;
    this.requestUpdate();
    void this.updateComplete.then(() => this.querySelector('textarea')?.focus());
  }

  #saveEdit(): void {
    const objective = this.#draft.trim();
    if (!objective) return;
    void this.#command('edit', {objective});
  }

  /** Resume normally; at the round cap, ask for a higher cap instead of sending a refused resume. */
  #beginResume(): void {
    const goal = this.#goal;
    if (!goal) return;
    if (goal.roundsLeft > 0) {
      void this.#command('resume');
      return;
    }
    this.#capDraft = String(goal.roundsStarted + 1);
    this.#capError = null;
    this.#failure = null;
    this.#editing = false;
    this.#capPrompt = true;
    this.requestUpdate();
    void this.updateComplete.then(() => this.querySelector<HTMLInputElement>('#goal-cap-input')?.focus());
  }

  #cancelCapPrompt(): void {
    this.#capPrompt = false;
    this.#capError = null;
    this.requestUpdate();
  }

  async #raiseCapAndResume(): Promise<void> {
    const goal = this.#goal;
    const conversationId = this.conversationId ?? null;
    if (!goal || !conversationId || this.#busy) return;
    const cap = Number(this.#capDraft);
    if (!Number.isSafeInteger(cap) || cap <= goal.roundsStarted) {
      this.#capError = msg('The cap must be a whole number above the rounds already started.', {
        id: 'goalBar.invalidCap',
      });
      this.requestUpdate();
      return;
    }
    this.#busy = true;
    this.#capError = null;
    this.requestUpdate();
    let edited = false;
    try {
      const raised = await sendGoalCommand(
        conversationId,
        {action: 'edit', maxGoalRounds: cap},
        this.lifetime,
      );
      if (!this.isConnected || this.conversationId !== conversationId) return;
      if (raised === null) throw new Error('goal edit returned no goal');
      this.#goal = raised;
      edited = true;
      const resumed = await sendGoalCommand(conversationId, {action: 'resume'}, this.lifetime);
      if (!this.isConnected || this.conversationId !== conversationId) return;
      if (resumed === null) throw new Error('goal resume returned no goal');
      this.#goal = resumed;
      this.#capPrompt = false;
      this.#failure = null;
      this.#announceResume();
    } catch (error) {
      if (!this.isConnected) return;
      if (edited) {
        this.#capPrompt = false;
        this.#failure = this.#commandFailure(error);
      } else {
        this.#capError = this.#commandFailure(error);
      }
    } finally {
      this.#busy = false;
      if (this.isConnected) this.requestUpdate();
    }
  }

  /** The design system's dense icon control: a mark a reader can press in a line of words. */
  #iconButton(
    name: IconName,
    label: string,
    onClick: () => void,
    options: {danger?: boolean; hint?: string; ariaDescription?: string} = {},
  ): TemplateResult {
    return html`
      <button
        class="dl-icon-button ${options.danger ? styles.danger : ''}"
        type="button"
        aria-label=${label}
        aria-description=${options.ariaDescription ?? nothing}
        title=${options.hint ?? label}
        ?disabled=${this.#busy}
        @click=${onClick}
      >${icon(name, {size: 'xs'})}</button>
    `;
  }

  #editingBody(): TemplateResult {
    return html`
      <textarea
        class=${styles.edit}
        .value=${this.#draft}
        placeholder=${msg('Objective', {id: 'goalBar.objective'})}
        aria-label=${msg('Objective', {id: 'goalBar.objective'})}
        @input=${(event: Event) => { this.#draft = (event.target as HTMLTextAreaElement).value; }}
      ></textarea>
      <div class=${styles.editActions}>
        <button class="dl-btn ${styles.save}" type="button" ?disabled=${this.#busy} @click=${this.#saveEdit}>
          ${msg('Save', {id: 'goalBar.save'})}
        </button>
        <button class="dl-btn" type="button" @click=${() => { this.#editing = false; this.requestUpdate(); }}>
          ${msg('Cancel', {id: 'goalBar.cancel'})}
        </button>
      </div>
    `;
  }

  #capPromptBody(): TemplateResult {
    const goal = this.#goal;
    if (!goal) return html``;
    return html`
      <section
        class=${styles.capPrompt}
        role="dialog"
        aria-labelledby="goal-cap-title"
        aria-describedby="goal-cap-copy"
      >
        <div class=${styles.capPromptHeading}>
          <span class=${styles.capPromptTitle} id="goal-cap-title">
            ${msg('Goal reached its round cap', {id: 'goalBar.raiseCapTitle'})}
          </span>
          <span class=${styles.capPromptCopy} id="goal-cap-copy">
            ${msg('This goal has used all its rounds. Raise the cap to continue.', {
              id: 'goalBar.raiseCapCopy',
            })}
          </span>
        </div>
        <div class=${styles.capPromptRow}>
          <label class=${styles.capPromptLabel} for="goal-cap-input">
            ${msg('Max rounds', {id: 'goalBar.maxRounds'})}
          </label>
          <input
            id="goal-cap-input"
            class=${styles.capPromptInput}
            type="number"
            inputmode="numeric"
            min=${goal.roundsStarted + 1}
            .value=${this.#capDraft}
            aria-label=${msg('Max rounds', {id: 'goalBar.maxRounds'})}
            @input=${(event: Event) => {
              this.#capDraft = (event.target as HTMLInputElement).value;
            }}
            @keydown=${(event: KeyboardEvent) => {
              if (event.key === 'Enter') {
                event.preventDefault();
                void this.#raiseCapAndResume();
              }
              if (event.key === 'Escape') {
                event.preventDefault();
                this.#cancelCapPrompt();
              }
            }}
          >
          <span class=${styles.capPromptHint}>
            ${msg('Must be above the rounds already started.', {id: 'goalBar.raiseCapHint'})}
          </span>
        </div>
        ${this.#capError
          ? html`<p class=${styles.capPromptError} role="alert">${this.#capError}</p>`
          : nothing}
        <div class=${styles.capPromptActions}>
          <button class="dl-btn" type="button" ?disabled=${this.#busy} @click=${() => this.#cancelCapPrompt()}>
            ${msg('Cancel', {id: 'goalBar.cancel'})}
          </button>
          <button
            class="dl-btn ${styles.capSave}"
            type="button"
            ?disabled=${this.#busy}
            @click=${() => void this.#raiseCapAndResume()}
          >
            ${msg('Raise & resume', {id: 'goalBar.raiseCapAndResume'})}
          </button>
        </div>
      </section>
    `;
  }

  protected override render(): TemplateResult | typeof nothing {
    const goal = this.#goal;
    if (!goal) return nothing;
    const canPause = goal.phase === 'active';
    const canResume = goal.phase === 'paused' || goal.phase === 'blocked';
    const finished = goal.phase === 'complete' || goal.phase === 'blocked';
    const description = goal.blockedReason
      ? `${goal.objective} — ${goal.blockedReason}`
      : goal.objective;
    return html`
      <div class=${styles.region}>
        <div class=${styles.stack}>
          <section
            class=${this.#editing ? `${styles.card} ${styles.editingCard}` : styles.card}
            data-phase=${goal.phase}
            aria-label=${msg('Goal', {id: 'goalBar.label'})}
            aria-description=${description}
          >
            ${this.#editing ? this.#editingBody() : html`
              <span class=${styles.pill} data-phase=${goal.phase} title=${description}>
                <span class=${styles.dot} aria-hidden="true"></span>${goalPhaseLabel(goal)}
              </span>
              <span class=${styles.objective} title=${goal.objective}>${goal.objective}</span>
              <span class=${styles.rounds} title=${msg('Round', {id: 'goalBar.round'})}>
                ${msg('Round', {id: 'goalBar.round'})} ${goal.roundsStarted}/${goal.maxRounds}
              </span>
              ${goal.blockedReason
                ? html`<span class=${styles.reason} title=${goal.blockedReason}>${goal.blockedReason}</span>`
                : nothing}
              <span class=${styles.actions}>
                ${canPause
                  ? this.#iconButton('pause', msg('Pause', {id: 'goalBar.pause'}), () => void this.#command('pause'), {
                    hint: msg(
                      'Pause stops the goal now. A running goal round is cancelled; your own running turn is not.',
                      {id: 'goalBar.pauseHint'},
                    ),
                    ariaDescription: msg(
                      'Pause stops the goal now. A running goal round is cancelled; your own running turn is not.',
                      {id: 'goalBar.pauseHint'},
                    ),
                  })
                  : nothing}
                ${canResume
                  ? this.#iconButton('play', msg('Resume', {id: 'goalBar.resume'}), () => void this.#beginResume())
                  : nothing}
                ${goal.phase === 'complete'
                  ? nothing
                  : this.#iconButton('edit', msg('Edit', {id: 'goalBar.edit'}), this.#beginEdit)}
                ${this.#iconButton(
                  'close',
                  msg('Clear', {id: 'goalBar.clear'}),
                  () => void this.#command('clear'),
                  {danger: true},
                )}
              </span>
              ${this.#failure
                ? html`<span class=${styles.failure} role="alert" title=${this.#failure}>${this.#failure}</span>`
                : nothing}
            `}
          </section>
          ${goal.phase === 'paused' || finished
            ? html`<span class=${styles.noticeGroup} role="status" aria-live="polite">
              ${goal.phase === 'paused'
                ? html`<span class=${styles.notice} data-phase="paused">
                  ${msg(
                    'Paused. The goal round in progress was cancelled; no next round will start. Your own running turn is not stopped.',
                    {id: 'goalBar.pausedNote'},
                  )}
                </span>`
                : html`<span class=${styles.notice} data-phase=${goal.phase}>
                  ${goal.phase === 'complete'
                    ? msg('Goal complete. No further round will start.', {id: 'goalBar.completeNotice'})
                    : goal.roundsLeft === 0
                      ? msg('Goal reached its round cap. Raise the cap to continue.', {
                        id: 'goalBar.capBlockedNotice',
                      })
                      : msg('Goal blocked. Resume to continue.', {id: 'goalBar.blockedNotice'})}
                </span>`}
            </span>`
            : nothing}
          ${this.#capPrompt && !this.#editing ? this.#capPromptBody() : nothing}
        </div>
      </div>
    `;
  }
}

customElements.define('dl-goal-bar', DlGoalBar);

declare global {
  interface HTMLElementTagNameMap {
    'dl-goal-bar': DlGoalBar;
  }
}
