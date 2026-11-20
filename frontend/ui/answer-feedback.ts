// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The thumbs under one answered message, and the reasons a thumb down is given.
 *
 * The element holds the state of one judgement: the one the server keeps for the message it names, whether
 * the reasons are open, and whether a request is out. It is an element of its own because the chat draws a
 * settled turn once and caches it, so state kept anywhere else would not be redrawn. It sends its own
 * requests and, like the rest of the app, waits for the server: a control is shown as pressed only once the
 * server has said it kept the judgement.
 *
 * The note a reader types travels with the reason they choose next, and stands on its own where they choose
 * none: a note a judgement does not already hold has a small Send of its own, on a fresh thumb down as much as
 * on a stored one. A reader who just clicks a reason sees no such button.
 *
 * An act in the reasons never takes them away. What closes them is the reader's own pause — three seconds
 * with nothing done in them — restarted by every act and held off while words are unsaved or a request is
 * out. Only an Escape, a thumb up, and pressing the lit thumb down close them at once.
 *
 * It draws no box of its own. Its thumbs and its editor are items of the wrapping flex row that holds it —
 * the action row under an answer, or the status row of a child — and the editor takes a line under that row.
 */

import {msg} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {live} from 'lit/directives/live.js';
import {
  type AnswerJudgement,
  deleteAnswerFeedback,
  FEEDBACK_NOTE_LIMIT,
  FEEDBACK_REASONS,
  type FeedbackReason,
  putAnswerFeedback,
} from '../api/feedback.ts';
import {ApiError} from '../api/wire.ts';
import {icon, isImeKey, type IconSize} from '../design-system/index.ts';
import {raise} from '../lib/dom.ts';
import {refusalText} from '../lib/errors.ts';
import {LightElement} from '../lib/lit-host.ts';
import {catalogText, type LocalizedText, localizedText} from '../lib/localized-text.ts';
import {type AppHandles, productionHandles} from '../stores/app-handles.ts';
import {judgementKey} from '../stores/kept-judgements.ts';
import styles from '../styles/answer-feedback.module.css';
import {requestToast} from './toast-request.ts';

/** Where the element sits: in the action row under an answer, or at the end of a child's status row. */
type FeedbackLayout = 'row' | 'block';

const REASON_LABELS: Record<FeedbackReason, string> = {
  unsupported: 'Unsupported claims',
  factual_error: 'Factual error',
  too_verbose: 'Too verbose',
  off_topic: 'Off topic',
  irrelevant_citations: 'Irrelevant citations',
};

const SAVE_FAILED = catalogText('Your feedback could not be saved.', 'answerFeedback.saveFailed');
const REMOVE_FAILED = catalogText('Your feedback could not be removed.', 'answerFeedback.removeFailed');

/** How long the reasons stay open after the reader's last act in them: the pause that closes them. A reader
 *  who has only opened them is still choosing, and nothing closes them until they have acted once. */
const IDLE_CLOSE_MILLISECONDS = 3000;

/** What the reasons do once the server has answered: close at once, or stay open for the reader's pause. */
type EditorClose = 'now' | 'when-idle';

let editors = 0;

/** Take a judgement back; the server answers 204 whether or not there was one. */
async function takeBack(runId: string, agent: string, sequence: number): Promise<null> {
  await deleteAnswerFeedback(runId, agent, sequence);
  return null;
}

export class DlAnswerFeedback extends LightElement {
  static properties = {
    handles: {attribute: false},
    runId: {attribute: false},
    agent: {attribute: false},
    sequence: {attribute: false},
    judgement: {attribute: false},
    enabled: {attribute: false},
    layout: {type: String},
    kept: {state: true},
    editing: {state: true},
    pending: {state: true},
  };

  declare handles: AppHandles;
  declare runId: string;
  /** Who wrote the message: `main` for the Run's own agent, else a Child Session's id. */
  declare agent: string;
  /** The message's own place in that agent's Session, which the server states with the turn or the child. */
  declare sequence: number;
  /** The judgement as the page read it with the turn or the child; what the page has since seen the server
   *  keep takes precedence (see `kept-judgements.ts`). */
  declare judgement: AnswerJudgement | null;
  /** Whether the reader is offered the thumbs. A refused rating turns it off. */
  declare enabled: boolean;
  declare layout: FeedbackLayout;
  /** The judgement now showing: the one the server last said it kept for the message. */
  declare kept: AnswerJudgement | null;
  /** Whether the reasons are open. */
  declare editing: boolean;
  /** Whether a request is out. */
  declare pending: boolean;

  readonly #editorId = `answer-feedback-editor-${++editors}`;
  /** The note as typed; it travels with the next rating, unless Send is there to keep it by itself. */
  #note = '';
  /** Whether the Enter being typed is a line break the reader asked for (Shift+Enter) and not a send. */
  #allowNextLineBreak = false;
  /** Counts the requests this element has made, so an answer that is no longer about what shows is let go. */
  #flight = 0;
  /** The timer that closes the reasons once the reader's pause is up; null while none is armed. */
  #idleClose: ReturnType<typeof setTimeout> | null = null;

  constructor() {
    super();
    this.handles = productionHandles();
    this.runId = '';
    this.agent = 'main';
    this.sequence = 0;
    this.judgement = null;
    this.enabled = false;
    this.layout = 'row';
    this.kept = null;
    this.editing = false;
    this.pending = false;
  }

  override connectedCallback(): void {
    super.connectedCallback();
    this.addEventListener('keydown', this.#keydown, {signal: this.lifetime});
    // Every act in the reasons restarts the pause that closes them. The click that opens them is not one: the
    // element is not editing yet when it lands, so a reader who is still choosing is never interrupted.
    for (const type of ['pointerdown', 'keydown', 'input', 'focusin'] as const) {
      this.addEventListener(type, this.#act, {signal: this.lifetime});
    }
  }

  override disconnectedCallback(): void {
    this.#stopIdleClose();
    super.disconnectedCallback();
  }

  protected override willUpdate(changed: PropertyValues<this>): void {
    // Another message: nothing the reader did to the last one carries over.
    if (changed.has('runId') || changed.has('agent') || changed.has('sequence')) {
      this.#flight += 1;
      this.#stopIdleClose();
      this.editing = false;
      this.pending = false;
      this.#note = '';
    }
    if (changed.has('runId') || changed.has('agent') || changed.has('sequence') || changed.has('judgement') || changed.has('handles')) {
      this.kept = this.handles.judgements.current(this.#key(), this.judgement);
    }
  }

  #key(): string {
    return judgementKey(this.runId, this.agent, this.sequence);
  }

  #thumb(which: 'up' | 'down'): HTMLElement | null {
    return this.querySelector<HTMLElement>(`[data-thumb="${which}"]`);
  }

  #noteField(): HTMLTextAreaElement | null {
    return this.querySelector<HTMLTextAreaElement>('textarea');
  }

  // ── Gestures ──

  /** A thumb up rates the message good, replacing a thumb down, and closes the reasons a thumb down had open;
   *  on a message already rated good it takes the rating back. */
  #thumbUp = (): void => {
    if (this.pending) return;
    const {runId, agent, sequence} = this;
    if (this.kept?.value === 'up') {
      void this.#change(() => takeBack(runId, agent, sequence), {close: 'now', receipt: true}, REMOVE_FAILED);
      return;
    }
    void this.#change(
      () => putAnswerFeedback(runId, agent, sequence, {value: 'up', reasons: [], note: null}),
      {close: 'now', receipt: true},
      SAVE_FAILED,
    );
  };

  /** A thumb down opens the reasons, and stores nothing until one is chosen. Pressed again while they are
   *  open, it takes a stored thumb down back, or just closes them where nothing is stored. */
  #thumbDown = (): void => {
    if (this.pending) return;
    if (!this.editing) {
      this.#note = this.kept?.value === 'down' ? this.kept.note ?? '' : '';
      this.editing = true;
      void this.updateComplete.then(() => {
        this.querySelector<HTMLElement>(`#${this.#editorId}`)?.scrollIntoView({block: 'nearest'});
      });
      return;
    }
    if (this.kept?.value === 'down') {
      const {runId, agent, sequence} = this;
      void this.#change(() => takeBack(runId, agent, sequence), {close: 'now', receipt: true}, REMOVE_FAILED);
    } else {
      this.#closeEditor();
    }
  };

  /** A reason not yet chosen is added and the whole choice stored at once; a reason chosen is taken away and
   *  the rest stored without a word, since the pill the reader pressed has already let go, or, when none is
   *  left, the rating goes, because a thumb down that says nothing — no reason and no words — is no judgement
   *  at all. Either way the reasons stay for the reader's own pause, so a choice never takes the form away
   *  from a reader still working in it. */
  #reason(reason: FeedbackReason): void {
    if (this.pending) return;
    const {runId, agent, sequence} = this;
    const chosen = this.kept?.value === 'down' ? this.kept.reasons : [];
    const note = this.#note.trim() || null;
    if (!chosen.includes(reason)) {
      const reasons = FEEDBACK_REASONS.filter((each) => each === reason || chosen.includes(each));
      void this.#change(
        () => putAnswerFeedback(runId, agent, sequence, {value: 'down', reasons, note}),
        {close: 'when-idle', receipt: true},
        SAVE_FAILED,
      );
      return;
    }
    const reasons = chosen.filter((each) => each !== reason);
    if (reasons.length > 0 || note !== null) {
      // The last reason can go and the rating stay, where the reader's own words still say why.
      void this.#change(
        () => putAnswerFeedback(runId, agent, sequence, {value: 'down', reasons, note}),
        {close: 'when-idle', receipt: false},
        SAVE_FAILED,
      );
    } else {
      void this.#change(() => takeBack(runId, agent, sequence), {close: 'when-idle', receipt: true}, REMOVE_FAILED);
    }
  }

  #keydown = (event: KeyboardEvent): void => {
    if (event.key !== 'Escape' || !this.editing) return;
    // An Escape that cancels an IME composition belongs to the composition.
    if (isImeKey(event)) return;
    // The Inspector closes on an Escape nobody handled; this one closed the reasons.
    event.preventDefault();
    this.#closeEditor();
  };

  #typed = (event: Event): void => {
    this.#note = (event.currentTarget as HTMLTextAreaElement).value;
    // Whether Send shows depends on what is typed.
    this.requestUpdate();
  };

  /** Whether the note as typed is the one the message's judgement holds: nothing Send would keep, and nothing
   *  the reasons' close would throw away. */
  #noteSaved(): boolean {
    const held = this.kept?.value === 'down' ? this.kept.note : null;
    return (this.#note.trim() || null) === held;
  }

  /** Whether the note has something of its own to send: it is not the note the message's thumb down already
   *  holds (none holds the empty note, so clearing one counts), and the judgement left with it still says
   *  why. A thumb down with neither a reason nor words is no judgement, so taking away the last of them is
   *  the lit thumb's gesture and never Send's. */
  #noteToSend(): boolean {
    if (this.#noteSaved()) return false;
    const held = this.kept?.value === 'down' ? this.kept : null;
    return this.#note.trim() !== '' || (held?.reasons.length ?? 0) > 0;
  }

  /** Keep the note as this message's judgement: a thumb down with the reasons already stored, if any, and the
   *  note as typed, which is none once it is empty. Where no thumb down is stored yet this is the whole
   *  judgement — a thumb down that says why in the reader's own words alone. */
  #sendNote = (): void => {
    if (this.pending || !this.#noteToSend()) return;
    const {runId, agent, sequence} = this;
    const reasons = this.kept?.value === 'down' ? this.kept.reasons : [];
    const note = this.#note.trim() || null;
    void this.#change(
      () => putAnswerFeedback(runId, agent, sequence, {value: 'down', reasons, note}),
      {close: 'when-idle', receipt: true},
      SAVE_FAILED,
    );
  };

  // Enter sends the note where there is something to send, and Shift+Enter breaks the line, as in the composers;
  // Enter that commits an IME composition is the composition's, never a send. Where Send is not shown, Enter
  // breaks the line.
  #noteKeydown = (event: KeyboardEvent): void => {
    if (event.key === 'Enter') this.#allowNextLineBreak = event.shiftKey;
  };

  #noteBeforeInput = (event: InputEvent): void => {
    if (event.inputType !== 'insertLineBreak') return;
    if (event.isComposing || this.#allowNextLineBreak || !this.#noteToSend()) {
      this.#allowNextLineBreak = false;
      return;
    }
    event.preventDefault();
    this.#allowNextLineBreak = false;
    this.#sendNote();
  };

  #noteKeyup = (event: KeyboardEvent): void => {
    if (event.key === 'Enter') this.#allowNextLineBreak = false;
  };

  // ── The reader's pause ──

  /** What an act on one of the controls does: restart the pause that closes the reasons. What counts is a
   *  control the reader can act on — the thumbs, the reason pills, the note field, Send — and not the box the
   *  element happens to occupy: a press on the space around them is not the reader working in the form. */
  #act = (event: Event): void => {
    const target = event.target;
    const control = target instanceof Element ? target.closest('button, textarea') : null;
    if (!control || !this.contains(control)) return;
    this.#armIdleClose();
  };

  /** Arm the one timer that closes the reasons. Every act in them restarts it, so they go only after three
   *  seconds with nothing done in them — and nothing is armed before the reader has acted once, because a
   *  reader who has only opened them is still choosing. */
  #armIdleClose(): void {
    this.#stopIdleClose();
    if (!this.editing) return;
    this.#idleClose = setTimeout(() => {
      this.#idleClose = null;
      this.#pauseEnded();
    }, IDLE_CLOSE_MILLISECONDS);
  }

  #stopIdleClose(): void {
    if (this.#idleClose !== null) clearTimeout(this.#idleClose);
    this.#idleClose = null;
  }

  /** The reader's pause is up. Words the judgement does not hold are theirs to keep, and a request that is
   *  out is still to be answered: either one holds the reasons open for another pause. */
  #pauseEnded(): void {
    if (this.pending || !this.#noteSaved()) {
      this.#armIdleClose();
      return;
    }
    this.#closeEditor();
  }

  #closeEditor(): void {
    this.#stopIdleClose();
    // A reader inside the reasons would lose their place with them: it goes to the thumb that opened them.
    const inside = this.querySelector(`#${this.#editorId}`)?.contains(document.activeElement) ?? false;
    this.editing = false;
    if (inside) void this.updateComplete.then(() => { this.#thumb('down')?.focus(); });
  }

  // ── Requests ──

  /** Send one change to this message's judgement and, once the server has answered, show what it kept — the
   *  judgement, or none once it is taken back — and, when asked, say so in a receipt. Where the reader is
   *  still at work in the reasons the answer arms their pause instead of taking the form away, and the thumbs
   *  hold still while it is out. */
  async #change(
    send: () => Promise<AnswerJudgement | null>,
    after: {close: EditorClose; receipt: boolean},
    failure: LocalizedText,
  ): Promise<void> {
    const key = this.#key();
    const flight = ++this.#flight;
    this.pending = true;
    try {
      const kept = await send();
      // The page keeps what the server said even when this element no longer shows the message.
      this.handles.judgements.keep(key, kept);
      if (this.#flight !== flight) return;
      this.kept = kept;
      if (after.close === 'now') {
        this.#closeEditor();
      } else {
        this.#armIdleClose();
        // Send goes away the moment the note it kept is the note the judgement holds, so a reader who was on
        // it lands on the field it belonged to rather than on nothing. Every other control stays where it is.
        const atWork = this.contains(document.activeElement);
        void this.updateComplete.then(() => {
          if (atWork && !this.contains(document.activeElement)) this.#noteField()?.focus();
        });
      }
      if (after.receipt) {
        requestToast(this, {
          message: kept === null
            ? msg('Feedback removed.', {id: 'answerFeedback.removed'})
            : msg('Thanks for the feedback.', {id: 'answerFeedback.thanks'}),
        });
      }
    } catch (error) {
      if (this.#flight === flight) this.#refused(error, failure);
    } finally {
      if (this.#flight === flight) this.pending = false;
    }
  }

  #refused(error: unknown, failure: LocalizedText): void {
    // The server's own refusal of a reader it does not offer feedback — 403 in its envelope as an authorization
    // failure, with no kind of its own — says feedback is off for this reader, which the page learned at load and
    // which has changed since: say so once and offer it no more. A bare 403 came from something in front of the
    // server, and a kind names another matter (the origin guard), so neither takes the thumbs away.
    if (
      error instanceof ApiError
      && error.status === 403
      && error.errorType === 'auth'
      && error.errorKind === null
    ) {
      // The controls go, and a reader on one of them keeps their place.
      if (this.contains(document.activeElement)) this.#focusNeighbour();
      this.enabled = false;
      raise(this, 'dl-answer-feedback-off');
      requestToast(this, {
        tone: 'error',
        message: msg('Feedback is turned off for you.', {id: 'answerFeedback.turnedOff'}),
      });
      return;
    }
    requestToast(this, {tone: 'error', message: localizedText(refusalText(error, failure))});
  }

  /** Put focus on what stays when the controls it is on go away: in the action row the button before the thumbs
   *  (Fork), else the row's first; in a child's page its title, where a reader who has opened the child belongs. */
  #focusNeighbour(): void {
    if (this.layout === 'block') {
      this.closest('dl-agent-session')?.focusTitle();
      return;
    }
    const buttons = [...this.parentElement?.querySelectorAll<HTMLElement>(':scope > button') ?? []];
    const before = buttons.filter((button) => button.compareDocumentPosition(this) & Node.DOCUMENT_POSITION_FOLLOWING);
    (before.at(-1) ?? buttons[0])?.focus();
  }

  // ── Drawing ──

  /** The thumb's classes: the shared icon button, drawn at the row's own scale there (see
   *  styles/answer-feedback.module.css and `.dl-icon-button--dense`), and at its neighbours' size in a
   *  child's status row, where the controls it sits among are `--size-button` tall. */
  get #thumbClass(): string {
    return this.layout === 'row'
      ? `dl-icon-button dl-icon-button--dense ${styles.thumb}`
      : `dl-icon-button ${styles.thumb}`;
  }

  /** The mark's size follows the same rule: the row's other marks are `xs`, a child's row is `sm`. */
  get #markSize(): IconSize {
    return this.layout === 'row' ? 'xs' : 'sm';
  }

  protected override render(): TemplateResult | typeof nothing {
    if (!this.enabled) return nothing;
    const up = this.kept?.value === 'up';
    const down = this.kept?.value === 'down';
    const busy = this.pending ? 'true' : 'false';
    const unavailable = this.pending ? 'true' : nothing;
    return html`
      <span class=${styles.thumbs} data-layout=${this.layout} aria-busy=${busy}
            ?data-action-part=${this.layout === 'row'}>
        <button type="button" class=${this.#thumbClass} data-thumb="up"
                aria-label=${msg('Good answer', {id: 'answerFeedback.good'})}
                aria-pressed=${up ? 'true' : 'false'} aria-disabled=${unavailable}
                @click=${this.#thumbUp}>${icon(up ? 'thumbs-up-filled' : 'thumbs-up', {size: this.#markSize})}</button>
        <button type="button" class=${this.#thumbClass} data-thumb="down"
                aria-label=${msg('Bad answer', {id: 'answerFeedback.bad'})}
                aria-pressed=${down ? 'true' : 'false'} aria-disabled=${unavailable}
                aria-expanded=${this.editing ? 'true' : 'false'}
                aria-controls=${this.editing ? this.#editorId : nothing}
                @click=${this.#thumbDown}>${icon(down ? 'thumbs-down-filled' : 'thumbs-down', {size: this.#markSize})}</button>
      </span>
      ${this.editing ? this.#editor(busy, unavailable) : nothing}
    `;
  }

  #editor(busy: string, unavailable: string | typeof nothing): TemplateResult {
    const chosen = this.kept?.value === 'down' ? this.kept.reasons : [];
    const label = msg('Add a note (optional)', {id: 'answerFeedback.noteLabel'});
    return html`
      <div class=${styles.editor} id=${this.#editorId} data-layout=${this.layout} role="group"
           aria-label=${msg('What was wrong with this answer?', {id: 'answerFeedback.reasonsLabel'})}
           aria-busy=${busy}>
        <div class=${styles.chips}>
          ${FEEDBACK_REASONS.map((reason) => html`
            <button type="button" class=${styles.chip} data-reason=${reason}
                    aria-pressed=${chosen.includes(reason) ? 'true' : 'false'} aria-disabled=${unavailable}
                    @click=${() => { this.#reason(reason); }}>${
                      // A chosen reason is told from the others by its check as well as its colour.
                      chosen.includes(reason) ? icon('check', {size: 'xs'}) : nothing
                    }${msg(REASON_LABELS[reason], {id: `answerFeedback.reason.${reason}`})}</button>
          `)}
        </div>
        <div class=${styles.noteField}>
          <textarea class=${styles.note} rows="2" maxlength=${FEEDBACK_NOTE_LIMIT}
                    aria-label=${label} placeholder=${label}
                    .value=${live(this.#note)} @input=${this.#typed}
                    @keydown=${this.#noteKeydown} @beforeinput=${this.#noteBeforeInput}
                    @keyup=${this.#noteKeyup}></textarea>
          ${this.#noteToSend() ? html`
            <button type="button" class=${styles.send} data-send
                    aria-label=${msg('Send note', {id: 'answerFeedback.sendNote'})}
                    aria-disabled=${unavailable} @click=${this.#sendNote}>${
                      // A request in flight is told in a shape as well as in dimming.
                      icon(this.pending ? 'status-dot' : 'send', {size: 'sm'})
                    }</button>
          ` : nothing}
        </div>
      </div>
    `;
  }
}

customElements.define('dl-answer-feedback', DlAnswerFeedback);

declare global {
  interface HTMLElementTagNameMap {
    'dl-answer-feedback': DlAnswerFeedback;
  }

  interface HTMLElementEventMap {
    /** A rating was refused because feedback is off for this reader: the Shell stops offering it. */
    'dl-answer-feedback-off': CustomEvent<void>;
  }
}
