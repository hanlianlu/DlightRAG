// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** What the Run left in Profile Memory, said under its answer, with the way to take each back.
 *
 * Two writers leave memory in a Run's name: the Agent, with its memory tools while it answers, and the
 * Session Memory Pass, which reads the conversation a few seconds after the answer. Neither asked the reader,
 * so the answer says what they kept: one more part at the end of the action row, a bookmark and "Remembered
 * 2". It opens a small popover that lists those records, each with its kind and an Undo, and a way into
 * Settings to manage the rest. A part that comes late is not a toast and floats beside nothing: it joins the
 * row, fades in without moving the words around it, and is announced once, politely, to a reader who was
 * here for the answer and has not asked to hear it.
 *
 * What the Run left is read by the page's store (`stores/memory-notices.ts`), which also knows when to look
 * again; this element draws that, and takes a change back on the reader's word. It draws nothing while there
 * is nothing: for a Run that left no memory, for an owner who turned Profile Memory off (the server lists
 * none then), and for a reader the read failed for.
 *
 * It draws no box of its own: its part is an item of the wrapping flex row that holds it.
 */

import {msg, str} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import {forgetMemory, undoMemoryChange} from '../api/memory.ts';
import {icon} from '../design-system/index.ts';
import {raise} from '../lib/dom.ts';
import {LightElement, StoreController} from '../lib/lit-host.ts';
import {TriggerPopover} from '../lib/popover.ts';
import {type AppHandles, productionHandles} from '../stores/app-handles.ts';
import type {MemoryNoticeChange} from '../stores/memory-notices.ts';
import styles from '../styles/memory-notice.module.css';
import {memoryKindLabel, memorySummary} from './memory-words.ts';
import type {ToastRequestDetail} from './toast.ts';
import {requestToast} from './toast-request.ts';

type ToastAction = NonNullable<ToastRequestDetail['action']>;

/** How long the announcement stays in the live region: long enough to be read out, short enough that it is
 *  not left beside the button for a reader who browses the page afterwards. */
const ANNOUNCEMENT_MILLISECONDS = 6000;

let notices = 0;

export class DlMemoryNotice extends LightElement {
  static properties = {
    handles: {attribute: false},
    runId: {attribute: false},
    placement: {state: true},
    fresh: {state: true},
    busy: {state: true},
    announcement: {state: true},
  };

  declare handles: AppHandles;
  /** The Run whose answer this sits under. */
  declare runId: string;
  /** Whether the popover opens above the part or below it, decided by the room there is when it opens. */
  declare placement: 'above' | 'below';
  /** The part came while the reader was here, so it fades in; one drawn with the page does not. */
  declare fresh: boolean;
  /** The changes whose undo is out. */
  declare busy: ReadonlySet<string>;
  declare announcement: string;

  readonly #id = `memory-notice-${++notices}`;
  readonly #popup = new TriggerPopover(this, {
    trigger: () => this.querySelector<HTMLElement>(`#${this.#id}-trigger`),
    enter: () => { void this.#opened(); },
  });
  /** How many changes were drawn the last time, or null before the first draw. */
  #drawn: number | null = null;
  #announcementTimer: ReturnType<typeof setTimeout> | null = null;

  constructor() {
    super();
    this.handles = productionHandles();
    this.runId = '';
    this.placement = 'below';
    this.fresh = false;
    this.busy = new Set();
    this.announcement = '';
    /** Store reads: memoryNotices.changes. */
    new StoreController(this, () => [this.handles.memoryNotices]);
  }

  override connectedCallback(): void {
    super.connectedCallback();
    // A window that is resized under an open popover moves the room it was placed in.
    window.addEventListener('resize', this.#resized, {signal: this.lifetime});
  }

  override disconnectedCallback(): void {
    if (this.#announcementTimer !== null) clearTimeout(this.#announcementTimer);
    this.#announcementTimer = null;
    super.disconnectedCallback();
  }

  protected override willUpdate(changed: PropertyValues<this>): void {
    if (changed.has('runId') || changed.has('handles')) {
      // Another answer: nothing that came to the last one is this one's.
      this.#drawn = null;
      this.fresh = false;
      this.busy = new Set();
      if (this.#popup.open) this.#popup.close();
    }
    // A turn answered before the page opened has not been read, and one the owner's own changes made stale is
    // read again; one this page watched is read by its watch. The store reads each at most once at a time.
    if (this.runId) this.handles.memoryNotices.ensure(this.runId);
    const total = this.#changes().length;
    // It arrived while the reader was here: from nothing to something, after the element was first drawn.
    if (this.#drawn === 0 && total > 0) {
      this.fresh = true;
      if (this.handles.memoryNotices.announces(this.runId)) this.#announce();
    }
    this.#drawn = total;
  }

  #changes(): readonly MemoryNoticeChange[] {
    return this.runId ? this.handles.memoryNotices.changes(this.runId) : [];
  }

  /** The count the part says: the changes still standing, not the ones the reader took back. */
  #standing(): number {
    return this.#changes().filter((change) => !change.undone).length;
  }

  #label(): string {
    const standing = this.#standing();
    return standing > 0
      ? msg(str`Remembered ${standing}`, {id: 'memoryNotice.remembered'})
      : msg('Undone', {id: 'memoryNotice.undone'});
  }

  #announce(): void {
    this.handles.memoryNotices.announced(this.runId);
    this.announcement = this.#label();
    if (this.#announcementTimer !== null) clearTimeout(this.#announcementTimer);
    this.#announcementTimer = setTimeout(() => {
      this.#announcementTimer = null;
      this.announcement = '';
    }, ANNOUNCEMENT_MILLISECONDS);
  }

  // ── The popover ──

  #trigger(): HTMLElement | null {
    return this.querySelector<HTMLElement>(`#${this.#id}-trigger`);
  }

  #popover(): HTMLElement | null {
    return this.querySelector<HTMLElement>(`#${this.#id}-popover`);
  }

  /** The popover is open: put it where there is room, then give the reader its first control. */
  async #opened(): Promise<void> {
    this.#place();
    await this.updateComplete;
    // It may have been put away again while it was placing itself.
    if (this.#popup.open) this.#focusFirst();
  }

  /** Where the popover goes, in the room the chat gives it. The chat scrolls and shares the window with the
   *  conversations and the panels beside it, so its own box is the room, not the window. Along the line it starts
   *  where the part starts and is moved no further than keeps it, whole, inside that box; across, it opens below the
   *  part when the room below holds it, so the answer's own words stay in view, else above when the room above does,
   *  else wherever there is more. A sheet from the bottom edge lays itself out and is left alone. */
  #place(): void {
    const trigger = this.#trigger();
    const popover = this.#popover();
    if (!trigger || !popover || getComputedStyle(popover).position === 'fixed') return;
    const box = trigger.closest('#chat-area')?.getBoundingClientRect();
    const room = box ?? {left: 0, right: window.innerWidth, top: 0, bottom: window.innerHeight};
    const edge = Number.parseFloat(getComputedStyle(document.documentElement).fontSize) / 2;
    // Its width first, since it decides its height; then where it would stand if nothing moved it.
    popover.style.setProperty('--notice-room', `${Math.max(0, room.right - room.left - 2 * edge)}px`);
    popover.style.setProperty('--notice-shift', '0px');
    const natural = popover.getBoundingClientRect();
    const part = trigger.getBoundingClientRect();
    const first = room.left + edge;
    const left = Math.min(Math.max(part.left, first), Math.max(first, room.right - edge - natural.width));
    popover.style.setProperty('--notice-shift', `${left - natural.left}px`);
    const above = part.top - room.top;
    const below = room.bottom - part.bottom;
    const height = natural.height;
    this.placement = below >= height || (above < height && below >= above) ? 'below' : 'above';
  }

  #resized = (): void => {
    if (this.#popup.open) this.#place();
  };

  #focusFirst(): void {
    const popover = this.#popover();
    popover?.querySelector<HTMLElement>('[data-undo]:not([aria-disabled="true"]), [data-manage]')?.focus();
  }

  /** Focus left the part: the popover goes with it, unless focus went to the part's own popover. */
  #focusLeft = (event: FocusEvent): void => {
    const to = event.relatedTarget;
    if (this.#popup.open && to instanceof Node && !this.contains(to)) this.#popup.close();
  };

  #manage = (): void => {
    // The popover goes first and the part holds focus, so Settings gives focus back to something that is there.
    this.#popup.close();
    this.#trigger()?.focus();
    raise(this, 'dl-settings-request', {page: 'memory'});
  };

  // ── Taking a change back ──

  /** Take one change back: an addition by forgetting the record, a correction or a merge by undoing the change,
   *  each the way Settings does. What happened is told in a receipt, or in an error where it was refused. */
  async #undo(change: MemoryNoticeChange): Promise<void> {
    if (change.undone || this.busy.has(change.changeId)) return;
    const runId = this.runId;
    // The button pressed is replaced by words once the change is taken back; focus must not go with it.
    const atWork = this.contains(document.activeElement);
    this.busy = new Set([...this.busy, change.changeId]);
    try {
      if (change.action === 'add') {
        const receipt = await forgetMemory(change.memoryId);
        this.handles.memoryNotices.markUndone(runId, change.changeId);
        const undoable = receipt.outcome === 'changed' && receipt.changeId !== '';
        this.#tell(memorySummary({
          live: true, operation: 'forget', outcome: receipt.outcome, changeId: receipt.changeId,
          intentId: null, body: receipt.body || change.body,
        }), undoable ? this.#takeBackForgetting(runId, change, receipt.changeId) : undefined);
      } else {
        const receipt = await undoMemoryChange(change.changeId);
        if (receipt.outcome !== 'changed') {
          this.#refused(memorySummary({
            live: true, operation: 'undo', outcome: 'conflict', changeId: null, intentId: null, body: '',
          }));
          return;
        }
        this.handles.memoryNotices.markUndone(runId, change.changeId);
        // A correction brings back the one record it replaced, which is worth naming; a merge brings back
        // several, and naming one would say too little.
        this.#tell(memorySummary({
          live: true, operation: 'undo', outcome: 'changed', changeId: receipt.changeId, intentId: null,
          body: change.action === 'update' ? receipt.body : '',
        }));
      }
      void this.#moveOn(atWork);
    } catch {
      this.#refused(msg('Could not undo this memory.', {id: 'memoryNotice.undoFailed'}));
    } finally {
      const rest = new Set(this.busy);
      rest.delete(change.changeId);
      this.busy = rest;
    }
  }

  /** The Undo a forgotten record's receipt offers: it brings the record back, and the record is then no
   *  change of this Run's, so the notice lets it go rather than say it was taken back. */
  #takeBackForgetting(
    runId: string,
    change: MemoryNoticeChange,
    forgetChangeId: string,
  ): ToastAction {
    return {
      actionLabel: msg('Undo', {id: 'settings.memory.undo'}),
      onAction: async () => {
        const restored = await undoMemoryChange(forgetChangeId);
        if (restored.outcome !== 'changed') throw new Error('Memory undo conflicted');
        this.handles.memoryNotices.release(runId, change.changeId);
        return msg('Profile Memory change undone.', {id: 'settings.memory.changeUndone'});
      },
    };
  }

  #tell(message: string, action?: ToastAction): void {
    if (this.isConnected) requestToast(this, {message, action});
  }

  #refused(message: string): void {
    if (this.isConnected) requestToast(this, {message, tone: 'error'});
  }

  /** The button the reader pressed is gone, replaced by the words "Undone": focus goes to the next thing they can
   *  do here, or to the way into Settings. */
  async #moveOn(atWork: boolean): Promise<void> {
    await this.updateComplete;
    if (atWork && this.#popup.open && !this.contains(document.activeElement)) this.#focusFirst();
  }

  // ── Drawing ──

  protected override render(): TemplateResult {
    const changes = this.#changes();
    return html`
      ${this.announcement !== '' || this.handles.memoryNotices.announces(this.runId) ? html`
        <span class="dl-sr-only" role="status" aria-live="polite">${this.announcement}</span>
      ` : nothing}
      ${changes.length === 0 ? nothing : this.#part(changes)}
    `;
  }

  #part(changes: readonly MemoryNoticeChange[]): TemplateResult {
    const open = this.#popup.open;
    return html`
      <span class=${styles.notice} data-action-part ?data-fresh=${this.fresh} @focusout=${this.#focusLeft}>
        <button type="button" class=${styles.trigger} id="${this.#id}-trigger"
                aria-haspopup="dialog" aria-expanded=${open ? 'true' : 'false'}
                aria-controls="${this.#id}-popover" @click=${this.#popup.toggle}>
          <span class=${styles.mark} aria-hidden="true">${icon('memory-notice', {size: 'xs'})}</span>${this.#label()}
        </button>
        <div class="dl-popover dl-anchored ${styles.popover}" id="${this.#id}-popover"
             role="dialog" aria-labelledby="${this.#id}-title" data-placement=${this.placement} ?hidden=${!open}>
          <p class=${styles.title} id="${this.#id}-title">${
            msg('Remembered from this conversation', {id: 'memoryNotice.title'})}</p>
          <ul class=${styles.list}>
            ${repeat(changes, (change) => change.changeId, (change) => this.#row(change))}
          </ul>
          <button type="button" class=${styles.manage} data-manage @click=${this.#manage}>${
            msg('Manage memory in Settings', {id: 'memoryNotice.manage'})}</button>
        </div>
      </span>
    `;
  }

  #row(change: MemoryNoticeChange): TemplateResult {
    const waiting = this.busy.has(change.changeId);
    return html`
      <li class=${styles.row} data-change=${change.changeId}>
        <span class=${styles.kind}>${memoryKindLabel(change.kind)}</span>
        <p class=${styles.text}>${change.body}</p>
        ${change.undone ? html`
          <span class=${styles.done}>${msg('Undone', {id: 'memoryNotice.undone'})}</span>
        ` : html`
          <button type="button" class=${styles.undo} data-undo
                  aria-label=${msg(str`Undo: ${change.body}`, {id: 'memoryNotice.undoAria'})}
                  aria-disabled=${waiting ? 'true' : nothing}
                  @click=${() => { void this.#undo(change); }}>${msg('Undo', {id: 'memoryNotice.undo'})}</button>
        `}
      </li>
    `;
  }
}

customElements.define('dl-memory-notice', DlMemoryNotice);

declare global {
  interface HTMLElementTagNameMap {
    'dl-memory-notice': DlMemoryNotice;
  }
}
