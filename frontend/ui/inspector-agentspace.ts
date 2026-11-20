// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The Agentspace Inspector: the files this conversation's Agent works in, as the Runs that settled
 * them left them — one restorable copy per settled Run, what each copy changed, and one file's
 * versions.
 *
 * Every read comes from the durable registers (`web_conversation_*` and
 * `dlightrag_session_workspace_*`): the panel never walks a filesystem and never shows bytes, only
 * the manifest the delivery path itself trusts. The reader opens one copy in place, opens one file's
 * versions as a second grain under a back row, and resets the agentspace when the conversation is at
 * rest — reset is refused while a Run holds the Session.
 *
 * The durable registers and the wire keep their `workspace` names ([ADR 0075]); the reader's words
 * are Agentspace and Run.
 */

import {msg, str} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import {
  type ConversationAgentspace,
  readConversationAgentspace,
  readAgentspaceFileHistory,
  readAgentspaceSnapshot,
  resetConversationAgentspace,
  type AgentspaceChange,
  type AgentspaceChangeCounts,
  type AgentspaceFile,
  type AgentspaceFileVersion,
  type AgentspaceSnapshot,
} from '../api/conversations.ts';
import {ApiError} from '../api/wire.ts';
import {icon} from '../design-system/index.ts';
import {getLocale} from '../i18n/locale.ts';
import {capitalized, recentDay, shortDate, timeOfDay} from '../lib/date-format.ts';
import {formatFileSize} from '../lib/file-size.ts';
import {LightElement, StoreController} from '../lib/lit-host.ts';
import {productionHandles, type AppHandles} from '../stores/app-handles.ts';
import styles from '../styles/inspector-agentspace.module.css';
import {requestToast} from './toast-request.ts';

const _NO_CONVERSATION = 'Open a conversation to see its Agentspace.';
const _LOADING = 'Reading the Agentspace…';
const _FAILED = 'The Agentspace could not be read.';
const _RETRY = 'Try again';
const _LEDE = 'The files the agent works in during this conversation. Each finished Run keeps a copy; open one to see what it changed.';
const _EMPTY = 'Nothing is saved yet. The files of the first answer to finish appear here.';
const _BRANCHES = 'Branches';
const _READING_COPY = 'Reading this saved copy…';
const _COPY_FAILED = 'This saved copy could not be read.';
const _NO_FILES = 'This copy carried no files.';
const _CHANGED_HEADING = 'Changed in this Run';
const _NO_CHANGED = 'No file changed in this Run.';
const _HISTORY_LEDE = 'Every saved copy that carried this file, oldest first.';
const _BACK = 'Back to saved copies';
const _NO_HISTORY = 'No saved copy carried this file.';
const _READING_HISTORY = 'Reading this file’s versions…';
const _HISTORY_FAILED = 'This file’s versions could not be read.';
const _RESET = 'Reset Agentspace…';
const _RESET_NOTE = 'This deletes every saved copy for this conversation. The conversation and its answers stay.';
const _RESET_CONFIRM = 'Reset Agentspace';
const _CANCEL = 'Cancel';
const _RESET_FAILED = 'The Agentspace could not be reset.';
const _RESET_REFUSED = 'This conversation is working; reset it after the current answer finishes.';

/** The first eight characters of a Run id, which is what a reader can match against the traces. */
function runShort(runId: string): string {
  return runId.slice(0, 8);
}

/** The day a copy settled: the reader's own word for a recent day, and its calendar date after. */
function dayLabel(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const now = new Date();
  const recent = recentDay(date, now, getLocale());
  return recent === null ? shortDate(date, now, getLocale()) : capitalized(recent, getLocale());
}

/** The event a copy raises to open the Run it saved in the Inspector's traces dock. */
export interface AgentspaceRunOpenDetail {
  runId: string;
}

function byPath(left: AgentspaceFile, right: AgentspaceFile): number {
  return left.relativePath < right.relativePath ? -1 : left.relativePath > right.relativePath ? 1 : 0;
}

export class DlInspectorAgentspace extends LightElement {
  static properties = {
    handles: {attribute: false},
    active: {type: Boolean},
  };

  declare handles: AppHandles;
  declare active: boolean;
  #space: ConversationAgentspace | null = null;
  /** The Run whose copy the reader opened, or null while every line is closed. */
  #openRunId: string | null = null;
  #openFiles: readonly AgentspaceFile[] = [];
  #sliceLoading = false;
  #sliceFailed = false;
  /** The file whose versions replace the timeline, or null while the timeline is showing. */
  #filePath: string | null = null;
  #versions: readonly AgentspaceFileVersion[] = [];
  #historyLoading = false;
  #historyFailed = false;
  /** The conversation the shown space belongs to, so a switch reloads instead of re-drawing it. */
  #loadedConversationId: string | null = null;
  #busy = false;
  #failure: string | null = null;
  #confirmReset = false;

  constructor() {
    super();
    this.handles = productionHandles();
    this.active = false;
    new StoreController(this, () => [this.handles.conversations]);
  }

  override connectedCallback(): void {
    super.connectedCallback();
    void this.#load();
  }

  protected override updated(changed: PropertyValues<this>): void {
    super.updated(changed);
    if (!this.active) return;
    // The panel stays open across a conversation switch, so it follows the store: a different
    // conversation is a different space, and drawing the old one would name another Session's Runs.
    const conversationId = this.handles.conversations.activeConversationId;
    if (changed.has('active') || conversationId !== this.#loadedConversationId) void this.#load();
  }

  async #load(): Promise<void> {
    const conversationId = this.handles.conversations.activeConversationId;
    this.#loadedConversationId = conversationId;
    this.#confirmReset = false;
    if (conversationId === null) {
      this.#space = null;
      this.#closeSlice();
      this.requestUpdate();
      return;
    }
    try {
      const space = await readConversationAgentspace(conversationId, this.lifetime);
      if (!this.isConnected || this.#loadedConversationId !== conversationId) return;
      this.#space = space;
      this.#failure = null;
      if (this.#openRunId !== null && !space.snapshots.some((item) => item.runId === this.#openRunId)) {
        this.#closeSlice();
      }
    } catch {
      if (!this.isConnected) return;
      this.#failure = _FAILED;
      this.#space = null;
    }
    this.requestUpdate();
  }

  /** Open one Run's copy in place, or close it when it is already open. One copy is open at a time. */
  async #toggleRun(snapshot: AgentspaceSnapshot): Promise<void> {
    if (this.#openRunId === snapshot.runId) {
      this.#closeSlice();
      return;
    }
    const conversationId = this.handles.conversations.activeConversationId;
    if (conversationId === null) return;
    const runId = snapshot.runId;
    this.#openRunId = runId;
    this.#openFiles = [];
    this.#sliceFailed = false;
    this.#sliceLoading = true;
    this.#filePath = null;
    this.#versions = [];
    this.requestUpdate();
    try {
      const detail = await readAgentspaceSnapshot(conversationId, runId, this.lifetime);
      if (!this.isConnected || this.#openRunId !== runId) return;
      this.#openFiles = detail.files;
    } catch {
      if (this.isConnected && this.#openRunId === runId) this.#sliceFailed = true;
    } finally {
      if (this.isConnected && this.#openRunId === runId) {
        this.#sliceLoading = false;
        this.requestUpdate();
      }
    }
  }

  #closeSlice(): void {
    this.#openRunId = null;
    this.#openFiles = [];
    this.#sliceLoading = false;
    this.#sliceFailed = false;
    this.#filePath = null;
    this.#versions = [];
    this.#historyLoading = false;
    this.requestUpdate();
  }

  /** Read one file's versions and show them in place of the timeline. */
  async #openFile(file: AgentspaceFile): Promise<void> {
    const conversationId = this.handles.conversations.activeConversationId;
    if (conversationId === null) return;
    this.#filePath = file.relativePath;
    this.#versions = [];
    this.#historyLoading = true;
    this.#historyFailed = false;
    this.requestUpdate();
    try {
      const history = await readAgentspaceFileHistory(
        conversationId,
        file.relativePath,
        this.lifetime,
      );
      if (!this.isConnected || this.#filePath !== file.relativePath) return;
      this.#versions = history.versions;
    } catch {
      if (this.isConnected && this.#filePath === file.relativePath) this.#historyFailed = true;
    } finally {
      if (this.isConnected && this.#filePath === file.relativePath) {
        this.#historyLoading = false;
        this.requestUpdate();
      }
    }
  }

  #backToTimeline = (): void => {
    this.#filePath = null;
    this.#versions = [];
    this.#historyLoading = false;
    this.#historyFailed = false;
    this.requestUpdate();
  };

  /** Ask the Shell to open this Run in the traces dock, where what the copy changed can be read. */
  #openTrace(snapshot: AgentspaceSnapshot): void {
    this.dispatchEvent(new CustomEvent<AgentspaceRunOpenDetail>('dl-agentspace-open-run', {
      bubbles: true,
      composed: true,
      detail: {runId: snapshot.runId},
    }));
  }

  #askReset = (): void => {
    this.#confirmReset = true;
    this.requestUpdate();
  };

  #cancelReset = (): void => {
    this.#confirmReset = false;
    this.requestUpdate();
  };

  #reset = async (): Promise<void> => {
    const conversationId = this.handles.conversations.activeConversationId;
    if (conversationId === null || this.#busy) return;
    this.#busy = true;
    try {
      await resetConversationAgentspace(conversationId, this.lifetime);
      if (!this.isConnected) return;
      this.#space = null;
      this.#confirmReset = false;
      this.#closeSlice();
      await this.#load();
    } catch (error) {
      if (!this.isConnected) return;
      const refused = error instanceof ApiError && error.errorKind === 'session_busy';
      requestToast(this, {
        tone: 'error',
        message: refused
          ? msg(_RESET_REFUSED, {id: 'inspectorAgentspace.resetRefused'})
          : msg(_RESET_FAILED, {id: 'inspectorAgentspace.resetFailed'}),
      });
    } finally {
      this.#busy = false;
      if (this.isConnected) this.requestUpdate();
    }
  };

  #changeLabel(change: AgentspaceChange): string {
    if (change === 'added') return msg('added', {id: 'inspectorAgentspace.changeAdded'});
    if (change === 'changed') return msg('changed', {id: 'inspectorAgentspace.changeChanged'});
    return msg('unchanged', {id: 'inspectorAgentspace.changeUnchanged'});
  }

  #fileCount(count: number): string {
    return count === 1
      ? msg('1 file', {id: 'inspectorAgentspace.oneFile'})
      : msg(str`${count} files`, {id: 'inspectorAgentspace.files'});
  }

  /** The signs a diff already speaks, with the words kept for assistive technology. */
  #changeChips(changes: AgentspaceChangeCounts): TemplateResult | typeof nothing {
    const chips: TemplateResult[] = [];
    if (changes.added > 0) {
      chips.push(html`
        <span class=${styles.change} data-change="added">
          <span aria-hidden="true">+${changes.added}</span>
          <span class="dl-sr-only">${msg(str`${changes.added} added`, {
            id: 'inspectorAgentspace.added',
          })}</span>
        </span>`);
    }
    if (changes.changed > 0) {
      chips.push(html`
        <span class=${styles.change} data-change="changed">
          <span aria-hidden="true">~${changes.changed}</span>
          <span class="dl-sr-only">${msg(str`${changes.changed} changed`, {
            id: 'inspectorAgentspace.changed',
          })}</span>
        </span>`);
    }
    if (changes.removed > 0) {
      chips.push(html`
        <span class=${styles.change} data-change="removed">
          <span aria-hidden="true">−${changes.removed}</span>
          <span class="dl-sr-only">${msg(str`${changes.removed} removed`, {
            id: 'inspectorAgentspace.removed',
          })}</span>
        </span>`);
    }
    return chips.length === 0 ? nothing : html`<span class=${styles.changes}>${chips}</span>`;
  }

  #branches(space: ConversationAgentspace): TemplateResult | typeof nothing {
    if (space.heads.length <= 1) return nothing;
    return html`
      <section class=${styles.branches} aria-label=${msg(_BRANCHES, {
        id: 'inspectorAgentspace.branchesAria',
      })}>
        <h3 class=${styles.branchesTitle}>${msg(_BRANCHES, {id: 'inspectorAgentspace.branches'})}</h3>
        <ul class=${styles.branchList} role="list">
          ${repeat(space.heads, (head) => head.laneId, (head) => html`
            <li class=${styles.branch}>
              <span class=${styles.branchName}>${head.laneId}</span>
              <span class=${styles.meta}>${msg(str`Run ${runShort(head.headRunId)}`, {
                id: 'inspectorAgentspace.run',
              })}</span>
            </li>
          `)}
        </ul>
      </section>
    `;
  }

  /** The copies under the day they settled on, newest first: one date per day, one clock per copy. */
  #timeline(space: ConversationAgentspace): TemplateResult {
    const days: {label: string; snapshots: AgentspaceSnapshot[]}[] = [];
    for (const snapshot of space.snapshots) {
      const label = dayLabel(snapshot.createdAt);
      const day = days.at(-1);
      if (day !== undefined && day.label === label) day.snapshots.push(snapshot);
      else days.push({label, snapshots: [snapshot]});
    }
    return html`
      <div class=${styles.days}>
        ${days.map((day) => html`
          <section class=${styles.day}>
            <h3 class=${styles.dayLabel}>${day.label}</h3>
            <ol class=${styles.timeline} role="list">
              ${repeat(day.snapshots, (snapshot) => snapshot.runId, (snapshot) => this.#entry(snapshot))}
            </ol>
          </section>`)}
      </div>
    `;
  }

  #entry(snapshot: AgentspaceSnapshot): TemplateResult {
    const open = this.#openRunId === snapshot.runId;
    const filesId = `agentspace-${snapshot.runId}-files`;
    return html`
      <li class=${styles.entry}>
        <button type="button" class=${styles.runButton}
                aria-expanded=${open ? 'true' : 'false'}
                aria-controls=${open ? filesId : nothing}
                @click=${() => void this.#toggleRun(snapshot)}>
          <span class=${styles.rail}>
            <span class=${styles.clock}>${timeOfDay(snapshot.createdAt, getLocale())}</span>
          </span>
          <span class=${styles.summary}>
            <span class=${styles.rowTop}>
              <span class=${styles.runLabel}>${msg(str`Run ${runShort(snapshot.runId)}`, {
                id: 'inspectorAgentspace.run',
              })}</span>
              ${this.#changeChips(snapshot.changes)}
            </span>
            <span class=${styles.rowFoot}>
              <span class=${styles.meta}>${this.#fileCount(snapshot.fileCount)}</span>
            </span>
          </span>
          ${icon('disclosure', {size: 'xs', className: styles.disclosure})}
        </button>
        ${open ? this.#copyOf(snapshot, filesId) : nothing}
      </li>
    `;
  }

  #copyOf(snapshot: AgentspaceSnapshot, filesId: string): TemplateResult {
    return html`
      <section class=${styles.files} id=${filesId}
               aria-label=${msg(str`Files saved by Run ${runShort(snapshot.runId)}`, {
                 id: 'inspectorAgentspace.filesAria',
               })}>
        ${this.#sliceFailed
          ? html`<p class="dl-note ${styles.quiet}" role="alert">${msg(_COPY_FAILED, {
            id: 'inspectorAgentspace.copyFailed',
          })}</p>`
          : this.#sliceLoading
            ? html`<p class="dl-note ${styles.quiet}" role="status">${msg(_READING_COPY, {
              id: 'inspectorAgentspace.readingCopy',
            })}</p>`
            : this.#copyFiles(snapshot)}
      </section>
    `;
  }

  #copyFiles(snapshot: AgentspaceSnapshot): TemplateResult {
    if (this.#openFiles.length === 0) {
      return html`<p class="dl-note ${styles.quiet}">${msg(_NO_FILES, {
        id: 'inspectorAgentspace.noFiles',
      })}</p>`;
    }
    const changed = this.#openFiles.filter((file) => file.change !== 'unchanged').sort(byPath);
    const unchanged = this.#openFiles.filter((file) => file.change === 'unchanged').sort(byPath);
    return html`
      ${changed.length === 0
        ? html`<p class="dl-note ${styles.quiet}">${msg(_NO_CHANGED, {
          id: 'inspectorAgentspace.noChanged',
        })}</p>`
        : html`
          <h3 class=${styles.filesHeading}>${msg(_CHANGED_HEADING, {
            id: 'inspectorAgentspace.changedHeading',
          })}</h3>
          <ul class=${styles.fileList} role="list">
            ${repeat(changed, (file) => file.relativePath, (file) => this.#fileRow(file))}
          </ul>`}
      ${snapshot.changes.removed === 0 ? nothing : html`
        <p class=${styles.removedNote}>${snapshot.changes.removed === 1
          ? msg('1 file was removed in this Run.', {id: 'inspectorAgentspace.oneRemoved'})
          : msg(str`${snapshot.changes.removed} files were removed in this Run.`, {
            id: 'inspectorAgentspace.removedFiles',
          })}</p>`}
      ${unchanged.length === 0 ? nothing : html`
        <details class=${styles.carried}>
          <summary class=${styles.carriedToggle}>
            ${icon('disclosure', {size: 'xs', className: styles.carriedMark})}
            ${unchanged.length === 1
              ? msg('1 unchanged file', {id: 'inspectorAgentspace.oneUnchanged'})
              : msg(str`${unchanged.length} unchanged files`, {
                id: 'inspectorAgentspace.unchangedFiles',
              })}
          </summary>
          <ul class=${styles.fileList} role="list">
            ${repeat(unchanged, (file) => file.relativePath, (file) => this.#fileRow(file))}
          </ul>
        </details>`}
      <div class=${styles.copyActions}>
        <button type="button" class="dl-text-action" data-open-trace
                @click=${() => this.#openTrace(snapshot)}>
          ${icon('open-external', {size: 'xs'})}${msg('Open this Run’s agent traces', {
            id: 'inspectorAgentspace.openTrace',
          })}
        </button>
      </div>
    `;
  }

  #fileRow(file: AgentspaceFile): TemplateResult {
    return html`
      <li>
        <button type="button" class=${styles.fileRow} data-change=${file.change}
                @click=${() => void this.#openFile(file)}>
          <span class=${styles.marker} aria-hidden="true"></span>
          <span class=${styles.path} title=${file.relativePath}>${file.relativePath}</span>
          <span class=${styles.size}>${formatFileSize(file.sizeBytes)}</span>
        </button>
      </li>
    `;
  }

  /** One file's versions, under a back row; the timeline returns exactly as it was left. */
  #versionsGrain(): TemplateResult {
    const path = this.#filePath ?? '';
    return html`
      <div class=${styles.panel}>
        <div class=${styles.grainHeader}>
          <button type="button" class="dl-text-action" data-back @click=${this.#backToTimeline}>
            ${icon('previous', {size: 'xs'})}${msg(_BACK, {id: 'inspectorAgentspace.back'})}
          </button>
          <span class=${styles.grainPath}>${path}</span>
        </div>
        <p class=${styles.lede}>${msg(_HISTORY_LEDE, {id: 'inspectorAgentspace.historyLede'})}</p>
        ${this.#historyLoading
          ? html`<p class="dl-note ${styles.quiet}" role="status">${msg(_READING_HISTORY, {
            id: 'inspectorAgentspace.readingHistory',
          })}</p>`
          : this.#historyFailed
            ? html`<p class="dl-note ${styles.quiet}" role="alert">${msg(_HISTORY_FAILED, {
              id: 'inspectorAgentspace.historyFailed',
            })}</p>`
            : this.#versions.length === 0
            ? html`<p class="dl-note ${styles.quiet}">${msg(_NO_HISTORY, {
              id: 'inspectorAgentspace.noHistory',
            })}</p>`
            : html`
              <ol class=${styles.historyList} role="list">
                ${repeat(this.#versions, (version) => version.runId, (version) => html`
                  <li class=${styles.version}>
                    <span class=${styles.rail}>
                      <span class=${styles.when}>${dayLabel(version.createdAt)}</span>
                      <span class=${styles.clock}>${timeOfDay(version.createdAt, getLocale())}</span>
                    </span>
                    <span class=${styles.versionBody}>
                      <span class=${styles.runLabel}>${msg(str`Run ${runShort(version.runId)}`, {
                        id: 'inspectorAgentspace.run',
                      })}</span>
                      <span class=${styles.badge} data-change=${version.change}>${
                        this.#changeLabel(version.change)
                      }</span>
                      <span class=${styles.meta}>${formatFileSize(version.sizeBytes)}</span>
                    </span>
                  </li>
                `)}
              </ol>`}
      </div>
    `;
  }

  #resetSection(): TemplateResult {
    if (this.#confirmReset) {
      return html`
        <div class=${styles.reset}>
          <p class=${styles.resetNote}>${msg(_RESET_NOTE, {
            id: 'inspectorAgentspace.resetNote',
          })}</p>
          <div class=${styles.resetActions}>
            <button type="button" class="dl-text-action ${styles.confirmReset}"
                    data-confirm-reset ?disabled=${this.#busy} @click=${this.#reset}>
              ${msg(_RESET_CONFIRM, {id: 'inspectorAgentspace.confirmReset'})}
            </button>
            <button type="button" class="dl-text-action" data-cancel-reset
                    @click=${this.#cancelReset}>${msg(_CANCEL, {
                      id: 'inspectorAgentspace.cancelReset',
                    })}</button>
          </div>
        </div>
      `;
    }
    return html`
      <div class=${styles.reset}>
        <button type="button" class="dl-text-action ${styles.resetAction}" data-reset
                ?disabled=${this.#busy} @click=${this.#askReset}>
          ${icon('remove', {size: 'xs'})}${msg(_RESET, {id: 'inspectorAgentspace.reset'})}
        </button>
      </div>
    `;
  }

  #failedState(): TemplateResult {
    return html`
      <div class=${styles.panel}>
        <p class=${styles.failed} role="alert">${msg(_FAILED, {
          id: 'inspectorAgentspace.failed',
        })}</p>
        <button type="button" class="dl-text-action" data-retry @click=${() => void this.#load()}>
          ${msg(_RETRY, {id: 'inspectorAgentspace.retry'})}
        </button>
      </div>
    `;
  }

  protected override render(): TemplateResult {
    const conversationId = this.handles.conversations.activeConversationId;
    if (conversationId === null) {
      return html`<div class=${styles.panel}><p class="dl-note ${styles.quiet}">${msg(
        _NO_CONVERSATION,
        {id: 'inspectorAgentspace.noConversation'},
      )}</p></div>`;
    }
    if (this.#failure !== null) return this.#failedState();
    if (this.#filePath !== null) return this.#versionsGrain();
    const space = this.#space;
    if (space === null) {
      return html`<div class=${styles.panel}><p class="dl-note ${styles.quiet}" role="status">${
        msg(_LOADING, {id: 'inspectorAgentspace.loading'})
      }</p></div>`;
    }
    return html`
      <div class=${styles.panel}>
        <p class=${styles.lede}>${msg(_LEDE, {id: 'inspectorAgentspace.lede'})}</p>
        ${this.#branches(space)}
        ${space.snapshots.length === 0
          ? html`<p class="dl-note ${styles.quiet}">${msg(_EMPTY, {
            id: 'inspectorAgentspace.empty',
          })}</p>`
          : html`${this.#timeline(space)}${this.#resetSection()}`}
      </div>
    `;
  }
}

customElements.define('dl-inspector-agentspace', DlInspectorAgentspace);

declare global {
  interface HTMLElementTagNameMap {
    'dl-inspector-agentspace': DlInspectorAgentspace;
  }

  interface HTMLElementEventMap {
    'dl-agentspace-open-run': CustomEvent<AgentspaceRunOpenDetail>;
  }
}
