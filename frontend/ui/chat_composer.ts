// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, str, updateWhenLocaleChanges} from '@lit/localize';
import {html, nothing, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import type {AnswerMode} from '../lib/answer_request.ts';
import {formatFileSize} from '../lib/file_size.ts';
import {LightElement} from '../lib/lit_host.ts';
import {attachmentStore, type PendingAttachment} from '../stores/attachmentStore.ts';
import chatStyles from '../styles/chat.module.css';
import {
  acceptsAttachmentUpload,
  attachmentsEnabled,
  classifyAttachmentFile,
  type AttachmentPolicy,
} from './attachment_policy.ts';
import {detectDropItems, type RelativeFile} from './folder-upload.ts';

const STORAGE_KEY = 'dlightrag.answerMode';
const MODES = ['auto', 'fast', 'research'] as const satisfies readonly AnswerMode[];
export type {AnswerMode} from '../lib/answer_request.ts';

const MODE_LABELS: Record<AnswerMode, string> = {
  auto: 'Auto',
  fast: 'Fast',
  research: 'Research',
};

export interface ComposerSubmitDetail {
  query: string;
  mode: AnswerMode | null;
}

export interface ComposerSteerDetail {
  query: string;
}

export interface ComposerWorkspaceDropDetail {
  files: readonly RelativeFile[];
  folderName: string | null;
}

function storedMode(): AnswerMode | null {
  try {
    const value = localStorage.getItem(STORAGE_KEY);
    return MODES.includes(value as AnswerMode) ? value as AnswerMode : null;
  } catch {
    return null;
  }
}

/** Lit-owned draft, attachment admission, answer mode, and keyboard interaction. */
export class DlChatComposer extends LightElement {
  static properties = {
    running: {type: Boolean},
    submissionPending: {type: Boolean},
    stopping: {type: Boolean},
    attachmentPolicy: {attribute: false},
    attachmentAccept: {type: String},
    draft: {state: true},
    mode: {state: true},
    modeOpen: {state: true},
    multiline: {state: true},
    dragActive: {state: true},
    attachments: {state: true},
  };

  declare running: boolean;
  declare submissionPending: boolean;
  declare stopping: boolean;
  declare attachmentPolicy: AttachmentPolicy | null;
  declare attachmentAccept: string;
  declare draft: string;
  declare mode: AnswerMode;
  declare modeOpen: boolean;
  declare multiline: boolean;
  declare dragActive: boolean;
  declare attachments: readonly PendingAttachment[];

  #unsubscribe: (() => void) | null = null;
  #dragCounter = 0;
  #allowNextLineBreak = false;
  #requestMode: AnswerMode | null;

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.running = false;
    this.submissionPending = false;
    this.stopping = false;
    this.attachmentPolicy = null;
    this.attachmentAccept = '';
    this.draft = '';
    this.#requestMode = storedMode();
    this.mode = this.#requestMode ?? 'auto';
    this.modeOpen = false;
    this.multiline = false;
    this.dragActive = false;
    this.attachments = [...attachmentStore.list()];
  }

  get hasDraft(): boolean {
    return Boolean(this.draft) || attachmentStore.size > 0;
  }

  override connectedCallback(): void {
    super.connectedCallback();
    // Attachment store reads: list(), size, imageCount.
    this.#unsubscribe ??= attachmentStore.subscribe(() => {
      this.attachments = [...attachmentStore.list()];
    });
    document.addEventListener('click', this.#closeModeMenu);
    document.addEventListener('dragenter', this.#dragEnter);
    document.addEventListener('dragleave', this.#dragLeave);
    document.addEventListener('dragover', this.#dragOver);
    document.addEventListener('drop', this.#drop);
    document.addEventListener('paste', this.#paste);
  }

  override disconnectedCallback(): void {
    super.disconnectedCallback();
    this.#unsubscribe?.();
    this.#unsubscribe = null;
    document.removeEventListener('click', this.#closeModeMenu);
    document.removeEventListener('dragenter', this.#dragEnter);
    document.removeEventListener('dragleave', this.#dragLeave);
    document.removeEventListener('dragover', this.#dragOver);
    document.removeEventListener('drop', this.#drop);
    document.removeEventListener('paste', this.#paste);
  }

  focusInput(): void {
    void this.updateComplete.then(() => this.#input()?.focus());
  }

  clearDraft(): void {
    this.clearText();
    attachmentStore.clear();
  }

  clearText(): void {
    this.draft = '';
    this.multiline = false;
    void this.updateComplete.then(() => this.#resize());
  }

  clearSubmittedText(query: string): boolean {
    if (this.draft.trim() !== query) return false;
    this.clearText();
    return true;
  }

  restoreSubmission(query: string, requestMode: AnswerMode | null): void {
    this.draft = query;
    this.#requestMode = requestMode;
    this.mode = requestMode ?? 'auto';
    this.modeOpen = false;
    try {
      if (requestMode === null) localStorage.removeItem(STORAGE_KEY);
      else localStorage.setItem(STORAGE_KEY, requestMode);
    } catch {
      // The restored mode still applies for this page when storage is blocked.
    }
    void this.updateComplete.then(() => {
      this.#resize();
      this.#input()?.focus();
    });
  }

  addFiles(files: Iterable<File>): void {
    for (const file of files) this.#addAttachment(file);
  }

  protected override render(): TemplateResult {
    const hasText = Boolean(this.draft.trim());
    const stop = this.running && !hasText;
    const steer = this.running && hasText;
    const attachmentsAvailable = this.attachmentPolicy
      ? attachmentsEnabled(this.attachmentPolicy)
      : false;
    return html`
      <div class="drop-overlay ${this.dragActive ? 'active' : ''}" aria-hidden="true">
        <div class="drop-overlay-content">${msg('Drop files or folders here', {id: 'chatComposer.dropCopy'})}</div>
      </div>
      <div class="composer" id="composer">
        <div class="composer-inner">
          <div class="thumbnail-strip" id="thumbnail-strip">
            ${repeat(this.attachments, (item) => item.id, (item) => this.#attachment(item))}
          </div>
          <form id="query-form" class="composer-form ${this.multiline ? 'multiline' : ''}"
                @submit=${this.#submitForm}>
            <button type="button" class="composer-plus" id="composer-plus" aria-label=${msg('Attach files', {id: 'chatComposer.attachFiles'})}
                    ?disabled=${!attachmentsAvailable}
                    aria-disabled=${attachmentsAvailable ? 'false' : 'true'}
                    title=${attachmentsAvailable ? nothing : msg('Attachments are currently unavailable.', {id: 'chatComposer.attachmentsUnavailable'})}
                    @click=${this.#openAttachmentPicker}>
              <svg class="composer-plus-icon" width="24" height="24" viewBox="0 0 24 24"
                   fill="none" stroke="currentColor" stroke-linecap="round"
                   stroke-linejoin="round" aria-hidden="true" focusable="false">
                <path d="M12 5v14"></path><path d="M5 12h14"></path>
              </svg>
            </button>
            <textarea name="query" aria-label=${msg('Message', {id: 'chatComposer.messageAria'})} placeholder=${msg('Ask anything', {id: 'chatComposer.placeholder'})}
                      class="composer-input" rows="1" autocomplete="off"
                      .value=${this.draft}
                      @input=${this.#inputChanged}
                      @keydown=${this.#inputKeydown}
                      @beforeinput=${this.#beforeInput}
                      @keyup=${this.#inputKeyup}></textarea>
            <div class="composer-mode">
              <button type="button" class="composer-mode-trigger" id="composer-mode"
                      aria-haspopup="menu" aria-expanded=${String(this.modeOpen)}
                      aria-label=${msg(str`Answer mode: ${MODE_LABELS[this.mode]}`, {id: `chatComposer.modeAria.${this.mode}`})}
                      @click=${this.#toggleModeMenu} @keydown=${this.#modeTriggerKeydown}>
                ${msg(MODE_LABELS[this.mode], {id: `chatComposer.mode.${this.mode}`})}
              </button>
              <div class="composer-mode-menu" id="composer-mode-menu" role="menu" aria-label=${msg('Answer mode', {id: 'chatComposer.modeMenuAria'})}
                   ?hidden=${!this.modeOpen} @keydown=${this.#modeMenuKeydown}>
                ${MODES.map((mode) => html`
                  <button type="button" role="menuitemradio" data-mode=${mode}
                          aria-checked=${String(this.mode === mode)} tabindex="-1"
                          @click=${() => this.#selectMode(mode)}>${msg(MODE_LABELS[mode], {id: `chatComposer.mode.${mode}`})}</button>
                `)}
              </div>
            </div>
            <button type="submit"
                    class="composer-send ${stop ? 'is-stop' : ''} ${steer ? 'is-steer' : ''}"
                    aria-label=${this.submissionPending
                      ? msg('Submitting', {id: 'chatComposer.submitting'})
                      : this.running
                        ? (hasText ? msg('Steer', {id: 'chatComposer.steer'}) : msg('Stop', {id: 'chatComposer.stop'}))
                        : msg('Send', {id: 'chatComposer.send'})}
                    ?disabled=${this.submissionPending
                      || (!hasText && !this.running) || this.stopping}
                    @click=${this.#sendClicked}>
              <svg class="composer-send-icon composer-send-icon--send" width="18" height="18"
                   viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                   stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">
                <line x1="22" y1="2" x2="11" y2="13"></line>
                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
              </svg>
              <svg class="composer-send-icon composer-send-icon--stop" width="16" height="16"
                   viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" focusable="false">
                <rect x="6" y="6" width="12" height="12" rx="2"></rect>
              </svg>
            </button>
          </form>
        </div>
        <input class="hidden" type="file" id="attachment-input"
               accept=${this.attachmentAccept} multiple
               @change=${this.#attachmentInputChanged}>
      </div>
    `;
  }

  #attachment(item: PendingAttachment): TemplateResult {
    if (item.kind === 'image') {
      return html`
        <div class=${chatStyles.thumbnailItem}>
          <img class=${chatStyles.thumbnailImg} src=${item.objectUrl} alt=${item.file.name}>
          <button type="button" class=${chatStyles.thumbnailRemove}
                  aria-label=${msg(str`Remove ${item.file.name}`, {id: 'chatComposer.removeAttachment'})}
                  @click=${() => attachmentStore.remove(item.id)}>×</button>
        </div>
      `;
    }
    return html`
      <span class=${chatStyles.documentChip} data-document-attachment="true">
        <span class=${chatStyles.documentChipInfo}>
          <span class=${chatStyles.documentChipName}>${item.file.name}</span>
          <span class=${chatStyles.documentChipMeta}>${formatFileSize(item.file.size)}</span>
        </span>
        <button type="button" class=${chatStyles.documentChipRemove}
                aria-label=${msg(str`Remove ${item.file.name}`, {id: 'chatComposer.removeAttachment'})}
                @click=${() => attachmentStore.remove(item.id)}>×</button>
      </span>
    `;
  }

  #input(): HTMLTextAreaElement | null {
    return this.querySelector<HTMLTextAreaElement>('.composer-input');
  }

  #inputChanged(event: Event): void {
    this.draft = (event.currentTarget as HTMLTextAreaElement).value;
    void this.updateComplete.then(() => this.#resize());
  }

  #resize(): void {
    const input = this.#input();
    if (!input) return;
    const computed = getComputedStyle(input);
    const lineHeight = parseFloat(computed.lineHeight) || 24;
    const maxHeight = parseFloat(computed.maxHeight) || 160;
    input.style.height = 'auto';
    const contentHeight = input.scrollHeight;
    this.multiline = this.draft.includes('\n') || contentHeight > lineHeight * 1.5;
    input.style.height = `${Math.min(contentHeight, maxHeight)}px`;
    input.style.overflowY = contentHeight > maxHeight ? 'auto' : 'hidden';
  }

  #inputKeydown(event: KeyboardEvent): void {
    if (event.key === 'Escape' && this.running) {
      this.#cancelIntent();
      return;
    }
    if (event.key === 'Enter') this.#allowNextLineBreak = event.shiftKey;
  }

  #beforeInput(event: InputEvent): void {
    if (event.inputType !== 'insertLineBreak') return;
    if (event.isComposing || this.#allowNextLineBreak) {
      this.#allowNextLineBreak = false;
      return;
    }
    event.preventDefault();
    this.#allowNextLineBreak = false;
    this.#primaryIntent();
  }

  #inputKeyup(event: KeyboardEvent): void {
    if (event.key === 'Enter') this.#allowNextLineBreak = false;
  }

  #submitForm = (event: SubmitEvent): void => {
    event.preventDefault();
    this.#primaryIntent();
  };

  #sendClicked = (event: MouseEvent): void => {
    if (!this.running) return;
    event.preventDefault();
    this.#primaryIntent();
  };

  #primaryIntent(): void {
    if (this.submissionPending) return;
    const query = this.draft.trim();
    if (this.running) {
      if (query) {
        this.dispatchEvent(new CustomEvent<ComposerSteerDetail>('dl-composer-steer', {
          bubbles: true,
          composed: true,
          detail: {query},
        }));
      } else {
        this.#cancelIntent();
      }
      return;
    }
    if (!query) return;
    this.draft = '';
    this.dispatchEvent(new CustomEvent<ComposerSubmitDetail>('dl-composer-submit', {
      bubbles: true,
      composed: true,
      detail: {query, mode: this.#requestMode},
    }));
    void this.updateComplete.then(() => this.#resize());
  }

  #cancelIntent(): void {
    this.dispatchEvent(new CustomEvent('dl-composer-cancel', {
      bubbles: true,
      composed: true,
    }));
  }

  #openAttachmentPicker = (): void => {
    this.querySelector<HTMLInputElement>('input[type="file"]')?.click();
  };

  #attachmentInputChanged = (event: Event): void => {
    const input = event.currentTarget as HTMLInputElement;
    for (const file of Array.from(input.files ?? [])) this.#addAttachment(file);
    input.value = '';
  };

  #addAttachment(file: File): void {
    const policy = this.attachmentPolicy;
    if (!policy) return;
    const kind = classifyAttachmentFile(file, policy.extensions);
    if (kind === 'unsupported') return;
    if (!acceptsAttachmentUpload(
      file,
      {total: attachmentStore.size, images: attachmentStore.imageCount},
      policy,
    )) return;
    attachmentStore.add(file, kind);
  }

  #toggleModeMenu = (event: Event): void => {
    event.stopPropagation();
    this.modeOpen = !this.modeOpen;
    if (this.modeOpen) void this.updateComplete.then(() => this.#focusMode(this.mode));
  };

  #selectMode(mode: AnswerMode): void {
    this.mode = mode;
    this.#requestMode = mode;
    this.modeOpen = false;
    try {
      localStorage.setItem(STORAGE_KEY, mode);
    } catch {
      // The selected mode still applies for this page when storage is blocked.
    }
    this.focusInput();
  }

  #modeTriggerKeydown = (event: KeyboardEvent): void => {
    if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return;
    event.preventDefault();
    this.modeOpen = true;
    const mode = event.key === 'ArrowUp' ? MODES[MODES.length - 1] : MODES[0];
    void this.updateComplete.then(() => this.#focusMode(mode));
  };

  #modeMenuKeydown = (event: KeyboardEvent): void => {
    const target = event.target as HTMLButtonElement;
    const index = MODES.indexOf(target.dataset.mode as AnswerMode);
    if (event.key === 'Escape') {
      event.preventDefault();
      this.modeOpen = false;
      this.querySelector<HTMLButtonElement>('.composer-mode-trigger')?.focus();
      return;
    }
    if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    let next = index;
    if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = MODES.length - 1;
    else if (event.key === 'ArrowDown') next = (index + 1) % MODES.length;
    else next = (index - 1 + MODES.length) % MODES.length;
    this.#focusMode(MODES[next]);
  };

  #focusMode(mode: AnswerMode): void {
    this.querySelector<HTMLButtonElement>(`[data-mode="${mode}"]`)?.focus();
  }

  #closeModeMenu = (): void => {
    if (this.modeOpen) this.modeOpen = false;
  };

  #dragEnter = (event: DragEvent): void => {
    event.preventDefault();
    if (event.dataTransfer?.types.includes('Files')) {
      this.#dragCounter += 1;
      this.dragActive = true;
    }
  };

  #dragLeave = (event: DragEvent): void => {
    event.preventDefault();
    this.#dragCounter = Math.max(0, this.#dragCounter - 1);
    if (this.#dragCounter === 0) this.dragActive = false;
  };

  #dragOver = (event: DragEvent): void => {
    event.preventDefault();
  };

  #drop = (event: DragEvent): void => {
    event.preventDefault();
    this.#dragCounter = 0;
    this.dragActive = false;
    const items = event.dataTransfer?.items;
    if (!items || items.length === 0) return;
    void detectDropItems(items, (image) => this.#addAttachment(image)).then((result) => {
      if (result.files.length === 0) return;
      this.dispatchEvent(
        new CustomEvent<ComposerWorkspaceDropDetail>('dl-composer-workspace-drop', {
          bubbles: true,
          composed: true,
          detail: result,
        }),
      );
    });
  };

  #paste = (event: ClipboardEvent): void => {
    const items = event.clipboardData?.items;
    if (!items) return;
    for (const item of Array.from(items)) {
      if (!item.type.startsWith('image/')) continue;
      const file = item.getAsFile();
      if (file) this.#addAttachment(file);
    }
  };
}

customElements.define('dl-chat-composer', DlChatComposer);

declare global {
  interface HTMLElementTagNameMap {
    'dl-chat-composer': DlChatComposer;
  }
}
