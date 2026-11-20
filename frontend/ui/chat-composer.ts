// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, str} from '@lit/localize';
import {html, nothing, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import {
  type DlMenu,
  icon,
  type IconName,
  isImeKey,
  menuButtonFocus,
  type MenuDismissDetail,
  type MenuFocus,
} from '../design-system/index.ts';
import type {AgentEffort} from '../api/agent-effort.ts';
import type {AnswerDelivery, AnswerMode} from '../api/answer-request.ts';
import {listSkills, type SkillSummary} from '../api/skills.ts';
import {raise} from '../lib/dom.ts';
import {formatFileSize} from '../lib/file-size.ts';
import {localizedText} from '../lib/localized-text.ts';
import {LightElement, StoreController} from '../lib/lit-host.ts';
import {commandSuggestions, findWebCommand, type WebCommand} from '../lib/web-commands.ts';
import {productionHandles, type AppHandles} from '../stores/app-handles.ts';
import {
  committedSkillDirective,
  parseSkillDirective,
  skillDirectiveState,
  skillGhostSuffix,
} from '../lib/skill-directive.ts';
import type {PendingAttachment} from '../stores/attachment-store.ts';
import {
  EMPTY_AGENT_EFFORT_OFFER,
  EFFORT_LABELS,
  type AgentEffortOffer,
  offeredLevels,
  storeAgentEffort,
  storedAgentEffort,
} from '../lib/agent-effort.ts';
import chatStyles from '../styles/chat.module.css';
import {
  acceptsAttachmentUpload,
  attachmentsEnabled,
  classifyAttachmentFile,
  type AttachmentPolicy,
} from '../lib/attachment-policy.ts';
import {detectDropItems, type RelativeFile} from './folder-upload.ts';
import {readStored, writeStored} from '../lib/storage.ts';

const STORAGE_KEY = 'dlightrag.answerMode';
/** Where an unsent message waits: a reload, a provider return, or a crash must not lose it. */
const DRAFT_STORAGE_KEY = 'dlightrag.composerDraft';
/** How long typing settles before the draft is written; a page going away writes at once. */
const DRAFT_WRITE_MILLISECONDS = 400;
const MODES = ['auto', 'fast', 'research'] as const satisfies readonly AnswerMode[];

type Picker = 'mode' | 'effort';

/** One glyph per level: the lucide dial with its needle at that level's stop.

 * The needle carries no accessible meaning, the label and aria do, so a level the
 * deployment does not offer never renders one of these.
 */
const EFFORT_ICONS: Record<AgentEffort, IconName> = {
  low: 'effort-low',
  high: 'effort-high',
  max: 'effort-max',
};

const MODE_LABELS: Record<AnswerMode, string> = {
  auto: 'Auto',
  fast: 'Fast',
  research: 'Research',
};

/** The id of a skill's option in the menu: the box points at the active one with it. A skill name is
 *  lowercase letters, digits, and hyphens, so it is safe in an id as it stands. */
function skillOptionId(skillName: string): string {
  return `skill-option-${skillName}`;
}

function commandOptionId(commandName: string): string {
  return `command-option-${commandName}`;
}

type SlashMenuItem =
  | {readonly kind: 'skill'; readonly skill: SkillSummary}
  | {readonly kind: 'command'; readonly command: WebCommand};

/** What `dl-composer-submit` carries. The event is cancelable: a listener that cannot take the message
 *  now cancels it, and the composer keeps the words in its box. */
export interface ComposerSubmitDetail {
  query: string;
  mode: AnswerMode | null;
  requestedSkill: string | null;
  /** The caller's own agent effort; null asks for the deployment default. */
  effort: AgentEffort | null;
  /** The delivery the reader chose while a Run is live; absent when the conversation is idle. */
  delivery?: AnswerDelivery;
}

export interface ComposerWorkspaceDropDetail {
  files: readonly RelativeFile[];
  folderName: string | null;
}

function storedMode(): AnswerMode | null {
  const value = readStored(STORAGE_KEY);
  return MODES.includes(value as AnswerMode) ? value as AnswerMode : null;
}

/** Lit-owned draft, attachment admission, answer mode and effort, and keyboard interaction. */
export class DlChatComposer extends LightElement {
  static properties = {
    handles: {attribute: false},
    running: {type: Boolean},
    live: {type: Boolean},
    submissionPending: {type: Boolean},
    stopping: {type: Boolean},
    attachmentPolicy: {attribute: false},
    attachmentAccept: {type: String},
    agentEffortOffer: {attribute: false},
    draft: {state: true},
    mode: {state: true},
    effort: {state: true},
    picker: {state: true},
    multiline: {state: true},
    dragActive: {state: true},
    skills: {state: true},
    skillNotice: {state: true},
    skillMenuOpen: {state: true},
    skillActive: {state: true},
  };

  declare handles: AppHandles;
  declare running: boolean;
  /** Whether the conversation has a live Run, whether or not this page is following it. */
  declare live: boolean;
  declare submissionPending: boolean;
  declare stopping: boolean;
  declare attachmentPolicy: AttachmentPolicy | null;
  declare attachmentAccept: string;
  declare agentEffortOffer: AgentEffortOffer;
  declare draft: string;
  declare mode: AnswerMode;
  declare effort: AgentEffort | null;
  /** The one menu of the two beside the box that is open, if any. */
  declare picker: Picker | null;
  declare multiline: boolean;
  declare dragActive: boolean;
  declare skills: readonly SkillSummary[];
  declare skillNotice: boolean;
  declare skillMenuOpen: boolean;
  declare skillActive: number;

  #dragCounter = 0;
  #allowNextLineBreak = false;
  #draftWrite: ReturnType<typeof setTimeout> | null = null;
  #draftRestored = false;
  #requestMode: AnswerMode | null;
  #requestEffort: AgentEffort | null = null;
  #delivery: AnswerDelivery = 'queue';
  #effortInitialized = false;
  readonly #modeButton = this.#buttonFor('mode');
  readonly #effortButton = this.#buttonFor('effort');

  constructor() {
    super();
    this.handles = productionHandles();
    this.running = false;
    this.submissionPending = false;
    this.stopping = false;
    this.attachmentPolicy = null;
    this.attachmentAccept = '';
    this.draft = '';
    this.#requestMode = storedMode();
    this.mode = this.#requestMode ?? 'auto';
    this.effort = null;
    this.picker = null;
    this.agentEffortOffer = EMPTY_AGENT_EFFORT_OFFER;
    this.multiline = false;
    this.dragActive = false;
    this.skills = [];
    this.skillNotice = false;
    this.skillMenuOpen = false;
    this.skillActive = -1;
    // Attachment store reads: list(), size, imageCount.
    new StoreController(this, () => [this.handles.attachments]);
  }

  override willUpdate(): void {
    if (this.#effortInitialized || offeredLevels(this.agentEffortOffer).length === 0) return;
    this.#effortInitialized = true;
    this.#requestEffort = storedAgentEffort(this.agentEffortOffer);
    this.effort = this.#requestEffort ?? this.agentEffortOffer.default;
  }

  get hasDraft(): boolean {
    return Boolean(this.draft) || this.handles.attachments.size > 0;
  }

  override connectedCallback(): void {
    super.connectedCallback();
    this.#restoreDraft();
    const {lifetime: signal} = this;
    // A page going away takes the box with it: whatever the debounce still owes is written now,
    // so a reload - a provider return included - finds the message where the reader left it.
    window.addEventListener('pagehide', this.#flushDraft, {signal});
    document.addEventListener('visibilitychange', this.#flushDraft, {signal});
    document.addEventListener('click', this.#closeMenus, {signal});
    document.addEventListener('dragenter', this.#dragEnter, {signal});
    document.addEventListener('dragleave', this.#dragLeave, {signal});
    document.addEventListener('dragover', this.#dragOver, {signal});
    document.addEventListener('drop', this.#drop, {signal});
    document.addEventListener('paste', this.#paste, {signal});
  }

  override disconnectedCallback(): void {
    super.disconnectedCallback();
    // A write still owed must not land once the box is gone: a sign-out forgets the draft, and
    // a debounce firing behind it would put the words back.
    if (this.#draftWrite) {
      clearTimeout(this.#draftWrite);
      this.#draftWrite = null;
    }
  }

  focusInput(): void {
    void this.updateComplete.then(() => this.#input()?.focus());
  }

  clearDraft(): void {
    this.clearText();
    this.handles.attachments.clear();
  }

  clearText(): void {
    this.draft = '';
    this.multiline = false;
    this.skillMenuOpen = false;
    this.skillActive = -1;
    // The box is empty on purpose: what waits in storage goes with it.
    this.#flushDraft();
    void this.updateComplete.then(() => this.#resize());
  }

  /** What the reader last left in the box, once per element: a reconnect is not a reload. */
  #restoreDraft(): void {
    if (this.#draftRestored) return;
    this.#draftRestored = true;
    this.draft = readStored(DRAFT_STORAGE_KEY) ?? '';
  }

  #scheduleDraftWrite(): void {
    if (this.#draftWrite) clearTimeout(this.#draftWrite);
    this.#draftWrite = setTimeout(() => {
      this.#draftWrite = null;
      writeStored(DRAFT_STORAGE_KEY, this.draft || null);
    }, DRAFT_WRITE_MILLISECONDS);
  }

  /** Write the draft now, whether or not typing has settled. */
  #flushDraft = (): void => {
    if (this.#draftWrite) {
      clearTimeout(this.#draftWrite);
      this.#draftWrite = null;
    }
    writeStored(DRAFT_STORAGE_KEY, this.draft || null);
  };

  clearSubmittedText(query: string): boolean {
    if (this.draft.trim() !== query) return false;
    this.clearText();
    return true;
  }

  restoreSubmission(
    query: string,
    requestMode: AnswerMode | null,
    requestedSkill: string | null = null,
    requestEffort: AgentEffort | null = null,
  ): void {
    const restored = requestedSkill ? `/skill:${requestedSkill} ${query}` : query;
    // Words the reader has typed since are theirs too: the message they asked to edit goes first and
    // what they were writing follows a blank line, so editing one never costs them the other.
    const typed = this.draft;
    this.draft = typed.trim() === '' || typed.trim() === restored.trim() ? restored : `${restored}\n\n${typed}`;
    this.#flushDraft();
    this.#requestMode = requestMode;
    this.mode = requestMode ?? 'auto';
    this.picker = null;
    this.#effortInitialized = true;
    this.#requestEffort = requestEffort;
    this.effort = requestEffort ?? this.agentEffortOffer.default;
    storeAgentEffort(requestEffort);
    writeStored(STORAGE_KEY, requestMode);
    void this.updateComplete.then(() => {
      this.#resize();
      this.#input()?.focus();
    });
  }

  protected override render(): TemplateResult {
    const hasText = Boolean(this.draft.trim());
    const stop = this.running && !hasText;
    const delivery = this.live && hasText;
    const attachmentsAvailable = this.attachmentPolicy
      ? attachmentsEnabled(this.attachmentPolicy)
      : false;
    // The box keeps focus while the slash menu is open, so it points at the menu and at the row the
    // arrows are on. A textarea may not take role=combobox or aria-expanded (ARIA in HTML), so it
    // states the list it completes from and where the highlight is; one status says how many skills
    // match, and the command rows carry their own names and hints.
    const suggestions = this.#skillSuggestions();
    const commands = commandSuggestions(this.draft);
    const menuItems = this.#menuItems(suggestions, commands);
    const skillMenuShown = this.skillMenuOpen && menuItems.length > 0;
    const activeItem = skillMenuShown && this.skillActive >= 0
      ? menuItems[this.skillActive]
      : undefined;
    return html`
      <div class="drop-overlay ${this.dragActive ? 'active' : ''}" aria-hidden="true">
        <div class="drop-overlay-content">${msg('Drop files or folders here', {id: 'chatComposer.dropCopy'})}</div>
      </div>
      <div class="composer" id="composer">
        <div class="composer-inner">
          <div class="skill-menu dl-anchored dl-anchored--above" id="skill-menu" role="listbox"
               aria-label=${msg('Commands and skills', {id: 'chatComposer.slashMenuAria'})}
               ?hidden=${!skillMenuShown}>
            ${repeat(suggestions, (skill) => skill.name, (skill, index) => html`
              <button type="button" role="option" id=${skillOptionId(skill.name)}
                      class="skill-menu-item ${index === this.skillActive ? 'active' : ''}"
                      aria-selected=${String(index === this.skillActive)}
                      title=${skill.description}
                      @click=${() => this.#applySkill(skill.name)}
                      @mousemove=${() => { this.skillActive = index; }}>
                <span class="skill-menu-name">${skill.name}</span>
                <span class="skill-menu-desc">${skill.description}</span>
                <span class="skill-menu-source ${skill.source}">${skill.source === 'owner'
                  ? msg('Mine', {id: 'chatComposer.skillSource.owner'})
                  : skill.source === 'global'
                    ? msg('Global', {id: 'chatComposer.skillSource.global'})
                    : msg('Built-in', {id: 'chatComposer.skillSource.builtin'})}</span>
              </button>
            `)}
            ${commands.map((command, index) => {
              const itemIndex = suggestions.length + index;
              return html`
                <button type="button" role="option" id=${commandOptionId(command.name)}
                        class="command-menu-item ${itemIndex === this.skillActive ? 'active' : ''}"
                        aria-selected=${String(itemIndex === this.skillActive)}
                        title=${localizedText(command.description)}
                        @click=${() => this.#applyCommand(command)}
                        @mousemove=${() => { this.skillActive = itemIndex; }}>
                  <span class="command-menu-name">/${command.name}</span>
                  <span class="command-menu-desc">${localizedText(command.description)}</span>
                  <span class="command-menu-hint">${localizedText(command.inputHint)}</span>
                </button>`;
            })}
          </div>
          <span class="dl-sr-only" role="status" aria-live="polite" data-skill-status>${skillMenuShown && suggestions.length > 0
            ? this.#skillCountText(suggestions.length)
            : ''}</span>
          <div class="thumbnail-strip" id="thumbnail-strip">
            ${repeat(this.handles.attachments.list(), (item) => item.id, (item) => this.#attachment(item))}
          </div>
          <form id="query-form" class="composer-form ${this.multiline ? 'multiline' : ''}"
                @submit=${this.#submitForm}>
            <button type="button" class="composer-plus" id="composer-plus" aria-label=${msg('Attach files', {id: 'chatComposer.attachFiles'})}
                    ?disabled=${!attachmentsAvailable}
                    aria-disabled=${attachmentsAvailable ? 'false' : 'true'}
                    title=${attachmentsAvailable ? nothing : msg('Attachments are currently unavailable.', {id: 'chatComposer.attachmentsUnavailable'})}
                    @click=${this.#openAttachmentPicker}>
              ${icon('attach', {size: 'lg', className: 'composer-plus-icon'})}
            </button>
            <div class="composer-input-wrap">
              <div class="composer-input-mirror" aria-hidden="true">${this.draft}<span class="skill-ghost-text">${this.#ghostText()}</span></div>
              <textarea name="query" aria-label=${msg('Message', {id: 'chatComposer.messageAria'})} placeholder=${msg('Ask, or assign.', {id: 'chatComposer.placeholder'})}
                        class="composer-input" rows="1" autocomplete="off"
                        aria-autocomplete="list"
                        aria-controls=${skillMenuShown ? 'skill-menu' : nothing}
                        aria-activedescendant=${activeItem ? this.#menuItemId(activeItem) : nothing}
                        .value=${this.draft}
                        @input=${this.#inputChanged}
                        @keydown=${this.#inputKeydown}
                        @beforeinput=${this.#beforeInput}
                        @keyup=${this.#inputKeyup}></textarea>
            </div>
            <div class="composer-mode">
              <button type="button" class="composer-mode-trigger" id="composer-mode"
                      aria-haspopup="menu" aria-controls="composer-mode-menu"
                      aria-expanded=${String(this.picker === 'mode')}
                      aria-label=${msg(str`Answer mode: ${MODE_LABELS[this.mode]}`, {id: `chatComposer.modeAria.${this.mode}`})}
                      @click=${this.#modeButton.toggle} @keydown=${this.#modeButton.keydown}>
                ${msg(MODE_LABELS[this.mode], {id: `chatComposer.mode.${this.mode}`})}
              </button>
              <dl-menu class="composer-mode-menu dl-anchored dl-anchored--above dl-anchored--end"
                   id="composer-mode-menu" role="menu" aria-label=${msg('Answer mode', {id: 'chatComposer.modeMenuAria'})}
                   ?hidden=${this.picker !== 'mode'} @dl-menu-dismiss=${this.#modeButton.dismissed}>
                ${MODES.map((mode) => html`
                  <button type="button" role="menuitemradio" data-mode=${mode}
                          aria-checked=${String(this.mode === mode)} tabindex="-1"
                          @click=${() => this.#selectMode(mode)}>${msg(MODE_LABELS[mode], {id: `chatComposer.mode.${mode}`})}</button>
                `)}
              </dl-menu>
            </div>
            ${this.#deliveryControl(delivery)}
            ${this.#effortControl()}
            <button type="submit"
                    class="composer-send ${stop ? 'is-stop' : ''}"
                    aria-label=${this.submissionPending
                      ? msg('Submitting', {id: 'chatComposer.submitting'})
                      : delivery
                        ? (this.#delivery === 'steer'
                          ? msg('Steer', {id: 'chatComposer.steer'})
                          : msg('Queue', {id: 'chatComposer.queue'}))
                        : this.running
                          ? msg('Stop', {id: 'chatComposer.stop'})
                          : msg('Send', {id: 'chatComposer.send'})}
                    ?disabled=${this.submissionPending
                      || (!hasText && !this.live && !this.running) || this.stopping}
                    @click=${this.#sendClicked}>
              ${icon('send', {size: 'md', className: 'composer-send-icon composer-send-icon--send'})}
              ${icon('stop', {size: 'sm', className: 'composer-send-icon composer-send-icon--stop'})}
            </button>
          </form>
          ${this.#commandHint()}
          ${this.skillNotice ? html`
            <div class="skill-notice" role="alert">
              ${msg('A skill directive needs a question.', {id: 'chatComposer.skillQuestionRequired'})}
            </div>` : nothing}
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
                  @click=${() => this.handles.attachments.remove(item.id)}>${icon('close', {size: 'xs'})}</button>
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
                @click=${() => this.handles.attachments.remove(item.id)}>${icon('close', {size: 'xs'})}</button>
      </span>
    `;
  }

  #input(): HTMLTextAreaElement | null {
    return this.querySelector<HTMLTextAreaElement>('.composer-input');
  }

  #inputChanged(event: Event): void {
    this.draft = (event.currentTarget as HTMLTextAreaElement).value;
    this.#scheduleDraftWrite();
    this.skillNotice = false;
    const skillMenuOpen = skillDirectiveState(this.draft) !== null;
    const menuOpening = skillMenuOpen && !this.skillMenuOpen;
    if (skillMenuOpen) this.picker = null;
    this.skillMenuOpen = skillMenuOpen;
    this.skillActive = -1;
    // A Skill published or deleted since the menu last opened shows up when it opens again.
    if (menuOpening) {
      void listSkills()
        .then((skills) => { this.skills = skills; })
        .catch(() => {});
    }
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
    const mirror = this.querySelector<HTMLDivElement>('.composer-input-mirror');
    if (mirror) mirror.scrollTop = input.scrollTop;
  }

  #inputKeydown(event: KeyboardEvent): void {
    // A key that confirms or cancels an IME composition is the composition's: it must not stop the Run,
    // move through the skill menu, or pick from it.
    if (isImeKey(event)) return;
    if (event.key === 'Escape' && this.running) {
      this.#cancelIntent();
      return;
    }
    const menuItems = this.#menuItems();
    const menuOpen = this.skillMenuOpen && menuItems.length > 0;
    const highlighted = this.skillActive >= 0 ? menuItems[this.skillActive] : undefined;
    if (event.key === 'Tab' && menuOpen) {
      event.preventDefault();
      // Tab takes the first row for a reader who has not moved onto one.
      this.#acceptMenuItem(highlighted ?? menuItems[0]);
      return;
    }
    if (event.key === 'ArrowDown' && menuOpen) {
      event.preventDefault();
      this.skillActive = (this.skillActive + 1) % menuItems.length;
      this.#scrollActiveItem();
      return;
    }
    if (event.key === 'ArrowUp' && menuOpen) {
      event.preventDefault();
      this.skillActive = (this.skillActive - 1 + menuItems.length) % menuItems.length;
      this.#scrollActiveItem();
      return;
    }
    if (event.key === 'Enter' && menuOpen && highlighted) {
      event.preventDefault();
      this.#acceptMenuItem(highlighted);
      return;
    }
    if (event.key === 'Escape' && this.skillMenuOpen) {
      this.skillMenuOpen = false;
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
    if (!this.live && !this.running) return;
    event.preventDefault();
    this.#primaryIntent();
  };

  #primaryIntent(): void {
    if (this.submissionPending) return;
    const query = this.draft.trim();
    if (!query) {
      if (this.running) this.#cancelIntent();
      return;
    }
    // A registered command is the conversation's control surface, not a message to the model:
    // even while a Run is active it must reach the chat as a command, never as delivery.
    if (this.live && findWebCommand(query) !== null) {
      this.#offerSubmission(query);
      return;
    }
    this.#offerSubmission(query, this.live ? this.#delivery : undefined);
  }

  /** Offer one message to the chat, which decides what a goal command or Skill directive means. */
  #offerSubmission(query: string, delivery?: AnswerDelivery): void {
    // A registered command is the conversation's command surface, not a Skill directive: keep
    // the words whole and leave the decision to the chat, which knows this conversation.
    const directive = findWebCommand(query) === null ? parseSkillDirective(query) : null;
    let submitQuery = query;
    let requestedSkill: string | null = null;
    if (directive !== null) {
      if (!directive.query) {
        this.skillNotice = true;
        return;
      }
      submitQuery = directive.query;
      requestedSkill = directive.skill;
    }
    // The message is offered before the box lets go of it: a listener that cannot take it now cancels
    // the event, and the words stay in the box and in storage for the reader to send again.
    const taken = this.dispatchEvent(new CustomEvent<ComposerSubmitDetail>('dl-composer-submit', {
      detail: {
        query: submitQuery,
        mode: this.#requestMode,
        requestedSkill,
        // Fast answers run no agent turns, so they never carry an effort.
        effort: this.mode === 'fast' ? null : this.#requestEffort,
        ...(delivery ? {delivery} : {}),
      },
      bubbles: true,
      composed: true,
      cancelable: true,
    }));
    if (!taken) return;
    this.draft = '';
    // The message left the box: what waits in storage must not hand it back on the next load.
    this.#flushDraft();
    this.skillNotice = false;
    this.skillMenuOpen = false;
    this.skillActive = -1;
    void this.updateComplete.then(() => this.#resize());
  }

  #applySkill(name: string): void {
    this.draft = committedSkillDirective(this.draft, name);
    // The box now says something else: storage follows it, not the letters typed before.
    this.#flushDraft();
    this.skillNotice = false;
    this.skillMenuOpen = false;
    this.skillActive = -1;
    void this.updateComplete.then(() => {
      const input = this.#input();
      if (input) {
        input.focus();
        input.setSelectionRange(input.value.length, input.value.length);
      }
    });
  }

  #menuItems(
    suggestions: readonly SkillSummary[] = this.#skillSuggestions(),
    commands: readonly WebCommand[] = commandSuggestions(this.draft),
  ): readonly SlashMenuItem[] {
    return [
      ...suggestions.map((skill) => ({kind: 'skill' as const, skill})),
      ...commands.map((command) => ({kind: 'command' as const, command})),
    ];
  }

  #menuItemId(item: SlashMenuItem): string {
    return item.kind === 'skill'
      ? skillOptionId(item.skill.name)
      : commandOptionId(item.command.name);
  }

  #acceptMenuItem(item: SlashMenuItem): void {
    if (item.kind === 'skill') this.#applySkill(item.skill.name);
    else this.#applyCommand(item.command);
  }

  #applyCommand(command: WebCommand): void {
    this.draft = `/${command.name} `;
    this.#flushDraft();
    this.skillNotice = false;
    this.skillMenuOpen = false;
    this.skillActive = -1;
    void this.updateComplete.then(() => {
      const input = this.#input();
      if (input) {
        input.focus();
        input.setSelectionRange(input.value.length, input.value.length);
      }
    });
  }

  #commandHint(): TemplateResult | typeof nothing {
    const invocation = findWebCommand(this.draft.trim());
    if (invocation === null || invocation.args === '') return nothing;
    return html`<div class="command-hint" role="status">${localizedText(invocation.command.inputHint)}</div>`;
  }

  #ghostText(): string {
    const items = this.#menuItems();
    if (items.length === 0) return '';
    const item = this.skillActive >= 0 ? items[this.skillActive] : items[0];
    return item.kind === 'skill' ? skillGhostSuffix(this.draft, item.skill.name) : '';
  }

  /** The row the reader's arrows are on, once they have moved onto one. */
  #scrollActiveItem(): void {
    void this.updateComplete.then(() => {
      this.querySelector<HTMLButtonElement>('.skill-menu-item.active, .command-menu-item.active')
        ?.scrollIntoView({block: 'nearest'});
    });
  }

  /** How many skills the menu offers. It is the text of one status that re-renders as the reader types, so
   *  it is read out when the menu opens and when the count changes, and not again for the same count. */
  #skillCountText(count: number): string {
    return count === 1
      ? msg('1 skill available', {id: 'chatComposer.skillCountOne'})
      : msg(str`${count} skills available`, {id: 'chatComposer.skillCountMany'});
  }

  #skillSuggestions(): readonly SkillSummary[] {
    const state = skillDirectiveState(this.draft);
    if (state === null || this.skills.length === 0) return [];
    if (state.kind === 'canonical' && state.prefix === '') return this.skills;
    return this.skills.filter((skill) => skill.name.startsWith(state.prefix));
  }

  #cancelIntent(): void {
    raise(this, 'dl-composer-cancel');
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
      {total: this.handles.attachments.size, images: this.handles.attachments.imageCount},
      policy,
    )) return;
    this.handles.attachments.add(file, kind);
  }

  /** Fast answers run no agent turns, so the control never appears there. */
  /** The reader's one switch while a Run is live and they are writing: hold the words for the next
   *  turn, or steer now. One quiet button states the current delivery and a press switches it, so an
   *  empty composer stays calm: before this, two chips sat in the row for the whole Run, before the
   *  reader had sent anything. The queue is the default because a second Run is never the answer. */
  #deliveryControl(visible: boolean): TemplateResult | typeof nothing {
    if (!this.live || !visible) return nothing;
    const current = this.#delivery === 'steer'
      ? msg('Steer', {id: 'chatComposer.steer'})
      : msg('Queue', {id: 'chatComposer.queue'});
    return html`
      <button type="button" class="composer-delivery"
              aria-label="${msg('Delivery', {id: 'chatComposer.deliveryAria'})}: ${current}"
              @click=${this.#toggleDelivery}>
        ${current}
      </button>
    `;
  }

  /** Switch the delivery a live send uses: queue by default, steer on request. */
  #toggleDelivery = (): void => {
    this.#delivery = this.#delivery === 'queue' ? 'steer' : 'queue';
    this.requestUpdate();
  };

  #effortControl(): TemplateResult | typeof nothing {
    const levels = offeredLevels(this.agentEffortOffer);
    if (levels.length === 0 || this.mode === 'fast') return nothing;
    const displayed = this.#displayedEffort(levels);
    return html`
      <div class="composer-effort">
        <button type="button" class="composer-effort-trigger" id="composer-effort"
                aria-haspopup="menu" aria-controls="composer-effort-menu"
                aria-expanded=${String(this.picker === 'effort')}
                aria-label=${displayed
                  ? msg(str`Agent effort: ${EFFORT_LABELS[displayed]}`, {id: `chatComposer.effortAria.${displayed}`})
                  : msg('Agent effort', {id: 'chatComposer.effortAria'})}
                @click=${this.#effortButton.toggle} @keydown=${this.#effortButton.keydown}>
          ${icon(displayed ? EFFORT_ICONS[displayed] : 'effort', {size: 'sm', className: 'composer-effort-icon'})}
          <span class="composer-effort-label">${displayed
            ? msg(EFFORT_LABELS[displayed], {id: `chatComposer.effort.${displayed}`})
            : msg('Effort', {id: 'chatComposer.effortLabel'})}</span>
        </button>
        <dl-menu class="composer-effort-menu dl-anchored dl-anchored--above dl-anchored--end"
             id="composer-effort-menu" role="menu"
             aria-label=${msg('Agent effort', {id: 'chatComposer.effortMenuAria'})}
             ?hidden=${this.picker !== 'effort'} @dl-menu-dismiss=${this.#effortButton.dismissed}>
          ${levels.map((level) => html`
            <button type="button" role="menuitemradio" data-effort=${level}
                    aria-checked=${String(displayed === level)} tabindex="-1"
                    @click=${() => this.#selectEffort(level)}>
              ${icon(EFFORT_ICONS[level], {size: 'sm', className: 'composer-effort-icon'})}
              <span>${msg(EFFORT_LABELS[level], {id: `chatComposer.effort.${level}`})}</span>
              ${this.agentEffortOffer.default === level
                ? html`<span class="composer-effort-default">${msg('Default', {id: 'chatComposer.effortDefault'})}</span>`
                : nothing}
            </button>
          `)}
        </dl-menu>
      </div>
    `;
  }

  /** The level this composer shows: the caller's choice, else the deployment's. */
  #displayedEffort(levels: readonly AgentEffort[]): AgentEffort | null {
    return this.effort && levels.includes(this.effort) ? this.effort : null;
  }

  #selectEffort(level: AgentEffort): void {
    this.#effortInitialized = true;
    this.#requestEffort = level;
    this.effort = level;
    this.picker = null;
    storeAgentEffort(level);
    this.focusInput();
  }

  /** The three answers a menu button's owner gives, the same for each of the two pickers. */
  #buttonFor(picker: Picker) {
    return {
      toggle: (event: Event): void => {
        event.stopPropagation();
        if (this.picker === picker) this.#closePickers();
        else this.#openPicker(picker, 'first');
      },
      keydown: (event: KeyboardEvent): void => {
        const focus = menuButtonFocus(event);
        if (!focus) return;
        event.preventDefault();
        this.#openPicker(picker, focus);
      },
      dismissed: (event: CustomEvent<MenuDismissDetail>): void => {
        if (this.picker === picker) this.picker = null;
        if (event.detail.restoreFocus) this.querySelector<HTMLButtonElement>(`.composer-${picker}-trigger`)?.focus();
      },
    };
  }

  /** Open one picker menu, closing every other popup, with focus where its button asked. */
  #openPicker(picker: Picker, focus: MenuFocus): void {
    this.picker = picker;
    this.skillMenuOpen = false;
    this.skillActive = -1;
    void this.updateComplete.then(() => {
      this.querySelector<DlMenu>(`#composer-${picker}-menu`)?.focusItem(focus);
    });
  }

  /** One composer popup at a time: a click elsewhere settles them all. */
  #closePickers(): void {
    this.picker = null;
    this.skillMenuOpen = false;
    this.skillActive = -1;
  }

  #selectMode(mode: AnswerMode): void {
    this.mode = mode;
    this.#requestMode = mode;
    this.picker = null;
    writeStored(STORAGE_KEY, mode);
    this.focusInput();
  }

  #closeMenus = (): void => {
    this.#closePickers();
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
      raise(this, 'dl-composer-workspace-drop', result);
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

/** Forget the unsent message, so a sign-out leaves no owner's words for the next one. */
export function forgetStoredDraft(): void {
  writeStored(DRAFT_STORAGE_KEY, null);
}

declare global {
  interface HTMLElementTagNameMap {
    'dl-chat-composer': DlChatComposer;
  }

  interface HTMLElementEventMap {
    'dl-composer-submit': CustomEvent<ComposerSubmitDetail>;
    'dl-composer-cancel': CustomEvent<void>;
    'dl-composer-workspace-drop': CustomEvent<ComposerWorkspaceDropDetail>;
  }
}
