// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Settings Dialog Feature: the native dialog, its navigation, and the one page beside it.
 *
 * The dialog owns what is common to every page: opening and closing, focus, which page shows, and
 * the title, description, and notices around it. Each page is an element of its own that owns its
 * data and reports a typed summary for the navigation row to show. Pages are mounted while the
 * dialog is open and hidden while another shows, so a page that polls keeps its status fresh;
 * closing tears them down. Profile Memory is the one page that stays in the dialog while it is
 * closed, because a live Memory change arrives with an Undo whenever Chat says so.
 *
 * On a phone the same markup is two levels: the section list, then one page with a Back button.
 */

import {msg, str, updateWhenLocaleChanges} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import type {MemoryOperationEvent} from '../api/memory.ts';
import {type IconName, icon, rovingFocusKeydown} from '../design-system/index.ts';
import {PHONE_DIALOG_MEDIA} from '../lib/breakpoints.ts';
import {LightElement, MediaController} from '../lib/lit-host.ts';
import {type AppHandles, productionHandles} from '../stores/app-handles.ts';
import styles from '../styles/settings-dialog.module.css';
import {publishModalState, showOwnedModal} from './modal.ts';
import './settings-agent-accounts.ts';
import './settings-connections.ts';
import './settings-conversations.ts';
import './settings-language.ts';
import './settings-memory.ts';
import type {SettingsSection, SettingsSummary} from './settings-summary.ts';
import './toast.ts';
import {requestToast} from './toast-request.ts';
import type {ToastRequestDetail} from './toast.ts';

type Group = 'agent' | 'data' | 'general';

interface SectionDefinition {
  section: SettingsSection;
  group: Group;
  icon: IconName;
  label: () => string;
  description: () => string;
}

const SECTIONS: readonly SectionDefinition[] = [
  {
    section: 'connections',
    group: 'agent',
    icon: 'connections',
    label: () => msg('Connections', {id: 'settings.connections'}),
    description: () => msg(
      'External MCP servers that Research runs can call. Turning the first one on asks you to confirm once.',
      {id: 'settings.connectionsDescription'},
    ),
  },
  {
    section: 'agent-accounts',
    group: 'agent',
    icon: 'agent-accounts',
    label: () => msg('Agent Accounts', {id: 'settings.agentAccounts'}),
    description: () => msg(
      'Accounts the agent registered on websites. DlightRAG generates and seals each password; nobody can view it.',
      {id: 'settings.agentAccountsDescription'},
    ),
  },
  {
    section: 'memory',
    group: 'agent',
    icon: 'profile-memory',
    label: () => msg('Profile Memory', {id: 'settings.profileMemory'}),
    description: () => msg(
      'Preferences and facts the agent remembers about you across conversations.',
      {id: 'settings.memoryDescription'},
    ),
  },
  {
    section: 'conversations',
    group: 'data',
    icon: 'conversations',
    label: () => msg('Conversation Sessions', {id: 'settings.conversationSessions'}),
    description: () => msg('Conversations retain 365 days', {id: 'settings.retentionNote'}),
  },
  {
    section: 'language',
    group: 'general',
    icon: 'language',
    label: () => msg('Language', {id: 'settings.language'}),
    description: () => msg('The language of the interface.', {id: 'settings.languageDescription'}),
  },
];

const GROUPS: readonly {group: Group; label: () => string}[] = [
  {group: 'agent', label: () => msg('Agent', {id: 'settings.groupAgent'})},
  {group: 'data', label: () => msg('Data', {id: 'settings.groupData'})},
  {group: 'general', label: () => msg('General', {id: 'settings.groupGeneral'})},
];

/** A short status for a desktop row, and the full line a phone's list shows under the name. */
function statusOf(summary: SettingsSummary | undefined): {short: string; detail: string} {
  const none = {short: '', detail: ''};
  if (!summary) return none;
  switch (summary.section) {
    case 'connections':
      return summary.total === 0
        ? {short: '', detail: msg('MCP · none yet', {id: 'settings.status.connectionsNone'})}
        : {
          short: `${summary.enabled}/${summary.total}`,
          detail: msg(str`MCP · ${summary.enabled} of ${summary.total} enabled`, {id: 'settings.status.connections'}),
        };
    case 'agent-accounts':
      return {
        short: String(summary.count),
        detail: summary.count === 0
          ? msg('None yet', {id: 'settings.status.accountsNone'})
          : summary.count === 1
            ? msg('1 website', {id: 'settings.status.accountsOne'})
            : msg(str`${summary.count} websites`, {id: 'settings.status.accounts'}),
      };
    case 'memory':
      if (summary.enabled === null) return none;
      if (!summary.enabled) return {short: '', detail: msg('Off', {id: 'settings.status.memoryOff'})};
      return summary.count === null
        ? {short: '', detail: msg('On', {id: 'settings.status.memoryOn'})}
        : {short: String(summary.count), detail: msg(str`On · ${summary.count} stored`, {id: 'settings.status.memoryStored'})};
    case 'conversations':
      return {
        short: String(summary.count),
        detail: summary.count === 1
          ? msg('1 conversation · kept 365 days', {id: 'settings.status.conversationsOne'})
          : msg(str`${summary.count} conversations · kept 365 days`, {id: 'settings.status.conversations'}),
      };
    case 'language':
      return {
        short: '',
        detail: summary.preference === 'auto'
          ? msg('Automatic', {id: 'settings.language.automatic'})
          : summary.preference === 'en'
            ? msg('English', {id: 'settings.language.english'})
            : '中文',
      };
  }
}

/** Owns the dialog's lifecycle, its navigation, and native Dialog semantics. */
export class DlSettingsDialog extends LightElement {
  static properties = {
    handles: {attribute: false},
    deleteAllConversations: {attribute: false},
    mounted: {state: true},
    page: {state: true},
    level: {state: true},
    summaries: {state: true},
  };

  declare handles: AppHandles;
  declare deleteAllConversations: (returnFocus?: HTMLElement | null) => Promise<boolean>;
  /** The pages are in the dialog: it is open, or it is about to be. */
  declare mounted: boolean;
  declare page: SettingsSection;
  /** What a phone shows: the section list, or the page the reader chose. */
  declare level: 'list' | 'page';
  declare summaries: Partial<Record<SettingsSection, SettingsSummary>>;

  #events: AbortController | null = null;
  #returnFocus: HTMLElement | null = null;
  readonly #phone = new MediaController(this, PHONE_DIALOG_MEDIA);

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.handles = productionHandles();
    this.deleteAllConversations = async () => false;
    this.mounted = false;
    this.page = 'connections';
    this.level = 'list';
    this.summaries = {};
  }

  override connectedCallback(): void {
    super.connectedCallback();
    this.#events = new AbortController();
  }

  override disconnectedCallback(): void {
    this.#events?.abort();
    this.#events = null;
    document.body.classList.remove('settings-open');
    super.disconnectedCallback();
  }

  protected override updated(changed: PropertyValues<this>): void {
    // A page opens at its top, whatever the one before it was scrolled to.
    if (changed.has('page')) this.querySelector<HTMLElement>('[data-page-body]')?.scrollTo({top: 0});
  }

  /** Open Settings on a page. A page that is named opens on a phone too; otherwise a phone opens on the list. */
  async open(returnFocus?: HTMLElement | null, page?: SettingsSection): Promise<void> {
    const signal = this.#events?.signal;
    if (!signal || signal.aborted) return;
    this.#returnFocus = returnFocus ?? (
      document.activeElement instanceof HTMLElement ? document.activeElement : null
    );
    if (this.mounted && this.#dialog()?.open === false) {
      // The last session closed and its close event is still on its way: end it here, so this
      // one reads on pages of its own and not on the ones the last left behind.
      this.mounted = false;
      await this.updateComplete;
      if (signal.aborted) return;
    }
    this.page = page ?? 'connections';
    this.level = page === undefined ? 'list' : 'page';
    this.mounted = true;
    await this.updateComplete;
    if (signal.aborted) return;
    const dialog = this.#dialog();
    if (!dialog) return;
    if (!dialog.open) {
      dialog.returnValue = '';
      showOwnedModal(this, dialog);
      document.body.classList.add('settings-open');
    }
    this.#focusEntry();
  }

  /** Hand one live Profile Memory fact from Chat to the page that turns it into a receipt. */
  handleMemoryOperation(event: MemoryOperationEvent): void {
    this.querySelector('dl-settings-memory')?.handleOperation(event);
  }

  protected override render(): TemplateResult {
    const phone = this.#phone.matches;
    const definition = SECTIONS.find((item) => item.section === this.page) ?? SECTIONS[0]!;
    const isPage = (section: SettingsSection): boolean => this.page === section;
    return html`
      <dialog id="settings-dialog" class="settings-dialog" aria-labelledby="settings-title"
              @click=${this.#scrimClick} @close=${this.#closed}
              @dl-settings-summary=${this.#summarized} @dl-toast-request=${this.#toastRequested}>
        <div class=${styles.frame} data-level=${this.level}>
          <dl-icon-button class=${styles.close} name="close" size="sm"
            aria-label=${msg('Close settings', {id: 'settings.close'})}
            @click=${this.#close}></dl-icon-button>
          <nav class=${styles.nav} aria-label=${msg('Settings', {id: 'settings.title'})}
               @keydown=${this.#navKeydown}>
            <h2 id="settings-title" class=${styles.navTitle}>${msg('Settings', {id: 'settings.title'})}</h2>
            ${GROUPS.map(({group, label}) => html`
              <div class=${styles.group} role="group" aria-labelledby="settings-group-${group}">
                <span id="settings-group-${group}" class=${styles.groupLabel}>${label()}</span>
                <div class=${styles.groupItems}>
                  ${SECTIONS.filter((item) => item.group === group).map((item) => this.#navItem(item, phone))}
                </div>
              </div>`)}
          </nav>
          <section class=${styles.pane} role="region" aria-labelledby="settings-page-title">
            <header class=${styles.paneHeader}>
              <dl-icon-button class=${styles.back} name="previous" size="md"
                aria-label=${msg('Back', {id: 'settings.back'})}
                @click=${this.#back}></dl-icon-button>
              <h3 id="settings-page-title" class=${styles.paneTitle} tabindex="-1">${definition.label()}</h3>
            </header>
            <div class=${styles.paneBody} data-page-body>
              <p class=${styles.description}>${definition.description()}</p>
              ${this.mounted ? html`
                <dl-settings-connections ?hidden=${!isPage('connections')}></dl-settings-connections>
                <dl-settings-agent-accounts ?hidden=${!isPage('agent-accounts')}></dl-settings-agent-accounts>` : nothing}
              <dl-settings-memory .active=${this.mounted} .current=${isPage('memory')}
                ?hidden=${!isPage('memory')}></dl-settings-memory>
              ${this.mounted ? html`
                <dl-settings-conversations .handles=${this.handles} .deleteAll=${this.#deleteAll}
                  ?hidden=${!isPage('conversations')}></dl-settings-conversations>
                <dl-settings-language ?hidden=${!isPage('language')}></dl-settings-language>` : nothing}
            </div>
            ${this.mounted ? html`
              <dl-toast-region class=${styles.notice} role="status" aria-live="polite"></dl-toast-region>` : nothing}
          </section>
        </div>
      </dialog>
    `;
  }

  /** One row of the navigation: a desktop shows its short status, a phone's list its full line. */
  #navItem(item: SectionDefinition, phone: boolean): TemplateResult {
    const status = statusOf(this.summaries[item.section]);
    const label = `settings-nav-${item.section}-label`;
    const detail = `settings-nav-${item.section}-detail`;
    // Only a dialog that is showing has a current page, and a phone's list shows no page at all.
    const current = this.mounted && this.page === item.section && (!phone || this.level === 'page');
    return html`
      <button class="dl-nav-item" type="button" data-section=${item.section}
        aria-current=${current ? 'page' : nothing}
        aria-labelledby=${label} aria-describedby=${status.detail ? detail : nothing}
        @click=${() => { void this.#select(item.section); }}>
        <span class="dl-nav-item-icon">${icon(item.icon, {size: 'sm'})}</span>
        <span class="dl-nav-item-text">
          <span id=${label} class="dl-nav-item-label">${item.label()}</span>
          <span id=${detail} class="dl-nav-item-detail ${styles.statusDetail}">${status.detail}</span>
        </span>
        <span class="dl-nav-item-status ${styles.statusShort}" aria-hidden="true">${status.short}</span>
        <span class=${styles.chevron} aria-hidden="true">${icon('disclosure', {size: 'sm'})}</span>
      </button>`;
  }

  #dialog(): HTMLDialogElement | null {
    return this.querySelector<HTMLDialogElement>('#settings-dialog');
  }

  #pageTitle(): HTMLElement | null {
    return this.querySelector<HTMLElement>('#settings-page-title');
  }

  /** Where focus starts: the page the reader is on, or the top of the list on a phone. */
  #focusEntry(): void {
    if (!this.#phone.matches) {
      this.querySelector<HTMLElement>('.dl-nav-item[aria-current="page"]')?.focus();
    } else if (this.level === 'list') {
      this.querySelector<HTMLElement>('.dl-nav-item')?.focus();
    } else {
      this.#pageTitle()?.focus();
    }
  }

  async #select(section: SettingsSection): Promise<void> {
    this.page = section;
    this.level = 'page';
    if (!this.#phone.matches) return;
    // The list is gone from a phone's screen: the page's title is where the reader now is.
    await this.updateComplete;
    this.#pageTitle()?.focus();
  }

  #back = async (): Promise<void> => {
    const leaving = this.page;
    this.level = 'list';
    await this.updateComplete;
    this.querySelector<HTMLElement>(`.dl-nav-item[data-section="${leaving}"]`)?.focus();
  };

  #close = (): void => {
    this.#dialog()?.close();
  };

  #navKeydown = (event: KeyboardEvent): void => {
    const items = [...this.querySelectorAll<HTMLElement>('.dl-nav-item')];
    if (items.includes(event.target as HTMLElement)) rovingFocusKeydown(event, items);
  };

  #scrimClick = (event: MouseEvent): void => {
    const dialog = this.#dialog();
    if (dialog && event.target === dialog) dialog.close();
  };

  #summarized = (event: CustomEvent<SettingsSummary>): void => {
    event.stopPropagation();
    this.summaries = {...this.summaries, [event.detail.section]: event.detail};
  };

  /** While Settings is open its own region shows a page's notice, above the scrim; otherwise the shell does. */
  #toastRequested = (event: CustomEvent<ToastRequestDetail>): void => {
    const toast = this.#dialog()?.open ? this.querySelector('dl-toast-region') : null;
    if (!toast) return;
    event.stopPropagation();
    if (event.detail.action) toast.showAction(event.detail.message, event.detail.action);
    else toast.show(event.detail.message, event.detail.duration);
  };

  #closed = (): void => {
    // The event is queued when the dialog closes, so it can arrive after Settings opened again;
    // it then reports the session before this one, which has nothing left to tear down.
    if (this.#dialog()?.open) return;
    // A notice that still offers Undo outlives the dialog: the shell's region takes it over.
    const toast = this.querySelector('dl-toast-region');
    const notice = toast?.request;
    if (notice?.action && !toast?.pending) requestToast(this, {message: notice.message, action: notice.action});
    this.mounted = false;
    this.summaries = {};
    publishModalState(this);
    document.body.classList.remove('settings-open');
    const returnFocus = this.#returnFocus;
    this.#returnFocus = null;
    if (returnFocus?.isConnected && !returnFocus.inert) returnFocus.focus();
  };

  /** Deleting every conversation is the sidebar's command; Settings has nothing left to show once it ran. */
  #deleteAll = async (returnFocus?: HTMLElement | null): Promise<boolean> => {
    const deleted = await this.deleteAllConversations(returnFocus);
    if (deleted) this.#dialog()?.close();
    return deleted;
  };
}

customElements.define('dl-settings-dialog', DlSettingsDialog);

declare global {
  interface HTMLElementTagNameMap {
    'dl-settings-dialog': DlSettingsDialog;
  }
}
