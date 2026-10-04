// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Settings → Agent Accounts: the owner's sign-up switch and the accounts the agent registered.
 *
 * DlightRAG mints and seals every password, so this page has nothing secret to show or take: it
 * lists where the agent has an account, how it signs in there, and when, and it can remove one.
 * The view comes from one route and every command answers the fresh view, so the page holds no
 * state of its own beyond what is in flight.
 */

import {msg, str, updateWhenLocaleChanges} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import {
  type AgentAccount,
  type AgentAccountsView,
  getAgentAccounts,
  removeAgentAccount,
  setAgentAccountRegistration,
} from '../api/agent-accounts.ts';
import {ApiError} from '../api/wire.ts';
import {icon} from '../design-system/index.ts';
import {getLocale} from '../i18n/locale.ts';
import {capitalized, recentDay, shortDate} from '../lib/date-format.ts';
import {LightElement, NarrowController} from '../lib/lit-host.ts';
import shared from '../styles/settings-page.module.css';
import styles from '../styles/settings-agent-accounts.module.css';
import {modalResult} from './modal.ts';
import {switchCard} from './settings-parts.ts';
import {reportSettingsSummary} from './settings-summary.ts';
import {requestToast} from './toast-request.ts';

/** The narrowest page, in rem, that has room for the table's five columns. */
const TABLE_REM = 36;

/** The one sentence under the sign-up switch: the deployment's answer first, then the owner's. */
function registrationCaption(view: AgentAccountsView): string {
  if (!view.available) {
    return msg('This deployment has not enabled Agent Accounts. Stored accounts can still be removed here.', {
      id: 'agentAccounts.unavailable',
    });
  }
  if (!view.registration.allowed) {
    return msg('This deployment does not allow sign-ups, so the agent only signs in with the accounts below.', {
      id: 'agentAccounts.notAllowed',
    });
  }
  return view.registration.enabled
    ? msg('The agent may register on a website when it needs to', {id: 'agentAccounts.signUpsOn'})
    : msg('Off: the agent only signs in with the accounts below', {id: 'agentAccounts.signUpsOff'});
}

/** How the agent signs in: the email first, and the username under it when there are both. */
function identityOf(account: AgentAccount): {primary: string | null; secondary: string | null} {
  if (account.email && account.username) return {primary: account.email, secondary: account.username};
  return {primary: account.email ?? account.username, secondary: null};
}

export class DlSettingsAgentAccounts extends LightElement {
  static properties = {
    view: {state: true},
    error: {state: true},
    pending: {state: true},
    removing: {state: true},
  };

  declare view: AgentAccountsView | null;
  /** The first read failed; with a view in hand a failed reload only says so in a toast. */
  declare error: boolean;
  /** A command is in flight, so no second one starts. */
  declare pending: boolean;
  /** The website the remove dialog is asking about, so its copy can name it. */
  declare removing: string | null;

  #events: AbortController | null = null;
  /** A table needs room for its columns: below this width each account is three lines instead. */
  readonly #narrow = new NarrowController(this, TABLE_REM);

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.view = null;
    this.error = false;
    this.pending = false;
    this.removing = null;
  }

  override connectedCallback(): void {
    super.connectedCallback();
    this.#events = new AbortController();
    void this.#load();
  }

  override disconnectedCallback(): void {
    this.#events?.abort();
    this.#events = null;
    super.disconnectedCallback();
  }

  protected override updated(changed: PropertyValues<this>): void {
    if (changed.has('view') && this.view) {
      reportSettingsSummary(this, {section: 'agent-accounts', count: this.view.accounts.length});
    }
  }

  async #load(): Promise<void> {
    const signal = this.#events?.signal;
    if (!signal || signal.aborted) return;
    try {
      const view = await getAgentAccounts(signal);
      if (signal.aborted) return;
      this.view = view;
      this.error = false;
    } catch {
      if (signal.aborted) return;
      if (this.view) {
        requestToast(this, {message: msg('Could not load agent accounts.', {id: 'agentAccounts.loadFailed'}), duration: 3000});
      } else {
        this.error = true;
      }
    }
  }

  #retry = (): void => {
    this.error = false;
    void this.#load();
  };

  #toggleRegistration = async (event: Event): Promise<void> => {
    const signal = this.#events?.signal;
    const view = this.view;
    const toggle = event.currentTarget as HTMLElement;
    if (!signal || signal.aborted || !view || this.pending) return;
    const focused = document.activeElement === toggle;
    this.pending = true;
    try {
      const fresh = await setAgentAccountRegistration(!view.registration.enabled, signal);
      if (!signal.aborted) this.view = fresh;
    } catch {
      if (!signal.aborted) {
        requestToast(this, {message: msg('Could not save the sign-up setting.', {id: 'agentAccounts.saveFailed'}), duration: 3000});
      }
    } finally {
      if (!signal.aborted) {
        this.pending = false;
        await this.updateComplete;
        // A switch that was disabled for the request drops focus in some engines; give it back.
        if (focused) toggle.focus();
      }
    }
  };

  async #remove(account: AgentAccount, trigger: HTMLElement): Promise<void> {
    const signal = this.#events?.signal;
    const dialog = this.querySelector<HTMLDialogElement>('#agent-accounts-remove');
    if (!signal || signal.aborted || !dialog || this.pending) return;
    this.removing = account.site;
    await this.updateComplete;
    const outcome = await modalResult(this, dialog, () => trigger.focus(), signal);
    this.removing = null;
    if (outcome !== 'remove' || signal.aborted) return;
    const index = this.view?.accounts.findIndex((item) => item.site === account.site) ?? 0;
    this.pending = true;
    try {
      this.view = await removeAgentAccount(account.site, signal);
    } catch (error) {
      if (signal.aborted) return;
      // The account is already gone, here or in another tab: the fresh view says so.
      if (error instanceof ApiError && error.status === 404) await this.#load();
      else requestToast(this, {message: msg('Could not remove the account.', {id: 'agentAccounts.removeFailed'}), duration: 3000});
    } finally {
      if (!signal.aborted) {
        this.pending = false;
        await this.updateComplete;
        // The row the reader was on is gone: land on the one that took its place, else on the switch,
        // else (it is off for the deployment) on the note that there is nothing left.
        const removals = this.querySelectorAll<HTMLElement>('[data-remove]');
        (removals[Math.min(index, removals.length - 1)]
          ?? this.querySelector<HTMLElement>('#agent-accounts-registration:not(:disabled)')
          ?? this.querySelector<HTMLElement>('#agent-accounts-empty-title'))?.focus();
      }
    }
  }

  /** Removing one account, named for the website it belongs to. */
  #removeButton(account: AgentAccount): TemplateResult {
    return html`<dl-icon-button name="remove" size="sm" class=${shared.iconAction}
      data-remove=${account.site} ?disabled=${this.pending}
      aria-label=${msg(str`Remove the account for ${account.site}`, {id: 'agentAccounts.removeLabel'})}
      @click=${(event: Event) => { void this.#remove(account, event.currentTarget as HTMLElement); }}
    ></dl-icon-button>`;
  }

  #site(account: AgentAccount): TemplateResult {
    return html`<span class=${styles.site}>
      <span class=${styles.tile} aria-hidden="true">${account.site.charAt(0).toLocaleUpperCase()}</span>
      <span class=${styles.siteName} title=${account.site}>${account.site}</span>
    </span>`;
  }

  /** The table row: the columns say what the phone's sentence says. */
  #tableRow(account: AgentAccount, now: Date, locale: string): TemplateResult {
    const {primary, secondary} = identityOf(account);
    const recent = account.lastUsedAt ? recentDay(account.lastUsedAt, now, locale) : null;
    return html`
      <tr data-site=${account.site}>
        <th scope="row" class=${styles.cell}>${this.#site(account)}</th>
        <td class=${styles.cell}>
          <span class=${styles.identity}>
            ${primary === null
              ? html`<span aria-hidden="true">—</span><span class="dl-sr-only">${
                msg('Not recorded', {id: 'agentAccounts.noIdentity'})}</span>`
              : html`<span class=${styles.primary} title=${primary}>${primary}</span>`}
            ${secondary === null ? nothing : html`<span class=${styles.secondary} title=${secondary}>${secondary}</span>`}
          </span>
        </td>
        <td class="${styles.cell} ${styles.date}">${shortDate(account.createdAt, now, locale)}</td>
        <td class="${styles.cell} ${styles.date}">${account.lastUsedAt === null
          ? html`<span aria-hidden="true">—</span><span class="dl-sr-only">${
            msg('Never', {id: 'agentAccounts.never'})}</span>`
          : recent !== null
            ? capitalized(recent, locale)
            : shortDate(account.lastUsedAt, now, locale)}</td>
        <td class="${styles.cell} ${styles.action}">${this.#removeButton(account)}</td>
      </tr>`;
  }

  #table(accounts: readonly AgentAccount[], now: Date, locale: string): TemplateResult {
    return html`
      <table class=${styles.table}>
        <thead>
          <tr>
            <th scope="col" class="${styles.cell} ${styles.websiteColumn}">${msg('Website', {id: 'agentAccounts.website'})}</th>
            <th scope="col" class=${styles.cell}>${msg('Sign-in', {id: 'agentAccounts.signIn'})}</th>
            <th scope="col" class="${styles.cell} ${styles.date}">${msg('Registered', {id: 'agentAccounts.registered'})}</th>
            <th scope="col" class="${styles.cell} ${styles.date}">${msg('Last sign-in', {id: 'agentAccounts.lastSignIn'})}</th>
            <th scope="col" class="${styles.cell} ${styles.action}"><span class="dl-sr-only">${
              msg('Remove', {id: 'agentAccounts.remove'})}</span></th>
          </tr>
        </thead>
        <tbody>
          ${repeat(accounts, (account) => account.site, (account) => this.#tableRow(account, now, locale))}
        </tbody>
      </table>`;
  }

  /** Where a table has no room, each account is three lines: the website, how it signs in, and when it last did. */
  #list(accounts: readonly AgentAccount[], now: Date, locale: string): TemplateResult {
    return html`
      <ul class="${shared.list} ${shared.divided}">
        ${repeat(accounts, (account) => account.site, (account) => {
          const {primary} = identityOf(account);
          const when = account.lastUsedAt
            ? recentDay(account.lastUsedAt, now, locale) ?? shortDate(account.lastUsedAt, now, locale)
            : null;
          return html`
            <li class=${styles.item} data-site=${account.site}>
              <span class=${styles.tile} aria-hidden="true">${account.site.charAt(0).toLocaleUpperCase()}</span>
              <span class=${styles.itemText}>
                <span class=${styles.siteName} title=${account.site}>${account.site}</span>
                ${primary === null ? nothing : html`<span class=${styles.secondary} title=${primary}>${primary}</span>`}
                <span class=${styles.meta}>${when === null
                  ? msg(str`Registered ${shortDate(account.createdAt, now, locale)}, not signed in since`, {
                    id: 'agentAccounts.registeredNever',
                  })
                  : msg(str`Signed in ${when}`, {id: 'agentAccounts.signedIn'})}</span>
              </span>
              ${this.#removeButton(account)}
            </li>`;
        })}
      </ul>`;
  }

  #empty(): TemplateResult {
    return html`
      <div class="${shared.card} ${styles.empty}">
        <span class=${styles.emptyIcon}>${icon('agent-accounts', {size: 'md'})}</span>
        <h4 id="agent-accounts-empty-title" class=${styles.emptyTitle} tabindex="-1">${
          msg('No accounts yet', {id: 'agentAccounts.emptyTitle'})}</h4>
        <span class=${styles.emptyBody}>${msg(
          'When Research meets a website that needs a free account, the agent signs up under its own identity and the account appears here.',
          {id: 'agentAccounts.emptyBody'},
        )}</span>
      </div>`;
  }

  #page(view: AgentAccountsView): TemplateResult {
    const now = new Date();
    const locale = getLocale();
    const usable = view.available && view.registration.allowed;
    return html`
      <div class=${shared.stack}>
        ${switchCard({
          id: 'agent-accounts-registration',
          label: msg('Allow new sign-ups', {id: 'agentAccounts.allowSignUps'}),
          caption: registrationCaption(view),
          checked: usable && view.registration.enabled,
          disabled: !usable || this.pending,
          muted: !usable,
          onToggle: this.#toggleRegistration,
        })}
        ${view.accounts.length === 0 ? this.#empty() : html`
          <div class=${shared.card}>
            ${this.#narrow.narrow
              ? this.#list(view.accounts, now, locale)
              : this.#table(view.accounts, now, locale)}
          </div>`}
      </div>`;
  }

  protected override render(): TemplateResult {
    return html`
      ${this.view ? this.#page(this.view) : this.error ? html`
        <div class=${styles.failed}>
          <p class=${shared.note} role="alert">${msg('Could not load agent accounts.', {id: 'agentAccounts.loadFailed'})}</p>
          <button type="button" class="dl-btn" @click=${this.#retry}>${msg('Retry', {id: 'agentAccounts.retry'})}</button>
        </div>` : html`<p class=${shared.hint} role="status">${
          msg('Loading agent accounts…', {id: 'agentAccounts.loading'})}</p>`}
      <dialog id="agent-accounts-remove" class="confirm-dialog" aria-labelledby="agent-accounts-remove-title">
        <form method="dialog" novalidate>
          <h2 id="agent-accounts-remove-title">${this.removing
            ? msg(str`Remove the account for ${this.removing}?`, {id: 'agentAccounts.removeTitle'})
            : msg('Remove this account?', {id: 'agentAccounts.removeTitleEmpty'})}</h2>
          <p>${msg(
            'DlightRAG deletes the saved sign-in email and sealed password, and the agent can no longer sign in there. The account itself stays on the website.',
            {id: 'agentAccounts.removeBody'},
          )}</p>
          <div class="dl-dialog-actions">
            <button type="submit" value="cancel">${msg('Cancel', {id: 'agentAccounts.cancel'})}</button>
            <button type="submit" value="remove" class="dl-dialog-danger">${
              msg('Remove account', {id: 'agentAccounts.removeConfirm'})}</button>
          </div>
        </form>
      </dialog>`;
  }
}

customElements.define('dl-settings-agent-accounts', DlSettingsAgentAccounts);

declare global {
  interface HTMLElementTagNameMap {
    'dl-settings-agent-accounts': DlSettingsAgentAccounts;
  }
}
