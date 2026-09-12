// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Settings → Connections → MCP owns forms, consent, refresh and cancellation. */
import {msg, updateWhenLocaleChanges} from '@lit/localize';
import {html, nothing, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import {beginConnectionAuthorization, changeConnection, getConnections, type Connection, type ConnectionsView} from '../api/connections.ts';
import {LightElement} from '../lib/lit-host.ts';
import styles from '../styles/settings-connections.module.css';

export class DlSettingsConnections extends LightElement {
  static properties = {view: {state: true}, pending: {state: true}, error: {state: true}, consent: {state: true}, authorizationUrl: {state: true}};
  declare view: ConnectionsView | null;
  declare pending: boolean;
  declare error: boolean;
  declare consent: string | null;
  declare authorizationUrl: string | null;
  #events: AbortController | null = null;
  #timer: ReturnType<typeof setTimeout> | null = null;
  #generation = 0;

  constructor() {
    super(); updateWhenLocaleChanges(this);
    this.authorizationUrl = null; this.view = null; this.pending = false; this.error = false; this.consent = null;
  }
  override connectedCallback(): void {
    super.connectedCallback(); this.#events = new AbortController(); void this.#load();
  }
  override disconnectedCallback(): void {
    this.authorizationUrl = null;
    this.#events?.abort(); this.#events = null; this.#generation++;
    if (this.#timer) clearTimeout(this.#timer);
    this.querySelectorAll<HTMLInputElement>('input[type=password]').forEach((input) => {input.value = '';});
    super.disconnectedCallback();
  }
  async #load(): Promise<void> {
    const signal = this.#events?.signal;
    if (!signal || signal.aborted || this.pending) return;
    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = null;
    const generation = ++this.#generation;
    try {
      const view = await getConnections(signal);
      if (!signal.aborted && generation === this.#generation) {this.view = view; this.error = false;}
    } catch {if (!signal.aborted && generation === this.#generation) this.error = true;}
    if (!signal.aborted) this.#timer = setTimeout(() => {if (!this.pending) void this.#load();}, 5000);
  }
  async #change(command: Parameters<typeof changeConnection>[1], returnFocus?: HTMLElement): Promise<void> {
    const signal = this.#events?.signal;
    if (!this.view || this.pending || !signal || signal.aborted) return;
    this.#generation++; this.pending = true; this.error = false;
    if (this.#timer) clearTimeout(this.#timer);
    try {
      const view = await changeConnection(this.view.revision, command, signal);
      if (!signal.aborted) {this.view = view; this.consent = null;}
    } catch {if (!signal.aborted) this.error = true;}
    finally {
      if (!signal.aborted) {
        this.pending = false; await this.updateComplete;
        if (returnFocus?.isConnected) returnFocus.focus();
        else this.querySelector<HTMLElement>('[data-connections-root]')?.focus();
        this.#timer = setTimeout(() => {void this.#load();}, 5000);
      }
    }
  }
  #create = (event: Event): void => {
    const label = this.querySelector<HTMLInputElement>('[data-new-label]');
    const endpoint = this.querySelector<HTMLInputElement>('[data-new-endpoint]');
    if (!label?.reportValidity() || !endpoint?.reportValidity()) return;
    void this.#change({kind: 'create', label: label.value, endpoint: endpoint.value}, event.currentTarget as HTMLElement);
  };
  #bearer(item: Connection, event: Event): void {
    const button = event.currentTarget as HTMLElement;
    const input = button.closest('article')?.querySelector<HTMLInputElement>('input[type=password]');
    if (!input?.value) return;
    const bearer = input.value; input.value = '';
    const endpoint = button.closest('article')?.querySelector<HTMLInputElement>('[data-endpoint]')?.value;
    void this.#change({kind: 'bearer', connectionId: item.connectionId, bearer, ...(endpoint && endpoint !== item.endpoint ? {endpoint} : {})}, button);
  }
  async #oauth(item: Connection, event: Event): Promise<void> {
    const input = (event.currentTarget as HTMLElement).closest('article')?.querySelector<HTMLInputElement>('[data-endpoint]');
    const signal = this.#events?.signal;
    if (!this.view || this.pending || !signal || signal.aborted || !input?.reportValidity()) return;
    this.pending = true; this.error = false; this.authorizationUrl = null; this.#generation++;
    if (this.#timer) clearTimeout(this.#timer);
    try {
      const url = await beginConnectionAuthorization(this.view.revision, item.connectionId, input.value, signal);
      if (!signal.aborted) this.authorizationUrl = url;
    } catch {if (!signal.aborted) this.error = true;}
    finally {if (!signal.aborted) {this.pending = false; this.#timer = setTimeout(() => {void this.#load();}, 5000);}}
  }
  #edit(item: Connection, event: Event): void {
    const button = event.currentTarget as HTMLElement;
    const article = button.closest('article');
    const label = article?.querySelector<HTMLInputElement>('[data-label]');
    const endpoint = article?.querySelector<HTMLInputElement>('[data-endpoint]');
    if (!label?.reportValidity() || !endpoint?.reportValidity()) return;
    void this.#change({kind: 'edit', connectionId: item.connectionId, label: label.value, endpoint: endpoint.value}, button);
  }
  #item(item: Connection): TemplateResult {
    const command = (kind: 'disable' | 'delete' | 'probe' | 'revoke', event: Event) => {
      void this.#change({kind, connectionId: item.connectionId}, event.currentTarget as HTMLElement);
    };
    const status = {
      ready: msg('Ready', {id: 'connections.ready'}),
      degraded: msg('Degraded', {id: 'connections.degraded'}),
      'needs-auth': msg('Needs authorization', {id: 'connections.needsAuth'}),
      refreshing: msg('Refreshing', {id: 'connections.refreshing'}),
      revoked: msg('Revoked', {id: 'connections.revoked'}),
      disabled: msg('Disabled', {id: 'connections.disabled'}),
    }[item.status];
    const authentication = item.authentication === 'none' ? msg('Unauthenticated', {id: 'connections.unauthenticated'}) : item.authentication === 'bearer' ? msg('Personal bearer', {id: 'connections.personalBearer'}) : 'OAuth';
    return html`<article class=${styles.card} aria-label=${item.label}>
      <h4>${item.label}</h4>
      ${item.authorizationStatus === 'pending' ? html`<p role="status">${msg('Authorization pending', {id: 'connections.oauthPending'})}</p>` : nothing}
      ${item.authorizationStatus === 'failed' ? html`<p role="alert">${msg('Authorization failed or expired. Restart authorization.', {id: 'connections.oauthFailed'})}</p>` : nothing}
      ${item.authorizationStatus === 'succeeded' ? html`<p role="status">${msg('Authorization published', {id: 'connections.oauthSucceeded'})}</p>` : nothing}
      <p>${item.enabled ? msg('Enabled', {id: 'connections.enabled'}) : msg('Disabled', {id: 'connections.disabled'})} · ${status} · ${authentication}</p>
      <label>${msg('Label', {id: 'connections.label'})}<input data-label required maxlength="100" .value=${item.label}></label>
      <label>${msg('Endpoint', {id: 'connections.endpoint'})}<input data-endpoint type="url" required maxlength="2048" .value=${item.endpoint}></label>
      ${item.authentication !== 'none' ? html`<p>${msg('Changing an authenticated endpoint requires a new bearer or explicit OAuth authorization below. The current endpoint and grant remain live until candidate authorization and discovery succeed.', {id: 'connections.endpointAuthorization'})}</p>` : nothing}
      <button class="dl-btn" type="button" ?disabled=${this.pending} @click=${(event: Event) => this.#edit(item, event)}>${msg('Save', {id: 'connections.save'})}</button>
      <details><summary>${msg('Read-only tool catalogue', {id: 'connections.catalogue'})} (${item.tools.length})</summary>
        <p>${msg('Last complete catalogue:', {id: 'connections.catalogueAge'})} ${item.catalogueCreatedAt ?? '—'}</p>
        ${item.tools.map((tool) => html`<section><strong>${tool.remoteName}</strong><p>${tool.description}</p><pre>${JSON.stringify(tool.inputSchema, null, 2)}</pre></section>`)}
      </details>
      <div class=${styles.actions}>
        <button class="dl-btn" type="button" ?disabled=${this.pending} @click=${(event: Event) => command('probe', event)}>${msg('Probe', {id: 'connections.probe'})}</button>
        ${item.enabled ? html`<button class="dl-btn" type="button" ?disabled=${this.pending} @click=${(event: Event) => command('disable', event)}>${msg('Disable', {id: 'connections.disable'})}</button>`
          : html`<button data-enable class="dl-btn" type="button" ?disabled=${this.pending || !item.catalogueCreatedAt || item.status === 'revoked'} @click=${() => {this.consent = item.connectionId;}}>${msg('Enable', {id: 'connections.enable'})}</button>`}
        <button class="dl-btn" type="button" ?disabled=${this.pending} @click=${(event: Event) => command('revoke', event)}>${msg('Revoke credentials', {id: 'connections.revoke'})}</button>
        <button class="dl-btn dl-btn-danger-text" type="button" ?disabled=${this.pending} @click=${(event: Event) => command('delete', event)}>${msg('Delete', {id: 'connections.delete'})}</button>
      </div>
      ${this.consent === item.connectionId ? html`<div class=${styles.warning} role="alert">
        <p>${msg('Enabling authorizes ALL current and future tools. Tools can read and can modify, send, or delete data in your granted external accounts, including shared workspaces. DlightRAG cannot prove that a remote server isolates your external data. OAuth scope growth still requires provider consent. There are no per-call confirmations.', {id: 'connections.warning'})}</p>
        <button data-consent class="dl-btn" type="button" ?disabled=${this.pending} @click=${(event: Event) => {void this.#change({kind: 'enable', connectionId: item.connectionId, consentVersion: 1}, event.currentTarget as HTMLElement);}}>${msg('Accept and enable all tools', {id: 'connections.accept'})}</button>
        <button class="dl-btn" type="button" ?disabled=${this.pending} @click=${() => {this.consent = null;}}>${msg('Cancel', {id: 'connections.cancel'})}</button>
      </div>` : nothing}
      <label>${msg('Replace personal bearer (write-only)', {id: 'connections.bearer'})}<input type="password" autocomplete="off" maxlength="8192"></label>
      <button class="dl-btn" type="button" ?disabled=${this.pending} @click=${(event: Event) => this.#bearer(item, event)}>${msg('Replace bearer and probe', {id: 'connections.replaceBearer'})}</button>
      <button data-oauth class="dl-btn" type="button" ?disabled=${this.pending} @click=${(event: Event) => {void this.#oauth(item, event);}}>${msg('Authorize OAuth for this endpoint', {id: 'connections.oauthBegin'})}</button>
    </article>`;
  }
  protected override render(): TemplateResult {
    return html`<section data-connections-root tabindex="-1" @keydown=${(event: KeyboardEvent) => {if (event.key === 'Enter' && event.target instanceof HTMLInputElement) event.preventDefault();}} class=${styles.root} aria-label=${msg('MCP Connections', {id: 'connections.title'})}>
      <h3>MCP</h3>
      ${this.view?.singleUser ? html`<p>${msg('Local single-user Connections', {id: 'connections.singleUser'})}</p>` : nothing}
      <p>${msg('Enabled Connections are automatically included in your future Research Runs.', {id: 'connections.answerAutomatic'})}</p>
      ${this.authorizationUrl ? html`<p><a data-oauth-continue href=${this.authorizationUrl} rel="noreferrer noopener">${msg('Continue to provider authorization', {id: 'connections.oauthContinue'})}</a></p>` : nothing}
      <p>${msg('Authorization must finish while the initiating worker is alive. Expired tokens refresh automatically within consented scopes. Rejected refresh or new scopes require authorization here.', {id: 'connections.oauthRestart'})}</p>
      <p>${msg('Disable blocks new dispatch after the authorization gate. Already in-flight work is only best-effort cancelled and cannot be rolled back.', {id: 'connections.disableWarning'})}</p>
      ${this.error ? html`<p role="alert">${msg('Connection request failed or the revision changed. Reload and retry; an authenticated endpoint change needs new authorization.', {id: 'connections.error'})}</p><button type="button" class="dl-btn" ?disabled=${this.pending} @click=${() => {void this.#load();}}>${msg('Reload', {id: 'connections.reload'})}</button>` : nothing}
      ${!this.view ? html`<p role="status">${msg('Loading Connections…', {id: 'connections.loading'})}</p>` : repeat(this.view.connections, (item) => item.connectionId, (item) => this.#item(item))}
      <div class=${styles.card}>
        <h4>${msg('New Streamable HTTP Connection', {id: 'connections.new'})}</h4>
        <label>${msg('Label', {id: 'connections.label'})}<input data-new-label required maxlength="100"></label>
        <label>${msg('Endpoint', {id: 'connections.endpoint'})}<input data-new-endpoint type="url" required maxlength="2048" placeholder="https://example.com/mcp"></label>
        <p>${msg('Created disabled and unauthenticated. Add a personal bearer or authorize OAuth if needed, then probe before enabling.', {id: 'connections.draft'})}</p>
        <button type="button" class="dl-btn" ?disabled=${this.pending || !this.view} @click=${this.#create}>${msg('Create disabled Connection', {id: 'connections.create'})}</button>
      </div>
    </section>`;
  }
}
customElements.define('dl-settings-connections', DlSettingsConnections);
declare global {interface HTMLElementTagNameMap {'dl-settings-connections': DlSettingsConnections;}}
