// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Settings → Connections: one MCP group of Connection cards.

 * The group is collapsed by default and its summary reads `N of M enabled`, so nothing here
 * reports on tools: the Capability Catalogue belongs to the Agent, and the UI never receives it.
 * A switch owns discovery, because enabling a Connection with no confirmed catalogue requires a
 * probe first and the user should not have to know that. Enabling is gated once per owner session
 * by the whole-Connection warning; the backend records `consent_version=1` on every enable.
 */

import {msg, str, updateWhenLocaleChanges} from '@lit/localize';
import {html, nothing, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import {
  beginConnectionAuthorization,
  changeConnection,
  getConnections,
  type Connection,
  type ConnectionsView,
  type Preset,
} from '../api/connections.ts';
import {icon} from '../design-system/index.ts';
import {LightElement} from '../lib/lit-host.ts';
import {modalResult} from './modal.ts';
import styles from '../styles/settings-connections.module.css';

const POLL_MILLISECONDS = 5000;
type Authentication = Connection['authentication'];
type Command = Parameters<typeof changeConnection>[1];

const MODE_LABEL: Record<Authentication, () => string> = {
  none: () => msg('None', {id: 'connections.modeNone'}),
  bearer: () => msg('Bearer', {id: 'connections.modeBearer'}),
  oauth: () => msg('OAuth', {id: 'connections.modeOauth'}),
};

export class DlSettingsConnections extends LightElement {
  static properties = {
    view: {state: true},
    expanded: {state: true},
    openCard: {state: true},
    editingEndpoint: {state: true},
    draftAuth: {state: true},
    pending: {state: true},
    busy: {state: true},
    deleting: {state: true},
    adding: {state: true},
    error: {state: true},
    authorizationUrl: {state: true},
  };

  declare view: ConnectionsView | null;
  declare expanded: boolean;
  declare openCard: string | null;
  declare editingEndpoint: string | null;
  declare draftAuth: Record<string, Authentication>;
  /** The Connection whose switch or credential control owns the in-flight command. */
  declare pending: string | null;
  /** Any command is in flight, including create, which has no Connection id yet. */
  declare busy: boolean;
  /** The Connection the delete dialog is asking about, so the copy can name it. */
  declare deleting: string | null;
  declare adding: boolean;
  declare error: boolean;
  declare authorizationUrl: string | null;

  #events: AbortController | null = null;
  #timer: ReturnType<typeof setTimeout> | null = null;
  #generation = 0;
  #acknowledged = false;
  /** The tab the last preset asked for, applied to the Connection the next create produces. */
  #presetAuth: Authentication | null = null;

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.view = null;
    this.expanded = false;
    this.openCard = null;
    this.editingEndpoint = null;
    this.draftAuth = {};
    this.pending = null;
    this.busy = false;
    this.deleting = null;
    this.adding = false;
    this.error = false;
    this.authorizationUrl = null;
    /** One enable per owner session is enough to record the standing authorization. */
    this.#acknowledged = false;
  }

  override connectedCallback(): void {
    super.connectedCallback();
    this.#events = new AbortController();
    void this.#load();
  }

  override disconnectedCallback(): void {
    this.authorizationUrl = null;
    this.#events?.abort();
    this.#events = null;
    this.#generation++;
    if (this.#timer) clearTimeout(this.#timer);
    this.querySelectorAll<HTMLInputElement>('input[type=password]').forEach((input) => {
      input.value = '';
    });
    super.disconnectedCallback();
  }

  /** Settings asks for the expanded group on the OAuth return path only. */
  expand(): void {
    this.expanded = true;
  }

  async #load(): Promise<void> {
    const signal = this.#events?.signal;
    if (!signal || signal.aborted || this.busy) return;
    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = null;
    const generation = ++this.#generation;
    try {
      const view = await getConnections(signal);
      if (!signal.aborted && generation === this.#generation) {
        this.view = view;
        this.error = false;
        // An owner who already runs an enabled Connection answered the warning before, so the
        // standing authorization covers every later enable in this session too.
        if (view.connections.some((connection) => connection.enabled)) this.#acknowledged = true;
      }
    } catch {
      if (!signal.aborted && generation === this.#generation) this.error = true;
    }
    if (!signal.aborted) {
      this.#timer = setTimeout(() => {
        if (!this.pending) void this.#load();
      }, POLL_MILLISECONDS);
    }
  }

  /** Apply one command, then restore focus so a replaced control never loses the user. */
  async #command(command: Command, returnFocus?: HTMLElement | null): Promise<ConnectionsView | null> {
    const signal = this.#events?.signal;
    if (!this.view || this.busy || !signal || signal.aborted) return null;
    this.#generation++;
    this.busy = true;
    this.pending = command.kind === 'create' ? null : command.connectionId;
    this.error = false;
    if (this.#timer) clearTimeout(this.#timer);
    try {
      const view = await changeConnection(this.view.revision, command, signal);
      if (signal.aborted) return null;
      this.view = view;
      this.authorizationUrl = null;
      return view;
    } catch {
      if (!signal.aborted) this.error = true;
      return null;
    } finally {
      if (!signal.aborted) await this.#settle(returnFocus);
    }
  }

  #authOf(connection: Connection): Authentication {
    return this.draftAuth[connection.connectionId] ?? connection.authentication;
  }

  /** One home per observed status: what it says, how it reads, and what it implies. */
  #statusOf(connection: Connection): {label: string; tone: string; note?: string} {
    switch (connection.status) {
      case 'ready':
        return {label: msg('Ready', {id: 'connections.ready'}), tone: styles.dotReady};
      case 'refreshing':
        return {label: msg('Refreshing', {id: 'connections.refreshing'}), tone: styles.dotReady};
      case 'degraded':
        return {
          label: msg('Not responding', {id: 'connections.notResponding'}),
          tone: styles.dotProblem,
          note: msg('Research keeps using what was last confirmed.', {id: 'connections.lastConfirmed'}),
        };
      case 'needs-auth':
        return {
          label: msg('Needs authorization', {id: 'connections.needsAuth'}),
          tone: styles.dotProblem,
          note: msg('Reauthorize to use this server again.', {id: 'connections.reauthorize'}),
        };
      case 'revoked':
        return {
          label: msg('Revoked', {id: 'connections.revoked'}),
          tone: styles.dotIdle,
          note: msg(
            'The stored authorization was revoked. Authorize again to use this server.',
            {id: 'connections.revokedNote'},
          ),
        };
      default:
        return {label: msg('Disabled', {id: 'connections.disabled'}), tone: styles.dotIdle};
    }
  }

  /** The row stays quiet while a Connection is off or plainly working; the light says which. */
  #meta(connection: Connection): string {
    if (this.pending === connection.connectionId) return msg('Connecting…', {id: 'connections.connecting'});
    const parts = [MODE_LABEL[connection.authentication]()];
    if (connection.status !== 'disabled' && connection.status !== 'ready') {
      parts.push(this.#statusOf(connection).label);
    }
    return parts.join(' · ');
  }

  /**
   * The light means connected: green while working, red while enabled but unanswered, and no
   * colour at all while it is switched off, because being off is not a fault.
   */
  #dotClass(connection: Connection): string {
    if (!connection.enabled || this.pending === connection.connectionId) return styles.dotIdle;
    return this.#statusOf(connection).tone;
  }

  #note(connection: Connection): TemplateResult | typeof nothing {
    if (connection.authorizationStatus === 'failed') {
      return html`<p class=${styles.note} role="alert">${msg(
        'Authorization failed or expired. Authorize again to use this server.',
        {id: 'connections.oauthFailed'},
      )}</p>`;
    }
    if (connection.authorizationStatus === 'pending') {
      return html`<p class=${styles.note} role="status">${msg('Authorization pending.', {id: 'connections.oauthPending'})}</p>`;
    }
    const {note} = this.#statusOf(connection);
    if (!note) return nothing;
    return html`<p class=${styles.note} role="alert">${note}</p>`;
  }

  #labelOf(connectionId: string): string {
    return this.view?.connections.find((item) => item.connectionId === connectionId)?.label ?? '';
  }

  #endpointOf(connectionId: string): string {
    return this.view?.connections.find((item) => item.connectionId === connectionId)?.endpoint ?? '';
  }

  #switchOf(connection: Connection): HTMLElement | null {
    return this.querySelector<HTMLElement>(`[data-switch="${connection.connectionId}"]`);
  }

  /** Open one owned confirm dialog and return its outcome, with focus restored either way. */
  async #ask(dialogSelector: string, trigger: HTMLElement | null): Promise<string> {
    const dialog = this.querySelector<HTMLDialogElement>(dialogSelector);
    if (!dialog) return '';
    return await modalResult(this, dialog, () => trigger?.focus(), this.#events?.signal);
  }

  async #settle(returnFocus?: HTMLElement | null): Promise<void> {
    this.busy = false;
    this.pending = null;
    await this.updateComplete;
    const target = returnFocus?.isConnected
      ? returnFocus
      : this.querySelector<HTMLElement>('[data-connections-root]');
    target?.focus();
    this.#timer = setTimeout(() => {
      void this.#load();
    }, POLL_MILLISECONDS);
  }

  /**
   * Switching on can require two commands: a Connection with no confirmed catalogue is checked
   * first, and a failed check reports itself instead of leaving a dead Connection enabled.
   */
  async #turnOn(connection: Connection): Promise<void> {
    if (!this.#acknowledged) {
      if (await this.#ask('#connections-consent', this.#switchOf(connection)) !== 'enable') return;
      this.#acknowledged = true;
    }
    // The switch owns discovery: a Connection whose last observation is neither working nor
    // in-flight is checked first, and a check that reports an authentication failure leaves it off
    // rather than enabling a server that cannot answer.
    if (connection.status !== 'ready' && connection.status !== 'refreshing') {
      const checked = await this.#command(
        {kind: 'probe', connectionId: connection.connectionId},
        this.#switchOf(connection),
      );
      const current = checked?.connections.find((item) => item.connectionId === connection.connectionId);
      if (current?.status !== 'ready') return;
    }
    await this.#command(
      {kind: 'enable', connectionId: connection.connectionId, consentVersion: 1},
      this.#switchOf(connection),
    );
  }

  #switch(connection: Connection): TemplateResult {
    const pending = this.pending === connection.connectionId;
    return html`<button class="dl-switch dl-switch--sm" type="button" role="switch" data-switch=${connection.connectionId}
      aria-checked=${String(connection.enabled)} ?disabled=${pending}
      aria-label=${connection.enabled
        ? msg(str`Disable ${connection.label} in Research`, {id: 'connections.switchOff'})
        : msg(str`Enable ${connection.label} in Research`, {id: 'connections.switchOn'})}
      @click=${() => {
        if (connection.enabled) {
          void this.#command({kind: 'disable', connectionId: connection.connectionId});
        } else {
          void this.#turnOn(connection);
        }
      }}></button>`;
  }

  #labelField(connection: Connection): TemplateResult {
    return html`<label class=${styles.field}>
      <span class=${styles.fieldLabel}>${msg('Label', {id: 'connections.label'})}</span>
      <input class=${styles.input} data-label=${connection.connectionId} .value=${connection.label}
        maxlength="100" @change=${(event: Event) => {
          const label = (event.target as HTMLInputElement).value;
          if (!label || label === connection.label) return;
          void this.#command(
            {kind: 'edit', connectionId: connection.connectionId, label, endpoint: connection.endpoint},
            event.target as HTMLElement,
          );
        }}>
    </label>`;
  }

  #endpointRow(connection: Connection): TemplateResult {
    if (connection.authentication !== 'none') {
      return html`<div class=${styles.endpointRow}>
        <span class=${styles.endpoint}>${connection.endpoint}</span>
      </div>`;
    }
    if (this.editingEndpoint !== connection.connectionId) {
      return html`<div class=${styles.endpointRow}>
        <span class=${styles.endpoint}>${connection.endpoint}</span>
        <button class="dl-btn ${styles.blockAction}" type="button"
          @click=${() => {
            this.editingEndpoint = connection.connectionId;
          }}>${msg('Change endpoint', {id: 'connections.changeEndpoint'})}</button>
      </div>`;
    }
    return html`<div class=${styles.stackTight}>
      <label class=${styles.field}>
        <span class=${styles.fieldLabel}>${msg('Endpoint', {id: 'connections.endpoint'})}</span>
        <input class=${styles.input} type="url" maxlength="2048"
          data-endpoint=${connection.connectionId} .value=${connection.endpoint}>
      </label>
      <div class=${styles.actionsEnd}>
        <button class="dl-btn" type="button" @click=${() => {
          this.editingEndpoint = null;
        }}>${msg('Cancel', {id: 'connections.cancel'})}</button>
        <button class="primary-btn" type="button" data-save-endpoint=${connection.connectionId}
          @click=${(event: Event) => {
            const input = this.querySelector<HTMLInputElement>(`[data-endpoint="${connection.connectionId}"]`);
            if (!input?.reportValidity()) return;
            this.editingEndpoint = null;
            void this.#command(
              {kind: 'edit', connectionId: connection.connectionId, label: connection.label, endpoint: input.value},
              event.currentTarget as HTMLElement,
            );
          }}>${msg('Save', {id: 'connections.save'})}</button>
      </div>
    </div>`;
  }

  #authBlock(connection: Connection, auth: Authentication): TemplateResult {
    if (auth === 'bearer') {
      return html`<div class=${styles.stackTight}>
        <label class=${styles.field}>
          <span class=${styles.fieldLabel}>${msg('Personal bearer (write-only)', {id: 'connections.bearer'})}</span>
          <input class=${styles.input} type="password" autocomplete="off" maxlength="8192"
            data-bearer=${connection.connectionId}>
        </label>
        <button class="dl-btn ${styles.blockAction}" type="button"
          @click=${(event: Event) => {
            const input = this.querySelector<HTMLInputElement>(`[data-bearer="${connection.connectionId}"]`);
            if (!input?.value) return;
            const bearer = input.value;
            input.value = '';
            void this.#command({
              kind: 'bearer',
              connectionId: connection.connectionId,
              bearer,
              endpoint: connection.endpoint,
            }, event.currentTarget as HTMLElement);
          }}>${msg('Save bearer', {id: 'connections.saveBearer'})}</button>
      </div>`;
    }
    if (auth === 'oauth') {
      return html`<div class=${styles.stackTight}>
        <button class="dl-btn ${styles.blockAction}" type="button" data-oauth=${connection.connectionId}
          @click=${(event: Event) => {
            void this.#beginAuthorization(connection, event.currentTarget as HTMLElement);
          }}>${msg('Authorize with OAuth', {id: 'connections.oauthBegin'})}</button>
        ${this.authorizationUrl
          ? html`<p><a data-oauth-continue href=${this.authorizationUrl} rel="noreferrer noopener"
            >${msg('Continue to provider authorization', {id: 'connections.oauthContinue'})}</a></p>`
          : nothing}
        <p class=${styles.hint}>${msg(
          'Authorization must finish in this session. Tokens refresh automatically inside the consented scopes.',
          {id: 'connections.oauthRestart'},
        )}</p>
      </div>`;
    }
    return html`<p class=${styles.hint}>${msg(
      'No credential. Anyone who can reach this URL can use it.',
      {id: 'connections.noCredential'},
    )}</p>`;
  }

  async #beginAuthorization(connection: Connection, returnFocus: HTMLElement): Promise<void> {
    const signal = this.#events?.signal;
    if (!this.view || this.busy || !signal || signal.aborted) return;
    this.busy = true;
    this.pending = connection.connectionId;
    this.error = false;
    this.authorizationUrl = null;
    this.#generation++;
    if (this.#timer) clearTimeout(this.#timer);
    try {
      const url = await beginConnectionAuthorization(
        this.view.revision, connection.connectionId, connection.endpoint, signal,
      );
      if (!signal.aborted) this.authorizationUrl = url;
    } catch {
      if (!signal.aborted) this.error = true;
    } finally {
      if (!signal.aborted) await this.#settle(returnFocus);
    }
  }

  async #requestDelete(connection: Connection): Promise<void> {
    this.deleting = connection.connectionId;
    await this.updateComplete;
    const trigger = this.querySelector<HTMLElement>(`[data-delete="${connection.connectionId}"]`);
    const outcome = await this.#ask('#connections-delete', trigger);
    this.deleting = null;
    if (outcome === 'delete') {
      void this.#command({kind: 'delete', connectionId: connection.connectionId});
    }
  }

  #card(connection: Connection): TemplateResult {
    const auth = this.#authOf(connection);
    const expanded = this.openCard === connection.connectionId;
    return html`<article class=${styles.card}>
      <div class=${styles.cardHeader}>
        <button class=${styles.cardToggle} type="button" aria-expanded=${String(expanded)}
          data-card=${connection.connectionId}
          @click=${() => {
            this.openCard = expanded ? null : connection.connectionId;
            this.editingEndpoint = null;
          }}>
          <span class="${styles.dot} ${this.#dotClass(connection)}">${icon('status-dot', {size: 'xs'})}</span>
          <span class=${styles.rowText}>
            <span class=${styles.rowLabel}>${connection.label}</span>
            <span class=${styles.rowMeta}>${this.#meta(connection)}</span>
          </span>
          <span class=${styles.chevron}>${icon('disclosure', {size: 'sm'})}</span>
        </button>
        ${this.#switch(connection)}
      </div>
      ${expanded ? html`<div class=${styles.cardBody}>
        ${this.#note(connection)}
        ${this.#labelField(connection)}
        ${this.#endpointRow(connection)}
        <div class=${styles.stackTight}>
          <span class=${styles.sectionLabel}>${msg('Authentication', {id: 'connections.authentication'})}</span>
          <div class=${styles.segmented}>
            ${(['none', 'bearer', 'oauth'] as const).map((mode) => html`
              <button class="${styles.segment} ${mode === auth ? styles.segmentActive : ''}" type="button"
                aria-pressed=${String(mode === auth)}
                @click=${() => {
                  this.draftAuth = {...this.draftAuth, [connection.connectionId]: mode};
                }}>${MODE_LABEL[mode]()}</button>`)}
          </div>
          ${this.#authBlock(connection, auth)}
        </div>
        <div class=${styles.dangerRow}>
          <button class="dl-btn dl-btn-danger-text" type="button" data-delete=${connection.connectionId}
            @click=${() => {
              void this.#requestDelete(connection);
            }}>${msg('Delete', {id: 'connections.delete'})}</button>
        </div>
      </div>` : nothing}
    </article>`;
  }

  /**
   * A preset only ever fills the form: it selects no endpoint policy, stores no credential, and
   * grants no authority -- creating the Connection stays the one act that does any of that.
   */
  #applyPreset(preset: Preset): void {
    const label = this.querySelector<HTMLInputElement>('[data-new-label]');
    const endpoint = this.querySelector<HTMLInputElement>('[data-new-endpoint]');
    if (label) label.value = preset.label;
    if (endpoint) endpoint.value = preset.endpoint;
    this.#presetAuth = preset.defaultAuthentication;
  }

  /**
   * Create, then open the new Connection on the tab its preset implies: an endpoint that answers
   * unauthenticated starts on None, and one that cannot be used without an account starts on the
   * choice that account needs, so the owner never has to guess which tab to pick.
   */
  async #create(label: string, endpoint: string, returnFocus: HTMLElement | null): Promise<void> {
    const presetAuth = this.#presetAuth;
    const known = new Set(
      this.view?.connections.map((connection) => connection.connectionId) ?? [],
    );
    this.#presetAuth = null;
    const view = await this.#command({kind: 'create', label, endpoint}, returnFocus);
    const created = view?.connections.find((connection) => !known.has(connection.connectionId));
    if (!created || !presetAuth) return;
    this.draftAuth = {...this.draftAuth, [created.connectionId]: presetAuth};
    this.expanded = true;
    this.openCard = created.connectionId;
  }

  #createForm(): TemplateResult {
    const presets = this.view?.presets ?? [];
    return html`<div class=${styles.stackTight}>
      ${presets.length === 0 ? nothing : html`
        <div class=${styles.presetRow} role="group"
          aria-label=${msg('Presets', {id: 'connections.presets'})}>
          ${presets.map((preset) => html`
            <button class=${styles.presetChip} type="button" data-preset=${preset.presetId}
              aria-label=${msg(str`Use the ${preset.label} preset`, {id: 'connections.usePreset'})}
              @click=${() => {
                this.#applyPreset(preset);
              }}>${preset.label}</button>`)}
        </div>`}
      <label class=${styles.field}>
        <span class=${styles.fieldLabel}>${msg('Label', {id: 'connections.label'})}</span>
        <input class=${styles.input} data-new-label required maxlength="100">
      </label>
      <label class=${styles.field}>
        <span class=${styles.fieldLabel}>${msg('Endpoint', {id: 'connections.endpoint'})}</span>
        <input class=${styles.input} data-new-endpoint type="url" required maxlength="2048"
          placeholder="https://example.com/mcp">
      </label>
      <p class=${styles.hint}>${msg('Starts inactive.', {id: 'connections.startsInactive'})}</p>
      <button class="primary-btn ${styles.blockAction}" type="button" ?disabled=${this.busy}
        @click=${(event: Event) => {
          const label = this.querySelector<HTMLInputElement>('[data-new-label]');
          const endpoint = this.querySelector<HTMLInputElement>('[data-new-endpoint]');
          if (!label?.reportValidity() || !endpoint?.reportValidity()) return;
          this.adding = false;
          void this.#create(label.value, endpoint.value, event.currentTarget as HTMLElement);
        }}>${msg('Add connection', {id: 'connections.add'})}</button>
    </div>`;
  }

  #group(): TemplateResult {
    const total = this.view?.connections.length ?? 0;
    const enabled = this.view?.connections.filter((connection) => connection.enabled).length ?? 0;
    const attention = this.view?.connections.filter((connection) => (
      connection.enabled && (connection.status === 'degraded' || connection.status === 'needs-auth')
    )).length ?? 0;
    return html`<button class=${styles.groupRow} type="button" aria-expanded=${String(this.expanded)}
      data-connections-root @click=${() => {
        this.expanded = !this.expanded;
      }}>
      <span class=${styles.groupName}>MCP</span>
      <span class=${styles.groupMeta}>${total === 0
        ? msg('add a connection', {id: 'connections.groupEmpty'})
        : msg(str`${enabled} of ${total} enabled`, {id: 'connections.groupSummary'})}</span>
      ${attention > 0
        ? html`<span class="sr-only">${attention === 1
          ? msg('1 connection needs attention', {id: 'connections.attentionOne'})
          : msg(str`${attention} connections need attention`, {id: 'connections.attentionMany'})}</span>`
        : nothing}
      <span class=${styles.groupChevron}>${icon('disclosure', {size: 'sm'})}</span>
    </button>`;
  }

  protected override render(): TemplateResult {
    const connections = this.view?.connections ?? [];
    return html`<section class=${styles.root} aria-label=${msg('MCP Connections', {id: 'connections.title'})}>
      ${this.#group()}
      ${this.expanded ? html`
        ${this.error ? html`<p class=${styles.note} role="alert">${msg(
          'Connection request failed or the revision changed. Reload and retry.',
          {id: 'connections.error'},
        )}</p>
        <button class="dl-btn" type="button" ?disabled=${this.busy} @click=${() => {
          void this.#load();
        }}>${msg('Reload', {id: 'connections.reload'})}</button>` : nothing}
        ${!this.view ? html`<p class=${styles.hint} role="status">${msg(
          'Loading Connections…', {id: 'connections.loading'})}</p>` : nothing}
        ${connections.length === 0 && this.view
          ? html`<p class=${styles.hint}>${msg('No MCP connections yet.', {id: 'connections.empty'})}</p>`
          : nothing}
        ${repeat(connections, (connection) => connection.connectionId, (connection) => this.#card(connection))}
        <article class=${styles.card}>
          <button class=${styles.addRow} type="button" @click=${() => {
            this.adding = !this.adding;
          }}>${icon('add', {size: 'sm'})}${msg('Add MCP connection', {id: 'connections.new'})}</button>
          ${this.adding ? html`<div class=${styles.cardBody}>${this.#createForm()}</div>` : nothing}
        </article>` : nothing}

      <dialog class="confirm-dialog" id="connections-consent" aria-labelledby="connections-consent-title">
        <form method="dialog" novalidate>
          <h2 id="connections-consent-title">${msg(
            'Enable external servers for Research?', {id: 'connections.consentTitle'})}</h2>
          <p>${msg(
            'While a Connection is switched on, DlightRAG calls that server during Research runs without asking each time. It can read and can modify, send, or delete data in the account you authorized, including shared workspaces, and a server may add capabilities later.',
            {id: 'connections.consentBody'},
          )}</p>
          <div class="dl-dialog-actions">
            <button type="submit" value="cancel">${msg('Cancel', {id: 'connections.cancel'})}</button>
            <button type="submit" value="enable" class="dl-dialog-danger">${msg(
              'I understand', {id: 'connections.consentAccept'})}</button>
          </div>
        </form>
      </dialog>

      <dialog class="confirm-dialog" id="connections-delete" aria-labelledby="connections-delete-title">
        <form method="dialog" novalidate>
          <h2 id="connections-delete-title">${this.deleting
            ? msg(str`Delete ${this.#labelOf(this.deleting)}?`, {id: 'connections.deleteTitle'})
            : msg('Delete this Connection?', {id: 'connections.deleteTitleEmpty'})}</h2>
          <p>${msg(str`This removes the endpoint ${this.deleting ? this.#endpointOf(this.deleting) : ''}, the label, and the stored credential. Adding this server again means entering the endpoint from scratch and authorizing again.`,
            {id: 'connections.deleteBody'})}</p>
          <p>${msg(
            'To stop using it without losing the configuration, switch it off instead.',
            {id: 'connections.deleteEscape'},
          )}</p>
          <div class="dl-dialog-actions">
            <button type="submit" value="cancel">${msg('Cancel', {id: 'connections.cancel'})}</button>
            <button type="submit" value="delete" class="dl-dialog-danger">${msg(
              'Delete connection', {id: 'connections.deleteConfirm'})}</button>
          </div>
        </form>
      </dialog>
    </section>`;
  }
}
customElements.define('dl-settings-connections', DlSettingsConnections);
declare global {interface HTMLElementTagNameMap {'dl-settings-connections': DlSettingsConnections;}}
