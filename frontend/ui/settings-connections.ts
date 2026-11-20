// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Settings → Connections: the MCP Connection cards.

 * The page says `N of M enabled`, and each card states its published catalogue's tool count and
 * the estimated token cost of its declarations; the Capability Catalogue itself belongs to the
 * Agent, and the UI never receives it. A switch owns discovery, because enabling a Connection
 * with no confirmed catalogue requires a probe first and the user should not have to know that.
 * Enabling is gated once per owner session by the whole-Connection warning; the backend records
 * `consent_version=1` on every enable.
 */

import {msg, str} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import {
  beginConnectionAuthorization,
  changeConnection,
  ConnectionsApiError,
  getConnections,
  type Connection,
  type ConnectionsView,
  type EndpointHeaderPair,
  type Preset,
} from '../api/connections.ts';
import {icon} from '../design-system/index.ts';
import {getLocale} from '../i18n/locale.ts';
import {LightElement} from '../lib/lit-host.ts';
import {formatCompactCount} from '../lib/number-format.ts';
import shared from '../styles/settings-page.module.css';
import styles from '../styles/settings-connections.module.css';
import {modalResult} from './modal.ts';
import {reportSettingsSummary} from './settings-summary.ts';
import {tokenCountLabel} from './token-count.ts';

const POLL_MILLISECONDS = 5000;
/** An authorization in flight settles seconds after the browser comes back, and the reader waits. */
const AUTHORIZATION_POLL_MILLISECONDS = 1000;
/** An authorization nobody finishes stays pending for its whole lifetime, so the quick look is spent. */
const AUTHORIZATION_POLL_ATTEMPTS = 30;
type Authentication = Connection['authentication'];
type Command = Parameters<typeof changeConnection>[1];

const MODE_LABEL: Record<Authentication, () => string> = {
  none: () => msg('None', {id: 'connections.modeNone'}),
  bearer: () => msg('Bearer', {id: 'connections.modeBearer'}),
  header: () => msg('Header', {id: 'connections.modeHeader'}),
  oauth: () => msg('OAuth', {id: 'connections.modeOauth'}),
};

/** What a Connection holds, as the reader chooses it: nothing, one token, or a provider's grant.
 *  Where a static token travels is the second, smaller choice — `bearer` is its default answer. */
type CredentialType = 'none' | 'token' | 'oauth';
const CREDENTIAL_TYPES = ['none', 'token', 'oauth'] as const;
const STATIC_TOKEN = 'token';

const CREDENTIAL_TYPE_LABEL: Record<CredentialType, () => string> = {
  none: () => MODE_LABEL.none(),
  token: () => msg('Static token', {id: 'connections.typeToken'}),
  oauth: () => MODE_LABEL.oauth(),
};

function credentialType(auth: Authentication): CredentialType {
  if (auth === 'none') return 'none';
  return auth === 'oauth' ? 'oauth' : STATIC_TOKEN;
}

/** What a row says while an authorization is the story of a Connection that holds no credential. */
const AUTHORIZATION_LABEL: Record<'pending' | 'failed' | 'changed', () => string> = {
  pending: () => msg('Authorization in progress', {id: 'connections.authorizing'}),
  failed: () => msg('Authorization failed', {id: 'connections.authorizationFailed'}),
  changed: () => msg('Authorization not saved', {id: 'connections.authorizationChanged'}),
};

class DlSettingsConnections extends LightElement {
  static properties = {
    view: {state: true},
    openCard: {state: true},
    editingEndpoint: {state: true},
    draftAuth: {state: true},
    draftHeaders: {state: true},
    draftCredential: {state: true},
    newHeaders: {state: true},
    pending: {state: true},
    busy: {state: true},
    deleting: {state: true},
    adding: {state: true},
    error: {state: true},
    refusal: {state: true},
  };

  declare view: ConnectionsView | null;
  declare openCard: string | null;
  declare editingEndpoint: string | null;
  declare draftAuth: Record<string, Authentication>;
  /** The endpoint headers a card is editing, seeded from the Connection on first touch. */
  declare draftHeaders: Record<string, EndpointHeaderPair[]>;
  /** What a card's credential form holds: the name and prefix a *new* credential would travel
   *  in. The stored one is read back from the projection (it is not a secret; the token is), and
   *  this map only holds what the reader has typed since. */
  declare draftCredential: Record<string, {header: string; prefix: string}>;
  /** The create form's stated headers: what a preset fills, and the owner may edit. */
  declare newHeaders: EndpointHeaderPair[];
  /** The Connection whose switch or credential control owns the in-flight command. */
  declare pending: string | null;
  /** Any command is in flight, including create, which has no Connection id yet. */
  declare busy: boolean;
  /** The Connection the delete dialog is asking about, so the copy can name it. */
  declare deleting: string | null;
  declare adding: boolean;
  declare error: boolean;
  /** What the Connections module said when it refused, when it said anything: product copy it
   *  wrote to be read, never remote text, and the reason the banner exists. */
  declare refusal: string | null;
  /** Where a started authorization sends the browser; a test replaces it to keep its own page. */
  navigateTo: (url: string) => void = (url) => {
    window.location.assign(url);
  };

  #timer: ReturnType<typeof setTimeout> | null = null;
  // Not a LatestRequest: a command, a newer read, or a disconnect only makes a read still in flight stale,
  // so its reply is ignored when it lands. Nothing is aborted: the signal is the element's, and commands share it.
  #generation = 0;
  /** Quick looks spent since a Connection last reported an authorization in flight. */
  #quickPolls = 0;
  #acknowledged = false;
  /** The tab the last preset asked for, applied to the Connection the next create produces. */
  #presetAuth: Authentication | null = null;

  constructor() {
    super();
    this.view = null;
    this.openCard = null;
    this.editingEndpoint = null;
    this.draftAuth = {};
    this.draftHeaders = {};
    this.draftCredential = {};
    this.newHeaders = [];
    this.pending = null;
    this.busy = false;
    this.deleting = null;
    this.adding = false;
    this.error = false;
    this.refusal = null;
    /** One enable per owner session is enough to record the standing authorization. */
    this.#acknowledged = false;
  }

  override connectedCallback(): void {
    super.connectedCallback();
    void this.#load();
  }

  override disconnectedCallback(): void {
    this.#generation++;
    if (this.#timer) clearTimeout(this.#timer);
    this.querySelectorAll<HTMLInputElement>('input[type=password]').forEach((input) => {
      input.value = '';
    });
    super.disconnectedCallback();
  }

  protected override updated(changed: PropertyValues<this>): void {
    if (changed.has('view') && this.view) {
      reportSettingsSummary(this, {
        section: 'connections',
        enabled: this.view.connections.filter((connection) => connection.enabled).length,
        total: this.view.connections.length,
      });
    }
  }

  /**
   * How long until the next read. A Connection whose authorization is still in flight is about to
   * change: the provider's code is deposited before the browser is sent back, and the token
   * exchange and publication follow within seconds, so a reader who has just returned watches the
   * card settle almost at once instead of a poll later. A first look at a settled page, and an
   * authorization abandoned at the provider, both keep the steady pace.
   */
  #pollDelay(): number {
    const authorizing = this.view?.connections
      .some((connection) => connection.authorizationStatus === 'pending') ?? false;
    if (!authorizing) {
      this.#quickPolls = 0;
      return POLL_MILLISECONDS;
    }
    this.#quickPolls += 1;
    return this.#quickPolls <= AUTHORIZATION_POLL_ATTEMPTS
      ? AUTHORIZATION_POLL_MILLISECONDS
      : POLL_MILLISECONDS;
  }

  async #load(): Promise<void> {
    const signal = this.lifetime;
    if (signal.aborted || this.busy) return;
    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = null;
    const generation = ++this.#generation;
    try {
      const view = await getConnections(signal);
      if (!signal.aborted && generation === this.#generation) {
        this.view = view;
        this.error = false;
        this.refusal = null;
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
      }, this.#pollDelay());
    }
  }

  /**
   * Say one command to the server, and say it again once if the owner's revision moved.
   *
   * A revision moves for things the reader did not do and cannot see: the background refresh, and
   * the authorization they just finished. Refusing their word because the page had not read that
   * yet would answer "the revision changed, reload and retry" at somebody who only flipped a
   * switch, so the page reads what the owner holds now and says the same word once more.
   */
  async #submit(command: Command, revision: string, signal: AbortSignal): Promise<ConnectionsView> {
    try {
      return await changeConnection(revision, command, signal);
    } catch (error) {
      if (signal.aborted || !(error instanceof ConnectionsApiError) || error.status !== 409) throw error;
      const current = await getConnections(signal);
      if (signal.aborted) throw error;
      this.view = current;
      return await changeConnection(current.revision, command, signal);
    }
  }

  /** Apply one command, then restore focus so a replaced control never loses the user. */
  async #command(command: Command, returnFocus?: HTMLElement | null): Promise<ConnectionsView | null> {
    const signal = this.lifetime;
    const read = this.view;
    if (!read || this.busy || signal.aborted) return null;
    this.#generation++;
    this.busy = true;
    this.pending = command.kind === 'create' ? null : command.connectionId;
    this.error = false;
    this.refusal = null;
    if (this.#timer) clearTimeout(this.#timer);
    try {
      const view = await this.#submit(command, read.revision, signal);
      if (signal.aborted) return null;
      this.view = view;
      return view;
    } catch (error) {
      if (!signal.aborted) {
        this.error = true;
        this.refusal = error instanceof ConnectionsApiError ? error.message : null;
      }
      return null;
    } finally {
      if (!signal.aborted) await this.#settle(returnFocus);
    }
  }

  /**
   * The OAuth attempt the server still reports, for a Connection that holds no credential yet.
   *
   * It is the whole story of such a Connection: the browser may have left for the provider's
   * consent page and come back to a fresh page, where the tab the reader had chosen is gone and
   * nothing is stored yet. Reading the credential alone would show "None" at the reader who has
   * just authorized, so the row, the light, and the card all read this instead.
   */
  #authorizationOf(connection: Connection): 'pending' | 'failed' | 'changed' | null {
    if (connection.authentication !== 'none') return null;
    const status = connection.authorizationStatus;
    return status === null || status === 'succeeded' ? null : status;
  }

  /**
   * The OAuth attempt this card should report, when one is its story.
   *
   * A Connection authorizing for the first time holds no credential yet, and one whose stored
   * grant is OAuth can have a re-authorization that failed or was superseded; both are answered
   * by the flow row. A *token* Connection keeps its last flow row for as long as the row lives,
   * and that row says nothing about the credential it holds now -- `#authorizationOf` above is
   * about the first of those cases only, because it also decides the row's own name and light.
   */
  #flowOutcome(connection: Connection): 'pending' | 'failed' | 'changed' | null {
    if (connection.authentication !== 'none' && connection.authentication !== 'oauth') return null;
    const status = connection.authorizationStatus;
    return status === null || status === 'succeeded' ? null : status;
  }

  #authOf(connection: Connection): Authentication {
    const chosen = this.draftAuth[connection.connectionId];
    if (chosen) return chosen;
    if (this.#authorizationOf(connection)) return 'oauth';
    return connection.authentication;
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
        // A credential that did not work says what to do next, and what that is depends on what
        // the card is showing: only OAuth has an authorization to redo, and a static token can
        // fail because it travels in the wrong header, which the reader can change right here.
        return {
          label: msg('Needs authorization', {id: 'connections.needsAuth'}),
          tone: styles.dotProblem,
          note: this.#refusalAdvice(connection),
        };
      case 'no-tools':
        // Not a fault of the server: it answered, and what it offers under the headers this
        // Connection states is nothing. The headers are the thing to look at, so the note says so.
        return {
          label: msg('No tools', {id: 'connections.noTools'}),
          tone: styles.dotProblem,
          note: msg(
            'This server published no tools under the request headers this Connection states. Check them, then probe again.',
            {id: 'connections.noToolsNote'},
          ),
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

  /**
   * What the row says about a credential. One word when the transport is the default, and the
   * header itself when it is not: "TAT or UAT?" is the question a reader cannot answer from a
   * stored page otherwise, and it is the whole reason the server reads the name back.
   */
  #credentialSentence(connection: Connection): string {
    const auth = this.#authOf(connection);
    if (auth !== 'header') return MODE_LABEL[auth]();
    const {header} = this.#credentialDraft(connection);
    return header === ''
      ? MODE_LABEL.header()
      : msg(str`Credential in ${header}`, {id: 'connections.metaHeaderCredential'});
  }

  /**
   * What a credential that did not work means, for the credential the card is showing.
   *
   * `needs-auth` has two causes and they are not distinguishable from the card: the server
   * refused the credential, or this deployment cannot open its envelope at all (a lost key ring,
   * a retired key) -- in which case nothing was sent and no header the reader picks will help.
   * So the advice names both moves instead of asserting a refusal that may not have happened.
   */
  #refusalAdvice(connection: Connection): string {
    switch (this.#authOf(connection)) {
      case 'oauth':
        return msg('Reauthorize to use this server again.', {id: 'connections.reauthorize'});
      case 'header':
        return msg(
          'This credential was refused, or this deployment cannot open it. Check the header name, the prefix, and the kind of token — a user token and an app token are not interchangeable — then save it again.',
          {id: 'connections.refusedHeader'},
        );
      case 'bearer':
        return msg(
          'This credential was refused, or this deployment cannot open it. If the service does not take Authorization: Bearer, move the credential to a header of its own; otherwise save it again.',
          {id: 'connections.refusedBearer'},
        );
      default:
        return msg(
          'The server asks for a credential. Add a static token, or authorize with OAuth if it uses one.',
          {id: 'connections.refusedWithout'},
        );
    }
  }

  /** The row stays quiet while a Connection is off or plainly working; the light says which. */
  #meta(connection: Connection): string {
    if (this.pending === connection.connectionId) return msg('Connecting…', {id: 'connections.connecting'});
    const parts = [this.#credentialSentence(connection)];
    const authorization = this.#authorizationOf(connection);
    if (authorization) {
      parts.push(AUTHORIZATION_LABEL[authorization]());
    } else if (connection.status !== 'disabled' && connection.status !== 'ready') {
      parts.push(this.#statusOf(connection).label);
    }
    return parts.join(' · ');
  }

  /**
   * The light means connected: green while working, red while enabled but unanswered, and no
   * colour at all while it is switched off, because being off is not a fault.
   */
  #dotClass(connection: Connection): string {
    if (this.pending === connection.connectionId) return styles.dotIdle;
    // A refused or superseded authorization is this Connection's fault, looked at or not.
    const authorization = this.#authorizationOf(connection);
    if (authorization === 'failed' || authorization === 'changed') return styles.dotProblem;
    if (!connection.enabled) return styles.dotIdle;
    return this.#statusOf(connection).tone;
  }

  #note(connection: Connection): TemplateResult | typeof nothing {
    const authorization = this.#flowOutcome(connection);
    if (authorization === 'changed') {
      return html`<p class=${shared.note} role="alert">${msg(
        'This connection changed while you were authorizing, so nothing was saved. Authorize again.',
        {id: 'connections.oauthChanged'},
      )}</p>`;
    }
    if (authorization === 'failed') {
      return html`<p class=${shared.note} role="alert">${msg(
        'Authorization failed or expired. Authorize again to use this server.',
        {id: 'connections.oauthFailed'},
      )}</p>`;
    }
    if (authorization === 'pending') {
      return html`<p class=${shared.note} role="status">${msg('Authorization pending.', {id: 'connections.oauthPending'})}</p>`;
    }
    const {note} = this.#statusOf(connection);
    if (!note) return nothing;
    return html`<p class=${shared.note} role="alert">${note}</p>`;
  }

  /**
   * What the published catalogue's definitions cost one Research request: one quiet line, drawn
   * only when the projection holds an estimate. The token figure is the runtime's own estimate,
   * so the card and the Run cannot disagree; a disabled Connection's figure is hypothetical, and
   * the line states what switching it on would add rather than hiding it. Nothing is drawn until
   * a catalogue is published, because there is nothing to estimate.
   */
  #declarationCost(connection: Connection): TemplateResult | typeof nothing {
    const tokens = connection.declarationTokens;
    if (tokens === null || tokens === undefined) return nothing;
    const tokensLabel = tokenCountLabel(tokens);
    const tools = connection.toolCount;
    if (tools === null || tools === undefined) {
      return html`<span class=${shared.hint}>${msg(
        str`≈ ${tokensLabel} per request`,
        {id: 'connections.declarationCostTokens'},
      )}</span>`;
    }
    if (tools === 1) {
      return html`<span class=${shared.hint}>${msg(
        str`≈ ${tokensLabel} per request · 1 tool`,
        {id: 'connections.declarationCostOne'},
      )}</span>`;
    }
    const toolCount = formatCompactCount(tools, getLocale());
    return html`<span class=${shared.hint}>${msg(
      str`≈ ${tokensLabel} per request · ${toolCount} tools`,
      {id: 'connections.declarationCostMany'},
    )}</span>`;
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
    return await modalResult(this, dialog, () => trigger?.focus(), this.lifetime);
  }

  async #settle(returnFocus?: HTMLElement | null): Promise<void> {
    this.busy = false;
    this.pending = null;
    await this.updateComplete;
    // A control that went away (a deleted card, a closed form) hands focus to the page's first card.
    const target = returnFocus?.isConnected
      ? returnFocus
      : this.querySelector<HTMLElement>('[data-card], [data-add]');
    target?.focus();
    this.#timer = setTimeout(() => {
      void this.#load();
    }, this.#pollDelay());
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
    return html`<button class="dl-switch dl-switch--dense" type="button" role="switch" data-switch=${connection.connectionId}
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
      <input class=${styles.input} .value=${connection.label}
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

  #endpointField(connection: Connection): TemplateResult {
    if (connection.authentication !== 'none' || this.editingEndpoint !== connection.connectionId) {
      return html`<div class=${styles.field}>
        <span class=${styles.fieldLabel}>${msg('Endpoint', {id: 'connections.endpoint'})}</span>
        <span class=${styles.endpointRow}>
          <span class=${styles.endpoint}>${connection.endpoint}</span>
          ${connection.authentication === 'none' ? html`<button class="dl-btn" type="button"
            @click=${() => {
              this.editingEndpoint = connection.connectionId;
            }}>${msg('Change endpoint', {id: 'connections.changeEndpoint'})}</button>` : nothing}
        </span>
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
        <button class="primary-btn" type="button"
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

  /**
   * The credential, in two steps: what it is, and — for a static token — which header carries it.
   *
   * `Authorization: Bearer` is the default answer, not a fourth kind, so a service that wants a
   * vendor header is the same form with one radio moved. The name and prefix shown here are the
   * stored credential's own, read back from the server: they are not secrets, and a reader who
   * reloaded the page still needs to know whether this Connection is sending TAT or UAT.
   */
  #authBlock(connection: Connection, auth: Authentication): TemplateResult {
    if (auth === 'oauth') {
      return html`<div class=${styles.stackTight}>
        <span class=${styles.inline}>
          <button class="dl-btn" type="button"
            @click=${(event: Event) => {
              void this.#beginAuthorization(connection, event.currentTarget as HTMLElement);
            }}>${msg('Authorize with OAuth', {id: 'connections.oauthBegin'})}</button>
          <span class="${shared.hint} ${styles.inlineHint}">${msg(
            'Authorization must finish in this session. Tokens refresh automatically inside the consented scopes.',
            {id: 'connections.oauthRestart'},
          )}</span>
        </span>
      </div>`;
    }
    if (auth !== 'none') {
      return this.#staticTokenBlock(connection, auth);
    }
    return html`<p class=${shared.hint}>${msg(
      'No credential. Anyone who can reach this URL can use it.',
      {id: 'connections.noCredential'},
    )}</p>`;
  }

  /** One token, and the header the reader says it travels in: the default, or one of their own. */
  #staticTokenBlock(connection: Connection, auth: Authentication): TemplateResult {
    const custom = auth === 'header';
    const {header, prefix} = this.#credentialDraft(connection);
    const stored = connection.credentialHeader;
    return html`<div class=${styles.stackTight}>
      <div class=${styles.transportChoices}>
        <label class=${styles.transportChoice}>
          <input type="radio" name="transport-${connection.connectionId}" value="bearer"
            .checked=${!custom} data-transport="default" data-connection=${connection.connectionId}
            @change=${() => {
              this.draftAuth = {...this.draftAuth, [connection.connectionId]: 'bearer'};
            }}>
          <span>${msg('In Authorization: Bearer', {id: 'connections.transportDefault'})}</span>
        </label>
        <label class=${styles.transportChoice}>
          <input type="radio" name="transport-${connection.connectionId}" value="header"
            .checked=${custom} data-transport="custom" data-connection=${connection.connectionId}
            @change=${() => {
              this.draftAuth = {...this.draftAuth, [connection.connectionId]: 'header'};
            }}>
          <span>${msg('In a header of my own', {id: 'connections.transportCustom'})}</span>
        </label>
      </div>
      ${custom ? html`<span class=${styles.inline}>
        <label class="${styles.field} ${styles.headerName}">
          <span class=${styles.fieldLabel}>${msg('Credential header name', {id: 'connections.credentialHeaderName'})}</span>
          <input class=${styles.input} maxlength="64" autocomplete="off" placeholder="X-Lark-MCP-TAT" required
            data-credential-header=${connection.connectionId} .value=${header}>
        </label>
        <label class="${styles.field} ${styles.headerValue}">
          <span class=${styles.fieldLabel}>${msg('Value prefix', {id: 'connections.credentialHeaderPrefix'})}</span>
          <input class=${styles.input} maxlength="64" autocomplete="off" placeholder="Bearer "
            data-credential-prefix=${connection.connectionId} .value=${prefix}>
        </label>
      </span>` : nothing}
      ${custom && stored === null && connection.authentication === 'header'
        ? html`<span class=${shared.hint} role="status">${msg(
          'This deployment cannot read its key ring, so the stored header name cannot be shown here.',
          {id: 'connections.credentialUnreadable'},
        )}</span>`
        : nothing}
      <span class=${styles.inline}>
        <input class="${styles.input} ${styles.inlineInput}" type="password" autocomplete="off"
          maxlength="8192" required aria-label=${msg('Credential value', {id: 'connections.credentialToken'})}
          data-credential-token=${connection.connectionId}>
        <button class="dl-btn" type="button"
          @click=${(event: Event) => this.#saveCredential(connection, event.currentTarget as HTMLElement)}
          >${msg('Save credential', {id: 'connections.saveHeader'})}</button>
      </span>
      <span class=${shared.hint}>${msg(
        'Written once and never shown again: the token, and the header it travels in with its prefix. Only the header name and prefix are read back.',
        {id: 'connections.headerHint'},
      )}</span>
    </div>`;
  }

  /** The name and prefix a save would use: what the reader typed, else what is stored, else the
   *  preset's own default for this endpoint -- never a guess at the token. */
  #credentialDraft(connection: Connection): {header: string; prefix: string} {
    const typed = this.draftCredential[connection.connectionId];
    if (typed) return typed;
    if (connection.credentialHeader !== null) {
      return {header: connection.credentialHeader, prefix: connection.credentialPrefix ?? ''};
    }
    // A stored credential the deployment cannot read is said, never guessed: offering the
    // preset's name here would read as a read-back beside the sentence saying it cannot be read.
    if (this.#storedCredentialUnreadable(connection)) return {header: '', prefix: ''};
    return this.#presetCredentialFor(connection) ?? {header: '', prefix: ''};
  }

  #saveCredential(connection: Connection, returnFocus: HTMLElement): void {
    const token = this.querySelector<HTMLInputElement>(`[data-credential-token="${connection.connectionId}"]`);
    if (!token?.reportValidity()) return;
    const value = token.value;
    if (this.#authOf(connection) === 'header') {
      const field = this.querySelector<HTMLInputElement>(`[data-credential-header="${connection.connectionId}"]`);
      const prefix = this.querySelector<HTMLInputElement>(`[data-credential-prefix="${connection.connectionId}"]`);
      if (!field?.reportValidity()) return;
      this.draftCredential = {...this.draftCredential, [connection.connectionId]: {
        header: field.value, prefix: prefix?.value ?? '',
      }};
      token.value = '';
      void this.#command({
        kind: 'header',
        connectionId: connection.connectionId,
        header: field.value,
        prefix: prefix?.value ?? '',
        token: value,
        endpoint: connection.endpoint,
      }, returnFocus);
      return;
    }
    token.value = '';
    void this.#command({
      kind: 'bearer',
      connectionId: connection.connectionId,
      bearer: value,
      endpoint: connection.endpoint,
    }, returnFocus);
  }

  /**
   * The header a shipped preset states for an endpoint, as the default a new credential starts
   * from. The endpoint decides it, not the owner's label: renaming a Connection does not change
   * how the service behind it wants a credential sent. A credential the owner already used in
   * this session wins over it, and the stored one is sealed, so this is only ever a default for
   * the next save.
   */
  #presetCredentialFor(connection: Connection): {header: string; prefix: string} | null {
    const preset = this.view?.presets.find((item) => item.endpoint === connection.endpoint);
    return preset ? {header: preset.credentialHeader, prefix: preset.credentialPrefix} : null;
  }

  /** The rows a card is editing: its own draft once touched, the Connection's statement before. */
  #headerRows(connection: Connection): EndpointHeaderPair[] {
    return this.draftHeaders[connection.connectionId]
      ?? connection.endpointHeaders.map(([name, value]) => [name, value] as EndpointHeaderPair);
  }

  #setHeaderRows(connection: Connection, rows: EndpointHeaderPair[]): void {
    this.draftHeaders = {...this.draftHeaders, [connection.connectionId]: rows};
  }

  #withoutHeader(rows: EndpointHeaderPair[], index: number): EndpointHeaderPair[] {
    return rows.filter((_, position) => position !== index);
  }

  /**
   * One stated header as the form states it: its name, its value, and the way to drop it.
   *
   * ``scope`` is what the row belongs to -- a Connection id, or the create form -- so both forms
   * state a header the same way and a test reaches either by the same names.
   */
  #statedHeaderRow(
    row: EndpointHeaderPair,
    index: number,
    scope: string,
    write: (position: 0 | 1, value: string) => void,
    remove: () => void,
    colliding = false,
  ): TemplateResult {
    return html`<span class=${styles.headerRow}>
      <input class="${styles.input} ${styles.headerName} ${colliding ? styles.inputInvalid : ''}"
        maxlength="64" autocomplete="off" aria-invalid=${String(colliding)}
        aria-label=${msg('Header name', {id: 'connections.statedHeaderName'})}
        data-header-name=${scope} data-index=${index} .value=${row[0]}
        @input=${(event: Event) => write(0, String((event.target as HTMLInputElement).value))}>
      <input class="${styles.input} ${styles.headerValue}" maxlength="1024" autocomplete="off"
        aria-label=${msg('Header value', {id: 'connections.statedHeaderValue'})}
        data-header-value=${scope} data-index=${index} .value=${row[1]}
        @input=${(event: Event) => write(1, String((event.target as HTMLInputElement).value))}>
      <button class="dl-btn" type="button"
        aria-label=${msg('Remove header', {id: 'connections.removeHeader'})}
        @click=${remove}>${msg('Remove', {id: 'connections.removeHeaderAction'})}</button>
    </span>`;
  }

  /** The credential's own header, shown where the request is assembled and edited above it. */
  #credentialRow(connection: Connection): TemplateResult {
    const header = this.#credentialHeaderName(connection);
    const name = header === ''
      ? msg('The header this deployment cannot read', {id: 'connections.credentialUnknownHeader'})
      : header;
    // The link is only a link when the control it names is on the page: a stored OAuth grant
    // has no transport to edit, and the credential section above says where it stands.
    const editable = this.#authOf(connection) !== 'oauth';
    return html`<div class=${styles.credentialRow}>
      <span class=${styles.tag}>${msg('credential', {id: 'connections.credentialTag'})}</span>
      <span class=${styles.credentialName}>${name}:</span>
      <span class=${styles.credentialValue}>••••••</span>
      <span class=${styles.credentialOwn}>${editable
        ? html`<a href="#credential-${connection.connectionId}"
          @click=${(event: Event) => {
            event.preventDefault();
            this.querySelector<HTMLElement>(
              `[data-transport][data-connection="${connection.connectionId}"]`,
            )?.focus();
          }}>${msg('edit above', {id: 'connections.credentialEditAbove'})}</a>`
        : nothing}</span>
    </div>`;
  }

  /**
   * The header this Connection's credential travels in, as the reader or the server states it.
   *
   * `bearer` and `oauth` are the default instance and travel in `Authorization`, so they name
   * themselves. A `header` credential names its own, and when the deployment cannot read it the
   * answer is empty -- the card says it cannot read it rather than guessing.
   */
  #credentialHeaderName(connection: Connection): string {
    if (this.#authOf(connection) !== 'header') return 'Authorization';
    return this.#credentialDraft(connection).header.trim();
  }

  /** A stored header credential this deployment cannot read: not the same as holding none. */
  #storedCredentialUnreadable(connection: Connection): boolean {
    return connection.authentication === 'header'
      && connection.credentialHeader === null
      && connection.status !== 'revoked';
  }

  /** What a stated header is, said the same way wherever one is edited. */
  #headersHint(): TemplateResult {
    return html`<span class=${shared.hint}>${msg(
      'Headers every request to this endpoint states. They are configuration, not secrets: a credential is stored separately and never here, and a server may filter what it offers by one of them.',
      {id: 'connections.endpointHeadersHint'},
    )}</span>`;
  }

  #endpointHeadersEditor(connection: Connection): TemplateResult {
    const rows = this.#headerRows(connection);
    // The credential's own header is known here, so a row that would collide with it is marked
    // as the reader types: the server still refuses it, and this is the same fact said earlier.
    const credential = this.#credentialHeaderName(connection);
    const collides = (name: string): boolean => (
      credential !== '' && name.trim().toLowerCase() === credential.toLowerCase()
    );
    return html`<div class=${styles.headerSection}>
      <div class=${styles.stackTight}>
        <span class=${styles.fieldLabel}>${msg('Request headers', {id: 'connections.requestHeaders'})}</span>
        ${this.#headersHint()}
        ${connection.authentication === 'none' || connection.status === 'revoked'
          ? nothing
          : this.#credentialRow(connection)}
        ${rows.some(([name]) => collides(name))
          ? html`<span class=${styles.collisionNote} role="status">${msg(
            'A row below names the credential\'s own header; the credential is sent last and would replace it.',
            {id: 'connections.headerCollision'},
          )}</span>`
          : nothing}
        ${rows.map((row, index) => this.#statedHeaderRow(
          row,
          index,
          connection.connectionId,
          (position, value) => {
            const next = [...this.#headerRows(connection)];
            const current = next[index] ?? ['', ''];
            next[index] = position === 0 ? [value, current[1]] : [current[0], value];
            this.#setHeaderRows(connection, next);
          },
          () => this.#setHeaderRows(connection, this.#withoutHeader(rows, index)),
          collides(row[0]),
        ))}
        <div class=${styles.actionsEnd}>
          <button class="dl-btn" type="button"
            @click=${() => this.#setHeaderRows(connection, [...rows, ['', '']])}
            >${msg('Add header', {id: 'connections.addHeader'})}</button>
          <button class="primary-btn" type="button" ?disabled=${this.busy}
            @click=${(event: Event) => {
              const stated = this.#readHeaderRows(rows);
              if (stated === null) return;
              void this.#command({
                kind: 'edit',
                connectionId: connection.connectionId,
                label: connection.label,
                endpoint: connection.endpoint,
                endpointHeaders: stated,
              }, event.currentTarget as HTMLElement).then((saved) => {
                if (!saved) return;  // A refusal keeps what the owner typed, to correct and retry.
                // What is published is the server's statement: re-seed the rows from it.
                const {[connection.connectionId]: _saved, ...rest} = this.draftHeaders;
                this.draftHeaders = rest;
              });
            }}>${msg('Save headers', {id: 'connections.saveHeaders'})}</button>
        </div>
      </div>
    </div>`;
  }

  /**
   * What the form states: a row that names nothing and states nothing is dropped, and one that
   * states a value under no name is refused rather than silently left out of the save.
   */
  #readHeaderRows(rows: EndpointHeaderPair[]): EndpointHeaderPair[] | null {
    const named = rows.filter(([name]) => name.trim() !== '');
    if (rows.some(([name, value]) => name.trim() === '' && value.trim() !== '')) {
      this.error = true;
      this.refusal = msg('A stated header needs a name.', {
        id: 'connections.statedHeaderNeedsName',
      });
      return null;
    }
    return named.map(([name, value]) => [name, value]);
  }

  async #beginAuthorization(connection: Connection, returnFocus: HTMLElement): Promise<void> {
    const signal = this.lifetime;
    if (!this.view || this.busy || signal.aborted) return;
    this.busy = true;
    this.pending = connection.connectionId;
    this.error = false;
    this.refusal = null;
    this.#generation++;
    if (this.#timer) clearTimeout(this.#timer);
    try {
      const url = await beginConnectionAuthorization(
        this.view.revision, connection.connectionId, connection.endpoint, signal,
      );
      if (signal.aborted) return;
      // One click is the whole gesture: the provider's consent page is where the reader is
      // headed, so the card sends them there. A reader the browser brings back -- Back from the
      // provider, or a fresh page -- starts again from this same control, and starting again
      // supersedes the attempt they left rather than resuming it behind their back.
      this.navigateTo(url);
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
    return html`<article class=${shared.card}>
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
        <div class=${styles.fields}>
          ${this.#labelField(connection)}
          ${this.#endpointField(connection)}
        </div>
        ${this.#declarationCost(connection)}
        <div class=${styles.stackTight}>
          <span class=${styles.fieldLabel}>${msg('Credential', {id: 'connections.credential'})}</span>
          <div class=${styles.segmented}>
            ${CREDENTIAL_TYPES.map((type) => html`
              <button class="${styles.segment} ${type === credentialType(auth) ? styles.segmentActive : ''}" type="button"
                aria-pressed=${String(type === credentialType(auth))}
                @click=${() => {
                  // Choosing a kind keeps the transport already chosen: a reader who moves from
                  // OAuth back to a token does not silently lose the header they had stated.
                  const chosen = type === STATIC_TOKEN
                    ? (auth === 'header' ? 'header' : 'bearer')
                    : type;
                  this.draftAuth = {...this.draftAuth, [connection.connectionId]: chosen};
                }}>${CREDENTIAL_TYPE_LABEL[type]()}</button>`)}
          </div>
          ${this.#authBlock(connection, auth)}
        </div>
        ${this.#endpointHeadersEditor(connection)}
        <div class="${shared.dangerRow} ${styles.deleteRow}">
          <span class=${shared.hint}>${msg(
            'Deleting removes the endpoint, the label and the stored credential; to pause it, switch it off instead.',
            {id: 'connections.deleteHint'},
          )}</span>
          <button class="dl-btn dl-btn-danger-text" type="button" data-delete=${connection.connectionId}
            @click=${() => {
              void this.#requestDelete(connection);
            }}>${msg('Delete…', {id: 'connections.delete'})}</button>
        </div>
      </div>` : nothing}
    </article>`;
  }

  /**
   * A preset only ever fills the form: it selects no endpoint policy, stores no credential, and
   * grants no authority -- creating the Connection stays the one act that does any of that.
   *
   * What it fills is everything the owner would otherwise have to know: the endpoint, the
   * non-secret headers that endpoint needs stated, and the tab with the header a credential
   * would travel in. The credential itself is never part of a preset.
   */
  #applyPreset(preset: Preset): void {
    const label = this.querySelector<HTMLInputElement>('[data-new-label]');
    const endpoint = this.querySelector<HTMLInputElement>('[data-new-endpoint]');
    if (label) label.value = preset.label;
    if (endpoint) endpoint.value = preset.endpoint;
    this.#presetAuth = preset.defaultAuthentication;
    this.newHeaders = preset.endpointHeaders.map(([name, value]) => [name, value]);
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
    const stated = this.#readHeaderRows(this.newHeaders);
    if (stated === null) return;
    const view = await this.#command(
      {kind: 'create', label, endpoint, endpointHeaders: stated},
      returnFocus,
    );
    const created = view?.connections.find((connection) => !known.has(connection.connectionId));
    if (!created) return;
    // The form's statement is now the Connection's, so the next create starts from nothing.
    this.newHeaders = [];
    if (!presetAuth) return;
    this.draftAuth = {...this.draftAuth, [created.connectionId]: presetAuth};
    this.openCard = created.connectionId;
  }

  /** The create form's rows are stated under this scope until a Connection holds them. */
  static readonly #CREATE_SCOPE = 'new';

  #createForm(): TemplateResult {
    const presets = this.view?.presets ?? [];
    return html`<div class=${styles.stackTight}>
      ${presets.length === 0 ? nothing : html`
        <div class=${styles.presetRow} role="group"
          aria-label=${msg('Presets', {id: 'connections.presets'})}>
          ${presets.map((preset) => html`
            <button class=${styles.presetChip} type="button"
              aria-label=${msg(str`Use the ${preset.label} preset`, {id: 'connections.usePreset'})}
              @click=${() => {
                this.#applyPreset(preset);
              }}>${preset.label}</button>`)}
        </div>`}
      <div class=${styles.fields}>
        <label class=${styles.field}>
          <span class=${styles.fieldLabel}>${msg('Label', {id: 'connections.label'})}</span>
          <input class=${styles.input} data-new-label required maxlength="100">
        </label>
        <label class=${styles.field}>
          <span class=${styles.fieldLabel}>${msg('Endpoint', {id: 'connections.endpoint'})}</span>
          <input class=${styles.input} data-new-endpoint type="url" required maxlength="2048"
            placeholder="https://example.com/mcp">
        </label>
      </div>
      <span class=${shared.hint}>${msg(
        'Not one of the presets? Create it with the endpoint alone: the card then asks two things — the credential (none, a static token in Authorization: Bearer or in a header of its own, or OAuth) and the headers every request to this endpoint states — and says what the server answered once you save a credential or switch the Connection on.',
        {id: 'connections.manualPath'},
      )}</span>
      <div class=${styles.stackTight}>
        <span class=${styles.fieldLabel}>${msg('Request headers', {id: 'connections.requestHeaders'})}</span>
        ${this.#headersHint()}
        ${this.newHeaders.map((row, index) => this.#statedHeaderRow(
          row,
          index,
          DlSettingsConnections.#CREATE_SCOPE,
          (position, value) => {
            const next = [...this.newHeaders];
            const current = next[index] ?? ['', ''];
            next[index] = position === 0 ? [value, current[1]] : [current[0], value];
            this.newHeaders = next;
          },
          () => {
            this.newHeaders = this.newHeaders.filter((_, position) => position !== index);
          },
        ))}
        <div class=${styles.actionsEnd}>
          <button class="dl-btn" type="button" data-new-header-add
            @click=${() => {
              this.newHeaders = [...this.newHeaders, ['', '']];
            }}>${msg('Add header', {id: 'connections.addHeader'})}</button>
        </div>
      </div>
      <div class=${styles.footer}>
        <span class=${shared.hint}>${msg('Starts inactive.', {id: 'connections.startsInactive'})}</span>
        <button class="primary-btn" type="button" ?disabled=${this.busy}
          @click=${(event: Event) => {
            const label = this.querySelector<HTMLInputElement>('[data-new-label]');
            const endpoint = this.querySelector<HTMLInputElement>('[data-new-endpoint]');
            if (!label?.reportValidity() || !endpoint?.reportValidity()) return;
            this.adding = false;
            void this.#create(label.value, endpoint.value, event.currentTarget as HTMLElement);
          }}>${msg('Add connection', {id: 'connections.add'})}</button>
      </div>
    </div>`;
  }

  /** The small MCP label and how many of the Connections are on. */
  #summary(): TemplateResult {
    const total = this.view?.connections.length ?? 0;
    const enabled = this.view?.connections.filter((connection) => connection.enabled).length ?? 0;
    const attention = this.view?.connections.filter((connection) => (
      connection.enabled && this.#statusOf(connection).tone === styles.dotProblem
    )).length ?? 0;
    return html`<p class=${styles.summary}>
      <span class=${styles.summaryName}>MCP</span>
      <span class=${styles.summaryMeta}>${total === 0
        ? msg('add a connection', {id: 'connections.summaryEmpty'})
        : msg(str`${enabled} of ${total} enabled`, {id: 'connections.summary'})}</span>
      ${attention > 0
        ? html`<span class="dl-sr-only">${attention === 1
          ? msg('1 connection needs attention', {id: 'connections.attentionOne'})
          : msg(str`${attention} connections need attention`, {id: 'connections.attentionMany'})}</span>`
        : nothing}
    </p>`;
  }

  protected override render(): TemplateResult {
    const connections = this.view?.connections ?? [];
    return html`<div class=${styles.root}>
      ${this.error ? html`<p class=${shared.note} role="alert">${this.refusal ?? msg(
        'Connection request failed or the revision changed. Reload and retry.',
        {id: 'connections.error'},
      )}</p>
      <button class="dl-btn" type="button" ?disabled=${this.busy} @click=${() => {
        void this.#load();
      }}>${msg('Reload', {id: 'connections.reload'})}</button>` : nothing}
      ${!this.view ? html`<p class=${shared.hint} role="status">${msg(
        'Loading Connections…', {id: 'connections.loading'})}</p>` : this.#summary()}
      ${connections.length === 0 && this.view
        ? html`<p class=${shared.hint}>${msg('No MCP connections yet.', {id: 'connections.empty'})}</p>`
        : nothing}
      <div class=${styles.cards}>
        ${repeat(connections, (connection) => connection.connectionId, (connection) => this.#card(connection))}
        <article class=${shared.card}>
          <button class=${styles.addRow} type="button" data-add @click=${() => {
            this.adding = !this.adding;
          }}>${icon('add', {size: 'sm'})}${msg('Add MCP connection', {id: 'connections.new'})}</button>
          ${this.adding ? html`<div class=${styles.cardBody}>${this.#createForm()}</div>` : nothing}
        </article>
      </div>

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
    </div>`;
  }
}
customElements.define('dl-settings-connections', DlSettingsConnections);
declare global {interface HTMLElementTagNameMap {'dl-settings-connections': DlSettingsConnections;}}
