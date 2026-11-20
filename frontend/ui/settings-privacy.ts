// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Settings → Privacy: the owner's one switch for letting their conversations improve DlightRAG.
 *
 * The permission is off until the owner turns it on and is theirs alone, so the page holds one switch
 * and says what it covers. No switch is drawn until the setting has been read, because an unchecked
 * one would claim "off" for a state nobody read. Every press is committed pessimistically, so the
 * switch shows what the server stores.
 */

import {msg, str} from '@lit/localize';
import {html, type PropertyValues, type TemplateResult} from 'lit';
import {getPrivacySettings, putPrivacySettings, type PrivacySettings} from '../api/privacy.ts';
import {getLocale} from '../i18n/locale.ts';
import {shortDate} from '../lib/date-format.ts';
import {LightElement} from '../lib/lit-host.ts';
import shared from '../styles/settings-page.module.css';
import styles from '../styles/settings-privacy.module.css';
import {commitSwitch, loadState, switchCard} from './settings-parts.ts';
import {reportSettingsSummary} from './settings-summary.ts';

export class DlSettingsPrivacy extends LightElement {
  static properties = {
    settings: {state: true},
    error: {state: true},
    pending: {state: true},
  };

  /** What the server stores for the owner; null until the first read answers. */
  declare settings: PrivacySettings | null;
  /** The first read failed. Once one has succeeded the page never reads again: every command answers with the stored state. */
  declare error: boolean;
  /** The switch's command is in flight, so it cannot be pressed again. */
  declare pending: boolean;

  constructor() {
    super();
    this.settings = null;
    this.error = false;
    this.pending = false;
  }

  override connectedCallback(): void {
    super.connectedCallback();
    void this.#load();
  }

  protected override updated(changed: PropertyValues<this>): void {
    if (changed.has('settings') && this.settings) {
      reportSettingsSummary(this, {section: 'privacy', contribute: this.settings.contribute});
    }
  }

  async #load(): Promise<void> {
    const signal = this.lifetime;
    if (signal.aborted) return;
    this.error = false;
    try {
      const settings = await getPrivacySettings(signal);
      if (signal.aborted) return;
      this.settings = settings;
    } catch {
      if (!signal.aborted) this.error = true;
    }
  }

  #toggle = async (event: Event): Promise<void> => {
    const settings = this.settings;
    if (!settings) return;
    await commitSwitch(this, event, {
      signal: this.lifetime,
      send: (signal) => putPrivacySettings(!settings.contribute, signal),
      accept: (reply) => { this.settings = reply; },
      failure: msg('Could not save privacy settings.', {id: 'settings.privacy.saveFailed'}),
    });
  };

  /** What a contribution covers and what it never does, in the words the owner decides on. */
  #scope(): TemplateResult {
    return html`
      <dl class=${styles.scope}>
        <dt class=${styles.term}>${msg('May include', {id: 'settings.privacy.includesTerm'})}</dt>
        <dd class=${styles.statement}>${msg(
          'Your messages and the agent\'s answers, its reasoning and tool calls, the passages it read, and your feedback on answers with its reasons and notes.',
          {id: 'settings.privacy.includes'},
        )}</dd>
        <dt class=${styles.term}>${msg('Never includes', {id: 'settings.privacy.excludesTerm'})}</dt>
        <dd class=${styles.statement}>${msg(
          'The passwords of your agent accounts and the secrets of your connections.',
          {id: 'settings.privacy.excludes'},
        )}</dd>
        <dt class=${styles.term}>${msg('How it works', {id: 'settings.privacy.worksTerm'})}</dt>
        <dd class=${styles.statement}>${msg(
          'It applies to conversations you start after you turn it on, and stops as soon as you turn it off.',
          {id: 'settings.privacy.works'},
        )}</dd>
      </dl>`;
  }

  /** The sentence under the switch. The unit of consent is the conversation, so while the permission is
   * on it names the day from which conversations count, and reads as true next week as it does today. */
  #caption(settings: PrivacySettings): string {
    if (!settings.contribute) {
      return msg('Off: DlightRAG does not use your conversations to evaluate or train models', {
        id: 'settings.privacy.contributeOff',
      });
    }
    const since = shortDate(settings.since, new Date(), getLocale());
    return msg(str`Conversations started since ${since} may be used to evaluate and train models`, {
      id: 'settings.privacy.contributeSince',
    });
  }

  #page(settings: PrivacySettings): TemplateResult {
    return html`
      <div class=${shared.stack}>
        ${switchCard({
          id: 'privacy-contribute',
          label: msg('Use my conversations to improve DlightRAG', {id: 'settings.privacy.contribute'}),
          caption: this.#caption(settings),
          checked: settings.contribute,
          disabled: this.pending,
          onToggle: this.#toggle,
        })}
        ${this.#scope()}
      </div>`;
  }

  protected override render(): TemplateResult {
    return this.settings ? this.#page(this.settings) : loadState({
      host: this,
      failed: this.error,
      loading: msg('Loading privacy settings…', {id: 'settings.privacy.loading'}),
      failure: msg('Could not load privacy settings.', {id: 'settings.privacy.loadFailed'}),
      retry: msg('Retry', {id: 'settings.privacy.retry'}),
      onRetry: () => this.#load(),
    });
  }
}

customElements.define('dl-settings-privacy', DlSettingsPrivacy);

declare global {
  interface HTMLElementTagNameMap {
    'dl-settings-privacy': DlSettingsPrivacy;
  }
}
