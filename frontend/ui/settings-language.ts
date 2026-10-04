// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Settings → Language: follow the browser, or pick English or Chinese. */

import {msg, updateWhenLocaleChanges} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {currentLanguagePreference, setLanguagePreference} from '../i18n/locale.ts';
import {type LanguagePreference, parseLanguagePreference} from '../lib/language.ts';
import {LightElement} from '../lib/lit-host.ts';
import shared from '../styles/settings-page.module.css';
import {reportSettingsSummary} from './settings-summary.ts';

const PREFERENCES: readonly LanguagePreference[] = ['auto', 'en', 'zh'];

/** What a preference is called. The Chinese name is always in Chinese, so a reader who cannot read
 *  the current language can still find it. */
export function languageLabel(preference: LanguagePreference): string {
  switch (preference) {
    case 'auto':
      return msg('Automatic', {id: 'settings.language.automatic'});
    case 'en':
      return msg('English', {id: 'settings.language.english'});
    case 'zh':
      return '中文';
  }
}

export class DlSettingsLanguage extends LightElement {
  static properties = {
    preference: {state: true},
  };

  declare preference: LanguagePreference;

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.preference = currentLanguagePreference();
  }

  protected override updated(changed: PropertyValues<this>): void {
    if (changed.has('preference')) {
      reportSettingsSummary(this, {section: 'language', preference: this.preference});
    }
  }

  #choose = (event: Event): void => {
    const input = event.currentTarget as HTMLInputElement;
    const preference = parseLanguagePreference(input.value);
    this.preference = preference;
    void setLanguagePreference(preference);
  };

  #choice(value: LanguagePreference): TemplateResult {
    const caption = value === 'auto'
      ? msg('Follows the browser language', {id: 'settings.language.automaticHint'})
      : undefined;
    return html`
      <label class="dl-dialog-checkbox dl-dialog-checkbox--row">
        <input type="radio" name="language" value=${value}
               .checked=${this.preference === value}
               aria-labelledby="language-${value}-label"
               aria-describedby=${caption ? `language-${value}-caption` : nothing}
               @change=${this.#choose}>
        <span class=${shared.rowText}>
          <span id="language-${value}-label" class=${shared.rowLabel}>${languageLabel(value)}</span>
          ${caption ? html`<span id="language-${value}-caption" class=${shared.rowCaption}>${caption}</span>` : nothing}
        </span>
      </label>`;
  }

  protected override render(): TemplateResult {
    return html`
      <div class=${shared.stack}>
        <div class="${shared.card} ${shared.divided}" role="radiogroup"
             aria-label=${msg('Language', {id: 'settings.language'})}>
          ${PREFERENCES.map((value) => this.#choice(value))}
        </div>
      </div>`;
  }
}

customElements.define('dl-settings-language', DlSettingsLanguage);

declare global {
  interface HTMLElementTagNameMap {
    'dl-settings-language': DlSettingsLanguage;
  }
}
