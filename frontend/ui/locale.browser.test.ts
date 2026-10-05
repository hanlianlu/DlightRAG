// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, updateWhenLocaleChanges} from '@lit/localize';
import {expect} from '@esm-bundle/chai';
import {LitElement, html} from 'lit';
import {memorySettings, mountSettings, openSettings, wire} from '../testing/settings.ts';
import {
  getLocale,
  initializeLanguagePreference,
  setLanguagePreference,
} from '../i18n/locale.ts';
import {LANGUAGE_STORAGE_KEY} from '../lib/language.ts';
import {LightElement} from '../lib/lit-host.ts';
import {radioNamed, waitFor} from '../testing/dom.ts';

class LocaleProbe extends LitElement {
  constructor() {
    super();
    updateWhenLocaleChanges(this);
  }

  protected override render(): unknown {
    return html`<p>${msg('Loading DlightRAG…', {id: 'bootstrap.loading'})}</p>`;
  }
}

// Registers nothing: the base class owns following the language.
class LightProbe extends LightElement {
  protected override render(): unknown {
    return html`<p>${msg('Loading DlightRAG…', {id: 'bootstrap.loading'})}</p>`;
  }
}

customElements.define('dl-locale-probe', LocaleProbe);
customElements.define('dl-light-probe', LightProbe);

beforeEach(() => {
  window.localStorage.removeItem(LANGUAGE_STORAGE_KEY);
});

afterEach(async () => {
  await setLanguagePreference('auto');
  window.localStorage.removeItem(LANGUAGE_STORAGE_KEY);
  document.body.replaceChildren();
});

it('initializes to the source locale when no preference is stored', async () => {
  await initializeLanguagePreference();

  expect(getLocale()).to.equal('en');
  expect(document.documentElement.lang).to.equal('en');
  expect(window.localStorage.getItem(LANGUAGE_STORAGE_KEY)).to.equal(null);
});

it('resolves a stored zh preference and renders localized content', async () => {
  window.localStorage.setItem(LANGUAGE_STORAGE_KEY, 'zh');
  await initializeLanguagePreference();

  expect(getLocale()).to.equal('zh');
  expect(document.documentElement.lang).to.equal('zh');

  const probe = new LocaleProbe();
  document.body.appendChild(probe);
  await probe.updateComplete;

  expect(probe.shadowRoot?.textContent).to.contain('正在加载 DlightRAG…');
});

it('a light element redraws in the new language without registering for it', async () => {
  const probe = new LightProbe();
  document.body.appendChild(probe);
  await probe.updateComplete;
  expect(probe.textContent).to.contain('Loading DlightRAG…');

  await setLanguagePreference('zh');

  await waitFor(() => probe.textContent!.includes('正在加载 DlightRAG…'));
});

it('switching back restores source strings and clears the stored preference', async () => {
  await setLanguagePreference('zh');
  expect(getLocale()).to.equal('zh');

  await setLanguagePreference('auto');

  expect(window.localStorage.getItem(LANGUAGE_STORAGE_KEY)).to.equal(null);
  expect(getLocale()).to.equal('en');
  expect(document.documentElement.lang).to.equal('en');
});

it('settings language radios apply and persist the preference', async () => {
  const originalFetch = window.fetch;
  window.fetch = wire({'GET /web/api/memory/settings': () => memorySettings(false)}).fetch;
  try {
    const {settings} = mountSettings();
    await openSettings(settings, 'language');

    const radio = radioNamed(settings, '中文')!;
    radio.checked = true;
    radio.dispatchEvent(new Event('change'));
    await waitFor(() => getLocale() === 'zh');
    await settings.updateComplete;

    expect(window.localStorage.getItem(LANGUAGE_STORAGE_KEY)).to.equal('zh');
    expect(getLocale()).to.equal('zh');
    expect(document.documentElement.lang).to.equal('zh');
    // The dialog speaks the new language at once, and the navigation says which one is chosen.
    expect(settings.querySelector('#settings-title')!.textContent).to.equal('设置');
    const language = settings.querySelector('nav [data-section="language"]')!;
    await waitFor(() => language.textContent!.includes('中文'));
    expect(language.textContent).to.contain('语言');
  } finally {
    window.fetch = originalFetch;
  }
});
