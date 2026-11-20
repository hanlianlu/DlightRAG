// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, updateWhenLocaleChanges} from '@lit/localize';
import {expect} from '@esm-bundle/chai';
import {LitElement, html} from 'lit';
import {memorySettings, mountSettings, openSettings, wire} from '../testing/settings.ts';
import {
  activateLocale,
  getLocale,
  initializeLanguagePreference,
  setLanguagePreference,
  setLocale,
} from '../i18n/locale.ts';
import {defineDesignSystemElements, type DlIconButton} from '../design-system/index.ts';
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

// A control named by the host's `aria-label`, as the product's close and back buttons are.
class IconButtonProbe extends LightElement {
  protected override render(): unknown {
    return html`<dl-icon-button name="close"
      aria-label=${msg('Close panel', {id: 'inspector.closePanel'})}></dl-icon-button>`;
  }
}

customElements.define('dl-locale-probe', LocaleProbe);
customElements.define('dl-light-probe', LightProbe);
customElements.define('dl-icon-button-probe', IconButtonProbe);
defineDesignSystemElements();

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
  expect(document.documentElement.lang).to.equal('zh-Hans');

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

it('an icon button names itself in the new language when the host label changes in place', async () => {
  const probe = new IconButtonProbe();
  document.body.appendChild(probe);
  await probe.updateComplete;
  const host = probe.querySelector<DlIconButton>('dl-icon-button')!;
  const inner = host.shadowRoot!.querySelector('button')!;
  expect(inner.getAttribute('aria-label')).to.equal('Close panel');

  await setLanguagePreference('zh');
  await waitFor(() => host.getAttribute('aria-label') === '关闭面板');
  expect(inner.getAttribute('aria-label')).to.equal('关闭面板');

  await setLanguagePreference('auto');
  await waitFor(() => host.getAttribute('aria-label') === 'Close panel');
  expect(inner.getAttribute('aria-label')).to.equal('Close panel');

  host.removeAttribute('aria-label');
  expect(inner.hasAttribute('aria-label')).to.equal(false);
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
    expect(document.documentElement.lang).to.equal('zh-Hans');
    // The dialog speaks the new language at once, and the navigation says which one is chosen.
    expect(settings.querySelector('#settings-title')!.textContent).to.equal('设置');
    const language = settings.querySelector('nav [data-section="language"]')!;
    await waitFor(() => language.textContent!.includes('中文'));
    expect(language.textContent).to.contain('语言');
  } finally {
    window.fetch = originalFetch;
  }
});

it('moves <html lang> only once the catalog is installed', async () => {
  await setLanguagePreference('en');
  let release!: () => void;
  const gate = new Promise<void>((resolve) => { release = resolve; });

  const switching = activateLocale('zh', async (locale) => {
    await gate;
    await setLocale(locale);
  });
  await new Promise((resolve) => setTimeout(resolve, 0));

  // The catalog is still loading: the page speaks English and says so.
  expect(getLocale()).to.equal('en');
  expect(document.documentElement.lang).to.equal('en');

  release();
  await switching;
  expect(getLocale()).to.equal('zh');
  expect(document.documentElement.lang).to.equal('zh-Hans');
});

it('keeps <html lang> on the language the page speaks when a catalog fails to load', async () => {
  const failing = async (): Promise<void> => { throw new Error('chunk failed to load'); };
  await setLanguagePreference('en');

  await activateLocale('zh', failing);
  expect(getLocale()).to.equal('en');
  expect(document.documentElement.lang).to.equal('en');

  await setLanguagePreference('zh');
  await activateLocale('en', failing);
  expect(getLocale()).to.equal('zh');
  expect(document.documentElement.lang).to.equal('zh-Hans');
});

it('names the script for Chinese and nothing more for English', async () => {
  await setLanguagePreference('zh');
  expect(document.documentElement.lang).to.equal('zh-Hans');
  expect(document.documentElement.matches(':lang(zh)'), 'a :lang(zh) rule still applies').to.equal(true);

  await setLanguagePreference('en');
  expect(document.documentElement.lang).to.equal('en');
});
