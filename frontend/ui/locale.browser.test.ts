// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, updateWhenLocaleChanges} from '@lit/localize';
import {expect} from '@esm-bundle/chai';
import {LitElement, html} from 'lit';
import './settings.ts';
import type {DlSettingsDialog} from './settings.ts';
import {
  getLocale,
  initializeLanguagePreference,
  setLanguagePreference,
} from '../i18n/locale.ts';
import {LANGUAGE_STORAGE_KEY} from '../lib/language.ts';

class LocaleProbe extends LitElement {
  constructor() {
    super();
    updateWhenLocaleChanges(this);
  }

  protected override render(): unknown {
    return html`<p>${msg('Loading DlightRAG…', {id: 'bootstrap.loading'})}</p>`;
  }
}

customElements.define('dl-locale-probe', LocaleProbe);

beforeEach(() => {
  window.localStorage.removeItem(LANGUAGE_STORAGE_KEY);
});

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error('condition did not become true');
}

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

it('switching back restores source strings and clears the stored preference', async () => {
  await setLanguagePreference('zh');
  expect(getLocale()).to.equal('zh');

  await setLanguagePreference('auto');

  expect(window.localStorage.getItem(LANGUAGE_STORAGE_KEY)).to.equal(null);
  expect(getLocale()).to.equal('en');
  expect(document.documentElement.lang).to.equal('en');
});

it('settings language radios apply and persist the preference', async () => {
  const settings = document.createElement('dl-settings-dialog') as DlSettingsDialog;
  document.body.appendChild(settings);
  await settings.updateComplete;

  const radio = settings.querySelector<HTMLInputElement>(
    '#language-options input[value="zh"]',
  )!;
  radio.checked = true;
  radio.dispatchEvent(new Event('change'));
  await waitFor(() => getLocale() === 'zh');
  await settings.updateComplete;

  expect(window.localStorage.getItem(LANGUAGE_STORAGE_KEY)).to.equal('zh');
  expect(getLocale()).to.equal('zh');
  expect(document.documentElement.lang).to.equal('zh');
});
