// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {setLanguagePreference} from '../i18n/locale.ts';
import {LANGUAGE_STORAGE_KEY} from '../lib/language.ts';

/** The shipped sign-in page, so the test reads the markup users get. */
async function loginPage(): Promise<Document> {
  const response = await fetch(new URL('../login.html', import.meta.url));
  if (!response.ok) throw new Error(`login.html: HTTP ${response.status}`);
  return new DOMParser().parseFromString(await response.text(), 'text/html');
}

it('localizes the sign-in page and never shows the error parameter text', async () => {
  const originalUrl = window.location.href;
  const originalTitle = document.title;
  window.localStorage.setItem(LANGUAGE_STORAGE_KEY, 'zh');
  const page = await loginPage();
  document.title = page.title;
  document.body.replaceChildren(...[...page.body.childNodes].map((node) => document.importNode(node, true)));
  const url = new URL(originalUrl);
  url.searchParams.set('next', '/web/conversations/c-1');
  url.searchParams.set('error', 'Call 555-0100 to unlock your account');
  window.history.replaceState(null, '', url);
  try {
    await import('./login.ts');

    expect(document.title).to.equal('登录 · DlightRAG');
    expect(document.querySelector('label[for="token"]')?.textContent).to.equal('访问令牌');
    expect(document.querySelector('form button[type="submit"]')?.textContent).to.equal('登录');
    const error = document.querySelector<HTMLElement>('.file-error')!;
    expect(error.hidden).to.equal(false);
    expect(error.textContent).to.equal('身份验证失败。请检查令牌后重试。');
    expect(document.body.textContent).not.to.contain('555-0100');
    expect(document.querySelector<HTMLInputElement>('input[name="next"]')!.value)
      .to.equal('/web/conversations/c-1');
  } finally {
    window.history.replaceState(null, '', originalUrl);
    document.title = originalTitle;
    await setLanguagePreference('auto');
    window.localStorage.removeItem(LANGUAGE_STORAGE_KEY);
    document.body.replaceChildren();
  }
});
