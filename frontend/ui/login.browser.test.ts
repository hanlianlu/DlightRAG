// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {setLanguagePreference} from '../i18n/locale.ts';
import {LANGUAGE_STORAGE_KEY} from '../lib/language.ts';

it('localizes the sign-in page and never shows the error parameter text', async () => {
  const originalUrl = window.location.href;
  const originalTitle = document.title;
  window.localStorage.setItem(LANGUAGE_STORAGE_KEY, 'zh');
  document.body.innerHTML = `
    <form method="post" action="/web/login">
      <input type="hidden" name="next" value="/web/">
      <label for="token">Access token</label>
      <input id="token" name="token" type="password">
      <div class="file-error" role="alert" hidden></div>
      <button class="primary-btn" type="submit">Sign in</button>
    </form>`;
  const url = new URL(originalUrl);
  url.searchParams.set('next', '/web/conversations/c-1');
  url.searchParams.set('error', 'Call 555-0100 to unlock your account');
  window.history.replaceState(null, '', url);
  try {
    await import('./login.ts');

    expect(document.title).to.equal('登录 · DlightRAG');
    expect(document.querySelector('label')?.textContent).to.equal('访问令牌');
    expect(document.querySelector('button')?.textContent).to.equal('登录');
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
