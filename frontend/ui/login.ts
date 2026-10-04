// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import '../design-system/index.css';
import '../styles/app.css';
import '../styles/layout.css';

import {msg} from '@lit/localize';
import {initializeLanguagePreference} from '../i18n/locale.ts';

// The language preference must resolve before any localized content on the
// login page renders, mirroring the main application entry.
await initializeLanguagePreference();

document.title = msg('Sign in · DlightRAG', {id: 'login.title'});
const label = document.querySelector<HTMLLabelElement>('label[for="token"]');
if (label) label.textContent = msg('Access token', {id: 'login.token'});
const submit = document.querySelector<HTMLButtonElement>('button[type="submit"]');
if (submit) submit.textContent = msg('Sign in', {id: 'login.submit'});

const params = new URLSearchParams(window.location.search);
const next = params.get('next');
const nextInput = document.querySelector<HTMLInputElement>('input[name="next"]');
if (nextInput && next?.startsWith('/web/')) nextInput.value = next;

// The server only ever reports a failed sign-in here. Its text is not shown:
// a link could carry any words in this parameter.
const errorElement = document.querySelector<HTMLElement>('.file-error');
if (params.has('error') && errorElement) {
  errorElement.textContent = msg('Authentication failed. Check the token and try again.', {
    id: 'login.failed',
  });
  errorElement.hidden = false;
}
