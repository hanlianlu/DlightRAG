// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import '../tokens/utopia.css';
import '../styles/global.css';
import '../styles/layout.css';

import {initializeLanguagePreference} from '../i18n/locale.ts';

// The language preference must resolve before any localized content on the
// login page renders, mirroring the main application entry.
await initializeLanguagePreference();

const params = new URLSearchParams(window.location.search);
const next = params.get('next');
const nextInput = document.querySelector<HTMLInputElement>('input[name="next"]');
if (nextInput && next?.startsWith('/web/')) nextInput.value = next;

const error = params.get('error');
const errorElement = document.querySelector<HTMLElement>('.file-error');
if (error && errorElement) {
  errorElement.textContent = error;
  errorElement.hidden = false;
}
