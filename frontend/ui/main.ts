// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import '../tokens/utopia.css';
import '../styles/global.css';
import '../styles/primitives.css';
import '../styles/layout.css';
import '../styles/panels.css';
import '../styles/artifacts.css';
import '../styles/files.css';
import '../styles/sources.css';

import type {DlApp} from './app.ts';
import {initializeLanguagePreference} from '../i18n/locale.ts';

// The language preference must resolve before the app element upgrades and
// renders, so the application module is imported dynamically after it.
await initializeLanguagePreference();

const appModule = await import('./app.ts');
void appModule;
const {initializeBrowserAdapters} = await import('./browser_adapters.ts');

// Dynamic imports can finish after DOMContentLoaded on a cold load. Start
// immediately when parsing already completed, otherwise wait for it once.
const start = (): void => {
  const app = document.querySelector<DlApp>('dl-app');
  if (app) void initializeBrowserAdapters(app);
};
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', start, {once: true});
} else {
  start();
}
