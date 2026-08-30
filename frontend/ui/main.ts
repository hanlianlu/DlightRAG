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
const {setupMathRendering} = await import('./mathjax.ts');
const {setupPanelSplits} = await import('./split_panel.ts');

// Vite's one-shot entry is the approved seam for the two browser/third-party
// adapters: MathJax loading/scheduling and Web Awesome split-panel binding.
document.addEventListener('DOMContentLoaded', () => {
  const app = document.querySelector<DlApp>('dl-app');
  if (!app) return;
  void app.ready.then(() => {
    setupPanelSplits();
    setupMathRendering();
  });
});
