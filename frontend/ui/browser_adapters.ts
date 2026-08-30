// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import type {DlApp} from './app.ts';
import {setupMathRendering} from './mathjax.ts';
import {setupPanelSplits} from './split_panel.ts';

let initialization: Promise<void> | null = null;

/** Initialize the document-lifetime browser adapters after the Shell is rendered. */
export function initializeBrowserAdapters(app: DlApp): Promise<void> {
  if (initialization !== null) return initialization;
  initialization = app.ready.then(() => {
    setupPanelSplits();
    setupMathRendering();
  });
  return initialization;
}
