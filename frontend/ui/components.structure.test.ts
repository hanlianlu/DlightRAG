// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Structural lock on the shared control layer.

 * Every <dialog> must carry one of the primitive dialog classes, and every
 * checkbox must live inside the .ui-dialog-checkbox primitive. New UI that
 * bypasses primitives fails here instead of shipping UA styling.
 */

import {readdirSync, readFileSync} from 'node:fs';
import {join, dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {test} from 'node:test';
import assert from 'node:assert/strict';

const UI_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', 'ui');

function uiSourceFiles(): string[] {
  const files: string[] = [];
  for (const name of readdirSync(UI_DIR)) {
    if (name.endsWith('.ts') && !name.endsWith('.test.ts')) files.push(join(UI_DIR, name));
  }
  return files;
}

const DIALOG_CLASSES = new Set(['confirm-dialog', 'settings-dialog', 'workspace-dialog']);

test('every dialog carries a primitive dialog class', () => {
  for (const file of uiSourceFiles()) {
    const source = readFileSync(file, 'utf8');
    const pattern = /<dialog([^>]*)>/g;
    for (const match of source.matchAll(pattern)) {
      const attributes = match[1];
      const classes = /class="([^"]*)"/.exec(attributes)?.[1]?.split(/\s+/) ?? [];
      assert.ok(
        classes.some((name) => DIALOG_CLASSES.has(name)),
        `${file}: <dialog> must carry one of ${[...DIALOG_CLASSES].join('/')}`,
      );
    }
  }
});

test('every checkbox lives inside the .ui-dialog-checkbox primitive', () => {
  for (const file of uiSourceFiles()) {
    const source = readFileSync(file, 'utf8');
    // A checkbox is legal only when a .ui-dialog-checkbox label appears
    // somewhere before it on the same template line region; walk label blocks.
    const labelPattern = /<label[^>]*class="[^"]*ui-dialog-checkbox[^"]*"[^>]*>([\s\S]*?)<\/label>/g;
    const barePattern = /<input[^>]*type="checkbox"[^>]*>/g;
    for (const match of source.matchAll(barePattern)) {
      const index = match.index ?? 0;
      const before = source.slice(0, index);
      const lastLabelOpen = before.lastIndexOf('ui-dialog-checkbox');
      const lastLabelClose = before.lastIndexOf('</label>');
      assert.ok(
        lastLabelOpen > lastLabelClose,
        `${file}: checkbox must sit inside a .ui-dialog-checkbox label`,
      );
    }
    void labelPattern;
  }
});
