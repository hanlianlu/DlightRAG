// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {THEME_STORAGE_KEY} from '../lib/theme.ts';
import type {DlThemeControl} from './theme.ts';
import './theme.ts';

const originalMatchMedia = window.matchMedia;

function media(): MediaQueryList {
  return {
    matches: false,
    media: '(prefers-color-scheme: dark)',
    onchange: null,
    addListener() {},
    removeListener() {},
    addEventListener() {},
    removeEventListener() {},
    dispatchEvent: () => true,
  };
}

function buttonNamed(root: ParentNode, name: string): HTMLButtonElement | null {
  return Array.from(root.querySelectorAll<HTMLButtonElement>('button'))
    .find((button) => (button.getAttribute('aria-label') || button.textContent?.trim()) === name)
    ?? null;
}

afterEach(() => {
  document.body.replaceChildren();
  window.matchMedia = originalMatchMedia;
  localStorage.removeItem(THEME_STORAGE_KEY);
  document.documentElement.setAttribute('data-theme', 'system');
});

it('owns its menu, roving selection, persistence, and root color capability', async () => {
  window.matchMedia = media;
  const control = document.createElement('dl-theme-control') as DlThemeControl;
  document.body.appendChild(control);
  await control.updateComplete;

  const trigger = buttonNamed(control, 'Appearance')!;
  trigger.click();
  await control.updateComplete;
  const menu = control.querySelector('[role="menu"][aria-label="Appearance"]')!;
  expect(menu.hasAttribute('hidden')).to.equal(false);
  expect(trigger.getAttribute('aria-expanded')).to.equal('true');

  const dark = buttonNamed(menu, 'Dark')!;
  dark.click();
  await control.updateComplete;
  expect(document.documentElement.getAttribute('data-theme')).to.equal('dark');
  expect(document.documentElement.getAttribute('data-color-mode')).to.equal('dark');
  expect(localStorage.getItem(THEME_STORAGE_KEY)).to.equal('dark');
  expect(dark.getAttribute('aria-checked')).to.equal('true');
  expect(menu.hasAttribute('hidden')).to.equal(true);
});
