// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Isolated living catalog for foundations, icons, primitives, and DS elements. */

import {html, render} from 'lit';
import '../design-system/index.css';
import '../styles/design-system.css';
import {
  defineDesignSystemElements,
  icon,
  ICON_REGISTRY,
  type IconName,
} from '../design-system/index.ts';

defineDesignSystemElements();

const TOKEN_NAMES = [
  'color-bg-base',
  'color-bg-surface',
  'color-text-primary',
  'color-text-muted',
  'color-accent-action',
  'color-danger',
  'color-border-subtle',
  'focus-ring-color',
  'radius-control',
  'radius-dialog',
  'duration-control',
] as const;

const host = document.getElementById('ds-primitives');
if (host) {
  render(html`
    <article class="ds-card">
      <h2>Theme and state matrix</h2>
      <div class="ds-stage">
        <button class="dl-btn" type="button" data-color-mode="dark">Dark</button>
        <button class="dl-btn" type="button" data-color-mode="light">Light</button>
        <button class="dl-btn" type="button" disabled>Disabled</button>
        <button class="dl-icon-button" type="button" aria-label="Add item">
          ${icon('add', {size: 'md'})}
        </button>
        <span class="ds-forced-colors-note">Forced-colors: native borders and Highlight focus ring.</span>
      </div>
    </article>

    <article class="ds-card">
      <h2>Semantic icon registry</h2>
      <div class="ds-icon-grid">
        ${(Object.keys(ICON_REGISTRY) as IconName[]).map((name) => html`
          <figure class="ds-icon-specimen">
            ${icon(name, {size: 'md'})}
            <figcaption><code>${name}</code><small>${ICON_REGISTRY[name].source}</small></figcaption>
          </figure>
        `)}
      </div>
    </article>

    <article class="ds-card">
      <h2>Native-first controls</h2>
      <div class="ds-viewport-matrix">
        <section class="ds-viewport ds-viewport--390" aria-label="390 pixel specimen">
          <strong>390px</strong>
          <div class="ds-stage">
            <button class="dl-btn" type="button">Standard</button>
            <button class="dl-btn dl-btn-danger-text" type="button">Destructive</button>
            <label class="dl-dialog-checkbox"><input type="checkbox" checked> Checkbox</label>
            <label class="dl-dialog-checkbox"><input type="radio" checked> Radio</label>
            <textarea class="dl-dialog-input" rows="2" placeholder="Text input"></textarea>
          </div>
        </section>
        <section class="ds-viewport ds-viewport--1440" aria-label="1440 pixel specimen">
          <strong>1440px</strong>
          <div class="ds-stage">
            <button class="dl-btn" type="button">Standard</button>
            <button class="dl-btn" type="button" disabled>Disabled</button>
            <button class="dl-btn" id="ds-dialog-open" type="button">Open dialog</button>
          </div>
        </section>
      </div>
      <dialog class="confirm-dialog" id="ds-dialog" aria-labelledby="ds-dialog-title">
        <form method="dialog">
          <h2 id="ds-dialog-title">Native dialog primitive</h2>
          <p>Focus, Escape, and modality remain browser-owned.</p>
          <div class="dl-dialog-actions">
            <button type="submit">Cancel</button>
            <button class="dl-dialog-danger" type="submit">Confirm</button>
          </div>
        </form>
      </dialog>
    </article>

    <article class="ds-card">
      <h2>Split layout</h2>
      <dl-split-layout class="ds-split" orientation="horizontal" primary="start"
                       size="180" min="100" max="300">
        <div class="ds-pane" slot="start">Primary pane</div>
        <div class="ds-pane" slot="end">Flexible pane</div>
      </dl-split-layout>
    </article>
  `, host);
}

const tokenHost = document.getElementById('ds-tokens');
if (tokenHost) {
  const styles = getComputedStyle(document.documentElement);
  render(html`
    <article class="ds-card">
      <h2>Runtime CSS authority</h2>
      <div class="ds-token-group">
        ${TOKEN_NAMES.map((name) => html`
          <div class="ds-token-row">
            <span class="ds-token-swatch" style=${`background:var(--${name})`}></span>
            <code>--${name}: ${styles.getPropertyValue(`--${name}`).trim()}</code>
          </div>
        `)}
      </div>
    </article>
  `, tokenHost);
}

document.querySelectorAll<HTMLButtonElement>('[data-color-mode]').forEach((button) => {
  button.addEventListener('click', () => {
    const mode = button.dataset.colorMode ?? 'dark';
    document.documentElement.setAttribute('data-color-mode', mode);
    document.documentElement.style.colorScheme = mode;
  });
});
document.getElementById('ds-dialog-open')?.addEventListener('click', () => {
  document.querySelector<HTMLDialogElement>('#ds-dialog')?.showModal();
});
