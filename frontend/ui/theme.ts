// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Theme Control Feature and document color-mode capability. */

import {msg, updateWhenLocaleChanges} from '@lit/localize';
import {html, type TemplateResult} from 'lit';
import {icon, type IconName} from '../design-system/index.ts';
import {
  parseThemePreference,
  resolveColorMode,
  THEME_STORAGE_KEY,
  type ThemePreference,
} from '../lib/theme.ts';
import {LightElement} from '../lib/lit_host.ts';
import {rovingArrowKeydown} from '../lib/listbox.ts';
import {createAutoDismiss} from '../lib/popover.ts';

function readPreference(): ThemePreference {
  try {
    const stored = parseThemePreference(window.localStorage.getItem(THEME_STORAGE_KEY));
    return stored === 'system'
      ? parseThemePreference(document.documentElement.getAttribute('data-theme'))
      : stored;
  } catch {
    return parseThemePreference(document.documentElement.getAttribute('data-theme'));
  }
}

function writePreference(preference: ThemePreference): void {
  try {
    if (preference === 'system') window.localStorage.removeItem(THEME_STORAGE_KEY);
    else window.localStorage.setItem(THEME_STORAGE_KEY, preference);
  } catch {
    // Theme choice remains active for this page when storage is blocked.
  }
}

/** Owns theme preference, menu accessibility, persistence, and system changes. */
export class DlThemeControl extends LightElement {
  static properties = {
    preference: {state: true},
    menuOpen: {state: true},
  };

  declare preference: ThemePreference;
  declare menuOpen: boolean;

  #events: AbortController | null = null;
  #media: MediaQueryList | null = null;
  readonly #dismiss = createAutoDismiss({
    getAnchor: () => this,
    isOpen: () => this.menuOpen,
    onDismiss: (reason) => this.#close(reason === 'escape'),
  });

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.preference = 'system';
    this.menuOpen = false;
  }

  override connectedCallback(): void {
    super.connectedCallback();
    this.preference = readPreference();
    this.#media = window.matchMedia('(prefers-color-scheme: dark)');
    this.#media.addEventListener('change', this.#mediaChanged);
    const events = new AbortController();
    this.#events = events;
    window.addEventListener('storage', this.#storageChanged, {signal: events.signal});
    this.#apply();
  }

  override disconnectedCallback(): void {
    this.#events?.abort();
    this.#events = null;
    this.#media?.removeEventListener('change', this.#mediaChanged);
    this.#media = null;
    this.#dismiss.deactivate();
    super.disconnectedCallback();
  }

  protected override updated(): void {
    this.#apply();
    if (this.menuOpen) this.#dismiss.activate();
    else this.#dismiss.deactivate();
  }

  protected override render(): TemplateResult {
    const appearance = msg('Appearance', {id: 'theme.appearance'});
    return html`
      <button id="theme-trigger" type="button" aria-label=${appearance} title=${appearance}
              aria-haspopup="menu" aria-controls="theme-menu"
              aria-expanded=${this.menuOpen ? 'true' : 'false'}
              @click=${this.#triggerClick} @keydown=${this.#triggerKeydown}>
        ${icon('moon', {size: 'sm', className: 'theme-icon theme-icon-moon'})}
        ${icon('sun', {size: 'sm', className: 'theme-icon theme-icon-sun'})}
      </button>
      <div id="theme-menu" role="menu" aria-label=${appearance} ?hidden=${!this.menuOpen}
           @keydown=${this.#menuKeydown}>
        ${this.#option('system', msg('System', {id: 'theme.system'}), 'system')}
        ${this.#option('light', msg('Light', {id: 'theme.light'}), 'sun')}
        ${this.#option('dark', msg('Dark', {id: 'theme.dark'}), 'moon')}
      </div>
    `;
  }

  #option(value: ThemePreference, label: string, iconName: IconName): TemplateResult {
    const checked = this.preference === value;
    return html`
      <button type="button" role="menuitemradio" data-theme-value=${value} aria-label=${label}
              aria-checked=${checked ? 'true' : 'false'} tabindex=${checked ? '0' : '-1'}
              @click=${() => this.#select(value)}>
        <span class="theme-menu-icon" aria-hidden="true">${icon(iconName, {size: 'sm'})}</span>
        <span class="theme-menu-label">${label}</span>
        <span class="theme-menu-check" aria-hidden="true">${icon('check', {size: 'xs'})}</span>
      </button>
    `;
  }

  #apply(): void {
    // Theme is an approved top-level browser capability; the root is its interface.
    const root = document.documentElement;
    const colorMode = resolveColorMode(this.preference, this.#media?.matches ?? false);
    root.setAttribute('data-theme', this.preference);
    root.setAttribute('data-color-mode', colorMode);
    root.style.colorScheme = colorMode;
  }

  #open(focusCurrent: boolean): void {
    this.menuOpen = true;
    if (focusCurrent) {
      void this.updateComplete.then(() => {
        this.querySelector<HTMLButtonElement>(
          `[data-theme-value="${this.preference}"]`,
        )?.focus();
      });
    }
  }

  #close(restoreFocus: boolean): void {
    if (!this.menuOpen) return;
    this.menuOpen = false;
    if (restoreFocus) {
      window.requestAnimationFrame(() => this.querySelector<HTMLButtonElement>('#theme-trigger')?.focus());
    }
  }

  #select(preference: ThemePreference): void {
    this.preference = preference;
    writePreference(preference);
    this.#close(true);
  }

  #triggerClick = (): void => {
    if (this.menuOpen) this.#close(false);
    else this.#open(false);
  };

  #triggerKeydown = (event: KeyboardEvent): void => {
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      this.#open(true);
      return;
    }
    if (event.key !== 'Enter' && event.key !== ' ') return;
    event.preventDefault();
    if (this.menuOpen) this.#close(false);
    else this.#open(true);
  };

  #menuKeydown = (event: KeyboardEvent): void => {
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopImmediatePropagation();
      this.#close(true);
      return;
    }
    const active = document.activeElement;
    if (active instanceof HTMLButtonElement && active.getAttribute('role') === 'menuitemradio'
        && (event.key === 'Enter' || event.key === ' ')) {
      event.preventDefault();
      this.#select(parseThemePreference(active.dataset.themeValue || null));
      return;
    }
    rovingArrowKeydown(event, '[role="menuitemradio"]');
  };

  #mediaChanged = (): void => {
    if (this.preference === 'system') this.#apply();
  };

  #storageChanged = (event: StorageEvent): void => {
    let storageArea: Storage | null = null;
    try {
      storageArea = window.localStorage;
    } catch {
      storageArea = null;
    }
    if (event.storageArea !== storageArea) return;
    if (event.key !== null && event.key !== THEME_STORAGE_KEY) return;
    this.preference = parseThemePreference(event.newValue);
  };
}

customElements.define('dl-theme-control', DlThemeControl);

declare global {
  interface HTMLElementTagNameMap {
    'dl-theme-control': DlThemeControl;
  }
}
