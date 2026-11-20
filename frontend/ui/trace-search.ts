// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The box a reader types into when they remember words and not where they were.

 * It owns the text and the pause after typing, so nothing outside can redraw what a reader is
 * writing and eat the space between two words. What is done with the words is the dock's, because
 * the answer narrows the dock's own panes rather than adding anything of its own.
 */

import {msg} from '@lit/localize';
import {html, nothing, type TemplateResult} from 'lit';
import {raise} from '../lib/dom.ts';
import {LightElement} from '../lib/lit-host.ts';
import styles from '../styles/trace-search.module.css';

/** The words a reader settled on, as they wrote them without their edges. */
export interface TraceQueryDetail {
  terms: string;
}

/** A Run whose words can be searched; the box needs nothing else of one. */
interface SearchableTraces {
  search(text: string): Promise<unknown>;
}

const SEARCH_DEBOUNCE_MILLISECONDS = 200;

export class DlTraceSearch extends LightElement {
  static override properties = {
    source: {attribute: false},
  };

  declare source: SearchableTraces | null;

  /** What the reader has typed, exactly: the box redraws from this, so trimming it here would
   * eat the space between two words on the next redraw. */
  #query = '';
  #timer: ReturnType<typeof setTimeout> | null = null;

  override disconnectedCallback(): void {
    this.#stopWaiting();
    super.disconnectedCallback();
  }

  protected override willUpdate(changed: Map<PropertyKey, unknown>): void {
    if (changed.has('source')) {
      this.#stopWaiting();
      this.#query = '';
    }
  }

  /** Say what the reader settled on; the dock narrows its panes to it, or widens them for ''. */
  #ask(terms: string): void {
    raise(this, 'dl-trace-query', {terms});
  }

  #stopWaiting(): void {
    if (this.#timer !== null) {
      clearTimeout(this.#timer);
      this.#timer = null;
    }
  }

  protected override render(): TemplateResult | typeof nothing {
    if (this.source === null) return nothing;
    return html`
      <div class=${styles.searchRow}>
        <input class="dl-input ${styles.searchInput}" type="search" .value=${this.#query}
               placeholder=${msg('Search every agent of this Run', {id: 'tracesPanel.search'})}
               aria-label=${msg('Search every agent of this Run', {id: 'tracesPanel.search'})}
               @input=${this.#typed}>
        <button type="button" class="dl-btn ${styles.searchClear}" ?hidden=${this.#query === ''}
                @click=${this.#cleared}>${msg('Clear', {id: 'tracesPanel.searchClear'})}</button>
      </div>`;
  }

  /** One search per pause, not one per keystroke. */
  #typed = (event: Event): void => {
    this.#query = (event.target as HTMLInputElement).value;
    this.#stopWaiting();
    this.#timer = setTimeout(() => {
      this.#timer = null;
      this.#ask(this.#query.trim());
    }, SEARCH_DEBOUNCE_MILLISECONDS);
    this.requestUpdate();
  };

  #cleared = (): void => {
    this.#stopWaiting();
    this.#query = '';
    this.#ask('');
    this.requestUpdate();
  };
}

customElements.define('dl-trace-search', DlTraceSearch);

declare global {
  interface HTMLElementTagNameMap {
    'dl-trace-search': DlTraceSearch;
  }

  interface HTMLElementEventMap {
    'dl-trace-query': CustomEvent<TraceQueryDetail>;
  }
}
