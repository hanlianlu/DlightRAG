// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, updateWhenLocaleChanges, str} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import type {PresentationSource} from '../api/conversations.ts';
import {icon} from '../design-system/index.ts';
import {LightElement} from '../lib/lit-host.ts';
import {setSanitizedLlmHtml} from '../lib/safe-html.ts';
import {safeExternalHttpHref, safeImageSrc, safeSameOriginHref} from '../lib/urls.ts';
import {renderMath} from '../lib/math.ts';
import type {ImageOpenDetail} from './image-lightbox.ts';

export interface InspectorSourcesStateDetail {
  hasSources: boolean;
  fullyExpanded: boolean;
}

/** Source list content owned by the Inspector. */
export class DlInspectorSources extends LightElement {
  static properties = {
    sources: {attribute: false},
    expandedRef: {state: true},
    onlyChunk: {state: true},
    activeRef: {state: true},
    activeChunk: {state: true},
    showAll: {state: true},
  };

  declare sources: PresentationSource[];
  declare expandedRef: string | null;
  declare onlyChunk: string | null;
  declare activeRef: string | null;
  declare activeChunk: string | null;
  declare showAll: boolean;

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.sources = [];
    this.expandedRef = null;
    this.onlyChunk = null;
    this.activeRef = null;
    this.activeChunk = null;
    this.showAll = false;
  }

  setSelection(ref?: string, chunk?: string): void {
    this.showAll = false;
    this.expandedRef = ref || null;
    this.onlyChunk = chunk || null;
    this.activeRef = ref && chunk ? ref : null;
    this.activeChunk = ref && chunk ? chunk : null;
  }

  expandAll(): void {
    this.showAll = true;
    this.expandedRef = null;
    this.onlyChunk = null;
  }

  collapseAll(): void {
    this.showAll = false;
    this.expandedRef = null;
    this.onlyChunk = null;
  }

  get fullyExpanded(): boolean {
    return this.sources.length > 0 && this.showAll;
  }

  protected override updated(changed: PropertyValues<this>): void {
    if (changed.has('sources')) {
      this.sources.forEach((source) => {
        source.chunks.forEach((chunk, index) => {
          const key = this.#chunkKey(chunk.chunk_idx, index);
          const host = this.querySelector<HTMLElement>(
            `[data-source-content="${CSS.escape(source.id)}:${CSS.escape(key)}"]`,
          );
          if (host) setSanitizedLlmHtml(host, chunk.content_html);
        });
      });
    }
    if (changed.has('sources') || changed.has('expandedRef') || changed.has('showAll')) {
      this.querySelectorAll('.source-doc.expanded .source-doc-chunks').forEach((element) => {
        renderMath(element);
      });
    }
    this.dispatchEvent(new CustomEvent<InspectorSourcesStateDetail>(
      'dl-inspector-sources-state-change',
      {
        bubbles: true,
        composed: true,
        detail: {
          hasSources: this.sources.length > 0,
          fullyExpanded: this.fullyExpanded,
        },
      },
    ));
  }

  #chunkKey(chunkIndex: number | null, fallback: number): string {
    return String(chunkIndex ?? fallback + 1);
  }

  #toggle(sourceId: string): void {
    if (!this.showAll && this.expandedRef === sourceId) {
      this.expandedRef = null;
      this.onlyChunk = null;
      return;
    }
    this.showAll = false;
    this.expandedRef = sourceId;
    this.onlyChunk = null;
  }

  #source(source: PresentationSource): TemplateResult {
    const expanded = this.showAll || this.expandedRef === source.id;
    const download = safeSameOriginHref(source.download_url);
    const external = safeExternalHttpHref(source.source_url);
    return html`
      <div class="source-doc${expanded ? ' expanded' : ''}" data-ref=${source.id}>
        <div class="source-doc-header">
          <button class="source-doc-toggle" type="button" aria-expanded=${String(expanded)}
                  @click=${() => { this.#toggle(source.id); }}>
            <span class="collapse-icon">${icon('disclosure', {size: 'xs'})}</span>
            <span class="source-doc-title">${source.title}</span>
            <span class="source-doc-badge">${source.id}</span>
            <span class="source-doc-count">${source.chunks.length}</span>
          </button>
          ${download ? html`
            <a href=${download} class="source-action-icon"
               title=${msg('Download source', {id: 'inspectorSources.downloadSource'})}
               aria-label=${msg('Download source', {id: 'inspectorSources.downloadSource'})} download>
              ${icon('download', {size: 'sm', className: 'source-action-icon-svg'})}
            </a>
          ` : nothing}
          ${external ? html`
            <a href=${external} class="source-action-icon"
               title=${msg('Open source', {id: 'inspectorSources.openSource'})}
               aria-label=${msg('Open source', {id: 'inspectorSources.openSource'})}
               target="_blank" rel="noopener noreferrer">
              ${icon('open-external', {size: 'sm', className: 'source-action-icon-svg'})}
            </a>
          ` : nothing}
        </div>
        <div class="source-doc-chunks" ?hidden=${!expanded}>
          ${repeat(
            source.chunks,
            (chunk, index) => this.#chunkKey(chunk.chunk_idx, index),
            (chunk, index) => {
              const key = this.#chunkKey(chunk.chunk_idx, index);
              const hidden = expanded && !this.showAll && this.onlyChunk !== null
                && key !== this.onlyChunk;
              const active = this.activeRef === source.id && this.activeChunk === key;
              const image = safeImageSrc(chunk.image_url || chunk.thumbnail_url);
              const thumbnail = safeImageSrc(chunk.thumbnail_url || chunk.image_url);
              return html`
                <div class="source-chunk${active ? ' active' : ''}" data-ref=${source.id}
                     data-chunk=${key} ?hidden=${hidden}>
                  <div class="source-chunk-header">
                    <span class="source-chunk-page">
                      ${chunk.page_number === null ? `#${key}` : `p.${chunk.page_number}`}
                    </span>
                  </div>
                  ${image && thumbnail ? html`
                    <div class="source-chunk-image">
                      <img src=${thumbnail}
                           alt=${msg(str`Page ${chunk.page_number ?? ''}`, {id: 'inspectorSources.pageAlt'})}
                           loading="lazy" role="button" tabindex="0"
                           aria-label=${msg('Open page image', {id: 'inspectorSources.openPageImage'})}
                           @click=${(event: Event) => this.#openImage(
                             image,
                             event.currentTarget as HTMLElement,
                           )}
                           @keydown=${(event: KeyboardEvent) => {
                             if (event.key !== 'Enter' && event.key !== ' ') return;
                             event.preventDefault();
                             this.#openImage(image, event.currentTarget as HTMLElement);
                           }}>
                    </div>
                  ` : nothing}
                  ${chunk.content_html ? html`
                    <div class="source-chunk-content"
                         data-source-content=${`${source.id}:${key}`}></div>
                  ` : nothing}
                </div>
              `;
            },
          )}
        </div>
      </div>
    `;
  }

  protected override render(): TemplateResult {
    return html`${repeat(this.sources, (source) => source.id, (source) => this.#source(source))}`;
  }

  #openImage(src: string, returnFocus: HTMLElement): void {
    const gallery = this.sources.flatMap((source) => source.chunks)
      .map((chunk) => safeImageSrc(chunk.image_url || chunk.thumbnail_url))
      .filter(Boolean);
    this.dispatchEvent(new CustomEvent<ImageOpenDetail>('dl-image-open', {
      bubbles: true,
      composed: true,
      detail: {src, gallery: [...new Set(gallery)], returnFocus},
    }));
  }

}

customElements.define('dl-inspector-sources', DlInspectorSources);

declare global {
  interface HTMLElementTagNameMap {
    'dl-inspector-sources': DlInspectorSources;
  }

  interface HTMLElementEventMap {
    'dl-inspector-sources-state-change': CustomEvent<InspectorSourcesStateDetail>;
  }
}
