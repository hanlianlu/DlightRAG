// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Minimal runtime model catalogue editor used by the Settings drawer. */

import {msg, str} from '@lit/localize';
import {html, type TemplateResult} from 'lit';
import {
  getModelCatalogue,
  ModelCatalogueRequestError,
  putModelCatalogueEntry,
  removeModelCatalogueEntry,
  type ModelCatalogue,
  type ModelCatalogueEntry,
} from '../api/model_catalogue.ts';
import {LightElement} from '../lib/lit_host.ts';

const NEW_ENTRY: Omit<ModelCatalogueEntry, 'source'> = {
  provider: 'openai',
  model: '',
  base_url: null,
  profile: {
    context_window_tokens: 32768,
    max_input_tokens: null,
    max_output_tokens: null,
    supports_images: false,
    reasoning: null,
  },
};

/** Owns catalogue reads, revision-guarded writes, and one raw complete-entry draft. */
export class DlModelCatalogue extends LightElement {
  static properties = {
    catalogue: {state: true},
    loading: {state: true},
    pending: {state: true},
    draft: {state: true},
    error: {state: true},
  };

  declare catalogue: ModelCatalogue | null;
  declare loading: boolean;
  declare pending: boolean;
  declare draft: string | null;
  declare error: string | null;

  #controller: AbortController | null = null;

  constructor() {
    super();
    this.catalogue = null;
    this.loading = false;
    this.pending = false;
    this.draft = null;
    this.error = null;
  }

  override disconnectedCallback(): void {
    this.#controller?.abort();
    this.#controller = null;
    super.disconnectedCallback();
  }

  async refresh(): Promise<void> {
    this.#controller?.abort();
    const controller = new AbortController();
    this.#controller = controller;
    this.loading = true;
    this.error = null;
    try {
      const catalogue = await getModelCatalogue(controller.signal);
      if (!controller.signal.aborted) this.catalogue = catalogue;
    } catch (error) {
      if (!controller.signal.aborted) this.error = this.#message(error);
    } finally {
      if (!controller.signal.aborted) this.loading = false;
      if (this.#controller === controller) this.#controller = null;
    }
  }

  protected override render(): TemplateResult {
    const models = this.catalogue?.models ?? [];
    return html`
      <div aria-busy=${this.loading ? 'true' : 'false'}>
        <div class="settings-actions">
          <button type="button" class="ui-btn"
                  ?disabled=${!this.catalogue || this.loading || this.pending}
                  @click=${this.#add}>${msg('Add model', {id: 'settings.modelCatalogue.add'})}</button>
          <button type="button" class="ui-btn" ?disabled=${this.loading || this.pending}
                  @click=${() => this.refresh()}>
            ${this.catalogue
              ? msg('Refresh', {id: 'settings.modelCatalogue.refresh'})
              : msg('Load catalogue', {id: 'settings.modelCatalogue.load'})}
          </button>
        </div>
        ${this.catalogue ? html`
          <p class="settings-note">
            ${msg(str`Revision ${this.catalogue.revision}`, {id: 'settings.modelCatalogue.revision'})}
          </p>
        ` : null}
        ${this.error ? html`<p role="alert" class="settings-note">${this.error}</p>` : null}
        <ul class="model-catalogue-list">
          ${models.map((entry) => html`
            <li>
              <code>${entry.provider}/${entry.model}</code>
              ${entry.base_url ? html`<small>${entry.base_url}</small>` : null}
              <span>${entry.source}</span>
              <button type="button" class="ui-btn" ?disabled=${this.pending}
                      @click=${() => this.#edit(entry)}>
                ${msg('Edit', {id: 'settings.modelCatalogue.edit'})}
              </button>
              ${entry.source === 'overlay' ? html`
                <button type="button" class="ui-btn ui-btn-danger-text"
                        ?disabled=${this.pending} @click=${() => this.#remove(entry)}>
                  ${msg('Remove override', {id: 'settings.modelCatalogue.remove'})}
                </button>
              ` : null}
            </li>
          `)}
        </ul>
        ${this.draft !== null ? html`
          <label for="model-catalogue-draft">
            ${msg('Complete model entry (JSON)', {id: 'settings.modelCatalogue.entryJson'})}
          </label>
          <textarea id="model-catalogue-draft" rows="16" spellcheck="false"
                    .value=${this.draft} @input=${this.#draftChanged}></textarea>
          <div class="settings-actions">
            <button type="button" class="ui-btn" ?disabled=${this.pending}
                    @click=${this.#save}>${msg('Save model', {id: 'settings.modelCatalogue.save'})}</button>
            <button type="button" class="ui-btn" ?disabled=${this.pending}
                    @click=${this.#cancel}>${msg('Cancel', {id: 'settings.cancel'})}</button>
          </div>
        ` : null}
      </div>
    `;
  }

  #add = (): void => {
    this.error = null;
    this.draft = JSON.stringify(NEW_ENTRY, null, 2);
  };

  #edit(entry: ModelCatalogueEntry): void {
    const payload: Omit<ModelCatalogueEntry, 'source'> = {
      provider: entry.provider,
      model: entry.model,
      base_url: entry.base_url,
      profile: entry.profile,
    };
    this.error = null;
    this.draft = JSON.stringify(payload, null, 2);
  }

  #cancel = (): void => {
    this.draft = null;
    this.error = null;
  };

  #draftChanged = (event: Event): void => {
    this.draft = (event.currentTarget as HTMLTextAreaElement).value;
  };

  #save = async (): Promise<void> => {
    if (!this.catalogue || this.draft === null || this.pending) return;
    let entry: Omit<ModelCatalogueEntry, 'source'>;
    try {
      entry = JSON.parse(this.draft) as Omit<ModelCatalogueEntry, 'source'>;
    } catch {
      this.error = msg('Entry JSON is invalid.', {id: 'settings.modelCatalogue.invalidJson'});
      return;
    }
    this.pending = true;
    this.error = null;
    try {
      this.catalogue = await putModelCatalogueEntry(entry, this.catalogue.revision);
      this.draft = null;
    } catch (error) {
      this.error = this.#message(error);
      if (error instanceof ModelCatalogueRequestError && error.status === 412) {
        await this.refresh();
        this.error = msg('The catalogue changed. Review the latest revision and retry.', {
          id: 'settings.modelCatalogue.conflict',
        });
      }
    } finally {
      this.pending = false;
    }
  };

  async #remove(entry: ModelCatalogueEntry): Promise<void> {
    if (!this.catalogue || this.pending) return;
    this.pending = true;
    this.error = null;
    try {
      this.catalogue = await removeModelCatalogueEntry(entry, this.catalogue.revision);
      this.draft = null;
    } catch (error) {
      this.error = this.#message(error);
      if (error instanceof ModelCatalogueRequestError && error.status === 412) {
        await this.refresh();
        this.error = msg('The catalogue changed. Review the latest revision and retry.', {
          id: 'settings.modelCatalogue.conflict',
        });
      }
    } finally {
      this.pending = false;
    }
  }

  #message(error: unknown): string {
    return error instanceof Error
      ? error.message
      : msg('Model catalogue request failed.', {id: 'settings.modelCatalogue.failed'});
  }
}

customElements.define('dl-model-catalogue', DlModelCatalogue);

declare global {
  interface HTMLElementTagNameMap {
    'dl-model-catalogue': DlModelCatalogue;
  }
}
