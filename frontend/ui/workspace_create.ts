// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {html, type TemplateResult} from 'lit';
import {WorkspaceApiError, createWorkspaceRequest} from '../api/workspaces.ts';
import {LightElement} from '../lib/lit_host.ts';
import {workspaceStore} from '../stores/workspaceStore.ts';
import type {ToastRequestDetail} from './toast.ts';

export interface WorkspaceCreatedDetail {
  workspace: string;
}

/** New-workspace row: input intent, asynchronous creation, and pending state. */
export class DlWorkspaceCreate extends LightElement {
  static properties = {pending: {state: true}};

  declare pending: boolean;

  #lifecycle: AbortController | null = null;

  constructor() {
    super();
    this.pending = false;
    this.className = 'ui-popover-create';
  }

  override connectedCallback(): void {
    super.connectedCallback();
    this.#lifecycle = new AbortController();
  }

  override disconnectedCallback(): void {
    this.#lifecycle?.abort();
    this.#lifecycle = null;
    this.pending = false;
    super.disconnectedCallback();
  }

  async #submit(): Promise<void> {
    const input = this.querySelector('input');
    const displayName = input?.value.trim();
    const lifecycle = this.#lifecycle;
    if (!input || !displayName || !lifecycle || this.pending) return;
    this.pending = true;
    try {
      const created = await createWorkspaceRequest(displayName, lifecycle.signal);
      if (
        lifecycle.signal.aborted || this.#lifecycle !== lifecycle || !this.isConnected
      ) return;
      workspaceStore.add({
        workspace: created.workspace,
        displayName: created.display_name,
        embeddingModel: '',
      });
      input.value = '';
      this.#requestToast({message: `Workspace ${created.display_name} created.`});
      this.dispatchEvent(new CustomEvent<WorkspaceCreatedDetail>('dl-workspace-created', {
        detail: {workspace: created.workspace},
        bubbles: true,
        composed: true,
      }));
    } catch (error) {
      if (
        !lifecycle.signal.aborted && this.#lifecycle === lifecycle && this.isConnected
      ) {
        this.#requestToast({
          message: error instanceof WorkspaceApiError
            ? error.message
            : 'Failed to create workspace',
          duration: 5000,
        });
      }
    } finally {
      if (this.#lifecycle === lifecycle) this.pending = false;
    }
  }

  #requestToast(detail: ToastRequestDetail): void {
    this.dispatchEvent(new CustomEvent<ToastRequestDetail>('dl-toast-request', {
      detail,
      bubbles: true,
      composed: true,
    }));
  }

  protected override render(): TemplateResult {
    return html`
      <input class="ui-popover-input" type="text" placeholder="New workspace..."
             aria-label="New workspace name" ?disabled=${this.pending}
             @click=${(event: Event) => { event.stopPropagation(); }}
             @keydown=${(event: KeyboardEvent) => {
               if (event.key !== 'Enter') return;
               event.preventDefault();
               void this.#submit();
             }}>
      <button class="ui-popover-create-btn" type="button" aria-label="Create workspace"
              ?disabled=${this.pending} @click=${(event: MouseEvent) => {
                event.stopPropagation();
                void this.#submit();
              }}>+</button>
    `;
  }
}

customElements.define('dl-workspace-create', DlWorkspaceCreate);

declare global {
  interface HTMLElementTagNameMap {
    'dl-workspace-create': DlWorkspaceCreate;
  }

  interface HTMLElementEventMap {
    'dl-workspace-created': CustomEvent<WorkspaceCreatedDetail>;
  }
}
