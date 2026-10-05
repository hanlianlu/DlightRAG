// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, str} from '@lit/localize';
import {html, type TemplateResult} from 'lit';
import {createWorkspaceRequest} from '../api/workspaces.ts';
import {icon} from '../design-system/index.ts';
import {apiErrorMessage} from '../lib/errors.ts';
import {LightElement} from '../lib/lit-host.ts';
import {productionHandles, type AppHandles} from '../stores/app-handles.ts';
import {requestToast} from './toast-request.ts';

export interface WorkspaceCreatedDetail {
  workspace: string;
}

/** New-workspace row: input intent, asynchronous creation, and pending state. */
export class DlWorkspaceCreate extends LightElement {
  static properties = {handles: {attribute: false}, pending: {state: true}};

  declare handles: AppHandles;
  declare pending: boolean;

  constructor() {
    super();
    this.handles = productionHandles();
    this.pending = false;
    this.className = 'dl-popover-create';
  }

  override disconnectedCallback(): void {
    this.pending = false;
    super.disconnectedCallback();
  }

  async #submit(): Promise<void> {
    const input = this.querySelector('input');
    const displayName = input?.value.trim();
    const signal = this.lifetime;
    if (!input || !displayName || signal.aborted || this.pending) return;
    this.pending = true;
    try {
      const created = await createWorkspaceRequest(displayName, signal);
      if (signal.aborted) return;
      this.handles.workspaces.add(created);
      this.handles.ingest.set(created.workspace);
      input.value = '';
      requestToast(this, {
        message: msg(str`Workspace ${created.displayName} created.`, {id: 'workspaceCreate.created'}),
      });
      this.dispatchEvent(new CustomEvent<WorkspaceCreatedDetail>('dl-workspace-created', {
        detail: {workspace: created.workspace},
        bubbles: true,
        composed: true,
      }));
    } catch (error) {
      if (!signal.aborted) {
        requestToast(this, {
          message: apiErrorMessage(
            error,
            msg('Failed to create workspace', {id: 'workspaceCreate.failed'}),
          ),
        });
      }
    } finally {
      if (!signal.aborted) this.pending = false;
    }
  }


  protected override render(): TemplateResult {
    return html`
      <input class="dl-popover-input" type="text"
             placeholder=${msg('New workspace...', {id: 'workspaceCreate.placeholder'})}
             aria-label=${msg('New workspace name', {id: 'workspaceCreate.nameAria'})} ?disabled=${this.pending}
             @click=${(event: Event) => { event.stopPropagation(); }}
             @keydown=${(event: KeyboardEvent) => {
               if (event.key !== 'Enter') return;
               event.preventDefault();
               void this.#submit();
             }}>
      <button class="dl-popover-create-btn" type="button"
              aria-label=${msg('Create workspace', {id: 'workspaceCreate.createAria'})}
              ?disabled=${this.pending} @click=${(event: MouseEvent) => {
                event.stopPropagation();
                void this.#submit();
              }}>
        ${icon('add', {size: 'sm', className: 'dl-popover-create-icon'})}
      </button>
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
