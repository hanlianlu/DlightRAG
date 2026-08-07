// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {html, type TemplateResult} from 'lit';
import {WorkspaceApiError, createWorkspaceRequest} from '../api/workspaces.ts';
import {LightElement} from '../lib/lit_host.ts';
import {workspaceStore} from '../stores/workspaceStore.ts';
import {showToast} from './toast.ts';

/** The "New workspace…" row shared by the scope and ingest popovers. */
export class WorkspaceCreate extends LightElement {
    static properties = {pending: {state: true}};

    declare pending: boolean;

    constructor() {
        super();
        this.pending = false;
        this.className = 'ui-popover-create';
    }

    async #submit(): Promise<void> {
        const input = this.querySelector('input');
        const displayName = input?.value.trim();
        if (!input || !displayName || this.pending) return;
        this.pending = true;
        try {
            const created = await createWorkspaceRequest(displayName);
            input.value = '';
            workspaceStore.add({
                workspace: created.workspace,
                displayName: created.display_name,
                embeddingModel: '',
            });
        } catch (error) {
            showToast(
                error instanceof WorkspaceApiError ? error.message : 'Failed to create workspace',
                5000,
            );
        } finally {
            this.pending = false;
        }
    }

    protected override render(): TemplateResult {
        return html`
            <input
                class="ui-popover-input"
                type="text"
                placeholder="New workspace..."
                ?disabled=${this.pending}
                @click=${(event: Event) => { event.stopPropagation(); }}
                @keydown=${(event: KeyboardEvent) => {
                    if (event.key !== 'Enter') return;
                    event.preventDefault();
                    void this.#submit();
                }}
            >
            <button
                class="ui-popover-create-btn"
                type="button"
                ?disabled=${this.pending}
                @click=${(event: MouseEvent) => {
                    event.stopPropagation();
                    void this.#submit();
                }}
            >+</button>
        `;
    }
}

customElements.define('workspace-create', WorkspaceCreate);
