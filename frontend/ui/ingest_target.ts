// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, updateWhenLocaleChanges, str} from '@lit/localize';
import {html, nothing, svg, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import type {WorkspaceRecord} from '../events/bus.ts';
import {LightElement, StoreController} from '../lib/lit_host.ts';
import {rovingArrowKeydown} from '../lib/listbox.ts';
import {createAutoDismiss} from '../lib/popover.ts';
import {ingestStore} from '../stores/ingestStore.ts';
import {workspaceStore} from '../stores/workspaceStore.ts';
import './workspace_create.ts';

const CARET = svg`<path d="M2.5 4 L5 6.5 L7.5 4"/>`;

/** Picks which workspace an upload lands in; shown only while Files is open. */
export class DlIngestTarget extends LightElement {
    static properties = {
        active: {attribute: false},
        open: {state: true},
    };

    declare active: boolean;
    declare open: boolean;

    readonly #dismiss = createAutoDismiss({
        getAnchor: () => this,
        isOpen: () => this.open,
        onDismiss: (reason) => { this.#dismissPopover(reason === 'escape'); },
    });

    constructor() {
        super();
        updateWhenLocaleChanges(this);
        this.active = false;
        this.open = false;
        /** Store reads: workspaceStore.records, ingestStore.workspace. */
        new StoreController(this, workspaceStore, ingestStore);
    }

    override disconnectedCallback(): void {
        this.open = false;
        this.#dismiss.deactivate();
        super.disconnectedCallback();
    }

    close(): void {
        this.open = false;
    }

    protected override updated(): void {
        const showing = this.active && this.open;
        this.classList.toggle('open', showing);
        if (showing) this.#dismiss.activate();
        else this.#dismiss.deactivate();
    }

    get #displayName(): string {
        const workspace = ingestStore.workspace;
        return workspaceStore.records.find((r) => r.workspace === workspace)?.displayName
            ?? workspace;
    }

    #renderOption(record: WorkspaceRecord) {
        const selected = record.workspace === ingestStore.workspace;
        return html`
            <button
                class="ui-popover-item"
                type="button"
                data-ingest-workspace-choice
                aria-pressed=${selected ? 'true' : 'false'}
                @click=${(event: Event) => {
                    event.stopPropagation();
                    this.open = false;
                    ingestStore.set(record.workspace);
                    void this.updateComplete.then(() => { this.#trigger()?.focus(); });
                }}
            >
                <span class="ingest-target-popover-radio${selected ? ' on' : ''}"></span>
                <span>${record.displayName}</span>
            </button>
        `;
    }

    #renderPopover() {
        const sorted = [...workspaceStore.records]
            .sort((left, right) => left.displayName.localeCompare(right.displayName));
        return html`
            <div
                class="ui-popover ui-popover--ingest"
                id="ingest-target-popover"
                role="dialog"
                aria-label=${msg('Select ingest workspace', {id: 'ingestTarget.selectWorkspaceAria'})}
                ?hidden=${!this.active || !this.open}
                @keydown=${(event: KeyboardEvent) => {
                    rovingArrowKeydown(event, '[data-ingest-workspace-choice]');
                }}
            >
                ${repeat(sorted, (record) => record.workspace, (record) => this.#renderOption(record))}
                <dl-workspace-create @dl-workspace-created=${this.#workspaceCreated}></dl-workspace-create>
            </div>
        `;
    }

    protected override render(): TemplateResult | typeof nothing {
        const displayName = this.#displayName;
        return html`
            ${this.active ? html`
                <button
                    class="ingest-target-pill"
                    id="ingest-target-trigger"
                    type="button"
                    aria-label=${msg(str`Files in ${displayName}; choose file workspace`, {id: 'ingestTarget.filesInAria'})}
                    aria-haspopup="dialog"
                    aria-expanded=${this.open ? 'true' : 'false'}
                    aria-controls="ingest-target-popover"
                    @click=${this.#togglePopover}
                >
                    <span class="ingest-target-dot"></span>
                    <span class="ingest-target-name">${displayName}</span>
                    <span class="ingest-target-caret">
                        <svg width="8" height="8" viewBox="0 0 10 10" fill="none"
                             stroke="currentColor" stroke-width="1.5">${CARET}</svg>
                    </span>
                </button>
            ` : nothing}
            ${this.#renderPopover()}
        `;
    }

    #trigger(): HTMLButtonElement | null {
        return this.querySelector<HTMLButtonElement>('#ingest-target-trigger');
    }

    #togglePopover = (event: Event): void => {
        event.stopPropagation();
        if (this.open) {
            this.open = false;
            return;
        }
        this.open = true;
        void this.updateComplete.then(() => {
            const selected = this.querySelector<HTMLButtonElement>(
                '[data-ingest-workspace-choice][aria-pressed="true"]',
            );
            (selected ?? this.querySelector<HTMLButtonElement>(
                '[data-ingest-workspace-choice]',
            ))?.focus();
        });
    };

    #dismissPopover(restoreFocus: boolean): void {
        this.open = false;
        if (restoreFocus) {
            void this.updateComplete.then(() => { this.#trigger()?.focus(); });
        }
    }

    #workspaceCreated = (): void => {
        const active = document.activeElement;
        const restoreFocus = active === document.body || this.contains(active);
        this.open = false;
        if (restoreFocus) {
            void this.updateComplete.then(() => { this.#trigger()?.focus(); });
        }
    };
}

customElements.define('dl-ingest-target', DlIngestTarget);

declare global {
    interface HTMLElementTagNameMap {
        'dl-ingest-target': DlIngestTarget;
    }
}
