// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {html, nothing, svg, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import type {WorkspaceRecord} from '../events/bus.ts';
import {BusController, LightElement} from '../lib/lit_host.ts';
import {rovingArrowKeydown} from '../lib/listbox.ts';
import {createAutoDismiss} from '../lib/popover.ts';
import {ingestStore} from '../stores/ingestStore.ts';
import {workspaceStore} from '../stores/workspaceStore.ts';

const CARET = svg`<path d="M2.5 4 L5 6.5 L7.5 4"/>`;

/** Picks which workspace an upload lands in; shown only while Files is open. */
export class IngestTarget extends LightElement {
    static properties = {
        active: {attribute: false},
        open: {state: true},
    };

    declare active: boolean;
    declare open: boolean;

    readonly #dismiss = createAutoDismiss({
        getAnchor: () => this,
        isOpen: () => this.open,
        onDismiss: () => { this.open = false; },
    });

    constructor() {
        super();
        this.active = false;
        this.open = false;
        new BusController(
            this, 'workspaceCreated', 'workspaceDeleted', 'ingestWorkspaceChanged',
        );
    }

    override disconnectedCallback(): void {
        super.disconnectedCallback();
        this.#dismiss.deactivate();
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
        const select = (event: Event): void => {
            event.stopPropagation();
            this.open = false;
            ingestStore.set(record.workspace);
        };
        return html`
            <div
                class="ui-popover-item"
                role="option"
                tabindex="0"
                aria-selected=${selected ? 'true' : 'false'}
                @click=${select}
                @keydown=${(event: KeyboardEvent) => {
                    if (event.key !== 'Enter' && event.key !== ' ') return;
                    event.preventDefault();
                    select(event);
                }}
            >
                <div class="ingest-target-popover-radio${selected ? ' on' : ''}"></div>
                <span>${record.displayName}</span>
            </div>
        `;
    }

    #renderPopover() {
        const sorted = [...workspaceStore.records]
            .sort((left, right) => left.displayName.localeCompare(right.displayName));
        return html`
            <div
                class="ui-popover ui-popover--ingest"
                role="listbox"
                aria-label="Select ingest workspace"
                @keydown=${(event: KeyboardEvent) => { rovingArrowKeydown(event, '[role="option"]'); }}
            >
                ${repeat(sorted, (record) => record.workspace, (record) => this.#renderOption(record))}
                <workspace-create></workspace-create>
            </div>
        `;
    }

    protected override render(): TemplateResult | typeof nothing {
        if (!this.active) return nothing;
        const displayName = this.#displayName;
        const toggle = (event: Event): void => {
            event.stopPropagation();
            this.open = !this.open;
        };
        return html`
            <span class="ingest-target-label">Files in:</span>
            <span
                class="ingest-target-pill"
                role="button"
                tabindex="0"
                aria-label="Files in ${displayName}; choose file workspace"
                aria-expanded=${this.open ? 'true' : 'false'}
                @click=${toggle}
                @keydown=${(event: KeyboardEvent) => {
                    if (event.key !== 'Enter' && event.key !== ' ') return;
                    event.preventDefault();
                    toggle(event);
                }}
            >
                <span class="ingest-target-dot"></span>
                <span class="ingest-target-name">${displayName}</span>
                <span class="ingest-target-caret">
                    <svg width="8" height="8" viewBox="0 0 10 10" fill="none"
                         stroke="currentColor" stroke-width="1.5">${CARET}</svg>
                </span>
            </span>
            ${this.open ? this.#renderPopover() : nothing}
        `;
    }
}

customElements.define('ingest-target', IngestTarget);

declare global {
    interface HTMLElementTagNameMap {
        'ingest-target': IngestTarget;
    }
}
