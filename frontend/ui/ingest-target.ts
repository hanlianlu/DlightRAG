// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, str} from '@lit/localize';
import {html, nothing, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import {icon, rovingFocusKeydown} from '../design-system/index.ts';
import {LightElement, StoreController} from '../lib/lit-host.ts';
import {TriggerPopover} from '../lib/popover.ts';
import {type AppHandles, productionHandles } from '../stores/app-handles.ts';
import type {WorkspaceRecord} from '../stores/workspace-store.ts';
import './workspace-create.ts';
import ingestStyles from '../styles/ingest-target.module.css';

/** Picks which workspace an upload lands in; shown only while Files is open. */
export class DlIngestTarget extends LightElement {
    static properties = {
        handles: {attribute: false},
        active: {attribute: false},
    };

    declare handles: AppHandles;
    declare active: boolean;

    readonly #popup = new TriggerPopover(this, {
        trigger: () => this.querySelector<HTMLButtonElement>('#ingest-target-trigger'),
        enter: () => {
            const choice = '[data-ingest-workspace-choice]';
            (this.querySelector<HTMLButtonElement>(`${choice}[aria-pressed="true"]`)
                ?? this.querySelector<HTMLButtonElement>(choice))?.focus();
        },
        showing: () => this.active,
    });

    constructor() {
        super();
        this.handles = productionHandles();
        this.active = false;
        /** Store reads: workspaces.records, ingest.workspace. */
        new StoreController(this, this.handles.workspaces, this.handles.ingest);
    }

    protected override updated(): void {
        this.classList.add(ingestStyles['ingest-target']);
        this.classList.toggle(ingestStyles.open, this.active && this.#popup.open);
    }

    get #displayName(): string {
        const workspace = this.handles.ingest.workspace;
        return this.handles.workspaces.records.find((r) => r.workspace === workspace)?.displayName
            ?? workspace;
    }

    #renderOption(record: WorkspaceRecord) {
        const selected = record.workspace === this.handles.ingest.workspace;
        return html`
            <button
                class="dl-popover-item"
                type="button"
                data-ingest-workspace-choice
                aria-pressed=${selected ? 'true' : 'false'}
                @click=${(event: Event) => {
                    event.stopPropagation();
                    this.handles.ingest.set(record.workspace);
                    this.#popup.close(true);
                }}
            >
                <span class=${`${ingestStyles['ingest-target-popover-radio']}${selected ? ` ${ingestStyles.on}` : ''}`}></span>
                <span>${record.displayName}</span>
            </button>
        `;
    }

    #renderPopover() {
        const sorted = [...this.handles.workspaces.records]
            .sort((left, right) => left.displayName.localeCompare(right.displayName));
        return html`
            <div
                class="dl-popover dl-popover--ingest dl-anchored"
                id="ingest-target-popover"
                role="dialog"
                aria-label=${msg('Select ingest workspace', {id: 'ingestTarget.selectWorkspaceAria'})}
                ?hidden=${!this.active || !this.#popup.open}
                @keydown=${(event: KeyboardEvent) => {
                    const popover = event.currentTarget as HTMLElement;
                    rovingFocusKeydown(
                        event,
                        [...popover.querySelectorAll<HTMLElement>('[data-ingest-workspace-choice]')],
                    );
                }}
            >
                ${repeat(sorted, (record) => record.workspace, (record) => this.#renderOption(record))}
                <dl-workspace-create .handles=${this.handles} @dl-workspace-created=${() => this.#popup.close(true)}></dl-workspace-create>
            </div>
        `;
    }

    protected override render(): TemplateResult | typeof nothing {
        const displayName = this.#displayName;
        return html`
            ${this.active ? html`
                <button
                    class=${ingestStyles['ingest-target-pill']}
                    id="ingest-target-trigger"
                    data-ingest-pill
                    type="button"
                    aria-label=${msg(str`Files in ${displayName}; choose file workspace`, {id: 'ingestTarget.filesInAria'})}
                    aria-haspopup="dialog"
                    aria-expanded=${this.#popup.open ? 'true' : 'false'}
                    aria-controls="ingest-target-popover"
                    @click=${this.#popup.toggle}
                >
                    <span class=${ingestStyles['ingest-target-dot']}></span>
                    <span class=${ingestStyles['ingest-target-name']} data-ingest-name>${displayName}</span>
                    <span class=${ingestStyles['ingest-target-caret']}>
                        ${icon('chevron-down', {size: 'xs'})}
                    </span>
                </button>
            ` : nothing}
            ${this.#renderPopover()}
        `;
    }
}

customElements.define('dl-ingest-target', DlIngestTarget);

declare global {
    interface HTMLElementTagNameMap {
        'dl-ingest-target': DlIngestTarget;
    }
}
