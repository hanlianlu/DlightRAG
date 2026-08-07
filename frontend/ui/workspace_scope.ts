// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {html, nothing, svg, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import type {WorkspaceRecord} from '../events/bus.ts';
import {BusController, LightElement} from '../lib/lit_host.ts';
import {rovingArrowKeydown} from '../lib/listbox.ts';
import {createAutoDismiss} from '../lib/popover.ts';
import {workspaceStore} from '../stores/workspaceStore.ts';
import workspaceStyles from '../styles/workspaces.module.css';

export interface WorkspaceDeleteDetail {
    workspace: string;
}

const CARET = svg`<path d="M2.5 4 L5 6.5 L7.5 4"/>`;

/** The search-scope control: derived label, and the multi-select popover. */
export class WorkspaceScope extends LightElement {
    static properties = {open: {state: true}};

    declare open: boolean;

    readonly #dismiss = createAutoDismiss({
        getAnchor: () => this,
        isOpen: () => this.open,
        onDismiss: () => { this.open = false; },
    });

    constructor() {
        super();
        this.open = false;
        new BusController(this, 'workspaceCreated', 'workspaceDeleted', 'workspaceToggled');
        this.addEventListener('click', (event) => { this.#toggleFromChrome(event); });
        this.addEventListener('keydown', (event) => {
            if (event.key !== 'Enter' && event.key !== ' ') return;
            this.#toggleFromChrome(event);
        });
    }

    #toggleFromChrome(event: Event): void {
        if (event.target instanceof Element && event.target.closest('.ui-popover')) return;
        event.preventDefault();
        this.open = !this.open;
    }

    override disconnectedCallback(): void {
        super.disconnectedCallback();
        this.#dismiss.deactivate();
    }

    close(): void {
        this.open = false;
    }

    protected override updated(): void {
        this.classList.toggle('open', this.open);
        if (this.open) this.#dismiss.activate();
        else this.#dismiss.deactivate();
    }

    get #allSelected(): boolean {
        const records = workspaceStore.records;
        return records.length > 0
            && records.every((record) => workspaceStore.active.includes(record.workspace));
    }

    get #label(): string {
        const total = workspaceStore.records.length;
        const active = workspaceStore.active;
        if (active.length === 0 || this.#allSelected) {
            return total > 0 ? `All workspaces (${total})` : 'All workspaces';
        }
        const anchor = active.includes(workspaceStore.primary) ? workspaceStore.primary : active[0];
        const name = workspaceStore.records.find((r) => r.workspace === anchor)?.displayName ?? anchor;
        return active.length === 1 ? name : `${name} + ${active.length - 1}`;
    }

    #check(selected: boolean) {
        return html`<div
            class="${workspaceStyles.workspacePopoverCheck}${selected ? ` ${workspaceStyles.on}` : ''}"
        ></div>`;
    }

    #renderAllOption() {
        const selected = this.#allSelected;
        const selectAll = (event: Event): void => {
            event.stopPropagation();
            workspaceStore.selectAll();
        };
        return html`
            <div
                class="ui-popover-item ${workspaceStyles.workspacePopoverAll}"
                role="option"
                tabindex="0"
                data-workspace-all="true"
                aria-selected=${selected ? 'true' : 'false'}
                @click=${selectAll}
                @keydown=${(event: KeyboardEvent) => {
                    if (event.key !== 'Enter' && event.key !== ' ') return;
                    event.preventDefault();
                    selectAll(event);
                }}
            >${this.#check(selected)}All workspaces</div>
        `;
    }

    #renderOption(record: WorkspaceRecord) {
        const selected = workspaceStore.active.includes(record.workspace);
        const toggle = (event: Event): void => {
            event.stopPropagation();
            workspaceStore.toggle(record.workspace);
        };
        return html`
            <div
                class="ui-popover-item ${workspaceStyles.workspacePopoverItem}"
                role="option"
                tabindex="0"
                aria-selected=${selected ? 'true' : 'false'}
                @click=${toggle}
                @keydown=${(event: KeyboardEvent) => {
                    if (event.key !== 'Enter' && event.key !== ' ') return;
                    event.preventDefault();
                    toggle(event);
                }}
            >
                ${this.#check(selected)}
                <span class=${workspaceStyles.workspacePopoverName}>${record.displayName}</span>
                <button
                    type="button"
                    class=${workspaceStyles.workspacePopoverDelete}
                    title="Delete workspace"
                    @click=${(event: MouseEvent) => {
                        event.stopPropagation();
                        this.open = false;
                        this.dispatchEvent(new CustomEvent<WorkspaceDeleteDetail>(
                            'workspace-delete',
                            {detail: {workspace: record.workspace}, bubbles: true, composed: true},
                        ));
                    }}
                >✕</button>
            </div>
        `;
    }

    #renderPopover() {
        const sorted = [...workspaceStore.records]
            .sort((left, right) => left.displayName.localeCompare(right.displayName));
        return html`
            <div
                class="ui-popover ui-popover--workspace"
                role="listbox"
                aria-label="Workspaces"
                @keydown=${(event: KeyboardEvent) => { rovingArrowKeydown(event, '[role="option"]'); }}
            >
                ${this.#renderAllOption()}
                ${repeat(sorted, (record) => record.workspace, (record) => this.#renderOption(record))}
                <workspace-create></workspace-create>
            </div>
        `;
    }

    protected override render(): TemplateResult {
        const multi = workspaceStore.active.length > 1 || this.#allSelected;
        return html`
            <span class="workspace-dot${multi ? ' multi' : ''}" id="workspace-dot"></span>
            <span class="workspace-label" id="workspace-label">${this.#label}</span>
            <svg class="workspace-caret" width="10" height="10" viewBox="0 0 10 10"
                 fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"
            >${CARET}</svg>
            ${this.open ? this.#renderPopover() : nothing}
        `;
    }
}

customElements.define('workspace-scope', WorkspaceScope);

declare global {
    interface HTMLElementTagNameMap {
        'workspace-scope': WorkspaceScope;
    }

    interface HTMLElementEventMap {
        'workspace-delete': CustomEvent<WorkspaceDeleteDetail>;
    }
}
