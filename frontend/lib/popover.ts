// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import type {ReactiveController, ReactiveControllerHost} from 'lit';
import {isImeKey} from '../design-system/index.ts';

interface TriggerPopoverOptions {
    /** The control the reader opened it with; closing by Escape or a choice gives focus back here. */
    trigger: () => HTMLElement | null;
    /** Move focus into the open popover; `which` is the end a menu button's key asked for. */
    enter: (which: 'first' | 'last') => void;
    /** Whether the host shows the surface the popover belongs to. A host that stops showing it has
     *  closed the popover: open is a state its surface shows, not one it remembers. Defaults to
     *  always shown, for a trigger that never leaves the page. */
    showing?: () => boolean;
}

/**
 * The state and dismissal of a popover its host opens from one trigger: it closes on a click
 * outside the host and on Escape, unless a modal dialog holds the page, when the host stops
 * showing the surface it belongs to, and when the host leaves the document. The host draws it
 * from `open`.
 *
 * Both listeners exist only while it is open. The outside click is armed on the next tick so the
 * click that opened it does not close it again.
 */
export class TriggerPopover implements ReactiveController {
    #open = false;
    readonly #host: ReactiveControllerHost & HTMLElement;
    readonly #options: TriggerPopoverOptions;
    #listening = false;
    #arming: ReturnType<typeof setTimeout> | undefined;

    constructor(host: ReactiveControllerHost & HTMLElement, options: TriggerPopoverOptions) {
        this.#host = host;
        this.#options = options;
        host.addController(this);
    }

    get open(): boolean {
        return this.#open;
    }

    /** The trigger's click. */
    toggle = (): void => {
        if (this.#open) this.close();
        else this.show();
    };

    show(which: 'first' | 'last' = 'first'): void {
        this.#open = true;
        this.#host.requestUpdate();
        void this.#host.updateComplete.then(() => { this.#options.enter(which); });
    }

    /** Close it; with `restoreFocus` the trigger takes focus back, unless the reader has moved on. */
    close(restoreFocus = false): void {
        this.#open = false;
        this.#host.requestUpdate();
        const active = document.activeElement;
        if (restoreFocus && (active === document.body || this.#host.contains(active))) {
            void this.#host.updateComplete.then(() => { this.#options.trigger()?.focus(); });
        }
    }

    /** A host that stops showing the surface has closed the popover for the render that hides it,
     *  so the state it would come back with is not a popover the reader left open. */
    hostUpdate(): void {
        if (this.#open && !(this.#options.showing?.() ?? true)) this.close();
    }

    hostUpdated(): void {
        if (this.#open) this.#listen();
        else this.#stop();
    }

    hostDisconnected(): void {
        if (this.#open) this.close();
        this.#stop();
    }

    #listen(): void {
        if (this.#listening) return;
        this.#listening = true;
        document.addEventListener('keydown', this.#escape, true);
        this.#arming = setTimeout(() => { document.addEventListener('click', this.#outside); }, 0);
    }

    #stop(): void {
        if (!this.#listening) return;
        this.#listening = false;
        clearTimeout(this.#arming);
        document.removeEventListener('click', this.#outside);
        document.removeEventListener('keydown', this.#escape, true);
    }

    readonly #outside = (event: MouseEvent): void => {
        if (event.target instanceof Node && !this.#host.contains(event.target)) this.close();
    };

    readonly #escape = (event: KeyboardEvent): void => {
        // This listener sees the key before the popover's own fields do, so an Escape that cancels a
        // composition in one of them is let through to it, not taken as the popover's dismissal.
        if (event.key !== 'Escape' || isImeKey(event) || !this.#open || document.querySelector('dialog[open]')) return;
        event.preventDefault();
        event.stopImmediatePropagation();
        this.close(true);
    };
}
