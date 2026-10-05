// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import type {ReactiveController, ReactiveControllerHost} from 'lit';

export interface TriggerPopoverOptions {
    /** The control the reader opened it with; closing by Escape or a choice gives focus back here. */
    trigger: () => HTMLElement | null;
    /** Move focus into the open popover; `which` is the end a menu button's key asked for. */
    enter: (which: 'first' | 'last') => void;
    /** Whether the host shows the popover at all while it is open. Defaults to always. */
    showing?: () => boolean;
}

/**
 * The state and dismissal of a popover its host opens from one trigger: it closes on a click
 * outside the host and on Escape, unless a modal dialog holds the page, and a host that leaves the
 * document closes it. The host draws it from `open`.
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

    hostUpdated(): void {
        if (this.#open && (this.#options.showing?.() ?? true)) this.#listen();
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
        if (event.key !== 'Escape' || !this.#open || document.querySelector('dialog[open]')) return;
        event.preventDefault();
        event.stopImmediatePropagation();
        this.close(true);
    };
}
