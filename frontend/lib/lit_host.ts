// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {LitElement, type ReactiveController, type ReactiveControllerHost} from 'lit';
import {bus, type DlightragEvents} from '../events/bus.ts';

/**
 * Lit host that renders into itself instead of a shadow root.
 *
 * The stylesheets address global class names and inherit design tokens through
 * the cascade, so a shadow boundary would sever every rule that styles these
 * components. Rendering light DOM also keeps the tree reachable for the
 * modules that query it directly, such as MathJax and Mermaid.
 *
 * Reactive fields must be declared and assigned in the constructor rather than
 * initialised as class fields: under `[[Define]]` semantics a class field
 * shadows the accessor Lit installs on the prototype and updates stop firing.
 */
export abstract class LightElement extends LitElement {
    protected override createRenderRoot(): HTMLElement {
        return this;
    }
}

/** Re-renders its host on bus events, releasing the subscriptions on removal. */
export class BusController implements ReactiveController {
    readonly #host: ReactiveControllerHost;
    readonly #events: readonly (keyof DlightragEvents)[];
    #release: (() => void)[] = [];

    constructor(host: ReactiveControllerHost, ...events: (keyof DlightragEvents)[]) {
        this.#host = host;
        this.#events = events;
        host.addController(this);
    }

    hostConnected(): void {
        const rerender = (): void => { this.#host.requestUpdate(); };
        this.#release = this.#events.map((event) => bus.on(event, rerender));
    }

    hostDisconnected(): void {
        for (const release of this.#release) release();
        this.#release = [];
    }
}
