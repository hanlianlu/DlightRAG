// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {LitElement, type ReactiveController, type ReactiveControllerHost} from 'lit';
import type {SubscribableStore} from '../stores/base.ts';

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

/** Re-renders its host from the focused domain stores it actually reads. */
export class StoreController implements ReactiveController {
    readonly #host: ReactiveControllerHost;
    readonly #stores: readonly SubscribableStore[];
    #release: (() => void)[] = [];

    constructor(host: ReactiveControllerHost, ...stores: SubscribableStore[]) {
        this.#host = host;
        this.#stores = stores;
        host.addController(this);
    }

    hostConnected(): void {
        const rerender = (): void => { this.#host.requestUpdate(); };
        this.#release = this.#stores.map((store) => store.subscribe(rerender));
    }

    hostDisconnected(): void {
        for (const release of this.#release) release();
        this.#release = [];
    }
}

/** Re-renders its host when a media query starts or stops matching. */
export class MediaController implements ReactiveController {
    readonly #host: ReactiveControllerHost;
    readonly #query: string;
    #list: MediaQueryList | null = null;

    constructor(host: ReactiveControllerHost, query: string) {
        this.#host = host;
        this.#query = query;
        host.addController(this);
    }

    /** Whether the query matches now, before the host connects as well as after. */
    get matches(): boolean {
        return (this.#list ?? window.matchMedia(this.#query)).matches;
    }

    hostConnected(): void {
        this.#list = window.matchMedia(this.#query);
        this.#list.addEventListener('change', this.#changed);
    }

    hostDisconnected(): void {
        this.#list?.removeEventListener('change', this.#changed);
        this.#list = null;
    }

    readonly #changed = (): void => { this.#host.requestUpdate(); };
}
