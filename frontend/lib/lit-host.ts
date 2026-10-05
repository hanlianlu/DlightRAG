// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {updateWhenLocaleChanges} from '@lit/localize';
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
    #lifetime = new AbortController();

    constructor() {
        super();
        // Every light element draws words, so none may stay behind when the language changes.
        updateWhenLocaleChanges(this);
        // Not connected yet, so nothing may start: the signal is spent until the first connect.
        this.#lifetime.abort();
    }

    /**
     * Aborts when this element leaves the document, so a request or a listener bound to it ends
     * with it. A new signal starts each time the element connects, and one that is not connected
     * holds a signal already aborted: work that checks `aborted` before it starts never starts.
     * Read it after `super.connectedCallback()`: before that the element is connected but the
     * signal is still the aborted one, and a listener registered with it silently does nothing.
     */
    protected get lifetime(): AbortSignal {
        return this.#lifetime.signal;
    }

    override connectedCallback(): void {
        this.#lifetime = new AbortController();
        super.connectedCallback();
    }

    override disconnectedCallback(): void {
        this.#lifetime.abort();
        super.disconnectedCallback();
    }

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

/**
 * Re-renders its host when a media query starts or stops matching, then calls `onChange`, so the
 * host's `updateComplete` there is the render that answers the change.
 */
export class MediaController implements ReactiveController {
    readonly #host: ReactiveControllerHost;
    readonly #query: string;
    readonly #onChange: (() => void) | undefined;
    #list: MediaQueryList | null = null;

    constructor(host: ReactiveControllerHost, query: string, onChange?: () => void) {
        this.#host = host;
        this.#query = query;
        this.#onChange = onChange;
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

    readonly #changed = (): void => {
        this.#host.requestUpdate();
        this.#onChange?.();
    };
}

/**
 * Re-renders its host when the host becomes narrower or wider than a width: a container query
 * for what only script can choose, such as which markup to draw. The host measures itself, so a
 * page in a pane answers for the pane and not for the viewport, and `rem` follows the reader's
 * text size. The host must be a box with a width (`display: block`). A host that is not showing
 * has no width to measure and keeps its last answer, which is wide until it has been measured.
 */
export class NarrowController implements ReactiveController {
    readonly #host: ReactiveControllerHost & HTMLElement;
    readonly #rem: number;
    #narrow = false;
    #observer: ResizeObserver | null = null;
    #frame = 0;

    constructor(host: ReactiveControllerHost & HTMLElement, rem: number) {
        this.#host = host;
        this.#rem = rem;
        host.addController(this);
    }

    /** Whether the host is narrower than the width it was given, as last measured. */
    get narrow(): boolean {
        return this.#narrow;
    }

    hostConnected(): void {
        this.#measure();
        this.#observer = new ResizeObserver(() => {
            // Drawing the other layout resizes the host in the very frame that reported its width, and
            // an observer that sees that loops; the next frame is the one to answer in.
            cancelAnimationFrame(this.#frame);
            this.#frame = requestAnimationFrame(() => {
                if (this.#measure()) this.#host.requestUpdate();
            });
        });
        this.#observer.observe(this.#host);
    }

    hostDisconnected(): void {
        cancelAnimationFrame(this.#frame);
        this.#observer?.disconnect();
        this.#observer = null;
    }

    /** Take the host's width; whether the answer changed. */
    #measure(): boolean {
        const width = this.#host.clientWidth;
        if (width === 0) return false;
        const narrow = width < this.#rem * Number.parseFloat(getComputedStyle(document.documentElement).fontSize);
        const changed = narrow !== this.#narrow;
        this.#narrow = narrow;
        return changed;
    }
}
