// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

export type WebRoute =
  | {kind: 'new'}
  | {kind: 'conversation'; conversationId: string}
  | {kind: 'not-found'};

export type NavigationSource = 'push' | 'replace' | 'pop';
export type NavigationGuard = (
  next: WebRoute,
  current: WebRoute,
  source: NavigationSource,
) => boolean | Promise<boolean>;
export type RouteListener = (
  route: WebRoute,
  source: NavigationSource,
) => void | Promise<void>;

interface RouteHistoryState extends Record<string, unknown> {
  dlightragRouteIndex?: number;
}

export interface NavigationSnapshot {
  href: string;
  state: RouteHistoryState | null;
}

export interface NavigationDriver {
  current(): NavigationSnapshot;
  push(path: string, state: RouteHistoryState): void;
  replace(path: string, state: RouteHistoryState): void;
  go(delta: number): void;
  subscribe(listener: (snapshot: NavigationSnapshot) => void): () => void;
}

export interface NavigateOptions {
  replace?: boolean;
  notify?: boolean;
  bypassGuard?: boolean;
}

const CONVERSATION_ROUTE_PATTERN = new URLPattern({
  pathname: '/web/conversations/:conversationId',
});

export function newChatRoute(): WebRoute {
  return {kind: 'new'};
}

export function conversationRoute(conversationId: string): WebRoute {
  return {kind: 'conversation', conversationId};
}

export function routePath(route: WebRoute): string {
  if (route.kind === 'new') return '/web/';
  if (route.kind === 'conversation') {
    return `/web/conversations/${encodeURIComponent(route.conversationId)}`;
  }
  return '/web/not-found';
}

export function parseWebRoute(value: string | URL): WebRoute {
  const url = value instanceof URL ? value : new URL(value, 'http://dlightrag.local');
  if (url.pathname === '/web' || url.pathname === '/web/') return newChatRoute();

  const match = CONVERSATION_ROUTE_PATTERN.exec(url);
  const conversationId = match?.pathname.groups.conversationId;
  if (!conversationId) return {kind: 'not-found'};
  try {
    return conversationRoute(decodeURIComponent(conversationId));
  } catch {
    return {kind: 'not-found'};
  }
}

export function routesEqual(left: WebRoute, right: WebRoute): boolean {
  if (left.kind !== right.kind) return false;
  if (left.kind === 'conversation' && right.kind === 'conversation') {
    return left.conversationId === right.conversationId;
  }
  return true;
}

class BrowserNavigationDriver implements NavigationDriver {
  readonly #window: Window;

  constructor(target: Window) {
    this.#window = target;
  }

  current(): NavigationSnapshot {
    return {
      href: this.#window.location.href,
      state: this.#window.history.state as RouteHistoryState | null,
    };
  }

  push(path: string, state: RouteHistoryState): void {
    this.#window.history.pushState(state, '', path);
  }

  replace(path: string, state: RouteHistoryState): void {
    this.#window.history.replaceState(state, '', path);
  }

  go(delta: number): void {
    this.#window.history.go(delta);
  }

  subscribe(listener: (snapshot: NavigationSnapshot) => void): () => void {
    const handle = (event: PopStateEvent): void => {
      listener({href: this.#window.location.href, state: event.state as RouteHistoryState | null});
    };
    this.#window.addEventListener('popstate', handle);
    return () => { this.#window.removeEventListener('popstate', handle); };
  }
}

export class WebRouter {
  readonly #driver: NavigationDriver;
  #route: WebRoute;
  #index = 0;
  #listener: RouteListener | null = null;
  #guard: NavigationGuard | null = null;
  #release: (() => void) | null = null;
  #guardingPop = false;
  #restoreChain: Promise<void> = Promise.resolve();
  #expectedPop: {
    index: number;
    resolve: (snapshot: NavigationSnapshot) => void;
  } | null = null;

  constructor(driver: NavigationDriver) {
    this.#driver = driver;
    const current = driver.current();
    this.#route = parseWebRoute(current.href);
    this.#index = current.state?.dlightragRouteIndex ?? 0;
    driver.replace(routePath(this.#route), {
      ...(current.state ?? {}),
      dlightragRouteIndex: this.#index,
    });
  }

  get current(): WebRoute {
    return this.#route;
  }

  setGuard(guard: NavigationGuard | null): void {
    this.#guard = guard;
  }

  start(listener: RouteListener): () => void {
    this.#listener = listener;
    this.#release?.();
    this.#release = this.#driver.subscribe((snapshot) => { void this.#onPop(snapshot); });
    return () => {
      this.#release?.();
      this.#release = null;
      if (this.#listener === listener) this.#listener = null;
    };
  }

  async navigate(route: WebRoute, options: NavigateOptions = {}): Promise<boolean> {
    if (this.#guardingPop) return false;
    const source: NavigationSource = options.replace ? 'replace' : 'push';
    this.#guardingPop = true;
    try {
      if (
        !options.bypassGuard
        && !routesEqual(route, this.#route)
        && this.#guard
        && !await this.#guard(route, this.#route, source)
      ) {
        return false;
      }

      const nextIndex = options.replace ? this.#index : this.#index + 1;
      const state = {dlightragRouteIndex: nextIndex};
      if (options.replace) this.#driver.replace(routePath(route), state);
      else this.#driver.push(routePath(route), state);
      this.#route = route;
      this.#index = nextIndex;
    } finally {
      this.#guardingPop = false;
    }
    if (options.notify !== false) await this.#listener?.(route, source);
    return true;
  }

  async #onPop(snapshot: NavigationSnapshot): Promise<void> {
    const arrivedIndex = snapshot.state?.dlightragRouteIndex;
    if (this.#expectedPop && arrivedIndex === this.#expectedPop.index) {
      const expected = this.#expectedPop;
      this.#expectedPop = null;
      expected.resolve(snapshot);
      return;
    }

    const route = parseWebRoute(snapshot.href);
    const targetIndex = snapshot.state?.dlightragRouteIndex;
    if (!this.#guard || routesEqual(route, this.#route)) {
      await this.#applyPop(route, targetIndex);
      return;
    }

    if (this.#guardingPop) {
      await this.#enqueueRestore(snapshot);
      return;
    }

    this.#guardingPop = true;
    try {
      // popstate fires after the browser has already changed the URL. Restore
      // the stable entry before opening an async confirmation, then travel to
      // the target exactly once only if the guard accepts it.
      await this.#enqueueRestore(snapshot);
      const allowed = await this.#guard(route, this.#route, 'pop');
      await this.#restoreChain;
      if (!allowed) return;

      if (typeof targetIndex === 'number') {
        const accepted = await this.#travel(this.#index, targetIndex);
        await this.#applyPop(parseWebRoute(accepted.href), targetIndex);
      } else {
        const nextIndex = this.#index + 1;
        this.#driver.push(routePath(route), {dlightragRouteIndex: nextIndex});
        this.#index = nextIndex;
        this.#route = route;
        await this.#listener?.(route, 'pop');
      }
    } finally {
      this.#guardingPop = false;
    }
  }

  #enqueueRestore(snapshot: NavigationSnapshot): Promise<void> {
    this.#restoreChain = this.#restoreChain.then(async () => {
      const targetIndex = snapshot.state?.dlightragRouteIndex;
      if (typeof targetIndex === 'number' && targetIndex !== this.#index) {
        await this.#travel(targetIndex, this.#index);
      } else {
        this.#driver.push(routePath(this.#route), {dlightragRouteIndex: this.#index});
      }
    });
    return this.#restoreChain;
  }

  #travel(fromIndex: number, toIndex: number): Promise<NavigationSnapshot> {
    if (fromIndex === toIndex) return Promise.resolve(this.#driver.current());
    return new Promise((resolve) => {
      this.#expectedPop = {index: toIndex, resolve};
      this.#driver.go(toIndex - fromIndex);
    });
  }

  async #applyPop(route: WebRoute, targetIndex: number | undefined): Promise<void> {
    this.#route = route;
    if (typeof targetIndex === 'number') this.#index = targetIndex;
    await this.#listener?.(route, 'pop');
  }
}

export function createBrowserRouter(target: Window): WebRouter {
  return new WebRouter(new BrowserNavigationDriver(target));
}
