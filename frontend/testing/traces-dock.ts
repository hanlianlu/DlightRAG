// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** What the browser tests of the Agent traces dock share: the dock mounted at a width, readers for its rows and
 *  pages, the timers a test holds, and the cleanup a test owes the next one. */

import {defineDesignSystemElements} from '../design-system/index.ts';
import {waitFor, buttonNamed} from './dom.ts';
import '../ui/inspector-traces.ts';
import type {TracesSource} from '../stores/traces-source.ts';
import type {DlInspectorTraces} from '../ui/inspector-traces.ts';

defineDesignSystemElements();

const originalFetch = window.fetch;

const originalNow = Date.now;

const originalSetTimeout = window.setTimeout;

const originalClearTimeout = window.clearTimeout;

const originalSetInterval = window.setInterval;

const originalClearInterval = window.clearInterval;

export function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return {promise, resolve};
}

export async function mount(source: TracesSource | null, width = 420): Promise<DlInspectorTraces> {
  const dock = document.createElement('dl-inspector-traces');
  // The dock measures itself: a pane at least 46rem (736px) wide shows the timeline beside the page.
  dock.style.cssText = `display:block;width:${width}px;height:600px`;
  dock.source = source;
  dock.active = true;
  document.body.append(dock);
  await dock.updateComplete;
  return dock;
}

/** The children's rows; the main agent heads the list and has its own tests. */
export function rows(dock: DlInspectorTraces): {id: string; text: string}[] {
  return [...dock.querySelectorAll<HTMLElement>('[data-agent-session]:not([data-agent-session="main"])')].map((button) => ({
    id: button.dataset.agentSession!,
    text: button.textContent!.replace(/\s+/g, ' ').trim(),
  }));
}

export function rowFor(dock: DlInspectorTraces, id: string): HTMLButtonElement {
  return dock.querySelector<HTMLButtonElement>(`[data-agent-session="${id}"]`)!;
}

export function session(dock: DlInspectorTraces) {
  return dock.querySelector('dl-agent-session')!;
}

/** The title of the child on show, once there is one. */
export function title(dock: DlInspectorTraces): string | undefined {
  return dock.querySelector('dl-agent-session h3')?.textContent?.trim();
}

/** The text of the child on show, as a reader takes it in. */
export function shown(dock: DlInspectorTraces): string {
  return (dock.querySelector('dl-agent-session')?.textContent ?? '').replace(/\s+/g, ' ').trim();
}

export function button(root: ParentNode, name: string): HTMLButtonElement | null {
  return buttonNamed<HTMLButtonElement>(root, name);
}

/** A batch that states something, the way the chat hands one to an open dock. */
export function stated(dock: DlInspectorTraces, runId: string): void {
  dock.applyEvents(runId, [{kind: 'tool', eventType: 'tool_start', payload: {}}]);
}

export async function openChild(dock: DlInspectorTraces, id: string): Promise<void> {
  rowFor(dock, id).click();
  await waitFor(() => (
    dock.querySelector(`[data-agent-session="${CSS.escape(id)}"][aria-current="true"]`) !== null
  ));
  await session(dock).updateComplete;
}

export const settle = (ms = 30) => new Promise<void>((resolve) => { originalSetTimeout(resolve, ms); });

/** Followed-activity refreshes share a one-second window with opening; run it at once. */
export function immediateFollowRefreshes(): void {
  window.setTimeout = ((handler: TimerHandler) => originalSetTimeout(handler, 0)) as typeof window.setTimeout;
}

/** Put back what a test of the dock leaves behind, after each test of the file that calls it. */
export function restoreDockFixtures(): void {
  afterEach(() => {
    // The dock remembers the agent each Run was left on; a test starts on the Run's own agent.
    for (const key of Object.keys(window.localStorage)) {
      if (key.startsWith('dlightrag-inspector-agent:')) window.localStorage.removeItem(key);
    }
    window.fetch = originalFetch;
    Date.now = originalNow;
    window.setTimeout = originalSetTimeout;
    window.clearTimeout = originalClearTimeout;
    window.setInterval = originalSetInterval;
    window.clearInterval = originalClearInterval;
    document.body.replaceChildren();
  });

  // ── The roster ──
}

/** Assistant messages numbered `from` to `to`, as the transcript route sends them. */
export function steps(from: number, to: number): Record<string, unknown>[] {
  return Array.from({length: to - from + 1}, (_, index) => (
    {sequence: from + index, role: 'assistant', content: `Step ${from + index}.`, tool_calls: []}
  ));
}
