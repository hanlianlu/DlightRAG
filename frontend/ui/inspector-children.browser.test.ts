// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {sendKeys, setViewport} from '@web/test-runner-commands';
import {CHILD_TRANSCRIPT_LIMIT} from '../api/conversations.ts';
import {defineDesignSystemElements} from '../design-system/index.ts';
import {NOW, ago, observation, question, receipt, refusal, roster, row, serve, sourceFor} from '../testing/children.ts';
import {linkStyles, waitFor} from '../testing/dom.ts';
import './inspector-children.ts';
import type {ChildrenSource, DlInspectorChildren} from './inspector-children.ts';

defineDesignSystemElements();

const originalFetch = window.fetch;
const originalNow = Date.now;
const originalSetTimeout = window.setTimeout;
const originalClearTimeout = window.clearTimeout;
const originalSetInterval = window.setInterval;
const originalClearInterval = window.clearInterval;

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return {promise, resolve};
}

async function mount(source: ChildrenSource | null, width = 420): Promise<DlInspectorChildren> {
  const dock = document.createElement('dl-inspector-children');
  // The dock measures itself: a pane at least 40rem wide shows the list beside the child.
  dock.style.cssText = `display:block;width:${width}px;height:600px`;
  dock.source = source;
  dock.active = true;
  document.body.append(dock);
  await dock.updateComplete;
  return dock;
}

function rows(dock: DlInspectorChildren): {id: string; text: string}[] {
  return [...dock.querySelectorAll<HTMLElement>('[data-child-session]')].map((button) => ({
    id: button.dataset.childSession!,
    text: button.textContent!.replace(/\s+/g, ' ').trim(),
  }));
}

function rowFor(dock: DlInspectorChildren, id: string): HTMLButtonElement {
  return dock.querySelector<HTMLButtonElement>(`[data-child-session="${id}"]`)!;
}

function session(dock: DlInspectorChildren) {
  return dock.querySelector('dl-child-session')!;
}

/** The title of the child on show, once there is one. */
function title(dock: DlInspectorChildren): string | undefined {
  return dock.querySelector('dl-child-session h3')?.textContent?.trim();
}

/** The text of the child on show, as a reader takes it in. */
function shown(dock: DlInspectorChildren): string {
  return session(dock).textContent!.replace(/\s+/g, ' ').trim();
}

function composer(dock: DlInspectorChildren): HTMLTextAreaElement {
  return session(dock).querySelector<HTMLTextAreaElement>('form textarea[data-draft]:not([data-reply])')!;
}

function button(root: ParentNode, name: string): HTMLButtonElement | null {
  return [...root.querySelectorAll<HTMLButtonElement>('button')].find((candidate) => (
    (candidate.getAttribute('aria-label') ?? candidate.textContent!.trim()) === name
  )) ?? null;
}

async function type(dock: DlInspectorChildren, field: HTMLTextAreaElement, text: string): Promise<void> {
  field.value = text;
  field.dispatchEvent(new Event('input', {bubbles: true}));
  await session(dock).updateComplete;
}

/** Watch the writes the page makes to a box's value. What the browser itself puts in the box, as an
 *  IME does while the reader composes, goes through `enter` and is not counted. */
function watchWrites(field: HTMLTextAreaElement) {
  const native = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')!;
  const writes: string[] = [];
  Object.defineProperty(field, 'value', {
    configurable: true,
    get: () => native.get!.call(field) as string,
    set: (value: string) => {
      writes.push(value);
      native.set!.call(field, value);
    },
  });
  const enter = (text: string, caret: number, composing: boolean) => {
    native.set!.call(field, text);
    field.setSelectionRange(caret, caret);
    field.dispatchEvent(new InputEvent('input', {
      inputType: 'insertCompositionText', isComposing: composing, bubbles: true,
    }));
  };
  return {writes, enter};
}

async function openChild(dock: DlInspectorChildren, id: string): Promise<void> {
  rowFor(dock, id).click();
  await waitFor(() => session(dock).querySelector('[data-draft]') !== null);
}

const settle = (ms = 30) => new Promise<void>((resolve) => { originalSetTimeout(resolve, ms); });

/** Followed-activity refreshes share a one-second window with opening; run it at once. */
function immediateFollowRefreshes(): void {
  window.setTimeout = ((handler: TimerHandler) => originalSetTimeout(handler, 0)) as typeof window.setTimeout;
}

/** Record the dock's timers instead of running them. */
function recordTimers() {
  const timers = new Map<number, {handler: () => void; delay: number}>();
  const cleared: number[] = [];
  window.setTimeout = ((handler: TimerHandler, delay?: number) => {
    const id = timers.size + 1;
    timers.set(id, {handler: handler as () => void, delay: delay ?? 0});
    return id;
  }) as typeof window.setTimeout;
  window.clearTimeout = ((id?: number) => { if (id !== undefined) cleared.push(id); }) as typeof window.clearTimeout;
  return {timers, cleared};
}

afterEach(() => {
  window.fetch = originalFetch;
  Date.now = originalNow;
  window.setTimeout = originalSetTimeout;
  window.clearTimeout = originalClearTimeout;
  window.setInterval = originalSetInterval;
  window.clearInterval = originalClearInterval;
  document.body.replaceChildren();
});

// ── The roster ──

it('lists each child with its state, how long it ran, and a settled child\'s summary', async () => {
  Date.now = () => NOW;
  serve({page: () => roster([
    row('a', 'running', {started_at: ago(134), pending_questions: 1}),
    row('b', 'succeeded', {started_at: ago(68), finished_at: ago(0), summary: 'Clause 9.2 caps the penalty.'}),
    row('c', 'failed', {started_at: ago(41), finished_at: ago(0), summary: 'The PDF has no text layer.'}),
    row('d', 'cancelled', {cancellation_origin: 'user', started_at: ago(12), finished_at: ago(0)}),
    row('e', 'cancelled', {cancellation_origin: 'parent'}),
    row('f', 'cancelled', {cancellation_origin: 'run'}),
  ])});
  const dock = await mount(sourceFor().source);
  await waitFor(() => rows(dock).length === 6);

  expect(rows(dock).map((child) => child.text)).to.deep.equal([
    'objective a Running · 2m 14s Question',
    'objective b Done · 1m 8s Clause 9.2 caps the penalty.',
    'objective c Failed · 41s The PDF has no text layer.',
    'objective d Cancelled by you · 12s',
    'objective e Cancelled by the agent',
    'objective f Stopped with the run',
  ]);
  expect(dock.querySelector('[data-load-older="children"]')).to.equal(null);
  expect(dock.textContent).to.contain('6 children');
  expect(dock.textContent).to.contain('1 running');
});

it('appends older pages without repeating a child, one request at a time', async () => {
  const older = deferred<Response>();
  let olderRequests = 0;
  serve({page: (cursor) => {
    if (cursor === null) return roster([row('newest')], 'older-1');
    olderRequests += 1;
    return older.promise;
  }});
  const dock = await mount(sourceFor().source);
  await waitFor(() => rows(dock).length === 1);

  const more = dock.querySelector<HTMLButtonElement>('[data-load-older="children"]')!;
  expect(more.textContent!.trim()).to.equal('Load older children');
  more.click();
  more.click();
  await dock.updateComplete;
  expect(olderRequests).to.equal(1);
  expect(more.getAttribute('aria-disabled')).to.equal('true');

  older.resolve(roster([row('newest'), row('older')]));
  await waitFor(() => rows(dock).length === 2);

  expect(rows(dock).map((child) => child.id)).to.deep.equal(['newest', 'older']);
  expect(dock.querySelector('[data-load-older="children"]')).to.equal(null);
  expect(dock.querySelector('[data-load-older-status="children"]')!.textContent).to.contain('Loaded 1 older child.');
});

it('keeps the loaded children when an older page fails, and offers it again', async () => {
  let attempts = 0;
  serve({page: (cursor) => {
    if (cursor === null) return roster([row('newest')], 'older-1');
    attempts += 1;
    return attempts === 1 ? new Response('unavailable', {status: 503}) : roster([row('older')]);
  }});
  const dock = await mount(sourceFor().source);
  await waitFor(() => rows(dock).length === 1);

  dock.querySelector<HTMLButtonElement>('[data-load-older="children"]')!.click();
  await waitFor(() => dock.querySelector('[data-load-older="children"]')?.textContent?.includes('Retry loading older children') === true);
  expect(rows(dock)).to.have.length(1);

  dock.querySelector<HTMLButtonElement>('[data-load-older="children"]')!.click();
  await waitFor(() => rows(dock).length === 2);
  expect(dock.querySelector('[data-load-older="children"]')).to.equal(null);
});

it('starts its traversal over on a refresh and drops an older page that arrives late', async () => {
  immediateFollowRefreshes();
  const older = deferred<Response>();
  let firstPages = 0;
  serve({page: (cursor) => {
    if (cursor !== null) return older.promise;
    firstPages += 1;
    return roster([row(firstPages === 1 ? 'first' : 'fresh')], 'older-1');
  }});
  const dock = await mount(sourceFor().source);
  await waitFor(() => rows(dock).length === 1);

  dock.querySelector<HTMLButtonElement>('[data-load-older="children"]')!.click();
  dock.refreshIfFollowing('run-1');
  await waitFor(() => firstPages === 2);
  older.resolve(roster([row('stale')]));
  await waitFor(() => rows(dock)[0]?.id === 'fresh');
  await settle();

  expect(rows(dock).map((child) => child.id)).to.deep.equal(['fresh']);
  expect(dock.querySelector('[data-load-older="children"]')!.textContent).to.contain('Load older children');
});

it('forgets its Run when the source goes: a page in flight is dropped and nothing is listed', async () => {
  const older = deferred<Response>();
  serve({page: (cursor) => (cursor === null ? roster([row('newest')], 'older-1') : older.promise)});
  const dock = await mount(sourceFor().source);
  await waitFor(() => rows(dock).length === 1);
  dock.querySelector<HTMLButtonElement>('[data-load-older="children"]')!.click();

  dock.source = null;
  await dock.updateComplete;
  older.resolve(roster([row('stale')]));
  await settle();

  expect(rows(dock)).to.have.length(0);
  expect(dock.textContent!.trim()).to.equal('');
});

it('says so when the first page cannot be loaded, and loads again on Retry', async () => {
  let attempts = 0;
  serve({page: () => {
    attempts += 1;
    return attempts === 1 ? new Response('unavailable', {status: 503}) : roster([row('a')]);
  }});
  const dock = await mount(sourceFor().source);
  await waitFor(() => dock.querySelector('[role="alert"]') !== null);

  expect(dock.querySelector('[role="alert"]')!.textContent!.trim()).to.equal('Child agents could not be loaded.');
  expect(rows(dock)).to.have.length(0);

  button(dock, 'Retry')!.click();
  await waitFor(() => rows(dock).length === 1);
  expect(dock.querySelector('[role="alert"]')).to.equal(null);
});

it('says when no child was started', async () => {
  serve({page: () => roster([])});
  const dock = await mount(sourceFor().source);
  await waitFor(() => dock.textContent!.includes('No child agents were started'));

  expect(dock.textContent).to.contain('They appear here when the agent splits a task.');
  expect(dock.querySelector('dl-child-session')).to.equal(null);
});

// ── Following the Run ──

it('refetches a followed roster at most once per interval, counting its opening', async () => {
  const {timers} = recordTimers();
  let pages = 0;
  serve({page: () => {
    pages += 1;
    return roster([row('a', 'running')]);
  }});
  const dock = await mount(sourceFor().source);
  await waitFor(() => pages === 1);

  for (let activity = 0; activity < 30; activity += 1) dock.refreshIfFollowing('run-1');
  dock.refreshIfFollowing('another-run');
  await settle(20);

  expect(pages).to.equal(1, 'activity right after opening waits for the window');
  const trailing = [...timers.values()].filter(({delay}) => delay > 0);
  expect(trailing).to.have.length(1, 'one trailing refresh carries it all');
  expect(trailing[0]!.delay).to.be.at.most(1000);
  trailing[0]!.handler();
  await waitFor(() => pages === 2);

  // That refresh opened a window of its own.
  dock.refreshIfFollowing('run-1');
  expect([...timers.values()].filter(({delay}) => delay > 0)).to.have.length(2);
});

it('drops a trailing refresh when the source goes or the dock leaves the page', async () => {
  const {timers, cleared} = recordTimers();
  let pages = 0;
  serve({page: () => {
    pages += 1;
    return roster([row('a', 'running')]);
  }});
  const dock = await mount(sourceFor().source);
  await waitFor(() => pages === 1);

  dock.refreshIfFollowing('run-1');
  const closing = [...timers.keys()].at(-1)!;
  dock.source = null;
  await dock.updateComplete;
  expect(cleared).to.include(closing);

  dock.source = sourceFor().source;
  await waitFor(() => pages === 2);
  dock.refreshIfFollowing('run-1');
  const leaving = [...timers.keys()].at(-1)!;
  dock.remove();
  expect(cleared).to.include(leaving);
});

it('does not follow while another Inspector kind is showing', async () => {
  immediateFollowRefreshes();
  let pages = 0;
  serve({page: () => {
    pages += 1;
    return roster([row('a', 'running')]);
  }});
  const dock = await mount(sourceFor().source);
  await waitFor(() => pages === 1);

  dock.active = false;
  await dock.updateComplete;
  dock.refreshIfFollowing('run-1');
  await settle(50);

  expect(pages).to.equal(1);
});

it('runs one clock that redraws the elapsed times without asking the server, and only while it is needed', async () => {
  let now = NOW;
  Date.now = () => now;
  const intervals = new Map<number, () => void>();
  const stopped: number[] = [];
  window.setInterval = ((handler: TimerHandler) => {
    const id = intervals.size + 100;
    intervals.set(id, handler as () => void);
    return id;
  }) as typeof window.setInterval;
  window.clearInterval = ((id?: number) => { if (id !== undefined) stopped.push(id); }) as typeof window.clearInterval;
  const requests = serve({page: () => roster([
    row('a', 'running', {started_at: ago(134)}),
    row('b', 'succeeded', {started_at: ago(600), finished_at: ago(540)}),
  ])});
  const dock = await mount(sourceFor().source);
  await waitFor(() => rows(dock).length === 2);

  expect(intervals.size).to.equal(1, 'one clock for every row and the child on show');
  expect(rows(dock)[0]!.text).to.contain('2m 14s');
  now += 5000;
  [...intervals.values()][0]!();
  await dock.updateComplete;
  expect(rows(dock)[0]!.text).to.contain('2m 19s');
  expect(rows(dock)[1]!.text).to.contain('1m 0s', 'a settled child\'s time does not move');
  expect(requests).to.have.length(1);

  dock.active = false;
  await dock.updateComplete;
  expect(stopped).to.deep.equal([100]);
});

it('keeps no clock when nothing runs', async () => {
  const intervals: number[] = [];
  window.setInterval = ((_handler: TimerHandler) => intervals.push(1)) as typeof window.setInterval;
  serve({page: () => roster([row('a', 'succeeded')])});
  const dock = await mount(sourceFor().source);
  await waitFor(() => rows(dock).length === 1);

  expect(intervals).to.have.length(0);
});

// ── One child at a time, or beside the list ──

it('shows the list or one child in a narrow dock, and Back returns to the row that opened it', async () => {
  serve({
    page: () => roster([row('a', 'running'), row('b', 'succeeded')]),
    observe: (id) => observation(row(id, id === 'a' ? 'running' : 'succeeded')),
  });
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 2);
  const listPane = dock.querySelector<HTMLElement>('ul')!.parentElement!;
  expect(listPane.hidden).to.equal(false);
  expect(session(dock).hidden).to.equal(true);

  rowFor(dock, 'b').click();
  await waitFor(() => title(dock) === 'objective b');
  expect(listPane.hidden).to.equal(true);
  expect(session(dock).hidden).to.equal(false);
  expect(button(dock, 'All child agents')).to.not.equal(null);
  await waitFor(() => document.activeElement === session(dock).querySelector('h3'));
  expect(rowFor(dock, 'b').getAttribute('aria-current')).to.equal('true');

  button(dock, 'All child agents')!.click();
  await dock.updateComplete;
  expect(listPane.hidden).to.equal(false);
  expect(session(dock).hidden).to.equal(true);
  await waitFor(() => document.activeElement === rowFor(dock, 'b'));
  expect(button(dock, 'All child agents')).to.equal(null);
});

it('shows the list beside the newest child in a wide dock, and the child the reader picks after that', async () => {
  serve({
    page: () => roster([row('a', 'running'), row('b', 'succeeded')]),
    observe: (id) => observation(row(id, id === 'a' ? 'running' : 'succeeded')),
  });
  const dock = await mount(sourceFor().source, 800);
  await waitFor(() => rows(dock).length === 2);
  await waitFor(() => title(dock) === 'objective a');

  expect(rowFor(dock, 'a').getAttribute('aria-current')).to.equal('true');
  expect(button(dock, 'All child agents')).to.equal(null, 'both are showing, so there is nothing to go back to');
  expect(dock.querySelector<HTMLElement>('ul')!.parentElement!.hidden).to.equal(false);

  rowFor(dock, 'b').click();
  await waitFor(() => title(dock) === 'objective b');
  expect(rowFor(dock, 'b').getAttribute('aria-current')).to.equal('true');
  expect(rowFor(dock, 'a').hasAttribute('aria-current')).to.equal(false);
});

it('follows the width of the pane: narrower shows the child the reader picked, wider shows the list too', async () => {
  serve({
    page: () => roster([row('a', 'running'), row('b', 'succeeded')]),
    observe: (id) => observation(row(id, id === 'a' ? 'running' : 'succeeded')),
  });
  const dock = await mount(sourceFor().source, 800);
  await waitFor(() => title(dock) === 'objective a');
  const listPane = dock.querySelector<HTMLElement>('ul')!.parentElement!;
  rowFor(dock, 'b').click();
  await waitFor(() => title(dock) === 'objective b');

  dock.style.width = '420px';
  await waitFor(() => listPane.hidden === true);
  expect(button(dock, 'All child agents')).to.not.equal(null);
  expect(title(dock)).to.equal('objective b');

  dock.style.width = '800px';
  await waitFor(() => listPane.hidden === false);
  expect(button(dock, 'All child agents')).to.equal(null);
});

// ── The child on show ──

it('shows a running child\'s question with when it expires, and its activity as the main trace words it', async () => {
  Date.now = () => NOW;
  serve({
    page: () => roster([row('a', 'running', {started_at: ago(134), pending_questions: 1})]),
    observe: () => observation(row('a', 'running', {started_at: ago(134), pending_questions: 1}), {
      transcript: [
        {role: 'user', content: 'objective a'},
        {role: 'assistant', content: '', tool_calls: [{id: 't1', name: 'search_knowledge_base'}]},
        {role: 'tool', content: '12 results\nTop: MSA §9.2', tool_call_id: 't1', name: 'search_knowledge_base', is_error: false},
        {role: 'assistant', content: 'Checking the amendment.', tool_calls: [{id: 't2', name: 'read'}]},
      ],
      questions: [
        question('q1'),
        question('q0', {status: 'replied', reply: 'Swedish law.', reply_origin: 'parent', expires_at: null}),
        question('q9', {expires_at: ago(5)}),
      ],
    }),
  });
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');
  await waitFor(() => shown(dock).includes('Checking the amendment'));

  const text = shown(dock);
  expect(text).to.contain('Running 2m 14s · Query model');
  expect(text).to.contain('Asking the parent · expires in 4 min Question q1? Answer instead');
  expect(text.match(/Asking the parent/g), 'a question past its expiry is no longer waiting').to.have.length(1);
  expect(text).to.contain('Activity Searching the knowledge base 12 results');
  expect(text).to.contain('Checking the amendment. Running: Reading a document');
  expect(session(dock).querySelector('details')!.open, 'a running child\'s activity is open').to.equal(true);
  // Quiet history: what was answered, and what lapsed.
  expect(text).to.contain('Question q0? Answered · Parent Swedish law.');
  expect(text).to.contain('Question q9? Expired');
});

it('leads a settled child with its result, and folds its evidence and activity', async () => {
  const settled = row('a', 'succeeded', {
    started_at: ago(68), finished_at: ago(0), usage: {total_tokens: 9400}, summary: 'Clause 9.2 caps it at 6%.',
    result_handles: ['ev-1', 'ev-2'],
  });
  serve({
    page: () => roster([settled]),
    observe: () => observation(settled, {
      transcript: [
        {role: 'user', content: 'objective a'},
        {role: 'assistant', content: '', tool_calls: [{id: 't1', name: 'read'}]},
        {role: 'tool', content: 'ok', tool_call_id: 't1', name: 'read', is_error: false},
      ],
      result: {status: 'succeeded', summary: 'Clause 9.2 caps it at 6%.', handles: ['ev-1', 'ev-2'], operation_id: 'op-a'},
    }),
  });
  Date.now = () => NOW;
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');
  await waitFor(() => shown(dock).includes('Activity · 1 step'));

  const text = shown(dock);
  expect(text.indexOf('Result')).to.be.lessThan(text.indexOf('Activity'));
  expect(text).to.contain('Done 1m 8s · Query model · 9.4K tokens');
  expect(text).to.contain('Result Clause 9.2 caps it at 6%.');
  expect(text).to.contain('Evidence · 2');
  const [evidence, activity] = session(dock).querySelectorAll('details');
  expect(evidence!.open).to.equal(false);
  expect(activity!.open, 'a settled child\'s activity is folded').to.equal(false);
  expect(evidence!.textContent).to.contain('ev-1');
  expect(button(session(dock), 'Cancel child'), 'nothing is left to cancel').to.equal(null);
});

it('says its steps are the latest ones once the transcript fills the page it asked for', async () => {
  immediateFollowRefreshes();
  const says = (count: number) => Array.from({length: count}, (_, index) => (
    {role: 'assistant', content: `Step ${index}`, tool_calls: []}
  ));
  let count = CHILD_TRANSCRIPT_LIMIT - 1;
  const running = row('a', 'running', {started_at: ago(5)});
  const requests = serve({
    page: () => roster([running]),
    observe: () => observation(running, {transcript: says(count)}),
  });
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');
  await waitFor(() => shown(dock).includes(`Step ${CHILD_TRANSCRIPT_LIMIT - 2}`));

  expect(shown(dock)).to.not.contain('Latest');
  expect(requests.find((request) => request.path.endsWith('/children/a'))!.search)
    .to.equal(`?limit=${CHILD_TRANSCRIPT_LIMIT}`);

  count = CHILD_TRANSCRIPT_LIMIT;
  dock.refreshIfFollowing('run-1');
  await waitFor(() => shown(dock).includes(`Latest ${CHILD_TRANSCRIPT_LIMIT} steps`));
});

it('shows a token count only when the child reports a finite total', async () => {
  Date.now = () => NOW;
  const read = async (usage: Record<string, number> | null) => {
    const settled = row('a', 'succeeded', {started_at: ago(10), finished_at: ago(0), usage});
    serve({page: () => roster([settled]), observe: () => observation(settled)});
    const dock = await mount(sourceFor().source, 420);
    await waitFor(() => rows(dock).length === 1);
    await openChild(dock, 'a');
    await waitFor(() => shown(dock).includes('Done'));
    const text = shown(dock);
    dock.remove();
    return text;
  };

  expect(await read({input_tokens: 5, output_tokens: 7})).to.not.contain('tokens');
  expect(await read(null)).to.not.contain('tokens');
  expect(await read({total_tokens: 12})).to.contain('Done 10s · Query model · 12 tokens');
});

// ── Commands ──

it('sends a steer to a running child with the Operation it was typed for, and says what became of it', async () => {
  const running = row('a', 'running', {started_at: ago(5)});
  const requests = serve({
    page: () => roster([running]),
    observe: () => observation(running),
    control: () => receipt('steer', 'queued'),
  });
  const {source, controls} = sourceFor();
  const dock = await mount(source, 420);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');

  expect(shown(dock)).to.contain("Delivered at the child's next safe checkpoint.");
  const field = composer(dock);
  expect(field.placeholder).to.equal('Steer this child…');
  field.focus();
  await type(dock, field, '  focus on dates ');
  session(dock).querySelector('form')!.requestSubmit();
  await waitFor(() => shown(dock).includes('Queued. The child has not necessarily followed it yet.'));

  expect(controls.map(([id, action, content, reauthorize, operationId]) => [id, action, content, reauthorize, operationId]))
    .to.deep.equal([['a', 'steer', 'focus on dates', false, 'op-a']]);
  // The command settled, so the child is read again and its roster row follows.
  const reads = (path: string) => requests.filter((request) => request.method === 'GET' && request.path === path).length;
  await waitFor(() => reads('/web/api/answer/run-1/children') === 2 && reads('/web/api/answer/run-1/children/a') >= 2);
  const sent = requests.find((request) => request.method === 'POST')!;
  expect(sent.path).to.equal('/web/api/answer/run-1/children/a/control');
  expect(sent.body).to.deep.equal({action: 'steer', content: 'focus on dates', reauthorize_user_cancelled: false});
  expect(sent.headers.get('Idempotency-Key')).to.match(/^[0-9a-f-]{36}$/);
  expect(field.value, 'an accepted command empties its box').to.equal('');
  expect(document.activeElement, 'and leaves the reader where they were typing').to.equal(field);
});

it('continues a settled child, with a reauthorization the reader gives for work they cancelled', async () => {
  const stopped = row('a', 'cancelled', {cancellation_origin: 'user', started_at: ago(12), finished_at: ago(0)});
  const requests = serve({
    page: () => roster([stopped]),
    observe: () => observation(stopped),
    control: () => receipt('continue', 'accepted', {operation_id: 'op-a2'}),
  });
  const {source, controls} = sourceFor();
  const dock = await mount(source, 420);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');

  expect(composer(dock).placeholder).to.equal('Continue this child…');
  expect(shown(dock)).to.contain('Starts a new operation on this child.');
  const reauthorize = session(dock).querySelector<HTMLInputElement>('input[type="checkbox"]')!;
  expect(reauthorize.closest('label')!.textContent!.trim()).to.equal('Reauthorize this user-cancelled work');
  reauthorize.click();
  await type(dock, composer(dock), 'try again');
  session(dock).querySelector('form')!.requestSubmit();
  await waitFor(() => controls.length === 1);

  expect(controls[0]!.slice(0, 5)).to.deep.equal(['a', 'continue', 'try again', true, 'op-a']);
  const sent = requests.find((request) => request.method === 'POST')!;
  expect(sent.body).to.deep.equal({action: 'continue', content: 'try again', reauthorize_user_cancelled: true});
});

it('offers no reauthorization for a child the reader did not cancel', async () => {
  const failed = row('a', 'failed', {started_at: ago(12), finished_at: ago(0)});
  serve({page: () => roster([failed]), observe: () => observation(failed), control: () => receipt('continue', 'accepted')});
  const {source, controls} = sourceFor();
  const dock = await mount(source, 420);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');

  expect(session(dock).querySelector('input[type="checkbox"]')).to.equal(null);
  await type(dock, composer(dock), 'retry');
  session(dock).querySelector('form')!.requestSubmit();
  await waitFor(() => controls.length === 1);
  expect(controls[0]![3]).to.equal(false);
});

it('asks before it cancels a child, and keeps it running if the reader says so', async () => {
  const running = row('a', 'running', {started_at: ago(5)});
  const requests = serve({
    page: () => roster([running]),
    observe: () => observation(running),
    control: () => receipt('cancel', 'cancellation_requested'),
  });
  const {source, controls} = sourceFor();
  const dock = await mount(source, 420);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');

  button(session(dock), 'Cancel child')!.click();
  await waitFor(() => shown(dock).includes('Cancel this child? Its work so far is kept.'));
  const confirmation = session(dock).querySelector<HTMLElement>('[role="group"]')!;
  expect(confirmation.getAttribute('aria-labelledby')).to.equal(confirmation.querySelector('p')!.id);
  await waitFor(() => document.activeElement === button(confirmation, 'Keep running'));

  button(confirmation, 'Keep running')!.click();
  await waitFor(() => !shown(dock).includes('Cancel this child?'));
  expect(requests.filter((request) => request.method === 'POST')).to.have.length(0);
  await waitFor(() => document.activeElement === button(session(dock), 'Cancel child'));

  button(session(dock), 'Cancel child')!.click();
  await waitFor(() => session(dock).querySelector('[role="group"]') !== null);
  button(session(dock).querySelector<HTMLElement>('[role="group"]')!, 'Cancel child')!.click();
  await waitFor(() => shown(dock).includes('Cancellation requested.'));

  expect(controls.map((call) => call.slice(0, 5))).to.deep.equal([['a', 'cancel', '', false, 'op-a']]);
  expect(session(dock).querySelector('[role="group"]')).to.equal(null);
});

it('drops the question about cancelling a child that settles before the reader answers it', async () => {
  immediateFollowRefreshes();
  let status = 'running';
  const current = () => row('a', status, {started_at: ago(5), finished_at: status === 'running' ? null : ago(0)});
  serve({page: () => roster([current()]), observe: () => observation(current())});
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');
  button(session(dock), 'Cancel child')!.click();
  await waitFor(() => shown(dock).includes('Cancel this child?'));

  status = 'succeeded';
  dock.refreshIfFollowing('run-1');
  await waitFor(() => shown(dock).includes('Done'));

  expect(shown(dock)).to.not.contain('Cancel this child?');
  expect(button(session(dock), 'Cancel child')).to.equal(null);
});

const FINISHED_NOTE = 'This answer has finished, so its child agents can no longer be steered or continued.';

it('offers a box and a Cancel only while the Run can take what they send', async () => {
  const cases: [string | null, boolean][] = [
    [null, true], ['queued', true], ['running', true],
    ['succeeded', false], ['failed', false], ['cancelled', false],
    ['a-status-the-server-adds-later', true],
  ];
  for (const [runStatus, commandable] of cases) {
    serve({
      page: () => roster([
        row('a', 'running', {started_at: ago(5)}),
        row('b', 'succeeded', {started_at: ago(9), finished_at: ago(1), summary: 'A summary.'}),
      ], null, runStatus),
      observe: (id) => observation(id === 'a'
        ? row('a', 'running', {started_at: ago(5)})
        : row('b', 'succeeded', {started_at: ago(9), finished_at: ago(1), summary: 'A summary.'})),
    });
    const dock = await mount(sourceFor().source, 800);
    await waitFor(() => rows(dock).length === 2);
    for (const id of ['a', 'b']) {
      rowFor(dock, id).click();
      await waitFor(() => title(dock) === `objective ${id}` && shown(dock).includes('Activity'));

      const where = `a ${id === 'a' ? 'running' : 'settled'} child of a Run that is ${runStatus ?? 'not said to be anything'}`;
      expect(session(dock).querySelector('form') !== null, `${where}: its box`).to.equal(commandable);
      expect(button(session(dock), 'Cancel child') !== null, `${where}: its Cancel`).to.equal(commandable && id === 'a');
      expect(shown(dock).includes(FINISHED_NOTE), `${where}: the note`).to.equal(!commandable);
      // Opening on a Run that is over says nothing out loud: only its ending under the reader does.
      expect(session(dock).querySelector('[role="status"]')!.textContent!.trim()).to.equal('');
    }
    dock.remove();
  }
});

it('puts a note where the box was when the Run ends under a reader, and keeps what the child did', async () => {
  immediateFollowRefreshes();
  let runStatus = 'running';
  const running = row('a', 'running', {started_at: ago(5)});
  const requests = serve({
    page: () => roster([running], null, runStatus),
    observe: () => observation(running, {
      transcript: [{role: 'assistant', content: 'Checking the amendment.', tool_calls: []}],
    }),
  });
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');
  await waitFor(() => shown(dock).includes('Checking the amendment.'));
  await type(dock, composer(dock), 'one more thing');
  button(session(dock), 'Cancel child')!.click();
  await waitFor(() => shown(dock).includes('Cancel this child?'));

  runStatus = 'succeeded';
  dock.refreshIfFollowing('run-1');
  await waitFor(() => shown(dock).includes(FINISHED_NOTE));

  expect(session(dock).querySelector('form')).to.equal(null);
  expect(button(session(dock), 'Cancel child')).to.equal(null);
  expect(shown(dock), 'the question about cancelling lapses with the Run').to.not.contain('Cancel this child?');
  expect(shown(dock)).to.contain('Checking the amendment.');
  expect(session(dock).querySelector('[role="status"]')!.textContent!.trim()).to.equal(FINISHED_NOTE);
  expect(requests.filter((request) => request.method === 'POST'), 'nothing was sent').to.have.length(0);
});

it('answers a question for the parent through a box the reader opens', async () => {
  const running = row('a', 'running', {started_at: ago(5), pending_questions: 1});
  const requests = serve({
    page: () => roster([running]),
    observe: () => observation(running, {questions: [question('req-a')]}),
    reply: () => Response.json({
      run_id: 'run-1', request_id: 'req-a', action: 'reply', outcome: 'replied',
    }, {status: 202}),
  });
  const {source, replies} = sourceFor();
  Date.now = () => NOW;
  const dock = await mount(source, 420);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');
  await waitFor(() => button(session(dock), 'Answer instead') !== null);
  expect(session(dock).querySelector('[data-reply]')).to.equal(null);

  button(session(dock), 'Answer instead')!.click();
  await waitFor(() => session(dock).querySelector('[data-reply]') !== null);
  const reply = session(dock).querySelector<HTMLTextAreaElement>('[data-reply]')!;
  await waitFor(() => document.activeElement === reply);

  button(session(dock), 'Cancel answer')!.click();
  await waitFor(() => session(dock).querySelector('[data-reply]') === null);
  await waitFor(() => document.activeElement === button(session(dock), 'Answer instead'));

  button(session(dock), 'Answer instead')!.click();
  await waitFor(() => session(dock).querySelector('[data-reply]') !== null);
  await type(dock, session(dock).querySelector<HTMLTextAreaElement>('[data-reply]')!, 'Use the report');
  session(dock).querySelector<HTMLFormElement>('form[data-request]')!.requestSubmit();
  await waitFor(() => shown(dock).includes('Reply sent.'));

  expect(replies.map(([requestId, content]) => [requestId, content])).to.deep.equal([['req-a', 'Use the report']]);
  const sent = requests.find((request) => request.method === 'POST')!;
  expect(sent.path).to.equal('/web/api/answer/run-1/child-guidance/req-a/reply');
  expect(sent.body).to.deep.equal({content: 'Use the report'});
  expect(session(dock).querySelector('[data-reply]'), 'the box closes once the reply is accepted').to.equal(null);
});

it('sends on Enter, breaks the line on Shift+Enter, and never sends on the Enter of an IME composition', async () => {
  const running = row('a', 'running', {started_at: ago(5)});
  serve({page: () => roster([running]), observe: () => observation(running), control: () => receipt('steer', 'queued')});
  const {source, controls} = sourceFor();
  const dock = await mount(source, 420);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');
  const field = composer(dock);
  field.focus();

  await sendKeys({type: 'first'});
  await sendKeys({press: 'Shift+Enter'});
  await sendKeys({type: 'second'});
  expect(field.value).to.equal('first\nsecond');
  expect(controls).to.have.length(0);

  // The Enter that commits a composition is the composition's.
  const composing = new InputEvent('beforeinput', {
    inputType: 'insertLineBreak', isComposing: true, bubbles: true, cancelable: true,
  });
  field.dispatchEvent(composing);
  expect(composing.defaultPrevented).to.equal(false);
  expect(controls).to.have.length(0);

  await sendKeys({press: 'Enter'});
  await waitFor(() => controls.length === 1);
  expect(controls[0]![2]).to.equal('first\nsecond');
});

it('writes nothing to a box while the reader composes in it, whether the steer box or a reply box', async () => {
  Date.now = () => NOW;
  const running = row('a', 'running', {started_at: ago(5), pending_questions: 1});
  serve({page: () => roster([running]), observe: () => observation(running, {questions: [question('req-a')]})});
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');
  await waitFor(() => button(session(dock), 'Answer instead') !== null);
  button(session(dock), 'Answer instead')!.click();
  await waitFor(() => session(dock).querySelector('[data-reply]') !== null);
  const reply = session(dock).querySelector<HTMLTextAreaElement>('[data-reply]')!;

  for (const field of [composer(dock), reply]) {
    await type(dock, field, 'ab');
    const {writes, enter} = watchWrites(field);
    field.focus();
    field.setSelectionRange(1, 1);

    // Composing between the two letters: the browser keeps the text and the caret it is composing at.
    field.dispatchEvent(new CompositionEvent('compositionstart', {bubbles: true}));
    enter('a\u65e5b', 2, true);
    await session(dock).updateComplete;
    expect([field.value, field.selectionStart, field.selectionEnd]).to.deep.equal(['a\u65e5b', 2, 2]);
    enter('a\u65e5\u672cb', 3, true);
    await session(dock).updateComplete;
    expect([field.value, field.selectionStart, field.selectionEnd]).to.deep.equal(['a\u65e5\u672cb', 3, 3]);

    enter('a\u65e5\u672c\u8a9eb', 4, false);
    field.dispatchEvent(new CompositionEvent('compositionend', {bubbles: true}));
    await session(dock).updateComplete;
    expect([field.value, field.selectionStart, field.selectionEnd]).to.deep.equal(['a\u65e5\u672c\u8a9eb', 4, 4]);
    expect(writes, 'the page never wrote the box while it was composed in').to.deep.equal([]);
    expect(document.activeElement).to.equal(field);
    // The page has the text all the same: the Send button took it.
    const send = field.closest('form')!.querySelector('button[type="submit"]')!;
    expect(send.hasAttribute('aria-disabled')).to.equal(false);
  }
});

it('names each refusal, and says when the child is gone', async () => {
  const running = row('a', 'running', {started_at: ago(5)});
  let answer: () => Response = () => refusal('terminal_child');
  serve({page: () => roster([running]), observe: () => observation(running), control: () => answer()});
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');

  const send = async (text: string, expected: string) => {
    await type(dock, composer(dock), text);
    session(dock).querySelector('form')!.requestSubmit();
    await waitFor(() => shown(dock).includes(expected));
  };
  await send('one', 'This child is already terminal and was not revived.');
  expect(composer(dock).value, 'a refused command keeps what was typed').to.equal('one');
  answer = () => refusal('queue_full');
  await send('two', 'The pending control queue is full.');
  answer = () => new Response('unavailable', {status: 503});
  await send('three', 'The child intervention could not be sent.');
  expect(session(dock).querySelector('[role="status"]')!.textContent!.trim()).to.equal('The child intervention could not be sent.');

  answer = () => Response.json({detail: 'Answer child not found'}, {status: 404});
  await send('four', 'That child is no longer available.');
  expect(session(dock).querySelector('[role="status"]')!.textContent!.trim()).to.equal('That child is no longer available.');
  expect(session(dock).querySelector('form')).to.equal(null);
});

it('words every outcome a command can meet', async () => {
  const running = row('a', 'running', {started_at: ago(5)});
  let answer: () => Response = () => receipt('steer', 'queued');
  serve({
    page: () => roster([running]),
    observe: () => observation(running),
    control: () => answer(),
    reply: () => answer(),
  });
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');

  const outcomes: [Response, string][] = [
    [receipt('steer', 'queued'), 'Queued. The child has not necessarily followed it yet.'],
    [receipt('steer', 'consumed'), 'Consumed at a safe checkpoint. This does not prove the model complied.'],
    [receipt('continue', 'accepted'), 'Continuation accepted as a new operation.'],
    [receipt('cancel', 'cancellation_requested'), 'Cancellation requested.'],
    [refusal('terminal_child'), 'This child is already terminal and was not revived.'],
    [refusal('run_terminal'), 'The parent run is terminal, so this child cannot continue.'],
    [refusal('child_running'), 'This child is still running.'],
    [refusal('reauthorization_required'), 'User-cancelled work needs explicit reauthorization.'],
    [refusal('queue_full'), 'The pending control queue is full.'],
    [refusal('idempotency_conflict'), 'This submission id was already used for a different request.'],
    [refusal('unknown_outcome'), 'The child outcome is unknown.'],
  ];
  for (const [response, words] of outcomes) {
    answer = () => response.clone();
    await type(dock, composer(dock), 'go');
    session(dock).querySelector('form')!.requestSubmit();
    await waitFor(() => session(dock).querySelector('[role="status"]')!.textContent!.trim() === words);
    await waitFor(() => !composer(dock).readOnly);
  }
});

it('keeps child A\'s late receipt, busy state and answer off child B\'s page', async () => {
  const late = deferred<Response>();
  serve({
    page: () => roster([row('a', 'running', {started_at: ago(5)}), row('b', 'running', {started_at: ago(5)})]),
    observe: (id) => observation(row(id, 'running', {started_at: ago(5)})),
    control: () => late.promise,
  });
  const {source, controls} = sourceFor();
  const dock = await mount(source, 420);
  await waitFor(() => rows(dock).length === 2);
  await openChild(dock, 'a');
  await type(dock, composer(dock), 'only child A');
  session(dock).querySelector('form')!.requestSubmit();
  await waitFor(() => controls.length === 1);
  expect(composer(dock).readOnly, 'A\'s own box waits for its command').to.equal(true);

  button(dock, 'All child agents')!.click();
  await dock.updateComplete;
  await openChild(dock, 'b');
  await type(dock, composer(dock), 'child B draft');
  expect(composer(dock).readOnly).to.equal(false);

  late.resolve(receipt('steer', 'queued', {child_session_id: 'a'}));
  await settle();
  await session(dock).updateComplete;

  expect(shown(dock)).to.not.contain('Queued.');
  expect(composer(dock).value).to.equal('child B draft');
  expect(composer(dock).readOnly).to.equal(false);
  // Back on A, its receipt has landed: the box is free and empty again.
  button(dock, 'All child agents')!.click();
  await dock.updateComplete;
  await openChild(dock, 'a');
  expect(composer(dock).value).to.equal('');
  expect(composer(dock).readOnly).to.equal(false);
});

it('does not attach a late receipt to the same child of a Run opened afterwards', async () => {
  const late = deferred<Response>();
  serve({
    page: () => roster([row('a', 'running', {started_at: ago(5)})]),
    observe: () => observation(row('a', 'running', {started_at: ago(5)})),
    control: () => late.promise,
  });
  const first = sourceFor();
  const dock = await mount(first.source, 420);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');
  await type(dock, composer(dock), 'first Run');
  session(dock).querySelector('form')!.requestSubmit();
  await waitFor(() => first.controls.length === 1);

  dock.source = sourceFor().source;
  await waitFor(() => rows(dock).length === 1 && session(dock).hidden === true);
  await openChild(dock, 'a');
  expect(composer(dock).value, 'what was typed for the earlier Run is gone').to.equal('');
  late.resolve(receipt('steer', 'queued'));
  await settle();

  expect(shown(dock)).to.not.contain('Queued.');
  expect(composer(dock).readOnly).to.equal(false);
});

it('keeps what the reader typed across refreshes and across children, for each Operation apart', async () => {
  immediateFollowRefreshes();
  let operation = 'op-a';
  serve({
    page: () => roster([
      row('a', 'running', {operation_id: operation, started_at: ago(5)}),
      row('b', 'running', {started_at: ago(5)}),
    ]),
    observe: (id) => observation(id === 'a'
      ? row('a', 'running', {operation_id: operation, started_at: ago(5)})
      : row('b', 'running', {started_at: ago(5)})),
  });
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 2);
  await openChild(dock, 'a');
  const aBox = composer(dock);
  await type(dock, aBox, 'for A, first Operation');
  aBox.focus();
  aBox.setSelectionRange(4, 9);

  // A refresh redraws the page around the box without taking the text, the focus, or the selection.
  dock.refreshIfFollowing('run-1');
  await settle(60);
  expect(composer(dock)).to.equal(aBox);
  expect(aBox.value).to.equal('for A, first Operation');
  expect(document.activeElement).to.equal(aBox);
  expect([aBox.selectionStart, aBox.selectionEnd]).to.deep.equal([4, 9]);

  // Another child has a box of its own, and the first one is waiting when the reader returns.
  button(dock, 'All child agents')!.click();
  await dock.updateComplete;
  await openChild(dock, 'b');
  expect(composer(dock).value).to.equal('');
  await type(dock, composer(dock), 'for B');
  button(dock, 'All child agents')!.click();
  await dock.updateComplete;
  await openChild(dock, 'a');
  expect(composer(dock).value).to.equal('for A, first Operation');

  // A new Operation of the same child starts with an empty box, and the old one comes back with it.
  operation = 'op-a2';
  dock.refreshIfFollowing('run-1');
  await waitFor(() => composer(dock).value === '');
  await type(dock, composer(dock), 'for A, second Operation');
  operation = 'op-a';
  dock.refreshIfFollowing('run-1');
  await waitFor(() => composer(dock).value === 'for A, first Operation');
});

it('reads a running child again after each roster refresh, and a settled one only when its row changed', async () => {
  immediateFollowRefreshes();
  const stateOf = {a: row('a', 'running', {started_at: ago(5)}), b: row('b', 'succeeded', {started_at: ago(9), finished_at: ago(1)})};
  const requests = serve({
    page: () => roster([stateOf.a, stateOf.b]),
    observe: (id) => observation(id === 'a' ? stateOf.a : stateOf.b),
  });
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 2);
  const observed = (id: string) => requests.filter((request) => request.path.endsWith(`/children/${id}`)).length;
  const refreshed = async () => {
    const before = requests.filter((request) => request.path.endsWith('/children')).length;
    dock.refreshIfFollowing('run-1');
    await waitFor(() => requests.filter((request) => request.path.endsWith('/children')).length > before);
    await settle(40);
  };

  await openChild(dock, 'a');
  await waitFor(() => observed('a') === 1);
  await refreshed();
  expect(observed('a'), 'a running child\'s transcript grows without its row changing').to.equal(2);

  button(dock, 'All child agents')!.click();
  await dock.updateComplete;
  await openChild(dock, 'b');
  await waitFor(() => observed('b') === 1);
  await refreshed();
  expect(observed('b'), 'a settled child that has not changed is not read again').to.equal(1);

  stateOf.b = row('b', 'succeeded', {started_at: ago(9), finished_at: ago(1), summary: 'A late summary.'});
  await refreshed();
  expect(observed('b'), 'a settled child whose row changed is').to.equal(2);
});

it('says so when the child on show is no longer on the roster', async () => {
  immediateFollowRefreshes();
  let children = [row('a', 'running'), row('b', 'running')];
  serve({page: () => roster(children), observe: (id) => observation(row(id, 'running'))});
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 2);
  await openChild(dock, 'a');

  children = [row('b', 'running')];
  dock.refreshIfFollowing('run-1');
  await waitFor(() => shown(dock).includes('That child is no longer available.'));

  expect(shown(dock)).to.contain('Pick another from the list.');
  expect(session(dock).querySelector('[role="status"]')!.textContent!.trim()).to.equal('That child is no longer available.');
  expect(session(dock).querySelector('form')).to.equal(null);
  button(dock, 'All child agents')!.click();
  await dock.updateComplete;
  expect(rows(dock).map((child) => child.id)).to.deep.equal(['b']);
});

it('says so when the server no longer knows the child it was asked about', async () => {
  serve({
    page: () => roster([row('a', 'running')]),
    observe: () => Response.json({detail: 'Answer child not found'}, {status: 404}),
  });
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 1);
  rowFor(dock, 'a').click();
  await waitFor(() => shown(dock).includes('That child is no longer available.'));
});

it('keeps what is on screen when a later read of the child fails, and offers a retry when the first does', async () => {
  immediateFollowRefreshes();
  let fail = true;
  let reads = 0;
  serve({
    page: () => roster([row('a', 'running', {started_at: ago(5)})]),
    observe: () => {
      reads += 1;
      return fail ? new Response('unavailable', {status: 503}) : observation(row('a', 'running', {started_at: ago(5)}), {
        transcript: [{role: 'assistant', content: 'Still here.', tool_calls: []}],
      });
    },
  });
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 1);
  rowFor(dock, 'a').click();
  await waitFor(() => session(dock).querySelector('[role="alert"]') !== null);
  expect(session(dock).querySelector('[role="alert"]')!.textContent!.trim()).to.equal('Child details could not be loaded.');

  fail = false;
  button(session(dock), 'Retry')!.click();
  await waitFor(() => shown(dock).includes('Still here.'));

  fail = true;
  const before = reads;
  dock.refreshIfFollowing('run-1');
  await waitFor(() => reads > before);
  await settle(40);
  expect(shown(dock)).to.contain('Still here.');
  expect(session(dock).querySelector('[role="alert"]')).to.equal(null);
});

it('names a child by its id when it has no objective', async () => {
  serve({page: () => roster([row('a', 'running', {objective: null})])});
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 1);

  expect(rows(dock)[0]!.text).to.equal('a Running');
});

// ── The live edge of a child's activity ──

describe('the page of a running child', () => {
  let unlink: () => void;
  before(async () => {
    // Scrolling needs the page laid out as the product lays it out.
    unlink = await linkStyles([
      '../design-system/index.css',
      '../styles/child-session.module.css',
      '../styles/inspector-children.module.css',
    ].map((href) => new URL(href, import.meta.url).href));
  });
  after(() => { unlink(); });

  const says = (count: number) => Array.from({length: count}, (_, index) => (
    {role: 'assistant', content: `Step ${index}`, tool_calls: []}
  ));
  const page = (dock: DlInspectorChildren) => session(dock).querySelector<HTMLElement>('[data-page]')!;
  /** How far the page lies below what is on screen, in pixels. */
  const below = (box: HTMLElement) => box.scrollHeight - box.clientHeight - box.scrollTop;
  const frames = () => new Promise<void>((resolve) => {
    requestAnimationFrame(() => { requestAnimationFrame(() => { resolve(); }); });
  });
  /** The reader scrolls: the page is moved, and given the frames it takes to see it. */
  const scrollTo = async (box: HTMLElement, top: number) => {
    box.scrollTop = top;
    await frames();
  };

  it('opens on its title, keeps the bottom in view for a reader who scrolled there, and lets go when they scroll up', async () => {
    immediateFollowRefreshes();
    let steps = 40;
    const running = row('a', 'running', {started_at: ago(5)});
    serve({page: () => roster([running]), observe: () => observation(running, {transcript: says(steps)})});
    const dock = await mount(sourceFor().source, 420);
    await waitFor(() => rows(dock).length === 1);
    await openChild(dock, 'a');
    await waitFor(() => shown(dock).includes('Step 39'));
    const box = page(dock);
    expect(box.scrollHeight, 'the page is taller than the dock').to.be.greaterThan(box.clientHeight + 100);
    const arrive = async (total: number) => {
      steps = total;
      dock.refreshIfFollowing('run-1');
      await waitFor(() => shown(dock).includes(`Step ${total - 1}`));
    };

    // A reader who has not scrolled stays on the title as steps arrive.
    expect(box.scrollTop).to.equal(0);
    await arrive(44);
    expect(box.scrollTop).to.equal(0);

    // One who has scrolled to the bottom is kept there.
    await scrollTo(box, box.scrollHeight);
    await arrive(48);
    expect(below(box)).to.be.lessThan(1);
    await arrive(52);
    expect(below(box)).to.be.lessThan(1);

    // And once they scroll up the page stays where they are.
    await scrollTo(box, 100);
    await arrive(56);
    expect(box.scrollTop).to.equal(100);
    expect(below(box)).to.be.greaterThan(100);
  });

  it('opens each child on its title, even from the bottom of another', async () => {
    immediateFollowRefreshes();
    const loadingB = deferred<Response>();
    const running = (id: string) => row(id, 'running', {started_at: ago(5)});
    serve({
      page: () => roster([running('a'), running('b')]),
      observe: (id) => (id === 'b' ? loadingB.promise : observation(running(id), {transcript: says(40)})),
    });
    const dock = await mount(sourceFor().source, 800);
    await waitFor(() => rows(dock).length === 2);
    await waitFor(() => title(dock) === 'objective a' && shown(dock).includes('Step 39'));
    const box = page(dock);
    await scrollTo(box, box.scrollHeight);
    expect(below(box)).to.be.lessThan(1);

    // The next child is still loading, so its page is short, when the scroll back to the top is seen.
    rowFor(dock, 'b').click();
    await waitFor(() => title(dock) === 'objective b');
    await scrollTo(box, 0);
    loadingB.resolve(observation(running('b'), {transcript: says(40)}));
    await waitFor(() => shown(dock).includes('Step 39'));
    expect(page(dock) === box, 'the same page is on show').to.equal(true);
    expect(box.scrollTop, 'the next child opens on its title').to.equal(0);
    dock.refreshIfFollowing('run-1');
    await settle(60);
    expect(box.scrollTop, 'and nothing carries the earlier child\'s place over').to.equal(0);
  });

  it('lets go of a reader who opens a step at the bottom, which grows the page without scrolling it', async () => {
    immediateFollowRefreshes();
    let steps = 30;
    const running = row('a', 'running', {started_at: ago(5)});
    const transcript = () => [
      ...says(steps),
      {role: 'assistant', content: '', tool_calls: [{id: 't1', name: 'read'}]},
      {role: 'tool', content: Array.from({length: 40}, (_, line) => `line ${line}`).join('\n'), tool_call_id: 't1', name: 'read', is_error: false},
    ];
    serve({page: () => roster([running]), observe: () => observation(running, {transcript: transcript()})});
    const dock = await mount(sourceFor().source, 420);
    await waitFor(() => rows(dock).length === 1);
    await openChild(dock, 'a');
    await waitFor(() => shown(dock).includes('Reading a document'));
    const box = page(dock);
    await scrollTo(box, box.scrollHeight);

    // The reader opens the last step: its whole result lengthens the page below where they stand.
    session(dock).querySelector<HTMLElement>('summary:has(+ pre)')!.click();
    await waitFor(() => below(box) > 100);
    const stood = box.scrollTop;
    steps = 34;
    dock.refreshIfFollowing('run-1');
    await waitFor(() => shown(dock).includes('Step 33'));

    expect(box.scrollTop, 'a reader who is no longer at the bottom is not pulled back to it').to.equal(stood);
  });

  it('does not follow a child that has settled', async () => {
    immediateFollowRefreshes();
    let steps = 40;
    let summary = 'First summary.';
    const settled = () => row('a', 'succeeded', {started_at: ago(9), finished_at: ago(1), summary});
    serve({page: () => roster([settled()]), observe: () => observation(settled(), {transcript: says(steps)})});
    const dock = await mount(sourceFor().source, 420);
    await waitFor(() => rows(dock).length === 1);
    await openChild(dock, 'a');
    await waitFor(() => shown(dock).includes('Activity · 40 steps'));
    // Its activity is folded; the reader opens it and goes to the bottom.
    session(dock).querySelector<HTMLDetailsElement>('details:has(ol)')!.open = true;
    const box = page(dock);
    await waitFor(() => box.scrollHeight > box.clientHeight + 100);
    // WebKit lays an opened fold out a frame later, and takes back a position set before it has.
    await frames();
    await scrollTo(box, box.scrollHeight);
    const stood = box.scrollTop;
    expect(stood).to.be.greaterThan(100);

    steps = 46;
    summary = 'A later summary.';
    dock.refreshIfFollowing('run-1');
    await waitFor(() => shown(dock).includes('Step 45'));
    expect(box.scrollTop).to.equal(stood);
    expect(below(box)).to.be.greaterThan(50);
  });
});

// ── On a phone ──

describe('on a phone', () => {
  const originalViewport = {width: window.innerWidth, height: window.innerHeight};
  let unlink: () => void;
  before(async () => {
    await setViewport({width: 390, height: 844});
    unlink = await linkStyles([
      '../design-system/index.css',
      '../styles/global.css',
      '../styles/layout.css',
      '../styles/child-session.module.css',
      '../styles/inspector-children.module.css',
    ].map((href) => new URL(href, import.meta.url).href));
  });
  after(async () => {
    unlink();
    await setViewport(originalViewport);
  });

  it('gives everything the reader can press the product\'s hit target, in the list and on a child\'s page', async () => {
    const running = row('a', 'running', {started_at: ago(5), pending_questions: 1});
    serve({
      page: () => roster([running, row('b', 'succeeded', {summary: 'Done.', result_handles: ['ev-1']})]),
      observe: () => observation(running, {
        transcript: [
          {role: 'assistant', content: 'Looking.', tool_calls: [{id: 't1', name: 'read'}]},
          {role: 'tool', content: 'first line\nsecond line', tool_call_id: 't1', name: 'read', is_error: false},
        ],
        controls: [{
          control_sequence: 1, kind: 'steer', content: 'focus', origin: 'user', consumed: true,
          consumed_at: ago(3), created_at: ago(4), operation_id: 'op-a',
        }],
        questions: [question('q1')],
      }),
    });
    Date.now = () => NOW;
    const probe = document.createElement('div');
    probe.style.height = 'var(--control-hit-target)';
    document.body.append(probe);
    const target = probe.getBoundingClientRect().height;
    probe.remove();
    const dock = await mount(sourceFor().source, 390);
    await waitFor(() => rows(dock).length === 2);
    const pressable = () => [...dock.querySelectorAll<HTMLElement>('button, summary')]
      .filter((element) => element.getClientRects().length > 0)
      .map((element) => ({
        name: (element.getAttribute('aria-label') ?? element.textContent!).replace(/\s+/g, ' ').trim(),
        height: element.getBoundingClientRect().height,
      }));

    expect(target, 'the product\'s hit target is known').to.be.greaterThan(40);
    const list = pressable();
    expect(list).to.have.length(2);
    await openChild(dock, 'a');
    await waitFor(() => shown(dock).includes('Looking.') && shown(dock).includes('Control history'));
    const page = pressable();
    // The page offers Back, Cancel, Answer instead, Send, the Activity and Control history folds, and a step.
    expect(page.length).to.be.greaterThan(6);
    const short = [...list, ...page].filter((item) => item.height < target - 0.5);
    expect(short, 'what is pressable and shorter than the hit target').to.deep.equal([]);
  });
});
