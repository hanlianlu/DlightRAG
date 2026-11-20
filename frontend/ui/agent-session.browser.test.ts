// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** One agent's page on its own: what it shows before the roster arrives, what it reads again when the
 *  dock says the Run moved, and the clock it keeps for itself. */

import {expect} from '@esm-bundle/chai';
import type {AgentChildStatus} from '../api/conversations.ts';
import {defineDesignSystemElements} from '../design-system/index.ts';
import {activityPage, ago, answeredTurn, NOW, observation, row, serve, sourceFor} from '../testing/traces.ts';
import {waitFor} from '../testing/dom.ts';
import type {TracesSource} from '../stores/traces-source.ts';
import type {DlAgentSession} from './agent-session.ts';
import './agent-session.ts';

defineDesignSystemElements();

/** One agent as the roster states it, with only the fields a test reads set. */
function listed(id: string, extra: Partial<AgentChildStatus> = {}): AgentChildStatus & {childSessionId: string} {
  return {
    childSessionId: id,
    status: 'running',
    objective: `objective ${id}`,
    modelRole: 'query',
    usage: null,
    operationId: `op-${id}`,
    operationSequence: 1,
    operationStatus: null,
    cancellationOrigin: null,
    summary: null,
    resultHandles: [],
    startedAt: ago(134),
    finishedAt: null,
    pendingQuestions: 0,
    resultSequence: null,
    feedback: null,
    ...extra,
  };
}

function mountSession(
  source: TracesSource | null,
  childSessionId: string,
  entry: AgentChildStatus | null,
): DlAgentSession {
  const session = document.createElement('dl-agent-session') as DlAgentSession;
  session.source = source;
  session.childSessionId = childSessionId;
  session.entry = entry;
  document.body.append(session);
  return session;
}

/** The page's words as a reader takes them in, without the template's layout whitespace. */
const text = (session: DlAgentSession): string => (session.textContent ?? '').replace(/\s+/g, ' ').trim();

const transcriptReads = (requests: {path: string; search: string}[]) =>
  requests.filter((request) => request.path.endsWith('/transcript'));

const childReads = (requests: {path: string; search: string}[], id: string) =>
  requests.filter((request) => request.path.endsWith(`/children/${id}`));

const realNow = Date.now;
const realSetInterval = window.setInterval;
const realClearInterval = window.clearInterval;
const realSetTimeout = window.setTimeout;

const settle = (ms = 30) => new Promise<void>((resolve) => { realSetTimeout(resolve, ms); });

/** Capture the page's own clock, so a test decides when it ticks. */
function captureTicks(): {ticks: Map<number, () => void>; stopped: number[]} {
  const ticks = new Map<number, () => void>();
  const stopped: number[] = [];
  window.setInterval = ((handler: TimerHandler) => {
    const id = ticks.size + 1;
    ticks.set(id, handler as () => void);
    return id;
  }) as typeof window.setInterval;
  window.clearInterval = ((id?: number) => { if (id !== undefined) stopped.push(id); }) as typeof window.clearInterval;
  return {ticks, stopped};
}

afterEach(() => {
  document.body.replaceChildren();
  Date.now = realNow;
  window.setInterval = realSetInterval;
  window.clearInterval = realClearInterval;
});

it('draws the page before the roster arrives, and reads its transcript in parallel', async () => {
  const requests = serve({
    // The child's own row is still on its way, as when the dock mounts the page and the roster together.
    observe: () => new Promise<Response>(() => {}),
    transcript: () => activityPage([{role: 'assistant', content: 'Reading in the dark.', tool_calls: []}], {running: true}),
  });
  const session = mountSession(sourceFor().source, 'a', null);
  await session.updateComplete;

  expect(text(session), 'the page waits rather than saying the child is gone').to.contain('Loading child details…');
  expect(text(session)).to.not.contain('That child is no longer available.');
  expect(session.querySelector('h3')?.textContent, 'the heading names the agent it waits for by its role, never its id')
    .to.equal('Child agent');
  expect(session.querySelector<HTMLElement>('p[role="status"]')!.classList.contains('dl-note'),
    'the quiet line is the design system\'s').to.equal(true);
  expect(session.querySelector('dl-activity'), 'the transcript reads while the roster is still loading').to.not.equal(null);
  await waitFor(() => transcriptReads(requests).length === 1);
  expect(transcriptReads(requests)[0]!.search).to.equal('?child=a');
});

it('never names a child by its Session id, even before the roster arrives', async () => {
  const id = '3f2a7c1e-2b44-4e87-9f0d-8c6c1f5b9a10';
  serve({
    observe: () => new Promise<Response>(() => {}),
    transcript: () => activityPage([], {running: true}),
  });
  const session = mountSession(sourceFor().source, id, null);
  await session.updateComplete;
  expect(session.querySelector('h3')?.textContent).to.equal('Child agent');
  expect(text(session)).to.not.contain('3f2a7c1e');
});

it('reads the main agent\'s transcript while the roster is still loading', async () => {
  const requests = serve({
    transcript: () => activityPage([{role: 'assistant', content: 'Hello.', tool_calls: []}], {running: true}),
  });
  const session = mountSession(sourceFor('run-1', answeredTurn({state: 'streaming'})).source, 'main', null);
  await session.updateComplete;

  expect(text(session)).to.contain('Main agent');
  expect(text(session)).to.contain('Loading child details…');
  expect(session.querySelector('dl-activity')).to.not.equal(null);
  await waitFor(() => transcriptReads(requests).length === 1);
  expect(transcriptReads(requests)[0]!.search, 'the main agent\'s page reads without a child name').to.equal('');
});

it('fills the page in when the roster row arrives behind it', async () => {
  serve({observe: () => new Promise<Response>(() => {})});
  const session = mountSession(sourceFor().source, 'a', null);
  await session.updateComplete;
  expect(text(session)).to.contain('Loading child details…');

  session.entry = listed('a');
  await waitFor(() => text(session).includes('Running'));
  expect(session.querySelector('h3')?.textContent).to.contain('objective a');
  expect(session.querySelector('dl-activity'), 'the transcript pane it mounted early stays').to.not.equal(null);
});

it('reads its child again when the dock refreshes the page, and its activity with it', async () => {
  let observations = 0;
  const running = row('a', 'running', {started_at: ago(5)});
  const requests = serve({
    observe: () => {
      observations += 1;
      return observation(running, {transcript: [{role: 'assistant', content: 'First.', tool_calls: []}]});
    },
  });
  const session = mountSession(sourceFor().source, 'a', listed('a', {startedAt: ago(5)}));
  await waitFor(() => transcriptReads(requests).length === 1 && observations === 1);
  await settle(20);

  session.refresh();
  await waitFor(() => requests.some((request) => (
    request.path.endsWith('/transcript') && request.search.includes('since=')
  )));
  expect(childReads(requests, 'a').length, 'the dock\'s events say the Run moved, so the row is read again')
    .to.be.greaterThan(1);
  expect(text(session)).to.contain('First.');
});

it('reads the main agent\'s transcript again when the dock refreshes the page', async () => {
  const requests = serve({
    transcript: () => activityPage([{role: 'assistant', content: 'The answer.', tool_calls: []}], {running: true}),
  });
  const session = mountSession(
    sourceFor('run-1', answeredTurn({state: 'streaming'})).source,
    'main',
    listed('main', {startedAt: ago(5)}),
  );
  await waitFor(() => transcriptReads(requests).length === 1);
  await settle(20);

  session.refresh();
  await waitFor(() => requests.some((request) => (
    request.path.endsWith('/transcript') && request.search.includes('since=')
  )));
  expect(text(session)).to.contain('The answer.');
});

it('reads nothing when the dock hands it a new row object that says the same thing', async () => {
  Date.now = () => NOW;
  const running = row('a', 'running', {started_at: ago(5)});
  const requests = serve({
    observe: () => observation(running, {transcript: [{role: 'assistant', content: 'Steps.', tool_calls: []}]}),
  });
  const entry = listed('a', {startedAt: ago(5)});
  const session = mountSession(sourceFor().source, 'a', entry);
  await waitFor(() => transcriptReads(requests).length === 1 && childReads(requests, 'a').length === 1);
  await settle(20);

  // A dock that re-renders hands a new object with the same values: that is not news about the child.
  session.entry = {...entry};
  await session.updateComplete;
  await settle(20);
  expect(transcriptReads(requests), 'a redraw alone asks for no transcript').to.have.length(1);
  expect(childReads(requests, 'a'), 'nor for the child\'s details').to.have.length(1);

  session.entry = {...entry, summary: 'Said something new.'};
  await waitFor(() => childReads(requests, 'a').length === 2);
});

it('says so when the roster drops the child it was showing, and when the server no longer knows it', async () => {
  serve({observe: () => observation(row('a', 'running'))});
  const listedSession = mountSession(sourceFor().source, 'a', listed('a'));
  await waitFor(() => listedSession.querySelector('h3') !== null);
  expect(listedSession.querySelector('dl-activity')).to.not.equal(null);

  listedSession.entry = null;
  await waitFor(() => text(listedSession).includes('That child is no longer available.'));
  expect(text(listedSession)).to.contain('Pick another from the list.');
  expect(listedSession.querySelector('dl-activity'), 'a gone child has no page to read').to.equal(null);
  listedSession.remove();

  serve({observe: () => Response.json({detail: 'Answer child not found'}, {status: 404})});
  const unknown = mountSession(sourceFor().source, 'b', listed('b'));
  await waitFor(() => text(unknown).includes('That child is no longer available.'));
});

it('keeps the transcript when the child\'s details fail, and offers a retry', async () => {
  serve({
    observe: () => new Response('unavailable', {status: 503}),
    transcript: () => activityPage([{role: 'assistant', content: 'Still here.', tool_calls: []}], {running: true}),
  });
  const session = mountSession(sourceFor().source, 'a', listed('a'));

  await waitFor(() => session.querySelector('[role="alert"]') !== null);
  expect(session.querySelector('[role="alert"]')!.textContent!.trim()).to.equal('Child details could not be loaded.');
  expect(text(session)).to.contain('Still here.');
  expect(session.querySelector('dl-activity')).to.not.equal(null);
});

it('keeps its own clock: a running child\'s time moves, and the clock leaves with the page', async () => {
  let now = NOW;
  Date.now = () => now;
  const {ticks, stopped} = captureTicks();
  const requests = serve({observe: () => observation(row('a', 'running', {started_at: ago(134)}))});
  const session = mountSession(sourceFor().source, 'a', listed('a'));
  await session.updateComplete;
  await waitFor(() => ticks.size === 1);
  expect(text(session)).to.contain('Running 2m 14s');
  const reads = transcriptReads(requests).length;

  now += 5000;
  [...ticks.values()][0]!();
  await session.updateComplete;
  expect(text(session)).to.contain('Running 2m 19s');
  expect(transcriptReads(requests).length, 'a tick asks the server for nothing').to.equal(reads);

  session.remove();
  expect(stopped, 'the clock goes when the page does').to.deep.equal([1]);
});

it('keeps its own clock for the main agent too', async () => {
  let now = NOW;
  Date.now = () => now;
  const {ticks} = captureTicks();
  serve({});
  const session = mountSession(
    sourceFor('run-1', answeredTurn({state: 'streaming'})).source,
    'main',
    listed('main', {startedAt: ago(134)}),
  );
  await session.updateComplete;
  await waitFor(() => ticks.size === 1);
  expect(text(session)).to.contain('Running 2m 14s');

  now += 5000;
  [...ticks.values()][0]!();
  await session.updateComplete;
  expect(text(session)).to.contain('Running 2m 19s');
});

it('keeps no clock once the child on show has settled', async () => {
  const {ticks} = captureTicks();
  serve({observe: () => observation(row('a', 'succeeded', {started_at: ago(68), finished_at: ago(0)}))});
  const session = mountSession(
    sourceFor().source,
    'a',
    listed('a', {status: 'succeeded', startedAt: ago(68), finishedAt: ago(0)}),
  );
  await session.updateComplete;
  await settle(20);

  expect(ticks.size).to.equal(0);
});
