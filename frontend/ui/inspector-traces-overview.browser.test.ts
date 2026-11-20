// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The Run overview in the dock: the 44px band's one summary line, the dossier under it, the
 *  finding a reader picks opening its step in its agent's page, and the states a Run can be in
 *  before it has one. The bare component's own surface is covered beside this file; these tests are
 *  the dock's, over the product's own routes. */

import {expect} from '@esm-bundle/chai';
import {answeredTurn, activityPage, NOW, overview, roster, row, serve, sourceFor} from '../testing/traces.ts';
import {mount, restoreDockFixtures, rows, session, settle, shown, stated} from '../testing/traces-dock.ts';
import {waitFor} from '../testing/dom.ts';
import './inspector-traces.ts';
import type {DlInspectorTraces} from './inspector-traces.ts';

restoreDockFixtures();

/** One Run's whole-Run facts: a main agent and one child, a failed call on the main agent's first
 *  message, and a delivery check — enough for every dossier section to have something to say. */
function runOverview(): Record<string, unknown> {
  return overview({
    totals: {
      steps: 6,
      kinds: {thinking: 0, tool: 3, say: 1, answer: 1, instruction: 0, delivery: 1},
      decisions: 2,
      marks: {failed: 1, degraded: 0, repeated: 1, redirect: 0},
      tool_time_ms: 2400,
    },
    agents: [
      {
        agent: 'main', label: 'main', order: 0, steps: 3, tool_calls: 1,
        marks: {failed: 1, degraded: 0, repeated: 1, redirect: 0},
        tool_time_ms: 1200, first_at: '2026-10-05T12:00:00Z', last_at: '2026-10-05T12:00:30Z',
      },
      {
        agent: 'a', label: 'objective a', order: 1, steps: 3, tool_calls: 2,
        marks: {failed: 0, degraded: 0, repeated: 0, redirect: 0},
        tool_time_ms: 1200, first_at: '2026-10-05T12:00:10Z', last_at: '2026-10-05T12:00:50Z',
      },
    ],
    findings: [
      {
        agent: 'main', sequence: 1, kind: 'tool', at: '2026-10-05T12:00:20Z', name: 'read',
        subject: 'northwind-msa.pdf', outcome: 'failed', marks: ['failed'],
        excerpt: 'The file could not be read.',
      },
      {
        agent: 'a', sequence: 4, kind: 'delivery', at: '2026-10-05T12:00:45Z', name: null,
        subject: null, outcome: null, marks: [], excerpt: 'The record names no verifier.',
      },
    ],
    findings_total: 2,
  });
}

/** The main agent's transcript: one failed read, on its first message — the step the finding names. */
function mainTranscript(): Response {
  return activityPage([
    {role: 'assistant', content: '', tool_calls: [{id: 't1', name: 'read'}]},
    {
      role: 'tool', content: 'text layer missing', tool_call_id: 't1', name: 'read', is_error: true,
    },
  ], {running: false});
}

/** The Run's roster: its own agent and one child, settled. */
const servingRun = () => ({
  page: () => roster([row('a', 'succeeded')], null, 'succeeded', {
    startedAt: '2026-10-05T12:00:00Z', finishedAt: '2026-10-05T12:01:00Z',
  }),
  observe: () => observationOf('a'),
  transcript: (agent: string | null) => (
    agent === null
      ? mainTranscript()
      : activityPage([{role: 'assistant', content: 'The child finished.'}], {running: false})
  ),
  overview: () => Response.json(runOverview()),
});

function observationOf(id: string): Response {
  return Response.json({
    run_id: 'run-1',
    child: row(id, 'succeeded', {started_at: '2026-10-05T12:00:10Z', finished_at: '2026-10-05T12:00:50Z'}),
    controls: [], questions: [], result: null,
  });
}

const dossier = (dock: DlInspectorTraces): Element | null => dock.querySelector('dl-run-overview');

/** Open the Run quality details: the findings a reader can follow live behind that disclosure. */
async function openQuality(dock: DlInspectorTraces): Promise<void> {
  const toggle = dock.querySelector<HTMLButtonElement>('[data-quality-toggle]');
  if (toggle !== null && toggle.getAttribute('aria-expanded') !== 'true') {
    toggle.click();
    await dock.updateComplete;
  }
}

it('states the Run in one summary line: its state, duration, agents, steps and tokens', async () => {
  Date.now = () => NOW;
  serve(servingRun());
  const dock = await mount(sourceFor().source, 800);
  await waitFor(() => dossier(dock) !== null);

  const band = dock.querySelector<HTMLElement>('[data-overview] [data-overview-summary]')!;
  const words = (band.textContent ?? '').replace(/\s+/g, ' ').trim();
  expect(words, 'the Run state').to.contain('Done');
  expect(words, 'how long it took').to.contain('1m');
  expect(words, 'how many agents').to.contain('2 agents');
  expect(words, 'how many steps').to.contain('6 steps');
  expect(words, 'what it spent').to.contain('9K tokens');
  expect(dock.querySelector('[data-overview]')!.getAttribute('data-overview'), 'the band leads the dossier')
    .to.equal('');
});

it('draws the dossier: the bar, what was delivered, the agents and a calm Run quality line', async () => {
  serve(servingRun());
  const dock = await mount(sourceFor().source, 800);
  await waitFor(() => dossier(dock) !== null);
  const panel = dossier(dock)!;
  expect(panel.querySelector('[data-bar]'), 'the Run\'s composition').to.not.equal(null);
  expect(panel.querySelector('[data-delivery]'), 'what the Run delivered').to.not.equal(null);
  expect(panel.querySelector('[data-ledger]'), 'its agents as a ledger').to.not.equal(null);
  expect(panel.querySelector('[data-agent-row][data-agent="main"]')).to.not.equal(null);
  expect(panel.querySelector('[data-agent-row][data-agent="a"]')).to.not.equal(null);
  const attention = panel.querySelector<HTMLElement>('[data-attention]')!;
  expect(attention.textContent).to.contain('Failed 1');
  expect(attention.textContent).to.contain('Repeated 1');
  expect(panel.querySelectorAll('[data-finding]').length, 'both findings are listed').to.equal(2);
});

it('opens a finding in its agent\'s page and rings the step where it sits', async () => {
  serve(servingRun());
  const dock = await mount(sourceFor().source, 800);
  await waitFor(() => dossier(dock) !== null);
  await openQuality(dock);
  const finding = dock.querySelector<HTMLElement>('[data-finding][data-agent="main"]')!;
  expect(finding.dataset.sequence).to.equal('1');

  finding.click();
  await waitFor(() => session(dock).childSessionId === 'main');
  // The page's timeline opens on the step and rings it, so a reader can see what the finding was.
  await waitFor(() => dock.querySelector('dl-agent-session [data-sequence="1"][data-focus="true"]') !== null);
  expect(dock.querySelector('dl-agent-session details')!.hasAttribute('open'), 'the fold opens to show it')
    .to.equal(true);
});

it('opens the agent a finding belongs to when that is not the agent on show', async () => {
  serve(servingRun());
  const dock = await mount(sourceFor().source, 800);
  await waitFor(() => dossier(dock) !== null);
  await openQuality(dock);
  dock.querySelector<HTMLElement>('[data-finding][data-agent="a"]')!.click();
  await waitFor(() => session(dock).childSessionId === 'a');
  await waitFor(() => (dock.textContent ?? '').includes('The child finished.'));
  expect(dock.querySelector('dl-agent-session [data-focus="true"]'), 'the delivery step was not on the page')
    .to.equal(null);
});

it('selects an agent from its ledger row, and clears the finding\'s ring', async () => {
  serve(servingRun());
  const dock = await mount(sourceFor().source, 800);
  await waitFor(() => dossier(dock) !== null);
  await openQuality(dock);
  dock.querySelector<HTMLElement>('[data-finding][data-agent="main"]')!.click();
  await waitFor(() => dock.querySelector('dl-agent-session [data-focus="true"]') !== null);

  dock.querySelector<HTMLElement>('[data-agent-row][data-agent="a"]')!.click();
  await waitFor(() => session(dock).childSessionId === 'a');
  expect(dock.querySelector('dl-agent-session [data-focus="true"]')).to.equal(null);
});

it('marks the agent on show as current across the dossier, and moves the mark with the page', async () => {
  serve(servingRun());
  const dock = await mount(sourceFor().source, 800);
  await waitFor(() => dossier(dock) !== null);

  // The Inspector opens on the Run's own agent, and the dossier says which page that is.
  await waitFor(() => dock.querySelector('[data-agent-row][data-agent="main"]')?.getAttribute('aria-current') === 'true');
  expect(dock.querySelector('[data-agent-row][data-agent="a"]')!.hasAttribute('aria-current')).to.equal(false);

  dock.querySelector<HTMLElement>('[data-agent-row][data-agent="a"]')!.click();
  await waitFor(() => session(dock).childSessionId === 'a');
  await waitFor(() => dock.querySelector('[data-agent-row][data-agent="a"]')?.getAttribute('aria-current') === 'true');
  expect(dock.querySelector('[data-agent-row][data-agent="main"]')!.hasAttribute('aria-current')).to.equal(false);

  dock.querySelector<HTMLElement>('[data-dimension="agent"]')!.click();
  await waitFor(
    () => dock.querySelector('[data-legend-agent="a"]')?.getAttribute('aria-current') === 'true',
  );
  expect(dock.querySelector('[data-bar-segment][data-key="a"]')!.getAttribute('aria-current'))
    .to.equal('true');
  expect(dock.querySelector('[data-bar-segment][data-key="main"]')!.hasAttribute('aria-current'))
    .to.equal(false);
});

it('never asks for the retired Run timeline, in either panel mode', async () => {
  const served = serve(servingRun());
  const dock = await mount(sourceFor().source, 800);
  await waitFor(() => dossier(dock) !== null);

  // A search switches the panel to the Run's matching steps, and a follow read asks again.
  const box = dock.querySelector<HTMLInputElement>('input[type="search"]')!;
  box.value = 'northwind';
  box.dispatchEvent(new Event('input', {bubbles: true}));
  await waitFor(() => dock.querySelector('[data-step-search]') !== null);
  stated(dock, 'run-1');
  await settle(60);

  const retired = served.filter((request) => (
    request.path.endsWith('/timeline') || request.path.endsWith('/timeline/step')
  ));
  expect(retired.map((request) => request.path)).to.deep.equal([]);
});

it('says it is loading, and that the Run has no panel until it has a step', async () => {
  let release!: (response: Response) => void;
  const pending = new Promise<Response>((resolve) => { release = resolve; });
  const served = serve({
    page: () => roster([row('a', 'running')]),
    overview: () => pending,
    transcript: () => mainTranscript(),
  });
  const dock = await mount(sourceFor().source, 800);
  await waitFor(() => dock.querySelector('[data-overview-loading]') !== null);
  expect(served.some((request) => request.path.endsWith('/overview'))).to.equal(true);
  release(Response.json(runOverview()));
  await waitFor(() => dossier(dock) !== null);
  expect(dock.querySelector('[data-overview-loading]')).to.equal(null);

  // A Run that states no steps draws no Run panel at all: the agent page is the view.
  document.body.replaceChildren();
  serve({page: () => roster([], null, 'running'), transcript: () => mainTranscript()});
  const empty = await mount(sourceFor().source, 800);
  await waitFor(() => shown(empty).includes('Main agent'));
  await settle(30);
  expect(empty.querySelector('[data-overview]')).to.equal(null);
  expect(rows(empty)).to.deep.equal([]);
});

it('says the overview could not be read, and keeps the agents readable beside it', async () => {
  serve({
    page: () => roster([row('a', 'succeeded')], null, 'succeeded'),
    overview: () => new Response('unavailable', {status: 503}),
    transcript: () => mainTranscript(),
  });
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => dock.querySelector('[data-overview-failed]') !== null);
  expect(rows(dock)).to.have.length(1);
  expect(shown(dock)).to.contain('Main agent');
});

it('draws the delivery card from the answer the chat holds', async () => {
  const answer = {
    answerText: 'Clause 9.2 allows it.',
    parts: [],
    sources: [{id: '1', title: 'northwind-msa.pdf', sourceUrl: null, downloadUrl: null, chunks: []}],
    linkCards: [], evidenceImages: [], artifacts: [],
    artifactOutcome: {status: 'complete', issues: []},
  } as never;
  serve(servingRun());
  const dock = await mount(sourceFor('run-1', answeredTurn({
    state: 'succeeded', presentation: answer, usage: {usage_details: {total_tokens: 9000}},
  })).source, 800);
  await waitFor(() => dossier(dock) !== null);
  const delivery = dock.querySelector<HTMLElement>('dl-run-overview [data-delivery]')!;
  expect(delivery.getAttribute('data-delivery-state')).to.equal('delivered');
  expect(delivery.textContent).to.contain('Delivered');
  expect(delivery.textContent).to.contain('Sources');
});
