// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The Run overview on its own: what its four sections state, and what a reader's press raises.
 *
 *  The element is mounted with the product's own read (the overview's wire through `serve` and the api
 *  parse), so the tests exercise the same translation the Inspector will: the wire's snake_case, the
 *  capped findings order, and the agent keys. The page owns a one-second clock for its moving times,
 *  so the suite fixes `Date.now` at the fixture's `NOW` and drives the ticks itself.
 *
 *  Layout assertions live in this file rather than a product-styles one: until the Inspector mounts the
 *  component in Phase 2, the application build does not include its stylesheet, so the product-styles
 *  page cannot project its class names. This file links the design system's tokens and the feature's
 *  own stylesheet, which is the same cascade the product page provides, and reads the settled layout.
 */

import {expect} from '@esm-bundle/chai';
import type {
  AgentChildStatus,
  AgentRunStatus,
  AnswerPresentation,
  ArtifactAttestation,
  RunOverview,
} from '../api/conversations.ts';
import {getAnswerRunOverview} from '../api/conversations.ts';
import {setLanguagePreference} from '../i18n/locale.ts';
import {waitFor} from '../testing/dom.ts';
import {NOW, overview as overviewWire, serve, sourceFor} from '../testing/traces.ts';
import type {DlRunOverview, ListedChild} from './run-overview.ts';
import './run-overview.ts';

const originalFetch = window.fetch;
const originalNow = Date.now;
const originalSetInterval = window.setInterval;
const originalClearInterval = window.clearInterval;

/** One agent as the roster states it, with only the fields a row reads set. */
function listed(
  id: string,
  extra: Partial<AgentChildStatus> = {},
): AgentChildStatus & {childSessionId: string} {
  return {
    childSessionId: id,
    status: 'succeeded',
    objective: id === 'main' ? undefined : `Read ${id}`,
    modelRole: 'query',
    usage: null,
    operationId: null,
    operationSequence: null,
    operationStatus: null,
    cancellationOrigin: null,
    summary: null,
    resultHandles: [],
    startedAt: null,
    finishedAt: null,
    pendingQuestions: 0,
    resultSequence: null,
    feedback: null,
    ...extra,
  };
}

const run = (extra: Partial<AgentRunStatus> = {}): AgentRunStatus => ({
  status: 'succeeded',
  startedAt: '2026-10-05T12:00:00Z',
  finishedAt: '2026-10-05T12:01:00Z',
  ...extra,
});

/** One Artifact as a presentation states it; the delivery card reads its label, file and record. */
function artifact(resourceId: string, extra: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    resource_id: resourceId,
    media_type: 'text/markdown',
    label: `File ${resourceId}`,
    filename: `${resourceId}.md`,
    byte_size: 120,
    digest: 'abc',
    presentation: 'markdown',
    status: 'available',
    uri: `artifact://${resourceId}`,
    width: null,
    height: null,
    data_url: null,
    download_url: null,
    presentation_url: null,
    issue: null,
    attestation: null,
    ...extra,
  };
}

function attestation(
  status: 'attested' | 'unattested' | 'unknown',
  extra: Partial<ArtifactAttestation> = {},
): ArtifactAttestation {
  return {
    status,
    skill: status === 'attested' ? 'research-report' : null,
    verifier: status === 'attested' ? 'verify.py' : null,
    reason: status === 'unattested' ? 'no_skill_loaded' : null,
    basis: null,
    requiredReadsMissing: null,
    ...extra,
  };
}

/** One Answer as the Run published it; a test states the parts the delivery card reads. The pieces
 *  are wire-shaped where a test only reads their words, so the helper states a valid Answer's
 *  defaults and lets a test replace the parts it names. */
function answer(extra: Record<string, unknown> = {}): AnswerPresentation {
  return {
    answerText: 'The answer.',
    parts: [],
    sources: [
      {id: '1', title: 'northwind-msa.pdf', sourceUrl: null, downloadUrl: null, chunks: []},
      {id: '2', title: 'amendment-3.pdf', sourceUrl: null, downloadUrl: null, chunks: []},
    ],
    linkCards: [],
    evidenceImages: [],
    artifacts: [artifact('summary', {attestation: attestation('attested')})],
    artifactOutcome: {status: 'complete', issues: []},
    ...extra,
  } as unknown as AnswerPresentation;
}

/** Read one overview the way the dock does, and mount the element with it. */
async function readOverview(wire: Record<string, unknown> = overviewWire()): Promise<RunOverview> {
  serve({overview: () => Response.json(wire)});
  return await getAnswerRunOverview('run-1');
}

/** The component in the document, handed whatever a test needs it to show. */
async function mount(
  data: RunOverview | null,
  options: {
    run?: AgentRunStatus | null;
    presentation?: AnswerPresentation | null;
    entries?: ListedChild[];
    lead?: ListedChild | null;
    loading?: boolean;
    failed?: boolean;
    selected?: string | null;
  } = {},
): Promise<DlRunOverview> {
  const element = document.createElement('dl-run-overview') as DlRunOverview;
  element.overview = data;
  element.run = options.run === undefined ? null : options.run;
  element.presentation = options.presentation ?? null;
  element.entries = options.entries ?? [];
  element.lead = options.lead === undefined ? listed('main') : options.lead;
  element.loading = options.loading ?? false;
  element.failed = options.failed ?? false;
  element.selected = options.selected ?? null;
  document.body.append(element);
  await element.updateComplete;
  return element;
}

/** The element's words as a reader takes them in, without the template's layout whitespace. */
const text = (node: Element): string => (node.textContent ?? '').replace(/\s+/g, ' ').trim();

const agentRows = (element: DlRunOverview): HTMLElement[] => [
  ...element.querySelectorAll<HTMLElement>('[data-agent-row]'),
];

const findingRows = (element: DlRunOverview): HTMLElement[] => [
  ...element.querySelectorAll<HTMLElement>('[data-finding]'),
];

/** The element raised these intents, as `[name, detail]` pairs. */
function raised(element: DlRunOverview): Array<[string, unknown]> {
  const events: Array<[string, unknown]> = [];
  element.addEventListener('dl-run-agent-select', (event) => {
    events.push(['agent', (event as CustomEvent).detail]);
  });
  element.addEventListener('dl-run-finding-select', (event) => {
    events.push(['finding', (event as CustomEvent).detail]);
  });
  return events;
}

function agentButton(element: DlRunOverview, agent: string): HTMLButtonElement | null {
  return element.querySelector<HTMLButtonElement>(`[data-legend-agent="${CSS.escape(agent)}"]`);
}

function agentRow(element: DlRunOverview, agent: string): HTMLButtonElement | null {
  return element.querySelector<HTMLButtonElement>(`[data-agent-row][data-agent="${CSS.escape(agent)}"]`);
}

beforeEach(() => {
  Date.now = () => NOW;
});

afterEach(() => {
  window.fetch = originalFetch;
  Date.now = originalNow;
  window.setInterval = originalSetInterval;
  window.clearInterval = originalClearInterval;
  document.body.replaceChildren();
});

/** The page's own one-second clock, recorded instead of run: a test decides when it ticks. */
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

it('draws the Run by kind, and the same Run by agent when the reader switches', async () => {
  const wire = overviewWire({
    totals: {
      steps: 8,
      kinds: {thinking: 2, tool: 3, say: 1, answer: 1, instruction: 0, delivery: 1},
      decisions: 2,
      marks: {failed: 1, degraded: 1, repeated: 0, redirect: 0},
      tool_time_ms: 2400,
    },
    agents: [
      {
        agent: 'main', label: 'main', order: 0, steps: 5, tool_calls: 2,
        marks: {failed: 1, degraded: 0, repeated: 0, redirect: 0},
        tool_time_ms: 1600, first_at: '2026-10-05T12:00:00Z', last_at: '2026-10-05T12:00:40Z',
      },
      {
        agent: 'child-1', label: 'Read the filing', order: 1, steps: 3, tool_calls: 1,
        marks: {failed: 0, degraded: 1, repeated: 0, redirect: 0},
        tool_time_ms: 800, first_at: '2026-10-05T12:00:10Z', last_at: '2026-10-05T12:00:50Z',
      },
    ],
  });
  const element = await mount(await readOverview(wire));

  // Kind is the default reading, and the bar is one part per kind the Run holds.
  const byKind = element.querySelector<HTMLButtonElement>('[data-dimension="kind"]')!;
  const byAgent = element.querySelector<HTMLButtonElement>('[data-dimension="agent"]')!;
  expect(byKind.getAttribute('aria-pressed')).to.equal('true');
  expect(byAgent.getAttribute('aria-pressed')).to.equal('false');
  expect(element.querySelectorAll('[data-bar-segment]').length).to.equal(5);
  expect(
    [...element.querySelectorAll<HTMLElement>('[data-bar-segment]')].map((part) => part.dataset.key),
    'the product order of kinds',
  ).to.deep.equal(['thinking', 'tool', 'say', 'answer', 'delivery']);
  expect(element.querySelector<HTMLElement>('[data-key="tool"]')!.style.flexGrow).to.equal('3');
  // The legend names every segment and its exact count; the kind bar is decorative, but every part
  // still states what it is on hover.
  expect(element.querySelector('[data-bar]')!.getAttribute('aria-hidden')).to.equal('true');
  const kindParts = [...element.querySelectorAll<HTMLElement>('[data-bar-segment]')];
  expect(kindParts.every((part) => part.tagName === 'SPAN'), 'kinds are not controls').to.equal(true);
  expect(element.querySelector<HTMLElement>('[data-key="tool"] [class*="barTip"]')!.textContent)
    .to.equal('Tool calls · 3');
  expect(element.querySelector<HTMLElement>('[data-key="thinking"]')!.getAttribute('data-edge')).to.equal('first');
  expect(element.querySelector<HTMLElement>('[data-key="delivery"]')!.getAttribute('data-edge')).to.equal('last');
  const kindLegend = text(element.querySelector<HTMLElement>('[data-legend="kind"]')!);
  expect(kindLegend).to.contain('Thoughts 2');
  expect(kindLegend).to.contain('Tool calls 3');
  expect(kindLegend).to.contain('Delivery note 1');

  byAgent.click();
  await element.updateComplete;
  expect(byAgent.getAttribute('aria-pressed')).to.equal('true');
  expect(byKind.getAttribute('aria-pressed')).to.equal('false');
  expect(
    [...element.querySelectorAll<HTMLElement>('[data-bar-segment]')].map((part) => part.dataset.key),
  ).to.deep.equal(['main', 'child-1']);
  const agentLegend = element.querySelector<HTMLElement>('[data-legend="agent"]')!;
  expect(text(agentLegend)).to.contain('Main agent 5');
  expect(text(agentLegend)).to.contain('Read the filing 3');
  // By agent the bar's parts are real buttons: each names itself, states its count, and opens its page.
  const agentParts = [...element.querySelectorAll<HTMLElement>('[data-bar-segment]')];
  expect(agentParts.every((part) => part.tagName === 'BUTTON')).to.equal(true);
  expect(element.querySelector('[data-bar]')!.getAttribute('aria-hidden'), 'the agent bar is a group')
    .to.equal(null);
  expect(element.querySelector<HTMLButtonElement>('[data-bar-segment][data-key="child-1"]')!.getAttribute('aria-label'))
    .to.equal('Read the filing: 3 steps');
  expect(element.querySelector<HTMLElement>('[data-key="child-1"] [class*="barTip"]')!.textContent)
    .to.equal('Read the filing · 3 steps');

  byKind.click();
  await element.updateComplete;
  expect(element.querySelectorAll('[data-bar-segment]').length).to.equal(5);
});

it('raises the agent a legend segment or a ledger row names', async () => {
  const wire = overviewWire({
    agents: [
      {
        agent: 'main', label: 'main', order: 0, steps: 5, tool_calls: 2,
        marks: {failed: 0, degraded: 0, repeated: 0, redirect: 0}, tool_time_ms: 0,
        first_at: '2026-10-05T12:00:00Z', last_at: '2026-10-05T12:00:30Z',
      },
      {
        agent: 'child-1', label: 'Read the filing', order: 1, steps: 3, tool_calls: 1,
        marks: {failed: 0, degraded: 0, repeated: 0, redirect: 0}, tool_time_ms: 0,
        first_at: '2026-10-05T12:00:10Z', last_at: '2026-10-05T12:00:50Z',
      },
    ],
    findings: [],
    findings_total: 0,
  });
  const element = await mount(await readOverview(wire), {
    run: run(),
    lead: listed('main', {status: 'succeeded'}),
    entries: [listed('child-1', {status: 'succeeded'})],
  });
  const events = raised(element);
  element.querySelector<HTMLButtonElement>('[data-dimension="agent"]')!.click();
  await element.updateComplete;
  element.querySelector<HTMLButtonElement>('[data-bar-segment][data-key="child-1"]')!.click();
  agentButton(element, 'child-1')!.click();
  agentRow(element, 'main')!.click();
  expect(events).to.deep.equal([
    ['agent', {agent: 'child-1'}],
    ['agent', {agent: 'child-1'}],
    ['agent', {agent: 'main'}],
  ]);
  expect(agentButton(element, 'child-1')!.getAttribute('aria-label')).to.equal('Read the filing: 3 steps');
});

it('marks the agent on show in the bar, the legend and the ledger, with no row chevron', async () => {
  const wire = overviewWire({
    agents: [
      {
        agent: 'main', label: 'main', order: 0, steps: 5, tool_calls: 2,
        marks: {failed: 0, degraded: 0, repeated: 0, redirect: 0}, tool_time_ms: 0,
        first_at: '2026-10-05T12:00:00Z', last_at: '2026-10-05T12:00:30Z',
      },
      {
        agent: 'child-1', label: 'Read the filing', order: 1, steps: 3, tool_calls: 1,
        marks: {failed: 0, degraded: 0, repeated: 0, redirect: 0}, tool_time_ms: 0,
        first_at: '2026-10-05T12:00:10Z', last_at: '2026-10-05T12:00:50Z',
      },
    ],
    findings: [],
    findings_total: 0,
  });
  const element = await mount(await readOverview(wire), {
    run: run(),
    lead: listed('main', {status: 'succeeded'}),
    entries: [listed('child-1', {status: 'succeeded'})],
    selected: 'child-1',
  });
  element.querySelector<HTMLButtonElement>('[data-dimension="agent"]')!.click();
  await element.updateComplete;

  expect(element.querySelector('[data-bar-segment][data-key="child-1"]')!.getAttribute('aria-current')).to.equal('true');
  expect(element.querySelector('[data-bar-segment][data-key="main"]')!.hasAttribute('aria-current')).to.equal(false);
  expect(element.querySelector('[data-legend-agent="child-1"]')!.getAttribute('aria-current')).to.equal('true');
  expect(element.querySelector('[data-legend-agent="main"]')!.hasAttribute('aria-current')).to.equal(false);
  expect(agentRow(element, 'child-1')!.getAttribute('aria-current')).to.equal('true');
  expect(agentRow(element, 'main')!.hasAttribute('aria-current')).to.equal(false);

  // A ledger row navigates to its agent's page; it is not an expand/next control, so no chevron.
  expect(agentRow(element, 'child-1')!.querySelector('[class*="rowDisclosure"]')).to.equal(null);
  expect(agentRow(element, 'child-1')!.textContent).to.not.contain('>');
  expect(agentRow(element, 'child-1')!.querySelectorAll('svg').length, 'only the state glyph remains')
    .to.equal(1);
});

it('places each agent\'s span over the Run clock and dots its decisions', async () => {
  const wire = overviewWire({
    agents: [
      {
        agent: 'main', label: 'main', order: 0, steps: 3, tool_calls: 1,
        marks: {failed: 1, degraded: 0, repeated: 0, redirect: 0}, tool_time_ms: 0,
        first_at: '2026-10-05T12:00:00Z', last_at: '2026-10-05T12:00:30Z',
      },
      {
        agent: 'child-1', label: 'Read the filing', order: 1, steps: 3, tool_calls: 1,
        marks: {failed: 0, degraded: 0, repeated: 0, redirect: 0}, tool_time_ms: 0,
        first_at: '2026-10-05T12:00:10Z', last_at: '2026-10-05T12:00:50Z',
      },
    ],
    findings: [
      {
        agent: 'main', sequence: 2, kind: 'tool', at: '2026-10-05T12:00:20Z', name: 'read',
        subject: 'northwind-msa.pdf', outcome: 'failed', marks: ['failed'], excerpt: 'No file.',
      },
      {
        agent: 'child-1', sequence: 5, kind: 'delivery', at: '2026-10-05T12:00:45Z', name: null,
        subject: null, outcome: null, marks: [], excerpt: 'No verifier.',
      },
    ],
    findings_total: 2,
  });
  const element = await mount(await readOverview(wire), {
    run: run({startedAt: '2026-10-05T12:00:00Z', finishedAt: '2026-10-05T12:01:00Z'}),
    lead: listed('main', {status: 'succeeded'}),
    entries: [listed('child-1', {status: 'succeeded'})],
  });
  const mainLane = agentRow(element, 'main')!.querySelector<HTMLElement>('[data-lane-span]')!;
  // The Run clock is 60 seconds: main acts for the first 30, the child from 10s to 50s.
  expect(mainLane.style.insetInlineStart).to.equal('0%');
  expect(mainLane.style.width).to.equal('50%');
  const childLane = agentRow(element, 'child-1')!.querySelector<HTMLElement>('[data-lane-span]')!;
  expect(Number.parseFloat(childLane.style.insetInlineStart)).to.be.closeTo(100 / 6, 0.01);
  expect(Number.parseFloat(childLane.style.width)).to.be.closeTo(100 * 40 / 60, 0.01);
  expect(mainLane.getAttribute('title'), 'the span names the agent and its time').to.equal('Main agent · 30s');
  const mainDots = agentRow(element, 'main')!.querySelectorAll('[data-lane-dot]');
  expect(mainDots.length).to.equal(1);
  expect(mainDots[0]!.getAttribute('data-mark')).to.equal('failed');
  expect(mainDots[0]!.getAttribute('title'), 'a dot names its mark and its time').to.match(/^Failed · .+:/);
  expect(Number.parseFloat((mainDots[0] as HTMLElement).style.insetInlineStart)).to.be.closeTo(100 / 3, 0.01);
  const childDots = agentRow(element, 'child-1')!.querySelectorAll('[data-lane-dot]');
  expect(childDots[0]!.getAttribute('data-mark')).to.equal('delivery');
  // The row's numbers are duration · tokens · steps · tool calls: the marks it states live behind
  // the quality fold, so the default ledger carries no developer vocabulary.
  const rowWords = text(agentRow(element, 'main')!);
  expect(rowWords).to.contain('3 steps');
  expect(rowWords).to.contain('1 tool call');
  expect(rowWords).to.not.contain('Failed');
  expect(rowWords).to.not.contain('Repeated');
});

it('says a running Run has delivered nothing yet, and a failed one delivered nothing', async () => {
  const running = await mount(await readOverview(), {
    run: run({status: 'running', finishedAt: null}),
    lead: listed('main', {status: 'running', startedAt: '2026-10-05T12:00:00Z'}),
  });
  expect(running.querySelector('[data-delivery]')!.getAttribute('data-delivery-state')).to.equal('running');
  expect(text(running.querySelector<HTMLElement>('[data-delivery]')!)).to.contain(
    'The Run is still going, so nothing is delivered yet.',
  );
  expect(running.querySelector('[data-delivery-counts]'), 'a running Run has no verdict to number').to.equal(null);

  const failed = await mount(await readOverview(), {
    run: run({status: 'failed'}),
    lead: listed('main', {status: 'failed'}),
  });
  expect(failed.querySelector('[data-delivery]')!.getAttribute('data-delivery-state')).to.equal('failed');
  expect(text(failed.querySelector<HTMLElement>('[data-delivery]')!)).to.contain(
    'The Run failed, so nothing was delivered.',
  );
  expect(failed.querySelector('[data-delivery]')!.querySelector('[aria-label*="Attested"]')).to.equal(null);
});

it('states what a delivered Run published, with its Attestation records and issues', async () => {
  const wire = overviewWire({
    totals: {
      steps: 6,
      kinds: {thinking: 0, tool: 2, say: 1, answer: 1, instruction: 0, delivery: 2},
      decisions: 1,
      marks: {failed: 0, degraded: 0, repeated: 0, redirect: 0},
      tool_time_ms: 0,
    },
    findings: [
      {
        agent: 'main', sequence: 7, kind: 'delivery', at: '2026-10-05T12:00:50Z', name: null,
        subject: null, outcome: null, marks: [], excerpt: 'The record names no verifier.',
      },
    ],
    findings_total: 1,
  });
  const presentation = answer({
    artifacts: [
      artifact('summary', {attestation: attestation('attested')}),
      artifact('notes', {attestation: attestation('unattested')}),
      artifact('scan', {
        status: 'unavailable',
        issue: {kind: 'missing_bytes', description: 'The bytes were never stored.', resource_id: 'scan'},
      }),
    ],
    artifactOutcome: {
      status: 'partial',
      issues: [{kind: 'missing_bytes', description: 'One file could not be published.', resourceId: 'scan'}],
    },
  });
  const element = await mount(await readOverview(wire), {
    run: run(),
    presentation,
    lead: listed('main', {status: 'succeeded'}),
  });
  const delivery = element.querySelector<HTMLElement>('[data-delivery]')!;
  expect(delivery.getAttribute('data-delivery-state')).to.equal('delivered');
  expect(text(delivery)).to.contain('Delivered');
  expect(delivery.querySelector('[data-delivery-count="sources"] dd')!.textContent).to.equal('2');
  expect(delivery.querySelector('[data-delivery-count="artifacts"] dd')!.textContent).to.equal('3');
  // A pre-delivery warning is not a number of checks: the Run quality line owns that count.
  expect(delivery.querySelector('[data-delivery-count="checks"]'), 'no delivery-checks stat').to.equal(null);
  expect(delivery.querySelector('[data-delivery-text-only]')).to.equal(null);
  expect(delivery.querySelectorAll('[data-artifact]').length).to.equal(3);
  expect(text(delivery)).to.contain('File summary');
  expect(text(delivery)).to.contain('Attested by research-report');
  expect(text(delivery)).to.contain('Not attested');
  expect(text(delivery)).to.contain('Unavailable');
  expect(text(delivery.querySelector<HTMLElement>('[data-attestation-buckets]')!)).to.contain('Attested 1');
  expect(text(delivery.querySelector<HTMLElement>('[data-attestation-buckets]')!)).to.contain('Not attested 1');
  expect(text(delivery.querySelector<HTMLElement>('[data-artifact-issues]')!)).to.contain(
    'One file could not be published.',
  );
  expect(text(delivery), 'the card does not repeat the Answer\'s text').to.not.contain('The answer.');
});

it('draws a source count only when there are sources', async () => {
  const element = await mount(await readOverview(), {
    run: run(),
    presentation: answer({sources: [], artifacts: [artifact('summary')]}),
    lead: listed('main'),
  });
  const delivery = element.querySelector<HTMLElement>('[data-delivery]')!;
  expect(delivery.querySelector('[data-delivery-count="sources"]')).to.equal(null);
  expect(delivery.querySelector('[data-delivery-count="artifacts"] dd')!.textContent).to.equal('1');
  expect(delivery.querySelectorAll('[data-artifact]').length).to.equal(1);
  expect(delivery.querySelector('[data-delivery-text-only]')).to.equal(null);
});

it('says a text-only answer in one quiet line, with no zero counts and no empty list', async () => {
  const element = await mount(await readOverview(), {
    run: run(),
    presentation: answer({sources: [], artifacts: []}),
    lead: listed('main'),
  });
  const delivery = element.querySelector<HTMLElement>('[data-delivery]')!;
  expect(delivery.querySelector('[data-delivery-counts]'), 'nothing to count').to.equal(null);
  expect(delivery.querySelector('[data-artifacts]'), 'no empty artifact list').to.equal(null);
  const sentence = delivery.querySelector<HTMLElement>('[data-delivery-text-only]')!;
  expect(text(sentence)).to.equal('Text-only answer: no cited sources or published files.');
  expect(sentence.classList.contains('dl-note'), 'a quiet line is the design system\'s').to.equal(true);
});

it('orders findings failed, degraded, redirected, repeated, delivery — and raises the exact step', async () => {
  const wire = overviewWire({
    totals: {
      steps: 6,
      kinds: {thinking: 0, tool: 2, say: 0, answer: 1, instruction: 0, delivery: 1},
      decisions: 4,
      marks: {failed: 1, degraded: 1, repeated: 1, redirect: 1},
      tool_time_ms: 0,
    },
    findings: [
      {
        agent: 'child-1', sequence: 9, kind: 'tool', at: '2026-10-05T12:00:40Z', name: 'read',
        subject: null, outcome: 'succeeded', marks: ['repeated'], excerpt: 'Read again.',
      },
      {
        agent: 'main', sequence: 8, kind: 'delivery', at: '2026-10-05T12:00:50Z', name: null,
        subject: null, outcome: null, marks: [], excerpt: 'The record names no verifier.',
      },
      {
        agent: 'main', sequence: 2, kind: 'tool', at: '2026-10-05T12:00:15Z', name: 'search_web',
        subject: 'northwind merger', outcome: 'failed', marks: ['failed'], excerpt: 'No results.',
      },
      {
        agent: 'main', sequence: 4, kind: 'tool', at: '2026-10-05T12:00:20Z', name: 'search_web',
        subject: 'northwind merger', outcome: 'degraded', marks: ['degraded'], excerpt: 'Degraded.',
      },
      {
        agent: 'child-1', sequence: 6, kind: 'tool', at: '2026-10-05T12:00:30Z', name: 'read',
        subject: null, outcome: 'succeeded', marks: ['redirect'], excerpt: 'Turned.',
      },
    ],
    findings_total: 5,
  });
  const element = await mount(await readOverview(wire), {
    run: run(),
    lead: listed('main', {status: 'succeeded'}),
    entries: [listed('child-1', {status: 'succeeded'})],
  });
  // The default section is one calm line; the triage detail is one disclosure away.
  const quality = element.querySelector<HTMLElement>('[data-quality]')!;
  expect(quality.getAttribute('data-quality-state')).to.equal('noted');
  expect(text(quality.querySelector<HTMLElement>('[data-quality-line]')!)).to.equal(
    '5 details were recorded during the Run (retries, degraded searches, or delivery notes).',
  );
  expect(text(quality.querySelector<HTMLElement>('[data-quality-line]')!)).to.not.contain('Failed');
  const details = quality.querySelector<HTMLElement>('#run-overview-quality-details')!;
  expect(details.hidden, 'the details start folded').to.equal(true);
  const head = quality.querySelector<HTMLElement>('[data-quality-head]')!;
  const toggle = head.querySelector<HTMLButtonElement>('[data-quality-toggle]')!;
  // The toggle is section chrome in the head's trailing slot, and its label carries no count: the
  // calm line above already states it.
  expect(toggle.parentElement, 'the toggle sits in the section head').to.equal(head);
  expect(head.querySelector('#run-overview-quality')).to.not.equal(null);
  expect(toggle.textContent!.trim()).to.equal('Show details');
  expect(toggle.textContent!).to.not.contain('5');
  expect(toggle.getAttribute('aria-expanded')).to.equal('false');
  expect(toggle.hasAttribute('hidden')).to.equal(false);
  // The expanded body is a bounded, focusable, labelled scroll region, not an unbounded block.
  expect(details.getAttribute('role')).to.equal('region');
  expect(details.getAttribute('tabindex')).to.equal('0');
  expect(details.getAttribute('aria-label')).to.equal('Run details');
  toggle.click();
  await element.updateComplete;
  expect(details.hidden).to.equal(false);
  expect(toggle.getAttribute('aria-expanded')).to.equal('true');
  expect(toggle.textContent!.trim()).to.equal('Hide details');
  expect(quality.querySelector('#run-overview-quality-details')).to.equal(details);
  expect(toggle.getAttribute('aria-controls')).to.equal('run-overview-quality-details');
  // Closing puts the section back to its head and calm line.
  toggle.click();
  await element.updateComplete;
  expect(details.hidden).to.equal(true);
  expect(toggle.textContent!.trim()).to.equal('Show details');
  toggle.click();
  await element.updateComplete;
  expect(details.hidden).to.equal(false);

  expect(findingRows(element).map((row) => `${row.dataset.agent}:${row.dataset.sequence}`)).to.deep.equal([
    'main:2', 'main:4', 'child-1:6', 'child-1:9', 'main:8',
  ]);
  const events = raised(element);
  findingRows(element)[0]!.click();
  expect(events).to.deep.equal([['finding', {agent: 'main', sequence: 2}]]);
  // A finding states its mark in words, its agent's name, and the tool's own verb.
  const first = findingRows(element)[0]!;
  expect(text(first)).to.contain('Failed');
  expect(text(first)).to.contain('Main agent');
  expect(text(first)).to.contain('Searching the web');
  expect(text(first)).to.contain('northwind merger');
  expect(text(first)).to.contain('No results.');
  expect(text(element.querySelector<HTMLElement>('[data-attention-counts]')!)).to.contain('Failed 1');
  expect(text(element.querySelector<HTMLElement>('[data-attention-counts]')!)).to.contain('Delivery note 1');
});

it('names an agent the roster page does not hold rather than dropping its finding', async () => {
  const wire = overviewWire({
    findings: [{
      agent: 'child-9', sequence: 3, kind: 'tool', at: '2026-10-05T12:00:20Z', name: 'read',
      subject: null, outcome: 'failed', marks: ['failed'], excerpt: 'No file.',
    }],
    findings_total: 1,
  });
  const element = await mount(await readOverview(wire));
  expect(findingRows(element).length).to.equal(1);
  element.querySelector<HTMLButtonElement>('[data-quality-toggle]')!.click();
  await element.updateComplete;
  expect(text(findingRows(element)[0]!)).to.contain('Child agent');
  // The ledger still states only the agents the overview names.
  expect(agentRows(element).map((row) => row.dataset.agent)).to.deep.equal(['main']);
});

it('states a clean Run as one calm line, with nothing to open', async () => {
  const wire = overviewWire({
    totals: {
      steps: 1,
      kinds: {thinking: 0, tool: 0, say: 0, answer: 1, instruction: 0, delivery: 0},
      decisions: 0,
      marks: {failed: 0, degraded: 0, repeated: 0, redirect: 0},
      tool_time_ms: 0,
    },
    findings: [],
    findings_total: 0,
  });
  const element = await mount(await readOverview(wire));
  const quality = element.querySelector<HTMLElement>('[data-quality]')!;
  expect(quality.getAttribute('data-quality-state')).to.equal('clean');
  expect(text(quality.querySelector<HTMLElement>('[data-quality-line]')!)).to.equal(
    'The Run completed with nothing to flag.',
  );
  expect(quality.querySelector('[data-quality-toggle]'), 'nothing to disclose on a clean Run').to.equal(null);
  expect(quality.querySelector('[data-quality-head] button'), 'the head is alone').to.equal(null);
  expect(quality.querySelector('#run-overview-quality-details')).to.equal(null);
  expect(element.querySelector('[data-attention-empty]')).to.equal(null);
  expect(text(quality)).to.not.contain('Failed');
});

it('says when it is still reading, and when the read failed', async () => {
  const loading = document.createElement('dl-run-overview') as DlRunOverview;
  loading.loading = true;
  document.body.append(loading);
  await loading.updateComplete;
  expect(text(loading.querySelector<HTMLElement>('[data-overview-loading]')!)).to.equal(
    'Reading the Run overview…',
  );

  const failed = document.createElement('dl-run-overview') as DlRunOverview;
  failed.failed = true;
  document.body.append(failed);
  await failed.updateComplete;
  expect(text(failed.querySelector<HTMLElement>('[data-overview-failed]')!)).to.equal(
    'The Run overview could not be read.',
  );
  expect(failed.querySelector('[data-composition]')).to.equal(null);
});

it('draws its copy in the language showing, placeholders included', async () => {
  await setLanguagePreference('zh');
  try {
    const element = await mount(await readOverview(overviewWire({
      agents: [
        {
          agent: 'main', label: 'main', order: 0, steps: 3, tool_calls: 2,
          marks: {failed: 1, degraded: 0, repeated: 0, redirect: 0}, tool_time_ms: 0,
          first_at: '2026-10-05T12:00:00Z', last_at: '2026-10-05T12:00:30Z',
        },
        {
          agent: 'child-1', label: '读招股书', order: 1, steps: 2, tool_calls: 1,
          marks: {failed: 0, degraded: 0, repeated: 0, redirect: 0}, tool_time_ms: 0,
          first_at: '2026-10-05T12:00:10Z', last_at: '2026-10-05T12:00:50Z',
        },
      ],
    })), {
      run: run(),
      lead: listed('main', {status: 'succeeded'}),
      entries: [listed('child-1', {status: 'succeeded'})],
    });
    const words = text(element);
    expect(words).to.contain('构成');
    expect(words).to.contain('交付');
    expect(words).to.contain('代理');
    expect(words).to.contain('主代理');
    expect(words).to.contain('运行质量');
    // The legend's counts are templates: the number and the word must both be there.
    expect(words).to.contain('3 步');
    expect(words).to.contain('失败 1');
    // A child is named by its objective, and its row's steps are its own.
    expect(words).to.contain('读招股书');
    // The calm line names no mark: the words above are the folded details it controls.
    const line = text(element.querySelector<HTMLElement>('[data-quality-line]')!);
    expect(line).to.equal('运行期间记录了 1 条细节（重试、降级或交付提示）。');
    const toggle = element.querySelector<HTMLButtonElement>('[data-quality-toggle]')!;
    expect(toggle.textContent!.trim()).to.equal('查看细节');
    expect(toggle.textContent!).to.not.contain('1');
    expect(element.querySelector('#run-overview-quality-details')!.getAttribute('aria-label'))
      .to.equal('运行细节');
  } finally {
    await setLanguagePreference('en');
  }
});

it('keeps a running Run\'s lane moving with its own one-second clock', async () => {
  const {ticks} = captureTicks();
  let tick = NOW;
  Date.now = () => tick;
  const wire = overviewWire({
    agents: [
      {
        agent: 'main', label: 'main', order: 0, steps: 3, tool_calls: 1,
        // The agent starts five minutes into its Run, so its lane is a share of a clock that grows.
        marks: {failed: 0, degraded: 0, repeated: 0, redirect: 0}, tool_time_ms: 0,
        first_at: '2026-10-05T12:05:00Z', last_at: '2026-10-05T12:06:00Z',
      },
    ],
    findings: [],
    findings_total: 0,
  });
  const running = await mount(await readOverview(wire), {
    run: run({status: 'running', finishedAt: null, startedAt: '2026-10-05T12:00:00Z'}),
    lead: listed('main', {status: 'running', startedAt: '2026-10-05T12:05:00Z'}),
  });
  await waitFor(() => running.querySelector('[data-lane-span]') !== null);
  expect(ticks.size, 'a running page owns one tick').to.equal(1);
  const lane = running.querySelector<HTMLElement>('[data-lane-span]')!;
  const before = `${lane.style.insetInlineStart}/${lane.style.width}`;
  const durationBefore = text(running.querySelector<HTMLElement>('[data-metric="duration"]')!);

  tick = NOW + 60_000;
  ticks.get(1)!();
  await running.updateComplete;
  const after = `${lane.style.insetInlineStart}/${lane.style.width}`;
  expect(after, 'the lane reaches further as the Run goes on').to.not.equal(before);
  expect(text(running.querySelector<HTMLElement>('[data-metric="duration"]')!))
    .to.not.equal(durationBefore);

  // A settled page draws no moving time, so the clock stops.
  document.body.replaceChildren();
  const settled = await mount(await readOverview(wire), {
    run: run(),
    lead: listed('main', {status: 'succeeded', startedAt: '2026-10-05T12:00:00Z', finishedAt: '2026-10-05T12:01:00Z'}),
  });
  await settled.updateComplete;
  expect(ticks.size, 'a settled page schedules no tick').to.equal(1);
});

it('draws an empty Run without inventing numbers', async () => {
  serve({});
  const empty = await sourceFor().source.overview!(new AbortController().signal);
  const element = await mount(empty, {run: run({status: 'running', finishedAt: null}), lead: listed('main')});
  expect(element.querySelectorAll('[data-bar-segment]').length).to.equal(0);
  expect(text(element.querySelector<HTMLElement>('[data-composition-empty]')!)).to.equal('No steps yet.');
  expect(agentRows(element).length, 'the Run still has its own agent').to.equal(1);
  expect(text(agentRows(element)[0]!)).to.not.contain('tokens');
});
