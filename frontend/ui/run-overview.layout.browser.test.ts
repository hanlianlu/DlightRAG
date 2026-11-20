// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The Run overview as the shipped stylesheets lay it out (the product-styles page of the runner):
 *  a dock-wide column that never pushes sideways, an objective that ends in an ellipsis, a bar whose
 *  parts are the Run's counts, finger-sized presses from 320px up, and no serious accessibility
 *  violations in either color mode. The Inspector mounts the component, so its class names and rules
 *  ship in the application's own stylesheets — these tests read what a reader actually gets. */

import {expect} from '@esm-bundle/chai';
import {resetMouse, sendMouse, setViewport} from '@web/test-runner-commands';
import type {
  AgentChildStatus,
  AgentRunStatus,
  AnswerPresentation,
  RunOverview,
} from '../api/conversations.ts';
import {getAnswerRunOverview} from '../api/conversations.ts';
import {pressArea, seriousViolations} from '../testing/axe.ts';
import {waitFor} from '../testing/dom.ts';
import {NOW, overview as overviewWire, serve} from '../testing/traces.ts';
import type {DlRunOverview, ListedChild} from './run-overview.ts';
import './run-overview.ts';

const originalViewport = {width: window.innerWidth, height: window.innerHeight};
const originalNow = Date.now;

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

/** One Answer as the Run published it; a test states the parts the delivery card reads. */
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
    artifacts: [artifact('summary')],
    artifactOutcome: {status: 'complete', issues: []},
    ...extra,
  } as unknown as AnswerPresentation;
}

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
  } = {},
): Promise<DlRunOverview> {
  const element = document.createElement('dl-run-overview') as DlRunOverview;
  element.overview = data;
  element.run = options.run === undefined ? run() : options.run;
  element.presentation = options.presentation ?? null;
  element.entries = options.entries ?? [];
  element.lead = options.lead === undefined ? listed('main') : options.lead;
  element.loading = false;
  element.failed = false;
  document.body.append(element);
  await element.updateComplete;
  return element;
}

const agentRows = (element: DlRunOverview): HTMLElement[] => [
  ...element.querySelectorAll<HTMLElement>('[data-agent-row]'),
];

const wideObjective = Array.from({length: 14}, (_, index) => `objective word ${index}`).join(' ');

/** The harness the panel is drawn in: a dock's own width, on the product's page. */
let harness: HTMLElement;

before(async () => {
  Date.now = () => NOW;
  await setViewport({width: 1280, height: 900});
  harness = document.createElement('div');
  harness.style.cssText = 'display:block;width:400px';
  document.body.append(harness);
});

after(async () => {
  Date.now = originalNow;
  harness.remove();
  await setViewport(originalViewport);
});

/** The mounted panel in the harness, after the stylesheet settles. */
async function drawn(wire: Record<string, unknown> = overviewWire(), width = 400): Promise<DlRunOverview> {
  if (!harness.isConnected) document.body.append(harness);
  harness.style.width = `${width}px`;
  const element = await mount(await readOverview(wire), {
    run: run(),
    lead: listed('main'),
    entries: [listed('child-1', {status: 'running', usage: {total_tokens: 1234}})],
    presentation: answer(),
  });
  harness.replaceChildren(element);
  await waitFor(() => element.querySelector<HTMLElement>('[data-lane-span]') !== null);
  return element;
}

afterEach(() => {
  harness.replaceChildren();
});

it('keeps the column inside a dock-wide panel, with the Run\'s parts proportional', async () => {
  const element = await drawn(overviewWire({
    totals: {
      steps: 9,
      kinds: {thinking: 0, tool: 6, say: 0, answer: 0, instruction: 0, delivery: 3},
      decisions: 1,
      marks: {failed: 1, degraded: 0, repeated: 0, redirect: 0},
      tool_time_ms: 0,
    },
  }));
  const box = element.getBoundingClientRect();
  expect(box.width).to.be.at.most(400.5);
  expect(element.scrollWidth, 'nothing pushes the column sideways').to.be.at.most(element.clientWidth + 1);
  const parts = [...element.querySelectorAll<HTMLElement>('[data-bar-segment]')];
  const bars = parts.map((part) => part.getBoundingClientRect().width);
  expect(parts.length).to.equal(2);
  // Six tool calls against three delivery checks: two to one, within a pixel of rounding.
  expect(bars[0]! / bars[1]!).to.be.closeTo(2, 0.05);
  expect(bars[0]!).to.be.greaterThan(100);
});

it('clamps a child\'s objective to one line and keeps the whole text in its title', async () => {
  const element = await drawn(overviewWire({
    agents: [
      {
        agent: 'main', label: 'main', order: 0, steps: 3, tool_calls: 2,
        marks: {failed: 1, degraded: 0, repeated: 0, redirect: 0}, tool_time_ms: 0,
        first_at: '2026-10-05T12:00:00Z', last_at: '2026-10-05T12:00:30Z',
      },
      {
        agent: 'child-1', label: wideObjective, order: 1, steps: 3, tool_calls: 1,
        marks: {failed: 0, degraded: 0, repeated: 0, redirect: 0}, tool_time_ms: 0,
        first_at: '2026-10-05T12:00:10Z', last_at: '2026-10-05T12:00:50Z',
      },
    ],
    findings: [],
    findings_total: 0,
  }));
  const child = agentRows(element).find((row) => row.dataset.agent === 'child-1')!;
  const name = child.querySelector<HTMLElement>('[class*="rowName"]')!;
  expect(getComputedStyle(name).overflow).to.equal('hidden');
  expect(getComputedStyle(name).textOverflow).to.equal('ellipsis');
  expect(getComputedStyle(name).whiteSpace).to.equal('nowrap');
  expect(name.scrollWidth).to.be.greaterThan(name.clientWidth);
  expect(name.getAttribute('title')).to.equal(wideObjective);
  // One line of the name, not a row grown by the words.
  const line = Number.parseFloat(getComputedStyle(name).lineHeight);
  expect(name.getBoundingClientRect().height).to.be.closeTo(line, 0.5);
});

it('keeps every press a finger can find, and the numbers readable, from 320px up', async () => {
  const busy = overviewWire({
    agents: [
      {
        agent: 'main', label: 'main', order: 0, steps: 5, tool_calls: 2,
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
  for (const width of [400, 320]) {
    const element = await drawn(busy, width);
    element.querySelector<HTMLElement>('[data-dimension="agent"]')!.click();
    await element.updateComplete;
    // The panel's own controls, as the default surface draws them: the two dimension toggles, the
    // agent legend buttons, the ledger rows, and the quality disclosure. (The delivery record's
    // disclosure is the Answer's own control, and is tested where it lives.)
    const visible = (selector: string): HTMLElement[] => [...element.querySelectorAll<HTMLElement>(selector)]
      .filter((control) => control.getClientRects().length > 0);
    const controls = visible('button[data-dimension], [data-legend-agent], [data-agent-row], [data-quality-toggle]');
    expect(controls.length).to.be.greaterThan(6);
    for (const control of controls) {
      const area = pressArea(control);
      const name = control.dataset.dimension ?? control.dataset.agent ?? control.dataset.sequence ?? 'quality';
      expect(area.height, `${width}px: ${name}`).to.be.at.least(23.5);
    }
    // The finding rows are behind the quality disclosure, and are a finger's target there too.
    element.querySelector<HTMLButtonElement>('[data-quality-toggle]')!.click();
    await element.updateComplete;
    const findings = visible('[data-finding]');
    expect(findings.length).to.equal(2);
    for (const finding of findings) {
      expect(pressArea(finding).height, `${width}px: finding ${finding.dataset.sequence}`).to.be.at.least(23.5);
    }
    expect(element.scrollWidth, `${width}px: no sideways scroll`).to.be.at.most(element.clientWidth + 1);
    // The metrics line wraps instead of running out of the row.
    for (const metrics of element.querySelectorAll<HTMLElement>('[data-metrics]')) {
      expect(metrics.getBoundingClientRect().right).to.be.at.most(element.getBoundingClientRect().right + 1);
    }
    harness.replaceChildren();
  }
});

/** One Run with enough findings to overflow the quality region's bound. */
function manyFindings(count: number): Record<string, unknown> {
  return overviewWire({
    totals: {
      steps: count * 2,
      kinds: {thinking: 0, tool: count, say: 0, answer: 1, instruction: 0, delivery: 0},
      decisions: count,
      marks: {failed: count, degraded: 0, repeated: 0, redirect: 0},
      tool_time_ms: 0,
    },
    agents: [{
      agent: 'main', label: 'main', order: 0, steps: count * 2, tool_calls: count,
      marks: {failed: count, degraded: 0, repeated: 0, redirect: 0}, tool_time_ms: 0,
      first_at: '2026-10-05T12:00:00Z', last_at: '2026-10-05T12:00:30Z',
    }],
    findings: Array.from({length: count}, (_, index) => ({
      agent: 'main', sequence: index + 2, kind: 'tool', at: '2026-10-05T12:00:20Z',
      name: 'read', subject: `northwind-${index}.pdf`, outcome: 'failed',
      marks: ['failed'], excerpt: `The file ${index} could not be read.`,
    })),
    findings_total: count,
  });
}

it('keeps the quality disclosure in the section head, over a bounded scroll region', async () => {
  const element = await drawn(manyFindings(30));
  const quality = element.querySelector<HTMLElement>('[data-quality]')!;
  const head = quality.querySelector<HTMLElement>('[data-quality-head]')!;
  const line = quality.querySelector<HTMLElement>('[data-quality-line]')!;
  const toggle = quality.querySelector<HTMLButtonElement>('[data-quality-toggle]')!;
  const details = quality.querySelector<HTMLElement>('#run-overview-quality-details')!;

  // Section chrome: the control is the head's trailing element, on the head's own line, not a stray
  // link under the calm sentence.
  expect(toggle.parentElement, 'the toggle sits in the section head').to.equal(head);
  expect(head.querySelector('#run-overview-quality')).to.not.equal(null);
  const headBox = head.getBoundingClientRect();
  const toggleBox = toggle.getBoundingClientRect();
  expect(toggleBox.top).to.be.at.least(headBox.top - 0.5);
  expect(toggleBox.bottom).to.be.at.most(headBox.bottom + 0.5);
  expect(headBox.right - toggleBox.right, 'trailing in the head row').to.be.at.most(1);
  expect(toggleBox.bottom, 'not a lone row under the calm line')
    .to.be.at.most(line.getBoundingClientRect().top + 0.5);
  expect(pressArea(toggle).height).to.be.at.least(23.5);
  const mark = toggle.querySelector<SVGElement>('svg')!;
  const shut = getComputedStyle(mark).transform;
  expect(getComputedStyle(mark).transitionProperty).to.contain('transform');

  // Folded: the section is just its head and its calm line.
  const collapsed = quality.getBoundingClientRect().height;
  expect(details.hidden).to.equal(true);

  toggle.click();
  await element.updateComplete;
  for (const animation of mark.getAnimations()) animation.finish();
  expect(details.hidden).to.equal(false);
  expect(toggle.getAttribute('aria-expanded')).to.equal('true');
  expect(getComputedStyle(mark).transform, 'the primitive turns its own chevron').to.not.equal(shut);
  expect(getComputedStyle(mark).transform).to.contain('-1');

  // The expanded body is self-contained: bounded, scrolling within itself, and labelled for a
  // keyboard reader who scrolls it.
  const style = getComputedStyle(details);
  expect(style.overflowY).to.equal('auto');
  expect(style.overscrollBehavior).to.equal('contain');
  const bound = Number.parseFloat(style.maxBlockSize);
  expect(bound, 'bounded by the region rule').to.be.at.most(22 * 16 + 0.5);
  expect(bound, 'and never past the viewport share').to.be.at.most(window.innerHeight * 0.45 + 0.5);
  expect(details.getAttribute('role')).to.equal('region');
  expect(details.getAttribute('tabindex')).to.equal('0');
  expect(details.getAttribute('aria-label')).to.equal('Run details');
  expect(details.scrollHeight, 'thirty findings scroll inside it').to.be.greaterThan(details.clientHeight + 50);
  const regionBox = details.getBoundingClientRect();
  expect(regionBox.height).to.be.at.most(bound + 0.5);
  // The section is head + line + the region, not the whole list: opening it stays bounded.
  const open = quality.getBoundingClientRect().height;
  expect(open).to.be.at.most(headBox.height + line.getBoundingClientRect().height + regionBox.height + 24);
  expect(open, 'the dossier does not grow past the region').to.be.lessThan(600);
  expect(open).to.be.greaterThan(collapsed);

  details.focus();
  expect(document.activeElement, 'the region is focusable').to.equal(details);
  details.scrollTop = 120;
  expect(details.scrollTop, 'the reader can scroll it from the keyboard').to.be.greaterThan(0);
  expect(await seriousViolations(element)).to.deep.equal([]);

  // Closing puts the section back to head and line.
  toggle.click();
  await element.updateComplete;
  expect(details.hidden).to.equal(true);
  expect(quality.getBoundingClientRect().height).to.be.closeTo(collapsed, 1);
});

it('makes the agent bar pressable with a ring, and marks the shown agent', async () => {
  const element = await drawn(overviewWire({
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
  }));
  element.selected = 'child-1';
  element.querySelector<HTMLElement>('[data-dimension="agent"]')!.click();
  await element.updateComplete;

  const part = element.querySelector<HTMLElement>('[data-bar-segment][data-key="child-1"]')!;
  expect(part.tagName).to.equal('BUTTON');
  expect(part.getAttribute('aria-current')).to.equal('true');
  // The strip is six pixels tall; the pointer target reaches above and below it.
  const box = part.getBoundingClientRect();
  const above = document.elementFromPoint(box.left + box.width / 2, box.top - 4);
  const below = document.elementFromPoint(box.left + box.width / 2, box.bottom + 4);
  expect(above === part || part.contains(above), 'the part answers above the strip').to.equal(true);
  expect(below === part || part.contains(below), 'and below it').to.equal(true);

  // A keyboard reader who reaches the part sees the product's focus ring.
  part.focus({focusVisible: true});
  expect(part.matches(':focus-visible')).to.equal(true);
  expect(getComputedStyle(part).outlineStyle).to.equal('solid');

  // The shown agent is stated for assistive tech — and for nobody else: a row that is always
  // current must not wear the selection token the conversations list uses for a mode.
  const currentRow = element.querySelector<HTMLElement>('[data-agent-row][data-agent="child-1"]')!;
  const otherRow = element.querySelector<HTMLElement>('[data-agent-row][data-agent="main"]')!;
  const legendCurrent = element.querySelector<HTMLElement>('[data-legend-agent="child-1"]')!;
  const legendOther = element.querySelector<HTMLElement>('[data-legend-agent="main"]')!;
  expect(currentRow.getAttribute('aria-current')).to.equal('true');
  expect(otherRow.hasAttribute('aria-current')).to.equal(false);
  expect(legendCurrent.getAttribute('aria-current')).to.equal('true');

  // No permanent fill on the current row or legend entry: the current and the other compute alike.
  expect(getComputedStyle(currentRow).backgroundColor, 'no permanent selected fill on the row')
    .to.equal(getComputedStyle(otherRow).backgroundColor);
  expect(getComputedStyle(legendCurrent).backgroundColor, 'no permanent selected fill on the legend')
    .to.equal(getComputedStyle(legendOther).backgroundColor);

  // The current bar part is not outlined either once focus has left it.
  part.blur();
  expect(getComputedStyle(part).outlineStyle, 'no permanent selected outline on the bar')
    .to.equal('none');
  expect(getComputedStyle(part).boxShadow).to.equal('none');

  // Hover and focus remain the only emphasis a reader gets: a pointer over the row still lights it,
  // and leaves it plain again when the pointer moves on.
  const rowBox = currentRow.getBoundingClientRect();
  await sendMouse({type: 'move', position: [Math.round(rowBox.left + 8), Math.round(rowBox.top + rowBox.height / 2)]});
  expect(getComputedStyle(currentRow).backgroundColor, 'hover still lights the row')
    .to.not.equal(getComputedStyle(otherRow).backgroundColor);
  await resetMouse();
  expect(getComputedStyle(currentRow).backgroundColor, 'and the row is plain again after it')
    .to.equal(getComputedStyle(otherRow).backgroundColor);
  part.focus({focusVisible: true});
  expect(getComputedStyle(part).outlineStyle, 'focus is still drawn').to.equal('solid');
  part.blur();
});

it('marks the selected dimension, and only the selected one', async () => {
  const element = await drawn();
  const agentToggle = element.querySelector<HTMLElement>('[data-dimension="agent"]')!;
  const kindToggle = element.querySelector<HTMLElement>('[data-dimension="kind"]')!;
  const before = getComputedStyle(kindToggle).backgroundColor;
  agentToggle.click();
  await element.updateComplete;
  expect(agentToggle.getAttribute('aria-pressed')).to.equal('true');
  expect(kindToggle.getAttribute('aria-pressed')).to.equal('false');
  // The accent moves with the choice: the pressed button wears it, the other gives it back.
  expect(getComputedStyle(kindToggle).backgroundColor).to.not.equal(before);
  expect(getComputedStyle(agentToggle).backgroundColor).to.equal(before);
});

it('has no serious accessibility violations in either color mode', async () => {
  const busy = overviewWire({
    agents: [
      {
        agent: 'main', label: 'main', order: 0, steps: 5, tool_calls: 2,
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
  harness.style.width = '400px';
  if (!harness.isConnected) document.body.append(harness);
  // A delivered Answer whose Artifacts carry no attestation record: the delivery card's own
  // shapes, without the Answer's separate disclosure control under test elsewhere.
  const panel = await mount(await readOverview(busy), {
    run: run(),
    lead: listed('main'),
    entries: [listed('child-1')],
    presentation: answer({artifacts: [artifact('summary')]}),
  });
  harness.replaceChildren(panel);
  await panel.updateComplete;
  expect(await seriousViolations(panel), 'collapsed').to.deep.equal([]);

  // The detail behind the disclosure is judged too: the triage list is what a reader can open.
  panel.querySelector<HTMLButtonElement>('[data-quality-toggle]')!.click();
  await panel.updateComplete;
  expect(panel.querySelector<HTMLElement>('#run-overview-quality-details')!.hidden).to.equal(false);
  expect(await seriousViolations(panel), 'expanded').to.deep.equal([]);
});
