// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The dock as the shipped stylesheets lay it out (the product-styles page of the runner): scrolling that follows a
 *  reader, a long Evidence list, and the hit target on a phone. */

import {expect} from '@esm-bundle/chai';
import {setViewport} from '@web/test-runner-commands';
import {activityPage, ago, NOW, observation, overview, question, roster, row, serve, sourceFor} from '../testing/traces.ts';
import {deferred, immediateFollowRefreshes, mount, openChild, restoreDockFixtures, rowFor, rows, session, settle, shown, stated, steps, title} from '../testing/traces-dock.ts';
import {pressArea} from '../testing/axe.ts';
import {waitFor} from '../testing/dom.ts';
import './inspector-traces.ts';
import type {DlInspectorTraces} from './inspector-traces.ts';

restoreDockFixtures();

describe('the page of a child', () => {

  const says = (count: number) => Array.from({length: count}, (_, index) => (
    {role: 'assistant', content: `Step ${index}`, tool_calls: []}
  ));
  const page = (dock: DlInspectorTraces) => session(dock).querySelector<HTMLElement>('[data-scroller]')!;
  const pane = (dock: DlInspectorTraces) => session(dock).querySelector<HTMLElement>('[data-activity-pane]')!;
  /** How far a box lies below what is on screen, in pixels. */
  const below = (box: HTMLElement) => box.scrollHeight - box.clientHeight - box.scrollTop;
  const frames = () => new Promise<void>((resolve) => {
    requestAnimationFrame(() => { requestAnimationFrame(() => { resolve(); }); });
  });
  /** The reader scrolls: the box is moved, and given the frames it takes to see it. */
  const scrollTo = async (box: HTMLElement, top: number) => {
    box.scrollTop = top;
    await frames();
  };

  it('opens its activity on the newest step, keeps it in view for a reader at the bottom, and lets go when they scroll up', async () => {
    immediateFollowRefreshes();
    let steps = 40;
    const running = row('a', 'running', {started_at: ago(5)});
    serve({
      page: () => roster([running]),
      observe: () => observation(running),
      transcript: (agent) => (agent === null ? undefined : activityPage(says(steps), {running: true})),
    });
    const dock = await mount(sourceFor().source, 420);
    await waitFor(() => rows(dock).length === 1);
    await openChild(dock, 'a');
    await waitFor(() => shown(dock).includes('Step 39'));
    const box = pane(dock);
    expect(box.scrollHeight, 'the activity is taller than its pane').to.be.greaterThan(box.clientHeight + 100);
    expect(below(box), 'it opens on the newest step').to.be.lessThan(1);
    const arrive = async (total: number) => {
      steps = total;
      stated(dock, 'run-1');
      await waitFor(() => shown(dock).includes(`Step ${total - 1}`));
    };

    await arrive(44);
    expect(below(box)).to.be.lessThan(1);
    await arrive(48);
    expect(below(box)).to.be.lessThan(1);

    // Once the reader scrolls up the pane stays where they are.
    await scrollTo(box, 100);
    await arrive(52);
    expect(box.scrollTop).to.equal(100);
    expect(below(box)).to.be.greaterThan(100);
  });

  it('opens each child on its title, even from the bottom of another', async () => {
    immediateFollowRefreshes();
    const loadingB = deferred<Response>();
    const running = (id: string) => row(id, 'running', {started_at: ago(5)});
    serve({
      page: () => roster([running('a'), running('b')]),
      observe: (id) => (id === 'b' ? loadingB.promise : observation(running(id))),
      transcript: (agent) => (agent === null ? undefined : activityPage(says(40), {running: true})),
    });
    const dock = await mount(sourceFor().source, 800);
    // A dock shorter than the page, so the page itself scrolls.
    dock.style.height = '360px';
    await waitFor(() => rows(dock).length === 2);
    rowFor(dock, 'a').click();
    await waitFor(() => title(dock) === 'objective a' && shown(dock).includes('Step 39'));
    const box = page(dock);
    await scrollTo(box, box.scrollHeight);
    expect(box.scrollTop, 'the page is scrolled down').to.be.greaterThan(0);

    // The next child is still loading, so its page is short, when the scroll back to the top is seen.
    rowFor(dock, 'b').click();
    await waitFor(() => title(dock) === 'objective b');
    await scrollTo(box, 0);
    loadingB.resolve(observation(running('b')));
    await waitFor(() => shown(dock).includes('Step 39'));
    expect(page(dock) === box, 'the same page is on show').to.equal(true);
    expect(box.scrollTop, 'the next child opens on its title').to.equal(0);
    stated(dock, 'run-1');
    await settle(60);
    expect(box.scrollTop, 'and nothing carries the earlier child\'s place over').to.equal(0);
  });

  it('lets go of a reader who opens a step at the bottom, which grows the pane without scrolling it', async () => {
    immediateFollowRefreshes();
    let steps = 30;
    const running = row('a', 'running', {started_at: ago(5)});
    const transcript = () => [
      ...says(steps),
      {role: 'assistant', content: '', tool_calls: [{id: 't1', name: 'read'}]},
      {role: 'tool', content: Array.from({length: 40}, (_, line) => `line ${line}`).join('\n'), tool_call_id: 't1', name: 'read', is_error: false},
    ];
    serve({
      page: () => roster([running]),
      observe: () => observation(running),
      transcript: (agent) => (agent === null ? undefined : activityPage(transcript(), {running: true})),
    });
    const dock = await mount(sourceFor().source, 420);
    await waitFor(() => rows(dock).length === 1);
    await openChild(dock, 'a');
    await waitFor(() => shown(dock).includes('Reading a document'));
    const box = pane(dock);

    // The reader opens the last step: its whole result lengthens the pane below where they stand.
    session(dock).querySelector<HTMLElement>('summary:has(+ pre)')!.click();
    await waitFor(() => below(box) > 100);
    const stood = box.scrollTop;
    steps = 34;
    stated(dock, 'run-1');
    await waitFor(() => shown(dock).includes('Step 33'));

    expect(box.scrollTop, 'a reader who is no longer at the bottom is not pulled back to it').to.equal(stood);
  });

  it('offers the whole objective when the dock is narrowed until the title is cut off', async () => {
    // Long enough to overflow the three-line clamp at 300px in whatever sans-serif the machine falls back
    // to (the system font on a Mac, a narrower one on Linux CI, where a shorter objective fitted in three
    // lines), and short enough to fit it at 620px.
    const objective = 'Make the strongest case for terminating the Northwind supply agreement early, citing the notice clause, its penalty schedule and the supplier\'s three missed deliveries';
    const settled = row('a', 'succeeded', {objective, started_at: ago(9), finished_at: ago(1)});
    serve({page: () => roster([settled]), observe: () => observation(settled)});
    // A narrow dock either way, so nothing but the width changes: a settled child has no clock to redraw it.
    const dock = await mount(sourceFor().source, 620);
    await waitFor(() => rows(dock).length === 1);
    await openChild(dock, 'a');
    const more = session(dock).querySelector<HTMLElement>('[data-more]')!;
    expect(more.hidden, 'the objective fits at first').to.equal(true);

    // The observer reports in the next frame's rendering step, so frames are what is waited for.
    dock.style.width = '300px';
    await frames();
    expect(more.hidden, 'the title is cut off at 300px').to.equal(false);
    expect(more.textContent!.trim()).to.equal('Show full objective');
    // The control is the design system's disclosure, not a bespoke link: its mark, face and target.
    expect(more.classList.contains('dl-text-action')).to.equal(true);
    expect(more.classList.contains('dl-text-action--disclosure')).to.equal(true);
    expect(more.querySelector('svg'), 'the primitive draws its own chevron').to.not.equal(null);
    expect(pressArea(more).height).to.be.at.least(23.5);
    expect(more.getAttribute('aria-controls')).to.equal('agent-session-title');
    const heading = session(dock).querySelector<HTMLElement>('[data-title]')!;
    expect(heading.className, 'the objective is clamped until asked').to.contain('clamped');
    more.click();
    await session(dock).updateComplete;
    expect(more.getAttribute('aria-expanded')).to.equal('true');
    expect(heading.className, 'opening it shows the whole objective').to.not.contain('clamped');
    expect(more.textContent!.trim()).to.equal('Show less');
    more.click();
    await session(dock).updateComplete;
    expect(more.getAttribute('aria-expanded')).to.equal('false');
    expect(heading.className).to.contain('clamped');

    dock.style.width = '620px';
    await frames();
    expect(more.hidden, 'and whole again at 620px').to.equal(true);
  });

  it('does not follow a child that has settled', async () => {
    immediateFollowRefreshes();
    let steps = 40;
    let summary = 'First summary.';
    const settled = () => row('a', 'succeeded', {started_at: ago(9), finished_at: ago(1), summary});
    serve({
      page: () => roster([settled()]),
      observe: () => observation(settled()),
      transcript: (agent) => (agent === null ? undefined : activityPage(says(steps), {running: false})),
    });
    const dock = await mount(sourceFor().source, 420);
    await waitFor(() => rows(dock).length === 1);
    await openChild(dock, 'a');
    await waitFor(() => shown(dock).includes('Agent timeline · 40 steps'));
    // Its activity is folded; the reader opens it, and it opens on the newest step.
    session(dock).querySelector<HTMLDetailsElement>('details:has([data-activity-pane])')!.open = true;
    const box = pane(dock);
    await waitFor(() => box.scrollHeight > box.clientHeight + 100 && below(box) < 1);
    const stood = box.scrollTop;

    steps = 46;
    summary = 'A later summary.';
    stated(dock, 'run-1');
    await waitFor(() => shown(dock).includes('Step 45'));
    expect(box.scrollTop).to.equal(stood);
    expect(below(box)).to.be.greaterThan(50);
  });

  it('keeps the step a reader is at where it is while earlier steps come in above it', async () => {
    const running = row('a', 'running', {started_at: ago(5)});
    serve({
      page: () => roster([running]),
      observe: () => observation(running),
      transcript: (agent, before) => {
        if (agent !== 'a') return undefined;
        return before === null
          ? activityPage(steps(21, 40), {nextBefore: 21, running: true})
          : activityPage(steps(1, 20), {running: true});
      },
    });
    const dock = await mount(sourceFor().source, 420);
    await waitFor(() => rows(dock).length === 1);
    await openChild(dock, 'a');
    await waitFor(() => shown(dock).includes('Step 40.'));
    const box = pane(dock);
    const first = [...box.querySelectorAll('li')].find((item) => item.textContent!.includes('Step 21.'))!;
    await scrollTo(box, 60);
    const at = () => first.getBoundingClientRect().top - box.getBoundingClientRect().top;
    const before = at();

    session(dock).querySelector<HTMLButtonElement>('[data-load-older="activity"]')!.click();
    await waitFor(() => shown(dock).includes('Step 1.Step 2.'));
    await frames();

    expect(at()).to.be.closeTo(before, 1);
  });

  it('scrolls a long Evidence list in a pane of its own', async () => {
    const handles = Array.from({length: 40}, (_, index) => `[${index + 1}] source-${index + 1}.pdf`);
    const settled = row('a', 'succeeded', {started_at: ago(9), finished_at: ago(1), summary: 'Done.', result_handles: handles});
    serve({page: () => roster([settled]), observe: () => observation(settled)});
    const dock = await mount(sourceFor().source, 420);
    await waitFor(() => rows(dock).length === 1);
    await openChild(dock, 'a');
    await waitFor(() => shown(dock).includes('Evidence · 40'));
    session(dock).querySelector<HTMLDetailsElement>('details')!.open = true;
    const list = session(dock).querySelector<HTMLElement>('ul[aria-label="Evidence · 40"]')!;
    await waitFor(() => list.scrollHeight > list.clientHeight + 100);

    await scrollTo(list, 50);
    expect(list.scrollTop, 'the list scrolls within the page').to.equal(50);
  });
});

// ── On a phone ──

describe('on a phone', () => {
  const originalViewport = {width: window.innerWidth, height: window.innerHeight};
  before(async () => {
    await setViewport({width: 390, height: 844});
  });
  after(async () => {
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
    // The one switcher line leads what a reader can press.
    expect(list.length).to.be.at.least(1);
    expect(list[0]!.name).to.equal('Main agent Running');
    await openChild(dock, 'a');
    await waitFor(() => shown(dock).includes('Looking.') && shown(dock).includes('Instructions received'));
    const page = pressable();
    // The page keeps the switcher's line plus the Agent timeline and Instructions received folds.
    expect(page.length).to.be.greaterThan(2);
    const short = [...list, ...page].filter((item) => item.height < target - 0.5).map((item) => `${item.name}: ${item.height}`);
    expect(short, 'what is pressable and shorter than the hit target').to.deep.equal([]);
  });
});

// ── The Run overview panel, in the dock's two layouts ──

/** One Run's whole-Run facts, so the left column has a dossier to lay out. */
function runOverview(): Record<string, unknown> {
  return overview({
    totals: {
      steps: 6,
      kinds: {thinking: 0, tool: 3, say: 1, answer: 1, instruction: 0, delivery: 1},
      decisions: 1,
      marks: {failed: 1, degraded: 0, repeated: 0, redirect: 0},
      tool_time_ms: 2400,
    },
    agents: [
      {
        agent: 'main', label: 'main', order: 0, steps: 3, tool_calls: 1,
        marks: {failed: 1, degraded: 0, repeated: 0, redirect: 0},
        tool_time_ms: 1200, first_at: '2026-10-05T12:00:00Z', last_at: '2026-10-05T12:00:30Z',
      },
      {
        agent: 'a', label: 'objective a', order: 1, steps: 3, tool_calls: 2,
        marks: {failed: 0, degraded: 0, repeated: 0, redirect: 0},
        tool_time_ms: 1200, first_at: '2026-10-05T12:00:10Z', last_at: '2026-10-05T12:00:50Z',
      },
    ],
    findings: [{
      agent: 'main', sequence: 1, kind: 'tool', at: '2026-10-05T12:00:20Z', name: 'read',
      subject: 'northwind-msa.pdf', outcome: 'failed', marks: ['failed'], excerpt: 'No file.',
    }],
    findings_total: 1,
  });
}

describe('the Run overview panel', () => {
  it('is a 44px band over a scroller in the dock\'s left column, not a fold', async () => {
    serve({
      page: () => roster([row('a', 'succeeded')], null, 'succeeded'),
      observe: () => observation(row('a', 'succeeded')),
      overview: () => Response.json(runOverview()),
    });
    const dock = await mount(sourceFor().source, 900);
    await waitFor(() => dock.querySelector('dl-run-overview [data-run-overview]') !== null);

    const band = dock.querySelector<HTMLElement>('[data-overview-band]')!;
    expect(Math.round(band.getBoundingClientRect().height), 'the band is the switcher\'s own 44px')
      .to.equal(44);
    const column = dock.querySelector<HTMLElement>('[data-overview]')!.parentElement!;
    const pane = dock.querySelector('dl-agent-switcher')!.parentElement!;
    expect(column.getBoundingClientRect().width, 'the Run panel is the narrower column')
      .to.be.lessThan(pane.getBoundingClientRect().width);
    expect(column.getBoundingClientRect().right, 'the two columns meet without a gap')
      .to.be.closeTo(pane.getBoundingClientRect().left, 1);
    // The dossier scrolls in the column, so a long Run never pushes the page beside it down.
    const body = dock.querySelector<HTMLElement>('[data-overview-body]')!;
    expect(getComputedStyle(body).overflowY).to.equal('auto');
    const outside = [...dock.querySelectorAll<HTMLElement>('dl-run-overview *')]
      .filter((element) => element.getBoundingClientRect().right > column.getBoundingClientRect().right + 1)
      .map((element) => element.className);
    expect(outside, 'nothing in the dossier pushes past its column').to.deep.equal([]);
  });

  it('folds to one 44px summary line above the switcher in a narrow dock', async () => {
    serve({
      page: () => roster([row('a', 'succeeded')], null, 'succeeded'),
      observe: () => observation(row('a', 'succeeded')),
      overview: () => Response.json(runOverview()),
    });
    const dock = await mount(sourceFor().source, 420);
    await waitFor(() => dock.querySelector('[data-overview]') !== null);

    const fold = dock.querySelector<HTMLDetailsElement>('[data-overview]')!;
    expect(fold.tagName).to.equal('DETAILS');
    const line = fold.querySelector<HTMLElement>('summary')!;
    expect(Math.round(line.getBoundingClientRect().height)).to.equal(44);
    const switcher = dock.querySelector<HTMLElement>('[data-switcher-current]')!;
    expect(line.getBoundingClientRect().bottom).to.be.at.most(switcher.getBoundingClientRect().top + 0.5);
  });
});
