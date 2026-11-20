// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The Agent traces dock's roster: the children it lists, how it pages them, how it follows a Run,
 *  and the agent page mounted beside a roster that is still loading. */

import {expect} from '@esm-bundle/chai';
import {activityPage, ago, answeredTurn, NOW, observation, overview, roster, row, serve, sourceFor} from '../testing/traces.ts';
import {button, deferred, immediateFollowRefreshes, mount, openChild, restoreDockFixtures, rowFor, rows, session, settle, shown, stated, title} from '../testing/traces-dock.ts';
import {waitFor} from '../testing/dom.ts';
import './inspector-traces.ts';

restoreDockFixtures();

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
    'objective a Running · 2m 14s Asks parent',
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
  stated(dock, 'run-1');
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

  expect(dock.querySelector('[role="alert"]')!.textContent!.trim()).to.equal('Agents could not be loaded.');
  expect(rows(dock)).to.have.length(0);

  button(dock, 'Retry')!.click();
  await waitFor(() => rows(dock).length === 1);
  expect(dock.querySelector('[role="alert"]')).to.equal(null);
});

it('shows only the main agent for a Run that started no child, with no children to choose between', async () => {
  serve({page: () => roster([], null, 'succeeded')});
  const dock = await mount(sourceFor('run-1', answeredTurn()).source, 800);
  await waitFor(() => shown(dock).includes('No activity yet.'));
  const empty = [...dock.querySelectorAll<HTMLElement>('.dl-note')]
    .find((line) => line.textContent?.includes('No activity yet.'));
  expect(empty, 'the empty activity line is the design system\'s quiet line').to.not.equal(undefined);

  expect(title(dock)).to.equal('What changed?');
  expect(shown(dock)).to.contain('Main agent');
  expect(shown(dock)).to.contain('Done');
  expect(rows(dock), 'nothing to choose between').to.deep.equal([]);
  expect(dock.querySelector('[data-switcher-current]')!.textContent).to.contain('Main agent');
  expect(dock.querySelector('textarea, form, [data-draft]'), 'and nothing to steer here').to.equal(null);
});

it('mounts the agent page before the roster answers, so its transcript reads in parallel', async () => {
  const pending = deferred<Response>();
  const served = serve({
    page: () => pending.promise,
    transcript: () => activityPage([{role: 'assistant', content: 'the main transcript'}], {running: true}),
  });
  const dock = await mount(sourceFor().source, 800);

  // The roster is still on the wire: the page is up anyway and the status near the switcher says so.
  expect(dock.querySelector('dl-agent-session'), 'the page does not wait for the roster').to.not.equal(null);
  expect(dock.querySelector('[data-roster-loading]')).to.not.equal(null);
  expect(rows(dock)).to.have.length(0);
  // Its transcript is read while the roster is still loading, not after it.
  await waitFor(() => served.some((request) => request.path.endsWith('/transcript')));
  await waitFor(() => (dock.textContent ?? '').includes('the main transcript'));

  pending.resolve(roster([row('a')]));
  await waitFor(() => rows(dock).length === 1);
  expect(dock.querySelector('[data-roster-loading]'), 'the loading line goes with the roster').to.equal(null);
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

  for (let activity = 0; activity < 30; activity += 1) stated(dock, 'run-1');
  stated(dock, 'another-run');
  await settle(20);

  expect(pages).to.equal(1, 'activity right after opening waits for the window');
  const trailing = [...timers.values()].filter(({delay}) => delay > 0);
  expect(trailing).to.have.length(1, 'one trailing refresh carries it all');
  expect(trailing[0]!.delay).to.be.at.most(1000);
  trailing[0]!.handler();
  await waitFor(() => pages === 2);

  // That refresh opened a window of its own.
  stated(dock, 'run-1');
  expect([...timers.values()].filter(({delay}) => delay > 0)).to.have.length(2);
});

it('re-reads the Run\'s own pane when a Run with no children moves', async () => {
  immediateFollowRefreshes();
  const served = serve({
    page: () => roster([], null, 'running'),
    transcript: () => activityPage([{role: 'assistant', content: 'a step'}], {running: true}),
  });
  const dock = await mount(sourceFor().source, 800);
  const reads = () => served.filter((one) => one.path.endsWith('/transcript')).length;
  const pages = () => served.filter((one) => one.path.endsWith('/children')).length;
  await waitFor(() => reads() > 0);
  const openedPages = pages();

  // The Run moved. Its roster page is the same empty list it always was, and the pane the reader
  // is looking at is the Run's own agent: it still has to follow.
  stated(dock, 'run-1');
  await waitFor(() => pages() > openedPages);
  await waitFor(() => reads() > 1);
});

it('re-reads the Run\'s own pane after the first page arrived empty', async () => {
  immediateFollowRefreshes();
  let reads = 0;
  serve({
    page: () => roster([], null, 'running'),
    transcript: () => {
      reads += 1;
      return activityPage(reads === 1 ? [] : [{role: 'assistant', content: 'a step'}], {
        running: true,
      });
    },
  });
  const dock = await mount(sourceFor().source, 800);
  await waitFor(() => reads > 0);
  await waitFor(() => shown(dock).includes('No activity yet.'));

  // The first page landed before the Run wrote anything; the pane holds that empty page. A later
  // event must make it look again rather than leave the reader with the empty first read.
  stated(dock, 'run-1');
  await waitFor(() => reads > 1);
  await waitFor(() => shown(dock).includes('a step'));
});

it('keeps a live pane reading itself when the dock thinks the Run has settled', async () => {
  let reads = 0;
  serve({
    // The roster says settled, so the dock never pokes the page; the transcript page itself says
    // the agent is working. The pane must believe its own page rather than freeze on the first read.
    page: () => roster([], null, 'succeeded'),
    transcript: () => {
      reads += 1;
      return activityPage(reads === 1 ? [] : [{role: 'assistant', content: 'a step'}], {
        running: true,
      });
    },
  });
  const dock = await mount(sourceFor().source, 800);
  await waitFor(() => reads === 1);
  await waitFor(() => shown(dock).includes('No activity yet.'));
  await waitFor(() => shown(dock).includes('a step'), 3000);
  expect(reads, 'the pane read itself forward').to.be.greaterThan(1);
});

it('takes a Run\'s tokens as liveness, without reading a page for them', async () => {
  const served = serve({page: () => roster([], null, 'running')});
  const dock = await mount(sourceFor().source, 800);
  await waitFor(() => shown(dock).includes('Main agent'));
  const before = served.length;

  dock.applyEvents('run-1', [{kind: 'token', text: 'Half an answer'}]);
  await dock.updateComplete;

  expect(shown(dock), 'the Run\'s own agent is on screen').to.contain('Main agent');
  expect(served.length, 'a token is the chat\'s words, not a trace read').to.equal(before);
});

it('keeps a running child on show current, whatever the batch was about', async () => {
  immediateFollowRefreshes();
  const served = serve({page: () => roster([row('a', 'running')])});
  const dock = await mount(sourceFor().source);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');
  const reads = () => served.filter((one) => one.path.endsWith('/transcript')).length;
  const opened = reads();

  // The Run's log states nothing about a child's own work, so a token — any batch — keeps the
  // pane the reader is watching as current as the interval allows.
  dock.applyEvents('run-1', [{kind: 'token', text: 'still working'}]);
  await waitFor(() => reads() > opened);
});

it('appends what landed since the last message the pane holds', async () => {
  immediateFollowRefreshes();
  let landed = 0;
  const served = serve({
    page: () => roster([row('a', 'running')]),
    transcript: (_agent, _before, since) => {
      // A delta answers with what the reader does not hold yet, exactly as the route does.
      if (since !== null) {
        landed += 1;
        return activityPage([{sequence: landed, role: 'assistant', content: `step ${landed}`}]);
      }
      landed = 1;
      return activityPage([{sequence: 1, role: 'assistant', content: 'step 1'}]);
    },
  });
  const dock = await mount(sourceFor().source);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');
  await waitFor(() => shown(dock).includes('step 1'));

  // The Run stated something and a second step landed: the pane reads forward from what it holds
  // and keeps what it had — a delta is an append, not the page that replaces the history.
  stated(dock, 'run-1');

  await waitFor(() => shown(dock).includes('step 2'));
  expect(shown(dock), 'the history stays beside what landed').to.include('step 1');
  expect(served.some((one) => one.search.includes('since=')), 'read forward').to.equal(true);
});

it('reads the pages again when an event states something', async () => {
  immediateFollowRefreshes();
  const served = serve({page: () => roster([], null, 'running')});
  const dock = await mount(sourceFor().source, 800);
  await waitFor(() => served.length > 0);
  const before = served.length;

  dock.applyEvents('run-1', [{kind: 'tool', eventType: 'tool_start', payload: {}}]);
  await waitFor(() => served.length > before);
});

it('re-reads the pane on show when the Run moves again', async () => {
  immediateFollowRefreshes();
  const served = serve({
    page: () => roster([row('a', 'running')]),
    transcript: () => activityPage([{role: 'assistant', content: 'a step'}], {running: true}),
  });
  const dock = await mount(sourceFor().source);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');
  const reads = () => served.filter((one) => one.path.endsWith('/transcript')).length;
  const pages = () => served.filter((one) => one.path.endsWith('/children')).length;
  const opened = reads();
  expect(opened, 'opening a child reads its transcript').to.be.greaterThan(0);

  const shown = session(dock);
  const pane = shown.querySelector('dl-activity')!;
  expect(pane, 'the pane on show is the transcript').to.not.equal(null);
  expect(pane.loaded, 'the pane holds its first page').to.equal(true);
  const openedPages = pages();

  // The Run moved: the dock refreshes its roster, and what the reader is looking at follows it.
  stated(dock, 'run-1');
  await waitFor(() => pages() > openedPages);
  expect(shown.entry?.status, 'the row the pane draws says it is still running').to.equal('running');
  await waitFor(() => reads() > opened);
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

  stated(dock, 'run-1');
  const closing = [...timers.keys()].at(-1)!;
  dock.source = null;
  await dock.updateComplete;
  expect(cleared).to.include(closing);

  dock.source = sourceFor().source;
  await waitFor(() => pages === 2);
  stated(dock, 'run-1');
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
  stated(dock, 'run-1');
  await settle(50);

  expect(pages).to.equal(1);
});

// ── One pane, one switcher ──

/** One Run's whole-Run facts with two agents and one finding: what the Run panel draws. */
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
      agent: 'main', sequence: 2, kind: 'tool', at: '2026-10-05T12:00:20Z', name: 'read',
      subject: 'northwind-msa.pdf', outcome: 'failed', marks: ['failed'], excerpt: 'No file.',
    }],
    findings_total: 1,
  });
}

it('folds the Run overview into one summary line above the page in a narrow dock', async () => {
  serve({
    page: () => roster([row('a', 'running'), row('b', 'succeeded')]),
    observe: (id) => observation(row(id, id === 'a' ? 'running' : 'succeeded')),
    overview: () => Response.json(runOverview()),
  });
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 2);
  await waitFor(() => dock.querySelector('[data-overview]') !== null);

  // The Run panel is one summary line, shut, with the page under the switcher as the only pane.
  const panel = dock.querySelector<HTMLElement>('[data-overview]')!;
  expect(panel.tagName).to.equal('DETAILS');
  expect(panel.hasAttribute('open')).to.equal(false);
  expect(dock.querySelector('[data-overview-title]')!.textContent).to.contain('Run overview');
  const summary = dock.querySelector<HTMLElement>('[data-overview-summary]')!;
  expect(summary.textContent).to.contain('2 agents');
  expect(summary.textContent).to.contain('6 steps');
  expect(dock.querySelector('[data-switcher-current]')!.textContent).to.contain('Main agent');
  expect(dock.querySelector('[data-switcher-current]')!.textContent).to.contain('Running');

  rowFor(dock, 'b').click();
  await waitFor(() => title(dock) === 'objective b');
  expect(rowFor(dock, 'b').getAttribute('aria-current')).to.equal('true');
  expect(dock.querySelector('[data-switcher-current]')!.textContent).to.contain('Child 2');

  // The reader opens the summary: the Run's dossier shows above the page, which keeps its place.
  const fold = dock.querySelector<HTMLDetailsElement>('[data-overview]')!;
  fold.open = true;
  fold.dispatchEvent(new Event('toggle'));
  await dock.updateComplete;
  await waitFor(() => dock.querySelector('dl-run-overview [data-run-overview]') !== null);
  expect(dock.querySelector('dl-run-overview [data-ledger]')).to.not.equal(null);
  expect(title(dock)).to.equal('objective b');
});

it('opens a wide dock as the Run overview column beside the switcher and the agent page', async () => {
  serve({
    page: () => roster([row('a', 'running'), row('b', 'succeeded')]),
    observe: (id) => observation(row(id, id === 'a' ? 'running' : 'succeeded')),
    overview: () => Response.json(runOverview()),
  });
  const dock = await mount(sourceFor().source, 800);
  await waitFor(() => rows(dock).length === 2);
  await waitFor(() => dock.querySelector('dl-run-overview [data-run-overview]') !== null);

  // The Run panel is a column, not a fold, and its 44px band leads it.
  expect(dock.querySelector('[data-overview]')!.tagName).to.equal('SECTION');
  expect(dock.querySelector('[data-overview-title]')!.textContent).to.contain('Run overview');
  expect(dock.querySelector('[data-overview-summary]')!.textContent).to.contain('6 steps');
  const dossier = dock.querySelector('dl-run-overview')!;
  expect(dossier.querySelector('[data-delivery]'), 'what the Run delivered').to.not.equal(null);
  expect(dossier.querySelector('[data-attention]'), 'what needs attention').to.not.equal(null);
  expect(dossier.querySelector('[data-agent-row][data-agent="main"]')).to.not.equal(null);
  expect(dock.querySelector('[data-switcher-current]')!.textContent).to.contain('Main agent');
  const writing = [...dock.querySelectorAll('textarea, form, [data-draft], [data-cancel]')];
  expect(writing, 'the Inspector only reads').to.deep.equal([]);

  rowFor(dock, 'b').click();
  await waitFor(() => title(dock) === 'objective b');
  expect(dock.querySelector('[data-switcher-current]')!.textContent).to.contain('Child 2');
  rowFor(dock, 'main').click();
  await waitFor(() => session(dock).childSessionId === 'main');
  expect(rowFor(dock, 'main').getAttribute('aria-current')).to.equal('true');
});

it('remembers the agent a Run was left on when the dock is opened again', async () => {
  serve({
    page: () => roster([row('a', 'running'), row('b', 'succeeded')]),
    observe: (id) => observation(row(id, id === 'a' ? 'running' : 'succeeded')),
  });
  const first = await mount(sourceFor().source, 800);
  await waitFor(() => rows(first).length === 2);
  rowFor(first, 'b').click();
  await waitFor(() => title(first) === 'objective b');
  first.remove();

  const again = await mount(sourceFor().source, 800);
  await waitFor(() => rows(again).length === 2);
  await waitFor(() => title(again) === 'objective b');
  expect(rowFor(again, 'b').getAttribute('aria-current')).to.equal('true');
  expect(again.querySelector('[data-switcher-current]')!.textContent).to.contain('Child 2');

  // Another Run remembers its own agent: the Run's own, since nobody chose one there.
  await mount(sourceFor('run-2').source, 800);
  await waitFor(() => rows(again).length === 2);
  const other = await mount(sourceFor('run-2').source, 800);
  await waitFor(() => other.querySelector('[data-agent-session="main"]') !== null);
  expect(other.querySelector('[data-agent-session="main"]')!.getAttribute('aria-current')).to.equal('true');
});

it('follows the width of the pane: the narrow summary becomes the wide Run panel', async () => {
  serve({
    page: () => roster([row('a', 'running'), row('b', 'succeeded')]),
    observe: (id) => observation(row(id, id === 'a' ? 'running' : 'succeeded')),
    overview: () => Response.json(runOverview()),
  });
  const dock = await mount(sourceFor().source, 800);
  await waitFor(() => rows(dock).length === 2);
  await waitFor(() => dock.querySelector('[data-overview]')!.tagName === 'SECTION');
  rowFor(dock, 'b').click();
  await waitFor(() => title(dock) === 'objective b');

  dock.style.width = '420px';
  await waitFor(() => dock.querySelector('[data-overview]')!.tagName === 'DETAILS');
  expect(title(dock), 'the page the reader chose is still on show').to.equal('objective b');
  expect(dock.querySelector('[data-switcher-current]')!.textContent).to.contain('Child 2');

  dock.style.width = '800px';
  await waitFor(() => dock.querySelector('[data-overview]')!.tagName === 'SECTION');
  expect(title(dock)).to.equal('objective b');
});

// ── The child on show ──
