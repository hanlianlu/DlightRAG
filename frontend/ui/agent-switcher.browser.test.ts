// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The Inspector's one-line agent switcher: the line, the roster it opens, the keys that move
 *  between agents, and reading an agent ahead of its click. */

import {expect} from '@esm-bundle/chai';
import type {AgentChildStatus} from '../api/conversations.ts';
import {defineDesignSystemElements} from '../design-system/index.ts';
import {NOW, ago} from '../testing/traces.ts';
import type {AgentId} from '../lib/agent-id.ts';
import {agentIdOf, RUN_AGENT} from '../lib/agent-id.ts';
import type {AgentSwitchDetail, AgentSwitcherPaging, DlAgentSwitcher} from './agent-switcher.ts';
import './agent-switcher.ts';

defineDesignSystemElements();

/** One agent as the roster states it, with only the fields a test reads set. */
function listed(id: string, extra: Partial<AgentChildStatus> = {}): AgentChildStatus & {childSessionId: string} {
  return {
    childSessionId: id,
    status: 'running',
    objective: `objective ${id}`,
    modelRole: undefined,
    usage: null,
    operationId: 'op-a',
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

const MAIN = {...listed('main'), objective: 'Peer margins: how durable?'};

function mountSwitcher(overrides: Partial<DlAgentSwitcher> = {}): DlAgentSwitcher {
  const switcher = document.createElement('dl-agent-switcher') as DlAgentSwitcher;
  switcher.main = MAIN;
  switcher.entries = [listed('a'), listed('b')];
  switcher.selected = RUN_AGENT;
  switcher.active = true;
  Object.assign(switcher, overrides);
  document.body.append(switcher);
  return switcher;
}

const picks = (switcher: DlAgentSwitcher): AgentId[] => {
  const seen: AgentId[] = [];
  switcher.addEventListener('dl-agent-switch', (event) => {
    const agent = (event as CustomEvent<AgentSwitchDetail>).detail.agent;
    seen.push(agent);
    // What a dock does after a pick: the switcher draws the new agent as current.
    switcher.selected = agent;
    void switcher.updateComplete;
  });
  return seen;
};

const realNow = Date.now;
const realSetInterval = window.setInterval;
const realClearInterval = window.clearInterval;

beforeEach(() => { Date.now = () => NOW; });

afterEach(() => {
  document.body.replaceChildren();
  Date.now = realNow;
  window.setInterval = realSetInterval;
  window.clearInterval = realClearInterval;
});

it('shows one line: the current agent, where it stands, and its task', async () => {
  const switcher = mountSwitcher();
  await switcher.updateComplete;

  const line = switcher.querySelector<HTMLElement>('[data-switcher-current]')!;
  const text = line.textContent!.replace(/\s+/g, ' ').trim();
  expect(text).to.contain('Main agent');
  expect(text).to.contain('Running');
  expect(text).to.contain('2m 14s');
  expect(text).to.contain('Peer margins: how durable?');
  expect(line.getAttribute('aria-haspopup')).to.equal('listbox');
  expect(line.getAttribute('aria-expanded')).to.equal('false');
});

it('keeps its own clock, so a running time moves without the dock handing it one', async () => {
  let now = NOW;
  Date.now = () => now;
  const ticks = new Map<number, () => void>();
  const stopped: number[] = [];
  window.setInterval = ((handler: TimerHandler) => {
    const id = ticks.size + 1;
    ticks.set(id, handler as () => void);
    return id;
  }) as typeof window.setInterval;
  window.clearInterval = ((id?: number) => { if (id !== undefined) stopped.push(id); }) as typeof window.clearInterval;

  const switcher = mountSwitcher();
  await switcher.updateComplete;
  expect(ticks.size, 'a running row makes the switcher keep time').to.equal(1);

  now += 5000;
  [...ticks.values()][0]!();
  await switcher.updateComplete;
  expect(switcher.querySelector<HTMLElement>('[data-switcher-current]')!.textContent).to.contain('2m 19s');

  // Nothing counts any more: the clock stops rather than redraw a page whose times are fixed.
  switcher.main = {...MAIN, status: 'succeeded', startedAt: ago(600), finishedAt: ago(540)};
  switcher.entries = [];
  await switcher.updateComplete;
  expect(stopped).to.deep.equal([1]);

  switcher.remove();
  expect(stopped, 'a stopped clock does not stop twice').to.deep.equal([1]);
});

it('keeps no clock when no row is counting time', async () => {
  const intervals = new Map<number, () => void>();
  window.setInterval = ((handler: TimerHandler) => {
    const id = intervals.size + 1;
    intervals.set(id, handler as () => void);
    return id;
  }) as typeof window.setInterval;
  const settled = {status: 'succeeded' as const, startedAt: ago(600), finishedAt: ago(540)};
  const switcher = mountSwitcher({
    main: {...MAIN, ...settled},
    entries: [listed('a', settled), listed('b', settled)],
  });
  await switcher.updateComplete;

  expect(intervals.size).to.equal(0);
});

it('opens the roster: main first, then the children, each with its status and question', async () => {
  const switcher = mountSwitcher({
    entries: [
      listed('a', {pendingQuestions: 1}),
      listed('b', {status: 'succeeded', summary: 'Clause 9.2 caps it.', startedAt: ago(600), finishedAt: ago(540)}),
    ],
  });
  await switcher.updateComplete;
  const popover = switcher.querySelector<HTMLElement>('[data-switcher-popover]')!;
  expect(popover.hasAttribute('hidden'), 'the roster is not drawn open').to.equal(true);

  switcher.querySelector<HTMLButtonElement>('[data-switcher-current]')!.click();
  await switcher.updateComplete;
  expect(popover.hasAttribute('hidden')).to.equal(false);
  expect(switcher.querySelector('[data-switcher-current]')!.getAttribute('aria-expanded')).to.equal('true');

  const rows = [...switcher.querySelectorAll<HTMLElement>('[data-agent-session]')];
  expect(rows.map((row) => row.dataset.agentSession)).to.deep.equal(['main', 'a', 'b']);
  expect(rows[0]!.getAttribute('role')).to.equal('option');
  expect(rows[0]!.getAttribute('aria-selected')).to.equal('true');
  expect(rows[1]!.textContent).to.contain('Asks parent');
  expect(rows[1]!.textContent).to.contain('Running');
  expect(rows[2]!.textContent).to.contain('Clause 9.2 caps it.');
});

it('picks a row, and closes the list behind the pick', async () => {
  const switcher = mountSwitcher();
  const chosen = picks(switcher);
  await switcher.updateComplete;
  switcher.querySelector<HTMLButtonElement>('[data-switcher-current]')!.click();
  switcher.querySelector<HTMLElement>('[data-agent-session="b"]')!.click();
  await switcher.updateComplete;

  expect(chosen.map((agent) => agent.kind === 'run' ? 'main' : agent.id)).to.deep.equal(['b']);
  expect(switcher.querySelector('[data-switcher-popover]')!.hasAttribute('hidden')).to.equal(true);
});

it('moves to the previous and next agent with the shortcuts, wrapping at the ends', async () => {
  const switcher = mountSwitcher();
  const chosen = picks(switcher);
  await switcher.updateComplete;

  const step = (key: string, code: string): void => {
    document.dispatchEvent(new KeyboardEvent('keydown', {key, code, altKey: true, bubbles: true}));
  };
  step(']', 'BracketRight');
  step(']', 'BracketRight');
  step('[', 'BracketLeft');
  await switcher.updateComplete;

  // Next from main goes to the first child, next again to the second, previous comes back.
  expect(chosen.map((agent) => agent.kind === 'run' ? 'main' : agent.id)).to.deep.equal(['a', 'b', 'a']);
});

it('hands the main agent back on ⌘1, from anywhere in the Inspector', async () => {
  const switcher = mountSwitcher({selected: agentIdOf('b')});
  const chosen = picks(switcher);
  await switcher.updateComplete;

  document.dispatchEvent(new KeyboardEvent('keydown', {key: '1', code: 'Digit1', metaKey: true, bubbles: true}));
  await switcher.updateComplete;

  expect(chosen).to.have.length(1);
  expect(chosen[0]!.kind).to.equal('run');
});

it('reads the agent a pointer or a focus lands on before the click does', async () => {
  const warmed: AgentId[] = [];
  const switcher = mountSwitcher({warm: (agent) => { warmed.push(agent); }});
  await switcher.updateComplete;
  switcher.querySelector<HTMLElement>('[data-switcher-current]')!.click();
  await switcher.updateComplete;

  const rowElement = switcher.querySelector<HTMLElement>('[data-agent-session="a"]')!;
  rowElement.dispatchEvent(new MouseEvent('mouseenter'));
  rowElement.focus();

  expect(warmed.map((agent) => agent.kind === 'child' ? agent.id : 'main')).to.deep.equal(['a', 'a']);
});

it('picks the highlighted row with Enter and closes with Escape, ignoring keystrokes typed in a box', async () => {
  const switcher = mountSwitcher();
  const chosen = picks(switcher);
  await switcher.updateComplete;
  switcher.querySelector<HTMLButtonElement>('[data-switcher-current]')!.click();
  await switcher.updateComplete;

  // A reader typing elsewhere keeps their keys.
  const box = document.createElement('textarea');
  document.body.append(box);
  box.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', bubbles: true}));
  expect(chosen).to.have.length(0);
  box.remove();

  // ArrowDown highlights the first child; Enter picks it.
  document.dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowDown', bubbles: true}));
  document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', bubbles: true}));
  await switcher.updateComplete;
  expect(chosen.map((agent) => agent.kind === 'child' ? agent.id : 'main')).to.deep.equal(['a']);
  expect(switcher.querySelector('[data-switcher-popover]')!.hasAttribute('hidden')).to.equal(true);

  switcher.querySelector<HTMLButtonElement>('[data-switcher-current]')!.click();
  await switcher.updateComplete;
  document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
  await switcher.updateComplete;
  expect(switcher.querySelector('[data-switcher-popover]')!.hasAttribute('hidden')).to.equal(true);
});

it('offers the older children from inside the roster', async () => {
  const paging: AgentSwitcherPaging = {hasOlder: true, loading: false, failed: false};
  const switcher = mountSwitcher({paging});
  const loads: number[] = [];
  switcher.addEventListener('dl-agent-load-older', () => { loads.push(1); });
  await switcher.updateComplete;
  switcher.querySelector<HTMLButtonElement>('[data-switcher-current]')!.click();
  await switcher.updateComplete;

  const more = switcher.querySelector<HTMLButtonElement>('[data-load-older="children"]')!;
  expect(more.textContent!.trim()).to.equal('Load older children');
  more.click();
  expect(loads).to.deep.equal([1]);

  switcher.paging = {hasOlder: true, loading: true, failed: false};
  await switcher.updateComplete;
  expect(switcher.querySelector<HTMLButtonElement>('[data-load-older="children"]')!.getAttribute('aria-disabled'))
    .to.equal('true');
});

it('filters the roster to the agents a search found words in', async () => {
  const switcher = mountSwitcher({hits: new Map([['b', 2]])});
  await switcher.updateComplete;
  switcher.querySelector<HTMLButtonElement>('[data-switcher-current]')!.click();
  await switcher.updateComplete;

  expect([...switcher.querySelectorAll<HTMLElement>('[data-agent-session]')].map((row) => row.dataset.agentSession))
    .to.deep.equal(['b']);
  expect(switcher.querySelector('[data-agent-session="b"]')!.textContent).to.contain('2 matches');
});

it('never shows a Session id as a name, and keeps the roster\'s place when it can', async () => {
  const id = '3f2a7c1e-2b44-4e87-9f0d-8c6c1f5b9a10';
  const switcher = mountSwitcher({
    entries: [listed(id, {objective: ''}), listed('b', {objective: 'Check peers'})],
    selected: agentIdOf(id),
  });
  await switcher.updateComplete;
  // The roster states where it stands, so the closed line names its place.
  expect(switcher.querySelector('[data-switcher-current]')!.textContent).to.contain('Child 1');
  expect(switcher.textContent).to.not.contain('3f2a7c1e');

  // A child with no objective is named by the canonical fallback in the roster row too.
  switcher.querySelector<HTMLButtonElement>('[data-switcher-current]')!.click();
  await switcher.updateComplete;
  const childRow = switcher.querySelector<HTMLElement>(`[data-agent-session="${id}"]`)!;
  expect(childRow.textContent).to.contain('Child agent');
  expect(childRow.textContent).to.not.contain('3f2a7c1e');

  // A child the roster cannot place has no place to name: the fallback, never the id.
  switcher.entries = [listed('b', {objective: 'Check peers'})];
  switcher.selected = agentIdOf(id);
  await switcher.updateComplete;
  expect(switcher.querySelector('[data-switcher-current]')!.textContent).to.contain('Child agent');
  expect(switcher.textContent).to.not.contain('3f2a7c1e');
});

it('names what it cannot show', async () => {
  const switcher = mountSwitcher({entries: []});
  await switcher.updateComplete;
  expect(switcher.querySelector('[data-switcher-current]')!.textContent).to.contain('Main agent');

  switcher.entries = [];
  switcher.hits = new Map();
  await switcher.updateComplete;
  switcher.querySelector<HTMLButtonElement>('[data-switcher-current]')!.click();
  await switcher.updateComplete;
  expect(switcher.querySelector('[data-switcher-popover]')!.textContent).to.contain('No agent holds those words.');
});

it('filters the roster by what the reader types, and says so when nothing matches', async () => {
  const switcher = mountSwitcher({entries: [listed('a', {objective: 'Read the filing'}), listed('b', {objective: 'Check peers'})]});
  await switcher.updateComplete;
  switcher.querySelector<HTMLButtonElement>('[data-switcher-current]')!.click();
  await switcher.updateComplete;

  const filter = switcher.querySelector<HTMLInputElement>('[data-switcher-filter]')!;
  filter.value = 'filing';
  filter.dispatchEvent(new Event('input', {bubbles: true}));
  await switcher.updateComplete;
  expect([...switcher.querySelectorAll<HTMLElement>('[data-agent-session]')].map((row) => row.dataset.agentSession))
    .to.deep.equal(['a']);

  filter.value = 'nothing matches these words';
  filter.dispatchEvent(new Event('input', {bubbles: true}));
  await switcher.updateComplete;
  expect(switcher.querySelector<HTMLElement>('[data-switcher-popover]')!.textContent).to.contain('No agent matches that filter.');
});

it('names a waiting question and the main agent\'s objective in the roster', async () => {
  const switcher = mountSwitcher({entries: [listed('a', {pendingQuestions: 2})]});
  await switcher.updateComplete;
  switcher.querySelector<HTMLButtonElement>('[data-switcher-current]')!.click();
  await switcher.updateComplete;

  const popover = switcher.querySelector<HTMLElement>('[data-switcher-popover]')!;
  expect(popover.textContent).to.contain('Asks parent · 2');
  // The main row's name is its role, so its objective is stated beside it, as a child's name is.
  expect(popover.textContent).to.contain('Peer margins: how durable?');
});
