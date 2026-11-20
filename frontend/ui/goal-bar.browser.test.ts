// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The goal bar: what it reads, and the reader's controls. */

import {expect} from '@esm-bundle/chai';
import type {DlGoalBar} from './goal-bar.ts';
import './goal-bar.ts';

const originalFetch = window.fetch;

const goalWire = (overrides: Record<string, unknown> = {}): Record<string, unknown> => ({
  conversation_id: 'c-1',
  goal_id: 'goal-1',
  revision: 2,
  objective: 'finish the sealed model',
  phase: 'active',
  max_rounds: 8,
  rounds_started: 2,
  blocked_reason: null,
  ...overrides,
});

function serve(current: Record<string, unknown> | null) {
  const posts: {body: Record<string, unknown>; url: string}[] = [];
  let state = current === null ? null : {...current};
  let revision = state ? Number(state.revision) : 0;
  window.fetch = async (input, init) => {
    const url = String(input);
    if (init?.method !== 'POST') return Response.json(state);
    const body = JSON.parse(String(init.body)) as Record<string, unknown>;
    posts.push({url, body});
    if (body.action === 'clear') {
      state = null;
      return Response.json(null);
    }
    if (state === null) return Response.json(null);
    revision += 1;
    state = {...state, revision};
    if (body.action === 'pause') {
      state = {...state, phase: 'paused'};
    } else if (body.action === 'resume') {
      state = {...state, phase: 'active', blocked_reason: null};
    } else if (body.action === 'edit') {
      if (body.objective !== undefined) state = {...state, objective: body.objective};
      if (body.max_goal_rounds !== undefined) state = {...state, max_rounds: body.max_goal_rounds};
    }
    return Response.json(state);
  };
  return posts;
}

/** A server that refuses the next command with a goal module's own envelope. */
function serveRefusal(kind: string, message: string): void {
  window.fetch = async (_input, init) => {
    if (init?.method !== 'POST') return Response.json(goalWire());
    return Response.json({detail: {kind, message}}, {status: 409});
  };
}

async function settleBar(bar: DlGoalBar): Promise<void> {
  await bar.updateComplete;
  await new Promise((resolve) => setTimeout(resolve, 0));
  await bar.updateComplete;
}

async function mount(conversationId = 'c-1'): Promise<DlGoalBar> {
  const bar = document.createElement('dl-goal-bar') as DlGoalBar;
  document.body.append(bar);
  bar.conversationId = conversationId;
  await settleBar(bar);
  return bar;
}


/** The bar's controls are icon buttons: their accessible name is their aria-label. */
function control(bar: DlGoalBar, label: string): HTMLButtonElement | undefined {
  return [...bar.querySelectorAll<HTMLButtonElement>('button')].find(
    (button) => button.getAttribute('aria-label') === label || button.textContent?.trim() === label,
  );
}

function text(bar: DlGoalBar): string {
  return (bar.textContent ?? '').replace(/\s+/g, ' ').trim();
}

afterEach(() => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
});

it('draws nothing while the conversation carries no goal', async () => {
  serve(null);
  const bar = await mount();
  expect(bar.querySelector('section')).to.equal(null);
});

it('shows an active goal as ongoing, with Pause and no Resume', async () => {
  serve(goalWire());
  const bar = await mount();
  const words = text(bar);
  expect(words).to.contain('Ongoing goal');
  expect(words).to.contain('Round 2/8');
  expect(words).to.contain('finish the sealed model');
  expect(control(bar, 'Resume'), 'the platform continues an active goal on its own').to.equal(undefined);
  const pause = control(bar, 'Pause');
  expect(pause).to.not.equal(undefined);
  expect(pause?.getAttribute('title')).to.equal('Pause stops the goal now. A running goal round is cancelled; your own running turn is not.');
  expect(pause?.getAttribute('aria-description')).to.equal('Pause stops the goal now. A running goal round is cancelled; your own running turn is not.');
  expect(bar.querySelectorAll('[role="status"]'), 'an active goal has no announcement line')
    .to.have.length(0);
  expect(bar.querySelector('.pill')?.getAttribute('role'), 'the pill is visual, not a live region')
    .to.equal(null);
});

it('shows the phase, the round, the objective, and a blocked condition', async () => {
  serve(goalWire({phase: 'blocked', rounds_started: 3, blocked_reason: 'the filings portal is unreachable'}));
  const bar = await mount();
  const words = text(bar);
  expect(words).to.contain('Blocked goal');
  expect(words).to.contain('Round 3/8');
  expect(words).to.contain('finish the sealed model');
  expect(words).to.contain('the filings portal is unreachable');
  expect(words).to.contain('Goal blocked. Resume to continue.');
  expect(control(bar, 'Resume')).to.not.equal(undefined);
  expect(control(bar, 'Pause')).to.equal(undefined);
});

it('pauses the goal through the command the reader pressed', async () => {
  const posts = serve(goalWire());
  const bar = await mount();
  const pause = control(bar, 'Pause');
  expect(pause).to.not.equal(undefined);
  pause?.click();
  await settleBar(bar);
  expect(posts).to.deep.equal([
    {url: '/web/api/conversations/c-1/goal', body: {action: 'pause'}},
  ]);
  expect(text(bar)).to.contain('Paused goal');
  expect(control(bar, 'Resume')).to.not.equal(undefined);
  expect(text(bar), 'the pause consequence is a visible line, not tooltip-only')
    .to.contain('Paused. The goal round in progress was cancelled; no next round will start. Your own running turn is not stopped.');
  expect(text(bar), 'pause is not the goal ending; complete and blocked carry the notice')
    .to.not.contain('No further round will start.');
  expect(bar.querySelectorAll('[role="status"]'), 'one announced notice region, not several')
    .to.have.length(1);
  expect(bar.querySelector('[role="status"]')?.getAttribute('aria-live')).to.equal('polite');
});

it('says a blocked goal at its cap must raise the cap, not merely resume', async () => {
  serve(goalWire({
    phase: 'blocked',
    rounds_started: 8,
    max_rounds: 8,
    blocked_reason: 'the goal reached its round cap',
  }));
  const bar = await mount();
  expect(text(bar)).to.contain('Goal reached its round cap. Raise the cap to continue.');
  expect(text(bar)).to.not.contain('Goal blocked. Resume to continue.');
});

it('shows a complete goal with its notice and no continuation control', async () => {
  serve(goalWire({phase: 'complete'}));
  const bar = await mount();
  expect(text(bar)).to.contain('Complete goal');
  expect(text(bar)).to.contain('Goal complete. No further round will start.');
  expect(control(bar, 'Pause')).to.equal(undefined);
  expect(control(bar, 'Resume')).to.equal(undefined);
  expect(control(bar, 'Edit')).to.equal(undefined);
});

it('edits the objective in place, and clears the goal away', async () => {
  const posts = serve(goalWire());
  const bar = await mount();
  const edit = control(bar, 'Edit');
  edit?.click();
  await settleBar(bar);
  const textarea = bar.querySelector('textarea');
  expect(textarea?.value).to.equal('finish the sealed model');
  if (textarea) {
    textarea.value = 'accept the enterprise framing';
    textarea.dispatchEvent(new Event('input', {bubbles: true}));
  }
  const save = control(bar, 'Save');
  save?.click();
  await settleBar(bar);
  expect(posts[0]?.body).to.deep.equal({
    action: 'edit',
    objective: 'accept the enterprise framing',
  });
  expect(text(bar)).to.contain('accept the enterprise framing');

  const clear = control(bar, 'Clear');
  clear?.click();
  await settleBar(bar);
  expect(posts[1]?.body).to.deep.equal({action: 'clear'});
  expect(bar.querySelector('section')).to.equal(null);
});

it('announces a resume, and only a resume, so the chat can read the new round', async () => {
  serve(goalWire());
  const bar = await mount();
  let announced = 0;
  bar.addEventListener('dl-goal-resumed', () => {
    announced += 1;
  });

  control(bar, 'Pause')?.click();
  await settleBar(bar);
  expect(announced).to.equal(0);

  control(bar, 'Resume')?.click();
  await settleBar(bar);
  expect(announced).to.equal(1);
});

it('asks for a higher cap when Resume is pressed at the cap, then raises and resumes', async () => {
  const posts = serve(goalWire({
    phase: 'blocked',
    rounds_started: 3,
    max_rounds: 3,
    blocked_reason: 'the goal reached its round cap; edit the cap or scope a new objective to continue',
  }));
  const bar = await mount();
  control(bar, 'Resume')?.click();
  await settleBar(bar);
  const input = bar.querySelector<HTMLInputElement>('#goal-cap-input');
  expect(input).to.not.equal(null);
  expect(text(bar)).to.contain('Goal reached its round cap');
  if (input) {
    input.value = '5';
    input.dispatchEvent(new Event('input', {bubbles: true}));
  }
  control(bar, 'Raise & resume')?.click();
  await settleBar(bar);
  await settleBar(bar);
  expect(posts.map((post) => post.body)).to.deep.equal([
    {action: 'edit', max_goal_rounds: 5},
    {action: 'resume'},
  ]);
  expect(text(bar)).to.contain('Ongoing goal');
  expect(text(bar)).to.contain('Round 3/5');
});

it('refuses a cap that is not above the rounds already started', async () => {
  const posts = serve(goalWire({
    phase: 'blocked',
    rounds_started: 3,
    max_rounds: 3,
    blocked_reason: 'the goal reached its round cap; edit the cap or scope a new objective to continue',
  }));
  const bar = await mount();
  control(bar, 'Resume')?.click();
  await settleBar(bar);
  const input = bar.querySelector<HTMLInputElement>('#goal-cap-input');
  if (input) {
    input.value = '3';
    input.dispatchEvent(new Event('input', {bubbles: true}));
  }
  control(bar, 'Raise & resume')?.click();
  await settleBar(bar);
  expect(posts).to.deep.equal([]);
  expect(text(bar)).to.contain('whole number above the rounds already started');
});

it('localizes a known goal refusal by its stable code', async () => {
  serveRefusal('goal_invalid_transition', 'pause needs an active goal; this one is complete');
  const bar = await mount();
  control(bar, 'Clear')?.click();
  await settleBar(bar);
  expect(text(bar)).to.contain('The goal cannot make that change now.');
  expect(text(bar), 'the concrete condition the mapped copy would hide')
    .to.contain('pause needs an active goal; this one is complete');
  expect(text(bar)).to.not.contain('The goal command failed.');
  const failure = bar.querySelector<HTMLElement>('[role="alert"]');
  expect(failure?.getAttribute('title')).to.equal(
    'The goal cannot make that change now. pause needs an active goal; this one is complete',
  );
});

it('shows an unknown refusal\u2019s server sentence rather than the generic line', async () => {
  serveRefusal('a_later_domain_reason', 'the goal is sealed for this release');
  const bar = await mount();
  control(bar, 'Clear')?.click();
  await settleBar(bar);
  expect(text(bar)).to.contain('the goal is sealed for this release');
  expect(text(bar)).to.not.contain('The goal command failed.');
});
