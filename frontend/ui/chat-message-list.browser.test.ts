// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import type {AnswerPresentation} from '../api/conversations.ts';
import {setLanguagePreference} from '../i18n/locale.ts';
import type {ChatTurnView, ProcessEntry} from '../lib/chat-views.ts';
import {applyToolEvent, MAX_TOOL_ROWS} from '../lib/tool-events.ts';
import chatStyles from '../styles/chat.module.css';
import {waitFor} from '../testing/dom.ts';
import type {DlChatMessageList} from './chat-message-list.ts';
import './chat-message-list.ts';

afterEach(() => { document.body.replaceChildren(); });

function turn(overrides: Partial<ChatTurnView> = {}): ChatTurnView {
  return {
    id: 'turn-live',
    userText: 'What changed?',
    userAttachments: [],
    runId: 'run-live',
    state: 'pending',
    streamText: '',
    processEntries: [],
    presentation: null,
    usage: {},
    error: null,
    activity: null,
    cancelRequested: false,
    steeringMessages: [],
    toolRows: [],
    answerSequence: null,
    feedback: null,
    ...overrides,
  };
}

async function mount(first: ChatTurnView): Promise<DlChatMessageList> {
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  list.turns = [first];
  document.body.append(list);
  await list.updateComplete;
  return list;
}

/** The trace a live turn holds after `count` calls, its clock started `agoMs` in the past. */
function traceRows(count: number, agoMs = 0): ChatTurnView['toolRows'] {
  const startedAt = performance.now() - agoMs;
  let rows: ReturnType<typeof applyToolEvent> = [];
  for (let index = 0; index < count; index += 1) {
    rows = applyToolEvent(rows, 'tool_start', {
      tool_name: index === count - 1 ? 'search_web' : 'read',
      call_id: `call-${index}`,
    }, startedAt);
  }
  return rows;
}

const group = (list: DlChatMessageList): HTMLDetailsElement | null => (
  list.querySelector<HTMLDetailsElement>('[data-process-group]')
);

/** The merged-trace entries that hold those rows, in order: what the projection builds from events. */
function entriesOf(rows: ChatTurnView['toolRows']): readonly ProcessEntry[] {
  return rows.map((row) => ({kind: 'tool', row}));
}

const summaryText = (list: DlChatMessageList): string => (
  group(list)?.querySelector('summary')?.textContent?.replace(/\s+/g, ' ').trim() ?? ''
);

const answer: AnswerPresentation = {
  answerText: 'A found answer.',
  parts: [],
  sources: [],
  evidenceImages: [],
  linkCards: [],
  artifacts: [],
  artifactOutcome: {status: 'complete', issues: []},
};

/** What the turn's polite status says; null where it draws none. */
const status = (list: DlChatMessageList): HTMLElement | null => (
  list.querySelector<HTMLElement>('article [role="status"][aria-live="polite"]')
);

it('says a pending turn with nothing to report yet is in progress, while its dots stay silent', async () => {
  const list = await mount(turn());

  expect(status(list)?.textContent?.trim()).to.equal('Answer in progress...');
  expect(status(list)?.classList.contains('dl-sr-only')).to.equal(true);
  const dots = list.querySelector('[data-phase]');
  expect(dots?.getAttribute('data-phase')).to.equal('Answer in progress...');
  expect(dots?.getAttribute('aria-live')).to.equal('off');
});

it('announces that status once: a redraw of the same turn, frame after frame, leaves it untouched', async () => {
  const list = await mount(turn());
  const announced = status(list)!;
  const changes: MutationRecord[] = [];
  const observer = new MutationObserver((records) => { changes.push(...records); });
  observer.observe(announced, {childList: true, characterData: true, subtree: true});

  for (let frame = 0; frame < 5; frame += 1) {
    list.turns = [turn()];
    await list.updateComplete;
  }
  changes.push(...observer.takeRecords());
  observer.disconnect();

  expect(status(list), 'the same element is kept').to.equal(announced);
  expect(changes, 'nothing in it is rewritten, so nothing is read out again').to.have.length(0);
});

it('lets a phase the server names replace the in-progress line, and the stream tail stay silent', async () => {
  const list = await mount(turn());

  list.turns = [turn({state: 'streaming', activity: {kind: 'phase', phase: 'searching'}, streamText: 'Revenue'})];
  await list.updateComplete;

  expect(status(list)?.textContent?.trim()).to.equal('Searching knowledge base...');
  expect(list.querySelector('.stream-tail')?.getAttribute('aria-live')).to.equal('off');
});

it('says nothing once the turn is streaming with no phase to name', async () => {
  const list = await mount(turn({state: 'streaming', streamText: 'Revenue grew'}));

  expect(status(list), 'the growing text is not read out as it grows').to.equal(null);
});

// ── The process group ──

it('gathers a live turn\'s calls into one line that names the work, counts it, and times it', async () => {
  const rows = traceRows(3, 12_000);
  const list = await mount(turn({
    state: 'streaming',
    activity: {kind: 'tool'},
    toolRows: rows,
    processEntries: entriesOf(rows),
  }));

  const details = group(list);
  expect(details, 'a turn with calls draws the group').to.not.equal(null);
  expect(details!.open, 'a live turn opens on its trace, with the newest steps in view').to.equal(true);
  const text = summaryText(list);
  expect(text).to.contain('Searching the web', 'the newest call names what the turn is doing');
  expect(text).to.contain('3 tool calls', 'every call the viewer saw is counted');
  expect(text, 'the clock runs from the first call the viewer saw').to.match(/1[12]\.\ds/);
  expect(list.querySelectorAll('[data-phase]'), 'the phase line stands down for the group')
    .to.have.length(0);
});

it('keeps the reader\'s shut as the turn works, and opens on the rows again when it is pressed', async () => {
  const traceRowsHeld = traceRows(2);
  const list = await mount(turn({
    state: 'streaming',
    activity: {kind: 'tool'},
    toolRows: traceRowsHeld,
    processEntries: entriesOf(traceRowsHeld),
  }));
  const details = group(list)!;
  const summary = details.querySelector('summary')!;

  summary.click();
  await waitFor(() => !details.open);
  await waitFor(() => summary.getAttribute('aria-expanded') === 'false');

  // The reader shut it: a frame later, with more work in the trace, it stays shut.
  const more = traceRows(3);
  list.turns = [turn({
    state: 'streaming',
    activity: {kind: 'tool'},
    toolRows: more,
    processEntries: entriesOf(more),
  })];
  await list.updateComplete;
  expect(details.open, 'the reader\'s choice outlives the redraw').to.equal(false);

  summary.click();
  await waitFor(() => details.open);
  await waitFor(() => summary.getAttribute('aria-expanded') === 'true');
  const rows = [...details.querySelectorAll(`.${chatStyles.toolRow}`)]
    .map((row) => row.textContent?.replace(/\s+/g, ' ').trim());
  expect(rows).to.deep.equal(['Reading a document', 'Reading a document', 'Searching the web']);
});

it('counts every call the turn took, not only the rows the list still holds', async () => {
  const rows = traceRows(MAX_TOOL_ROWS + 4);
  const list = await mount(turn({
    state: 'streaming',
    activity: {kind: 'tool'},
    toolRows: rows,
    processEntries: entriesOf(rows),
  }));

  expect(group(list)!.querySelectorAll(`.${chatStyles.toolRow}`)).to.have.length(MAX_TOOL_ROWS);
  expect(summaryText(list)).to.contain(`${MAX_TOOL_ROWS + 4} tool calls`);
});

it('keeps every step of a live trace, so the block can scroll back through the Run', async () => {
  const rows = traceRows(2);
  const entries: ProcessEntry[] = [
    {kind: 'thinking', text: 'First I read the question.', open: false},
    {kind: 'tool', row: rows[0]!},
    {kind: 'thinking', text: 'Then I searched.', open: false},
    {kind: 'tool', row: rows[1]!},
    {kind: 'thinking', text: 'Then I read the cap.', open: false},
    {kind: 'thinking', text: 'Checking the filing.', open: true},
  ];
  const list = await mount(turn({state: 'streaming', toolRows: rows, processEntries: entries}));

  const trace = list.querySelector<HTMLElement>('[data-process-trace]')!;
  expect(trace.dataset.live, 'a running turn says its trace is live').to.equal('true');
  expect(traceKinds(list), 'the whole history is in the block, folds and rows alike')
    .to.deep.equal(['thinking', 'tool', 'thinking', 'tool', 'thinking', 'thinking']);
  expect(trace.querySelectorAll('[data-thinking]')).to.have.length(4);
  expect(trace.querySelectorAll(`.${chatStyles.toolRow}`)).to.have.length(2);
});

it('redraws the group\'s clock once a second while the turn works', async () => {
  const rows = traceRows(2, 2_000);
  const list = await mount(turn({
    state: 'streaming',
    activity: {kind: 'tool'},
    toolRows: rows,
    processEntries: entriesOf(rows),
  }));
  const clock = (): string => (
    group(list)?.querySelector(`.${chatStyles.processDuration}`)?.textContent?.trim() ?? ''
  );
  const before = clock();
  expect(before).to.match(/^2\.\ds$/);

  await new Promise((resolve) => setTimeout(resolve, 1_100));
  expect(clock(), 'the viewer clock keeps counting on its own').to.not.equal(before);
});

it('keeps the phase line while no call has landed, with no group to open', async () => {
  const list = await mount(turn({state: 'streaming', activity: {kind: 'phase', phase: 'searching'}}));

  expect(group(list)).to.equal(null);
  expect(list.querySelector('[data-phase]')?.getAttribute('data-phase'))
    .to.equal('Searching knowledge base...');
});

it('keeps a succeeded turn to its answer, with no process group left behind', async () => {
  const list = await mount(turn({
    state: 'succeeded',
    presentation: answer,
    toolRows: traceRows(3),
  }));

  expect(group(list), 'the Run going quiet is the product taste this keeps').to.equal(null);
  expect(list.querySelector('dl-answer-presentation')).to.not.equal(null);
});

it('keeps a failed turn to its words, with no process group to open', async () => {
  const list = await mount(turn({
    state: 'failed',
    error: {source: 'Service error. Please try again.', id: 'errors.service'},
    toolRows: traceRows(2),
  }));

  expect(group(list)).to.equal(null);
  expect(list.textContent).to.contain('Service error. Please try again.');
});

it('offers a failed turn\'s own words again instead of leaving only a message', async () => {
  const list = await mount(turn({
    state: 'failed',
    userText: 'Rebuild the sealed model',
    error: {source: 'Service error. Please try again.', id: 'errors.service'},
  }));
  const retried: string[] = [];
  list.addEventListener('dl-chat-retry-turn', (event) => {
    retried.push((event as CustomEvent<{query: string}>).detail.query);
  });

  list.querySelector<HTMLButtonElement>('[aria-label="Retry this message"]')!.click();

  expect(retried).to.deep.equal(['Rebuild the sealed model']);
});

it('draws the group\'s words in the language the reader switches to', async () => {
  const rows = traceRows(2, 2_000);
  const list = await mount(turn({
    state: 'streaming',
    activity: {kind: 'tool'},
    toolRows: rows,
    processEntries: entriesOf(rows),
  }));
  expect(summaryText(list)).to.contain('Searching the web');
  try {
    await setLanguagePreference('zh');
    await list.updateComplete;
    expect(summaryText(list)).to.contain('正在网络搜索');
    expect(summaryText(list)).to.contain('2 次工具调用');
    expect(status(list)?.textContent?.trim(), 'the announcement follows the language too')
      .to.equal('正在网络搜索 · 2 次工具调用');
  } finally {
    await setLanguagePreference('en');
  }
});

// ── The merged process trace: thinking folds and Tool rows, one rolling list ──

/** The trace a turn's group body draws, as the kinds of its entries in order. */
const traceKinds = (list: DlChatMessageList): string[] => (
  [...list.querySelectorAll<HTMLElement>('[data-process-group] .toolTrace > *')].map(
    (entry) => entry.dataset.thinking !== undefined ? 'thinking' : 'tool',
  )
);

it('draws the folds where they happened among the tool rows, the newest fold open', async () => {
  const rows = traceRows(2);
  const list = await mount(turn({
    state: 'streaming',
    streamText: 'Revenue grew.',
    activity: {kind: 'tool'},
    toolRows: rows,
    processEntries: [
      {kind: 'thinking', text: 'First I read the question.', open: false},
      {kind: 'tool', row: rows[0]!},
      {kind: 'thinking', text: 'Then I searched.', open: true},
    ],
  }));

  const group = list.querySelector<HTMLDetailsElement>('[data-process-group]')!;
  expect(group.textContent, 'the line counts the calls').to.contain('2 tool calls');
  expect(traceKinds(list), 'the folds and rows keep the order they happened in')
    .to.deep.equal(['thinking', 'tool', 'thinking']);

  const folds = [...group.querySelectorAll<HTMLDetailsElement>('[data-thinking]')];
  expect(folds.map((fold) => fold.open), 'the finished fold is shut; the one being written is open')
    .to.deep.equal([false, true]);
  expect(folds[0]!.textContent).to.contain('First I read the question.');
  expect(folds[1]!.textContent).to.contain('Then I searched.');
  expect(folds.map((fold) => fold.getAttribute('aria-live')), 'reasoning grows every frame, so it is not read out')
    .to.deep.equal(['off', 'off']);
  // The trace is the one list under the line, inside the process group and the conversation log.
  expect(group.querySelectorAll(`.${chatStyles.toolRow}`), 'the Tool row is the same row as before').to.have.length(1);
  expect(group.closest('[role="log"]')).to.not.equal(null);
});

it('opens a running turn\'s thinking inside its process group, with the line naming it before any call', async () => {
  const list = await mount(turn({
    state: 'streaming',
    streamText: 'Revenue grew.',
    processEntries: [
      {kind: 'thinking', text: 'First I read the question.', open: false},
      {kind: 'thinking', text: 'Checking the filing.', open: true},
    ],
  }));

  expect(traceKinds(list), 'the folds are the trace the group opens on').to.deep.equal(['thinking', 'thinking']);
  // No call has landed, so the one line says what the model is doing and counts nothing.
  expect(summaryText(list), 'the line names the thinking before any call').to.equal('Thinking');
  expect(summaryText(list)).to.not.contain('tool calls');

  // The words lead; the process group that holds the thinking follows them.
  const tail = list.querySelector('.stream-tail')!;
  expect(
    group(list)!.compareDocumentPosition(tail) & Node.DOCUMENT_POSITION_PRECEDING,
    'the group holds the thinking under the words',
  ).to.be.greaterThan(0);
});

it('names the server\'s phase on the thinking line while no call has landed', async () => {
  const list = await mount(turn({
    state: 'streaming',
    activity: {kind: 'phase', phase: 'researching'},
    processEntries: [{kind: 'thinking', text: 'Reading the filings.', open: true}],
  }));

  expect(summaryText(list)).to.equal('Researching sources...');
  expect(list.querySelector('[data-phase]'), 'the phase line stands down for the group').to.equal(null);
});

it('keeps the process line but clears its live trace once the turn has settled', async () => {
  const rows = traceRows(2);
  const list = await mount(turn({
    state: 'cancelled',
    streamText: 'The answer so far',
    toolRows: rows,
    // What the projection leaves a settled view: the line states what happened, the trace is gone.
    processEntries: [],
  }));

  const settled = group(list);
  expect(settled, 'the settled turn still says what it did').to.not.equal(null);
  expect(summaryText(list)).to.contain('2 tool calls');
  expect(settled!.querySelectorAll(`.${chatStyles.toolRow}`), 'no live row survives the settle')
    .to.have.length(0);
  expect(settled!.querySelector('[data-thinking]'), 'and no thinking fold does').to.equal(null);
  expect(list.textContent).to.contain('Stopped');
});

it('draws the Thinking label in the language the reader switches to', async () => {
  const list = await mount(turn({
    state: 'streaming',
    processEntries: [{kind: 'thinking', text: 'Checking.', open: true}],
  }));
  const label = (): string => (
    list.querySelector('[data-thinking] summary')?.textContent?.replace(/\s+/g, ' ').trim() ?? ''
  );
  expect(label()).to.equal('Thinking');
  try {
    await setLanguagePreference('zh');
    await list.updateComplete;
    expect(label()).to.equal('思考');
  } finally {
    await setLanguagePreference('en');
  }
});
