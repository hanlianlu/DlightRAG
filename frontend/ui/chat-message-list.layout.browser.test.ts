// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The live process block as the shipped stylesheets draw it: one bounded viewport that holds the
 *  newest steps and scrolls the rest inside itself, so a Run's thinking cannot run down the page. */

import {expect} from '@esm-bundle/chai';
import {PROCESS_TRACE_VISIBLE_ROWS, type ChatTurnView, type ProcessEntry} from '../lib/chat-views.ts';
import {applyToolEvent} from '../lib/tool-events.ts';
import chatStyles from '../styles/chat.module.css';
import type {DlChatMessageList} from './chat-message-list.ts';
import './chat-message-list.ts';

afterEach(() => { document.body.replaceChildren(); });

function turn(overrides: Partial<ChatTurnView> = {}): ChatTurnView {
  return {
    id: 'turn-live',
    userText: 'What changed?',
    userAttachments: [],
    runId: 'run-live',
    state: 'streaming',
    streamText: '',
    processEntries: [],
    presentation: null,
    usage: {},
    error: null,
    activity: {kind: 'tool'},
    cancelRequested: false,
    steeringMessages: [],
    toolRows: [],
    answerSequence: null,
    feedback: null,
    ...overrides,
  };
}

/** The tool rows a turn's trace holds after `count` calls. */
function traceRows(count: number): ChatTurnView['toolRows'] {
  let rows: ReturnType<typeof applyToolEvent> = [];
  for (let index = 0; index < count; index += 1) {
    rows = applyToolEvent(rows, 'tool_start', {
      tool_name: index === count - 1 ? 'search_web' : 'read',
      call_id: `call-${index}`,
    }, performance.now());
  }
  return rows;
}

function entriesOf(rows: ChatTurnView['toolRows']): readonly ProcessEntry[] {
  return rows.map((row) => ({kind: 'tool', row}));
}

async function mount(first: ChatTurnView): Promise<DlChatMessageList> {
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  list.turns = [first];
  document.body.append(list);
  await list.updateComplete;
  return list;
}

const traceOf = (list: DlChatMessageList): HTMLElement => (
  list.querySelector<HTMLElement>('[data-process-trace]')!
);

/** How far the trace is from its newest row. */
const fromEnd = (trace: HTMLElement): number => (
  trace.scrollHeight - trace.scrollTop - trace.clientHeight
);

it('bounds the live trace and scrolls the history inside it', async () => {
  const rows = traceRows(PROCESS_TRACE_VISIBLE_ROWS * 3);
  const list = await mount(turn({toolRows: rows, processEntries: entriesOf(rows)}));
  const trace = traceOf(list);
  const style = getComputedStyle(trace);

  expect(style.overflowY, 'the block scrolls its own history').to.equal('auto');
  const bound = Number.parseFloat(style.maxHeight);
  expect(Number.isFinite(bound) && bound > 0, 'the block states a max height').to.equal(true);
  expect(trace.clientHeight, 'the viewport never grows past its bound').to.be.at.most(bound + 1);
  expect(trace.scrollHeight, 'the history overflows the viewport').to.be.greaterThan(trace.clientHeight);
  expect(
    trace.querySelectorAll(`.${chatStyles.toolRow}`),
    'every step is in the block, not only the visible ones',
  ).to.have.length(PROCESS_TRACE_VISIBLE_ROWS * 3);
});

it('holds the newest step in view while the turn works', async () => {
  const rows = traceRows(3);
  const list = await mount(turn({toolRows: rows, processEntries: entriesOf(rows)}));

  // More steps arrive while the turn runs: the block follows them, and the page does not grow.
  const more = traceRows(PROCESS_TRACE_VISIBLE_ROWS * 3);
  list.turns = [turn({toolRows: more, processEntries: entriesOf(more)})];
  await list.updateComplete;

  const trace = traceOf(list);
  expect(fromEnd(trace), 'the newest step is held in view').to.be.at.most(8);
});

it('leaves the reader\'s place when they scroll up, and follows again at the end', async () => {
  const rows = traceRows(PROCESS_TRACE_VISIBLE_ROWS * 3);
  const list = await mount(turn({toolRows: rows, processEntries: entriesOf(rows)}));

  // The reader scrolls up into the history: the next redraw must not pull them back down.
  const trace = traceOf(list);
  trace.scrollTop = 0;
  trace.dispatchEvent(new Event('scroll'));
  const more = traceRows(PROCESS_TRACE_VISIBLE_ROWS * 4);
  list.turns = [turn({toolRows: more, processEntries: entriesOf(more)})];
  await list.updateComplete;
  expect(traceOf(list).scrollTop, 'the reader keeps their place').to.equal(0);

  // Back at the newest step, the block follows the Run again.
  const followed = traceOf(list);
  followed.scrollTop = followed.scrollHeight;
  followed.dispatchEvent(new Event('scroll'));
  const most = traceRows(PROCESS_TRACE_VISIBLE_ROWS * 5);
  list.turns = [turn({toolRows: most, processEntries: entriesOf(most)})];
  await list.updateComplete;
  expect(fromEnd(traceOf(list)), 'at its end the trace follows again').to.be.at.most(8);
});

it('keeps a streaming thought inside the bounded block, never in the page flow', async () => {
  const rows = traceRows(2);
  const list = await mount(turn({
    toolRows: rows,
    processEntries: [
      {kind: 'thinking', text: 'Read the filing. '.repeat(400), open: true},
      ...entriesOf(rows),
    ],
  }));

  const trace = traceOf(list);
  const fold = trace.querySelector<HTMLDetailsElement>('[data-thinking]');
  expect(fold, 'the thought is a row of the trace').to.not.equal(null);
  expect(trace.contains(fold)).to.equal(true);
  expect(trace.scrollHeight, 'the growing thought scrolls inside the block')
    .to.be.greaterThan(trace.clientHeight);
  expect(trace.clientHeight, 'and the block stays bounded')
    .to.be.at.most(Number.parseFloat(getComputedStyle(trace).maxHeight) + 1);
});
