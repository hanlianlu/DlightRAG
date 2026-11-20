// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The dock narrowed by a search: the list and the pane keep to what holds the words, and a long message folds
 *  unless it holds them. */

import {expect} from '@esm-bundle/chai';
import {RUN_AGENT} from '../lib/agent-id.ts';
import {activityPage, answeredTurn, overview, roster, row, searchPage, serve, sourceFor, stepSearchPage} from '../testing/traces.ts';
import {mount, restoreDockFixtures, rowFor, rows, session} from '../testing/traces-dock.ts';
import {waitFor} from '../testing/dom.ts';
import './inspector-traces.ts';

restoreDockFixtures();

it('shows the words that a Run\'s own agent holds, which the wire calls main', async () => {
  serve({
    page: () => roster([row('a', 'succeeded')], null, 'succeeded'),
    transcript: async () => activityPage([{role: 'assistant', content: 'the whole transcript'}]),
  });
  const source = {
    ...sourceFor('run-1', answeredTurn({state: 'succeeded'})).source,
    search: async (text: string) => ({
      query: text,
      terms: ['基于德国联邦银行'],
      total: 1,
      match: 'all',
      messages: [
        {role: 'tool', name: 'read', content: '…基于德国联邦银行…', snippet: '…基于德国联邦银行…',
         agent: RUN_AGENT, agentLabel: null, sequence: 1, toolCalls: [], toolCallId: 't1', isError: false},
      ],
    }),
  };
  const dock = await mount(source as never, 900);
  const box = dock.querySelector('input[type="search"]') as HTMLInputElement;
  box.value = '基于德国联邦银行';
  box.dispatchEvent(new Event('input', {bubbles: true}));
  await waitFor(() => dock.querySelectorAll('mark').length > 0);

  expect(dock.textContent ?? '', 'the Run\'s own agent shows what was found in it').to.contain('基于德国联邦银行');
  expect(rows(dock)[0]?.text ?? '', 'and its row counts them').to.not.contain('1 match');
  expect(
    (dock.querySelector('[data-agent-session="main"]')?.textContent ?? '').includes('1 match'),
    'the Run\'s own row carries the count',
  ).to.equal(true);
});

it('filters the list and the pane down to what the words were found in', async () => {
  serve({
    page: () => roster([row('a', 'succeeded'), row('b', 'succeeded')], null, 'succeeded'),
    transcript: async () => activityPage([{role: 'assistant', content: 'the whole transcript'}]),
  });
  const source = {
    ...sourceFor('run-1', answeredTurn({state: 'succeeded'})).source,
    search: async (text: string) => ({
      query: text,
      terms: ['branch'],
      total: 2,
      match: 'all',
      messages: [
        {role: 'assistant', content: 'the branch question', snippet: 'a look at the branch question',
         agent: {kind: 'child', id: 'a'}, agentLabel: null, sequence: 1, toolCalls: [], toolCallId: '', name: '', isError: false},
        {role: 'assistant', content: 'another branch', snippet: 'a second branch',
         agent: RUN_AGENT, agentLabel: null, sequence: 2, toolCalls: [], toolCallId: '', name: '', isError: false},
      ],
    }),
  };
  const dock = await mount(source as never, 900);
  await waitFor(() => rows(dock).length === 2);

  const box = dock.querySelector('input[type="search"]') as HTMLInputElement;
  box.value = 'branch';
  box.dispatchEvent(new Event('input', {bubbles: true}));
  await waitFor(() => dock.querySelectorAll('mark').length > 0);

  expect(rows(dock).map((one) => one.id), 'an agent with nothing to show is not shown').to.deep.equal(['a']);
  expect(rows(dock)[0]?.text.includes('1 match'), 'and the agent that is shown says how many').to.equal(true);
  expect(dock.querySelector('mark')?.textContent, 'the words are marked where they sit').to.equal('branch');
  expect(dock.querySelector('details')?.open, 'and what was found is shown, not folded away').to.equal(true);
  expect(dock.querySelectorAll('mark').length, 'every match is marked').to.be.greaterThan(0);
  const pane = dock.textContent ?? '';
  expect(pane.includes('the whole transcript'), 'the pane holds the matches, not the transcript').to.equal(false);
  // The Run's own agent answers first, so a reader lands on the matches it holds.
  expect(pane.includes('another branch'), 'and the matches that agent holds').to.equal(true);
  expect(pane.includes('the branch question'), 'and not another agent\'s').to.equal(false);

  (dock.querySelector('button[class*="searchClear"]') as HTMLButtonElement).click();
  await waitFor(() => rows(dock).length === 2);
  await waitFor(() => (dock.textContent ?? '').includes('the whole transcript'));
  expect(dock.querySelectorAll('mark').length, 'and the marks go with the words').to.equal(0);
});

it('shows a child its own matches when the words are in more than one agent', async () => {
  serve({
    page: () => roster([row('a', 'succeeded'), row('b', 'succeeded')], null, 'succeeded'),
    transcript: async () => activityPage([{role: 'assistant', content: 'the whole transcript'}]),
  });
  const source = {
    ...sourceFor('run-1', answeredTurn({state: 'succeeded'})).source,
    search: async (text: string) => ({
      query: text,
      terms: ['德国'],
      total: 2,
      match: 'all',
      messages: [
        {role: 'assistant', content: '主代理里的德国段落', snippet: '…德国…',
         agent: RUN_AGENT, agentLabel: null, sequence: 1, toolCalls: [], toolCallId: '', name: '', isError: false},
        {role: 'tool', content: '子代理里的德国记录', snippet: '…德国…',
         agent: {kind: 'child', id: 'a'}, agentLabel: 'Build the QUANTITATIVE SPINE', sequence: 2,
         toolCalls: [], toolCallId: 't1', name: 'read', isError: false},
      ],
    }),
  };
  const dock = await mount(source as never, 900);
  await waitFor(() => rows(dock).length === 2);
  const box = dock.querySelector('input[type="search"]') as HTMLInputElement;
  box.value = '德国';
  box.dispatchEvent(new Event('input', {bubbles: true}));
  await waitFor(() => dock.querySelectorAll('mark').length > 0);

  // The child whose words hold a match must show that match when a reader walks to it.
  rowFor(dock, 'a').click();
  await waitFor(() => (session(dock)?.textContent ?? '').includes('子代理里的德国记录'));
  expect(dock.textContent ?? '', 'the child shows the message the words were found in').to.contain('子代理里的德国记录');
  expect(dock.textContent ?? '', 'and not the other agent\'s').to.not.contain('主代理里的德国段落');
});

it('narrows the pane of the child a reader already has open', async () => {
  serve({
    page: () => roster([row('a', 'succeeded'), row('b', 'succeeded')], null, 'succeeded'),
    transcript: async () => activityPage([{role: 'assistant', content: 'THE WHOLE STEP TEXT',
      toolCalls: [], toolCallId: '', agent: 'a', agentLabel: null, snippet: '', isError: false}]),
  });
  const source = {
    ...sourceFor('run-1', answeredTurn({state: 'succeeded'})).source,
    search: async (text: string) => ({
      query: text,
      terms: ['联邦'],
      total: 1,
      match: 'all',
      messages: [
        {role: 'tool', name: 'read', content: '…此处提到联邦…', snippet: '…此处提到联邦…',
         agent: {kind: 'child', id: 'a'}, agentLabel: null, sequence: 30, toolCalls: [], toolCallId: 't1', isError: false},
      ],
    }),
  };
  const dock = await mount(source as never, 900);
  await waitFor(() => rows(dock).length === 2);
  // The reader is already reading this child when they search.
  rowFor(dock, 'a').click();
  await waitFor(() => (dock.textContent ?? '').includes('THE WHOLE STEP TEXT'));

  const box = dock.querySelector('input[type="search"]') as HTMLInputElement;
  box.value = '联邦';
  box.dispatchEvent(new Event('input', {bubbles: true}));
  await waitFor(() => dock.querySelectorAll('mark').length > 0);

  const pane = session(dock)?.textContent ?? '';
  expect(pane, 'the pane is the matches, not the step it was reading').to.not.contain('THE WHOLE STEP TEXT');
  expect(pane).to.contain('联邦');
});

it('folds a long message into a window instead of laying it out', async () => {
  const long = Array.from({length: 30}, (_, line) => `${line}: a paragraph of the agent's own words`).join('\n');
  serve({
    page: () => roster([row('a', 'succeeded')], null, 'succeeded'),
    transcript: async () => activityPage([{role: 'assistant', content: long, tool_calls: []}]),
  });
  const dock = await mount(sourceFor('run-1', answeredTurn({state: 'succeeded'})).source as never, 900);
  await waitFor(() => rows(dock).length === 1);
  rowFor(dock, 'a').click();
  await waitFor(() => (dock.textContent ?? '').includes('0: a paragraph'));

  const steps = [...(dock.querySelector('dl-activity ol')?.children ?? [])];
  const folded = steps.filter((item) => item.querySelector(':scope > details > summary') !== null);
  expect(folded.length, 'the long message is drawn as a fold').to.be.greaterThan(0);
  expect(dock.querySelector('dl-activity p'), 'and not as a paragraph of its whole length').to.equal(null);
  const details = folded[0]!.querySelector('details') as HTMLDetailsElement;
  expect(details.open, 'shut, so the page is as long as its steps').to.equal(false);
  (details.querySelector('summary') as HTMLElement).click();
  await new Promise((resolve) => setTimeout(resolve, 40));
  expect(details.open, 'and open when its line is pressed').to.equal(true);
  expect((details.textContent ?? '').includes('29: a paragraph'), 'holding the whole of it').to.equal(true);
});

it('leaves a message that holds the words open, never folded away', async () => {
  const body = Array.from({length: 20}, (_, line) => `line ${line} of a long result`).join('\n');
  serve({
    page: () => roster([row('a', 'succeeded')], null, 'succeeded'),
    transcript: async () => activityPage([{role: 'assistant', content: 'the whole transcript', tool_calls: []}]),
  });
  const source = {
    ...sourceFor('run-1', answeredTurn({state: 'succeeded'})).source,
    search: async (text: string) => ({
      query: text, terms: ['联邦'], total: 1, match: 'all',
      messages: [{role: 'assistant', content: `${body}\n提到了联邦\n${body}`, snippet: '', agent: RUN_AGENT,
        agentLabel: null, sequence: 1, toolCalls: [], toolCallId: '', name: '', isError: false}],
    }),
  };
  const dock = await mount(source as never, 900);
  await waitFor(() => rows(dock).length === 1);
  const box = dock.querySelector('input[type="search"]') as HTMLInputElement;
  box.value = '联邦';
  box.dispatchEvent(new Event('input', {bubbles: true}));
  await waitFor(() => dock.querySelectorAll('mark').length > 0);

  const mark = dock.querySelector('mark') as HTMLElement;
  const fold = mark.closest('details');
  expect(fold === null || fold.open, 'what a search found is not behind a closed fold').to.equal(true);
});

// ── The Run panel's step search: the only step list the Run carries ──

/** The Run's matching steps as the route states them, one row a test reads. */
function match(extra: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    agent: 'main',
    agent_label: 'main',
    kind: 'tool',
    name: 'read',
    subject: 'northwind-msa.pdf',
    excerpt: 'The cap is 6% of the remaining fees.',
    marks: [],
    ...extra,
  };
}

it('searches the Run\'s steps and opens the agent a match names', async () => {
  serve({
    page: () => roster([row('a', 'succeeded')], null, 'succeeded'),
    overview: () => Response.json(overview()),
    // The words are in both agents' transcripts too, so the switcher keeps both pages offered.
    search: (text) => searchPage([
      {role: 'assistant', content: 'The cap is 6%.', agent: 'main', snippet: 'the cap'},
      {role: 'assistant', content: 'The child looked too.', agent: 'a', snippet: 'looked'},
    ], {terms: [text], total: 2}),
    stepSearch: (text) => stepSearchPage([
      match({marks: ['failed']}),
      // A child row the roster cannot name: the fallback copy, never the Session's id.
      match({agent: 'a', agent_label: null, kind: 'say', name: null, excerpt: 'The child looked too.'}),
    ], {matched: 2, terms: [text]}),
  });
  const dock = await mount(sourceFor().source, 900);
  await waitFor(() => rows(dock).length === 1);

  const box = dock.querySelector('input[type="search"]') as HTMLInputElement;
  box.value = 'cap';
  box.dispatchEvent(new Event('input', {bubbles: true}));
  await new Promise((resolve) => setTimeout(resolve, 260));
  await waitFor(() => dock.querySelectorAll('[data-search-step]').length === 2);

  const panel = dock.querySelector<HTMLElement>('[data-step-search]')!;
  expect(panel.querySelector('[data-search-count]')!.textContent).to.contain('2 matching steps');
  const stepRows = [...panel.querySelectorAll<HTMLElement>('[data-search-step]')];
  expect(stepRows[0]!.textContent).to.contain('Reading a document');
  expect(stepRows[0]!.textContent).to.contain('northwind-msa.pdf');
  expect(stepRows[0]!.textContent).to.contain('The cap is 6% of the remaining fees.');
  // The Run's decisions are technical detail: the search row states no mark word or badge, while
  // the Run-quality details still carry them.
  expect(stepRows[0]!.querySelector('[data-run-mark]'), 'no developer badge in a search row').to.equal(null);
  expect(stepRows[0]!.textContent ?? '', 'a marked step reads without its mark').to.not.contain('Failed');
  expect(stepRows[1]!.textContent).to.contain('Child agent');

  // A row opens the agent that stated the step.
  stepRows[1]!.click();
  await waitFor(() => dock.querySelector('[data-agent-session="a"][aria-current="true"]') !== null);
  expect(dock.textContent ?? '').to.contain('The child looked too.');

  // Clearing the words restores the dossier, whose quality details still state the Run's decisions.
  dock.querySelector<HTMLButtonElement>('button[class*="searchClear"]')!.click();
  await waitFor(() => dock.querySelector('dl-run-overview [data-run-overview]') !== null);
  dock.querySelector<HTMLButtonElement>('[data-quality-toggle]')!.click();
  await dock.updateComplete;
  expect(dock.querySelector<HTMLElement>('[data-attention-counts]')!.textContent).to.contain('Failed 1');
});

it('loads the earlier matches from the cursor the page stated', async () => {
  const served = serve({
    page: () => roster([row('a', 'succeeded')], null, 'succeeded'),
    overview: () => Response.json(overview()),
    stepSearch: (_text, before) => before === null
      ? stepSearchPage([match({sequence: 4, excerpt: 'a newer match'})], {next_before: 'cursor-1', matched: 2})
      : stepSearchPage([match({sequence: 2, excerpt: 'an older match'})], {matched: 2}),
  });
  const dock = await mount(sourceFor().source, 900);
  await waitFor(() => rows(dock).length === 1);
  const box = dock.querySelector('input[type="search"]') as HTMLInputElement;
  box.value = 'match';
  box.dispatchEvent(new Event('input', {bubbles: true}));
  await new Promise((resolve) => setTimeout(resolve, 260));
  await waitFor(() => dock.querySelectorAll('[data-search-step]').length === 1);

  dock.querySelector<HTMLButtonElement>('[data-search-older]')!.click();
  await waitFor(() => dock.querySelectorAll('[data-search-step]').length === 2);
  const texts = [...dock.querySelectorAll<HTMLElement>('[data-search-step]')].map((one) => one.textContent);
  expect(texts[0]).to.contain('an older match');
  expect(texts[1]).to.contain('a newer match');
  const searches = served.filter((request) => request.path.endsWith('/steps/search'));
  expect(searches).to.have.length(2);
  expect(searches[1]!.search).to.contain('before=cursor-1');
});

it('says when no step holds the words, and when the search could not be read', async () => {
  serve({
    page: () => roster([row('a', 'succeeded')], null, 'succeeded'),
    overview: () => Response.json(overview()),
    transcript: () => activityPage([{role: 'assistant', content: 'the whole transcript'}]),
  });
  const first = await mount(sourceFor().source, 900);
  await waitFor(() => rows(first).length === 1);
  const box = first.querySelector('input[type="search"]') as HTMLInputElement;
  box.value = 'nothing holds this';
  box.dispatchEvent(new Event('input', {bubbles: true}));
  await new Promise((resolve) => setTimeout(resolve, 260));
  await waitFor(() => first.querySelector('[data-search-no-match]') !== null);

  document.body.replaceChildren();
  serve({
    page: () => roster([row('a', 'succeeded')], null, 'succeeded'),
    overview: () => Response.json(overview()),
    stepSearch: () => new Response('unavailable', {status: 503}),
  });
  const failed = await mount(sourceFor().source, 900);
  await waitFor(() => rows(failed).length === 1);
  const second = failed.querySelector('input[type="search"]') as HTMLInputElement;
  second.value = 'cap';
  second.dispatchEvent(new Event('input', {bubbles: true}));
  await new Promise((resolve) => setTimeout(resolve, 260));
  await waitFor(() => failed.querySelector('[data-search-failed]') !== null);
  expect(failed.querySelector('[data-search-failed]')!.textContent).to.contain('Retry');
});

it('clearing the words restores the Run overview the dock holds', async () => {
  serve({
    page: () => roster([row('a', 'succeeded')], null, 'succeeded'),
    overview: () => Response.json(overview()),
    transcript: () => activityPage([{role: 'assistant', content: 'the whole transcript'}]),
    stepSearch: () => stepSearchPage([match()]),
  });
  const dock = await mount(sourceFor().source, 900);
  await waitFor(() => dock.querySelector('dl-run-overview [data-run-overview]') !== null);
  const box = dock.querySelector('input[type="search"]') as HTMLInputElement;
  box.value = 'cap';
  box.dispatchEvent(new Event('input', {bubbles: true}));
  await new Promise((resolve) => setTimeout(resolve, 260));
  await waitFor(() => dock.querySelector('[data-step-search]') !== null);
  expect(dock.querySelector('dl-run-overview'), 'the step list took the panel').to.equal(null);

  dock.querySelector<HTMLButtonElement>('button[class*="searchClear"]')!.click();
  await waitFor(() => dock.querySelector('dl-run-overview [data-run-overview]') !== null);
  expect(dock.querySelector('[data-step-search]')).to.equal(null);
});

it('shows the matching steps above the page in a narrow dock, not folded', async () => {
  serve({
    page: () => roster([row('a', 'succeeded')], null, 'succeeded'),
    search: (text) => searchPage([
      {role: 'assistant', content: 'The cap is 6%.', agent: 'main', snippet: 'the cap'},
    ], {terms: [text], total: 1}),
    stepSearch: () => stepSearchPage([match({sequence: 3})], {matched: 1}),
  });
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 1);
  const box = dock.querySelector('input[type="search"]') as HTMLInputElement;
  box.value = 'cap';
  box.dispatchEvent(new Event('input', {bubbles: true}));
  await new Promise((resolve) => setTimeout(resolve, 260));
  await waitFor(() => dock.querySelectorAll('[data-search-step]').length === 1);

  const panel = dock.querySelector<HTMLElement>('[data-step-search]')!;
  expect(panel.tagName).to.equal('SECTION');
  const switcher = dock.querySelector<HTMLElement>('[data-switcher-current]')!;
  expect(panel.getBoundingClientRect().bottom).to.be.at.most(switcher.getBoundingClientRect().top + 0.5);
});

it('drops the words and the step list when the Run changes', async () => {
  serve({
    page: () => roster([row('a', 'succeeded')], null, 'succeeded'),
    search: (text) => searchPage([
      {role: 'assistant', content: 'The cap is 6%.', agent: 'main', snippet: 'the cap'},
    ], {terms: [text], total: 1}),
    stepSearch: () => stepSearchPage([match()], {matched: 1}),
  });
  const dock = await mount(sourceFor().source, 900);
  await waitFor(() => rows(dock).length === 1);
  const box = dock.querySelector('input[type="search"]') as HTMLInputElement;
  box.value = 'cap';
  box.dispatchEvent(new Event('input', {bubbles: true}));
  await new Promise((resolve) => setTimeout(resolve, 260));
  await waitFor(() => dock.querySelector('[data-step-search]') !== null);

  dock.source = sourceFor('run-2').source;
  await dock.updateComplete;
  await waitFor(() => dock.querySelector('[data-step-search]') === null);
  expect((dock.querySelector('input[type="search"]') as HTMLInputElement).value).to.equal('');
});
