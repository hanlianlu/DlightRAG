// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** What the dock shows of an agent: its question, steps, evidence and result, and how the page stands when the
 *  roster or the server changes under it. */

import {expect} from '@esm-bundle/chai';
import type {AnswerPresentation} from '../api/conversations.ts';
import {activityPage, ago, answeredTurn, NOW, observation, question, roster, row, serve, sourceFor} from '../testing/traces.ts';
import {button, deferred, immediateFollowRefreshes, mount, openChild, restoreDockFixtures, rowFor, rows, session, settle, shown, stated, steps, title} from '../testing/traces-dock.ts';
import {waitFor} from '../testing/dom.ts';
import './inspector-traces.ts';

restoreDockFixtures();

const transcriptRequests = (requests: {path: string; search: string}[]) => requests
  .filter((request) => request.path.endsWith('/transcript'))
  .map((request) => request.search);

it('shows a running child\'s question with when it expires, and its activity as the main trace words it', async () => {
  Date.now = () => NOW;
  serve({
    page: () => roster([row('a', 'running', {started_at: ago(134), pending_questions: 1})]),
    observe: () => observation(row('a', 'running', {started_at: ago(134), pending_questions: 1}), {
      transcript: [
        {role: 'user', content: 'objective a'},
        {role: 'assistant', content: '', tool_calls: [{id: 't1', name: 'search_knowledge_base'}]},
        {role: 'tool', content: '12 results\nTop: MSA §9.2', tool_call_id: 't1', name: 'search_knowledge_base', is_error: false},
        {role: 'assistant', content: 'Checking the amendment.', tool_calls: [{id: 't2', name: 'read'}]},
      ],
      questions: [
        question('q1'),
        question('q0', {status: 'replied', reply: 'Swedish law.', reply_origin: 'parent', expires_at: null}),
        question('q9', {expires_at: ago(5)}),
      ],
    }),
  });
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');
  await waitFor(() => shown(dock).includes('Checking the amendment'));

  const text = shown(dock);
  expect(text).to.contain('Running 2m 14s');
  expect(text).to.contain('Asking the parent · expires in 4 min Question q1?');
  expect(text.match(/Asking the parent/g), 'a question past its expiry is no longer waiting').to.have.length(1);
  expect(text).to.contain('Agent timeline Searching the knowledge base 12 results');
  expect(text).to.contain('Checking the amendment. Running: Reading a document');
  expect(session(dock).querySelector('details')!.open, 'a running child\'s activity is open').to.equal(true);
  // Quiet history: what was answered, and what lapsed.
  expect(text).to.contain('Question q0? Answered · Parent Swedish law.');
  expect(text).to.contain('Question q9? Expired');
});

// ── Opening ──

it('mounts the agent page at once, so its transcript reads while the roster is still loading', async () => {
  // A reader coming back to a child: the dock remembers the agent before the roster can name it.
  window.localStorage.setItem('dlightrag-inspector-agent:run-1', 'a');
  const page = deferred<Response>();
  const requests = serve({
    page: () => page.promise,
    observe: () => new Promise<Response>(() => {}),
    transcript: (agent) => agent === 'a'
      ? activityPage([{role: 'assistant', content: 'Reading already.', tool_calls: []}], {running: true})
      : undefined,
  });
  const dock = await mount(sourceFor().source, 420);

  expect(session(dock), 'the page is on show before the roster has answered').to.not.equal(null);
  expect(title(dock), 'the page names the agent it is waiting for by its role').to.equal('Child agent');
  await waitFor(() => transcriptRequests(requests).length === 1);
  expect(transcriptRequests(requests)[0], 'the child\'s transcript is read in parallel with the roster').to.equal('?child=a');

  page.resolve(roster([row('a', 'running', {objective: 'Read the filing', started_at: ago(5)})], null, 'running'));
  await waitFor(() => title(dock) === 'Read the filing');
  await waitFor(() => shown(dock).includes('Reading already.'));
});

it('times each tool step on its own line, from the call to the result that settled it', async () => {
  const running = row('a', 'running', {started_at: ago(134)});
  serve({
    page: () => roster([running]),
    observe: () => observation(running, {
      transcript: [
        {role: 'user', content: 'objective a', at: ago(200)},
        {role: 'assistant', content: 'Looking.', at: ago(130), tool_calls: [{id: 't1', name: 'search_knowledge_base'}]},
        {role: 'tool', content: '12 results', tool_call_id: 't1', name: 'search_knowledge_base', is_error: false, at: ago(127.5)},
        {role: 'assistant', content: 'Checking.', tool_calls: [{id: 't2', name: 'read'}]},
        {role: 'tool', content: 'ok', tool_call_id: 't2', name: 'read', is_error: false, at: ago(1)},
      ],
    }),
  });
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');
  await waitFor(() => shown(dock).includes('Checking.'));

  const text = shown(dock);
  expect(text).to.contain('Searching the knowledge base 2.5s 12 results');
  // A step whose call the wire gave no clock for states no time at all.
  expect(text).to.contain('Checking. Reading a document ok');
});

it('says who sent each steer, even when a short one is part of an older one', async () => {
  const running = row('a', 'running', {started_at: ago(5)});
  const control = (sequence: number, content: string, origin: string) => ({
    control_sequence: sequence, kind: 'steer', content, origin, consumed: true,
    consumed_at: ago(3), created_at: ago(4), operation_id: 'op-a',
  });
  serve({
    page: () => roster([running]),
    observe: () => observation(running, {
      transcript: [
        {role: 'user', content: 'objective a'},
        {role: 'user', content: 'Parent steer: Continue with the amendment review'},
        {role: 'user', content: 'User steer: Continue'},
      ],
      // Newest first, as the server lists them.
      controls: [control(2, 'Continue', 'user'), control(1, 'Continue with the amendment review', 'parent')],
    }),
  });
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');
  await waitFor(() => shown(dock).includes('Continue with the amendment review'));

  expect(shown(dock)).to.contain('Agent timeline Parent Continue with the amendment review You Continue');
});

it('shows a settled child\'s own work and evidence, without saying its summary twice', async () => {
  const settled = row('a', 'succeeded', {
    started_at: ago(68), finished_at: ago(0), usage: {total_tokens: 9400}, summary: 'Clause 9.2 caps it at 6%.',
    result_handles: ['ev-1', 'ev-2'],
  });
  serve({
    page: () => roster([settled]),
    observe: () => observation(settled, {
      transcript: [
        {role: 'user', content: 'objective a'},
        {role: 'assistant', content: '', tool_calls: [{id: 't1', name: 'read'}]},
        {role: 'tool', content: 'ok', tool_call_id: 't1', name: 'read', is_error: false},
      ],
      result: {status: 'succeeded', summary: 'Clause 9.2 caps it at 6%.', handles: ['ev-1', 'ev-2'], operation_id: 'op-a'},
    }),
  });
  Date.now = () => NOW;
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');
  await waitFor(() => shown(dock).includes('Agent timeline · 1 step'));

  const text = shown(dock);
  const headings = [...session(dock).querySelectorAll('h4')].map((one) => one.textContent?.trim());
  expect(
    headings.includes('Result'),
    'a child\'s summary is its last message and its row already says it: not a third time here',
  ).to.equal(false);
  expect(text).to.contain('Done 1m 8s · 9.4K tokens');
  expect(rows(dock)[0]?.text, 'the row is where a settled child\'s summary belongs').to.contain('Clause 9.2 caps it at 6%.');
  expect(text).to.contain('Evidence · 2');
  const [evidence, activity] = session(dock).querySelectorAll('details');
  expect(evidence!.open).to.equal(false);
  expect(activity!.open, 'a settled child\'s activity is folded').to.equal(false);
  expect(evidence!.textContent).to.contain('ev-1');
  expect(button(session(dock), 'Cancel child'), 'nothing is left to cancel').to.equal(null);
});

it('shows a child\'s newest steps first and the earlier ones on request', async () => {
  const running = row('a', 'running', {started_at: ago(5)});
  const requests = serve({
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

  expect(shown(dock)).to.not.contain('Step 20.');
  session(dock).querySelector<HTMLButtonElement>('[data-load-older="activity"]')!.click();
  await waitFor(() => shown(dock).includes('Step 1.Step 2.'));

  expect(session(dock).querySelector('[data-load-older="activity"]'), 'the start has been reached').to.equal(null);
  expect(transcriptRequests(requests)).to.deep.equal(['', '?child=a', '?child=a&before=21']);
});

it('keeps the earlier steps a reader has shown when the Run moves on again', async () => {
  immediateFollowRefreshes();
  let newest = steps(21, 40);
  let cursor: number | null = 21;
  const running = row('a', 'running', {started_at: ago(5)});
  serve({
    page: () => roster([running]),
    observe: () => observation(running),
    transcript: (agent, before) => {
      if (agent !== 'a') return undefined;
      return before === null
        ? activityPage(newest, {nextBefore: cursor, running: true})
        : activityPage(steps(1, 20), {running: true});
    },
  });
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');
  await waitFor(() => shown(dock).includes('Step 40.'));
  session(dock).querySelector<HTMLButtonElement>('[data-load-older="activity"]')!.click();
  await waitFor(() => shown(dock).includes('Step 1.Step 2.'));

  newest = steps(22, 41);
  cursor = 22;
  stated(dock, 'run-1');
  await waitFor(() => shown(dock).includes('Step 41.'));
  expect(shown(dock)).to.contain('Step 1.Step 2.');
  expect(shown(dock).match(/Step 25\./g), 'a step is shown once').to.have.length(1);

  // What landed is appended to what the reader has, whichever step the Run moved on to: a delta
  // read starts where the pane left off, so nothing it holds gives way.
  newest = steps(100, 119);
  cursor = 100;
  stated(dock, 'run-1');
  await waitFor(() => shown(dock).includes('Step 119.'));
  expect(shown(dock), 'the steps the reader had stay').to.contain('Step 1.Step 2.');
});

it('shows the main agent\'s token count, and repeats neither the answer nor the sources the chat holds', async () => {
  serve({page: () => roster([], null, 'succeeded')});
  const presentation = {
    answerText: 'Clause 9.2 caps the penalty at 6%.', parts: [], linkCards: [], evidenceImages: [], artifacts: [],
    artifactOutcome: null,
    sources: [
      {id: '1', title: 'northwind-msa.pdf', sourceUrl: null, downloadUrl: null, chunks: []},
      {id: '2', title: 'amendment-3.pdf', sourceUrl: null, downloadUrl: null, chunks: []},
    ],
  } as unknown as AnswerPresentation;
  const turn = answeredTurn({presentation, usage: {usage_details: {total_tokens: 9400}}});
  const dock = await mount(sourceFor('run-1', turn).source, 800);
  await waitFor(() => shown(dock).includes('Main agent'));
  await waitFor(() => shown(dock).includes('Final answer'));

  const text = shown(dock);
  expect(text).to.contain('Main agent What changed? Show full objective Done 9.4K tokens');
  expect(
    text.includes('Result') || text.includes('Evidence'),
    'the chat shows the answer and the sources it cites; this pane stands beside that chat',
  ).to.equal(false);
  // The answer is here, but folded like a tool call: shut, it takes one line and the steps read.
  expect(text).to.contain('Final answer');
  const answer = [...session(dock).querySelectorAll('details')].find(
    (candidate) => candidate.textContent?.includes('Final answer'),
  )!;
  expect(answer.open, 'and it stays shut until a reader opens it').to.equal(false);
  expect(answer.textContent).to.contain('Clause 9.2 caps the penalty at 6%.');
});

it('says why a Run failed in the main agent\'s result, and shows no Result for a Run that is still going', async () => {
  serve({page: () => roster([], null, 'failed')});
  const failed = await mount(sourceFor('run-1', answeredTurn({state: 'failed', error: {source: 'Service error. Please try again.', id: 'errors.service'}})).source, 800);
  await waitFor(() => shown(failed).includes('Failed'));
  expect(shown(failed)).to.contain('Result Service error. Please try again.');
  failed.remove();

  serve({page: () => roster([], null, 'running')});
  const going = await mount(sourceFor('run-1', answeredTurn({state: 'streaming', error: {source: 'ignored', id: null}})).source, 800);
  await waitFor(() => shown(going).includes('Running'));
  expect(shown(going)).to.not.contain('Result');
});

it('opens a child\'s Evidence line on the answer\'s source it cites, and leaves a line the answer does not hold as text', async () => {
  const settled = row('a', 'succeeded', {
    started_at: ago(68), finished_at: ago(0), summary: 'Clause 9.2 caps it.',
    result_handles: ['[1] northwind-msa.pdf', '[7] not-in-the-answer.pdf'],
  });
  serve({page: () => roster([settled]), observe: () => observation(settled)});
  const presentation = {
    answerText: '', parts: [], linkCards: [], evidenceImages: [], artifacts: [], artifactOutcome: null,
    sources: [{id: '1', title: 'northwind-msa.pdf', sourceUrl: null, downloadUrl: null, chunks: []}],
  } as unknown as AnswerPresentation;
  const dock = await mount(sourceFor('run-1', answeredTurn({presentation})).source, 420);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');
  await waitFor(() => shown(dock).includes('Evidence · 2'));

  const links = [...session(dock).querySelectorAll<HTMLElement>('[data-evidence-ref]')].map((link) => link.dataset.evidenceRef);
  expect(links).to.deep.equal(['1']);
  expect(shown(dock)).to.contain('[7] not-in-the-answer.pdf');
});

it('shows a token count only when the child reports a finite total', async () => {
  Date.now = () => NOW;
  const read = async (usage: Record<string, number> | null) => {
    const settled = row('a', 'succeeded', {started_at: ago(10), finished_at: ago(0), usage});
    serve({page: () => roster([settled]), observe: () => observation(settled)});
    const dock = await mount(sourceFor().source, 420);
    await waitFor(() => rows(dock).length === 1);
    await openChild(dock, 'a');
    await waitFor(() => shown(dock).includes('Done'));
    const text = shown(dock);
    dock.remove();
    return text;
  };

  expect(await read({input_tokens: 5, output_tokens: 7})).to.not.contain('tokens');
  expect(await read(null)).to.not.contain('tokens');
  expect(await read({total_tokens: 12})).to.contain('Done 10s · 12 tokens');
});

// ── Commands ──

it('reads a running child again after each roster refresh, and a settled one only when its row changed', async () => {
  immediateFollowRefreshes();
  const stateOf = {a: row('a', 'running', {started_at: ago(5)}), b: row('b', 'succeeded', {started_at: ago(9), finished_at: ago(1)})};
  const requests = serve({
    page: () => roster([stateOf.a, stateOf.b]),
    observe: (id) => observation(id === 'a' ? stateOf.a : stateOf.b),
  });
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 2);
  const observed = (id: string) => requests.filter((request) => request.path.endsWith(`/children/${id}`)).length;
  const refreshed = async () => {
    const before = requests.filter((request) => request.path.endsWith('/children')).length;
    stated(dock, 'run-1');
    await waitFor(() => requests.filter((request) => request.path.endsWith('/children')).length > before);
    await settle(40);
  };

  await openChild(dock, 'a');
  await waitFor(() => observed('a') === 1);
  await refreshed();
  expect(observed('a'), 'a running child\'s transcript grows without its row changing').to.equal(2);

  rowFor(dock, 'main').click();
  await dock.updateComplete;
  await openChild(dock, 'b');
  await waitFor(() => observed('b') === 1);
  await refreshed();
  expect(observed('b'), 'a settled child that has not changed is not read again').to.equal(1);

  stateOf.b = row('b', 'succeeded', {started_at: ago(9), finished_at: ago(1), summary: 'A late summary.'});
  await refreshed();
  expect(observed('b'), 'a settled child whose row changed is').to.equal(2);
});

it('says so when the child on show is no longer on the roster', async () => {
  immediateFollowRefreshes();
  let children = [row('a', 'running'), row('b', 'running')];
  serve({page: () => roster(children), observe: (id) => observation(row(id, 'running'))});
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 2);
  await openChild(dock, 'a');

  children = [row('b', 'running')];
  stated(dock, 'run-1');
  await waitFor(() => shown(dock).includes('That child is no longer available.'));

  expect(shown(dock)).to.contain('Pick another from the list.');
  expect(session(dock).querySelector(':scope > [role="status"]')!.textContent!.trim()).to.equal('That child is no longer available.');
  expect(session(dock).querySelector('form')).to.equal(null);
  rowFor(dock, 'main').click();
  await dock.updateComplete;
  expect(rows(dock).map((child) => child.id)).to.deep.equal(['b']);
});

it('says so when the server no longer knows the child it was asked about', async () => {
  serve({
    page: () => roster([row('a', 'running')]),
    observe: () => Response.json({detail: 'Answer child not found'}, {status: 404}),
  });
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 1);
  rowFor(dock, 'a').click();
  await waitFor(() => shown(dock).includes('That child is no longer available.'));
});

it('keeps what is on screen when a later read of the child fails, and offers a retry when the first does', async () => {
  immediateFollowRefreshes();
  let fail = true;
  let reads = 0;
  serve({
    page: () => roster([row('a', 'running', {started_at: ago(5)})]),
    observe: () => {
      reads += 1;
      return fail ? new Response('unavailable', {status: 503}) : observation(row('a', 'running', {started_at: ago(5)}));
    },
    // The activity is read apart from the child's details, and is on screen whichever way they go.
    transcript: (agent) => agent === 'a'
      ? activityPage([{role: 'assistant', content: 'Still here.', tool_calls: []}], {running: true})
      : undefined,
  });
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 1);
  rowFor(dock, 'a').click();
  await waitFor(() => session(dock).querySelector('[role="alert"]') !== null);
  expect(session(dock).querySelector('[role="alert"]')!.textContent!.trim()).to.equal('Child details could not be loaded.');

  fail = false;
  button(session(dock), 'Retry')!.click();
  await waitFor(() => shown(dock).includes('Still here.'));

  fail = true;
  const before = reads;
  stated(dock, 'run-1');
  await waitFor(() => reads > before);
  await settle(40);
  expect(shown(dock)).to.contain('Still here.');
  expect(session(dock).querySelector('[role="alert"]')).to.equal(null);
});

it('names a child by the canonical fallback, never its Session id, when it has no objective', async () => {
  const id = '3f2a7c1e-2b44-4e87-9f0d-8c6c1f5b9a10';
  serve({page: () => roster([row(id, 'running', {objective: null})])});
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 1);

  expect(rows(dock)[0]!.text).to.equal('Child agent Running');
  expect(dock.textContent ?? '', 'the Session id never reaches a reader as a name').to.not.contain('3f2a7c1e');
});

it('offers the thumbs on what a settled child answered once the server has said which message that is, and on no other agent', async () => {
  const judged = {value: 'down', reasons: ['off_topic'], note: null};
  const answered = row('a', 'succeeded', {result_sequence: 12, feedback: judged});
  const rows3 = [answered, row('b', 'succeeded'), row('c', 'running', {result_sequence: null})];
  serve({
    page: () => roster(rows3),
    observe: (id) => observation(rows3.find((each) => each.child_session_id === id)!),
  });
  const served = window.fetch;
  const ratings: Record<string, unknown>[] = [];
  window.fetch = async (input, init) => {
    if (!String(input).endsWith('/feedback')) return served(input, init);
    const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
    ratings.push(body);
    return Response.json({run_id: 'run-1', ...body, updated_at: ago(0)});
  };
  const dock = await mount(sourceFor().source, 420);
  dock.answerFeedbackEnabled = true;
  await waitFor(() => rows(dock).length === 3);
  const thumbs = () => [...session(dock).querySelectorAll<HTMLElement>('[data-thumb]')];

  // The Run's own agent is judged under the chat's turn.
  expect(thumbs()).to.have.length(0);
  await openChild(dock, 'b');
  expect(thumbs(), 'a child that has not said which message is its answer').to.have.length(0);
  await openChild(dock, 'c');
  expect(thumbs(), 'a child that is still running').to.have.length(0);

  await openChild(dock, 'a');
  await waitFor(() => thumbs().length === 2);
  const [up, down] = thumbs() as [HTMLButtonElement, HTMLButtonElement];
  const element = session(dock).querySelector('dl-answer-feedback')!;
  expect([element.runId, element.agent, element.sequence, element.layout]).to.deep.equal(['run-1', 'a', 12, 'block']);
  // The child's status row holds it.
  expect(element.parentElement?.classList.contains('status')).to.equal(true);
  expect([up.getAttribute('aria-pressed'), down.getAttribute('aria-pressed')]).to.deep.equal(['false', 'true']);

  up.click();
  await waitFor(() => ratings.length === 1 && up.getAttribute('aria-pressed') === 'true');
  expect(ratings).to.deep.equal([{agent: 'a', sequence: 12, value: 'up', reasons: [], note: null}]);

  // Looking at another child and back reads the child again, and what it says is older than the rating.
  await openChild(dock, 'b');
  expect(thumbs()).to.have.length(0);
  await openChild(dock, 'a');
  await waitFor(() => thumbs().length === 2);
  expect(thumbs().map((each) => each.getAttribute('aria-pressed'))).to.deep.equal(['true', 'false']);

  // Once the page learns the thumbs are off, the child's page stops offering them.
  dock.answerFeedbackEnabled = false;
  await dock.updateComplete;
  await session(dock).updateComplete;
  await waitFor(() => thumbs().length === 0);
});

// ── A child's page, laid out ──

it('folds an agent message\'s reasoning into a Thinking block above the words it spoke', async () => {
  const running = row('a', 'running', {started_at: ago(5)});
  serve({
    page: () => roster([running]),
    observe: () => observation(running, {
      transcript: [
        {role: 'user', content: 'objective a'},
        {
          role: 'assistant',
          content: 'Checking the amendment.',
          reasoning: 'The indemnity clause is the one to read.',
        },
        {role: 'assistant', content: 'No thought on record.'},
      ],
    }),
  });
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');
  await waitFor(() => shown(dock).includes('Checking the amendment.'));

  const says = [...session(dock)!.querySelectorAll<HTMLElement>('[data-step-kind="say"]')];
  expect(says, 'both things the agent said are steps').to.have.length(2);
  const fold = says[0]!.querySelector<HTMLDetailsElement>('details');
  expect(fold, 'the reasoning is folded inside the step').to.not.equal(null);
  expect(fold!.open, 'shut until a reader asks for it').to.equal(false);
  expect(fold!.querySelector('summary')?.textContent?.trim()).to.equal('Thinking');
  expect(fold!.textContent).to.contain('The indemnity clause is the one to read.');
  // The block leads: the words the message spoke come after it.
  const words = [...says[0]!.querySelectorAll('p')]
    .find((one) => one.textContent?.includes('Checking the amendment.'));
  expect(words).to.not.equal(undefined);
  expect(fold!.compareDocumentPosition(words!) & Node.DOCUMENT_POSITION_FOLLOWING).to.be.greaterThan(0);
  // A message that states no reasoning gets no empty fold.
  expect(says[1]!.querySelector('details')).to.equal(null);
});

it('keeps the reasoning of a tool choice and of the final answer in the pane', async () => {
  const settled = row('a', 'succeeded', {
    started_at: ago(68), finished_at: ago(0), summary: 'Clause 9.2 caps it at 6%.',
  });
  serve({
    page: () => roster([settled]),
    observe: () => observation(settled, {
      transcript: [
        {role: 'user', content: 'objective a'},
        // The turn that chose the search: reasoning and a call, but no words of its own.
        {
          role: 'assistant', content: '', reasoning: 'The filing is what to search.',
          tool_calls: [{id: 't1', name: 'search_knowledge_base'}],
        },
        {role: 'tool', content: '12 results', tool_call_id: 't1', name: 'search_knowledge_base', is_error: false},
        // The final answer, whose words the page folds into the conclusion: its reasoning goes too.
        {role: 'assistant', content: 'Clause 9.2 caps it at 6%.', reasoning: 'The cap is the answer.'},
      ],
      result: {status: 'succeeded', summary: 'Clause 9.2 caps it at 6%.', handles: [], operation_id: 'op-a'},
    }),
  });
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');
  await waitFor(() => shown(dock).includes('Clause 9.2 caps it at 6%.'));

  const steps = [...session(dock)!.querySelectorAll<HTMLElement>('[data-activity-pane] ol > li')];
  const thinking = steps.find((step) => step.dataset.stepKind === 'thinking')!;
  expect(thinking, 'the tool choice states the reasoning that chose it').to.not.equal(undefined);
  const thinkingFold = thinking.querySelector<HTMLDetailsElement>('details')!;
  expect(thinkingFold.open, 'shut until a reader asks for it').to.equal(false);
  expect(thinkingFold.querySelector('summary')?.textContent?.trim()).to.equal('Thinking');
  expect(thinkingFold.textContent).to.contain('The filing is what to search.');
  const tool = steps.find((step) => step.dataset.stepKind === 'tool')!;
  expect(
    thinking.compareDocumentPosition(tool) & Node.DOCUMENT_POSITION_FOLLOWING,
    'the thinking comes before the call it explains',
  ).to.be.greaterThan(0);

  const brief = session(dock)!.querySelector<HTMLElement>('[data-step-kind="brief"]')!;
  expect(brief, 'the page ends on the conclusion').to.not.equal(null);
  const briefFold = brief.querySelector<HTMLDetailsElement>(':scope > details')!;
  expect(briefFold, 'the conclusion is the same fold a tool call opens').to.not.equal(null);
  const briefThinking = briefFold.querySelector<HTMLDetailsElement>('details')!;
  expect(briefThinking, 'the conclusion keeps the reasoning that produced it').to.not.equal(null);
  expect(briefThinking.open, 'and keeps it folded').to.equal(false);
  expect(briefThinking.querySelector('summary')?.textContent?.trim()).to.equal('Thinking');
  expect(briefThinking.textContent).to.contain('The cap is the answer.');
  expect(briefFold.textContent).to.contain('Clause 9.2 caps it at 6%.');
});

it('names a pre-delivery warning as a delivery note, never a check', async () => {
  const settled = row('a', 'succeeded', {started_at: ago(9), finished_at: ago(1)});
  serve({
    page: () => roster([settled]),
    observe: () => observation(settled, {
      transcript: [{role: 'delivery', content: 'The record names no verifier.'}],
    }),
  });
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');
  await waitFor(() => shown(dock).includes('The record names no verifier.'));

  expect(shown(dock)).to.contain('Delivery note');
  expect(shown(dock), 'the retired word is gone from the timeline').to.not.contain('Delivery check');
});
