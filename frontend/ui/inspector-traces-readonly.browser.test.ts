// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The Inspector reads. It has no composer, no steer/continue/cancel/reply box, and a child's
 *  question to its parent is a state a reader can see, never a place to answer it. */

import {expect} from '@esm-bundle/chai';
import {ago, NOW, observation, question, roster, row, serve, sourceFor} from '../testing/traces.ts';
import {mount, openChild, restoreDockFixtures, rows, session, shown} from '../testing/traces-dock.ts';
import {waitFor} from '../testing/dom.ts';
import './inspector-traces.ts';

restoreDockFixtures();

/** Every control that would write something, as the Inspector must never offer one. */
const WRITING_CONTROLS = [
  'textarea',
  // Search and filter inputs narrow what is shown; they write nothing, so they are not writing controls.
  'input:not([type="search"])',
  'select',
  'form',
  '[contenteditable="true"]',
  'button[type="submit"]',
  '[data-draft]',
  '[data-reply]',
  '[data-cancel]',
  '[data-send]',
];

it('offers no box, form, or command control on either layout', async () => {
  const running = row('a', 'running', {started_at: ago(5)});
  const settled = row('b', 'succeeded', {started_at: ago(9), finished_at: ago(1), summary: 'Done.'});
  serve({page: () => roster([running, settled])});
  for (const width of [420, 900]) {
    const dock = await mount(sourceFor().source, width);
    await waitFor(() => rows(dock).length === 2);
    expect(dock.querySelector('[data-switcher-filter]'), 'the list filter reads, it does not write')
      .to.not.equal(null);
    for (const id of ['a', 'b']) {
      await openChild(dock, id);
      await dock.updateComplete;
      const writing = [...dock.querySelectorAll<HTMLElement>(WRITING_CONTROLS.join(','))];
      expect(
        writing.map((element) => `${element.tagName}:${element.className}`),
        `no writing control on ${id} at ${width}px`,
      ).to.deep.equal([]);
      for (const word of ['Steer', 'Continue', 'Cancel child', 'Answer instead', 'Reauthorize', 'Send']) {
        expect([...dock.querySelectorAll('button')].some((button) => button.textContent?.trim() === word), `${word} at ${width}px`)
          .to.equal(false);
        expect(shown(dock)).to.not.contain(word);
      }
    }
    dock.remove();
  }
});

it('states a child\'s question to its parent read-only, with who it asks and when it lapses', async () => {
  Date.now = () => NOW;
  const running = row('a', 'running', {started_at: ago(5), pending_questions: 1});
  serve({
    page: () => roster([running]),
    observe: () => observation(running, {questions: [question('q1')]}),
  });
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');
  await waitFor(() => shown(dock).includes('Question q1?'));

  const text = shown(dock);
  expect(text).to.contain('Asking the parent · expires in 4 min');
  expect(text).to.contain('Question q1?');
  expect(text).to.not.contain('Answer instead');
  expect(session(dock).querySelectorAll('textarea, form, [data-reply]')).to.have.length(0);
});

it('keeps the instructions it received as the read-only record it is', async () => {
  const running = row('a', 'running', {started_at: ago(5)});
  serve({
    page: () => roster([running]),
    observe: () => observation(running, {
      controls: [{
        control_sequence: 1, kind: 'steer', content: 'Look at the amendment', origin: 'user',
        consumed: true, consumed_at: ago(3), created_at: ago(4), operation_id: 'op-a',
      }],
    }),
  });
  const dock = await mount(sourceFor().source, 420);
  await waitFor(() => rows(dock).length === 1);
  await openChild(dock, 'a');
  await waitFor(() => shown(dock).includes('Instructions received'));

  const text = shown(dock);
  expect(text).to.contain('Instructions received · 1');
  // The sender leads the meta line: who spoke, then what became of it.
  expect(text).to.contain('You · Handled');
  expect(text).to.contain('Look at the amendment');
  expect(text, 'the admin words are gone').to.not.contain('Consumed');
  expect(text).to.not.contain('Control history');
  expect(session(dock).querySelectorAll('form, textarea, button[type="submit"]')).to.have.length(0);
});
