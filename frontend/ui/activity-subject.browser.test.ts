// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** A historical trace step carries the bounded subject the call reported, as plain text beside its verb. */

import {expect} from '@esm-bundle/chai';
import {waitFor} from '../testing/dom.ts';
import {activityPage, answeredTurn, roster, row, serve, sourceFor} from '../testing/traces.ts';
import {mount, restoreDockFixtures, rowFor, rows, session} from '../testing/traces-dock.ts';
import './inspector-traces.ts';

restoreDockFixtures();

it('draws the bounded tool subject beside the verb, escaped, and still shows the result excerpt', async () => {
  serve({
    page: () => roster([row('a', 'succeeded')], null, 'succeeded'),
    transcript: async () => activityPage([
      {
        role: 'assistant',
        content: '',
        tool_calls: [
          {id: 't1', name: 'search_web', subject: 'volvo <b>2026</b> tariff exposure'},
        ],
      },
      {role: 'tool', tool_call_id: 't1', name: 'search_web', content: '3 results', is_error: false},
    ]),
  });
  const dock = await mount(sourceFor('run-1', answeredTurn({state: 'succeeded'})).source as never, 900);
  await waitFor(() => rows(dock).length === 1);
  rowFor(dock, 'a').click();
  await waitFor(() => (session(dock)?.textContent ?? '').includes('volvo'));

  const step = session(dock)!.querySelector('dl-activity li')!;
  const subject = step.querySelector('[class*="subject"]');
  expect(subject?.textContent, 'the Tool-reported subject is on the step').to.equal(
    'volvo <b>2026</b> tariff exposure',
  );
  expect((session(dock)?.textContent ?? '').includes('3 results'), 'the result excerpt is unchanged').to.equal(true);
  expect(subject?.querySelector('b'), 'the subject is plain text, never HTML').to.equal(null);
});
