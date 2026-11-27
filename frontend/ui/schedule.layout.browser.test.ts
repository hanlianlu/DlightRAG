// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** What the shipped stylesheets do with the Scheduled section and its dialogs: the rows are as dense as a
 * conversation's and hold their words in a narrow sidebar, and the form's controls share one line and one height. */

import {expect} from '@esm-bundle/chai';
import {setViewport} from '@web/test-runner-commands';
import {waitFor} from '../testing/dom.ts';
import {
  choose,
  field,
  formOpen,
  mountSidebar,
  rowsShown,
  serving,
  useWorkspaces,
  wireSchedule,
} from '../testing/schedules.ts';
import dialogStyles from '../styles/schedule-dialogs.module.css';
import styles from '../styles/schedule-section.module.css';

const originalFetch = window.fetch;

afterEach(() => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
  document.body.className = '';
  window.localStorage.removeItem('dlightrag.schedule_section_open');
});

const LONG = 'Quarterly review of the supplier portal, its delivery commitments and the open contract disputes';

it('draws a Schedule as dense as a conversation row, with its words kept inside a narrow column and the quick actions clear of them', async () => {
  await setViewport({width: 1400, height: 900});
  serving({
    'GET /web/api/conversations': () => Response.json({items: [{
      conversation_id: 'c1', title: 'Research notes', created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z',
    }], next_cursor: null}),
  }, [wireSchedule(LONG), wireSchedule('Filing watch', {state: 'paused', next_fire_at: null})]);
  const column = document.createElement('div');
  column.style.cssText = 'height: 640px; position: relative; width: 240px;';
  document.body.append(column);
  const {sidebar, handles} = await mountSidebar(column);
  await handles.conversations.loadList();
  const rows = await rowsShown(sidebar, 2);
  await waitFor(() => sidebar.querySelector('dl-conversation-list .conversation-row') !== null);
  const conversation = sidebar.querySelector<HTMLElement>('dl-conversation-list .conversation-row')!;

  for (const row of rows) {
    expect(Math.abs(row.getBoundingClientRect().height - conversation.getBoundingClientRect().height), 'as tall as a conversation row').to.be.below(6);
    expect(row.scrollWidth, 'nothing spills out of the row').to.be.at.most(row.clientWidth + 1);
  }
  const name = rows[0]!.querySelector<HTMLElement>(`.${styles.name}`)!;
  expect(getComputedStyle(name).textOverflow).to.equal('ellipsis');
  expect(name.scrollWidth, 'a long name ends in an ellipsis').to.be.greaterThan(name.clientWidth);
  expect(rows[0]!.querySelector(`.${styles.meta}`)!.getBoundingClientRect().right).to.be.at.most(rows[0]!.getBoundingClientRect().right + 1);

  // Focus brings the quick actions in over the row's end, and the words give them room.
  const select = rows[0]!.querySelector<HTMLButtonElement>('button.conversation-select')!;
  select.focus();
  await waitFor(() => getComputedStyle(rows[0]!.querySelector(`.${styles.actions}`)!).opacity === '1');
  const actions = rows[0]!.querySelector(`.${styles.actions}`)!.getBoundingClientRect();
  expect(name.getBoundingClientRect().right).to.be.at.most(actions.left + 1);
  expect(actions.right).to.be.at.most(rows[0]!.getBoundingClientRect().right + 1);
});

it('lays the frequency as a row of chips, and the controls it needs on a line under it at the one control height', async () => {
  await setViewport({width: 1400, height: 900});
  serving({'POST /web/api/schedules/preview': () => Response.json({zone: 'UTC', slots: []})}, []);
  const {sidebar, handles} = await mountSidebar();
  useWorkspaces(handles, ['alpha'], ['alpha']);
  const add = sidebar.querySelector<HTMLButtonElement>('[data-new]')!;
  await waitFor(() => !add.disabled);
  add.click();
  const dialog = await formOpen(sidebar);
  const plain = field(dialog, 'name').getBoundingClientRect().height;
  const chips = field<HTMLElement>(dialog, 'frequency');

  expect(dialog.getBoundingClientRect().width, 'a dialog of about 560px').to.be.closeTo(560, 1);
  const row = [...chips.querySelectorAll<HTMLElement>('.dl-chip')].map((chip) => chip.getBoundingClientRect());
  expect(row, 'one chip for each frequency').to.have.length(7);
  expect(row.every((box) => Math.abs(box.top - row[0]!.top) < 0.5), 'the chips share one row').to.equal(true);
  for (const frequency of ['once', 'hourly', 'daily', 'weekdays', 'weekly', 'monthly']) {
    choose(chips, frequency);
    await dialog.closest('dl-schedule-form')!.updateComplete;
    // Weekly's days come between the frequency and its time, so the line reads Weekly, Mon, at 09:00.
    const line = frequency === 'weekly' ? chips.nextElementSibling!.nextElementSibling! : chips.nextElementSibling!;
    expect(line.className, `${frequency}: its controls stand on a line of their own`).to.equal(dialogStyles.line);
    const controls = [...line.querySelectorAll<HTMLElement>('input')].map((control) => control.getBoundingClientRect());
    expect(controls.length, frequency).to.be.greaterThan(0);
    for (const box of controls) {
      expect(box.height, `${frequency}: one control height`).to.be.closeTo(plain, 0.5);
      expect(box.top, `${frequency}: one line`).to.be.closeTo(controls[0]!.top, 0.5);
      expect(box.top, `${frequency}: under the chips`).to.be.at.least(chips.getBoundingClientRect().bottom - 0.5);
    }
  }
  choose(chips, 'custom');
  await dialog.closest('dl-schedule-form')!.updateComplete;
  expect(dialog.querySelector(`.${dialogStyles.line}`), 'a frequency with no controls leaves no empty line').to.equal(null);
});

it('fits a phone: the dialog stays inside the screen, its footer within reach, and nothing scrolls sideways', async () => {
  await setViewport({width: 360, height: 640});
  serving({'POST /web/api/schedules/preview': () => Response.json({zone: 'UTC', slots: []})}, []);
  const {sidebar, handles} = await mountSidebar();
  useWorkspaces(handles, ['alpha', 'beta'], ['alpha']);
  const add = sidebar.querySelector<HTMLButtonElement>('[data-new]')!;
  await waitFor(() => !add.disabled);
  add.click();
  const dialog = await formOpen(sidebar);
  choose(field<HTMLElement>(dialog, 'frequency'), 'monthly');
  dialog.querySelector<HTMLButtonElement>('[data-advanced]')!.click();
  await waitFor(() => field(dialog, 'ceiling') !== null);

  const box = dialog.getBoundingClientRect();
  expect(box.left).to.be.at.least(0);
  expect(box.right).to.be.at.most(window.innerWidth);
  expect(box.bottom).to.be.at.most(window.innerHeight);
  const body = dialog.querySelector<HTMLElement>('form > div')!;
  expect(body.scrollWidth).to.be.at.most(body.clientWidth + 1);
  const submit = dialog.querySelector<HTMLElement>('button[type=submit]')!.getBoundingClientRect();
  expect(submit.bottom).to.be.at.most(window.innerHeight);
  expect(submit.right).to.be.at.most(window.innerWidth);
});
