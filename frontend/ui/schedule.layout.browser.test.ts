// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** What the shipped stylesheets do with the Scheduled row and the Schedule dialogs: the row stands on the nav
 * column with New chat and Settings, the Schedules dialog's rows are at least a control high and its body scrolls
 * inside the dialog with the footer kept, and the form's controls share one line and one height. */

import {expect} from '@esm-bundle/chai';
import {sendMouse, setViewport} from '@web/test-runner-commands';
import {waitFor} from '../testing/dom.ts';
import {
  choose,
  field,
  listOpen,
  mountSidebar,
  newFormOpen,
  rowsShown,
  serving,
  useWorkspaces,
  wireSchedule,
} from '../testing/schedules.ts';
import dialogStyles from '../styles/schedule-dialogs.module.css';

const originalFetch = window.fetch;

afterEach(() => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
  document.body.className = '';
});

const LONG = 'Quarterly review of the supplier portal, its delivery commitments and the open contract disputes';

it('stands the Scheduled row on the column of New chat and Settings, pill and icon alike, as tall as they are, with its count at the end', async () => {
  await setViewport({width: 1400, height: 900});
  serving({
    'GET /web/api/conversations': () => Response.json({items: [{
      conversation_id: 'c1', title: 'Research notes', created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z',
    }], next_cursor: null}),
  }, [wireSchedule('Weekly review'), wireSchedule('Filing watch')]);
  const column = document.createElement('div');
  column.style.cssText = 'height: 640px; position: relative; width: 280px;';
  document.body.append(column);
  const {sidebar, handles} = await mountSidebar(column);
  await handles.conversations.loadList();
  await waitFor(() => sidebar.querySelector('#scheduled-btn [data-count]') !== null);
  await waitFor(() => sidebar.querySelector('dl-conversation-list .conversation-select') !== null);
  const scheduled = sidebar.querySelector<HTMLElement>('#scheduled-btn')!;

  const textLeft = (element: Element): number => {
    const range = document.createRange();
    range.selectNodeContents(element);
    return range.getBoundingClientRect().left;
  };
  // The pills start on one edge, and a row's control starts on its tint's edge, not inside it.
  const conversationRow = sidebar.querySelector('dl-conversation-list .conversation-row')!.getBoundingClientRect().left;
  for (const [name, left] of [
    ['New chat', sidebar.querySelector('#new-conversation-btn')!.getBoundingClientRect().left],
    ['Scheduled', scheduled.getBoundingClientRect().left],
    ['Settings', sidebar.querySelector('#settings-btn')!.getBoundingClientRect().left],
    ['a conversation\'s control', sidebar.querySelector('dl-conversation-list .conversation-select')!.getBoundingClientRect().left],
  ] as const) {
    expect(left, `${name} starts at a conversation row's edge`).to.be.closeTo(conversationRow, 0.5);
  }
  // It and Settings are the column's two nav rows, so they end on one edge too, and stand one control high.
  expect(scheduled.getBoundingClientRect().right, 'Scheduled ends where Settings does')
    .to.be.closeTo(sidebar.querySelector('#settings-btn')!.getBoundingClientRect().right, 0.5);
  expect(scheduled.getBoundingClientRect().height, 'as tall as Settings')
    .to.be.closeTo(sidebar.querySelector('#settings-btn')!.getBoundingClientRect().height, 0.5);

  const title = textLeft(sidebar.querySelector('dl-conversation-list .conversation-select')!);
  for (const [name, left] of [
    ['New chat icon', sidebar.querySelector('#new-conversation-btn .dl-icon')!.getBoundingClientRect().left],
    ['Scheduled icon', scheduled.querySelector('.dl-icon')!.getBoundingClientRect().left],
    ['Settings icon', sidebar.querySelector('#settings-btn .dl-icon')!.getBoundingClientRect().left],
    ['Recents label', textLeft(sidebar.querySelector('.dl-nav-group-label')!)],
  ] as const) {
    expect(left, `${name} starts where a conversation's title does`).to.be.closeTo(title, 0.5);
  }

  // The count stands inside the pill at its end, and the marks are drawn, not text glyphs.
  const count = scheduled.querySelector('[data-count]')!.getBoundingClientRect();
  expect(count.right).to.be.at.most(scheduled.getBoundingClientRect().right + 0.5);
  expect(count.left).to.be.greaterThan(textLeft(scheduled.querySelector('.dl-nav-item-label')!));
  expect(sidebar.textContent).not.to.contain('•••');
});

it('draws the Schedules dialog\'s rows at least a control high, one to a line, with a long name wrapping inside the row', async () => {
  await setViewport({width: 1400, height: 900});
  serving({}, [wireSchedule(LONG), wireSchedule('Filing watch', {state: 'paused', next_fire_at: null}), wireSchedule('Launch')]);
  const {sidebar} = await mountSidebar();
  const dialog = await listOpen(sidebar);
  const rows = await rowsShown(sidebar, 3);
  const control = dialog.querySelector('.dl-dialog-actions button')!.getBoundingClientRect().height;
  const list = dialog.querySelector('.dl-list-card')!.getBoundingClientRect();

  expect(dialog.getBoundingClientRect().width, 'a dialog of about 560px, as the detail is').to.be.closeTo(560, 1);
  for (const row of rows) {
    const box = row.getBoundingClientRect();
    expect(box.height, 'at least the control height').to.be.at.least(control - 0.5);
    expect(box.width, 'a row fills the list').to.be.closeTo(list.width - 2, 1.5);
    expect(row.querySelector('button')!.scrollWidth, 'nothing spills out of the row').to.be.at.most(row.clientWidth + 1);
  }
  // A hairline between rows, none above the first or under the last, and the list clips its rows' corners.
  expect(rows.map((row) => getComputedStyle(row).borderTopWidth)).to.deep.equal(['0px', '1px', '1px']);
  expect(getComputedStyle(dialog.querySelector('.dl-list-card')!).overflow).to.equal('clip');
  // The name is the first line and the cadence the second, under it.
  const name = rows[0]!.querySelector('[data-name]')!.getBoundingClientRect();
  const meta = rows[0]!.querySelector('[data-meta]')!.getBoundingClientRect();
  expect(meta.top).to.be.at.least(name.bottom - 0.5);
  expect(rows[0]!.getBoundingClientRect().height, 'a long name makes its row taller, not wider').to.be.greaterThan(rows[2]!.getBoundingClientRect().height);
  // Hover tints the whole row, and the keyboard's ring is drawn on it.
  const button = rows[1]!.querySelector('button')!;
  const box = button.getBoundingClientRect();
  expect(getComputedStyle(button).backgroundColor).to.equal('rgba(0, 0, 0, 0)');
  await sendMouse({type: 'move', position: [Math.round(box.left + box.width / 2), Math.round(box.top + box.height / 2)]});
  await waitFor(() => getComputedStyle(button).backgroundColor !== 'rgba(0, 0, 0, 0)');
  button.focus();
  expect(getComputedStyle(button).outlineStyle).to.equal('solid');
});

it('scrolls a long list inside the dialog and keeps the footer in place, within the screen', async () => {
  await setViewport({width: 1400, height: 520});
  serving({}, Array.from({length: 30}, (_, index) => wireSchedule(`Schedule number ${index + 1}`)));
  const {sidebar} = await mountSidebar();
  const dialog = await listOpen(sidebar);
  await rowsShown(sidebar, 30);
  const body = dialog.querySelector<HTMLElement>('[data-body]')!;
  const footer = dialog.querySelector<HTMLElement>('.dl-dialog-actions')!;
  const box = dialog.getBoundingClientRect();

  expect(box.top).to.be.at.least(0);
  expect(box.bottom).to.be.at.most(window.innerHeight);
  expect(getComputedStyle(body).overflowY).to.equal('auto');
  expect(body.scrollHeight, 'the list is taller than its window').to.be.greaterThan(body.clientHeight);
  expect(body.scrollWidth).to.be.at.most(body.clientWidth + 1);
  const before = footer.getBoundingClientRect();
  expect(before.bottom).to.be.at.most(box.bottom + 0.5);
  body.scrollTop = body.scrollHeight;
  expect(body.scrollTop, 'the body is what scrolled').to.be.greaterThan(0);
  expect(footer.getBoundingClientRect().top, 'the footer did not move').to.be.closeTo(before.top, 0.5);
  expect(dialog.scrollTop).to.equal(0);
});

it('fits a phone: the dialog stays inside the screen, its two footer buttons within reach and apart, and nothing scrolls sideways', async () => {
  await setViewport({width: 360, height: 640});
  serving({}, [wireSchedule(LONG), wireSchedule('Filing watch', {state: 'paused', next_fire_at: null})]);
  const {sidebar} = await mountSidebar();
  const dialog = await listOpen(sidebar);
  await rowsShown(sidebar, 2);

  const box = dialog.getBoundingClientRect();
  expect(box.left).to.be.at.least(0);
  expect(box.right).to.be.at.most(window.innerWidth);
  expect(box.bottom).to.be.at.most(window.innerHeight);
  const body = dialog.querySelector<HTMLElement>('[data-body]')!;
  expect(body.scrollWidth).to.be.at.most(body.clientWidth + 1);
  const [close, add] = [...dialog.querySelectorAll<HTMLElement>('.dl-dialog-actions button')].map((button) => button.getBoundingClientRect());
  expect(add!.right).to.be.at.most(window.innerWidth);
  expect(close!.right, 'the two do not overlap').to.be.at.most(add!.left);
});

it('lays the frequency as a row of chips, and the controls it needs on a line under it at the one control height', async () => {
  await setViewport({width: 1400, height: 900});
  serving({'POST /web/api/schedules/preview': () => Response.json({zone: 'UTC', slots: []})}, []);
  const {sidebar, handles} = await mountSidebar();
  useWorkspaces(handles, ['alpha'], ['alpha']);
  const dialog = await newFormOpen(sidebar);
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
  const dialog = await newFormOpen(sidebar);
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
