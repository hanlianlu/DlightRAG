// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The notice under an answer as the shipped stylesheets lay it out (the product-styles page of the runner):
 *  one more part of the action row in the row's own words and divider, a small popover beside it on a desktop,
 *  and a sheet from the bottom edge, with a finger's targets, on a phone. */

import {expect} from '@esm-bundle/chai';
import {emulateMedia, setViewport} from '@web/test-runner-commands';
import type {RunMemoryChange} from '../api/memory.ts';
import {defineDesignSystemElements, type DlSplitLayout} from '../design-system/index.ts';
import {setLanguagePreference} from '../i18n/locale.ts';
import type {ChatTurnView} from '../lib/chat-views.ts';
import {type AppHandles, createAppHandles} from '../stores/app-handles.ts';
import {MemoryNotices} from '../stores/memory-notices.ts';
import {pressArea, seriousViolations} from '../testing/axe.ts';
import {waitFor} from '../testing/dom.ts';
import './chat-message-list.ts';
import type {DlChatMessageList} from './chat-message-list.ts';
import './memory-notice.ts';

defineDesignSystemElements();

const originalViewport = {width: window.innerWidth, height: window.innerHeight};

const change = (
  id: string,
  action: RunMemoryChange['action'] = 'add',
  kind: RunMemoryChange['kind'] = 'preference',
  body = `The owner likes answers that open with a table, number ${id}.`,
): RunMemoryChange => ({
  memoryId: `memory-${id}`, kind, body, action, changeId: `change-${id}`, writer: 'conversation',
});

const CHANGES = [
  change('1', 'add', 'preference'),
  change('2', 'update', 'fact', 'Lives in Munich, and works from home on most days of the week.'),
  change('3', 'fuse', 'experience', 'Charts of more than five series read better as small multiples.'),
];

function turn(index: number): ChatTurnView {
  return {
    id: `turn-${index}`,
    userText: `Question ${index}`,
    userAttachments: [],
    runId: `run-${index}`,
    state: 'succeeded',
    streamText: '',
    processEntries: [],
    presentation: {
      answerText: 'Answer.',
      parts: [{
        type: 'markdown', text: 'Answer.', html: `<p>Answer ${index}.</p>`, artifact: null, evidenceImage: null, inline: false,
      }],
      sources: [], evidenceImages: [], linkCards: [], artifacts: [],
      artifactOutcome: {status: 'complete', issues: []},
    },
    usage: {usage_details: {total_tokens: 9400}},
    error: null,
    activity: null,
    cancelRequested: false,
    steeringMessages: [],
    toolRows: [],
    answerSequence: 7,
    feedback: null,
  };
}

/** The chat list with `earlier` answers before the last, which has left `changes`, in the app container that
 *  sizes and clips every product surface. */
async function answered(
  changes: readonly RunMemoryChange[],
  earlier = 0,
  sidebar: number | null = null,
): Promise<DlChatMessageList> {
  const handles: AppHandles = {
    ...createAppHandles(),
    memoryNotices: new MemoryNotices({read: async () => changes}),
  };
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  list.handles = handles;
  list.answerFeedbackEnabled = true;
  list.turns = Array.from({length: earlier + 1}, (_, index) => turn(index));
  const app = document.createElement('div');
  app.className = 'app';
  if (sidebar === null) {
    app.append(list);
  } else {
    // The conversations beside the chat, in the split the shell uses: the chat's panel is what is left.
    const split = document.createElement('dl-split-layout') as DlSplitLayout;
    split.className = 'panel-split';
    split.setAttribute('primary', 'start');
    split.setAttribute('orientation', 'horizontal');
    split.setAttribute('min', '0');
    split.setAttribute('label', 'Resize conversations');
    split.setAttribute('size', String(sidebar));
    const conversations = document.createElement('div');
    conversations.slot = 'start';
    conversations.id = 'conversations-standin';
    const panel = document.createElement('div');
    panel.slot = 'end';
    panel.style.minWidth = '0';
    panel.append(list);
    split.append(conversations, panel);
    app.append(split);
  }
  document.body.append(app);
  await list.updateComplete;
  await waitFor(() => list.querySelector('dl-memory-notice [aria-haspopup]') !== null);
  return list;
}

const trigger = (list: DlChatMessageList): HTMLButtonElement => (
  [...list.querySelectorAll<HTMLButtonElement>('dl-memory-notice [aria-haspopup]')].at(-1)!
);
const popover = (list: DlChatMessageList): HTMLElement => (
  [...list.querySelectorAll<HTMLElement>('dl-memory-notice [role="dialog"]')].at(-1)!
);
/** The row's Fork, whatever language names it: the second button of the last action row. */
const fork = (list: DlChatMessageList): HTMLButtonElement => (
  [...list.querySelectorAll<HTMLElement>('.dl-action-row')].at(-1)!.querySelectorAll<HTMLButtonElement>(':scope > button')[1]!
);

/** Axe reads the colour a word is painted in, so it reads it once the part has finished fading in. */
function finishAnimations(): void {
  for (const animation of document.getAnimations()) animation.finish();
}

async function open(list: DlChatMessageList): Promise<void> {
  trigger(list).click();
  await waitFor(() => !popover(list).hidden);
  // The popover places itself once it is open and has a size to measure.
  await list.querySelector('dl-memory-notice')?.updateComplete;
}

afterEach(async () => {
  document.body.replaceChildren();
  document.documentElement.dataset.colorMode = 'dark';
  await setViewport(originalViewport);
  await emulateMedia({reducedMotion: 'no-preference'});
});

describe('the part in the action row', () => {
  it('is pressed at 24 CSS pixels at least on a desktop, and the product\'s hit target on a phone', async () => {
    await setViewport({width: 1280, height: 900});
    const desktop = await answered(CHANGES);
    expect(pressArea(trigger(desktop)).height).to.be.at.least(24 - 0.5);
    expect(pressArea(trigger(desktop)).width).to.be.at.least(24 - 0.5);
    document.body.replaceChildren();

    await setViewport({width: 390, height: 844});
    const probe = document.createElement('div');
    probe.style.height = 'var(--control-hit-target)';
    document.body.append(probe);
    const target = probe.getBoundingClientRect().height;
    probe.remove();
    const phone = await answered(CHANGES);

    expect(target).to.be.greaterThan(40);
    expect(pressArea(trigger(phone)).height).to.be.at.least(target - 0.5);
  });

  it('fades in when it arrives, over the overlay duration', async () => {
    await setViewport({width: 1280, height: 900});
    const list = await answered(CHANGES);
    const part = trigger(list).closest<HTMLElement>('[data-action-part]')!;

    // It came after the list was drawn, so it fades in rather than appearing.
    expect(part.hasAttribute('data-fresh')).to.equal(true);
    expect(getComputedStyle(part).animationName).to.not.equal('none');
    expect(Number.parseFloat(getComputedStyle(part).animationDuration)).to.be.closeTo(0.2, 0.001);
  });

  it('does not animate for a reader who asked for no motion', async () => {
    await emulateMedia({reducedMotion: 'reduce'});
    await setViewport({width: 1280, height: 900});
    const list = await answered(CHANGES);
    const part = trigger(list).closest<HTMLElement>('[data-action-part]')!;

    expect(part.hasAttribute('data-fresh')).to.equal(true);
    expect(Number.parseFloat(getComputedStyle(part).animationDuration)).to.equal(0);
  });
});

describe('the popover on a desktop', () => {
  it('opens below the part inside the viewport, no wider than a small card, starting where the part starts', async () => {
    await setViewport({width: 1280, height: 900});
    const list = await answered(CHANGES);
    await open(list);

    const box = popover(list).getBoundingClientRect();
    const part = trigger(list).getBoundingClientRect();
    expect(getComputedStyle(popover(list)).position).to.equal('absolute');
    expect(popover(list).getAttribute('data-placement')).to.equal('below');
    expect(box.top, 'below the part').to.be.at.least(part.bottom - 0.5);
    expect(box.width).to.be.at.most(22 * 16 + 1);
    expect(box.left).to.be.at.least(0);
    expect(box.right).to.be.at.most(window.innerWidth);
    expect(box.left, 'it starts where the part starts').to.be.closeTo(part.left, 2);
    expect(box.bottom).to.be.at.most(window.innerHeight);
  });

  it('opens above when the answer is at the foot of the chat and there is no room below it', async () => {
    await setViewport({width: 1280, height: 440});
    const list = await answered(CHANGES, 6);
    const area = list.querySelector<HTMLElement>('#chat-area')!;
    area.scrollTop = area.scrollHeight;
    await open(list);

    const box = popover(list).getBoundingClientRect();
    const part = trigger(list).getBoundingClientRect();
    const room = area.getBoundingClientRect();
    expect(popover(list).getAttribute('data-placement')).to.equal('above');
    expect(box.bottom, 'above the part').to.be.at.most(part.top + 0.5);
    expect(box.top, 'inside the chat').to.be.at.least(room.top - 0.5);
    expect(box.left, 'it starts where the part starts').to.be.closeTo(part.left, 2);
  });

  it('keeps each press at least 24 CSS pixels each way', async () => {
    await setViewport({width: 1280, height: 900});
    const list = await answered(CHANGES, 4);
    await open(list);

    const small = [...popover(list).querySelectorAll<HTMLElement>('button')]
      .map((button) => ({name: button.textContent!.trim(), area: pressArea(button)}))
      .filter(({area}) => area.width < 24 - 0.5 || area.height < 24 - 0.5);
    expect(small.map(({name, area}) => `${name}: ${area.width} by ${area.height}`)).to.deep.equal([]);
  });

  it('has no serious axe violations open, in either color mode, with a change taken back', async () => {
    await setViewport({width: 1280, height: 900});
    const list = await answered(CHANGES, 4);
    const handles = (list.querySelector('dl-memory-notice') as HTMLElement & {handles: AppHandles}).handles;
    handles.memoryNotices.markUndone('run-4', 'change-2');
    await open(list);
    finishAnimations();

    expect(await seriousViolations(popover(list).closest<HTMLElement>('.dl-action-row')!)).to.deep.equal([]);
  });
});

/** The part is on the same line as Fork, or on a line of its own when the row has wrapped. */
const wrapped = (list: DlChatMessageList): boolean => (
  trigger(list).getBoundingClientRect().top - fork(list).getBoundingClientRect().top > 2
);

describe('the popover inside the chat panel, with the conversations open beside it', () => {
  /** Every edge of the open popover is inside the chat's box, and none of it is over the conversations. */
  async function expectInside(list: DlChatMessageList, label: string): Promise<void> {
    await open(list);
    const area = list.querySelector<HTMLElement>('#chat-area')!.getBoundingClientRect();
    const box = popover(list).getBoundingClientRect();
    const sidebar = document.querySelector('#conversations-standin')!.getBoundingClientRect();
    expect(box.left, `${label}: left edge`).to.be.at.least(area.left - 0.5);
    expect(box.right, `${label}: right edge`).to.be.at.most(area.right + 0.5);
    expect(box.left, `${label}: clear of the conversations`).to.be.at.least(sidebar.right - 0.5);
    expect(box.top, `${label}: top edge`).to.be.at.least(area.top - 0.5);
    expect(box.bottom, `${label}: bottom edge`).to.be.at.most(area.bottom + 0.5);
    // Every word of it is readable: nothing the popover holds is cut off by the box.
    for (const text of popover(list).querySelectorAll<HTMLElement>('p, button, span')) {
      const each = text.getBoundingClientRect();
      expect(each.left, `${label}: ${text.textContent?.trim()}`).to.be.at.least(area.left - 0.5);
      expect(each.right, `${label}: ${text.textContent?.trim()}`).to.be.at.most(area.right + 0.5);
    }
  }

  for (const language of ['en', 'zh'] as const) {
    afterEach(async () => { await setLanguagePreference('en'); });

    it(`never leaves the chat's box, from a narrow window to a wide one (${language})`, async () => {
      await setLanguagePreference(language);
      for (const width of [641, 800, 1024, 1280, 1920]) {
        await setViewport({width, height: 900});
        const list = await answered(CHANGES, 1, 280);
        await expectInside(list, `${width}px wide`);
        document.body.replaceChildren();
      }
    });

    it(`never leaves the chat's box wherever the part sits, at the end of a line or on a wrapped one (${language})`, async () => {
      await setLanguagePreference(language);
      await setViewport({width: 1280, height: 900});
      const seen = {wrapped: 0, farRight: 0, narrow: 0};
      // The Inspector or the conversations take the width the chat gives up: sweep the chat's width.
      for (let sidebar = 600; sidebar <= 960; sidebar += 30) {
        const list = await answered(CHANGES, 1, sidebar);
        const area = list.querySelector<HTMLElement>('#chat-area')!.getBoundingClientRect();
        const part = trigger(list).getBoundingClientRect();
        if (wrapped(list)) seen.wrapped += 1;
        if (part.right > area.right - area.width * 0.4) seen.farRight += 1;
        if (area.width < 22 * 16 + 16 + 40) seen.narrow += 1;
        await expectInside(list, `a chat ${Math.round(area.width)}px wide`);
        document.body.replaceChildren();
      }
      // The sweep does reach the cases it is named for.
      expect(seen.wrapped, 'a wrapped row was met').to.be.greaterThan(0);
      expect(seen.farRight, 'a part at the far right was met').to.be.greaterThan(0);
      expect(seen.narrow, 'a chat narrower than the popover\'s own width was met').to.be.greaterThan(0);
    });
  }

  it('stays inside the chat and keeps the card readable when the window is resized under it', async () => {
    await setViewport({width: 1280, height: 900});
    const list = await answered(CHANGES, 1, 280);
    await open(list);
    const split = document.querySelector<DlSplitLayout>('dl-split-layout')!;

    split.size = 700;
    window.dispatchEvent(new Event('resize'));
    await list.querySelector('dl-memory-notice')?.updateComplete;

    const area = list.querySelector<HTMLElement>('#chat-area')!.getBoundingClientRect();
    const box = popover(list).getBoundingClientRect();
    expect(box.left).to.be.at.least(area.left - 0.5);
    expect(box.right).to.be.at.most(area.right + 0.5);
  });
});

describe('the popover on a phone', () => {
  it('is a sheet from the bottom edge, as wide as the screen, and a finger\'s targets throughout', async () => {
    await setViewport({width: 390, height: 844});
    const list = await answered(CHANGES, 2);
    await open(list);

    const probe = document.createElement('div');
    probe.style.height = 'var(--control-hit-target)';
    document.body.append(probe);
    const target = probe.getBoundingClientRect().height;
    probe.remove();

    const box = popover(list).getBoundingClientRect();
    expect(getComputedStyle(popover(list)).position).to.equal('fixed');
    expect(box.left).to.be.closeTo(0, 0.5);
    expect(box.right).to.be.closeTo(window.innerWidth, 0.5);
    expect(box.bottom).to.be.closeTo(window.innerHeight, 0.5);
    expect(box.height).to.be.at.most(window.innerHeight * 0.7 + 1);
    const short = [...popover(list).querySelectorAll<HTMLElement>('button')]
      .map((button) => ({name: button.textContent!.trim(), area: pressArea(button)}))
      .filter(({area}) => area.height < target - 0.5 || area.width < target - 0.5);
    expect(short.map(({name, area}) => `${name}: ${area.width} by ${area.height}`)).to.deep.equal([]);
  });

  it('wraps long words inside the sheet and has no serious axe violations in either color mode', async () => {
    await setViewport({width: 390, height: 844});
    const long = [change('9', 'add', 'fact', `A${'very '.repeat(30)}long statement with a_link_like_${'x'.repeat(60)}.`)];
    const list = await answered([...CHANGES, ...long], 2);
    await open(list);

    finishAnimations();
    const box = popover(list).getBoundingClientRect();
    for (const text of popover(list).querySelectorAll<HTMLElement>('p')) {
      expect(text.getBoundingClientRect().right).to.be.at.most(box.right + 0.5);
      expect(text.scrollWidth).to.be.at.most(text.clientWidth + 1);
    }
    expect(await seriousViolations(popover(list))).to.deep.equal([]);
  });
});
