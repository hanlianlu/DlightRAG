// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {setViewport} from '@web/test-runner-commands';
import {type AnswerArtifact, type AnswerPresentation, getArtifactPresentationAt} from '../api/conversations.ts';
import {defineDesignSystemElements} from '../design-system/index.ts';
import type {ChatTurnView} from '../lib/chat-views.ts';
import {productionHandles} from '../stores/app-handles.ts';
import {bootstrapResponse} from '../testing/app.ts';
import {pressArea, seriousViolations} from '../testing/axe.ts';
import {buttonNamed, waitFor} from '../testing/dom.ts';
import {
  detailOpen,
  formOpen,
  listOpen,
  mountSidebar,
  rowButton,
  rowsShown,
  serving as servingSchedules,
  useWorkspaces,
  wireFire,
  wireSchedule,
} from '../testing/schedules.ts';
import {
  agentAccountsView,
  memoryPage,
  memorySettings,
  mountSettings,
  openSettings,
  ownerSkills,
  privacySettings,
  wire,
  wireAccount,
  wireSkill,
} from '../testing/settings.ts';
import {NOW, activityPage, ago, answeredTurn, observation, overview, question, roster, row, serve, sourceFor} from '../testing/traces.ts';
import {DEFAULT_CHANGES} from '../testing/workspaces.ts';
import './artifact-canvas.ts';
import './chat-composer.ts';
import {type DlChatComposer, forgetStoredDraft} from './chat-composer.ts';
import './app.ts';
import type {DlApp} from './app.ts';
import './chat-message-list.ts';
import type {DlChatMessageList} from './chat-message-list.ts';
import './conversation-sidebar.ts';
import './image-lightbox.ts';
import type {DlImageLightbox} from './image-lightbox.ts';
import './inspector.ts';
import type {DlInspector} from './inspector.ts';
import './run-dialogs.ts';
import type {DlContinuationDialog} from './run-dialogs.ts';
import './settings.ts';
import './theme.ts';
import type {DlThemeControl} from './theme.ts';
import './workspace-scope.ts';

defineDesignSystemElements();

const SAFE_PNG =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=';

// Every kind of gold text an answer draws: a link in prose and in a table, a
// citation, an evidence image's Source button, and enough references to fold.
const answerWire = {
  answer_text: 'Retrieval finds passages.',
  parts: [{
    type: 'markdown', text: 'Retrieval finds passages.', artifact: null, evidence_image: null, inline: false,
    html: '<p>Retrieval finds passages, as <a href="https://example.com/survey">the survey</a> shows '
      + '<cite class="citation-badge" data-ref="1" role="button" tabindex="0" aria-label="Reference 1">1</cite>.</p>'
      + '<table><thead><tr><th><a href="https://example.com/methods">Method</a></th></tr></thead>'
      + '<tbody><tr><td><a href="https://example.com/bm25">BM25</a></td></tr>'
      + '<tr><td><a href="https://example.com/dense">Dense</a></td></tr></tbody></table>',
  }],
  sources: Array.from({length: 6}, (_, index) => ({
    id: String(index + 1), title: `Source ${index + 1}`, source_url: null, download_url: null, chunks: [],
  })),
  evidence_images: [{
    id: 'image-1', chunk_id: 'chunk-1', source_ref: '2', url: SAFE_PNG, thumbnail_url: SAFE_PNG,
    label: 'Figure 1', answer_image_sent: true,
  }],
  artifacts: [],
  artifact_outcome: {status: 'complete', issues: []},
};

const workspaceRecords = ['default', 'research'].map((workspace) => ({
  workspace, displayName: workspace, embeddingModel: 'embed', changes: DEFAULT_CHANGES,
}));

/** Answer every fetch with the JSON `route` returns for its URL while `run` runs. */
async function withFetch<T>(route: (url: string) => unknown, run: () => Promise<T>): Promise<T> {
  const originalFetch = window.fetch;
  window.fetch = async (input) => Response.json(route(String(input)));
  try {
    return await run();
  } finally {
    window.fetch = originalFetch;
  }
}

// The product-styles page (web-test-runner.config.mjs) links the application's
// shipped stylesheets, and components render the class names they were built with.
before(async () => {
  const token = getComputedStyle(document.documentElement).getPropertyValue('--color-bg-base');
  expect(token, 'the shipped stylesheets are applied').to.not.equal('');
  // Axe judges only what is on screen, so each fixture renders whole in a tall
  // viewport. The default width keeps the compact shell.
  await setViewport({width: 800, height: 1600});
});

/** Mount `element` in the app container, which sizes and clips every product surface. */
function mountInApp<T extends HTMLElement>(element: T): T {
  const app = document.createElement('div');
  app.className = 'app';
  app.append(element);
  document.body.append(app);
  return element;
}

// Axe walks a whole fixture once for each color mode, and a page whose machine is busy needs longer than the
// default for the larger fixtures, which hold several pages or a whole shell.
beforeEach(function () {
  this.currentTest?.timeout(20_000);
});

afterEach(() => {
  document.body.replaceChildren();
  document.body.className = '';
  document.documentElement.dataset.colorMode = 'dark';
});

it('chat message list has no serious axe violations', async () => {
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  // The list holds the answer as the product parses it off the wire.
  const presentation = await withFetch(() => answerWire, () => getArtifactPresentationAt('/presentation'));
  const turn: ChatTurnView = {
    id: 'turn-a11y',
    userText: 'What is retrieval?',
    userAttachments: [],
    runId: 'run-a11y',
    state: 'succeeded',
    streamText: '',
    processEntries: [],
    presentation,
    usage: {},
    error: null,
    activity: null,
    cancelRequested: false,
    steeringMessages: [],
    answerSequence: null, feedback: null,
    toolRows: [],
  };
  list.turns = [turn];
  mountInApp(list);
  await list.updateComplete;
  await waitFor(() => Boolean(list.querySelector('.citation-badge')));
  expect(await seriousViolations(list)).to.deep.equal([]);
});

it('composer has no serious axe violations with the skill menu open and a skill highlighted', async () => {
  const skills = [
    {name: 'tdd', description: 'Build a feature test first', source: 'global'},
    {name: 'weekly-report', description: 'Summarise the week', source: 'owner'},
    {name: 'z-creator', description: 'Create a skill', source: 'builtin'},
  ];
  await withFetch(() => ({skills}), async () => {
    const composer = mountInApp(document.createElement('dl-chat-composer') as DlChatComposer);
    await composer.updateComplete;
    const input = composer.querySelector<HTMLTextAreaElement>('.composer-input')!;
    input.value = '/';
    input.dispatchEvent(new Event('input', {bubbles: true}));
    await waitFor(() => composer.querySelectorAll('.skill-menu-item').length === skills.length);
    input.dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowDown', bubbles: true, cancelable: true}));
    await composer.updateComplete;

    // The box points at the open menu and at the skill it highlights, and axe judges those references.
    expect(composer.querySelector<HTMLElement>('#skill-menu')!.hidden).to.equal(false);
    expect(input.getAttribute('aria-controls')).to.equal('skill-menu');
    expect(input.getAttribute('aria-activedescendant')).to.equal('skill-option-tdd');
    try {
      expect(await seriousViolations(composer)).to.deep.equal([]);
    } finally {
      // The slash typed above may have been stored while axe ran; no other test should find it in a box.
      forgetStoredDraft();
    }
  });
});

describe('the composer\'s other menus', () => {
  for (const picker of [
    {name: 'answer mode', trigger: '#composer-mode', menu: '#composer-mode-menu'},
    {name: 'agent effort', trigger: '#composer-effort', menu: '#composer-effort-menu'},
  ]) {
    it(`has no serious axe violations with the ${picker.name} menu open`, async () => {
      const composer = mountInApp(document.createElement('dl-chat-composer') as DlChatComposer);
      composer.agentEffortOffer = {levels: ['low', 'high', 'max'], default: 'high'};
      await composer.updateComplete;
      composer.querySelector<HTMLButtonElement>(picker.trigger)!.click();
      await composer.updateComplete;

      expect(composer.querySelector<HTMLElement>(picker.menu)!.hidden).to.equal(false);
      expect(composer.querySelector(picker.trigger)!.getAttribute('aria-expanded')).to.equal('true');
      expect(await seriousViolations(composer)).to.deep.equal([]);
    });
  }
});

it('theme menu has no serious axe violations while open', async () => {
  const control = mountInApp(document.createElement('dl-theme-control') as DlThemeControl);
  await control.updateComplete;
  buttonNamed(control, 'Appearance')!.click();
  await control.updateComplete;

  expect(control.querySelector('[role="menu"]')!.hasAttribute('hidden')).to.equal(false);
  expect(await seriousViolations(control)).to.deep.equal([]);
});

it('image lightbox has no serious axe violations with a gallery open', async () => {
  const trigger = document.createElement('button');
  trigger.textContent = 'Open chart';
  const lightbox = document.createElement('dl-image-lightbox') as DlImageLightbox;
  document.body.append(trigger, lightbox);
  await lightbox.open(SAFE_PNG, trigger, [SAFE_PNG, SAFE_PNG]);
  await lightbox.updateComplete;

  expect(lightbox.getAttribute('aria-hidden')).to.equal('false');
  expect(await seriousViolations(lightbox)).to.deep.equal([]);
});

it('continuation dialog has no serious axe violations while open', async () => {
  const continuation = mountInApp(document.createElement('dl-continuation-dialog') as DlContinuationDialog);
  continuation.open();
  const dialog = await (async () => {
    await waitFor(() => continuation.querySelector<HTMLDialogElement>('dialog')?.open === true);
    return continuation.querySelector<HTMLDialogElement>('dialog')!;
  })();

  expect(await seriousViolations(dialog)).to.deep.equal([]);
});

it('sign-in page has no serious axe violations, with and without its error', async () => {
  // The shipped markup, so the page judged is the page users get.
  const page = new DOMParser().parseFromString(
    await (await fetch(new URL('../login.html', import.meta.url))).text(),
    'text/html',
  );
  document.body.replaceChildren(...[...page.body.childNodes].map((node) => document.importNode(node, true)));
  const main = document.querySelector<HTMLElement>('main')!;
  const clean = await seriousViolations(main);

  const error = main.querySelector<HTMLElement>('.file-error')!;
  error.textContent = 'Authentication failed. Check the token and try again.';
  error.hidden = false;
  expect([...clean, ...await seriousViolations(main)]).to.deep.equal([]);
});

describe('the application shell', () => {
  const widths = [{name: 'phone', width: 390, height: 844}, {name: 'desktop', width: 1440, height: 900}];
  after(async () => { await setViewport({width: 800, height: 1600}); });

  for (const {name, width, height} of widths) {
    it(`has no serious axe violations on a ${name}, ${width}px wide`, async () => {
      await setViewport({width, height});
      const originalFetch = window.fetch;
      window.fetch = async (input) => bootstrapResponse(input, ['a', 'b'].map((id) => ({
        conversation_id: id, title: `Conversation ${id}`,
        created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z',
      })));
      try {
        const app = document.createElement('dl-app') as DlApp;
        document.body.append(app);
        await app.ready;
        await waitFor(() => productionHandles().conversations.listState === 'ready');
        await app.updateComplete;
        expect(await seriousViolations(app)).to.deep.equal([]);
      } finally {
        window.fetch = originalFetch;
      }
    });
  }
});

/** A settled answer under which the reader has rated, and the reasons open on a thumb down they gave, with the
 *  note they gave changed, so the note has its own Send. */
async function ratedAnswerWithReasons({changedNote = true} = {}): Promise<DlChatMessageList> {
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  const presentation = await withFetch(() => answerWire, () => getArtifactPresentationAt('/presentation'));
  list.turns = [{
    id: 'turn-rated',
    userText: 'What is retrieval?',
    userAttachments: [],
    runId: 'run-rated',
    state: 'succeeded',
    streamText: '',
    processEntries: [],
    presentation,
    usage: {usage_details: {total_tokens: 9400}},
    error: null,
    activity: null,
    cancelRequested: false,
    steeringMessages: [],
    toolRows: [],
    answerSequence: 7,
    feedback: {value: 'down', reasons: ['off_topic'], note: 'It answers a different question.'},
  }];
  list.answerFeedbackEnabled = true;
  mountInApp(list);
  await list.updateComplete;
  await waitFor(() => list.querySelector('[data-thumb="down"]') !== null);
  list.querySelector<HTMLButtonElement>('[data-thumb="down"]')!.click();
  await waitFor(() => list.querySelector('[data-reason]') !== null);
  if (changedNote) await changeNote(list);
  return list;
}

/** The reader adds to the note they gave, and the note has something to send. */
async function changeNote(list: DlChatMessageList): Promise<void> {
  const field = list.querySelector<HTMLTextAreaElement>('textarea')!;
  field.value = 'It answers a different question, and cites nothing.';
  field.dispatchEvent(new Event('input', {bubbles: true}));
  await waitFor(() => list.querySelector('[data-send]') !== null);
}

describe('the area a reader can press', () => {
  /** Two 10 by 10 buttons side by side, whose ::before is declared by `pseudo`. */
  function smallPair(pseudo: string): HTMLElement {
    const pair = document.createElement('div');
    pair.innerHTML = `<style>
      .small-pair { display: flex; gap: 2px; }
      .small-pair button { border: 0; height: 10px; padding: 0; position: relative; width: 10px; }
      .small-pair button::before { ${pseudo} }
    </style>
    <button type="button" aria-label="One"></button><button type="button" aria-label="Two"></button>`;
    pair.className = 'small-pair';
    document.body.append(pair);
    return pair;
  }

  const sizeOfFirst = (pair: HTMLElement): number[] => {
    const area = pressArea(pair.querySelector('button')!);
    return [area.width, area.height];
  };

  it('is the control and the invisible extension of it that is positioned and takes the pointer', async () => {
    const pair = smallPair("content: ''; inset: -7px; position: absolute;");

    expect(sizeOfFirst(pair)).to.deep.equal([24, 24]);
    // Two controls that small, so near each other, pass the target size only through the extension.
    expect((await seriousViolations(pair)).filter((found) => found.includes('target-size'))).to.deep.equal([]);
  });

  for (const [what, pseudo] of [
    ['draws nothing', "content: none; inset: -7px; position: absolute;"],
    ['is in the flow of the control', "content: ''; inset: -7px; position: relative;"],
    ['lets the pointer through', "content: ''; inset: -7px; pointer-events: none; position: absolute;"],
  ] as const) {
    it(`is the control alone when its ::before ${what}, so a decorative one does not pass a small target`, async () => {
      const pair = smallPair(pseudo);

      expect(sizeOfFirst(pair)).to.deep.equal([10, 10]);
      expect((await seriousViolations(pair)).some((found) => found.includes('target-size'))).to.equal(true);
    });
  }
});

it('the thumbs under a settled answer, with the reasons open on a chosen one and a note being changed, have no serious axe violations', async () => {
  const list = await ratedAnswerWithReasons();
  // A chosen reason, the others not, a pressed thumb and one that is not, and the note with its Send.
  expect(list.querySelector('[data-reason="off_topic"]')!.getAttribute('aria-pressed')).to.equal('true');
  expect(list.querySelector('[data-reason="unsupported"]')!.getAttribute('aria-pressed')).to.equal('false');
  expect(list.querySelector('[data-send]') !== null, 'the note has its own Send').to.equal(true);
  expect(await seriousViolations(list)).to.deep.equal([]);
});

describe('what the reader presses to judge an answer', () => {
  const originalViewport = {width: window.innerWidth, height: window.innerHeight};
  after(async () => { await setViewport(originalViewport); });

  const pressables = (list: DlChatMessageList) => [
    ...list.querySelectorAll<HTMLElement>('[data-thumb], [data-reason], [data-send]'),
  ].map((button) => ({
    name: button.getAttribute('aria-label') ?? button.textContent!.trim(),
    box: pressArea(button),
  }));

  it('is at least 24 CSS pixels each way on a desktop, with the thumbs in the action row and the reasons under it', async () => {
    await setViewport({width: 1280, height: 1600});
    const list = await ratedAnswerWithReasons();

    const found = pressables(list);
    // Two thumbs, five reasons, and the note's Send.
    expect(found).to.have.length(8);
    const small = found.filter(({box}) => box.width < 24 - 0.5 || box.height < 24 - 0.5);
    expect(small.map(({name, box}) => `${name}: ${box.width} by ${box.height}`)).to.deep.equal([]);

    const fork = [...list.querySelectorAll('button')].find((button) => button.textContent?.trim() === 'Fork')!;
    const up = list.querySelector('[data-thumb="up"]')!.getBoundingClientRect();
    const down = list.querySelector('[data-thumb="down"]')!.getBoundingClientRect();
    const group = list.querySelector('[role="group"]')!.getBoundingClientRect();
    const usage = list.querySelector('[class*="runUsage"]')!.getBoundingClientRect();
    // The thumbs share Fork's line, before the token count on it; the reasons are the line under.
    expect(Math.abs((up.top + up.bottom) / 2 - (fork.getBoundingClientRect().top + fork.getBoundingClientRect().bottom) / 2))
      .to.be.lessThan(3);
    expect(fork.getBoundingClientRect().right).to.be.lessThan(up.left);
    expect(down.right).to.be.lessThan(usage.left);
    expect(group.top).to.be.greaterThan(down.bottom - 0.5);
    expect(Math.abs(usage.top - up.top)).to.be.lessThan(up.height);
  });

  it('wears the row\'s own divider, rather than drawing a second kind of line beside it', async () => {
    await setViewport({width: 1280, height: 1600});
    const list = await ratedAnswerWithReasons();
    const pair = list.querySelector<HTMLElement>('[data-layout="row"]')!;
    const usage = list.querySelector<HTMLElement>('[class*="runUsage"]')!;
    const fork = [...list.querySelectorAll('button')].find((button) => button.textContent?.trim() === 'Fork')!;
    const divider = (element: Element): string => {
      const style = getComputedStyle(element);
      return [
        style.borderInlineStartWidth, style.borderInlineStartStyle,
        style.borderInlineStartColor, style.paddingInlineStart,
      ].join(' ');
    };
    // The thumbs arrive inside an element that is no box of its own, so their pair is hoisted onto the
    // row's line: one rule says what a divider in that row is, and every part of it — the words, the pair,
    // and the count — takes that same line and that same measure. A second line of the feature's own, or
    // a feature rule that zeroed a part's padding, would leave one of the three standing differently,
    // which is how this row's dividers drifted apart once already.
    for (const part of [fork, pair, usage]) {
      expect(getComputedStyle(part).borderInlineStartStyle, `${part.tagName.toLowerCase()} wears a line`)
        .to.not.equal('none');
    }
    const lines = [fork, pair, usage].map(divider);
    expect(new Set(lines).size, `one line and one measure: ${lines.join(' | ')}`).to.equal(1);

    // The row draws the line before every part but the first, and stands each side of it off by the measure
    // it gaps its parts by; the pair's marks are shorter than the words, and its line is as tall as theirs.
    const row = fork.closest<HTMLElement>('.dl-action-row')!;
    const traces = row.firstElementChild!;
    expect(traces.textContent, 'the first part is the row\'s first').to.contain('Agent traces');
    expect(getComputedStyle(traces).borderInlineStartStyle, 'the first part has nothing to part from').to.equal('none');
    expect(getComputedStyle(fork).paddingInlineStart, 'a part stands off the line by the row\'s gap')
      .to.equal(getComputedStyle(row).columnGap);
    expect(pair.getBoundingClientRect().height, 'the pair\'s line is as tall as the words beside it')
      .to.be.closeTo(fork.getBoundingClientRect().height, 0.5);
  });

  it('brings Send into the field\'s corner without moving the field or the words in it', async () => {
    await setViewport({width: 1280, height: 1600});
    const list = await ratedAnswerWithReasons({changedNote: false});
    const field = list.querySelector<HTMLTextAreaElement>('textarea')!;
    const measure = () => {
      const box = (element: Element) => {
        const {top, left, width, height} = element.getBoundingClientRect();
        return {top, left, width, height};
      };
      return {group: box(list.querySelector('[role="group"]')!), field: box(field), words: field.scrollHeight};
    };
    expect(list.querySelector('[data-send]') === null, 'the note as stored has nothing to send').to.equal(true);
    const before = measure();

    await changeNote(list);

    // The room for Send was the field's padding all along, so nothing moved when it came.
    expect(measure()).to.deep.equal(before);
    const send = list.querySelector('[data-send]')!.getBoundingClientRect();
    const box = field.getBoundingClientRect();
    expect(send.right, 'inside the right edge').to.be.at.most(box.right);
    expect(send.bottom, 'inside the bottom edge').to.be.at.most(box.bottom);
    expect(send.left, 'in the right-hand corner').to.be.greaterThan(box.left + box.width * 0.8);
    expect(send.top, 'in the bottom corner').to.be.greaterThan(box.top + box.height * 0.3);
    // The words keep clear of it, and the field no longer offers a grip beneath it.
    expect(Number.parseFloat(getComputedStyle(field).paddingRight)).to.be.at.least(send.width);
    expect(getComputedStyle(field).resize).to.equal('none');
  });

  it('is the product\'s hit target on a phone, and the reasons wrap inside the width', async () => {
    await setViewport({width: 390, height: 844});
    const list = await ratedAnswerWithReasons();
    const probe = document.createElement('div');
    probe.style.height = 'var(--control-hit-target)';
    document.body.append(probe);
    const target = probe.getBoundingClientRect().height;
    probe.remove();

    expect(target, 'the product\'s hit target is known').to.be.greaterThan(40);
    const short = pressables(list).filter(({box}) => box.height < target - 0.5 || box.width < target - 0.5);
    expect(short.map(({name, box}) => `${name}: ${box.width} by ${box.height}`)).to.deep.equal([]);
    // Wrapped, not cut off: every reason is inside the width the chat gives its messages.
    const area = list.querySelector<HTMLElement>('.chat-area')?.getBoundingClientRect() ?? list.getBoundingClientRect();
    const outside = pressables(list).filter(({box}) => box.right > area.right + 0.5 || box.left < area.left - 0.5);
    expect(outside.map(({name}) => name)).to.deep.equal([]);
  });
});

it('a Markdown Artifact in the Canvas has no serious axe violations', async () => {
  const artifact: AnswerArtifact = {
    resourceId: 'report', mediaType: 'text/markdown', label: 'Report', filename: 'report.md',
    byteSize: 20, digest: 'a'.repeat(64), presentation: 'markdown', status: 'available',
    uri: 'dlightrag://answer/run-a11y/artifacts/report', width: null, height: null,
    dataUrl: '/web/api/answer/run-a11y/artifacts/report',
    downloadUrl: '/web/api/answer/run-a11y/artifacts/report?download=1',
    presentationUrl: '/web/api/answer/run-a11y/artifacts/report/presentation', issue: null,
    attestation: null,
  };
  await withFetch(() => answerWire, async () => {
    const canvas = mountInApp(document.createElement('dl-artifact-canvas'));
    canvas.className = 'panel';
    await canvas.open(artifact);
    await waitFor(() => Boolean(canvas.querySelector('.citation-badge')));
    expect(await seriousViolations(canvas)).to.deep.equal([]);
  });
});

it('conversation list has no serious axe violations with older conversations to load', async () => {
  await withFetch(() => ({
    items: ['a', 'b'].map((id) => ({
      conversation_id: id, title: `Conversation ${id}`,
      created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z',
    })),
    next_cursor: 'older',
  }), async () => {
    const sidebar = mountInApp(document.createElement('dl-conversation-sidebar'));
    sidebar.enabled = true;
    await productionHandles().conversations.loadList();
    await sidebar.open();
    // The shell slides the compact drawer in by classing the body.
    document.body.classList.add('conversation-drawer-open');
    await waitFor(() => Boolean(sidebar.querySelector('[data-load-older]')));
    // The list is judged on the sidebar surface; the sidebar's own Settings button
    // controls a dialog only the full shell renders.
    expect(await seriousViolations(sidebar.querySelector('dl-conversation-list')!)).to.deep.equal([]);
  });
});

it('the Scheduled row, the Schedules dialog, a Schedule\'s detail and the form that makes one have no serious axe violations', async () => {
  servingSchedules({
    'GET /web/api/schedules/schedule-weekly-review/fires': () => Response.json({fires: [
      wireFire('f3', '2099-01-12T07:00:00.000000Z', 'succeeded', {conversation_id: 'conversation-3'}),
      wireFire('f2', '2099-01-11T07:00:00.000000Z', 'skipped_busy', {reason: 'its previous run is still working'}),
      wireFire('f1', '2099-01-10T07:00:00.000000Z', 'failed', {reason: 'the model refused'}),
    ], next_cursor: '2099-01-10T07:00:00.000000Z'}),
    'POST /web/api/schedules/preview': () => Response.json({zone: 'Europe/Stockholm', slots: [
      {at: '2099-01-05T07:00:00.000000Z', local: '2099-01-05T08:00:00+01:00'},
    ]}),
  }, [
    wireSchedule('Weekly review', {changed_by: 'agent', prompt: Array.from({length: 7}, (_, index) => `Line ${index + 1} of the prompt.`).join('\n')}),
    wireSchedule('Filing watch', {state: 'paused', pause_reason: 'paused by its owner', next_fire_at: null}),
    wireSchedule('Launch', {state: 'completed', cron: null, at: '2026-12-01T07:00:00Z', next_fire_at: null}),
  ]);
  const app = document.createElement('div');
  app.className = 'app';
  document.body.append(app);
  const {sidebar, handles} = await mountSidebar(app);
  useWorkspaces(handles, ['alpha', 'beta'], ['alpha']);
  // The shell slides the compact drawer in by classing the body.
  document.body.classList.add('conversation-drawer-open');
  const section = sidebar.querySelector('dl-schedule-section')!;
  await waitFor(() => section.querySelector('[data-count]') !== null);
  expect(await seriousViolations(section)).to.deep.equal([]);

  const list = await listOpen(sidebar);
  await rowsShown(sidebar, 3);
  expect(await seriousViolations(list)).to.deep.equal([]);

  rowButton(sidebar, 'schedule-weekly-review').click();
  const detail = await detailOpen(sidebar);
  await waitFor(() => detail.querySelectorAll('[data-fire]').length === 3);
  expect(await seriousViolations(detail)).to.deep.equal([]);
  detail.querySelector<HTMLButtonElement>('[data-action="edit"]')!.click();
  const form = await formOpen(sidebar);
  form.querySelector<HTMLButtonElement>('[data-advanced]')!.click();
  await waitFor(() => form.querySelector('[data-field="ceiling"]') !== null && form.querySelector('[data-next]') !== null);
  // Weekly shows the day chips, and Advanced shows every field.
  form.querySelector<HTMLInputElement>('[data-field="frequency"] input[value="weekly"]')!.click();
  await waitFor(() => form.querySelector('[data-day]') !== null);
  expect(await seriousViolations(form)).to.deep.equal([]);
});

it('the Schedules dialog has no serious axe violations while empty, loading, or when the list could not be read', async () => {
  servingSchedules({}, []);
  const app = document.createElement('div');
  app.className = 'app';
  document.body.append(app);
  const {sidebar} = await mountSidebar(app);
  document.body.classList.add('conversation-drawer-open');
  const list = await listOpen(sidebar);
  await waitFor(() => list.querySelector('[data-empty]') !== null);
  expect(await seriousViolations(list)).to.deep.equal([]);
  list.close();
  await waitFor(() => !list.open);
  document.body.replaceChildren();

  // A list that is still being read: the dialog says so, with New schedule not yet offered.
  servingSchedules({'GET /web/api/schedules': () => new Promise<Response>(() => {})});
  const reading = document.createElement('div');
  reading.className = 'app';
  document.body.append(reading);
  const pending = await mountSidebar(reading);
  const loading = await listOpen(pending.sidebar);
  expect(loading.textContent).to.contain('Loading schedules');
  expect(await seriousViolations(loading)).to.deep.equal([]);
  loading.close();
  await waitFor(() => !loading.open);
  document.body.replaceChildren();

  servingSchedules({'GET /web/api/schedules': () => new Response('unavailable', {status: 503})});
  const failing = document.createElement('div');
  failing.className = 'app';
  document.body.append(failing);
  const broken = await mountSidebar(failing);
  await waitFor(() => broken.handles.schedules.listState === 'error');
  const failed = await listOpen(broken.sidebar);
  await waitFor(() => failed.querySelector('[data-failed]') !== null);
  expect(await seriousViolations(failed)).to.deep.equal([]);
});

it('workspace picker has no serious axe violations while open', async () => {
  const {ingest, workspaces} = productionHandles();
  workspaces.init(workspaceRecords, ['default'], 'default');
  ingest.resetToPrimary();
  const scope = mountInApp(document.createElement('dl-workspace-scope'));
  scope.className = 'workspace-selector';
  await scope.updateComplete;
  scope.querySelector<HTMLButtonElement>('#workspace-trigger')!.click();
  await scope.updateComplete;
  expect(await seriousViolations(scope)).to.deep.equal([]);
});

it('inspector has no serious axe violations when closed', async () => {
  const inspector = document.createElement('dl-inspector') as DlInspector;
  mountInApp(inspector);
  await inspector.updateComplete;
  expect(await seriousViolations(inspector)).to.deep.equal([]);
});

it('inspector Files panel has no serious axe violations', async () => {
  const {ingest, workspaces} = productionHandles();
  workspaces.init(workspaceRecords, ['default'], 'default');
  ingest.resetToPrimary();
  await withFetch((url) => (url.includes('/files/failed')
    ? {workspace: 'default', next_cursor: 'more', failed: [{
      document_id: 'scan', file_name: 'scan.pdf', error: 'Parsing failed', updated_at: '2026-01-01T00:00:00Z',
    }]}
    : {workspace: 'default', next_cursor: null, files: [{file_name: 'notes.md', file_path: 'notes.md'}]}
  ), async () => {
    const inspector = mountInApp(document.createElement('dl-inspector') as DlInspector);
    await inspector.openFiles();
    await waitFor(() => Boolean(inspector.querySelector('#ingest-target-trigger')
      && inspector.querySelector('[data-load-older]')));
    expect(await seriousViolations(inspector)).to.deep.equal([]);
  });
});

// One Run's children as the dock draws them: a running child with a question, a running one, and each
// way a child settles. Elapsed times are drawn against the fixtures' own clock.
const dockChildren = [
  row('c1', 'running', {
    started_at: ago(134), pending_questions: 1,
    objective: 'Make the strongest case for terminating the Northwind supply agreement early',
  }),
  row('c2', 'running', {started_at: ago(112)}),
  row('c3', 'succeeded', {
    started_at: ago(300), finished_at: ago(232), usage: {total_tokens: 9400}, result_handles: ['ev-1', 'ev-2'],
    summary: 'Clause 9.2 caps the penalty at 6% of the remaining fees; notice is 90 days.',
    // The reader has judged what it answered, which the thumbs show once the dock offers them.
    result_sequence: 12, feedback: {value: 'down', reasons: ['too_verbose'], note: null},
  }),
  row('c4', 'failed', {
    started_at: ago(200), finished_at: ago(159), summary: 'The PDF has no text layer, and OCR is not available for this run.',
  }),
  row('c5', 'cancelled', {
    cancellation_origin: 'user', started_at: ago(400), finished_at: ago(388), summary: 'Child session cancelled.',
  }),
];

function dockObservation(id: string): Response {
  const child = dockChildren.find((candidate) => candidate.child_session_id === id)!;
  const call = (callId: string, name: string, content: string, isError = false) => [
    {role: 'assistant', content: '', tool_calls: [{id: callId, name}]},
    {role: 'tool', content, tool_call_id: callId, name, is_error: isError},
  ];
  return observation(child, {
    transcript: [
      {role: 'user', content: String(child.objective)},
      ...call('t1', 'search_knowledge_base', '12 results\nTop: Northwind MSA §9.2, early termination for convenience'),
      ...call('t2', 'read', 'Document not readable: no text layer', true),
      {role: 'assistant', content: 'The MSA allows termination for convenience after month 18.', tool_calls: [{id: 't3', name: 'read'}]},
      {role: 'user', content: 'User steer: focus on clause 9.2 only'},
    ],
    controls: [{
      control_sequence: 1, kind: 'steer', content: 'focus on clause 9.2 only', origin: 'user',
      consumed: true, consumed_at: ago(30), created_at: ago(40), operation_id: `op-${id}`,
    }],
    questions: id === 'c1' ? [
      question('q1'),
      question('q0', {status: 'replied', reply: 'Swedish law.', reply_origin: 'parent', expires_at: null}),
      question('q9', {expires_at: ago(5)}),
    ] : [],
    result: child.status === 'running' ? null : {
      status: child.status, summary: child.summary ?? '', handles: child.result_handles, operation_id: `op-${id}`,
    },
  });
}

/** The dock open on its Run, in a pane `width` wide: narrow shows one of the list and a child, wide both. */
async function openDock(width: number | null, turn?: ChatTurnView) {
  const inspector = mountInApp(document.createElement('dl-inspector') as DlInspector);
  if (width !== null) inspector.parentElement!.style.width = `${width}px`;
  await inspector.openTraces(sourceFor('run-1', turn).source);
  // The main agent heads the list, and the children hang off it.
  await waitFor(() => inspector.querySelectorAll('[data-agent-session]').length === dockChildren.length + 1);
  // The dock measures its pane a frame after it is shown.
  await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  return inspector;
}

/** Run `judge` with the dock's Run on the wire and the clock its fixtures are drawn against. */
async function withDock<T>(judge: () => Promise<T>): Promise<T> {
  const originalNow = Date.now;
  const originalFetch = window.fetch;
  Date.now = () => NOW;
  serve({
    page: () => roster(dockChildren),
    observe: dockObservation,
    // The Run panel is part of the dock's two-column surface: axe judges it beside the pages.
    overview: () => Response.json(overview({
      totals: {
        steps: 18,
        kinds: {thinking: 1, tool: 12, say: 2, answer: 1, instruction: 1, delivery: 1},
        decisions: 3,
        marks: {failed: 1, degraded: 1, repeated: 1, redirect: 0},
        tool_time_ms: 4200,
      },
      agents: [
        {
          agent: 'main', label: 'main', order: 0, steps: 6, tool_calls: 3,
          marks: {failed: 1, degraded: 0, repeated: 1, redirect: 0},
          tool_time_ms: 2000, first_at: '2026-10-05T12:00:00Z', last_at: '2026-10-05T12:01:00Z',
        },
        ...dockChildren.map((child, index) => ({
          agent: String(child.child_session_id), label: String(child.objective ?? ''), order: index + 1,
          steps: 2, tool_calls: 2,
          marks: {failed: 0, degraded: 0, repeated: 0, redirect: 0},
          tool_time_ms: 400, first_at: '2026-10-05T12:00:10Z', last_at: '2026-10-05T12:00:50Z',
        })),
      ],
      findings: [
        {
          agent: 'main', sequence: 3, kind: 'tool', at: '2026-10-05T12:00:20Z',
          name: 'read', subject: 'northwind-msa.pdf', outcome: 'failed', marks: ['failed'],
          excerpt: 'The PDF has no text layer.',
        },
        {
          agent: 'c1', sequence: 5, kind: 'delivery', at: '2026-10-05T12:00:45Z', name: null,
          subject: null, outcome: null, marks: [], excerpt: 'The record names no verifier.',
        },
      ],
      findings_total: 2,
    })),
    // The main agent has a long past: its newest page says there is an older one.
    transcript: (agent) => agent === null
      ? activityPage([
        {role: 'user', content: 'Which clause lets Northwind leave the supply agreement early?'},
        {role: 'assistant', content: '', tool_calls: [{id: 'm1', name: 'search_knowledge_base'}]},
        {role: 'tool', content: '12 results\nTop: Northwind MSA §9.2', tool_call_id: 'm1', name: 'search_knowledge_base'},
        {role: 'assistant', content: 'Clause 9.2 allows it after month 18.', tool_calls: []},
      ], {nextBefore: 1, running: true})
      : undefined,
  });
  try {
    return await judge();
  } finally {
    window.fetch = originalFetch;
    Date.now = originalNow;
  }
}

it('Agent traces has no serious axe violations as a list, narrow, with the Run quality details both folded and open', async () => {
  await withDock(async () => {
    const inspector = await openDock(420);
    expect(await seriousViolations(inspector), 'collapsed').to.deep.equal([]);

    // The narrow Run panel folds its dossier; opening it and then the quality disclosure is the one
    // path to the triage detail, and it is judged on the product page too.
    const overview = inspector.querySelector<HTMLDetailsElement>('[data-overview]')!;
    overview.open = true;
    overview.dispatchEvent(new Event('toggle'));
    await waitFor(() => inspector.querySelector('[data-quality-toggle]') !== null);
    inspector.querySelector<HTMLButtonElement>('[data-quality-toggle]')!.click();
    await waitFor(() => inspector.querySelector<HTMLElement>('#run-overview-quality-details')?.hidden === false);
    // The expanded body is one labelled, focusable scroll region a keyboard reader can operate.
    const details = inspector.querySelector<HTMLElement>('#run-overview-quality-details')!;
    expect(details.getAttribute('role')).to.equal('region');
    expect(details.getAttribute('tabindex')).to.equal('0');
    expect(details.getAttribute('aria-label')).to.equal('Run details');
    details.focus();
    expect(document.activeElement, 'the region takes focus').to.equal(details);
    expect(await seriousViolations(inspector), 'expanded').to.deep.equal([]);
  });
});

it('Agent traces keeps its Run composition accessible when it is grouped by agent', async () => {
  await withDock(async () => {
    const inspector = await openDock(null);
    await waitFor(() => inspector.querySelector('dl-run-overview [data-run-overview]') !== null);
    inspector.querySelector<HTMLElement>('[data-dimension="agent"]')!.click();
    await waitFor(() => inspector.querySelector('button[data-bar-segment]') !== null);

    // The bar is a labelled group of buttons, and exactly one names the page on show.
    const bar = inspector.querySelector<HTMLElement>('[data-bar]')!;
    expect(bar.getAttribute('role')).to.equal('group');
    expect(bar.getAttribute('aria-labelledby')).to.equal('run-overview-composition');
    const current = inspector.querySelectorAll('[data-bar-segment][aria-current="true"]');
    expect(current, 'one segment names the shown agent').to.have.length(1);
    expect(await seriousViolations(inspector)).to.deep.equal([]);
  });
});

it('the main agent\'s page, with its answer, evidence and earlier steps to show, has no serious axe violations beside the list', async () => {
  await withDock(async () => {
    const answer = {
      answerText: 'Clause 9.2 allows it after month 18.',
      sources: [{id: '1', title: 'northwind-msa.pdf', sourceUrl: null, downloadUrl: null, chunks: []}],
    } as unknown as AnswerPresentation;
    const inspector = await openDock(null, answeredTurn({
      presentation: answer, usage: {usage_details: {total_tokens: 9400}},
    }));
    await waitFor(() => inspector.querySelector('dl-agent-session dl-activity ol') !== null);
    expect(inspector.querySelector('dl-agent-session [data-load-older="activity"]')).to.not.equal(null);
    inspector.querySelector<HTMLElement>('dl-agent-session summary:has(+ pre)')!.click();
    inspector.querySelector<HTMLDetailsElement>('dl-agent-session details')!.open = true;
    // The Run panel's quality detail is part of the wide dock's surface, folded and open.
    const toggle = inspector.querySelector<HTMLButtonElement>('[data-quality-toggle]');
    if (toggle !== null) {
      toggle.click();
      await waitFor(() => inspector.querySelector<HTMLElement>('#run-overview-quality-details')?.hidden === false);
    }
    expect(await seriousViolations(inspector)).to.deep.equal([]);
  });
});

it('what a settled child answered can be judged from its page: the thumbs and reasons have no serious axe violations and the dock\'s hit target', async () => {
  await withDock(async () => {
    const inspector = await openDock(null);
    inspector.answerFeedbackEnabled = true;
    inspector.querySelector<HTMLElement>('[data-agent-session="c3"]')!.click();
    await waitFor(() => inspector.querySelector('dl-agent-session [data-thumb]') !== null);
    inspector.querySelector<HTMLButtonElement>('dl-agent-session [data-thumb="down"]')!.click();
    await waitFor(() => inspector.querySelector('dl-agent-session [data-reason]') !== null);
    // The reader adds a note to the thumb down they gave, so the note has its own Send.
    const field = inspector.querySelector<HTMLTextAreaElement>('dl-agent-session textarea[aria-label="Add a note (optional)"]')!;
    field.value = 'Too long.';
    field.dispatchEvent(new Event('input', {bubbles: true}));
    await waitFor(() => inspector.querySelector('dl-agent-session [data-send]') !== null);

    const probe = document.createElement('div');
    probe.style.height = 'var(--size-button)';
    document.body.append(probe);
    const hit = probe.getBoundingClientRect().height;
    probe.remove();
    const pressed = [...inspector.querySelectorAll<HTMLElement>(
      'dl-agent-session [data-thumb], dl-agent-session [data-reason], dl-agent-session [data-send]',
    )];
    // Two thumbs, five reasons and the note's Send, each as easy to hit as the dock's other controls.
    expect(pressed).to.have.length(8);
    expect(pressed.filter((button) => pressArea(button).height < hit - 0.5)).to.deep.equal([]);
    expect(inspector.querySelector('dl-agent-session [data-reason="too_verbose"]')!.getAttribute('aria-pressed')).to.equal('true');
    expect(await seriousViolations(inspector)).to.deep.equal([]);
  });
});

it('a running child with its question and activity has no serious axe violations, and no box to answer it', async () => {
  await withDock(async () => {
    const inspector = await openDock(null);
    inspector.querySelector<HTMLElement>('[data-agent-session="c1"]')!.click();
    await waitFor(() => inspector.querySelector('dl-agent-session dl-activity') !== null);
    await waitFor(() => inspector.querySelector('[data-switcher-current]')!.textContent!.includes('Child 1'));
    // The running child's question is a state, not a box: the Inspector only reads.
    expect(inspector.querySelector('dl-agent-session [data-draft], dl-agent-session form')).to.equal(null);
    expect(buttonNamed(inspector, 'Answer instead')).to.equal(null);
    // Every kind of step is on the page, one of them open to its whole result.
    const fold = inspector.querySelector<HTMLDetailsElement>('dl-agent-session dl-activity details');
    if (fold !== null) fold.open = true;
    expect(await seriousViolations(inspector)).to.deep.equal([]);
  });
});

it('a settled child with every fold open, and a child the reader cancelled, have no serious axe violations, narrow', async () => {
  await withDock(async () => {
    const inspector = await openDock(420);
    const open = async (id: string) => {
      inspector.querySelector<HTMLElement>(`[data-agent-session="${id}"]`)!.click();
      await waitFor(() => inspector.querySelector('dl-agent-session dl-activity') !== null);
      await waitFor(() => inspector.querySelector('dl-agent-session details') !== null);
      for (const fold of inspector.querySelectorAll<HTMLDetailsElement>('dl-agent-session details')) fold.open = true;
    };
    await open('c3');
    const settled = await seriousViolations(inspector);
    rowForMain(inspector);
    await open('c5');
    expect(inspector.querySelector('dl-agent-session input[type="checkbox"]')).to.equal(null);
    expect(inspector.querySelector('dl-agent-session form')).to.equal(null);
    expect([...settled, ...await seriousViolations(inspector)]).to.deep.equal([]);
  });
});

/** Go back to the Run's own agent from the switcher. */
function rowForMain(inspector: DlInspector): void {
  inspector.querySelector<HTMLElement>('[data-agent-session="main"]')!.click();
}

it('Settings MCP consent, credential, and create forms have no serious accessible-name or contrast violations', async () => {
  const originalFetch = window.fetch;
  window.fetch = async (input) => String(input).includes('/connections/')
    ? Response.json({revision: '1', connections: [{
      connection_id: 'a', label: 'Personal tools', endpoint: 'https://fixture.example/mcp',
      enabled: false, activation_epoch: 1, generation: 1, authentication: 'oauth',
      endpoint_headers: [], credential_header: null, credential_prefix: null,
      status: 'needs-auth', authorization_status: 'failed',
      tool_count: 7, declaration_tokens: 1_400,
    }], presets: [{preset_id: 'notion', label: 'Notion', endpoint: 'https://mcp.notion.com/mcp',
      default_authentication: 'oauth', endpoint_headers: [], credential_header: 'Authorization', credential_prefix: 'Bearer '}]})
    : Response.json({enabled: true, active_count: 0});
  try {
    // Connections renders on the Settings dialog surface, so it is judged there.
    const settings = mountInApp(document.createElement('dl-settings-dialog'));
    await settings.open();
    const feature = settings.querySelector('dl-settings-connections')!;
    await waitFor(() => Boolean(feature.view));
    await feature.updateComplete;
    feature.querySelector<HTMLButtonElement>('[data-card="a"]')!.click();
    buttonNamed(feature, 'Add MCP connection')!.click();
    await feature.updateComplete;
    const credentialForm = await seriousViolations(feature);
    // The consent modal makes the rest of Settings inert, so it is judged on its own.
    feature.querySelector<HTMLButtonElement>('[data-switch="a"]')!.click();
    await feature.updateComplete;
    const consent = await seriousViolations(feature);
    expect([...credentialForm, ...consent]).to.deep.equal([]);
  } finally {window.fetch = originalFetch;}
});

it('every page of the Settings dialog has no serious accessible-name or structure violations', async () => {
  const originalFetch = window.fetch;
  window.fetch = wire({
    'GET /web/api/connections/mcp': () => Response.json({revision: '1', presets: [], connections: [{
      connection_id: 'a', label: 'Personal tools', endpoint: 'https://fixture.example/mcp',
      enabled: true, activation_epoch: 1, generation: 1, authentication: 'bearer',
      endpoint_headers: [], credential_header: null, credential_prefix: null,
      status: 'degraded', authorization_status: null,
      tool_count: 1, declaration_tokens: 2_000,
    }]}),
    'GET /web/api/agent-accounts': () => Response.json(agentAccountsView([
      wireAccount('discourse.org', {last_used_at: '2026-10-04T15:13:01Z'}),
      wireAccount('ycombinator.com', {email: null, username: null}),
    ])),
    'GET /web/api/memory/settings': () => memorySettings(true, 2),
    'GET /web/api/memory': () => memoryPage([
      {id: 'one', body: 'Use concise answers'}, {id: 'two', kind: 'fact', body: 'Lives in Sweden'},
    ]),
    'GET /web/api/skills/mine': () => ownerSkills([
      wireSkill('csv-report', {description: 'Summarise a CSV file into a short report, with the columns it found and the rows it skipped.'}),
      wireSkill('old-style-guide', {enabled: false}),
    ]),
    'GET /web/api/skills/mine/csv-report/document': () => new Response('---\nname: csv-report\n---\nSummarise a CSV file.\n'),
    'GET /web/api/privacy/settings': () => privacySettings(true),
  }).fetch;
  try {
    const {settings} = mountSettings();
    const dialog = await openSettings(settings);
    for (const section of ['connections', 'agent-accounts', 'memory', 'skills', 'conversations', 'privacy', 'language']) {
      const row = settings.querySelector<HTMLButtonElement>(`nav [data-section="${section}"]`)!;
      row.click();
      await settings.updateComplete;
      if (section === 'connections') {
        const feature = settings.querySelector('dl-settings-connections')!;
        await waitFor(() => Boolean(feature.view));
        await feature.updateComplete;
        feature.querySelector<HTMLButtonElement>('[data-card="a"]')!.click();
        await feature.updateComplete;
      } else if (section === 'agent-accounts') {
        const feature = settings.querySelector('dl-settings-agent-accounts')!;
        await waitFor(() => Boolean(feature.view));
        await feature.updateComplete;
      } else if (section === 'memory') {
        await waitFor(() => settings.querySelectorAll('dl-settings-memory li').length === 2);
      } else if (section === 'skills') {
        // One Skill is off, and the first shows its SKILL.md.
        await waitFor(() => settings.querySelectorAll('dl-settings-skills [data-skill]').length === 2);
        settings.querySelector<HTMLButtonElement>('dl-settings-skills [data-view="csv-report"]')!.click();
        await waitFor(() => settings.querySelector('dl-settings-skills pre') !== null);
      } else if (section === 'privacy') {
        // The switch is on, so the page shows the caption of a permission that was given.
        await waitFor(() => settings.querySelector('#privacy-contribute[aria-checked="true"]') !== null);
      }
      expect(await seriousViolations(dialog), section).to.deep.equal([]);
    }
  } finally {window.fetch = originalFetch;}
});
