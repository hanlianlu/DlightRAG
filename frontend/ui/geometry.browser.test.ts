// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

// Runs on the product-styles page (web-test-runner.config.mjs): the application's shipped stylesheets are
// linked, and CSS Modules carry the class names the build gave them, which these tests read from the same
// imports the components use.

import {expect} from '@esm-bundle/chai';
import {emulateMedia, resetMouse, sendKeys, sendMouse} from '@web/test-runner-commands';
import type {AnswerPresentation} from '../api/conversations.ts';
import {productionHandles} from '../stores/app-handles.ts';
import activityStyles from '../styles/activity.module.css';
import answerStyles from '../styles/answer-presentation.module.css';
import chatStyles from '../styles/chat.module.css';
import {DEFAULT_CHANGES} from '../testing/workspaces.ts';
import type {AnswerPresentationElement} from './answer-presentation.ts';
import type {DlChatMessageList} from './chat-message-list.ts';
import './chat-message-list.ts';
import type {DlIngestTarget} from './ingest-target.ts';
import './ingest-target.ts';
import type {DlWorkspaceScope} from './workspace-scope.ts';
import './workspace-scope.ts';

function element(className: string): HTMLElement {
  const node = document.createElement('div');
  node.className = className;
  document.body.appendChild(node);
  return node;
}

interface Rgba {
  r: number;
  g: number;
  b: number;
  a: number;
}

function rgba(value: string): Rgba {
  const match = value.match(/^rgba?\(([^)]+)\)$/);
  if (!match) throw new Error(`unsupported computed color: ${value}`);
  const channels = match[1].split(',').map((part) => Number(part.trim()));
  return {r: channels[0], g: channels[1], b: channels[2], a: channels[3] ?? 1};
}

function composite(foreground: Rgba, background: Rgba): Rgba {
  const alpha = foreground.a + background.a * (1 - foreground.a);
  const channel = (front: number, back: number): number => (
    (front * foreground.a + back * background.a * (1 - foreground.a)) / alpha
  );
  return {
    r: channel(foreground.r, background.r),
    g: channel(foreground.g, background.g),
    b: channel(foreground.b, background.b),
    a: alpha,
  };
}

function contrastRatio(first: Rgba, second: Rgba): number {
  const luminance = (color: Rgba): number => {
    const linear = [color.r, color.g, color.b].map((channel) => {
      const normalized = channel / 255;
      return normalized <= 0.04045
        ? normalized / 12.92
        : ((normalized + 0.055) / 1.055) ** 2.4;
    });
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
  };
  const [lighter, darker] = [luminance(first), luminance(second)].sort((a, b) => b - a);
  return (lighter + 0.05) / (darker + 0.05);
}

afterEach(() => {
  document.body.replaceChildren();
});

/** What a token resolves to in a property, read from a probe's computed style: the shipped stylesheet is
 *  minified, so the text a token was written with is not what the page holds. */
function resolved(property: string, token: string): string {
  const probe = element('');
  probe.style.setProperty(property, `var(${token})`);
  return getComputedStyle(probe).getPropertyValue(property);
}

it('centres the composer input line on the composer, level with its buttons', () => {
  const form = element('composer-form');
  const plus = document.createElement('button');
  plus.className = 'composer-plus';
  const wrap = document.createElement('div');
  wrap.className = 'composer-input-wrap';
  const input = document.createElement('textarea');
  input.className = 'composer-input';
  input.rows = 1;
  wrap.appendChild(input);
  const send = document.createElement('button');
  send.className = 'composer-send';
  form.append(plus, wrap, send);

  const middle = (node: Element): number => {
    const box = node.getBoundingClientRect();
    return box.top + (box.height / 2);
  };
  expect(wrap.getBoundingClientRect().height).to.be.closeTo(input.getBoundingClientRect().height, 0.01);
  expect(middle(input)).to.be.closeTo(middle(form), 0.01);
  expect(middle(input)).to.be.closeTo(middle(send), 0.01);
});

it('uses the DlightRAG focus ring without changing the control geometry', () => {
  const reference = document.createElement('button');
  reference.className = answerStyles['answer-ref-item'];
  document.body.appendChild(reference);
  reference.focus();

  const ringProbe = element('ring-probe');
  ringProbe.style.color = 'var(--color-control-ring)';
  const referenceStyle = getComputedStyle(reference);
  expect(referenceStyle.outlineColor).to.equal(getComputedStyle(ringProbe).color);
  expect(referenceStyle.outlineStyle).to.equal('solid');
  expect(referenceStyle.outlineWidth).to.equal('2px');
  expect(referenceStyle.outlineOffset).to.equal('2px');
  expect(referenceStyle.borderRadius).to.equal('10px');
});

it('keeps reconnect text and focus indicators contrast-safe in every theme and state', () => {
  const notice = element(chatStyles.answerReconnect);
  const status = document.createElement('span');
  status.className = chatStyles.answerReconnectStatus;
  status.textContent = 'Connection lost while this answer is running.';
  const action = document.createElement('button');
  action.className = chatStyles.answerReconnectAction;
  action.textContent = 'Reconnect';
  notice.append(status, action);
  action.focus();

  for (const colorMode of ['dark', 'light']) {
    document.documentElement.dataset.colorMode = colorMode;
    for (const reconnectState of ['running', 'stopping']) {
      notice.dataset.reconnectState = reconnectState;
      const bodyBackground = rgba(getComputedStyle(document.body).backgroundColor);
      const noticeBackground = composite(
        rgba(getComputedStyle(notice).backgroundColor),
        bodyBackground,
      );
      const actionStyle = getComputedStyle(action);
      const actionBackground = composite(rgba(actionStyle.backgroundColor), noticeBackground);

      expect(contrastRatio(rgba(actionStyle.color), actionBackground)).to.be.at.least(4.5);
      expect(contrastRatio(rgba(actionStyle.outlineColor), noticeBackground)).to.be.at.least(3);
      expect(actionStyle.outlineStyle).to.equal('solid');
      expect(actionStyle.outlineWidth).to.equal('2px');
      expect(actionStyle.outlineOffset).to.equal('2px');
    }
  }
  document.documentElement.dataset.colorMode = 'dark';
});

/** A colour token's value in the current theme, read from a probe's computed style. */
function resolveToken(property: 'color' | 'backgroundColor', token: string): Rgba {
  const probe = element('');
  probe.style[property] = `var(${token})`;
  return rgba(getComputedStyle(probe)[property]);
}

const surfaces = ['--color-bg-base', '--color-bg-surface', '--color-bg-elevated', '--color-bg-subtle'];

// The ring sits 2px outside its control, so it reads against whatever surrounds the control:
// a surface, or a row tinted under the pointer or by selection.
it('keeps the focus ring at 3:1 on every surface and row tint in both themes', () => {
  for (const colorMode of ['dark', 'light']) {
    document.documentElement.dataset.colorMode = colorMode;
    const ring = resolveToken('color', '--focus-ring-color');
    for (const surface of surfaces) {
      const plain = resolveToken('backgroundColor', surface);
      for (const tint of ['', '--color-bg-hover', '--color-selected-row']) {
        const background = tint ? composite(resolveToken('backgroundColor', tint), plain) : plain;
        expect(contrastRatio(composite(ring, background), background), `${colorMode} ${surface} ${tint}`)
          .to.be.at.least(3);
      }
    }
  }
  document.documentElement.dataset.colorMode = 'dark';
});

// Accent text is the gold for small text. It sits on a surface, or on the tint its row, pill,
// chip, or link wears at rest or under the pointer. Gold-300, one step quieter in dark, read
// 4.26:1 under a pill's hover tint.
it('keeps accent text at 4.5:1 on every surface and tint in both themes', () => {
  const tints = [
    '', '--color-bg-hover', '--color-selected-row', '--color-row-alt', '--color-control-surface',
    '--color-accent-ghost', '--color-accent-hover', '--color-accent-surface', '--color-accent-hover-strong',
  ];
  for (const colorMode of ['dark', 'light']) {
    document.documentElement.dataset.colorMode = colorMode;
    const text = resolveToken('color', '--color-accent-text');
    for (const surface of surfaces) {
      const plain = resolveToken('backgroundColor', surface);
      for (const tint of tints) {
        const background = tint ? composite(resolveToken('backgroundColor', tint), plain) : plain;
        expect(contrastRatio(text, background), `${colorMode} ${surface} ${tint}`).to.be.at.least(4.5);
      }
    }
  }
  document.documentElement.dataset.colorMode = 'dark';
});

it('rings a keyboard-focused popover row, keeps its radio legible, and keeps the whole ring in view', async () => {
  const {ingest, workspaces} = productionHandles();
  const records = Array.from({length: 12}, (_, index) => ({
    workspace: `workspace-${index}`,
    displayName: `Workspace ${String(index).padStart(2, '0')}`,
    embeddingModel: 'embed',
    changes: DEFAULT_CHANGES,
  }));
  workspaces.init(records, ['workspace-0'], 'workspace-0');
  ingest.resetToPrimary();
  const houseRing = element('');
  houseRing.style.color = 'var(--focus-ring-color)';
  const hoverTint = element('');
  hoverTint.style.backgroundColor = 'var(--color-bg-hover)';
  const pickers = [
    {
      mount: (): DlWorkspaceScope => {
        const scope = document.createElement('dl-workspace-scope');
        scope.className = 'workspace-selector';
        return scope;
      },
      trigger: '#workspace-trigger', popover: '#workspace-popover',
      rows: '[data-workspace-choice]', radio: '.dl-popover-radio',
    },
    {
      mount: (): DlIngestTarget => {
        const target = document.createElement('dl-ingest-target');
        target.className = 'ingest-target';
        target.active = true;
        return target;
      },
      trigger: '#ingest-target-trigger', popover: '#ingest-target-popover',
      rows: '[data-ingest-workspace-choice]', radio: '.dl-popover-radio',
    },
  ];

  for (const picker of pickers) {
    const host = picker.mount();
    document.body.appendChild(host);
    await host.updateComplete;
    host.querySelector<HTMLButtonElement>(picker.trigger)!.focus();
    await sendKeys({press: 'Enter'});
    await host.updateComplete;
    const popover = host.querySelector<HTMLElement>(picker.popover)!;
    const rows = popover.querySelectorAll(picker.rows).length;
    expect(popover.scrollHeight, 'the list scrolls').to.be.greaterThan(popover.clientHeight);

    await sendKeys({press: 'ArrowDown'});
    const row = document.activeElement as HTMLElement;
    expect(row.getAttribute('aria-pressed')).to.equal('false');
    expect(row.matches(':focus-visible')).to.equal(true);
    for (const colorMode of ['dark', 'light']) {
      document.documentElement.dataset.colorMode = colorMode;
      await Promise.all(row.getAnimations({subtree: true}).map((animation) => animation.finished));
      const style = getComputedStyle(row);
      const surface = rgba(getComputedStyle(popover).backgroundColor);
      expect(style.outlineStyle).to.equal('solid');
      expect(style.outlineWidth).to.equal('2px');
      expect(style.outlineOffset).to.equal('2px');
      expect(style.outlineColor).to.equal(getComputedStyle(houseRing).color);
      // Outside the row, the ring's pixels were the popover before focus: one ratio covers
      // contrast with the adjacent colour and the change from the unfocused state.
      expect(contrastRatio(composite(rgba(style.outlineColor), surface), surface), colorMode).to.be.at.least(3);
      const fill = composite(rgba(style.backgroundColor), surface);
      const radio = rgba(getComputedStyle(row.querySelector(picker.radio)!).borderTopColor);
      expect(contrastRatio(radio, fill), `${colorMode} unselected radio on the focused row`).to.be.at.least(3);
    }
    document.documentElement.dataset.colorMode = 'dark';

    // A ring cut off at the scroll edge loses a whole side, so every step keeps all of it. The walk dispatches
    // the keydown the list's own handler steps on, instead of ~50 round trips to the browser's keyboard.
    const press = (key: string): void => {
      (document.activeElement as HTMLElement).dispatchEvent(new KeyboardEvent('keydown', {
        key, bubbles: true, cancelable: true,
      }));
    };
    for (const key of ['ArrowDown', 'ArrowUp']) {
      for (let step = 0; step < rows; step += 1) {
        press(key);
        const focused = document.activeElement as HTMLElement;
        const style = getComputedStyle(focused);
        const reach = Number.parseFloat(style.outlineOffset) + Number.parseFloat(style.outlineWidth);
        const box = focused.getBoundingClientRect();
        const port = popover.getBoundingClientRect();
        const top = port.top + popover.clientTop;
        const left = port.left + popover.clientLeft;
        const where = `${key} to ${focused.textContent?.trim()}`;
        expect(focused.matches(':focus-visible'), where).to.equal(true);
        expect(box.top - reach, where).to.be.at.least(top - 0.5);
        expect(box.bottom + reach, where).to.be.at.most(top + popover.clientHeight + 0.5);
        expect(box.left - reach, where).to.be.at.least(left - 0.5);
        expect(box.right + reach, where).to.be.at.most(left + popover.clientWidth + 0.5);
      }
    }

    const hovered = document.activeElement as HTMLElement;
    const box = hovered.getBoundingClientRect();
    const center: [number, number] = [
      Math.round(box.left + box.width / 2),
      Math.round(box.top + box.height / 2),
    ];
    try {
      await sendMouse({type: 'move', position: center});
      await Promise.all(hovered.getAnimations().map((animation) => animation.finished));
      expect(getComputedStyle(hovered).backgroundColor, 'hover keeps its tint')
        .to.equal(getComputedStyle(hoverTint).backgroundColor);
    } finally {
      await resetMouse();
    }
    host.remove();
  }
});

it('labels Canvas layout controls, truncates long titles, and preserves compact downloads', () => {
  expect(matchMedia('(width < 1200px)').matches).to.equal(true);
  const canvas = document.createElement('dl-artifact-canvas');
  canvas.className = 'open';
  const header = document.createElement('div');
  header.className = 'artifact-canvas-header';
  const heading = document.createElement('div');
  heading.className = 'artifact-canvas-heading';
  const title = document.createElement('h2');
  title.className = 'artifact-canvas-title';
  title.id = 'artifact-canvas-title';
  title.textContent = 'A very long generated Artifact title that must remain within its fixed header';
  heading.appendChild(title);
  const actions = document.createElement('div');
  actions.className = 'artifact-canvas-actions';
  const layouts = document.createElement('div');
  layouts.className = 'artifact-canvas-layout-actions';
  layouts.setAttribute('role', 'group');
  layouts.setAttribute('aria-labelledby', title.id);
  const download = document.createElement('a');
  download.className = 'dl-btn';
  download.textContent = 'Download';
  actions.append(layouts, download);
  header.append(heading, actions);
  canvas.appendChild(header);
  document.body.appendChild(canvas);

  const titleStyle = getComputedStyle(title);
  expect(titleStyle.overflow).to.equal('hidden');
  expect(titleStyle.textOverflow).to.equal('ellipsis');
  expect(titleStyle.whiteSpace).to.equal('nowrap');
  expect(layouts.getAttribute('aria-labelledby')).to.equal(title.id);
  expect(getComputedStyle(layouts).display).to.equal('none');
  expect(getComputedStyle(download).display).not.to.equal('none');
});

it('rounds and clips rich-content containers without rounding table cells', () => {
  const answer = element(`${chatStyles.aiMessageContent} ${chatStyles.markdownContent}`);
  const table = document.createElement('table');
  const row = table.insertRow();
  row.insertCell().textContent = 'A';
  answer.appendChild(table);

  const code = document.createElement('pre');
  answer.appendChild(code);
  const image = document.createElement('img');
  answer.appendChild(image);

  expect(getComputedStyle(table).borderRadius).to.equal('16px');
  expect(getComputedStyle(table).borderCollapse).to.equal('separate');
  expect(getComputedStyle(table).overflow).to.equal('hidden');
  expect(getComputedStyle(row.cells[0]).borderRadius).to.equal('0px');
  expect(getComputedStyle(code).borderRadius).to.equal('16px');
  expect(getComputedStyle(image).borderRadius).to.equal('16px');
});

const answerPresentation: AnswerPresentation = {
  answerText: '### Revenue\n\nRevenue grew.',
  parts: [{
    type: 'markdown',
    text: '### Revenue\n\nRevenue grew.',
    html: '<h3>Revenue</h3><p>Revenue grew.</p>',
    artifact: null,
    evidenceImage: null,
    inline: false,
  }],
  sources: [{id: '1', title: 'Annual report', sourceUrl: null, downloadUrl: null, chunks: []}],
  evidenceImages: [{
    id: 'chart',
    chunkId: 'chunk-1',
    sourceRef: '1',
    url: '/web/api/images/default/chunk-1?size=full',
    thumbnailUrl: '/web/api/images/default/chunk-1?size=thumb',
    label: 'Chart',
    answerImageSent: true,
  }],
  linkCards: [],
  artifacts: [],
  artifactOutcome: {status: 'complete', issues: []},
};

/** The same Answer in a chat turn and in the Artifact Canvas, whose content
 *  well mounts the presentation with no typography of its own. */
async function answerHosts(): Promise<[chat: HTMLElement, canvas: HTMLElement]> {
  const chat = document.createElement('dl-chat-message-list') as DlChatMessageList;
  chat.turns = [{
    id: 'turn-1',
    userText: 'How did revenue change?',
    userAttachments: [],
    runId: 'run-1',
    state: 'succeeded',
    streamText: '',
    processEntries: [],
    presentation: answerPresentation,
    usage: {},
    error: null,
    activity: null,
    cancelRequested: false,
    steeringMessages: [],
    answerSequence: null, feedback: null,
    toolRows: [],
  }];
  const canvas = document.createElement('dl-artifact-canvas');
  canvas.className = 'open';
  const content = document.createElement('div');
  content.className = 'artifact-canvas-content';
  const presented = document.createElement('dl-answer-presentation') as AnswerPresentationElement;
  presented.presentation = answerPresentation;
  content.appendChild(presented);
  canvas.appendChild(content);
  document.body.append(chat, canvas);
  await chat.updateComplete;
  await Promise.all([chat, canvas].map((host) => (
    host.querySelector<AnswerPresentationElement>('dl-answer-presentation')?.updateComplete
  )));
  return [chat, canvas];
}

function styleOf(node: Element | null, properties: readonly string[]): Record<string, string> {
  if (!node) throw new Error('missing answer node');
  const style = getComputedStyle(node);
  return Object.fromEntries(properties.map((property) => [property, style.getPropertyValue(property)]));
}

it('keeps Markdown typography inside Markdown parts in chat and in the Artifact Canvas', async () => {
  const [chat, canvas] = await answerHosts();
  const captionSize = element('');
  captionSize.style.fontSize = 'var(--font-size-caption)';

  // The presentation's own chrome renders alike whichever host wraps it: every
  // section title is one caption label and evidence thumbnails fill their tiles.
  const title = ['font-size', 'font-weight', 'color', 'text-transform', 'margin-top', 'margin-bottom'];
  const caption = styleOf(canvas.querySelector(`.${answerStyles['answer-references']} .${answerStyles['answer-section-title']}`), title);
  expect(caption['font-size']).to.equal(getComputedStyle(captionSize).fontSize);
  expect(caption['text-transform']).to.equal('uppercase');
  for (const [name, host] of [['chat', chat], ['canvas', canvas]] as const) {
    for (const selector of [
      '.answer-evidence > h3',
      `.${answerStyles['answer-references']} .${answerStyles['answer-section-title']}`,
    ]) {
      expect(styleOf(host.querySelector(selector), title), `${name} ${selector}`).to.deep.equal(caption);
    }
  }
  const thumbnail = ['height', 'margin-top', 'margin-bottom', 'border-radius'];
  expect(styleOf(chat.querySelector('.answer-evidence img'), thumbnail))
    .to.deep.equal(styleOf(canvas.querySelector('.answer-evidence img'), thumbnail));

  // A model-authored heading keeps the Markdown heading design in both hosts.
  for (const host of [chat, canvas]) {
    const body = styleOf(host.querySelector('[data-answer-part]'), ['font-size']);
    const heading = styleOf(host.querySelector('[data-answer-part] h3'), ['font-size', 'font-weight']);
    expect(Number.parseFloat(heading['font-size']))
      .to.be.closeTo(Number.parseFloat(body['font-size']) * 1.05, 0.01);
    expect(heading['font-weight']).to.equal('600');
  }
});

it('places popovers and menus from their trigger through one anchored primitive', () => {
  const px = (value: string): number => Number.parseFloat(value);
  const token = (name: string): number => {
    const probe = element('');
    probe.style.height = `var(${name})`;
    return probe.getBoundingClientRect().height;
  };
  const anchoredIn = (className: string): CSSStyleDeclaration => {
    const trigger = element('');
    trigger.style.position = 'relative';
    trigger.style.height = '40px';
    const surface = document.createElement('div');
    surface.className = className;
    trigger.appendChild(surface);
    return getComputedStyle(surface);
  };
  const gap = token('--space-3xs');

  const below = anchoredIn('dl-popover dl-anchored');
  expect(below.position).to.equal('absolute');
  expect(px(below.top)).to.be.closeTo(40 + gap, 0.5);
  expect(px(below.left)).to.equal(0);

  const theme = anchoredIn('dl-anchored dl-anchored--end');
  expect(px(theme.right)).to.equal(0);
  expect(theme.left).to.not.equal('0px');

  const mode = anchoredIn('composer-mode-menu dl-anchored dl-anchored--above dl-anchored--end');
  expect(px(mode.bottom)).to.be.closeTo(40 + gap, 0.5);
  expect(px(mode.right)).to.equal(0);

  const skills = anchoredIn('skill-menu dl-anchored dl-anchored--above');
  expect(px(skills.bottom)).to.be.closeTo(40 + token('--space-2xs'), 0.5);
  expect(px(skills.left)).to.equal(0);
  expect(px(skills.right)).to.equal(0);
});

/** A composer form as the product draws it: Attach, the input wrap holding the mirror and the textarea,
 *  and Send. */
function composerForm(): {
  form: HTMLElement;
  wrap: HTMLElement;
  mirror: HTMLElement;
  input: HTMLTextAreaElement;
} {
  const form = element('composer-form');
  const plus = document.createElement('button');
  plus.className = 'composer-plus';
  const wrap = document.createElement('div');
  wrap.className = 'composer-input-wrap';
  const mirror = document.createElement('div');
  mirror.className = 'composer-input-mirror';
  mirror.setAttribute('aria-hidden', 'true');
  const input = document.createElement('textarea');
  input.className = 'composer-input';
  input.rows = 1;
  wrap.append(mirror, input);
  const send = document.createElement('button');
  send.className = 'composer-send';
  form.append(plus, wrap, send);
  return {form, wrap, mirror, input};
}

it('draws the composer textarea and its mirror as one box that stays inside its wrap', () => {
  const {wrap, mirror, input} = composerForm();
  const box = (node: Element): DOMRect => node.getBoundingClientRect();

  expect(box(input).left, 'left edge').to.be.closeTo(box(mirror).left, 0.01);
  expect(box(input).right, 'right edge').to.be.closeTo(box(mirror).right, 0.01);
  expect(box(input).width, 'width').to.be.closeTo(box(mirror).width, 0.01);
  expect(box(input).left, 'inside the wrap on the left').to.be.at.least(box(wrap).left - 0.01);
  expect(box(input).right, 'inside the wrap on the right').to.be.at.most(box(wrap).right + 0.01);
  expect(input.scrollWidth, 'nothing spills sideways').to.be.at.most(input.clientWidth);

  // One width means one set of line breaks, so the ghost hint in the mirror stands where the text does.
  for (let words = 1; words <= 60; words += 1) {
    const draft = Array.from({length: words}, (_, index) => 'abcdefghij'.slice(0, 3 + (index % 8))).join(' ');
    input.value = draft;
    mirror.textContent = draft;
    expect(mirror.scrollHeight, `${words} words wrap alike`).to.equal(input.scrollHeight);
  }
});

// The composer is the one control that shows focus by its border and a soft halo rather than the 2px
// outline, so its border carries the change: at least the 3:1 the focus contract asks of any focus
// indicator, against the border it had at rest, over whatever surface the form is drawn on.
it('moves the composer border 3:1 against its rest on every surface when it takes focus', () => {
  const {form, input} = composerForm();
  const settle = (): void => {
    for (const animation of form.getAnimations()) animation.finish();
  };
  try {
    for (const colorMode of ['dark', 'light']) {
      document.documentElement.dataset.colorMode = colorMode;
      input.blur();
      settle();
      const rest = rgba(getComputedStyle(form).borderTopColor);
      input.focus();
      settle();
      expect(form.matches(':focus-within')).to.equal(true);
      const focused = rgba(getComputedStyle(form).borderTopColor);
      for (const surface of surfaces) {
        const back = resolveToken('backgroundColor', surface);
        expect(contrastRatio(composite(focused, back), composite(rest, back)), `${colorMode} ${surface}`)
          .to.be.at.least(3);
      }
    }
  } finally {
    document.documentElement.dataset.colorMode = 'dark';
  }
});

/**
 * The rules the shipped stylesheet keeps for `(pointer: coarse)`, read back as plain rules in the layer they
 * live in. A page cannot be given a touch pointer here, so the test applies the same rules unconditionally.
 */
function coarsePointerRules(): HTMLStyleElement {
  const found: string[] = [];
  const collect = (rules: CSSRuleList, layers: readonly string[]): void => {
    for (const rule of rules) {
      if (rule instanceof CSSLayerBlockRule) {
        collect(rule.cssRules, [...layers, rule.name]);
      } else if (rule instanceof CSSMediaRule && /pointer:\s*coarse/.test(rule.conditionText)) {
        const inner = [...rule.cssRules].map((each) => each.cssText).join('\n');
        found.push(layers.reduceRight((css, name) => `@layer ${name} {${css}}`, inner));
      }
    }
  };
  for (const sheet of document.styleSheets) collect(sheet.cssRules, []);
  expect(found, 'the stylesheets keep rules for a touch pointer').to.not.have.length(0);
  const style = document.createElement('style');
  style.textContent = found.join('\n');
  document.head.append(style);
  return style;
}

it('keeps every text-entry control at 16px or more under a touch pointer', () => {
  const fontSize = (node: Element): number => Number.parseFloat(getComputedStyle(node).fontSize);
  const control = (tag: string, className = '', type = ''): HTMLElement => {
    const node = document.createElement(tag);
    node.className = className;
    if (type) node.setAttribute('type', type);
    document.body.appendChild(node);
    return node;
  };
  const typed = [
    control('input', 'dl-input', 'search'),
    control('textarea', 'dl-dialog-input'),
    control('input', 'dl-dialog-input', 'text'),
    control('input', 'dl-popover-input', 'text'),
    control('input', '', 'text'),
    control('input', '', 'url'),
    control('input', '', 'password'),
    control('input'),
    control('textarea'),
    control('select'),
  ];
  const untyped = ['checkbox', 'radio', 'range', 'file', 'button', 'submit'].map((type) => control('input', '', type));
  const {form, mirror, input} = composerForm();
  const entry = element('');
  entry.style.fontSize = 'var(--font-size-touch-entry)';

  const before = [...typed, ...untyped, input, mirror].map(fontSize);
  expect(fontSize(entry), 'the entry floor').to.equal(16);
  for (const [index, node] of [...typed, input].entries()) {
    expect(before[index], `${node.className || node.tagName} is small before`).to.be.below(16);
  }
  const inputBefore = getComputedStyle(input).minHeight;

  const rules = coarsePointerRules();
  try {
    for (const node of typed) {
      expect(fontSize(node), `${node.tagName} ${node.className} ${node.getAttribute('type') ?? ''}`).to.equal(16);
    }
    untyped.forEach((node, index) => {
      expect(fontSize(node), `${node.getAttribute('type')} keeps its size`).to.equal(before[typed.length + index]);
    });

    // The composer's mirror lays out the same text at the same size, and the input line's minimum height
    // follows it, so the line holds its text.
    expect(fontSize(input)).to.equal(16);
    expect(fontSize(mirror)).to.equal(16);
    expect(Number.parseFloat(getComputedStyle(input).minHeight)).to.equal(16 * 1.5);
    expect(inputBefore).to.not.equal(getComputedStyle(input).minHeight);
    expect(input.getBoundingClientRect().height).to.be.at.least(16 * 1.5 - 0.01);
    expect(form.getBoundingClientRect().height).to.be.above(0);

    // A fixed-height field still holds the larger line.
    const field = typed[0];
    expect(Number.parseFloat(getComputedStyle(field).lineHeight)).to.be.at.most(field.clientHeight);
  } finally {
    rules.remove();
  }
});

it('sets the composer menus, notices and Run rows in the type scale', () => {
  const sized = (token: string): string => {
    const probe = element('');
    probe.style.fontSize = `var(${token})`;
    return getComputedStyle(probe).fontSize;
  };
  const within = (parent: string | null, tag: string, className: string): HTMLElement => {
    const host = parent ? element(parent) : document.body;
    const node = document.createElement(tag);
    node.className = className;
    host.appendChild(node);
    return node;
  };
  const roles: Array<[HTMLElement, string]> = [
    [within('composer-mode', 'button', 'composer-mode-trigger'), '--font-size-detail'],
    [within('composer-mode-menu', 'button', ''), '--font-size-detail'],
    [within(null, 'span', 'composer-effort-default'), '--font-size-caption'],
    [within('skill-menu', 'span', 'skill-menu-name'), '--font-size-detail'],
    [within('skill-menu', 'span', 'skill-menu-desc'), '--font-size-caption'],
    [within('skill-menu', 'span', 'skill-menu-source'), '--font-size-caption'],
    [within(null, 'div', 'skill-notice'), '--font-size-detail'],
    [within(null, 'div', chatStyles.toolRow), '--font-size-detail'],
    [within(null, 'span', chatStyles.toolDuration), '--font-size-caption'],
    [within(null, 'span', `${chatStyles.streamingDot} ${chatStyles.progressPhase}`), '--font-size-detail'],
  ];
  for (const [node, token] of roles) {
    expect(getComputedStyle(node).fontSize, `${node.className || node.tagName} takes ${token}`).to.equal(sized(token));
  }
});

it('zeroes the change durations under reduced motion and leaves the loop periods alone', async () => {
  const {form} = composerForm();
  const durations = ['--duration-instant', '--duration-control', '--duration-overlay', '--duration-shell'];
  const loops = ['--duration-spin', '--duration-pulse', '--duration-pulse-slow'];
  const seconds = (name: string): number => Number.parseFloat(resolved('transition-duration', name));
  try {
    for (const name of durations) expect(seconds(name), name).to.be.greaterThan(0);
    expect(getComputedStyle(form).transitionDuration).to.equal('0.2s, 0.2s');

    await emulateMedia({reducedMotion: 'reduce'});
    for (const name of durations) expect(seconds(name), name).to.equal(0);
    for (const name of loops) expect(seconds(name), name).to.be.greaterThan(0);
    expect(getComputedStyle(form).transitionDuration).to.equal('0s, 0s');
  } finally {
    await emulateMedia({reducedMotion: 'no-preference'});
  }
});

it('wraps an unbroken query and an unbroken answer inside the chat column', async () => {
  const unbroken = 'x'.repeat(400);
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  list.style.cssText = 'display:block;width:360px';
  list.turns = [{
    id: 'turn-1',
    userText: unbroken,
    userAttachments: [],
    runId: 'run-1',
    state: 'succeeded',
    streamText: '',
    processEntries: [],
    presentation: {
      ...answerPresentation,
      parts: [{...answerPresentation.parts[0]!, html: `<p>${unbroken}</p>`}],
    },
    usage: {},
    error: null,
    activity: null,
    cancelRequested: false,
    steeringMessages: [],
    answerSequence: null, feedback: null,
    toolRows: [],
  }];
  document.body.append(list);
  await list.updateComplete;
  await list.querySelector<AnswerPresentationElement>('dl-answer-presentation')?.updateComplete;

  const column = list.getBoundingClientRect();
  for (const className of [chatStyles.userMessage, chatStyles.aiMessageContent]) {
    const node = list.querySelector<HTMLElement>(`.${className}`)!;
    expect(node.scrollWidth, `${className} holds its words`).to.be.at.most(node.clientWidth);
    expect(node.getBoundingClientRect().right, `${className} stays in the column`).to.be.at.most(column.right + 0.5);
  }
});

it('lets a panel\'s content be shorter than what it holds, so the child that scrolls has something to scroll', () => {
  const panel = element('panel');
  panel.style.height = '240px';
  const content = document.createElement('div');
  content.className = 'panel-content';
  const scroller = document.createElement('div');
  scroller.style.cssText = 'overflow-y:auto;height:100%';
  const tall = document.createElement('div');
  tall.style.height = '1000px';
  scroller.append(tall);
  content.append(scroller);
  panel.append(content);

  expect(panel.getBoundingClientRect().height).to.equal(240);
  expect(content.getBoundingClientRect().height).to.be.at.most(240);
  expect(scroller.scrollHeight).to.be.greaterThan(scroller.clientHeight);
});

it('marks a searched word in the accent gold under text it can be read in, in both themes', () => {
  const pane = element(activityStyles.pane);
  pane.textContent = 'Clause ';
  const mark = document.createElement('mark');
  mark.textContent = 'nine';
  pane.append(mark);
  try {
    for (const colorMode of ['dark', 'light']) {
      document.documentElement.dataset.colorMode = colorMode;
      const surface = rgba(getComputedStyle(pane).backgroundColor);
      const style = getComputedStyle(mark);
      const fill = composite(rgba(style.backgroundColor), composite(surface, rgba(getComputedStyle(document.body).backgroundColor)));
      expect(style.backgroundColor, `${colorMode} the word is marked in gold`)
        .to.equal(resolved('background-color', '--color-gold-200'));
      expect(contrastRatio(rgba(style.color), fill), `${colorMode} the marked word reads`).to.be.at.least(4.5);
    }
  } finally {
    document.documentElement.dataset.colorMode = 'dark';
  }
});
