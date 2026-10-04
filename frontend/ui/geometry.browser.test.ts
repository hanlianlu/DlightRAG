// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {resetMouse, sendKeys, sendMouse} from '@web/test-runner-commands';
import {productionHandles} from '../stores/app-handles.ts';
import {DEFAULT_CHANGES} from '../testing/workspaces.ts';
import type {DlIngestTarget} from './ingest-target.ts';
import './ingest-target.ts';
import type {DlWorkspaceScope} from './workspace-scope.ts';
import './workspace-scope.ts';

const stylesheets = [
  '../design-system/index.css',
  '../styles/global.css',
  '../styles/layout.css',
  '../styles/shared-components.css',
  '../styles/panels.css',
  '../styles/inspector-files.module.css',
  '../styles/ingest-target.module.css',
  '../styles/workspaces.module.css',
  '../styles/failed-file-recovery.module.css',
  '../styles/inspector-sources.module.css',
  '../styles/answer-presentation.module.css',
  '../styles/artifacts.css',
  '../styles/chat.module.css',
  '../styles/lightbox.module.css',
];

before(async () => {
  await Promise.all(stylesheets.map(async (href) => new Promise<void>((resolve, reject) => {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = new URL(href, import.meta.url).href;
    link.addEventListener('load', () => { resolve(); }, {once: true});
    link.addEventListener('error', () => { reject(new Error(`could not load ${href}`)); }, {once: true});
    document.head.appendChild(link);
  })));
});

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

function radius(className: string): string {
  return getComputedStyle(element(className)).borderRadius;
}

afterEach(() => {
  document.body.replaceChildren();
});

it('publishes the approved Soft semantic radius ladder', () => {
  const tokens = getComputedStyle(document.documentElement);
  expect(tokens.getPropertyValue('--radius-control').trim()).to.equal('0.625rem');
  expect(tokens.getPropertyValue('--radius-card').trim()).to.equal('1rem');
  expect(tokens.getPropertyValue('--radius-popover').trim()).to.equal('1.125rem');
  expect(tokens.getPropertyValue('--radius-dialog').trim()).to.equal('1.375rem');
  expect(tokens.getPropertyValue('--radius-composer').trim()).to.equal('1.5rem');
});

it('assigns geometry by surface role rather than component size', () => {
  expect(radius('primary-btn')).to.equal('10px');
  expect(radius('userMessage')).to.equal('16px');
  expect(radius('source-chunk')).to.equal('16px');
  expect(radius('source-doc')).to.equal('0px');
  expect(radius('panel')).to.equal('0px');
  expect(radius('dl-popover')).to.equal('18px');
  expect(radius('workspace-dialog')).to.equal('22px');
  expect(radius('composer-form')).to.equal('24px');
  const app = element('app');
  expect(getComputedStyle(app).borderRadius).to.equal('0px');
  expect(getComputedStyle(app).clipPath).to.equal('none');
  expect(radius('workspace-selector-trigger')).to.equal('999px');
  expect(radius('imageLightboxPrev')).to.equal('0px 10px 10px 0px');
});

it('uses the DlightRAG focus ring without changing the control geometry', () => {
  const reference = document.createElement('button');
  reference.className = 'answer-ref-item';
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
  const notice = element('answerReconnect');
  const status = document.createElement('span');
  status.className = 'answerReconnectStatus';
  status.textContent = 'Connection lost while this answer is running.';
  const action = document.createElement('button');
  action.className = 'answerReconnectAction';
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

// The ring sits 2px outside its control, so it reads against whatever surrounds the control:
// a surface, or a row tinted under the pointer or by selection.
it('keeps the focus ring at 3:1 on every surface and row tint in both themes', () => {
  const probe = element('');
  const resolve = (property: 'color' | 'backgroundColor', value: string): Rgba => {
    probe.style[property] = value;
    return rgba(getComputedStyle(probe)[property]);
  };
  for (const colorMode of ['dark', 'light']) {
    document.documentElement.dataset.colorMode = colorMode;
    const ring = resolve('color', 'var(--focus-ring-color)');
    for (const surface of ['--color-bg-base', '--color-bg-surface', '--color-bg-elevated', '--color-bg-subtle']) {
      const plain = resolve('backgroundColor', `var(${surface})`);
      for (const tint of ['', '--color-bg-hover', '--color-selected-row']) {
        const background = tint ? composite(resolve('backgroundColor', `var(${tint})`), plain) : plain;
        expect(contrastRatio(composite(ring, background), background), `${colorMode} ${surface} ${tint}`)
          .to.be.at.least(3);
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
      rows: '[data-workspace-choice]', radio: '.workspacePopoverCheck',
    },
    {
      mount: (): DlIngestTarget => {
        const target = document.createElement('dl-ingest-target');
        target.className = 'ingest-target';
        target.active = true;
        return target;
      },
      trigger: '#ingest-target-trigger', popover: '#ingest-target-popover',
      rows: '[data-ingest-workspace-choice]', radio: '.ingest-target-popover-radio',
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

    // A ring cut off at the scroll edge loses a whole side, so every step keeps all of it.
    for (const key of ['ArrowDown', 'ArrowUp']) {
      for (let step = 0; step < rows; step += 1) {
        await sendKeys({press: key});
        const focused = document.activeElement as HTMLElement;
        const style = getComputedStyle(focused);
        const reach = Number.parseFloat(style.outlineOffset) + Number.parseFloat(style.outlineWidth);
        const box = focused.getBoundingClientRect();
        const port = popover.getBoundingClientRect();
        const top = port.top + popover.clientTop;
        const left = port.left + popover.clientLeft;
        const where = `${key} to ${focused.textContent?.trim()}`;
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
  const answer = element('aiMessageContent');
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
