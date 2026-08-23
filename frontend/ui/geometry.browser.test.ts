// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';

const stylesheets = [
  '../tokens/utopia.css',
  '../styles/global.css',
  '../styles/primitives.css',
  '../styles/layout.css',
  '../styles/panels.css',
  '../styles/files.css',
  '../styles/sources.css',
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
  expect(radius('ui-popover')).to.equal('18px');
  expect(radius('workspace-dialog')).to.equal('22px');
  expect(radius('composer-form')).to.equal('24px');
  const app = element('app');
  expect(getComputedStyle(app).borderRadius).to.equal('0px');
  expect(getComputedStyle(app).clipPath).to.equal('none');
  expect(radius('workspace-selector')).to.equal('999px');
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
