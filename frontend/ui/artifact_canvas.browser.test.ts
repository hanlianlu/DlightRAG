// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import type {AnswerArtifact} from '../api/conversations.ts';
import './artifact_canvas.ts';
import type {DlArtifactCanvas} from './artifact_canvas.ts';

const originalFetch = window.fetch;
const originalMatchMedia = window.matchMedia;

function desktopMedia(query: string): MediaQueryList {
  return {
    matches: false,
    media: query,
    onchange: null,
    addListener() {},
    removeListener() {},
    addEventListener() {},
    removeEventListener() {},
    dispatchEvent: () => true,
  };
}

function htmlArtifact(): AnswerArtifact {
  return {
    resource_id: 'artifact-html',
    role: 'primary_report',
    media_type: 'text/html',
    label: 'Interactive report',
    filename: 'report.html',
    byte_size: 42,
    digest: 'a'.repeat(64),
    presentation: 'html',
    status: 'available',
    uri: 'dlightrag://answer/run-1/artifacts/artifact-html',
    width: null,
    height: null,
    data_url: '/web/api/answer/run-1/artifacts/artifact-html',
    download_url: '/web/api/answer/run-1/artifacts/artifact-html?download=1',
    presentation_url: '/web/api/answer/run-1/artifacts/artifact-html/presentation',
    issue: null,
  };
}

afterEach(() => {
  window.fetch = originalFetch;
  window.matchMedia = originalMatchMedia;
  document.body.replaceChildren();
  document.body.className = '';
});

it('requires explicit user intent before creating the active iframe', async () => {
  window.matchMedia = desktopMedia;
  window.fetch = async () => new Response('<!doctype html><html><body>Report</body></html>');
  const returnFocus = document.createElement('button');
  document.body.appendChild(returnFocus);
  returnFocus.focus();
  const canvas = document.createElement('dl-artifact-canvas') as DlArtifactCanvas;
  canvas.activePreviewEnabled = true;
  document.body.appendChild(canvas);

  await canvas.open(htmlArtifact(), returnFocus);
  await canvas.updateComplete;

  expect(canvas.classList.contains('open')).to.equal(true);
  expect(canvas.inert).to.equal(false);
  expect(canvas.getAttribute('aria-modal')).to.equal('true');
  expect(canvas.querySelector('dl-active-artifact-frame')).to.equal(null);
  const consent = canvas.querySelector<HTMLButtonElement>('.artifact-active-consent .ui-btn');
  expect(consent?.textContent).to.contain('Open interactive report');
  consent?.click();
  await canvas.updateComplete;
  const frame = canvas.querySelector('dl-active-artifact-frame');
  expect(frame).not.to.equal(null);
  expect(frame?.active).to.equal(true);

  frame?.dispatchEvent(new CustomEvent('artifact-frame-escape', {bubbles: true, composed: true}));
  await canvas.updateComplete;
  expect(canvas.classList.contains('open')).to.equal(false);
  expect(canvas.inert).to.equal(true);
  expect(canvas.getAttribute('aria-hidden')).to.equal('true');
  expect(canvas.hasAttribute('aria-modal')).to.equal(false);
  expect(document.activeElement).to.equal(returnFocus);

  await canvas.open(htmlArtifact(), returnFocus);
  canvas.querySelector<HTMLButtonElement>('[aria-pressed][type="button"]')?.click();
  await canvas.updateComplete;
  expect(canvas.hasAttribute('aria-modal')).to.equal(false);
  canvas.close();
  await canvas.updateComplete;
  expect(canvas.querySelector('dl-active-artifact-frame')).to.equal(null);
  expect(document.activeElement).to.equal(returnFocus);
});

it('uses one modal predicate for ARIA, focus trapping, and outside inertness', async () => {
  window.matchMedia = desktopMedia;
  window.fetch = async () => new Response('<!doctype html><html><body>Report</body></html>');
  document.body.classList.add('conversation-sidebar-open');
  const sidebar = document.createElement('nav');
  sidebar.id = 'chat-sidebar';
  const topbar = document.createElement('header');
  topbar.className = 'topbar';
  const chat = document.createElement('main');
  chat.className = 'chat-area';
  const composer = document.createElement('div');
  composer.className = 'composer';
  const panel = document.createElement('aside');
  panel.id = 'panel';
  panel.className = 'open';
  const canvas = document.createElement('dl-artifact-canvas') as DlArtifactCanvas;
  document.body.append(sidebar, topbar, chat, composer, panel, canvas);

  await canvas.open(htmlArtifact());
  await canvas.updateComplete;

  expect(canvas.getAttribute('aria-modal')).to.equal('true');
  expect(document.body.classList.contains('artifact-canvas-modal')).to.equal(true);
  expect(sidebar.inert).to.equal(true);
  expect(topbar.inert).to.equal(true);
  expect(chat.inert).to.equal(true);
  expect(composer.inert).to.equal(true);
  expect(panel.inert).to.equal(true);

  const focusable = Array.from(canvas.querySelectorAll<HTMLButtonElement>('button'));
  focusable.at(-1)?.focus();
  focusable.at(-1)?.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Tab', bubbles: true, cancelable: true,
  }));
  expect(document.activeElement).to.equal(focusable[0]);

  canvas.prepareForInspector();
  await canvas.updateComplete;
  expect(canvas.layout).to.equal('side');
  expect(canvas.hasAttribute('aria-modal')).to.equal(false);
  expect(document.body.classList.contains('artifact-canvas-modal')).to.equal(false);
  expect(sidebar.inert).to.equal(false);
  expect(topbar.inert).to.equal(false);
  expect(chat.inert).to.equal(false);
  expect(composer.inert).to.equal(false);
  expect(panel.inert).to.equal(false);
});

it('operator-disabled HTML uses only the script-disabled static frame', async () => {
  window.fetch = async () => new Response('<!doctype html><html><body>Static</body></html>');
  const canvas = document.createElement('dl-artifact-canvas') as DlArtifactCanvas;
  canvas.activePreviewEnabled = false;
  document.body.appendChild(canvas);

  await canvas.open(htmlArtifact());
  await canvas.updateComplete;

  const frame = canvas.querySelector('dl-active-artifact-frame');
  expect(frame).not.to.equal(null);
  expect(frame?.active).to.equal(false);
  expect(canvas.textContent).not.to.contain('Open interactive report');
});

it('isolates PDF preview in a sandboxed no-referrer iframe', async () => {
  const artifact = {...htmlArtifact(),
    media_type: 'application/pdf',
    filename: 'report.pdf',
    presentation: 'pdf' as const,
  };
  const canvas = document.createElement('dl-artifact-canvas') as DlArtifactCanvas;
  document.body.appendChild(canvas);

  await canvas.open(artifact);
  await canvas.updateComplete;

  const iframe = canvas.querySelector<HTMLIFrameElement>('iframe.artifact-pdf');
  expect(iframe?.getAttribute('sandbox')).to.equal('');
  expect(iframe?.referrerPolicy).to.equal('no-referrer');
});

it('an unavailable Artifact renders a persistent safe issue without fetching', async () => {
  let fetched = false;
  window.fetch = async () => {
    fetched = true;
    return new Response();
  };
  const artifact = htmlArtifact();
  artifact.status = 'unavailable';
  artifact.issue = {
    kind: 'missing_file',
    description: 'Referenced Artifact is missing.',
    resource_id: artifact.resource_id,
  };
  const canvas = document.createElement('dl-artifact-canvas') as DlArtifactCanvas;
  document.body.appendChild(canvas);

  await canvas.open(artifact);
  await canvas.updateComplete;

  expect(canvas.querySelector('[role="alert"]')?.textContent).to.contain('missing');
  expect(fetched).to.equal(false);
});
