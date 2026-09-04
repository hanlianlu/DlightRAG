// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import type {AnswerArtifact} from '../api/conversations.ts';
import {defineDesignSystemElements} from '../design-system/index.ts';
import './artifact-canvas.ts';

defineDesignSystemElements();

import type {DlActiveArtifactFrame} from './active-artifact-frame.ts';
import type {DlArtifactCanvas} from './artifact-canvas.ts';
import './active-artifact-frame.ts';

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
    resourceId: 'artifact-html',
    mediaType: 'text/html',
    label: 'Interactive report',
    filename: 'report.html',
    byteSize: 42,
    digest: 'a'.repeat(64),
    presentation: 'html',
    status: 'available',
    uri: 'dlightrag://answer/run-1/artifacts/artifact-html',
    width: null,
    height: null,
    dataUrl: '/web/api/answer/run-1/artifacts/artifact-html',
    downloadUrl: '/web/api/answer/run-1/artifacts/artifact-html?download=1',
    presentationUrl: '/web/api/answer/run-1/artifacts/artifact-html/presentation',
    issue: null,
  };
}

function artifact(id: string, label: string, presentation: 'markdown' | 'html'): AnswerArtifact {
  const extension = presentation === 'markdown' ? 'md' : 'html';
  const mediaType = presentation === 'markdown' ? 'text/markdown' : 'text/html';
  return {
    ...htmlArtifact(),
    resourceId: id,
    mediaType: mediaType,
    label,
    filename: `${id}.${extension}`,
    presentation,
    uri: `dlightrag://answer/run-1/artifacts/${id}`,
    dataUrl: `/web/api/answer/run-1/artifacts/${id}`,
    downloadUrl: `/web/api/answer/run-1/artifacts/${id}?download=1`,
    presentationUrl: `/web/api/answer/run-1/artifacts/${id}/presentation`,
  };
}

// The canvas parses the presentation response through the api schema, so the
// stub must serve the snake_case wire shape.
function markdownPresentation(text: string): Record<string, unknown> {
  return {
    answer_text: text,
    parts: [{
      type: 'markdown', text, html: `<p>${text}</p>`, artifact: null,
      evidence_image: null, inline: false,
    }],
    sources: [],
    evidence_images: [],
    artifacts: [],
    artifact_outcome: {status: 'complete', issues: []},
  };
}

afterEach(() => {
  window.fetch = originalFetch;
  window.matchMedia = originalMatchMedia;
  document.body.replaceChildren();
  document.body.className = '';
});

it('ignores a previous Markdown response that finishes decoding after an Artifact switch', async () => {
  let finishStaleDecode!: (value: Record<string, unknown>) => void;
  let markDecodeStarted!: () => void;
  const staleDecode = new Promise<Record<string, unknown>>((resolve) => { finishStaleDecode = resolve; });
  const decodeStarted = new Promise<void>((resolve) => { markDecodeStarted = resolve; });
  const currentPresentation = markdownPresentation('Current report');
  window.fetch = async (input) => {
    if (String(input).includes('/stale-markdown/')) {
      return {
        ok: true,
        json: async () => {
          markDecodeStarted();
          return staleDecode;
        },
      } as Response;
    }
    return new Response(JSON.stringify(currentPresentation));
  };
  const canvas = document.createElement('dl-artifact-canvas') as DlArtifactCanvas;
  document.body.appendChild(canvas);

  const staleOpen = canvas.open(artifact('stale-markdown', 'Stale report', 'markdown'));
  await decodeStarted;
  await canvas.open(artifact('current-markdown', 'Current report', 'markdown'));
  await canvas.updateComplete;
  await canvas.querySelector('dl-answer-presentation')?.updateComplete;

  expect(canvas.querySelector('.artifact-canvas-title')?.textContent).to.equal('Current report');
  expect(canvas.querySelector<HTMLAnchorElement>('[download]')?.getAttribute('href'))
    .to.equal(`${window.location.origin}/web/api/answer/run-1/artifacts/current-markdown?download=1`);
  expect(canvas.querySelector('.answer-rich-content')?.textContent).to.equal('Current report');

  finishStaleDecode(markdownPresentation('Stale report'));
  await staleOpen;
  await canvas.updateComplete;
  await canvas.querySelector('dl-answer-presentation')?.updateComplete;

  expect(canvas.querySelector('.artifact-canvas-title')?.textContent).to.equal('Current report');
  expect(canvas.querySelector('.answer-rich-content')?.textContent).to.equal('Current report');
  expect(canvas.textContent).not.to.contain('Stale report');
});

it('ignores previous HTML text that finishes decoding after an Artifact switch', async () => {
  let finishStaleDecode!: (value: string) => void;
  let markDecodeStarted!: () => void;
  const staleDecode = new Promise<string>((resolve) => { finishStaleDecode = resolve; });
  const decodeStarted = new Promise<void>((resolve) => { markDecodeStarted = resolve; });
  window.fetch = async (input) => {
    if (String(input).includes('/stale-html')) {
      return {
        ok: true,
        text: async () => {
          markDecodeStarted();
          return staleDecode;
        },
      } as Response;
    }
    return new Response('<!doctype html><html><body>Current HTML</body></html>');
  };
  const canvas = document.createElement('dl-artifact-canvas') as DlArtifactCanvas;
  canvas.activePreviewEnabled = false;
  document.body.appendChild(canvas);

  const staleOpen = canvas.open(artifact('stale-html', 'Stale HTML', 'html'));
  await decodeStarted;
  await canvas.open(artifact('current-html', 'Current HTML', 'html'));
  await canvas.updateComplete;

  expect(canvas.querySelector('.artifact-canvas-title')?.textContent).to.equal('Current HTML');
  expect(canvas.querySelector<HTMLAnchorElement>('[download]')?.getAttribute('href'))
    .to.equal(`${window.location.origin}/web/api/answer/run-1/artifacts/current-html?download=1`);
  const frame = canvas.querySelector<DlActiveArtifactFrame>('dl-active-artifact-frame');
  expect(frame?.source).to.contain('Current HTML');

  finishStaleDecode('<!doctype html><html><body>Stale HTML</body></html>');
  await staleOpen;
  await canvas.updateComplete;

  expect(canvas.querySelector('.artifact-canvas-title')?.textContent).to.equal('Current HTML');
  expect(frame?.source).to.contain('Current HTML');
  expect(frame?.source).not.to.contain('Stale HTML');
});

it('opens the interactive frame directly without a consent step', async () => {
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
  expect(canvas.getAttribute('role')).to.equal('dialog');
  expect(canvas.getAttribute('aria-labelledby')).to.equal('artifact-canvas-title');
  expect(canvas.querySelector('#artifact-canvas-title')?.textContent?.trim())
    .to.equal('Interactive report');
  const layoutGroup = canvas.querySelector<HTMLElement>('.artifact-canvas-layout-actions')!;
  expect(layoutGroup.getAttribute('role')).to.equal('group');
  expect(layoutGroup.getAttribute('aria-labelledby')).to.equal('artifact-canvas-title');
  expect(canvas.getAttribute('aria-modal')).to.equal(null);
  expect(canvas.textContent).not.to.contain('Open interactive report');
  const frame = canvas.querySelector('dl-active-artifact-frame');
  expect(frame).not.to.equal(null);
  expect(frame?.active).to.equal(true);

  frame?.dispatchEvent(new CustomEvent('dl-artifact-frame-escape', {bubbles: true, composed: true}));
  await canvas.updateComplete;
  await new Promise((resolve) => requestAnimationFrame(resolve));
  expect(canvas.classList.contains('open')).to.equal(false);
  expect(canvas.inert).to.equal(true);
  expect(canvas.getAttribute('aria-hidden')).to.equal('true');
  expect(canvas.hasAttribute('aria-labelledby')).to.equal(false);
  expect(canvas.hasAttribute('aria-modal')).to.equal(false);
  expect(document.activeElement).to.equal(returnFocus);

  await canvas.open(htmlArtifact(), returnFocus);
  canvas.querySelector<HTMLButtonElement>('[aria-pressed][type="button"]')?.click();
  await canvas.updateComplete;
  expect(canvas.hasAttribute('aria-modal')).to.equal(false);
  canvas.close();
  await canvas.updateComplete;
  await new Promise((resolve) => requestAnimationFrame(resolve));
  expect(canvas.querySelector('dl-active-artifact-frame')).to.equal(null);
  expect(document.activeElement).to.equal(returnFocus);
});

it('publishes modal, overlay, and wide predicates for Shell composition', async () => {
  window.matchMedia = desktopMedia;
  window.fetch = async () => new Response('<!doctype html><html><body>Report</body></html>');
  const canvas = document.createElement('dl-artifact-canvas') as DlArtifactCanvas;
  const states: Array<{open: boolean; modal: boolean; overlay: boolean; wide: boolean}> = [];
  canvas.addEventListener('dl-artifact-canvas-state-change', (event) => {
    states.push(event.detail);
  });
  document.body.append(canvas);

  await canvas.open(htmlArtifact());
  await canvas.updateComplete;

  // HTML suggests wide: docked expansion, not a modal, no overlay.
  expect(canvas.layout).to.equal('wide');
  expect(canvas.getAttribute('aria-modal')).to.equal(null);
  expect(states.at(-1)).to.deep.equal({open: true, modal: false, overlay: false, wide: true});

  // Fullscreen is the modal; focus trapping applies there.
  const fullscreenButton = Array.from(canvas.querySelectorAll<HTMLButtonElement>('.artifact-canvas-layout-actions button'))
    .find((button) => button.textContent?.trim() === 'Fullscreen');
  fullscreenButton?.click();
  await canvas.updateComplete;
  expect(canvas.getAttribute('aria-modal')).to.equal('true');
  expect(states.at(-1)).to.deep.equal({open: true, modal: true, overlay: true, wide: false});
  const focusable = Array.from(canvas.querySelectorAll<HTMLElement>(
    'button:not([disabled]), dl-icon-button:not([disabled]), a[href]',
  ));
  focusable.at(-1)?.focus();
  focusable.at(-1)?.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Tab', bubbles: true, cancelable: true,
  }));
  expect(document.activeElement).to.equal(focusable[0]);

  canvas.prepareForInspector();
  await canvas.updateComplete;
  expect(canvas.layout).to.equal('side');
  expect(canvas.hasAttribute('aria-modal')).to.equal(false);
  expect(states.at(-1)).to.deep.equal({open: true, modal: false, overlay: false, wide: false});
});

it('does not restore stale focus when close is immediately followed by reopen', async () => {
  window.matchMedia = desktopMedia;
  const artifact = htmlArtifact();
  artifact.status = 'unavailable';
  artifact.issue = {
    kind: 'missing_file',
    description: 'Referenced Artifact is missing.',
    resourceId: artifact.resourceId,
  };
  const firstTrigger = document.createElement('button');
  const secondTrigger = document.createElement('button');
  const canvas = document.createElement('dl-artifact-canvas') as DlArtifactCanvas;
  document.body.append(firstTrigger, secondTrigger, canvas);

  await canvas.open(artifact, firstTrigger);
  canvas.close();
  const reopened = canvas.open(artifact, secondTrigger);
  await reopened;
  await new Promise((resolve) => requestAnimationFrame(resolve));

  expect(canvas.classList.contains('open')).to.equal(true);
  expect(document.activeElement).not.to.equal(firstTrigger);
  expect(canvas.contains(document.activeElement)).to.equal(true);
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
  expect(frame?.source).to.contain('Static');
  expect(canvas.textContent).not.to.contain('Open interactive report');
  expect(canvas.querySelector('details')).to.equal(null);
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
    resourceId: artifact.resourceId,
  };
  const canvas = document.createElement('dl-artifact-canvas') as DlArtifactCanvas;
  document.body.appendChild(canvas);

  await canvas.open(artifact);
  await canvas.updateComplete;

  expect(canvas.querySelector('[role="alert"]')?.textContent).to.contain('missing');
  expect(fetched).to.equal(false);
});
