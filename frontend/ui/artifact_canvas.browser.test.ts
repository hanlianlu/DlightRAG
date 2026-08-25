// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import type {AnswerArtifact} from '../api/conversations.ts';
import './artifact_canvas.ts';
import type {DlArtifactCanvas} from './artifact_canvas.ts';

const originalFetch = window.fetch;

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
  document.body.replaceChildren();
  document.body.classList.remove('artifact-canvas-open');
});

it('requires explicit user intent before creating the active iframe', async () => {
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
  expect(canvas.querySelector('dl-active-artifact-frame')).to.equal(null);
  const consent = canvas.querySelector<HTMLButtonElement>('.artifact-active-consent .ui-btn');
  expect(consent?.textContent).to.contain('Open interactive report');
  consent?.click();
  await canvas.updateComplete;
  const frame = canvas.querySelector('dl-active-artifact-frame');
  expect(frame).not.to.equal(null);
  expect(frame?.active).to.equal(true);

  canvas.close();
  await canvas.updateComplete;
  expect(canvas.querySelector('dl-active-artifact-frame')).to.equal(null);
  expect(document.activeElement).to.equal(returnFocus);
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
